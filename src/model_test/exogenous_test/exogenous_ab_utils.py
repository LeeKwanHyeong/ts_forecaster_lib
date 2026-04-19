from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "ts_forecaster_mpl"))

import matplotlib
import numpy as np
import polars as pl
import torch

from model_test.future_exo_utils import build_lookup_tuple, build_week_index
from model_test.model_test_utils import add_week
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_PLOT_COLOR = "#ff6b6b"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


@dataclass(frozen=True)
class WeeklyCalendarFutureExoCallback:
    date_type: str = "weekly"
    sincos: bool = True

    def __call__(self, start_idx, H, device: str = "cpu"):
        # top-level dataclass callable로 감싸서
        # DataLoader worker spawn 환경에서도 pickle 가능하게 유지한다.
        cb = compose_exo_calendar_cb(date_type=self.date_type, sincos=self.sincos)
        return cb(start_idx, H, device=device)


@dataclass(frozen=True)
class DenseLookupPartFutureExoFn:
    lookup: dict[tuple[str, int], tuple[Any, ...]]
    ww_list: list[int]
    ww_to_pos: dict[int, int]
    width: int

    def __call__(self, uid_list, start_idxs, H):
        # 각 배치 샘플의 uid + forecast start week 조합으로
        # 미래 구간의 part-specific exogenous lookup tensor를 만든다.
        batch = len(uid_list)
        horizon = int(H)
        out = np.zeros((batch, horizon, int(self.width)), dtype=np.float32)

        if int(self.width) <= 0:
            return out

        for b, (uid, start_ww) in enumerate(zip(uid_list, start_idxs)):
            pos = self.ww_to_pos.get(int(start_ww))
            if pos is None:
                continue

            future_ww = self.ww_list[pos : pos + horizon]
            for k, ww in enumerate(future_ww):
                vals = self.lookup.get((str(uid), int(ww)))
                if vals is None:
                    continue
                out[b, k, :] = np.asarray(vals, dtype=np.float32)

        return out


def make_dense_lookup_part_future_exo_fn(
    df: pl.DataFrame,
    *,
    id_col: str,
    date_col: str,
    value_cols: Sequence[str],
):
    # lookup dict와 week index를 한 번만 만들고 callable 객체에 담아
    # 매 배치마다 재계산하지 않도록 한다.
    lookup = build_lookup_tuple(df, id_col, date_col, list(value_cols))
    ww_list, ww_to_pos = build_week_index(df, date_col)
    return DenseLookupPartFutureExoFn(
        lookup=lookup,
        ww_list=ww_list,
        ww_to_pos=ww_to_pos,
        width=len(value_cols),
    )


def build_callback_future_exo_components(
    df: pl.DataFrame,
    *,
    id_col: str,
    date_col: str,
    lookup_future_cols: Sequence[str],
):
    # callback 모드는 calendar 계열 feature와
    # part-specific lookup feature를 분리 생성한 뒤 collate에서 concat한다.
    future_exo_cb = WeeklyCalendarFutureExoCallback(date_type="weekly", sincos=True)
    part_future_exo_fn = make_dense_lookup_part_future_exo_fn(
        df,
        id_col=id_col,
        date_col=date_col,
        value_cols=lookup_future_cols,
    )
    return future_exo_cb, part_future_exo_fn


def infer_plan_weeks(
    target_df: pl.DataFrame,
    *,
    date_col: str,
    horizon: int,
    plan_count: int,
    plan_start_week: Optional[int] = None,
    plan_end_week: Optional[int] = None,
) -> list[int]:
    # 평가에 사용할 plan week는
    # "해당 시점 이후 horizon 길이만큼 actual이 존재하는 주차"만 선택한다.
    weeks = (
        target_df.select(pl.col(date_col).cast(pl.Int64))
        .drop_nulls()
        .unique()
        .sort(date_col)
        .get_column(date_col)
        .to_list()
    )
    week_set = set(int(w) for w in weeks)
    eligible: list[int] = []

    for week in weeks:
        ww = int(week)
        if plan_start_week is not None and ww < int(plan_start_week):
            continue
        if plan_end_week is not None and ww > int(plan_end_week):
            continue
        if all(add_week(ww, step) in week_set for step in range(int(horizon))):
            eligible.append(ww)

    if not eligible:
        raise ValueError("No eligible plan weeks found with full future actual coverage.")

    take = min(int(plan_count), len(eligible))
    if take <= 0:
        raise ValueError(f"`plan_count` must be positive. got={plan_count}")
    return eligible[-take:]


def resolve_single_plan_week(
    target_df: pl.DataFrame,
    *,
    date_col: str,
    horizon: int,
    plan_week: int,
) -> int:
    # 단일 backtest plan week를 명시적으로 검증한다.
    # - 해당 주차가 target 테이블에 존재해야 함
    # - plan_week부터 horizon 길이만큼의 actual 주차가 전역적으로 존재해야 함
    weeks = (
        target_df.select(pl.col(date_col).cast(pl.Int64))
        .drop_nulls()
        .unique()
        .sort(date_col)
        .get_column(date_col)
        .to_list()
    )
    week_set = {int(w) for w in weeks}
    resolved = int(plan_week)

    if resolved not in week_set:
        raise ValueError(
            f"Requested plan_week={resolved} is not present in target_df[{date_col!r}]."
        )

    missing = [
        int(add_week(resolved, step))
        for step in range(int(horizon))
        if int(add_week(resolved, step)) not in week_set
    ]
    if missing:
        raise ValueError(
            "Requested plan_week does not have full future actual coverage for the requested horizon. "
            f"plan_week={resolved}, missing_weeks={missing[:5]}"
        )

    return resolved


def select_eval_ids_with_full_actual_coverage(
    target_df: pl.DataFrame,
    *,
    id_col: str,
    date_col: str,
    plan_weeks: Sequence[int],
    horizon: int,
) -> list[str]:
    # 예측/평가에 사용할 모든 forecast_week를 먼저 구한 뒤,
    # 그 주차를 전부 보유한 part만 evaluation 대상으로 남긴다.
    required_weeks = sorted(
        {
            int(add_week(int(plan_week), int(step)))
            for plan_week in plan_weeks
            for step in range(int(horizon))
        }
    )
    if not required_weeks:
        return []

    coverage = (
        target_df.filter(pl.col(date_col).cast(pl.Int64).is_in(required_weeks))
        .group_by(id_col)
        .agg(pl.col(date_col).n_unique().alias("n_required_present"))
        .filter(pl.col("n_required_present") == len(required_weeks))
        .select(pl.col(id_col).cast(pl.String))
        .sort(id_col)
    )
    return coverage.get_column(id_col).to_list()


@torch.no_grad()
def make_point_forecast_result_table(
    *,
    inference_loader,
    predictor,
    model_name: str,
    plan_week: int,
    horizon: int,
    device: str,
    max_parts: int = 100_000,
) -> pl.DataFrame:
    # 모델별 / 계획주별 예측을 long-format으로 저장하면
    # 이후 metric, join, plot 파이프라인을 공통 로직으로 처리하기 쉽다.
    schema = {
        "model_name": pl.String,
        "part_no": pl.String,
        "plan_week": pl.Int64,
        "forecast_week": pl.Int64,
        "horizon_step": pl.Int64,
        "prediction": pl.Float64,
    }
    rows: list[dict[str, Any]] = []
    n_parts = 0

    for batch in inference_loader:
        # LoadedPredictor wrapper가 기본적으로 eval 경로를 사용하므로
        # 여기서는 horizon/device만 넘겨 중복 keyword 충돌을 피한다.
        pred = predictor.predict(batch, horizon=int(horizon), device=device)
        point = np.asarray(pred["point"])
        if point.ndim == 1:
            point = point.reshape(1, -1)
        elif point.ndim != 2:
            point = point.reshape(point.shape[0], -1)

        if len(batch) == 5:
            _, uid_list, _, _, _ = batch
        elif len(batch) == 6:
            _, _, uid_list, _, _, _ = batch
        else:
            raise RuntimeError(f"Unsupported batch tuple length: {len(batch)}")

        batch_size = len(uid_list)
        if point.shape[0] == 1 and batch_size > 1:
            point = np.repeat(point, batch_size, axis=0)
        if point.shape[0] != batch_size:
            raise RuntimeError(
                f"Prediction batch mismatch for {model_name}: preds={point.shape[0]} uids={batch_size}"
            )

        for i in range(batch_size):
            if n_parts >= int(max_parts):
                break

            pid = uid_list[i]
            pid_str = str(pid.item()) if torch.is_tensor(pid) and pid.numel() == 1 else str(pid)
            for step, pred_val in enumerate(point[i, : int(horizon)].tolist()):
                rows.append(
                    {
                        "model_name": model_name,
                        "part_no": pid_str,
                        "plan_week": int(plan_week),
                        "forecast_week": int(add_week(plan_week, step)),
                        "horizon_step": int(step),
                        "prediction": float(pred_val),
                    }
                )
            n_parts += 1

        if n_parts >= int(max_parts):
            break

    return pl.DataFrame(rows, schema=schema)


def attach_actuals(
    forecast_df: pl.DataFrame,
    target_df: pl.DataFrame,
    *,
    id_col: str,
    date_col: str,
    y_col: str,
) -> pl.DataFrame:
    # forecast long table에 GT를 붙이고,
    # 음수 예측은 downstream metric/plot 해석을 위해 0으로 clip한다.
    actual_df = (
        target_df.select(
            [
                pl.col(id_col).cast(pl.String).alias("part_no"),
                pl.col(date_col).cast(pl.Int64).alias("forecast_week"),
                pl.col(y_col).cast(pl.Float64).alias("actual"),
            ]
        )
    )

    return (
        forecast_df.with_columns(pl.col("part_no").cast(pl.String))
        .join(actual_df, on=["part_no", "forecast_week"], how="left")
        .with_columns(
            pl.when(pl.col("prediction") < 0.0)
            .then(0.0)
            .otherwise(pl.col("prediction"))
            .alias("prediction")
        )
    )


def select_latest_revision(
    forecast_df: pl.DataFrame,
    *,
    extra_group_cols: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    if forecast_df.height == 0:
        return forecast_df

    # 동일한 forecast_week를 여러 revision이 예측한 경우
    # 가장 최신 plan_week 결과만 남겨서 최신 revision view를 만든다.
    # variant 같은 추가 비교축이 있으면 그 축을 보존한 채 latest를 뽑아야
    # raw/guard 또는 여러 inference setting 비교가 서로 섞이지 않는다.
    base_group_cols = ["model_name", "part_no", "forecast_week"]
    inferred_extra_cols = [
        col
        for col in ["variant"]
        if col in forecast_df.columns and col not in base_group_cols
    ]
    if extra_group_cols is None:
        group_cols = base_group_cols + inferred_extra_cols
    else:
        normalized_extra_cols = [
            col
            for col in extra_group_cols
            if col in forecast_df.columns and col not in base_group_cols
        ]
        group_cols = base_group_cols + normalized_extra_cols

    sort_cols = group_cols + ["plan_week"]
    return (
        forecast_df.sort(sort_cols, descending=[False] * len(group_cols) + [True])
        .group_by(group_cols, maintain_order=True)
        .first()
        .sort(group_cols)
    )


def compute_metric_tables(
    forecast_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # 전체 revision을 다 보는 metric과
    # latest revision 기준 metric을 함께 만들면
    # "모델 평균 성능"과 "현재 운영 관점 성능"을 같이 비교할 수 있다.
    valid = forecast_df.filter(pl.col("actual").is_not_null())
    if valid.height == 0:
        empty = pl.DataFrame()
        return empty, empty, empty

    valid = valid.with_columns(
        [
            (pl.col("prediction") - pl.col("actual")).alias("error"),
            (pl.col("prediction") - pl.col("actual")).abs().alias("abs_error"),
            ((pl.col("prediction") - pl.col("actual")) ** 2).alias("sq_error"),
            (
                2.0
                * (pl.col("prediction") - pl.col("actual")).abs()
                / (pl.col("prediction").abs() + pl.col("actual").abs() + 1e-6)
            ).alias("smape_term"),
        ]
    )

    def _summarize(group_cols: Sequence[str]) -> pl.DataFrame:
        summary = (
            valid.group_by(list(group_cols))
            .agg(
                [
                    pl.len().alias("n_rows"),
                    pl.col("part_no").n_unique().alias("n_parts"),
                    pl.col("plan_week").n_unique().alias("n_plan_weeks"),
                    pl.col("abs_error").mean().alias("mae"),
                    pl.col("sq_error").mean().sqrt().alias("rmse"),
                    pl.col("error").mean().alias("bias"),
                    pl.col("smape_term").mean().alias("smape"),
                    pl.col("abs_error").sum().alias("abs_error_sum"),
                    pl.col("actual").abs().sum().alias("abs_actual_sum"),
                ]
            )
            .with_columns(
                pl.when(pl.col("abs_actual_sum") > 0.0)
                .then(pl.col("abs_error_sum") / pl.col("abs_actual_sum"))
                .otherwise(None)
                .alias("wape")
            )
            .drop(["abs_error_sum", "abs_actual_sum"])
        )
        return summary

    overall = _summarize(["model_name"]).sort("mae")
    by_horizon = _summarize(["model_name", "horizon_step"]).sort(["horizon_step", "mae"])
    latest_summary = _summarize(["model_name"]) if valid.height > 0 else pl.DataFrame()
    latest = select_latest_revision(valid)
    if latest.height > 0:
        latest = latest.with_columns(
            [
                (pl.col("prediction") - pl.col("actual")).alias("error"),
                (pl.col("prediction") - pl.col("actual")).abs().alias("abs_error"),
                ((pl.col("prediction") - pl.col("actual")) ** 2).alias("sq_error"),
                (
                    2.0
                    * (pl.col("prediction") - pl.col("actual")).abs()
                    / (pl.col("prediction").abs() + pl.col("actual").abs() + 1e-6)
                ).alias("smape_term"),
            ]
        )
        latest_summary = (
            latest.group_by("model_name")
            .agg(
                [
                    pl.len().alias("n_rows"),
                    pl.col("part_no").n_unique().alias("n_parts"),
                    pl.col("forecast_week").n_unique().alias("n_target_weeks"),
                    pl.col("abs_error").mean().alias("mae"),
                    pl.col("sq_error").mean().sqrt().alias("rmse"),
                    pl.col("error").mean().alias("bias"),
                    pl.col("smape_term").mean().alias("smape"),
                    pl.col("abs_error").sum().alias("abs_error_sum"),
                    pl.col("actual").abs().sum().alias("abs_actual_sum"),
                ]
            )
            .with_columns(
                pl.when(pl.col("abs_actual_sum") > 0.0)
                .then(pl.col("abs_error_sum") / pl.col("abs_actual_sum"))
                .otherwise(None)
                .alias("wape")
            )
            .drop(["abs_error_sum", "abs_actual_sum"])
            .sort("mae")
        )

    return overall, by_horizon, latest_summary


def select_plot_parts(
    latest_df: pl.DataFrame,
    *,
    plot_part_count: int,
    actual_positive_only: bool = True,
) -> list[str]:
    if latest_df.height == 0:
        return []

    # 플롯은 실제 수요가 어느 정도 있는 part 위주로 고르면
    # zero-only 시계열보다 비교가 훨씬 잘 보인다.
    candidates = (
        latest_df.group_by("part_no")
        .agg(
            [
                pl.len().alias("n_rows"),
                (pl.col("actual") > 0).sum().alias("n_nonzero"),
                pl.col("actual").sum().alias("actual_sum"),
            ]
        )
        .sort(["actual_sum", "part_no"], descending=[True, False])
    )
    if actual_positive_only:
        candidates = candidates.filter(pl.col("n_nonzero") > 0)

    return candidates.head(int(plot_part_count)).get_column("part_no").to_list()


def plot_metric_summary(overall_df: pl.DataFrame, save_path: Path) -> None:
    if overall_df.height == 0:
        return

    # 여러 metric을 한 장에서 비교할 수 있게 2x2 요약 플롯으로 저장한다.
    metrics = ["mae", "rmse", "wape", "smape"]
    labels = overall_df.get_column("model_name").to_list()
    values = {
        metric: [
            float(v) if v is not None else float("nan")
            for v in overall_df.get_column(metric).to_list()
        ]
        for metric in metrics
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        ax.bar(labels, values[metric])
        ax.set_title(metric.upper())
        ax.set_xlabel("model")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latest_revision_aggregate(latest_df: pl.DataFrame, save_path: Path) -> None:
    if latest_df.height == 0:
        return

    # part 단위 noise를 줄이기 위해 forecast_week 기준 총합 곡선도 함께 본다.
    actual_df = (
        latest_df.select(["part_no", "forecast_week", "actual"])
        .unique(subset=["part_no", "forecast_week"])
        .group_by("forecast_week")
        .agg(pl.col("actual").sum().alias("actual_sum"))
        .sort("forecast_week")
    )
    pred_df = (
        latest_df.group_by(["model_name", "forecast_week"])
        .agg(pl.col("prediction").sum().alias("prediction_sum"))
        .sort(["model_name", "forecast_week"])
    )
    x_actual = [str(v) for v in actual_df.get_column("forecast_week").to_list()]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        x_actual,
        actual_df.get_column("actual_sum").to_list(),
        marker="o",
        linewidth=2.2,
        label="GT",
        color=GT_PLOT_COLOR,
    )

    for model_name in pred_df.get_column("model_name").unique().to_list():
        sub = pred_df.filter(pl.col("model_name") == model_name).sort("forecast_week")
        ax.plot(
            [str(v) for v in sub.get_column("forecast_week").to_list()],
            sub.get_column("prediction_sum").to_list(),
            marker="o",
            label=model_name,
        )

    ax.set_title("Latest Revision Aggregate Forecast vs GT")
    ax.set_xlabel("forecast_week")
    ax.set_ylabel("sum_qty")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latest_revision_part_grid(
    latest_df: pl.DataFrame,
    *,
    sampled_parts: Sequence[str],
    save_path: Path,
    ncols: int = 3,
) -> None:
    if latest_df.height == 0 or not sampled_parts:
        return

    # 운영적으로 중요한 일부 part를 골라
    # GT와 모델별 예측 궤적을 한 장의 grid로 비교한다.
    n_parts = len(sampled_parts)
    nrows = math.ceil(n_parts / int(ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, part_no in zip(axes_flat, sampled_parts):
        part_df = latest_df.filter(pl.col("part_no") == str(part_no)).sort("forecast_week")
        actual_df = (
            part_df.select(["forecast_week", "actual"])
            .unique(subset=["forecast_week"])
            .sort("forecast_week")
        )
        x = [str(v) for v in actual_df.get_column("forecast_week").to_list()]
        ax.plot(
            x,
            actual_df.get_column("actual").to_list(),
            marker="o",
            linewidth=2.0,
            label="GT",
            color=GT_PLOT_COLOR,
        )

        for model_name in part_df.get_column("model_name").unique().to_list():
            model_df = (
                part_df.filter(pl.col("model_name") == model_name)
                .select(["forecast_week", "prediction"])
                .sort("forecast_week")
            )
            ax.plot(
                [str(v) for v in model_df.get_column("forecast_week").to_list()],
                model_df.get_column("prediction").to_list(),
                marker="o",
                label=model_name,
            )

        ax.set_title(str(part_no), fontsize=10)
        ax.set_xlabel("forecast_week")
        ax.set_ylabel("qty")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for idx in range(n_parts, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle("Latest Revision GT vs Predictions by Part", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
