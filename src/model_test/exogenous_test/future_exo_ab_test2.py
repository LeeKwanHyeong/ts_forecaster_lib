from __future__ import annotations

import inspect
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def _configure_stdio_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()


THIS_FILE = Path(__file__).resolve()
for candidate in [THIS_FILE.parent, *THIS_FILE.parents, Path.cwd().resolve()]:
    try:
        if (candidate / "modeling_module").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break
    except Exception:
        continue

import matplotlib

matplotlib.use("Agg", force=True)
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from modeling_module import models as model_builders
from modeling_module.data_loader import MultiPartExoDataModule
from modeling_module.training.model_losses.loss_module import SpikeWeightedHuberLoss
from modeling_module.training.model_trainers.total_train import run_total_train_weekly
from modeling_module.utils.checkpoint import load_model_dict
from modeling_module.utils.exogenous_utils import compose_exo_calendar_cb
from model_test.future_exo_utils import (
    build_lookup_tuple,
    build_week_index,
    make_part_future_exo_fn,
)
from model_test.model_test_utils import (
    add_week,
    calc_accuracy,
    make_forecast_result_table,
)


BASE_COLS = ["part_no", "yyyyww", "demand_qty"]
PAST_EXO_COLS = ("in_wty_log", "out_wty_log")
PAST_EXO_CAT_COLS: Tuple[str, ...] = ()
FUTURE_EXO_BASE_COLS = [
    "warranty_end",
    "demand_log_ago",
    "order_cumsum_log",
    "warranty",
    "wty_ago27",
]
MODEL_NAME = "patchtst"
RUN_PATCHTST_QUANTILE = False
PATCHTST_FUTURE_EXO_FUSION_MODES = ("head_flatten", "token_cross_attn")
DEFAULT_PATCHTST_FUTURE_EXO_FUSION_MODE = "token_cross_attn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK = 52
HORIZON = 27
FREQ = "weekly"
SPLIT_MODE = "multi"
TRAIN_SHUFFLE = True
INFER_SHUFFLE = False
BASE_DIR = Path("E:/future_exo_test")
PLAN_WEEKS = list(range(202538, 202553))
BATCH_SIZES = [256, 128, 64]
SEEDS = [42, 32, 22]
MAX_PARTS = 10_000
WARMUP_EPOCHS = 2
SPIKE_EPOCHS = 1
SSL_PRETRAIN_EPOCHS = 1
LOSS = SpikeWeightedHuberLoss(delta=5.0)
SAMPLE_PART_COUNT = 50
SAMPLE_PLOT_NCOLS = 3

BUILD_PATCHTST = getattr(model_builders, "build_patchTST", None)
BUILD_PATCHTST_QUANTILE = getattr(model_builders, "build_patchTST_quantile", None)

PLANT_PARQUET_MAP = {
    "V101": Path("E:/V101_in_out_wty.parquet"),
    "V506": Path("E:/V506_in_out_wty.parquet"),
}

BASE_AB_CASES = {
    "past_o_future_o": {"use_past_exo": True, "use_future_exo": True},
    "past_o_future_x": {"use_past_exo": True, "use_future_exo": False},
    "past_x_future_o": {"use_past_exo": False, "use_future_exo": True},
    "past_x_future_x": {"use_past_exo": False, "use_future_exo": False},
}


def build_ab_cases() -> Dict[str, Dict[str, object]]:
    cases: Dict[str, Dict[str, object]] = {}
    for case_name, spec in BASE_AB_CASES.items():
        if bool(spec["use_future_exo"]):
            for fusion_mode in PATCHTST_FUTURE_EXO_FUSION_MODES:
                cases[f"{case_name}_{fusion_mode}"] = {
                    **spec,
                    "patchtst_future_exo_fusion_mode": fusion_mode,
                }
        else:
            cases[case_name] = {
                **spec,
                "patchtst_future_exo_fusion_mode": DEFAULT_PATCHTST_FUTURE_EXO_FUSION_MODE,
            }
    return cases


AB_CASES = build_ab_cases()


@dataclass(frozen=True)
class RunConfig:
    plant: str
    case_name: str
    use_past_exo: bool
    use_future_exo: bool
    batch_size: int
    seed: int
    lookback: int = LOOKBACK
    horizon: int = HORIZON
    freq: str = FREQ
    split_mode: str = SPLIT_MODE
    model_name: str = MODEL_NAME
    base_dir: str = str(BASE_DIR)
    plan_weeks: Tuple[int, ...] = tuple(PLAN_WEEKS)
    past_exo_cols: Tuple[str, ...] = PAST_EXO_COLS
    past_exo_cat_cols: Tuple[str, ...] = PAST_EXO_CAT_COLS
    future_exo_base_cols: Tuple[str, ...] = tuple(FUTURE_EXO_BASE_COLS)
    max_parts: int = MAX_PARTS
    warmup_epochs: int = WARMUP_EPOCHS
    spike_epochs: int = SPIKE_EPOCHS
    ssl_pretrain_epochs: int = SSL_PRETRAIN_EPOCHS
    sample_part_count: int = SAMPLE_PART_COUNT
    run_patchtst_quantile: bool = RUN_PATCHTST_QUANTILE
    patchtst_future_exo_fusion_mode: str = DEFAULT_PATCHTST_FUTURE_EXO_FUSION_MODE


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    parquet_dir: Path
    forecast_dir: Path
    metrics_dir: Path
    plots_dir: Path
    logs_dir: Path
    manifest_dir: Path


@dataclass(frozen=True)
class ExperimentObjects:
    future_exo_cb: Optional[Callable]
    part_future_exo_fn: Optional[Callable]
    past_exo_cols: Tuple[str, ...]
    past_exo_cat_cols: Tuple[str, ...]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_base_dirs(base_dir: Path) -> None:
    for sub in [
        base_dir,
        base_dir / "global_logs",
        base_dir / "global_summary",
        base_dir / "global_plots",
    ]:
        ensure_dir(sub)


def build_run_paths(cfg: RunConfig) -> RunPaths:
    run_dir = (
        Path(cfg.base_dir)
        / cfg.plant
        / cfg.case_name
        / f"batch_{cfg.batch_size}"
        / f"seed_{cfg.seed}"
    )
    return RunPaths(
        run_dir=ensure_dir(run_dir),
        checkpoints_dir=ensure_dir(run_dir / "checkpoints"),
        parquet_dir=ensure_dir(run_dir / "parquet"),
        forecast_dir=ensure_dir(run_dir / "parquet" / "forecast_by_plan"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        plots_dir=ensure_dir(run_dir / "plots"),
        logs_dir=ensure_dir(run_dir / "logs"),
        manifest_dir=ensure_dir(run_dir / "manifest"),
    )


def build_logger(log_path: Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8", errors="replace")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_plant_df(parquet_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    if "demand_cumsum" not in df.columns:
        raise ValueError(
            f"'demand_cumsum' column is missing: {parquet_path}. "
            "This preprocessing step depends on demand_cumsum to build demand_log_ago."
        )
    if "warranty" not in df.columns:
        raise ValueError(
            f"'warranty' column is missing: {parquet_path}. "
            "This preprocessing step depends on warranty to build demand_log_ago."
        )

    prepared = (
        df.with_columns(
            (
                -pl.col("demand_cumsum")
                .shift(pl.col("warranty").first() + 1)
                .over("part_no")
                .log1p()
                .alias("demand_log_ago")
            )
        )
        .fill_null(0.0)
        .with_columns(
            [
                pl.col("yyyyww").cast(pl.Int64).alias("yyyyww"),
                pl.col("demand_qty").cast(pl.Float64).alias("demand_qty"),
            ]
        )
        .select(BASE_COLS + list(PAST_EXO_COLS) + FUTURE_EXO_BASE_COLS)
        .sort(["part_no", "yyyyww"])
    )
    return prepared


def build_future_exo_components(
    df_for_lookup: pl.DataFrame,
    use_future_exo: bool,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    if not use_future_exo:
        return None, None

    lookup = build_lookup_tuple(
        df_for_lookup,
        "part_no",
        "yyyyww",
        FUTURE_EXO_BASE_COLS,
    )
    ww_list, ww_to_pos = build_week_index(df_for_lookup, "yyyyww")
    future_exo_cb = compose_exo_calendar_cb(date_type="weekly", sincos=True)
    part_future_exo_fn = make_part_future_exo_fn(lookup, ww_list, ww_to_pos)
    return future_exo_cb, part_future_exo_fn


def resolve_experiment_objects(
    train_df: pl.DataFrame,
    use_past_exo: bool,
    use_future_exo: bool,
) -> ExperimentObjects:
    past_exo_cols = PAST_EXO_COLS if use_past_exo else tuple()
    future_exo_cb, part_future_exo_fn = build_future_exo_components(
        train_df,
        use_future_exo=use_future_exo,
    )
    return ExperimentObjects(
        future_exo_cb=future_exo_cb,
        part_future_exo_fn=part_future_exo_fn,
        past_exo_cols=past_exo_cols,
        past_exo_cat_cols=PAST_EXO_CAT_COLS,
    )


def build_data_module(
    df: pl.DataFrame,
    batch_size: int,
    seed: int,
    shuffle: bool,
    past_exo_cols: Sequence[str],
    past_exo_cat_cols: Sequence[str],
    future_exo_cb: Optional[Callable],
    part_future_exo_fn: Optional[Callable],
) -> MultiPartExoDataModule:
    signature = inspect.signature(MultiPartExoDataModule)
    kwargs = {
        "df": df,
        "id_col": BASE_COLS[0],
        "date_col": BASE_COLS[1],
        "y_col": BASE_COLS[2],
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "batch_size": batch_size,
        "freq": FREQ,
        "shuffle": shuffle,
        "split_mode": SPLIT_MODE,
        "seed": seed,
    }

    if "past_exo_cont_cols" in signature.parameters:
        kwargs["past_exo_cont_cols"] = tuple(past_exo_cols)
    if "past_exo_cat_cols" in signature.parameters:
        kwargs["past_exo_cat_cols"] = tuple(past_exo_cat_cols)
    if "future_exo_cb" in signature.parameters:
        kwargs["future_exo_cb"] = future_exo_cb
    if "part_future_exo_fn" in signature.parameters:
        kwargs["part_future_exo_fn"] = part_future_exo_fn

    return MultiPartExoDataModule(**kwargs)


def run_training(
    train_df: pl.DataFrame,
    cfg: RunConfig,
    paths: RunPaths,
    logger: logging.Logger,
) -> None:
    logger.info("[TRAIN] building data module")
    objects = resolve_experiment_objects(
        train_df=train_df,
        use_past_exo=cfg.use_past_exo,
        use_future_exo=cfg.use_future_exo,
    )
    data_module = build_data_module(
        df=train_df,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=TRAIN_SHUFFLE,
        past_exo_cols=objects.past_exo_cols,
        past_exo_cat_cols=objects.past_exo_cat_cols,
        future_exo_cb=objects.future_exo_cb,
        part_future_exo_fn=objects.part_future_exo_fn,
    )

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    logger.info(
        "[TRAIN] start | plant=%s case=%s batch=%s seed=%s use_past_exo=%s "
        "use_future_exo=%s future_fusion=%s run_quantile=%s",
        cfg.plant,
        cfg.case_name,
        cfg.batch_size,
        cfg.seed,
        cfg.use_past_exo,
        cfg.use_future_exo,
        cfg.patchtst_future_exo_fusion_mode,
        cfg.run_patchtst_quantile,
    )

    signature = inspect.signature(run_total_train_weekly)
    models_to_run = (
        [cfg.model_name]
        if cfg.run_patchtst_quantile
        else ["patchtst_base" if cfg.model_name == "patchtst" else cfg.model_name]
    )
    model_architecture = {
        "patchtst": {
            "future_exo_fusion_mode": cfg.patchtst_future_exo_fusion_mode,
        }
    }
    kwargs = {
        "device": DEVICE,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "warmup_epochs": cfg.warmup_epochs,
        "spike_epochs": cfg.spike_epochs,
        "save_dir": str(paths.checkpoints_dir),
        "use_exogenous_mode": (cfg.use_past_exo or cfg.use_future_exo),
        "loss": LOSS,
        "models_to_run": models_to_run,
        "model_architecture": model_architecture,
        "use_ssl_mode": "full",
        "ssl_pretrain_epochs": cfg.ssl_pretrain_epochs,
    }

    if "use_past_exogenous" in signature.parameters:
        kwargs["use_past_exogenous"] = cfg.use_past_exo
    if "use_future_exogenous" in signature.parameters:
        kwargs["use_future_exogenous"] = cfg.use_future_exo
    if "allow_past_only_exogenous" in signature.parameters:
        kwargs["allow_past_only_exogenous"] = True

    if (
        cfg.use_past_exo
        and not cfg.use_future_exo
        and "use_future_exogenous" not in signature.parameters
        and "use_past_exogenous" not in signature.parameters
    ):
        logger.warning(
            "[TRAIN] trainer signature does not expose separate past/future exogenous flags. "
            "The past-only case may fail if exo_policy still requires future exo."
        )

    run_total_train_weekly(train_loader, val_loader, **kwargs)
    logger.info("[TRAIN] done")


def _first_existing_model(model_dict: Dict[str, object], candidate_keys: Sequence[str]):
    for key in candidate_keys:
        if key in model_dict and model_dict[key] is not None:
            return model_dict[key]
    return None


def load_models(checkpoints_dir: Path, logger: logging.Logger, *, load_quantile: bool):
    if BUILD_PATCHTST is None:
        raise RuntimeError("build_patchTST is not available in modeling_module.models")

    builders = {
        "patchtst": BUILD_PATCHTST,
        "PatchTST": BUILD_PATCHTST,
    }
    if load_quantile and BUILD_PATCHTST_QUANTILE is not None:
        builders["patchtst_quantile"] = BUILD_PATCHTST_QUANTILE
        builders["PatchTSTQuantile"] = BUILD_PATCHTST_QUANTILE

    model_dict = load_model_dict(str(checkpoints_dir), builders, device=DEVICE)
    checkpoint_files = sorted([p.name for p in checkpoints_dir.glob("*.pt")])
    logger.info("[LOAD] checkpoint dir=%s", checkpoints_dir)
    logger.info("[LOAD] checkpoint files=%s", checkpoint_files)
    logger.info("[LOAD] model keys=%s", list(model_dict.keys()))

    base_model = _first_existing_model(model_dict, ["patchtst", "PatchTST", MODEL_NAME])
    quantile_model = (
        _first_existing_model(
            model_dict,
            ["patchtst_quantile", "PatchTSTQuantile", "patchtst-quantile"],
        )
        if load_quantile
        else None
    )

    if base_model is None:
        raise ValueError(f"Could not find '{MODEL_NAME}' checkpoint in {checkpoints_dir}")

    return base_model, quantile_model


def make_result_table_compat(
    inference_loader,
    base_model,
    quantile_model,
    plan_week: int,
    horizon: int,
    include_quantile: bool,
):
    signature = inspect.signature(make_forecast_result_table)
    kwargs = {
        "inference_loader": inference_loader,
        "horizon": horizon,
        "device": DEVICE,
        "max_parts": MAX_PARTS,
    }

    if "plan_week" in signature.parameters:
        kwargs["plan_week"] = plan_week
    elif "plan_dt" in signature.parameters:
        kwargs["plan_dt"] = plan_week

    if "base_model" in signature.parameters:
        kwargs["base_model"] = base_model
    elif "model" in signature.parameters:
        kwargs["model"] = base_model
    else:
        raise ValueError(
            "Could not find a base model argument in make_forecast_result_table signature."
        )

    quantile_param = signature.parameters.get("quantile_model")
    if quantile_param is not None:
        if quantile_model is not None:
            kwargs["quantile_model"] = quantile_model
        elif quantile_param.default is inspect._empty:
            # Base-only PatchTST AB tests intentionally skip quantile training.
            # The legacy helper still requires a quantile_model argument, so pass
            # the point model and blank the quantile output immediately after.
            kwargs["quantile_model"] = base_model

    out = make_forecast_result_table(**kwargs)
    if not include_quantile and "quantile_forecast" in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("quantile_forecast"))
    return out


def postprocess_forecast_df(
    result_df: pl.DataFrame,
    target_df: pl.DataFrame,
    plan_week: int,
) -> pl.DataFrame:
    rename_map = {
        "oper_part_no": "part_no",
        "forecast_week": "yyyyww",
    }
    forecast_df = result_df.select(
        [
            c
            for c in result_df.columns
            if c
            in {"oper_part_no", "plan_week", "forecast_week", "base_forecast", "quantile_forecast"}
        ]
    ).rename(rename_map)

    if "quantile_forecast" not in forecast_df.columns:
        forecast_df = forecast_df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("quantile_forecast")
        )

    final_df = (
        forecast_df.join(target_df, on=["part_no", "yyyyww"], how="left")
        .with_columns(
            [
                pl.when(pl.col("base_forecast") < 0)
                .then(0.0)
                .otherwise(pl.col("base_forecast"))
                .alias("base_forecast"),
                pl.when(
                    pl.col("quantile_forecast").is_not_null()
                    & (pl.col("quantile_forecast") < 0)
                )
                .then(0.0)
                .otherwise(pl.col("quantile_forecast"))
                .alias("quantile_forecast"),
                pl.col("demand_qty").fill_null(0.0).alias("demand_qty"),
                pl.lit(plan_week).cast(pl.Int64).alias("plan_week"),
                pl.col("yyyyww").cast(pl.Int64).alias("yyyyww"),
            ]
        )
        .sort(["part_no", "yyyyww"])
    )
    return final_df


def sample_parts_for_plot(
    target_df: pl.DataFrame,
    id_col: str = "part_no",
    y_col: str = "demand_qty",
    n_samples: int = 50,
    seed: int = 42,
    min_nonzero_count: int = 1,
) -> List[str]:
    candidates = (
        target_df.group_by(id_col)
        .agg(
            [
                pl.len().alias("n_obs"),
                (pl.col(y_col) > 0).sum().alias("n_nonzero"),
                pl.col(y_col).sum().alias("sum_qty"),
            ]
        )
        .filter(pl.col("n_nonzero") >= min_nonzero_count)
        .sort(id_col)
    )

    part_list = candidates.get_column(id_col).to_list()
    if len(part_list) == 0:
        raise ValueError("No candidate parts found for plotting.")

    sample_size = min(n_samples, len(part_list))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(part_list, size=sample_size, replace=False)
    return list(sampled)


def save_sampled_parts_manifest(
    sampled_parts: List[str],
    save_path: Path,
    id_col: str = "part_no",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({id_col: sampled_parts}).write_parquet(str(save_path))


def save_sample_lineplots_for_plan_week_long_3col(
    final_df: pl.DataFrame,
    sampled_parts: List[str],
    plan_week: int,
    save_dir: Path,
    id_col: str = "part_no",
    date_col: str = "yyyyww",
    actual_col: str = "demand_qty",
    base_fcst_col: str = "base_forecast",
    quantile_fcst_col: str = "quantile_forecast",
    ncols: int = 3,
    subplot_width: float = 6.0,
    subplot_height: float = 3.0,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_df = (
        final_df.filter(pl.col(id_col).is_in(sampled_parts))
        .with_columns(pl.col(date_col).cast(pl.Int64))
        .sort([id_col, date_col])
    )

    if plot_df.height == 0:
        return

    plot_parts = (
        plot_df.select(id_col).unique().sort(id_col).get_column(id_col).to_list()
    )

    n_parts = len(plot_parts)
    nrows = math.ceil(n_parts / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(subplot_width * ncols, subplot_height * nrows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax_idx, part_no in enumerate(plot_parts):
        ax = axes_flat[ax_idx]

        part_df = (
            plot_df.filter(pl.col(id_col) == part_no)
            .select([date_col, actual_col, base_fcst_col, quantile_fcst_col])
            .sort(date_col)
            .to_pandas()
        )

        x = part_df[date_col].astype(str)
        ax.plot(x, part_df[actual_col], marker="o", label="actual")
        ax.plot(x, part_df[base_fcst_col], marker="o", label="base_forecast")

        if quantile_fcst_col in part_df.columns and part_df[quantile_fcst_col].notna().any():
            ax.plot(
                x,
                part_df[quantile_fcst_col],
                marker="o",
                linestyle="--",
                label="quantile_forecast",
            )

        ax.set_title(str(part_no), fontsize=10)
        ax.set_xlabel("yyyyww")
        ax.set_ylabel("qty")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for empty_idx in range(n_parts, len(axes_flat)):
        fig.delaxes(axes_flat[empty_idx])

    fig.suptitle(f"Plan Week {plan_week} | Sampled Parts Actual vs Forecast", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = save_dir / f"plan_{plan_week}_sample_parts_3col.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def infer_one_plan(
    full_df: pl.DataFrame,
    plan_week: int,
    cfg: RunConfig,
    base_model,
    quantile_model,
    paths: RunPaths,
    logger: logging.Logger,
    sampled_parts: Optional[List[str]] = None,
) -> pl.DataFrame:
    inference_history_df = full_df.filter(pl.col("yyyyww") < plan_week)
    objects = resolve_experiment_objects(
        train_df=inference_history_df,
        use_past_exo=cfg.use_past_exo,
        use_future_exo=cfg.use_future_exo,
    )
    data_module = build_data_module(
        df=full_df,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=INFER_SHUFFLE,
        past_exo_cols=objects.past_exo_cols,
        past_exo_cat_cols=objects.past_exo_cat_cols,
        future_exo_cb=objects.future_exo_cb,
        part_future_exo_fn=objects.part_future_exo_fn,
    )
    inference_loader = data_module.get_inference_loader_at_plan(plan_dt=plan_week)

    result_df = make_result_table_compat(
        inference_loader=inference_loader,
        base_model=base_model,
        quantile_model=quantile_model,
        plan_week=plan_week,
        horizon=cfg.horizon,
        include_quantile=cfg.run_patchtst_quantile,
    )

    target_df = full_df.select(["part_no", "yyyyww", "demand_qty"])
    final_df = postprocess_forecast_df(
        result_df=result_df,
        target_df=target_df,
        plan_week=plan_week,
    )

    plan_result_path = paths.forecast_dir / f"final_df_{plan_week}.parquet"
    raw_result_path = paths.forecast_dir / f"raw_result_df_{plan_week}.parquet"
    final_df.write_parquet(plan_result_path)
    result_df.write_parquet(raw_result_path)

    if sampled_parts:
        save_sample_lineplots_for_plan_week_long_3col(
            final_df=final_df,
            sampled_parts=sampled_parts,
            plan_week=plan_week,
            save_dir=paths.plots_dir / "sample_lineplots",
            id_col="part_no",
            date_col="yyyyww",
            actual_col="demand_qty",
            base_fcst_col="base_forecast",
            quantile_fcst_col="quantile_forecast",
            ncols=SAMPLE_PLOT_NCOLS,
        )

    logger.info("[INFER] saved plan_week=%s -> %s", plan_week, plan_result_path)
    return final_df


def load_plan_forecasts(forecast_dir: Path, plan_weeks: Sequence[int]) -> Dict[int, pl.DataFrame]:
    dfs: Dict[int, pl.DataFrame] = {}
    for plan_week in plan_weeks:
        path = forecast_dir / f"final_df_{plan_week}.parquet"
        if path.exists():
            dfs[plan_week] = pl.read_parquet(path)
    return dfs


def build_revision_panel(
    forecast_dfs: Dict[int, pl.DataFrame],
    target_week: int,
    revisions: int = 8,
) -> pl.DataFrame:
    exclude_id = "part_no"
    base_df = None

    for rev_idx, delta in enumerate(range(revisions - 1, -1, -1), start=1):
        src_plan_week = add_week(target_week, -delta)
        if src_plan_week not in forecast_dfs:
            raise KeyError(
                f"Failed to build revision panel: missing plan_week={src_plan_week}."
            )

        src = (
            forecast_dfs[src_plan_week]
            .filter(pl.col("yyyyww") == target_week)
            .select([
                "part_no",
                "plan_week",
                "yyyyww",
                "base_forecast",
                "quantile_forecast",
                "demand_qty",
            ])
        )
        rename_map = {c: f"{rev_idx}_{c}" for c in src.columns if c != exclude_id}
        src = src.rename(rename_map)

        if base_df is None:
            base_df = src
        else:
            base_df = base_df.join(src, on="part_no", how="left")

    if base_df is None:
        raise ValueError(f"Could not build revision panel for target_week={target_week}.")

    demand_cols = [c for c in base_df.columns if c.endswith("_demand_qty")]
    if "2_demand_qty" in base_df.columns:
        base_df = base_df.rename({"2_demand_qty": "demand_qty"})
        demand_cols = [c for c in demand_cols if c != "2_demand_qty"]
    elif demand_cols:
        first_demand = demand_cols[0]
        base_df = base_df.rename({first_demand: "demand_qty"})
        demand_cols = [c for c in demand_cols if c != first_demand]

    if demand_cols:
        base_df = base_df.drop(demand_cols)

    return base_df


def aggregate_far_from_accuracy(df_acc: pl.DataFrame) -> pl.DataFrame:
    forecast_cols = [c for c in df_acc.columns if c.endswith("_base_forecast")]
    if not forecast_cols:
        raise ValueError("No '_base_forecast' columns found for FAR aggregation.")
    if "accu" not in df_acc.columns:
        raise ValueError("'accu' column is missing from calc_accuracy output.")
    if "fcst_qty" not in df_acc.columns:
        raise ValueError("'fcst_qty' column is missing from calc_accuracy output.")

    nonzero_filter = None
    for c in forecast_cols:
        cond = pl.col(c) != 0
        nonzero_filter = cond if nonzero_filter is None else (nonzero_filter & cond)

    filtered = df_acc.filter(nonzero_filter) if nonzero_filter is not None else df_acc
    if filtered.height == 0:
        return pl.DataFrame(
            {
                "weighted_far": [None],
                "fcst_qty_total": [0.0],
                "row_count": [df_acc.height],
                "nonzero_row_count": [0],
            }
        )

    fcst_qty_total = filtered.select(pl.col("fcst_qty").sum()).item()
    weighted_far = (
        filtered.with_columns(
            (
                pl.col("accu") * (pl.col("fcst_qty") / pl.col("fcst_qty").sum())
            ).alias("weighted_acc_part")
        )
        .select(pl.col("weighted_acc_part").sum())
        .item()
    )

    return pl.DataFrame(
        {
            "weighted_far": [weighted_far],
            "fcst_qty_total": [float(fcst_qty_total)],
            "row_count": [df_acc.height],
            "nonzero_row_count": [filtered.height],
        }
    )


def evaluate_far(
    forecast_dfs: Dict[int, pl.DataFrame],
    cfg: RunConfig,
    paths: RunPaths,
    logger: logging.Logger,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    target_weeks = list(range(cfg.plan_weeks[0] + 7, cfg.plan_weeks[-1] + 1))
    far_rows: List[pl.DataFrame] = []
    revision_panels: List[pl.DataFrame] = []

    for target_week in target_weeks:
        panel_df = build_revision_panel(
            forecast_dfs=forecast_dfs,
            target_week=target_week,
            revisions=8,
        )
        panel_df = panel_df.with_columns(pl.lit(target_week).alias("target_week"))
        panel_path = paths.parquet_dir / f"revision_panel_{target_week}.parquet"
        panel_df.write_parquet(panel_path)
        revision_panels.append(panel_df)

        df_acc = calc_accuracy(panel_df)
        df_far = aggregate_far_from_accuracy(df_acc).with_columns(
            [
                pl.lit(target_week).cast(pl.Int64).alias("target_week"),
                pl.lit(cfg.plant).alias("plant"),
                pl.lit(cfg.case_name).alias("case_name"),
                pl.lit(cfg.batch_size).cast(pl.Int64).alias("batch_size"),
                pl.lit(cfg.seed).cast(pl.Int64).alias("seed"),
            ]
        )
        far_rows.append(df_far)

    far_by_week = pl.concat(far_rows) if far_rows else pl.DataFrame()
    revision_concat = pl.concat(revision_panels) if revision_panels else pl.DataFrame()

    far_by_week.write_parquet(paths.metrics_dir / "far_by_target_week.parquet")
    revision_concat.write_parquet(paths.parquet_dir / "revision_panels_all.parquet")

    summary = (
        far_by_week.select(
            [
                pl.col("plant").first().alias("plant"),
                pl.col("case_name").first().alias("case_name"),
                pl.col("batch_size").first().alias("batch_size"),
                pl.col("seed").first().alias("seed"),
                pl.col("weighted_far").mean().alias("far_mean"),
                pl.col("weighted_far").median().alias("far_median"),
                pl.col("weighted_far").min().alias("far_min"),
                pl.col("weighted_far").max().alias("far_max"),
                pl.col("fcst_qty_total").sum().alias("fcst_qty_total_sum"),
                pl.len().alias("eval_target_week_count"),
            ]
        )
        if far_by_week.height > 0
        else pl.DataFrame()
    )
    if summary.height > 0:
        summary.write_parquet(paths.metrics_dir / "far_summary.parquet")

    logger.info("[EVAL] FAR evaluation done")
    return far_by_week, summary


def _safe_series_from_polars(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pdf = df.select([x_col, y_col]).to_pandas()
    x = pdf[x_col].to_numpy()
    y = pdf[y_col].to_numpy()
    return x, y


def plot_far_trend(far_by_week: pl.DataFrame, save_path: Path) -> None:
    if far_by_week.height == 0:
        return
    x, y = _safe_series_from_polars(
        far_by_week.sort("target_week"),
        "target_week",
        "weighted_far",
    )
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker="o")
    plt.title("FAR Trend by Target Week")
    plt.xlabel("target_week")
    plt.ylabel("weighted_far")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_inflection_summary(
    forecast_dfs: Dict[int, pl.DataFrame],
    target_weeks: Sequence[int],
) -> pl.DataFrame:
    rows = []
    for target_week in target_weeks:
        base_plan_week = target_week
        if base_plan_week not in forecast_dfs:
            continue
        df = forecast_dfs[base_plan_week].filter(pl.col("yyyyww") == target_week)
        if df.height == 0:
            continue
        rows.append(
            pl.DataFrame(
                {
                    "target_week": [target_week],
                    "actual_sum": [df.select(pl.col("demand_qty").sum()).item()],
                    "base_forecast_sum": [df.select(pl.col("base_forecast").sum()).item()],
                    "quantile_forecast_sum": [
                        df.select(pl.col("quantile_forecast").fill_null(0.0).sum()).item()
                    ],
                }
            )
        )
    return pl.concat(rows) if rows else pl.DataFrame()


def plot_inflection_summary(inflection_df: pl.DataFrame, save_path: Path) -> None:
    if inflection_df.height == 0:
        return
    pdf = inflection_df.sort("target_week").to_pandas()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(pdf["target_week"], pdf["actual_sum"], marker="o", label="actual_sum")
    plt.plot(
        pdf["target_week"],
        pdf["base_forecast_sum"],
        marker="o",
        label="base_forecast_sum",
    )
    if "quantile_forecast_sum" in pdf.columns and not np.allclose(
        pdf["quantile_forecast_sum"].to_numpy(),
        0.0,
    ):
        plt.plot(
            pdf["target_week"],
            pdf["quantile_forecast_sum"],
            marker="o",
            label="quantile_forecast_sum",
        )
    plt.title("Inflection Point Check (Aggregate)")
    plt.xlabel("target_week")
    plt.ylabel("sum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_run_manifest(cfg: RunConfig, paths: RunPaths) -> None:
    save_json(asdict(cfg), paths.manifest_dir / "run_config.json")


def summarize_all_runs(base_dir: Path) -> None:
    summary_files = list(base_dir.rglob("far_summary.parquet"))
    if not summary_files:
        return
    summaries = [pl.read_parquet(path) for path in summary_files]
    summary_df = pl.concat(summaries)
    global_summary_dir = ensure_dir(base_dir / "global_summary")
    global_plots_dir = ensure_dir(base_dir / "global_plots")
    summary_df.write_parquet(global_summary_dir / "all_run_far_summary.parquet")

    grouped = (
        summary_df.group_by(["plant", "case_name", "batch_size"])
        .agg(
            [
                pl.col("far_mean").mean().alias("far_mean_avg"),
                pl.col("far_mean").std().alias("far_mean_std"),
            ]
        )
        .sort(["plant", "case_name", "batch_size"])
    )
    grouped.write_parquet(global_summary_dir / "all_run_far_grouped.parquet")

    for plant in grouped["plant"].unique().to_list():
        pdf = grouped.filter(pl.col("plant") == plant).to_pandas()
        if pdf.empty:
            continue
        fig = plt.figure(figsize=(14, 6))
        for case_name, sub in pdf.groupby("case_name"):
            plt.plot(sub["batch_size"], sub["far_mean_avg"], marker="o", label=case_name)
        plt.title(f"{plant} | FAR Mean by Batch Size (seed average)")
        plt.xlabel("batch_size")
        plt.ylabel("far_mean_avg")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig.savefig(global_plots_dir / f"{plant}_far_by_batch_case.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_single_experiment(plant: str, case_name: str, batch_size: int, seed: int) -> None:
    cfg = RunConfig(
        plant=plant,
        case_name=case_name,
        use_past_exo=AB_CASES[case_name]["use_past_exo"],
        use_future_exo=AB_CASES[case_name]["use_future_exo"],
        batch_size=batch_size,
        seed=seed,
        run_patchtst_quantile=RUN_PATCHTST_QUANTILE,
        patchtst_future_exo_fusion_mode=str(
            AB_CASES[case_name]["patchtst_future_exo_fusion_mode"]
        ),
    )
    paths = build_run_paths(cfg)
    logger = build_logger(
        paths.logs_dir / "run.log",
        logger_name=f"future_exo_ab.{plant}.{case_name}.{batch_size}.{seed}",
    )
    save_run_manifest(cfg, paths)

    logger.info("=" * 120)
    logger.info("experiment start | %s", asdict(cfg))

    full_df = load_plant_df(PLANT_PARQUET_MAP[plant])
    train_cutoff = cfg.plan_weeks[0]
    train_df = full_df.filter(pl.col("yyyyww") < train_cutoff)

    if train_df.height == 0:
        raise ValueError(
            f"train_df is empty. plant={plant}, train_cutoff={train_cutoff}"
        )

    target_df = full_df.select(["part_no", "yyyyww", "demand_qty"])
    sampled_parts = sample_parts_for_plot(
        target_df=target_df,
        id_col="part_no",
        y_col="demand_qty",
        n_samples=cfg.sample_part_count,
        seed=cfg.seed,
        min_nonzero_count=1,
    )
    save_sampled_parts_manifest(
        sampled_parts=sampled_parts,
        save_path=paths.manifest_dir / "sampled_parts_for_plot.parquet",
        id_col="part_no",
    )

    logger.info(
        "data ready | full_rows=%s train_rows=%s unique_parts=%s sampled_parts=%s",
        full_df.height,
        train_df.height,
        train_df.select(pl.col("part_no").n_unique()).item(),
        len(sampled_parts),
    )

    run_training(train_df=train_df, cfg=cfg, paths=paths, logger=logger)
    base_model, quantile_model = load_models(
        paths.checkpoints_dir,
        logger=logger,
        load_quantile=cfg.run_patchtst_quantile,
    )

    for plan_week in cfg.plan_weeks:
        infer_one_plan(
            full_df=full_df,
            plan_week=plan_week,
            cfg=cfg,
            base_model=base_model,
            quantile_model=quantile_model,
            paths=paths,
            logger=logger,
            sampled_parts=sampled_parts,
        )

    forecast_dfs = load_plan_forecasts(paths.forecast_dir, cfg.plan_weeks)
    far_by_week, far_summary = evaluate_far(
        forecast_dfs=forecast_dfs,
        cfg=cfg,
        paths=paths,
        logger=logger,
    )

    target_weeks = list(range(cfg.plan_weeks[0] + 7, cfg.plan_weeks[-1] + 1))
    inflection_df = build_inflection_summary(forecast_dfs, target_weeks)
    if inflection_df.height > 0:
        inflection_df.write_parquet(paths.metrics_dir / "inflection_summary.parquet")

    plot_far_trend(far_by_week, paths.plots_dir / "far_trend.png")
    plot_inflection_summary(inflection_df, paths.plots_dir / "inflection_summary.png")

    if far_summary.height > 0:
        logger.info("FAR summary: %s", far_summary.to_dicts())
    logger.info("experiment done")


def main() -> None:
    setup_base_dirs(BASE_DIR)
    for plant in PLANT_PARQUET_MAP:
        for case_name in AB_CASES:
            for batch_size in BATCH_SIZES:
                for seed in SEEDS:
                    run_single_experiment(
                        plant=plant,
                        case_name=case_name,
                        batch_size=batch_size,
                        seed=seed,
                    )
    summarize_all_runs(BASE_DIR)


if __name__ == "__main__":
    main()
