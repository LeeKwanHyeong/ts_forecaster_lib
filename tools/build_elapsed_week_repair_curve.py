from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl


warnings.filterwarnings(
    "ignore",
    message="Sortedness of columns cannot be checked when 'by' groups provided",
)


DEFAULT_DEMAND = Path("raw_data/raw/tb_dyn_demand_dtl.parquet")
DEFAULT_SALES = Path("raw_data/raw/tb_dyn_sales_parts.parquet")
DEFAULT_OPER = Path("raw_data/raw/tb_mst_oper_part.parquet")
DEFAULT_OUT_DIR = Path("raw_data/derived")


@dataclass(frozen=True)
class InputPaths:
    demand: Path
    sales: Path
    oper: Path
    out_dir: Path


def _parse_args() -> InputPaths:
    parser = argparse.ArgumentParser(
        description=(
            "Build elapsed-week repair incidence curves from sales/demand/master parquet files. "
            "The output is a practical proxy for defect rate, not a serial-level failure probability."
        )
    )
    parser.add_argument("--demand", type=Path, default=DEFAULT_DEMAND)
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    parser.add_argument("--oper", type=Path, default=DEFAULT_OPER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    return InputPaths(
        demand=args.demand,
        sales=args.sales,
        oper=args.oper,
        out_dir=args.out_dir,
    )


def _yearweek_to_monday(yearweek: int) -> date:
    year = int(yearweek) // 100
    week = int(yearweek) % 100
    return date.fromisocalendar(year, week, 1)


def _build_week_index_map(all_weeks: list[int]) -> dict[int, int]:
    week_dates = sorted(_yearweek_to_monday(int(w)) for w in set(all_weeks))
    if not week_dates:
        return {}

    cur = min(week_dates)
    end = max(week_dates)
    dense_weeks: list[int] = []
    while cur <= end:
        iso = cur.isocalendar()
        dense_weeks.append((iso.year * 100) + iso.week)
        cur = date.fromordinal(cur.toordinal() + 7)
    return {w: i for i, w in enumerate(dense_weeks)}


def _load_inputs(paths: InputPaths) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    oper = pl.read_parquet(paths.oper).select(
        ["oper_part_no", "part_fam_cd", "demand_start_dt", "demand_end_dt", "warranty"]
    )
    sales = pl.read_parquet(paths.sales).select(["oper_part_no", "sales_dt", "sales_qty"])
    demand = pl.read_parquet(paths.demand).select(["oper_part_no", "demand_dt", "demand_qty"])
    return oper, sales, demand


def _prepare_week_features(
    oper: pl.DataFrame,
    sales: pl.DataFrame,
    demand: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    week_map = _build_week_index_map(
        sales["sales_dt"].to_list()
        + demand["demand_dt"].to_list()
        + oper["demand_start_dt"].to_list()
        + oper["demand_end_dt"].to_list()
    )
    if not week_map:
        raise ValueError("No week keys found in the input parquet files.")

    oper_w = oper.with_columns(
        [
            pl.col("demand_start_dt").replace_strict(week_map).alias("start_wk_idx"),
            pl.col("demand_end_dt").replace_strict(week_map).alias("end_wk_idx"),
            (pl.col("warranty") * 52 / 12).round(0).cast(pl.Int64).alias("warranty_weeks"),
        ]
    )

    sales_w = (
        sales.join(oper_w, on="oper_part_no", how="inner")
        .with_columns(pl.col("sales_dt").replace_strict(week_map).alias("wk_idx"))
        .group_by(
            ["oper_part_no", "part_fam_cd", "warranty", "warranty_weeks", "start_wk_idx", "end_wk_idx", "wk_idx"]
        )
        .agg(pl.col("sales_qty").sum().alias("sales_qty"))
        .sort(["oper_part_no", "wk_idx"])
    )

    sales_cum = sales_w.with_columns(
        [
            pl.col("sales_qty").cum_sum().over("oper_part_no").alias("cum_sales"),
        ]
    ).select(["oper_part_no", "wk_idx", "cum_sales"])

    demand_w = (
        demand.join(oper_w, on="oper_part_no", how="inner")
        .with_columns(pl.col("demand_dt").replace_strict(week_map).alias("wk_idx"))
        .group_by(
            ["oper_part_no", "part_fam_cd", "warranty", "warranty_weeks", "start_wk_idx", "end_wk_idx", "wk_idx"]
        )
        .agg(pl.col("demand_qty").sum().alias("demand_qty"))
        .sort(["oper_part_no", "wk_idx"])
    )

    demand_with_cum = demand_w.join_asof(
        sales_cum.sort(["oper_part_no", "wk_idx"]),
        on="wk_idx",
        by="oper_part_no",
        strategy="backward",
    ).rename({"cum_sales": "cum_sales_now"})

    sales_prev = sales_cum.rename({"wk_idx": "cutoff_wk_idx", "cum_sales": "cum_sales_prev"}).sort(
        ["oper_part_no", "cutoff_wk_idx"]
    )
    demand_with_prev = demand_with_cum.with_columns(
        (pl.col("wk_idx") - pl.col("warranty_weeks")).alias("cutoff_wk_idx")
    ).join_asof(
        sales_prev,
        on="cutoff_wk_idx",
        by="oper_part_no",
        strategy="backward",
    )

    part_week = (
        demand_with_prev.with_columns(
            [
                (pl.col("wk_idx") - pl.col("start_wk_idx")).alias("elapsed_week"),
                (pl.col("cum_sales_now").fill_null(0) - pl.col("cum_sales_prev").fill_null(0))
                .clip(lower_bound=0)
                .alias("active_warranty_base"),
                pl.col("cum_sales_now").fill_null(0).alias("cum_sales_total"),
            ]
        )
        .filter(pl.col("elapsed_week") >= 0)
        .with_columns(
            [
                pl.when(pl.col("active_warranty_base") > 0)
                .then(pl.col("demand_qty") / pl.col("active_warranty_base"))
                .otherwise(None)
                .alias("repair_incidence_rate"),
                pl.when(pl.col("warranty_weeks") > 0)
                .then(pl.col("elapsed_week") / pl.col("warranty_weeks"))
                .otherwise(None)
                .alias("warranty_progress"),
            ]
        )
    )

    group_curve = (
        part_week.group_by(["part_fam_cd", "warranty", "warranty_weeks", "elapsed_week"])
        .agg(
            [
                pl.col("demand_qty").sum().alias("demand_qty"),
                pl.col("active_warranty_base").sum().alias("active_warranty_base"),
                pl.col("cum_sales_total").sum().alias("cum_sales_total"),
                pl.col("oper_part_no").n_unique().alias("n_parts"),
                pl.len().alias("n_obs"),
            ]
        )
        .sort(["part_fam_cd", "warranty", "elapsed_week"])
        .with_columns(
            [
                pl.when(pl.col("active_warranty_base") > 0)
                .then(pl.col("demand_qty") / pl.col("active_warranty_base"))
                .otherwise(None)
                .alias("repair_incidence_rate"),
                pl.when(pl.col("cum_sales_total") > 0)
                .then(pl.col("demand_qty") / pl.col("cum_sales_total"))
                .otherwise(None)
                .alias("installed_base_rate"),
                pl.when(pl.col("warranty_weeks") > 0)
                .then(pl.col("elapsed_week") / pl.col("warranty_weeks"))
                .otherwise(None)
                .alias("warranty_progress"),
            ]
        )
        .with_columns(
            [
                pl.col("repair_incidence_rate")
                .rolling_mean(window_size=5, min_samples=1)
                .over(["part_fam_cd", "warranty"])
                .alias("repair_incidence_rate_smooth_5w")
            ]
        )
    )

    return part_week, group_curve


def _build_summary(curve: pl.DataFrame) -> pl.DataFrame:
    stable = curve.filter((pl.col("n_parts") >= 100) & (pl.col("active_warranty_base") > 0))
    summary = (
        stable.group_by(["part_fam_cd", "warranty", "warranty_weeks"])
        .agg(
            [
                pl.col("elapsed_week").min().alias("min_elapsed_week"),
                pl.col("elapsed_week").max().alias("max_elapsed_week"),
                pl.col("repair_incidence_rate_smooth_5w").max().alias("max_smooth_rate"),
                pl.col("elapsed_week").sort_by("repair_incidence_rate_smooth_5w").last().alias("peak_elapsed_week"),
                pl.col("repair_incidence_rate_smooth_5w").head(8).mean().alias("early_8w_mean"),
                pl.col("repair_incidence_rate_smooth_5w").tail(8).mean().alias("late_8w_mean"),
                pl.col("n_parts").max().alias("max_parts_supported"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("warranty_weeks") > 0)
                .then(pl.col("peak_elapsed_week") / pl.col("warranty_weeks"))
                .otherwise(None)
                .alias("peak_vs_warranty_ratio")
            ]
        )
        .sort(["part_fam_cd", "warranty"])
    )
    return summary


def main() -> None:
    paths = _parse_args()
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    oper, sales, demand = _load_inputs(paths)
    part_week, curve = _prepare_week_features(oper, sales, demand)
    summary = _build_summary(curve)

    part_week_path = paths.out_dir / "part_week_repair_incidence.parquet"
    curve_path = paths.out_dir / "elapsed_week_repair_curve_by_group.parquet"
    summary_path = paths.out_dir / "elapsed_week_repair_curve_summary.parquet"

    part_week.write_parquet(part_week_path)
    curve.write_parquet(curve_path)
    summary.write_parquet(summary_path)

    print(f"[saved] {part_week_path}")
    print(f"[saved] {curve_path}")
    print(f"[saved] {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
