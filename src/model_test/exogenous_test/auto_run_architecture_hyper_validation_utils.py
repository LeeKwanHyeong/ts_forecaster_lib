from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import polars as pl

from model_test.model_test_utils import calc_accuracy


def load_plan_forecasts(forecast_dir, plan_weeks: Sequence[int]) -> Dict[int, pl.DataFrame]:
    """Load per-plan forecast parquet files that were written during inference."""
    dfs: Dict[int, pl.DataFrame] = {}
    for plan_week in plan_weeks:
        path = forecast_dir / f"final_df_{plan_week}.parquet"
        if path.exists():
            dfs[plan_week] = pl.read_parquet(path)
    return dfs


def build_revision_panel(
    forecast_dfs: Dict[int, pl.DataFrame],
    ordered_plan_weeks: Sequence[int],
    target_week: int,
    revisions: int = 8,
) -> pl.DataFrame:
    """
    Build a revision panel for a single target week.

    The panel contains the last `revisions` forecasts that targeted the same
    week so downstream FAR logic can compare forecast revisions consistently.
    """
    exclude_id = "part_no"
    base_df = None

    ordered_plan_weeks = list(ordered_plan_weeks)
    if target_week not in ordered_plan_weeks:
        raise KeyError(f"target_week={target_week} is not in ordered_plan_weeks")

    target_idx = ordered_plan_weeks.index(target_week)
    if target_idx < revisions - 1:
        raise ValueError(
            f"Not enough prior plan weeks to build revision panel for target_week={target_week}"
        )

    revision_plan_weeks = ordered_plan_weeks[target_idx - (revisions - 1) : target_idx + 1]

    for rev_idx, src_plan_week in enumerate(revision_plan_weeks, start=1):
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
    """
    Turn row-level accuracy output into a weighted FAR summary.

    Only rows with non-zero forecast across all revision columns are used in the
    weighted aggregation so we avoid dividing importance into zero-volume rows.
    """
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
    cfg,
    paths,
    logger,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Evaluate FAR over the configured target-week range.

    Notes:
    - The first 7 plan weeks are skipped because an 8-revision panel is required.
    - The function persists both detailed-by-week output and the run-level FAR summary.
    """
    target_weeks = list(cfg.plan_weeks[7:])
    far_rows: List[pl.DataFrame] = []
    revision_panels: List[pl.DataFrame] = []

    for target_week in target_weeks:
        panel_df = build_revision_panel(
            forecast_dfs=forecast_dfs,
            ordered_plan_weeks=cfg.plan_weeks,
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
                pl.lit(cfg.ab_case_name).alias("ab_case_name"),
                pl.lit(cfg.arch_case_name).alias("arch_case_name"),
                pl.lit(cfg.batch_size).cast(pl.Int64).alias("batch_size"),
                pl.lit(cfg.seed).cast(pl.Int64).alias("seed"),
                pl.lit(cfg.d_model).cast(pl.Int64).alias("d_model"),
                pl.lit(cfg.n_layers).cast(pl.Int64).alias("n_layers"),
                pl.lit(cfg.d_ff).cast(pl.Int64).alias("d_ff"),
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
                pl.col("ab_case_name").first().alias("ab_case_name"),
                pl.col("arch_case_name").first().alias("arch_case_name"),
                pl.col("batch_size").first().alias("batch_size"),
                pl.col("seed").first().alias("seed"),
                pl.col("d_model").first().alias("d_model"),
                pl.col("n_layers").first().alias("n_layers"),
                pl.col("d_ff").first().alias("d_ff"),
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


def build_inflection_summary(
    forecast_dfs: Dict[int, pl.DataFrame],
    target_weeks: Sequence[int],
) -> pl.DataFrame:
    """
    Build aggregate actual/forecast sums for each target week.

    This summary is later plotted to inspect whether the forecast captures
    inflection or level changes at the portfolio level.
    """
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
