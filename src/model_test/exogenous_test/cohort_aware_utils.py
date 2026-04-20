from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import polars as pl


DEFAULT_ANALYTICS_FILENAMES = {
    "acf": "acf_memory_length_estimator.parquet",
    "calendar": "calendar_effect_extractor.parquet",
    "change_point": "change_point_detector.parquet",
    "part_cluster": "clustering_profiler_part_cluster.parquet",
    "selector": "forecast_model_selector.parquet",
    "intermittent": "intermittent_detector.parquet",
    "level": "level_scale_volatility_profiler.parquet",
    "lifecycle": "lifecycle_stage_detector.parquet",
    "obsolescence": "obsolescence_risk_scorer.parquet",
    "outlier": "outlier_spike_detector.parquet",
    "pattern": "pattern_similarity_embedding.parquet",
    "seasonality": "seasonality_detector_weekly.parquet",
    "trend": "trend_strength_analyzer.parquet",
}


def get_default_analytics_paths(repo_root: str | Path) -> dict[str, Path]:
    analytics_root = Path(repo_root) / "raw_data" / "analytics"
    return {key: analytics_root / filename for key, filename in DEFAULT_ANALYTICS_FILENAMES.items()}


def add_quantile_bucket(df: pl.DataFrame, col: str, out_col: str, labels: Sequence[str]) -> pl.DataFrame:
    valid = df.select(pl.col(col).drop_nulls())
    if valid.height == 0:
        return df.with_columns(pl.lit("missing").alias(out_col))

    qs = valid.select(
        [
            pl.col(col).quantile(0.25).alias("q1"),
            pl.col(col).quantile(0.50).alias("q2"),
            pl.col(col).quantile(0.75).alias("q3"),
        ]
    ).to_dicts()[0]
    q1, q2, q3 = qs["q1"], qs["q2"], qs["q3"]

    return df.with_columns(
        pl.when(pl.col(col).is_null())
        .then(pl.lit("missing"))
        .when(pl.col(col) <= q1)
        .then(pl.lit(labels[0]))
        .when(pl.col(col) <= q2)
        .then(pl.lit(labels[1]))
        .when(pl.col(col) <= q3)
        .then(pl.lit(labels[2]))
        .otherwise(pl.lit(labels[3]))
        .alias(out_col)
    )


def _require_files(analytics_paths: Mapping[str, Path], keys: Sequence[str]) -> None:
    missing = [str(analytics_paths[key]) for key in keys if key not in analytics_paths or not Path(analytics_paths[key]).exists()]
    if missing:
        raise FileNotFoundError("Missing analytics parquet(s): " + ", ".join(missing))


def load_part_profile_df(
    selected_ids: Sequence[str],
    analytics_paths: Mapping[str, str | Path],
    *,
    source_id_col: str = "oper_part_no",
) -> pl.DataFrame:
    resolved_paths = {key: Path(value) for key, value in analytics_paths.items()}
    _require_files(
        resolved_paths,
        [
            "intermittent",
            "lifecycle",
            "obsolescence",
            "outlier",
            "acf",
            "seasonality",
            "calendar",
            "trend",
            "level",
            "part_cluster",
            "selector",
        ],
    )

    selected_ids = [str(v) for v in selected_ids]
    id_expr = pl.col(source_id_col).cast(pl.String).alias(source_id_col)

    intermittent_df = (
        pl.read_parquet(resolved_paths["intermittent"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("demand_type"),
                pl.col("is_sparsity"),
                pl.col("ADI"),
                pl.col("CV2"),
                pl.col("zero_ratio").alias("zero_ratio_inter"),
                pl.col("nz_mean"),
            ]
        )
    )

    lifecycle_df = (
        pl.read_parquet(resolved_paths["lifecycle"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("stage"),
                pl.col("stage_code"),
                pl.col("age_periods"),
                pl.col("inactive_gap"),
                pl.col("recent_mean").alias("lifecycle_recent_mean"),
                pl.col("prev_mean").alias("lifecycle_prev_mean"),
            ]
        )
    )

    obsolescence_df = (
        pl.read_parquet(resolved_paths["obsolescence"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("obsolescence_score"),
                pl.col("time_since_last_demand"),
                pl.col("hazard_like_score"),
                pl.col("recent_zero_ratio"),
            ]
        )
    )

    spike_df = (
        pl.read_parquet(resolved_paths["outlier"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("outlier_count"),
                pl.col("spike_score"),
            ]
        )
    )

    acf_df = (
        pl.read_parquet(resolved_paths["acf"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("acf_decay_lag"),
                pl.col("sig_lag_count"),
                pl.col("max_acf"),
            ]
        )
    )

    seasonality_df = (
        pl.read_parquet(resolved_paths["seasonality"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .group_by(pl.col(source_id_col).cast(pl.String).alias(source_id_col))
        .agg(
            [
                pl.col("best_period").drop_nulls().first().alias("best_period"),
                pl.col("best_strength").drop_nulls().first().alias("best_strength"),
            ]
        )
    )

    calendar_df = (
        pl.read_parquet(resolved_paths["calendar"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("calendar_strength"),
                pl.col("calendar_peak_count"),
                pl.col("calendar_flag"),
            ]
        )
    )

    trend_df = (
        pl.read_parquet(resolved_paths["trend"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("trend_direction"),
                pl.col("trend_slope"),
                pl.col("trend_strength"),
                pl.col("monotonicity"),
            ]
        )
    )

    level_df = (
        pl.read_parquet(resolved_paths["level"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("mean").alias("level_mean"),
                pl.col("median").alias("level_median"),
                pl.col("std").alias("level_std"),
                pl.col("p95"),
                pl.col("cv").alias("level_cv"),
                pl.col("volatility_index"),
            ]
        )
    )

    cluster_df = (
        pl.read_parquet(resolved_paths["part_cluster"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("cluster_id"),
            ]
        )
    )

    selector_df = (
        pl.read_parquet(resolved_paths["selector"])
        .filter(pl.col(source_id_col).cast(pl.String).is_in(selected_ids))
        .select(
            [
                id_expr,
                pl.col("season_flag"),
                pl.col("calendar_flag").alias("selector_calendar_flag"),
                pl.col("spike_flag"),
                pl.col("regime_flag"),
            ]
        )
    )

    part_profile_df = intermittent_df
    for frame in [
        lifecycle_df,
        obsolescence_df,
        spike_df,
        acf_df,
        seasonality_df,
        calendar_df,
        trend_df,
        level_df,
        cluster_df,
        selector_df,
    ]:
        part_profile_df = part_profile_df.join(frame, on=source_id_col, how="left")

    return part_profile_df.sort(source_id_col)


def build_part_profile_buckets(part_profile_df: pl.DataFrame) -> pl.DataFrame:
    out = add_quantile_bucket(part_profile_df, "obsolescence_score", "obsolescence_bucket", ["Q1_low", "Q2", "Q3", "Q4_high"])
    out = add_quantile_bucket(out, "spike_score", "spike_bucket", ["Q1_low", "Q2", "Q3", "Q4_high"])
    out = add_quantile_bucket(out, "acf_decay_lag", "memory_bucket", ["short", "mid_short", "mid_long", "long"])
    out = add_quantile_bucket(out, "p95", "scale_bucket", ["small", "mid_small", "mid_large", "large"])
    return out


def select_target_cohort_parts(
    part_profile_df: pl.DataFrame,
    *,
    source_id_col: str = "oper_part_no",
    scale_buckets: Sequence[str] = ("small",),
    demand_types: Sequence[str] = ("erratic",),
    obsolescence_buckets: Sequence[str] = ("Q4_high",),
    require_tail_stage: bool = False,
) -> pl.DataFrame:
    out = (
        part_profile_df.select(
            [
                pl.col(source_id_col).cast(pl.String).alias(source_id_col),
                "stage",
                "demand_type",
                "obsolescence_bucket",
                "spike_bucket",
                "memory_bucket",
                "scale_bucket",
            ]
        )
        .unique(subset=[source_id_col])
    )

    mask = (
        pl.col("scale_bucket").is_in([str(v) for v in scale_buckets])
        & pl.col("demand_type").is_in([str(v) for v in demand_types])
        & pl.col("obsolescence_bucket").is_in([str(v) for v in obsolescence_buckets])
    )
    if require_tail_stage:
        mask = mask & pl.col("stage").is_in(["decline", "inactive"])

    return out.with_columns(mask.alias("target_segment")).filter(pl.col("target_segment"))


def oversample_train_df_with_synthetic_ids(
    train_df: pl.DataFrame,
    cohort_part_df: pl.DataFrame,
    *,
    id_col: str,
    cohort_id_col: str = "oper_part_no",
    extra_copies: int = 2,
    suffix_prefix: str = "__cohortw",
) -> pl.DataFrame:
    extra_copies = int(extra_copies)
    if extra_copies <= 0 or cohort_part_df.height == 0 or train_df.height == 0:
        return train_df

    target_ids = (
        cohort_part_df.select(pl.col(cohort_id_col).cast(pl.String))
        .unique()
        .get_column(cohort_id_col)
        .to_list()
    )
    if not target_ids:
        return train_df

    base_target_df = train_df.filter(pl.col(id_col).cast(pl.String).is_in(target_ids))
    if base_target_df.height == 0:
        return train_df

    copies = [train_df]
    for idx in range(extra_copies):
        copy_df = base_target_df.with_columns(
            (pl.col(id_col).cast(pl.String) + pl.lit(f"{suffix_prefix}{idx + 1:02d}")).alias(id_col)
        )
        copies.append(copy_df)
    return pl.concat(copies, how="vertical_relaxed")


def build_cohort_oversampling_summary(
    train_df_before: pl.DataFrame,
    train_df_after: pl.DataFrame,
    cohort_part_df: pl.DataFrame,
    *,
    id_col: str,
    cohort_id_col: str = "oper_part_no",
    extra_copies: int = 0,
) -> pl.DataFrame:
    original_train_parts = int(train_df_before.select(pl.col(id_col).cast(pl.String).n_unique()).item())
    oversampled_train_parts = int(train_df_after.select(pl.col(id_col).cast(pl.String).n_unique()).item())

    return pl.DataFrame(
        [
            {
                "original_train_rows": int(train_df_before.height),
                "oversampled_train_rows": int(train_df_after.height),
                "added_rows": int(train_df_after.height - train_df_before.height),
                "original_train_parts": original_train_parts,
                "oversampled_train_parts": oversampled_train_parts,
                "added_parts": int(oversampled_train_parts - original_train_parts),
                "hard_cohort_parts": int(cohort_part_df.select(pl.col(cohort_id_col).cast(pl.String).n_unique()).item()) if cohort_part_df.height > 0 else 0,
                "extra_copies_per_part": int(extra_copies),
            }
        ]
    )
