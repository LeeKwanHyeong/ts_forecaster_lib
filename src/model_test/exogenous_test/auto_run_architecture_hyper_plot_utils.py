from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _sanitize_filename(value: str) -> str:
    """Keep generated plot filenames stable across platforms."""
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _with_base_ab_case(
    summary_df: pl.DataFrame,
    base_ab_case_names: Sequence[str],
) -> pl.DataFrame:
    """
    Attach the coarse AB case name used in BASE_AB_CASES.

    Future-exogenous cases may be expanded into implementation variants such as
    `past_o_future_o_token_cross_attn`; this helper maps those variants back to
    `past_o_future_o` so final comparison plots remain stable as cases evolve.
    """
    if summary_df.height == 0:
        return summary_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("base_ab_case"))
    if "ab_case_name" not in summary_df.columns:
        raise ValueError("'ab_case_name' column is required for AB case plots.")
    if "far_mean" not in summary_df.columns:
        raise ValueError("'far_mean' column is required for AB case plots.")

    base_expr = pl.lit(None).cast(pl.Utf8)
    for base_case in reversed([str(v) for v in base_ab_case_names]):
        matched = (
            (pl.col("ab_case_name") == base_case)
            | pl.col("ab_case_name").str.starts_with(f"{base_case}_")
        )
        base_expr = pl.when(matched).then(pl.lit(base_case)).otherwise(base_expr)

    return (
        summary_df.with_columns(base_expr.alias("base_ab_case"))
        .filter(pl.col("base_ab_case").is_not_null())
    )


def _mean_far(summary_df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    """Aggregate run-level FAR summaries over seeds, batches, and architectures."""
    return (
        summary_df.group_by(list(group_cols))
        .agg(
            [
                pl.col("far_mean").mean().alias("far_mean_avg"),
                pl.col("far_mean").std().alias("far_mean_std"),
                pl.len().alias("run_count"),
            ]
        )
        .sort(list(group_cols))
    )


def _ordered_values(values: Sequence[str], preferred_order: Sequence[str] | None = None) -> List[str]:
    """Return values in preferred order first, then append any unseen values."""
    unique_values = [str(v) for v in values]
    if preferred_order is None:
        return sorted(set(unique_values))

    ordered: List[str] = []
    for value in preferred_order:
        value = str(value)
        if value in unique_values and value not in ordered:
            ordered.append(value)
    for value in sorted(set(unique_values)):
        if value not in ordered:
            ordered.append(value)
    return ordered


def _plot_single_bar(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    x_order: Sequence[str] | None = None,
) -> None:
    """Draw a simple one-series bar chart."""
    if df.height == 0:
        return

    pdf = df.select([x_col, y_col]).to_pandas()
    x_values = _ordered_values(pdf[x_col].astype(str).tolist(), x_order)
    y_lookup = {
        str(row[x_col]): float(row[y_col]) if row[y_col] is not None else np.nan
        for _, row in pdf.iterrows()
    }
    y_values = [y_lookup.get(value, np.nan) for value in x_values]

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(x_values)), 5))
    ax.bar(np.arange(len(x_values)), y_values, color="#87cfc4", edgecolor="white", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels(x_values, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bars(
    df: pl.DataFrame,
    *,
    x_col: str,
    hue_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    x_order: Sequence[str] | None = None,
    hue_order: Sequence[str] | None = None,
) -> None:
    """Draw grouped bars for plant/case comparison summaries."""
    if df.height == 0:
        return

    pdf = df.select([x_col, hue_col, y_col]).to_pandas()
    x_values = _ordered_values(pdf[x_col].astype(str).tolist(), x_order)
    hue_values = _ordered_values(pdf[hue_col].astype(str).tolist(), hue_order)
    x = np.arange(len(x_values), dtype=np.float64)
    width = min(0.8 / max(1, len(hue_values)), 0.28)

    fig, ax = plt.subplots(figsize=(max(10, 2.4 * len(x_values)), 6))
    for hue_idx, hue_value in enumerate(hue_values):
        offsets = x + (hue_idx - (len(hue_values) - 1) / 2.0) * width
        y_values = []
        for x_value in x_values:
            matched = pdf[
                (pdf[x_col].astype(str) == x_value)
                & (pdf[hue_col].astype(str) == hue_value)
            ]
            if matched.empty:
                y_values.append(np.nan)
            else:
                y_values.append(float(matched[y_col].iloc[0]))
        ax.bar(offsets, y_values, width=width, label=hue_value, edgecolor="white", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_far_mean_base_ab_case_comparisons(
    summary_df: pl.DataFrame,
    save_dir: Path,
    *,
    base_ab_case_names: Sequence[str],
    plant_order: Sequence[str] | None = None,
) -> Dict[str, Path]:
    """
    Save final FAR mean comparison plots for the active BASE_AB_CASES.

    Outputs:
    - one plant comparison plot per base AB case
    - one variant-level plant comparison plot when an AB case expands into
      multiple implementation variants, e.g. future fusion modes
    - two overall plots that compare base AB cases across all active plants
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    prepared = _with_base_ab_case(summary_df, base_ab_case_names)
    if prepared.height == 0:
        return {}

    base_order = _ordered_values(
        prepared.get_column("base_ab_case").to_list(),
        base_ab_case_names,
    )
    plant_order = _ordered_values(
        prepared.get_column("plant").to_list(),
        plant_order,
    )

    by_base_plant = _mean_far(prepared, ["base_ab_case", "plant"])
    by_variant_plant = _mean_far(prepared, ["base_ab_case", "ab_case_name", "plant"])
    by_base = _mean_far(prepared, ["base_ab_case"])

    by_base_plant.write_csv(save_dir / "far_mean_by_base_ab_case_plant.csv")
    by_variant_plant.write_csv(save_dir / "far_mean_by_ab_case_variant_plant.csv")
    by_base.write_csv(save_dir / "far_mean_by_base_ab_case.csv")

    paths: Dict[str, Path] = {}

    for base_case in base_order:
        base_case_df = by_base_plant.filter(pl.col("base_ab_case") == base_case)
        base_path = save_dir / f"{_sanitize_filename(base_case)}__far_mean_by_plant.png"
        _plot_single_bar(
            base_case_df,
            x_col="plant",
            y_col="far_mean_avg",
            title=f"{base_case} | FAR Mean by Plant",
            xlabel="plant",
            ylabel="far_mean_avg",
            save_path=base_path,
            x_order=plant_order,
        )
        paths[f"{base_case}_by_plant"] = base_path

        variant_df = by_variant_plant.filter(pl.col("base_ab_case") == base_case)
        variant_count = variant_df.select(pl.col("ab_case_name").n_unique()).item()
        if int(variant_count) > 1:
            variant_path = save_dir / f"{_sanitize_filename(base_case)}__far_mean_by_plant_variant.png"
            _plot_grouped_bars(
                variant_df,
                x_col="plant",
                hue_col="ab_case_name",
                y_col="far_mean_avg",
                title=f"{base_case} | FAR Mean by Plant and Variant",
                xlabel="plant",
                ylabel="far_mean_avg",
                save_path=variant_path,
                x_order=plant_order,
            )
            paths[f"{base_case}_by_plant_variant"] = variant_path

    overall_by_plant_path = save_dir / "overall__far_mean_by_plant_and_base_ab_case.png"
    _plot_grouped_bars(
        by_base_plant,
        x_col="plant",
        hue_col="base_ab_case",
        y_col="far_mean_avg",
        title="Overall FAR Mean by Plant and Base AB Case",
        xlabel="plant",
        ylabel="far_mean_avg",
        save_path=overall_by_plant_path,
        x_order=plant_order,
        hue_order=base_order,
    )
    paths["overall_by_plant_and_base_ab_case"] = overall_by_plant_path

    overall_by_case_path = save_dir / "overall__far_mean_by_base_ab_case_and_plant.png"
    _plot_grouped_bars(
        by_base_plant,
        x_col="base_ab_case",
        hue_col="plant",
        y_col="far_mean_avg",
        title="Overall FAR Mean by Base AB Case and Plant",
        xlabel="base_ab_case",
        ylabel="far_mean_avg",
        save_path=overall_by_case_path,
        x_order=base_order,
        hue_order=plant_order,
    )
    paths["overall_by_base_ab_case_and_plant"] = overall_by_case_path

    base_only_path = save_dir / "overall__far_mean_by_base_ab_case.png"
    _plot_single_bar(
        by_base,
        x_col="base_ab_case",
        y_col="far_mean_avg",
        title="Overall FAR Mean by Base AB Case",
        xlabel="base_ab_case",
        ylabel="far_mean_avg",
        save_path=base_only_path,
        x_order=base_order,
    )
    paths["overall_by_base_ab_case"] = base_only_path

    return paths


def sample_parts_for_plot(
    target_df: pl.DataFrame,
    id_col: str = "part_no",
    y_col: str = "demand_qty",
    n_samples: int = 50,
    seed: int = 42,
    min_nonzero_count: int = 1,
) -> List[str]:
    """
    Pick a stable random subset of parts for qualitative forecast plots.

    We bias the candidate pool toward parts that have at least a minimum amount
    of non-zero demand so that the sample plots are informative.
    """
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
    """Persist the sampled part list so future reruns can inspect the same set."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({id_col: sampled_parts}).write_parquet(str(save_path))


def _ordered_week_axis(
    weeks: Sequence[int | str],
    *,
    max_ticks: int = 14,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    Build a categorical week axis that preserves input order.

    This avoids matplotlib treating YYYYWW as a continuous numeric axis, which
    causes ugly gaps or broken labels when the series crosses year boundaries
    such as 202552 -> 202601.
    """
    labels = [str(int(w)) if not isinstance(w, str) else w for w in weeks]
    x = np.arange(len(labels), dtype=np.int64)
    if len(labels) <= max_ticks:
        tick_idx = list(range(len(labels)))
    else:
        step = max(1, math.ceil(len(labels) / max_ticks))
        tick_idx = list(range(0, len(labels), step))
        if tick_idx[-1] != len(labels) - 1:
            tick_idx.append(len(labels) - 1)
    tick_labels = [labels[i] for i in tick_idx]
    return x, labels, tick_idx, tick_labels


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
    """
    Save sampled part plots using categorical week labels.

    The categorical axis keeps forecast weeks readable even when the plan range
    spans multiple years.
    """
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

        x, _, tick_idx, tick_labels = _ordered_week_axis(part_df[date_col].to_list(), max_ticks=12)
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
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for empty_idx in range(n_parts, len(axes_flat)):
        fig.delaxes(axes_flat[empty_idx])

    fig.suptitle(f"Plan Week {plan_week} | Sampled Parts Actual vs Forecast", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = save_dir / f"plan_{plan_week}_sample_parts_3col.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_far_trend(far_by_week: pl.DataFrame, save_path: Path) -> None:
    """Plot FAR over target weeks on a categorical week axis."""
    if far_by_week.height == 0:
        return

    pdf = far_by_week.sort("target_week").select(["target_week", "weighted_far"]).to_pandas()
    x, _, tick_idx, tick_labels = _ordered_week_axis(pdf["target_week"].to_list(), max_ticks=14)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, pdf["weighted_far"].to_numpy(), marker="o")
    plt.title("FAR Trend by Target Week")
    plt.xlabel("target_week")
    plt.ylabel("weighted_far")
    plt.xticks(tick_idx, tick_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inflection_summary(inflection_df: pl.DataFrame, save_path: Path) -> None:
    """Plot aggregate actual/forecast sums by target week on a categorical axis."""
    if inflection_df.height == 0:
        return

    pdf = inflection_df.sort("target_week").to_pandas()
    x, _, tick_idx, tick_labels = _ordered_week_axis(pdf["target_week"].to_list(), max_ticks=14)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, pdf["actual_sum"], marker="o", label="actual_sum")
    plt.plot(
        x,
        pdf["base_forecast_sum"],
        marker="o",
        label="base_forecast_sum",
    )
    if "quantile_forecast_sum" in pdf.columns and not np.allclose(
        pdf["quantile_forecast_sum"].to_numpy(),
        0.0,
    ):
        plt.plot(
            x,
            pdf["quantile_forecast_sum"],
            marker="o",
            label="quantile_forecast_sum",
        )
    plt.title("Inflection Point Check (Aggregate)")
    plt.xlabel("target_week")
    plt.ylabel("sum")
    plt.xticks(tick_idx, tick_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_far_by_batch_case(grouped: pl.DataFrame, save_dir: Path) -> None:
    """
    Draw the run-summary comparison plot for each plant.

    The grouped frame is expected to already contain `case_label`,
    `batch_size`, and `far_mean_avg`.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for plant in grouped["plant"].unique().to_list():
        pdf = grouped.filter(pl.col("plant") == plant).to_pandas()
        if pdf.empty:
            continue

        fig = plt.figure(figsize=(16, 7))
        for case_label, sub in pdf.groupby("case_label"):
            plt.plot(sub["batch_size"], sub["far_mean_avg"], marker="o", label=case_label)

        plt.title(f"{plant} | FAR Mean by Batch Size (seed average)")
        plt.xlabel("batch_size")
        plt.ylabel("far_mean_avg")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(save_dir / f"{plant}_far_by_batch_case.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
