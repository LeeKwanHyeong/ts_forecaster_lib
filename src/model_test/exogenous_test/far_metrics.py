from __future__ import annotations

from typing import Final

import polars as pl


REVISION_WEIGHTS: Final[tuple[float, ...]] = (
    0.03,
    0.03,
    0.07,
    0.07,
    0.15,
    0.15,
    0.25,
    0.25,
)

_REQUIRED_COLUMNS: Final[set[str]] = {
    "model_name",
    "part_no",
    "plan_week",
    "forecast_week",
    "prediction",
    "actual",
}

_RESULT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "model_name": pl.String,
    "target_week": pl.Int64,
    "weighted_far": pl.Float64,
    "fcst_qty_total": pl.Float64,
    "row_count": pl.Int64,
    "nonzero_row_count": pl.Int64,
}


def _far_score_expr() -> pl.Expr:
    return (
        pl.when(
            (pl.col("prediction") == 0)
            | (pl.col("prediction") * 2 < pl.col("actual"))
        )
        .then(pl.lit(0.0))
        .otherwise(
            1.0
            - (pl.col("prediction") - pl.col("actual")).abs()
            / pl.col("prediction")
        )
        .alias("revision_far")
    )


def _validate_forecast_frame(forecasts: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(forecasts.columns))
    if missing:
        raise ValueError(f"FAR input is missing required columns: {missing}")
    duplicate = (
        forecasts.group_by(
            ["model_name", "part_no", "plan_week", "forecast_week"]
        )
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate.height:
        raise ValueError(
            "FAR input contains duplicate model/part/plan/forecast rows."
        )


def compute_revision_far(
    forecasts: pl.DataFrame,
    *,
    revisions: int = 8,
) -> pl.DataFrame:
    """Compute Samsung GCS weighted FAR from modern long-format forecasts.

    Revision weights preserve the established oldest-to-newest order:
    0.03, 0.03, 0.07, 0.07, 0.15, 0.15, 0.25, 0.25.
    A part contributes to the portfolio result only when all eight revisions
    are present and every revision forecast is non-zero.
    """

    if revisions != len(REVISION_WEIGHTS):
        raise ValueError(
            f"FAR requires exactly {len(REVISION_WEIGHTS)} revisions."
        )
    _validate_forecast_frame(forecasts)
    if forecasts.is_empty():
        return pl.DataFrame(schema=_RESULT_SCHEMA)

    normalized = forecasts.with_columns(
        pl.col("model_name").cast(pl.String),
        pl.col("part_no").cast(pl.String),
        pl.col("plan_week").cast(pl.Int64),
        pl.col("forecast_week").cast(pl.Int64),
        pl.col("prediction").cast(pl.Float64),
        pl.col("actual").cast(pl.Float64),
    ).filter(pl.col("actual").is_not_null())

    rows: list[dict[str, object]] = []
    for key, target_frame in normalized.partition_by(
        ["model_name", "forecast_week"],
        as_dict=True,
    ).items():
        model_name, target_week = key
        plan_weeks = sorted(
            int(value)
            for value in target_frame.get_column("plan_week").unique()
        )
        if len(plan_weeks) < revisions:
            continue
        selected_plan_weeks = plan_weeks[-revisions:]
        weights = pl.DataFrame(
            {
                "plan_week": selected_plan_weeks,
                "revision_weight": list(REVISION_WEIGHTS),
            }
        )
        selected = (
            target_frame.filter(
                pl.col("plan_week").is_in(selected_plan_weeks)
            )
            .join(weights, on="plan_week", how="inner")
            .with_columns(_far_score_expr())
        )
        inconsistent_actuals = (
            selected.group_by("part_no")
            .agg(pl.col("actual").n_unique().alias("actual_count"))
            .filter(pl.col("actual_count") > 1)
        )
        if inconsistent_actuals.height:
            raise ValueError(
                f"Actual demand changes across revisions for {model_name} "
                f"target_week={target_week}."
            )

        by_part = selected.group_by("part_no").agg(
            pl.len().alias("revision_count"),
            (pl.col("prediction") != 0).all().alias("all_nonzero"),
            (pl.col("revision_far") * pl.col("revision_weight"))
            .sum()
            .alias("accu"),
            (pl.col("prediction") * pl.col("revision_weight"))
            .sum()
            .alias("fcst_qty"),
        )
        complete = by_part.filter(pl.col("revision_count") == revisions)
        eligible = complete.filter(pl.col("all_nonzero"))
        fcst_qty_total = float(
            eligible.select(pl.col("fcst_qty").sum()).item() or 0.0
        )
        weighted_far = None
        if eligible.height and fcst_qty_total != 0.0:
            weighted_far = float(
                eligible.select(
                    (
                        pl.col("accu")
                        * pl.col("fcst_qty")
                        / pl.col("fcst_qty").sum()
                    ).sum()
                ).item()
            )
        rows.append(
            {
                "model_name": str(model_name),
                "target_week": int(target_week),
                "weighted_far": weighted_far,
                "fcst_qty_total": fcst_qty_total,
                "row_count": complete.height,
                "nonzero_row_count": eligible.height,
            }
        )

    if not rows:
        return pl.DataFrame(schema=_RESULT_SCHEMA)
    return pl.DataFrame(rows, schema=_RESULT_SCHEMA).sort(
        ["model_name", "target_week"]
    )


__all__ = ["REVISION_WEIGHTS", "compute_revision_far"]
