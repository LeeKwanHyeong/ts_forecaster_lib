"""Contract-first characterization for the anchored data path.

These tests freeze the temporal, selection, and loader behavior shared by the
public data API and the future high-level forecast API.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
import torch

from modeling_module import build_dataloader


def _inference_request(
    df: pl.DataFrame,
    *,
    freq: str,
    origin: date | int,
    lookback: int = 2,
    batch_size: int = 8,
    **overrides: object,
) -> dict[str, object]:
    request: dict[str, object] = {
        "df": df,
        "lookback": lookback,
        "horizon": 2,
        "freq": freq,
        "stage": "inference",
        "plan_dt": origin,
        "batch_size": batch_size,
        "shuffle": False,
    }
    request.update(overrides)
    return request


def test_weekly_polars_date_builds_the_expected_anchored_window() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "date": [date(2020, 12, 21), date(2020, 12, 28), date(2021, 1, 4)],
            "y": [52.0, 53.0, 1.0],
        },
        schema_overrides={"date": pl.Date},
    )

    batch = next(iter(build_dataloader(_inference_request(df, freq="weekly", origin=date(2021, 1, 4)))))

    assert batch[2] == ["A"]
    assert torch.equal(batch[0][0, :, 0], torch.tensor([52.0, 53.0]))


def test_iso_week_year_boundary_and_week_53_are_not_gregorian_year_keys() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["A", "A"],
            "date": [date(2020, 12, 28), date(2021, 1, 4)],
            "y": [53.0, 1.0],
        },
        schema_overrides={"date": pl.Date},
    )

    batch = next(
        iter(
            build_dataloader(
                _inference_request(df, freq="weekly", origin=date(2021, 1, 11), lookback=2)
            )
        )
    )

    assert torch.equal(batch[0][0, :, 0], torch.tensor([53.0, 1.0]))


def test_monthly_date_and_yyyymm_inputs_produce_equivalent_windows() -> None:
    date_df = pl.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "date": [date(2023, 11, 1), date(2023, 12, 1), date(2024, 1, 1)],
            "y": [11.0, 12.0, 1.0],
        },
        schema_overrides={"date": pl.Date},
    )
    int_df = date_df.with_columns(
        (pl.col("date").dt.year() * 100 + pl.col("date").dt.month()).alias("date")
    )

    date_batch = next(
        iter(build_dataloader(_inference_request(date_df, freq="monthly", origin=202401)))
    )
    int_batch = next(
        iter(build_dataloader(_inference_request(int_df, freq="monthly", origin=202401)))
    )

    assert torch.equal(date_batch[0], int_batch[0])
    assert torch.equal(date_batch[0][0, :, 0], torch.tensor([11.0, 12.0]))


def test_series_ids_subset_preserves_request_order() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B", "C", "C"],
            "date": [20240101, 20240102] * 3,
            "y": [1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
        }
    )

    batch = next(
        iter(
            build_dataloader(
                _inference_request(
                    df,
                    freq="daily",
                    origin=20240103,
                    series_ids=["C", "A", "C"],
                )
            )
        )
    )

    assert batch[2] == ["C", "A"]


@pytest.mark.parametrize("series_ids", [[], ["UNKNOWN"]])
def test_empty_or_unknown_series_selection_fails_fast(series_ids: list[str]) -> None:
    df = pl.DataFrame(
        {"unique_id": ["A", "A"], "date": [20240101, 20240102], "y": [1.0, 2.0]}
    )

    with pytest.raises(ValueError):
        build_dataloader(
            _inference_request(
                df,
                freq="daily",
                origin=20240103,
                series_ids=series_ids,
            )
        )


def test_inference_loader_options_are_forwarded() -> None:
    df = pl.DataFrame(
        {"unique_id": ["A", "A"], "date": [20240101, 20240102], "y": [1.0, 2.0]}
    )

    loader = build_dataloader(
        _inference_request(
            df,
            freq="daily",
            origin=20240103,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
    )

    assert loader.batch_size == 1
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.drop_last is True


def test_unknown_series_can_be_ignored_explicitly() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B"],
            "date": [20240101, 20240102] * 2,
            "y": [1.0, 2.0, 10.0, 20.0],
        }
    )

    batch = next(
        iter(
            build_dataloader(
                _inference_request(
                    df,
                    freq="daily",
                    origin=20240103,
                    series_ids=["UNKNOWN", "B"],
                    unknown_series_policy="ignore",
                )
            )
        )
    )

    assert batch[2] == ["B"]


def test_series_identity_order_is_independent_of_batch_size() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["A", "A", "B", "B", "C", "C"],
            "date": [20240101, 20240102] * 3,
            "y": [1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
        }
    )

    def collect_ids(batch_size: int) -> list[str]:
        loader = build_dataloader(
            _inference_request(
                df,
                freq="daily",
                origin=20240103,
                batch_size=batch_size,
                series_ids=["C", "A", "B"],
            )
        )
        return [series_id for batch in loader for series_id in batch[2]]

    assert collect_ids(1) == collect_ids(2) == collect_ids(8) == ["C", "A", "B"]


def test_capitalized_legacy_module_is_a_thin_identity_compatible_wrapper() -> None:
    from modeling_module.data_loader.MultiPartExoDataModule import (
        MultiPartExoDataModule as LegacyDataModule,
    )
    from modeling_module.data_loader.multi_part_exo_data_module import (
        MultiPartExoDataModule as AuthoritativeDataModule,
    )

    assert LegacyDataModule is AuthoritativeDataModule
