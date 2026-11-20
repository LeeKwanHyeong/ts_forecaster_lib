"""실제 데이터 실행기의 달력·분할·무수요·표본 선택 누수 방지 회귀 테스트."""

from dataclasses import asdict
import json

import numpy as np
import polars as pl
import pytest

from model_test.mora.real_data import (
    identity_calendar, materialize, plan_windows, restore_contract, step_week, week_step,
)
from model_test.mora.comparison_metrics import paired_comparison, point_metrics
from model_test.mora.run_chronos2_comparison import fold_preflight, make_query
from modeling_module.forecast_adapters.MORA import MORAConfig
from modeling_module.models.Chronos2.training.contracts import (
    Chronos2DataContract, Chronos2SplitContract, ForecastPartition, SeriesKey,
)


@pytest.fixture
def source(tmp_path):
    start = week_step(201801)
    steps = np.arange(start, start + 360)
    ids = [f"P{i:02d}" for i in range(20)]
    frame = pl.DataFrame({"oper_part_no": np.repeat(ids, len(steps)),
                          "demand_dt": np.tile([step_week(step) for step in steps], len(ids)),
                          "demand_qty": np.arange(len(ids) * len(steps), dtype=float)})
    path = tmp_path / "source.parquet"
    frame.write_parquet(path)
    keys = [SeriesKey((name,)) for name in ids]
    split = Chronos2SplitContract(
        ForecastPartition(tuple(keys[:12]), step_week(start), step_week(start + 305)),
        ForecastPartition(tuple(keys[12:16]), step_week(start + 306), step_week(start + 332)),
        ForecastPartition(tuple(keys[16:]), step_week(start + 333), step_week(start + 359)),
    )
    return path, frame, split


def test_iso_year_boundary():
    assert step_week(week_step(202053) + 1) == 202101
    assert week_step(202101) - week_step(202052) == 2
    with pytest.raises(ValueError):
        week_step(202153)


def test_manifest_restore_and_tamper(source, tmp_path):
    _, _, split = source
    contract = Chronos2DataContract("source", "a" * 64, provenance="user_provided", zero_semantics="observed_zero")
    path = tmp_path / "manifest.json"
    manifest = split.manifest(contract)
    path.write_text(json.dumps(manifest))
    original, restored_source, restored_split = restore_contract(path)
    assert original == json.loads(json.dumps(manifest))
    assert asdict(restored_split) == asdict(split)
    assert restored_source == contract
    manifest["split"]["seed"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        restore_contract(path)


def test_selection_uses_only_identity_dates_and_keeps_cutoffs(source, tmp_path):
    path, frame, split = source
    index = identity_calendar(path)
    assert "demand_qty" not in index.columns
    rows, excluded, audit = plan_windows(index, split)
    assert not excluded
    assert audit["train"]["selected_windows"] == 48
    assert audit["test"]["selected_windows"] == 4
    assert audit["validation"]["selected_windows"] == 4
    for row in rows:
        partition = getattr(split, row["split"])
        assert partition.target_start <= row["origin"] <= row["target_end"] <= partition.target_end
        assert week_step(row["target_end"]) - row["origin_step"] == 26
        assert row["origin_step"] - week_step(row["context_start"]) == 52
    changed = tmp_path / "changed.parquet"
    frame.with_columns((pl.col("demand_qty") * -100 + 777).alias("demand_qty")).write_parquet(changed)
    assert plan_windows(identity_calendar(changed), split) == (rows, excluded, audit)


def test_missing_test_week_excludes_part_without_zero_fill(source):
    path, frame, split = source
    missing = split.test.target_start
    frame.filter(~((pl.col("oper_part_no") == "P19") & (pl.col("demand_dt") == missing))).write_parquet(path)
    rows, excluded, audit = plan_windows(identity_calendar(path), split)
    assert excluded == [{"split": "test", "oper_part_no": "P19", "reason": "no_complete_L52_H27_in_allowed_dates"}]
    assert audit["test"]["eligible_parts"] == 3
    assert not any(row["oper_part_no"] == "P19" for row in rows)


def test_duplicate_identity_rejected(source):
    path, frame, _ = source
    pl.concat([frame, frame.head(1)]).write_parquet(path)
    with pytest.raises(ValueError, match="중복"):
        identity_calendar(path)


def test_materialize_preserves_observed_zero_and_array_order(source):
    path, frame, split = source
    frame.with_columns(pl.lit(0.0).alias("demand_qty")).write_parquet(path)
    rows, _, _ = plan_windows(identity_calendar(path), split)
    test = [row for row in rows if row["split"] == "test"]
    history, target = materialize(path, test)
    assert history.shape == (4, 52, 1)
    assert target.shape == (4, 27, 1)
    assert not history.any() and not target.any()
    query = make_query(test, history, target)
    assert query.series_ids == tuple(row["oper_part_no"] for row in test)
    assert np.array_equal(query.origins, [row["origin_step"] for row in test])


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), None])
def test_invalid_quantity_not_silently_imputed(source, bad):
    path, frame, split = source
    rows, _, _ = plan_windows(identity_calendar(path), split)
    test = [row for row in rows if row["split"] == "test"]
    frame.with_columns(pl.when((pl.col("oper_part_no") == test[0]["oper_part_no"]) &
                              (pl.col("demand_dt") == test[0]["origin"]))
                       .then(pl.lit(bad, dtype=pl.Float64)).otherwise(pl.col("demand_qty")).alias("demand_qty")).write_parquet(path)
    with pytest.raises(ValueError, match="결측"):
        materialize(path, test)


def test_purged_nested_preflight_without_quantity(source):
    path, _, split = source
    rows, _, _ = plan_windows(identity_calendar(path), split)
    result = fold_preflight([row for row in rows if row["split"] == "train"], MORAConfig(scope="pooled_series", min_fold_train=1))
    assert len(result) == 4
    assert all(len(block["inner"]) == 4 and block["fit"] > 0 for block in result)


def test_zero_demand_denominators_explicit():
    y = np.zeros((3, 27, 1))
    metrics = point_metrics(y, y + 2)
    assert metrics["wape"] is None
    assert metrics["macro_part_wape_positive_only"] is None
    assert metrics["zero_total_parts"] == 3
    assert metrics["zero_total_parts_mae"] == 2
    result = paired_comparison(y, y + 2, y + 1, bootstrap_draws=50)
    assert result["part_mae"] == {"improved": 3, "unchanged": 0, "worsened": 0, "tolerance": 1e-10}
    assert result["paired_part_bootstrap"]["percentile_95_ci"] == [-1, -1]


def test_metrics_ties_and_negative_values_visible():
    y = np.ones((2, 27, 1))
    p = -y
    result = paired_comparison(y, p, p, bootstrap_draws=20)
    assert result["part_mae"]["unchanged"] == 2
    assert result["baseline"]["negative_predictions"] == 54
    assert result["baseline"]["wape"] == 2
    assert result["mae_reduction_ratio"] == 0


@pytest.mark.parametrize("bad", [np.zeros((1, 2)), np.array([[[float("nan")]]])])
def test_metrics_reject_invalid_arrays(bad):
    with pytest.raises(ValueError):
        point_metrics(bad, bad)
