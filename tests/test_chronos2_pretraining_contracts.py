"""부품·정답 기간 분리, 원천 의미, ID 전용 배정의 회귀 테스트."""

from dataclasses import replace
import json

import polars as pl
import pytest

from modeling_module.data_loader.temporal import add_period
from modeling_module.models.Chronos2.training.contracts import (
    Chronos2DataContract, Chronos2SplitContract, ForecastPartition, ForecastWindow,
    SeriesKey, make_split_contract, manifest_sha256,
)
from model_test.chronos2.prepare_pretraining_contract import build_parser, run


def source_contract(**changes):
    return Chronos2DataContract(source_id="synthetic", source_sha256="a" * 64,
                               provenance="synthetic", zero_semantics="observed_zero", **changes)


def split_contract():
    return Chronos2SplitContract(
        ForecastPartition((SeriesKey(("A",)),), 202301, 202452),
        ForecastPartition((SeriesKey(("B",)),), 202501, 202527),
        ForecastPartition((SeriesKey(("C",)),), 202528, 202602))


def test_zero_meaning_and_provenance_require_explicit_confirmation():
    unknown = Chronos2DataContract("snapshot", "a" * 64)
    with pytest.raises(ValueError, match="확인"):
        unknown.assert_semantics_confirmed()
    source_contract().assert_semantics_confirmed()
    with pytest.raises(ValueError, match="마스크"):
        replace(unknown, zero_semantics="explicit_mask")
    replace(unknown, zero_semantics="explicit_mask", observed_mask_column="observed", provenance="user_provided").assert_semantics_confirmed()


@pytest.mark.parametrize("changes", [
    {"source_id": ""}, {"source_sha256": "x"}, {"series_columns": ()},
    {"time_column": "oper_part_no"}, {"target_column": "demand_dt"},
    {"provenance": "production_verified_automatically"}, {"zero_semantics": "fill_all_missing_with_zero"},
])
def test_invalid_data_contract_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(source_contract(), **changes)


@pytest.mark.parametrize("key", [(), ("",), (1,), "A"])
def test_invalid_series_keys_rejected(key):
    with pytest.raises(ValueError):
        SeriesKey(key)


def test_window_calendar_and_target_cutoff_are_strict():
    window = ForecastWindow(SeriesKey(("B",)), 202501)
    split = split_contract()
    split.validate_window("validation", window)
    assert window.context_weeks(52)[-1] == 202452
    assert window.target_weeks(27) == tuple(add_period(202501, i, "weekly") for i in range(27))
    assert ForecastWindow(window.series, 202101).context_weeks(1) == (202053,)
    with pytest.raises(ValueError, match="cutoff"):
        split.validate_window("validation", ForecastWindow(window.series, 202502))
    with pytest.raises(ValueError, match="분할"):
        split.validate_window("train", window)
    with pytest.raises(ValueError, match="중복"):
        split.validate_windows([("validation", window), ("validation", window)])


def test_parts_and_target_periods_cannot_overlap():
    split = split_contract()
    with pytest.raises(ValueError, match="부품 ID"):
        replace(split, test=replace(split.test, series=split.train.series))
    with pytest.raises(ValueError, match="기간"):
        replace(split, validation=replace(split.validation, target_start=202452))
    with pytest.raises(ValueError, match="horizon"):
        replace(split, test=replace(split.test, target_end=202552))
    with pytest.raises(ValueError):
        ForecastPartition((SeriesKey(("X",)),), 202553, 202601)


def test_composite_keys_are_explicit_and_manifest_hash_is_reproducible():
    split = split_contract()
    source = source_contract()
    first = split.manifest(source)
    second = json.loads(json.dumps(first))
    digest = second.pop("manifest_sha256")
    assert digest == manifest_sha256(second) and first == split.manifest(source)
    assert first["normalization_scope"] == first["pattern_classification_scope"] == "context_only"
    assert not first["quantity_and_window_quality_validated"] and not first["training_executed"]
    with pytest.raises(ValueError, match="식별값"):
        split.manifest(replace(source, series_columns=("company_cd", "oper_part_no")))
    assert SeriesKey(("COMPANY_A", "PART_001")) != SeriesKey(("COMPANY_B", "PART_001"))


def test_hash_split_is_70_15_15_and_keeps_prior_evaluation_out_of_test():
    keys = [SeriesKey((f"PART_{i:03d}",)) for i in range(100)]
    excluded = keys[:40]
    split = make_split_contract(source_contract(), keys, 201801, 202544, excluded_from_test=excluded)
    assert [len(getattr(split, name).series) for name in ("train", "validation", "test")] == [70, 15, 15]
    assert set(split.test.series).isdisjoint(excluded)
    assert (split.train.target_end, split.validation.target_start, split.validation.target_end,
            split.test.target_start, split.test.target_end) == (202442, 202443, 202517, 202518, 202544)
    assert make_split_contract(source_contract(), reversed(keys), 201801, 202544, excluded_from_test=excluded) == split
    assert make_split_contract(source_contract(), keys, 201801, 202544, excluded_from_test=excluded, seed=43) != split
    with pytest.raises(ValueError, match="후보"):
        make_split_contract(source_contract(), keys, 201801, 202544, excluded_from_test=keys)
    with pytest.raises(ValueError, match="SeriesKey"):
        make_split_contract(source_contract(), keys, 201801, 202544, excluded_from_test=["PART_001"])


def write_fixture(path, quantity=10.0):
    weeks = [add_period(202301, i, "weekly") for i in range(100)]
    pl.DataFrame([{"oper_part_no": f"PART_{i:03d}", "demand_dt": week, "demand_qty": quantity}
                  for i in range(20) for week in weeks]).write_parquet(path)


def test_prepare_manifest_reads_identity_not_future_demand_values(tmp_path):
    source = tmp_path / "source.parquet"
    previous = tmp_path / "cohort.csv"
    pl.DataFrame({"oper_part_no": ["PART_000", "PART_001"]}).write_csv(previous)
    write_fixture(source)
    def args(output):
        return build_parser().parse_args(["--source-parquet", str(source), "--output-dir", str(output),
            "--exclude-evaluated-cohort", str(previous), "--zero-semantics", "observed_zero", "--provenance", "synthetic"])
    first_args = args(tmp_path / "first")
    first = run(first_args)
    assert first["counts"] == {"train": 14, "validation": 3, "test": 3}
    assert first["test_overlap_with_prior_evaluation"] == 0 and not first["ready_for_training"]
    first_manifest = json.loads((first_args.output_dir / "split_manifest.json").read_text())
    write_fixture(source, quantity=99999.0)
    second_args = args(tmp_path / "second")
    second = run(second_args)
    second_manifest = json.loads((second_args.output_dir / "split_manifest.json").read_text())
    assert first_manifest["split"] == second_manifest["split"]
    assert first["source_sha256"] != second["source_sha256"]
    assert first["zero_semantics"] == "observed_zero"
    with pytest.raises(FileExistsError):
        run(first_args)


def test_prepare_manifest_rejects_duplicate_grain(tmp_path):
    source = tmp_path / "source.parquet"
    write_fixture(source)
    frame = pl.read_parquet(source)
    pl.concat([frame, frame.head(1)]).write_parquet(source)
    args = build_parser().parse_args(["--source-parquet", str(source), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="중복"):
        run(args)


def test_prepare_manifest_rejects_empty_source(tmp_path):
    source = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"oper_part_no": pl.String, "demand_dt": pl.Int64,
                         "demand_qty": pl.Float64}).write_parquet(source)
    args = build_parser().parse_args(["--source-parquet", str(source), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="행이 없습니다"):
        run(args)
    assert not args.output_dir.exists()
