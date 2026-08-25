from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from modeling_module.icl import ICLSplit
from tools.qualify_icl_backbones_5090 import (
    APPROVED_EXOGENOUS_FEATURES,
    _accuracy,
    _load_operation_part_source,
    _minimum_contiguous_rows,
    _sha256_payload,
    _split_target_contract,
    prepare_bundles,
)


def _week(start: date, offset: int) -> int:
    iso = (start + timedelta(weeks=offset)).isocalendar()
    return int(iso.year) * 100 + int(iso.week)


def _operation_parts(count: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site_cd": ["V100"] * count,
            "oper_part_no": [f"part-{index}" for index in range(count)],
            "demand_start_dt": [201801] * count,
            "demand_end_dt": [202652] * count,
            "warranty": [12 + 12 * (index % 4) for index in range(count)],
        },
        schema_overrides={"warranty": pl.Int16},
    )


def test_qualification_minimum_history_matches_non_overlapping_split_contract():
    assert _minimum_contiguous_rows(horizon=26, stride=26) == 286
    assert _minimum_contiguous_rows(horizon=27, stride=26) == 391
    assert (
        _minimum_contiguous_rows(
            horizon=27,
            stride=26,
            validation_episodes=0,
        )
        == 339
    )


def test_qualification_prepares_sealed_h26_and_h27_exogenous_artifacts(
    tmp_path: Path,
):
    start = date.fromisocalendar(2019, 1, 1)
    rows = []
    for part_index in range(3):
        for offset in range(420):
            rows.append(
                {
                    "oper_part_no": f"part-{part_index}",
                    "demand_dt": _week(start, offset),
                    "demand_qty": float(10 + part_index + offset % 9),
                }
            )
    source = tmp_path / "target.parquet"
    pl.DataFrame(rows).write_parquet(source)

    bundles = prepare_bundles(
        target_path=source,
        source_revision="a" * 64,
        output_root=tmp_path / "qualification",
        horizons=(26, 27),
        sample_series=2,
        stride=26,
        operation_parts=_operation_parts(3),
        exogenous_source_revision="approved-operation-part-r1",
    )

    assert set(bundles) == {26, 27}
    for horizon, bundle in bundles.items():
        assert bundle.manifest.dataset_kind == "exogenous"
        assert bundle.manifest.exogenous_schema is not None
        assert bundle.manifest.exogenous_schema.past_feature_names == (
            APPROVED_EXOGENOUS_FEATURES
        )
        assert bundle.manifest.exogenous_schema.future_feature_names == (
            APPROVED_EXOGENOUS_FEATURES
        )
        assert bundle.manifest.exogenous_schema.source_revision == (
            "approved-operation-part-r1"
        )
        assert bundle.manifest.split_counts["test"] == 2
        assert bundle.manifest.split_counts["validation"] == (
            2 if horizon == 26 else 0
        )
        assert len(bundle.for_split(ICLSplit.TEST)[0].query_target.weeks) == horizon
        ranges = _split_target_contract(bundle)
        if horizon == 26:
            assert ranges["train"]["target_end_week"] < ranges["validation"][
                "target_start_week"
            ]
            assert ranges["validation"]["target_end_week"] < ranges["test"][
                "target_start_week"
            ]
        else:
            assert "validation" not in ranges
            assert ranges["train"]["target_end_week"] < ranges["test"][
                "target_start_week"
            ]
        assert (tmp_path / "qualification" / f"h{horizon}" / "episodes" / "manifest.json").is_file()


def test_qualification_accuracy_uses_sealed_query_targets(tmp_path: Path):
    start = date.fromisocalendar(2019, 1, 1)
    rows = [
        {
            "oper_part_no": "part-1",
            "demand_dt": _week(start, offset),
            "demand_qty": float(10 + offset % 5),
        }
        for offset in range(420)
    ]
    source = tmp_path / "target.parquet"
    pl.DataFrame(rows).write_parquet(source)
    bundle = prepare_bundles(
        target_path=source,
        source_revision="b" * 64,
        output_root=tmp_path / "qualification",
        horizons=(26,),
        sample_series=1,
        stride=26,
        operation_parts=_operation_parts(2),
        exogenous_source_revision="approved-operation-part-r2",
    )[26]
    episode = bundle.for_split(ICLSplit.TEST)[0]
    predictions = pl.DataFrame(
        {
            "episode_id": [episode.episode_id] * 26,
            "horizon_step": list(range(26)),
            "point": [float(row[0]) for row in episode.query_target.target],
        }
    )

    metrics = _accuracy(predictions, bundle)

    assert metrics == {"points": 26, "mae": 0.0, "wape": 0.0}


def test_qualification_verifies_operation_part_source_revision(tmp_path: Path):
    frame = _operation_parts(2)
    source = tmp_path / "tb_mst_oper_part.parquet"
    frame.write_parquet(source)
    artifact = {
        "logical_name": "tb_mst_oper_part",
        "columns": [
            "site_cd",
            "oper_part_no",
            "demand_start_dt",
            "demand_end_dt",
            "warranty",
        ],
        "row_count": 2,
        "part_count": 2,
        "content_sha256": _sha256_payload(frame.sort("site_cd", "oper_part_no").to_dicts()),
    }
    manifest = {
        "contract_id": "demand-engine-operation-part-snapshot-v1",
        "contract_version": "1.0.0",
        "source_id": "dsdm.tb_mst_oper_part",
        "source_revision": "approved-operation-part-r3",
        "feature_schema_version": "1.0.0",
        "scope": {
            "company_cd": "DSE",
            "subs_cd": "C100",
            "plant_cd": "V100",
            "site_cd": "V100",
        },
        "artifact": artifact,
    }
    manifest["manifest_sha256"] = _sha256_payload(manifest)
    manifest_path = tmp_path / "operation_part_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    observed, contract = _load_operation_part_source(
        manifest_path,
        source,
        expected_site_cd="V100",
    )

    assert observed.height == 2
    assert contract["source_revision"] == "approved-operation-part-r3"
    assert contract["source_manifest_sha256"] == manifest["manifest_sha256"]
    assert contract["snapshot_content_sha256"] == artifact["content_sha256"]
