from __future__ import annotations

import json
from pathlib import Path

from tools.aggregate_icl_backbone_multiseed import aggregate_receipts
from tools.qualify_icl_backbones_5090 import _sha256_payload


def _receipt(
    path: Path,
    *,
    model_key: str,
    seed: int,
    mae: float,
    wape: float,
    bias: float = 0.02,
    negative_rate: float = 0.03,
) -> Path:
    receipt = {
        "qualification": {
            "status": "PASS",
            "seed": seed,
            "horizons": [26],
            "sample_series": 256,
            "batch_size": 4,
            "epochs": 5,
            "learning_rate": 1e-3 if model_key == "autotimes_base" else 1e-4,
            "exogenous_source_revision": "exo-r1",
        },
        "input": {"source_revision": "target-r1"},
        "backbone": {
            "model_id": "Qwen/Qwen2-0.5B",
            "revision": "91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
            "contract_sha256": "b" * 64,
        },
        "episodes": {"26": {"manifest_hash": "e" * 64}},
        "results": [
            {
                "model_key": model_key,
                "accuracy": {
                    "mae": mae,
                    "wape": wape,
                    "smape": wape * 0.8,
                    "bias": bias,
                    "raw_negative_rate": negative_rate,
                },
                "training": {
                    "seconds": 10.0 + seed,
                    "peak_allocated_mib": 2000.0,
                },
                "checkpoint": {
                    "sha256": f"{seed:064x}",
                    "reload_max_abs_delta": 0.0,
                },
            }
        ],
    }
    receipt["receipt_sha256"] = _sha256_payload(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_multiseed_aggregate_selects_stable_lower_wape_model(tmp_path: Path):
    paths = []
    for seed, mae in zip((11, 22, 33), (3.9, 4.0, 4.1)):
        paths.append(
            _receipt(
                tmp_path / f"autotimes-{seed}.json",
                model_key="autotimes_base",
                seed=seed,
                mae=mae,
                wape=0.56,
            )
        )
        paths.append(
            _receipt(
                tmp_path / f"sellm-{seed}.json",
                model_key="sellm_base",
                seed=seed,
                mae=mae + 0.1,
                wape=0.60,
            )
        )

    aggregate = aggregate_receipts(
        paths,
        expected_seeds=(11, 22, 33),
        max_mae_cv=0.10,
        max_abs_bias=0.10,
        max_raw_negative_rate=0.10,
    )

    assert aggregate["status"] == "PASS"
    assert aggregate["operational_candidates"] == ["autotimes_base", "sellm_base"]
    assert aggregate["recommended_default"] == "autotimes_base"
    assert aggregate["models"]["autotimes_base"]["status"] == "PASS"


def test_multiseed_aggregate_rejects_excessive_negative_output(tmp_path: Path):
    paths = []
    for seed in (11, 22, 33):
        paths.append(
            _receipt(
                tmp_path / f"autotimes-{seed}.json",
                model_key="autotimes_base",
                seed=seed,
                mae=4.0,
                wape=0.56,
            )
        )
        paths.append(
            _receipt(
                tmp_path / f"sellm-{seed}.json",
                model_key="sellm_base",
                seed=seed,
                mae=4.1,
                wape=0.60,
                negative_rate=0.25,
            )
        )

    aggregate = aggregate_receipts(
        paths,
        expected_seeds=(11, 22, 33),
        max_mae_cv=0.10,
        max_abs_bias=0.10,
        max_raw_negative_rate=0.10,
    )

    assert aggregate["operational_candidates"] == ["autotimes_base"]
    assert aggregate["models"]["sellm_base"]["status"] == "FAIL"
    assert not aggregate["models"]["sellm_base"]["gates"][
        "raw_negative_rate_within_limit"
    ]
