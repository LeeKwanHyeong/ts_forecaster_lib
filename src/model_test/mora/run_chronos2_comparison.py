"""고정 Chronos2 + MORA 실제 데이터 비교. 기존 모델·분할·가중치는 읽기 전용."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from modeling_module.forecast_adapters.contracts import ForecastQuery, ForecastFitData
from modeling_module.forecast_adapters.MORA import MORAAdapter, MORAConfig
from modeling_module.forecast_adapters.MORA.fitting import purged_folds
from modeling_module.models.Chronos2 import Chronos2Config, Chronos2Model
from .real_data import (SPLITS, file_sha256, identity_calendar, materialize, plan_windows,
                        restore_contract, step_week, write_json)
from .comparison_metrics import paired_comparison


OFFICIAL_WEIGHTS = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
OFFICIAL_CONFIG = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
BACKBONE_ID = "amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c:P50:L52:H27:cpu:fp32"


def log(event, **values):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}, ensure_ascii=False), flush=True)


def model_source_hashes(root):
    return {str(path.relative_to(root)): file_sha256(path)
            for path in sorted((root / "src/modeling_module/models").rglob("*.py"))}


def parameter_fingerprint(model):
    """파일뿐 아니라 메모리에 올라간 파라미터·버퍼와 동결 상태도 확인."""
    native = model.backbone._model
    if native.training or any(p.requires_grad or p.grad is not None for p in native.parameters()):
        raise RuntimeError("Chronos2가 완전히 동결된 평가 상태가 아닙니다.")
    digest = hashlib.sha256()
    for name, tensor in sorted(native.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_query(rows, history, baseline):
    return ForecastQuery(history, baseline, np.asarray([row["origin_step"] for row in rows], dtype=np.int64),
                         ("demand_qty",), tuple(row["oper_part_no"] for row in rows), BACKBONE_ID)


def fold_preflight(rows, config):
    """정답이나 모델 예측을 읽기 전에 outer·inner 검증의 표본 부족 여부 확인."""
    query = make_query(rows, np.zeros((len(rows), 52, 1)), np.zeros((len(rows), 27, 1)))
    outer = purged_folds(query, config.folds, config.min_fold_train)
    result = []
    for fit, held in outer:
        inner = purged_folds(query.take(fit), config.folds, config.min_fold_train)
        result.append({"fit": len(fit), "held": len(held),
                       "inner": [{"fit": len(a), "held": len(b)} for a, b in inner]})
    return result


def predict_and_save(model, rows, history, target, output, split_name, chunk_size=256):
    """동일한 공식 P50을 기준선과 보정 양쪽에 사용. 반복 추론 차이를 비교에서 제외."""
    started = time.perf_counter()
    baseline = np.empty((len(rows), 27, 1), dtype=np.float64)
    quantiles = np.empty((len(rows), 27, 3), dtype=np.float32)
    for start in range(0, len(rows), chunk_size):
        end = min(len(rows), start + chunk_size)
        result = model.predict(history[start:end])
        baseline[start:end] = result.point.numpy()
        quantiles[start:end] = result.quantiles.numpy()
        if start == 0 or (end // chunk_size) % 10 == 0 or end == len(rows):
            log("inference", split=split_name, complete=end, total=len(rows), seconds=time.perf_counter() - started)
    np.testing.assert_array_equal(baseline[:, :, 0], quantiles[:, :, 1])
    path = output / f"{split_name}_arrays.npz"
    np.savez_compressed(path, history=history, target=target, backbone=baseline, original_quantiles=quantiles,
                        origins=np.asarray([r["origin_step"] for r in rows]),
                        origin_weeks=np.asarray([r["origin"] for r in rows]),
                        series_ids=np.asarray([r["oper_part_no"] for r in rows]))
    return ForecastFitData(make_query(rows, history, baseline), target, "frozen_pretrained"), time.perf_counter() - started


def run(root, source_path, split_path, weights_path, output, train_windows_per_part=4):
    """한 번 고정한 계획으로 적합 → 강도 결정 → 미사용 시험 평가를 순차 실행."""
    import torch

    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if output.exists():
        raise FileExistsError(f"기존 실험 결과를 덮어쓰지 않습니다: {output}")
    manifest, source, split = restore_contract(split_path)
    if file_sha256(source_path) != source.source_sha256:
        raise ValueError("실제 parquet의 SHA-256이 기존 분할의 원천과 다릅니다.")
    if file_sha256(weights_path / "model.safetensors") != OFFICIAL_WEIGHTS or file_sha256(weights_path / "config.json") != OFFICIAL_CONFIG:
        raise ValueError("검증된 공식 가중치·설정이 아닙니다.")
    source_before = model_source_hashes(root)
    config = MORAConfig(period=52, scope="pooled_series")
    rows, excluded, quality = plan_windows(identity_calendar(source_path), split, train_windows_per_part)
    by_split = {name: [row for row in rows if row["split"] == name] for name in SPLITS}
    if any(not by_split[name] for name in SPLITS):
        raise ValueError("적합 창이 없는 분할이 있습니다.")
    preflight = fold_preflight(by_split["train"], config)
    output.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(rows).write_csv(output / "window_manifest.csv")
    pl.DataFrame(excluded, schema={"split": pl.String, "oper_part_no": pl.String, "reason": pl.String}).write_csv(output / "excluded_parts.csv")
    write_json(output / "eligibility.json", quality)
    experiment = {"status": "sealed_before_reading_quantities", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source": str(source_path), "source_sha256": source.source_sha256,
                  "split_manifest": str(split_path), "split_manifest_sha256": manifest["manifest_sha256"],
                  "window_manifest_sha256": file_sha256(output / "window_manifest.csv"),
                  "lookback": 52, "horizon": 27, "seed": 42, "train_windows_per_part": train_windows_per_part,
                  "train_window_policy": "up_to_cap_smallest_SHA256_ID_origin_seed_before_quantities",
                  "eligibility": quality, "mora": asdict(config), "fold_preflight": preflight,
                  "scope_note": "pooled_series research extension; one demand channel; unseen part IDs at validation/test",
                  "calibration_blocks": "four stable cross-sectional ID blocks at ONE validation origin, NOT four temporal folds",
                  "exogenous_used": False, "forecast_clipping": False, "quantile_calibration": False,
                  "bootstrap_draws": 2000, "bootstrap_unit": "part", "hyperparameter_search": False,
                  "weekly_index_epoch": "1970-01-05", "baseline": BACKBONE_ID,
                  "model_sources_before": source_before, "official_weights_sha256": OFFICIAL_WEIGHTS,
                  "official_config_sha256": OFFICIAL_CONFIG}
    write_json(output / "experiment.json", experiment)
    log("plan_sealed", output=str(output), eligibility=quality)
    cfg = Chronos2Config(model_path=str(weights_path), expected_weights_sha256=OFFICIAL_WEIGHTS,
                         expected_config_sha256=OFFICIAL_CONFIG, context_length=52, horizon=27,
                         device="cpu", dtype="float32", batch_size=32)
    model = Chronos2Model(cfg).load()
    before = parameter_fingerprint(model)
    inference_seconds = {}
    fitting_data = {}
    for name in ("train", "validation"):
        history, target = materialize(source_path, by_split[name])
        fitting_data[name], inference_seconds[name] = predict_and_save(model, by_split[name], history, target, output, name)
    log("mora_fit_start", train=len(fitting_data["train"].target), validation=len(fitting_data["validation"].target))
    fit_start = time.perf_counter()
    adapter = MORAAdapter(config).fit(fitting_data["train"], fitting_data["validation"])
    fit_seconds = time.perf_counter() - fit_start
    checkpoint = adapter.save(output / "adapter.npz")
    checkpoint_sha = file_sha256(checkpoint)
    write_json(output / "adapter_diagnostics.json", adapter.diagnostics)
    write_json(output / "fit_complete_before_test.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "adapter_sha256": checkpoint_sha,
        "experiment_sha256": file_sha256(output / "experiment.json"), "test_labels_loaded_for_evaluation": False,
        "test_predictions_consumed": False, "fit_seconds": fit_seconds})
    log("adapter_sealed_before_test", checkpoint_sha256=checkpoint_sha, fit_seconds=fit_seconds,
        calibration=adapter.diagnostics["calibration_details"])
    # 시험 정답은 보정 파라미터와 검증 강도가 고정된 뒤 처음 읽어 평가에만 사용.
    history, target = materialize(source_path, by_split["test"])
    test, inference_seconds["test"] = predict_and_save(model, by_split["test"], history, target, output, "test")
    correction_started = time.perf_counter()
    forecast = adapter.predict(test.query)
    correction_seconds = time.perf_counter() - correction_started
    np.testing.assert_array_equal(forecast.point[:, 0], test.query.backbone_prediction[:, 0])
    restored = MORAAdapter.load(checkpoint)
    np.testing.assert_array_equal(restored.predict(test.query).point, forecast.point)
    if file_sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("시험 평가 중 adapter 체크포인트가 변경되었습니다.")
    np.savez_compressed(output / "test_comparison.npz", target=target, baseline=test.query.backbone_prediction,
                        corrected=forecast.point, correction=forecast.correction,
                        origins=test.query.origins, series_ids=np.asarray(test.query.series_ids))
    h = np.arange(27)
    pl.DataFrame({"oper_part_no": np.repeat(test.query.series_ids, 27),
                  "origin": np.repeat([row["origin"] for row in by_split["test"]], 27),
                  "forecast_week": [step_week(int(origin) + int(offset)) for origin in test.query.origins for offset in h],
                  "relative_week": np.tile(h + 1, len(target)), "target": target.ravel(),
                  "chronos2_p50": test.query.backbone_prediction.ravel(), "mora_point": forecast.point.ravel()}).write_csv(output / "test_forecast.csv")
    comparison = paired_comparison(target, test.query.backbone_prediction, forecast.point)
    after = parameter_fingerprint(model)
    frozen_checks = {"parameter_fingerprint_before": before, "parameter_fingerprint_after": after,
                     "parameters_identical": before == after, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "model_sources_identical": source_before == model_source_hashes(root),
                     "source_identical": file_sha256(source_path) == source.source_sha256,
                     "weights_file_identical": file_sha256(weights_path / "model.safetensors") == OFFICIAL_WEIGHTS,
                     "config_file_identical": file_sha256(weights_path / "config.json") == OFFICIAL_CONFIG,
                     "split_manifest_identical": restore_contract(split_path)[0]["manifest_sha256"] == manifest["manifest_sha256"],
                     "adapter_identical_during_test": file_sha256(checkpoint) == checkpoint_sha,
                     "first_week_identical": bool(np.array_equal(forecast.point[:, 0], test.query.backbone_prediction[:, 0])),
                     "save_load_predictions_identical": True}
    if not all(value for key, value in frozen_checks.items() if key.endswith("identical")):
        raise RuntimeError("원천·모델·분할·보정 불변 검증에 실패했습니다.")
    summary = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "source_kind": "local_snapshot_user_provided_provenance", "zero_semantics": source.zero_semantics,
               "source_sha256": source.source_sha256, "split_manifest_sha256": manifest["manifest_sha256"],
               "experiment_sha256": file_sha256(output / "experiment.json"), "adapter_sha256": checkpoint_sha,
               "lookback": 52, "horizon": 27, "scope": config.scope,
               "queries": {name: len(by_split[name]) for name in SPLITS}, "eligibility": quality,
               "test_origin": sorted({row["origin"] for row in by_split["test"]}),
               "comparison": comparison, "frozen_checks": frozen_checks, "provenance": model.backbone.provenance(),
               "inference_seconds": inference_seconds, "mora_fit_seconds": fit_seconds,
               "mora_predict_seconds": correction_seconds, "total_seconds": time.perf_counter() - started,
               "calibration": adapter.diagnostics["calibration_details"],
               "versions": {name: importlib.metadata.version(name) for name in ("chronos-forecasting", "torch", "numpy", "scipy", "polars")},
               "caveats": ["Single held-out origin, not repeated-origin evaluation", "Pooled-series MORA is an explicit research extension",
                           "Validation scale groups are cross-sectional because validation has one H27 origin",
                           f"Up to {train_windows_per_part} ID/date-selected training windows per assigned part, not every rolling window",
                           "Corrected point forecasts only; original P10/P50/P90 quantiles are not recalibrated",
                           "No clipping or rounding; negative forecasts are counted", "Final holdout consumed by this experiment; future tuning needs a new holdout",
                           "Source is a user-provided local snapshot; live production provenance not independently attested"]}
    write_json(output / "summary.json", summary)
    paths = [path for path in sorted(output.iterdir()) if path.is_file()]
    write_json(output / "artifact_hashes.json", {path.name: file_sha256(path) for path in paths})
    log("complete", comparison={key: value for key, value in comparison.items() if key != "horizon"}, frozen=frozen_checks,
        total_seconds=summary["total_seconds"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("raw_data/master/tb_master_target.parquet"))
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/chronos2/pretraining_contract_20261119_seed42/split_manifest.json"))
    parser.add_argument("--weights", type=Path, default=Path("artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-windows-per-part", type=int, default=4)
    args = parser.parse_args()
    run(Path.cwd(), args.source.resolve(), args.split_manifest.resolve(), args.weights.resolve(), args.output_dir.resolve(), args.train_windows_per_part)
