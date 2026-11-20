"""MORA 합성 H27 실험과 증적 저장. 실제 수요·외부 모델·네트워크 사용 없음."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from modeling_module.forecast_adapters.MORA import MORAAdapter, MORAConfig
from .synthetic import make_synthetic, point_metrics


def run(output_dir, seed=42):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"기존 합성 결과를 덮어쓰지 않습니다: {output}")
    config = MORAConfig(period=13, correction_rank=12)
    train, validation, test = make_synthetic(config, seed=seed)
    adapter = MORAAdapter(config).fit(train, validation)
    forecast = adapter.predict(test.query)
    np.testing.assert_array_equal(forecast.point[:, 0], test.query.backbone_prediction[:, 0])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = adapter.save(output / "adapter.npz")
    restored = MORAAdapter.load(checkpoint)
    replay = restored.predict(test.query)
    np.testing.assert_array_equal(replay.point, forecast.point)
    np.savez_compressed(output / "evaluation_arrays.npz", history=test.query.history, origins=test.query.origins,
                        target=test.target, backbone=test.query.backbone_prediction,
                        corrected=forecast.point, correction=forecast.correction)
    with (output / "forecast.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query", "origin", "relative_week", "channel", "target", "backbone", "corrected"])
        for n, origin in enumerate(test.query.origins):
            for h in range(config.horizon):
                for c, name in enumerate(test.query.channel_names):
                    writer.writerow([n, int(origin), h, name, test.target[n, h, c],
                                     test.query.backbone_prediction[n, h, c], forecast.point[n, h, c]])
    before = point_metrics(test.target, test.query.backbone_prediction)
    after = point_metrics(test.target, forecast.point)
    summary = {"implementation": "paper_independent_v1", "author_source_code_copied": False, "seed": seed,
               "lookback": config.lookback, "horizon": config.horizon, "coarse_intervals": config.coarse_count,
               "correction_rank": config.correction_rank, "channels": len(test.query.channel_names),
               "queries": {"train": len(train.target), "calibration": len(validation.target), "test": len(test.target)},
               "baseline": before, "adapted": after, "mse_reduction_ratio": 1 - after["mse"] / before["mse"],
               "first_forecast_unchanged": True, "save_load_predictions_identical": True,
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "calibration": adapter.diagnostics["calibration_details"], "real_data_used": False,
               "backbone_training_executed": False,
               "caveat": "Deliberately underbiased synthetic history-only baseline; not business or paper benchmark performance"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.seed), ensure_ascii=False, indent=2))
