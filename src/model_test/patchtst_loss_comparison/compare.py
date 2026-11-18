"""저장된 예측값을 다시 평가하고 전후 비교 표와 그래프를 생성."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .common import SETTINGS, save_json, scores, week_date
else:
    from common import SETTINGS, save_json, scores, week_date


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    manifest = read(root / "data/manifest.json")
    data = np.load(root / "data/dataset.npz", allow_pickle=False)
    actual = data["test_y"].squeeze(-1)
    summaries, predictions, histories, rows = {}, {}, {}, []
    for group, seeds in [("primary", SETTINGS["seeds"]), ("lr_sensitivity", [42])]:
        for variant in ["before", "after"]:
            summaries[group, variant], predictions[group, variant], histories[group, variant] = [], [], []
            for seed in seeds:
                folder = root / group / f"seed_{seed}" / variant
                summary = read(folder / "summary.json")
                forecast = np.load(folder / "predictions.npz", allow_pickle=False)
                np.testing.assert_array_equal(actual, forecast["actual"])
                computed = scores(actual, forecast["prediction"])
                for key, value in computed.items():
                    if value is not None:
                        np.testing.assert_allclose(value, summary["test_metrics"][key], rtol=1e-12)
                summaries[group, variant].append(summary)
                predictions[group, variant].append(forecast["prediction"])
                histories[group, variant].append(read(folder / "pretrain_history.json"))
                rows.append({"group": group, "variant": variant, "seed": seed, "code_commit": summary["code_commit"], "pretrain_lr": summary["pretrain_lr"],
                             "pretrain_best_val": summary["pretrain_best_val"], "pretrain_best_epoch": summary["pretrain_best_epoch"],
                             "pretrain_last_gamma": summary["pretrain_last_diagnostics"]["affine_gamma"],
                             "pretrain_last_fixed_mse": summary["pretrain_last_diagnostics"]["fixed_scale_reconstruction_mse"],
                             "finetune_best_val": summary["finetune_best_val"], **computed,
                             "clipped_mae": summary["clipped_test_metrics"]["mae"],
                             "clipped_wape_pct": summary["clipped_test_metrics"]["wape_pct"]})
    write_csv(root / "comparison.csv", rows)
    x = data["test_x"].squeeze(-1)
    naive = {"last_value": np.repeat(x[:, -1:], actual.shape[1], axis=1),
             "recent_13_mean": np.repeat(x[:, -13:].mean(axis=1, keepdims=True), actual.shape[1], axis=1),
             "seasonal_52": x[:, :actual.shape[1]]}
    baseline_metrics = {key: scores(actual, value) for key, value in naive.items()}
    save_json(root / "baseline_metrics.json", baseline_metrics)

    before = np.stack(predictions["primary", "before"])
    after = np.stack(predictions["primary", "after"])
    # 각 부품의 시드 평균 절대오차 합을 계산한 뒤, 부품을 통째로 재표집.
    err_before = np.abs(before - actual).sum(axis=-1).mean(axis=0)
    err_after = np.abs(after - actual).sum(axis=-1).mean(axis=0)
    qty_part = np.abs(actual).sum(axis=-1)
    rng = np.random.default_rng(404)
    indices = rng.integers(0, len(actual), size=(4000, len(actual)))
    denom = qty_part[indices].sum(axis=1)
    valid = denom > 0
    delta = 100 * (err_after[indices].sum(axis=1) - err_before[indices].sum(axis=1))[valid] / denom[valid]
    ci = np.quantile(delta, [0.025, 0.975]).tolist()
    mean_metrics = {variant: {key: float(np.mean([s["test_metrics"][key] for s in summaries["primary", variant]]))
                              for key in ["mae", "rmse", "wape_pct", "bias_pct", "negative_prediction_pct"]}
                    for variant in ["before", "after"]}
    clipped_metrics = {variant: {key: float(np.mean([s["clipped_test_metrics"][key] for s in summaries["primary", variant]]))
                                 for key in ["mae", "rmse", "wape_pct"]} for variant in ["before", "after"]}
    delta_wape = mean_metrics["after"]["wape_pct"] - mean_metrics["before"]["wape_pct"]
    save_json(root / "aggregate.json", {"seed_mean_metrics": mean_metrics, "clipped_seed_mean_metrics": clipped_metrics,
                                       "after_minus_before_wape_pp": delta_wape,
                                       "paired_part_bootstrap_95ci_wape_pp": ci,
                                       "parts_with_lower_seed_mean_mae_after": int((err_after < err_before).sum()),
                                       "part_count": len(actual), "bootstrap_repetitions": 4000,
                                       "caveat": "single forecast origin, 64 stratified parts, 3 training seeds; not a production reproduction"})

    points = []
    for variant in ["before", "after"]:
        for seed, prediction in zip(SETTINGS["seeds"], predictions["primary", variant]):
            for index, part in enumerate(data["parts"]):
                for step in range(actual.shape[1]):
                    points.append({"variant": variant, "seed": seed, "oper_part_no": str(part),
                                   "demand_dt": int(data["weeks"][-actual.shape[1] + step]), "horizon": step + 1,
                                   "actual": float(actual[index, step]), "prediction": float(prediction[index, step])})
    write_csv(root / "test_predictions.csv", points)

    colors = {"before": "#b45b29", "after": "#076b83"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for variant in ["before", "after"]:
        traces = histories["primary", variant]
        epochs = np.arange(1, len(traces[0]) + 1)
        for ax, key in [(axes[0, 0], "validation_loss"), (axes[0, 1], "fixed_scale_reconstruction_mse")]:
            values = np.array([[r[key] for r in trace] for trace in traces])
            ax.plot(epochs, values.mean(axis=0), label=variant, color=colors[variant], lw=2)
            ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=colors[variant], alpha=0.12)
        gamma = np.array([[r["affine_gamma"] for r in trace] for trace in traces])
        axes[1, 0].plot(epochs, gamma.mean(axis=0), color=colors[variant], lw=2, label=variant)
        horizon_mae = np.abs(np.stack(predictions["primary", variant]) - actual).mean(axis=(0, 1))
        axes[1, 1].plot(np.arange(1, actual.shape[1] + 1), horizon_mae, color=colors[variant], lw=2, label=variant)
    axes[0, 0].set(title="Native SSL validation MSE", xlabel="Pretrain epoch", ylabel="Affine-dependent loss")
    axes[0, 1].set(title="SSL MSE in the same fixed scale", xlabel="Pretrain epoch", ylabel="Fixed-mask, de-affined MSE")
    axes[1, 0].set(title="Pretrain affine scale", xlabel="Pretrain epoch", ylabel="Gamma (after: fixed at 1)", ylim=(0, 1.05))
    axes[1, 1].set(title="Held-out forecast MAE by horizon", xlabel="Weeks ahead", ylabel="Demand quantity")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("PatchTST before / after | 64 parts, 3 seeds | bands: observed seed range", fontsize=13)
    fig.savefig(root / "comparison.png", dpi=170)
    plt.close(fig)

    with (root / "data/cohort.csv").open(encoding="utf-8", newline="") as stream:
        cohort = list(csv.DictReader(stream))
    strata = sorted({row["stratum"] for row in cohort})
    chosen = [min(row["oper_part_no"] for row in cohort if row["stratum"] == stratum) for stratum in strata][::2]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    dates = [week_date(int(w)) for w in data["weeks"][-(SETTINGS["lookback"] + SETTINGS["horizon"]):]]
    for ax, part in zip(axes.flat, chosen):
        index = data["parts"].tolist().index(part)
        ax.plot(dates[:52], x[index], color="#949b9c", label="Observed context")
        ax.plot(dates[52:], actual[index], color="#172c36", label="Actual", lw=2)
        for variant in ["before", "after"]:
            ax.plot(dates[52:], np.stack(predictions["primary", variant]).mean(axis=0)[index],
                    color=colors[variant], label=f"{variant} (seed mean)", ls="--" if variant == "before" else "-")
        ax.axvline(dates[52], color="#566b73", ls=":")
        ax.axhline(0, color="#777777", lw=0.5)
        ax.set(title=str(part), ylabel="Demand quantity")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Examples selected by training-cohort strata, not test performance | raw forecasts")
    fig.savefig(root / "forecast_examples.png", dpi=170)
    plt.close(fig)

    # 결론과 커밋 번호를 과거 실험에서 복사하지 않고 이번 실행 값으로 작성.
    lines = ["# PatchTST 수정 전후 학습 비교", "", "## 비교 범위", ""]
    for variant in ("before", "after"):
        commits = sorted({s["code_commit"] for s in summaries["primary", variant]})
        lines.append(f"- {variant} 코드: {', '.join(commits)}")
    lines += [
        f"- 원천: `{manifest['source']}`",
        f"- 원천 SHA-256: `{manifest['source_sha256']}`",
        f"- 입력 {SETTINGS['lookback']}주, 예측 {SETTINGS['horizon']}주, 시드 {SETTINGS['seeds']}",
        f"- 학습 정답 종료 {manifest['splits']['train_end']}, 검증 {manifest['splits']['val_start']}~{manifest['splits']['val_end']}, 테스트 {manifest['splits']['test_start']}~{manifest['splits']['test_end']}",
        "- 원천의 운영/합성 여부와 삼성 운영 당시 실행 조건의 일치 여부는 별도 확인 대상",
        "", "## 원 단위 예측", "",
        "아래 표는 시드별 지표의 평균이며 앙상블 예측 성능과 구분.", "",
        "| 지표 | 수정 전 | 수정 후 | 후 - 전 |",
        "|---|---:|---:|---:|",
    ]
    for key in ["mae", "rmse", "wape_pct", "bias_pct", "negative_prediction_pct"]:
        a, b = mean_metrics["before"][key], mean_metrics["after"][key]
        lines.append(f"| {key} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    lines += [
        "",
        f"- WAPE 차이의 부품 단위 paired bootstrap 95% 구간: [{ci[0]:+.6f}, {ci[1]:+.6f}] %p",
        "- 위 구간은 이번 표본 부품 재표집 결과이며 운영 모집단 전체의 불확실성을 대표하지 않는 범위",
        "- 낮은 native loss만으로 성능 향상이나 운영 zero-loss 원인 확정 불가",
        "", "## 동일 비음수 후처리", "",
        "| 지표 | 수정 전 | 수정 후 |", "|---|---:|---:|",
    ]
    for key in ["mae", "rmse", "wape_pct"]:
        lines.append(f"| {key} | {clipped_metrics['before'][key]:.6f} | {clipped_metrics['after'][key]:.6f} |")
    lines += ["", "원 예측과 음수 절단 결과는 별도 집계.", "", "## Pretrain 진단", "",
              "| 비교군 | 시드 | 최소 native 검증 MSE | 선택 epoch | 마지막 gamma | 마지막 고정 좌표 MSE |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['group']}/{row['variant']} | {row['seed']} | {row['pretrain_best_val']:.8e} | {row['pretrain_best_epoch']} | {row['pretrain_last_gamma']:.6f} | {row['pretrain_last_fixed_mse']:.6f} |")
    for variant in ["before", "after"]:
        zero_epochs = [
            [r["epoch"] for r in history if abs(r["validation_loss"]) < 0.000005]
            for history in histories["primary", variant]
        ]
        lines.append(f"- {variant} 시드별 .5f 기준 0으로 표시되는 검증 loss epoch: {zero_epochs}")
    lines += [
        "", "- Native MSE는 정규화 좌표계의 값이므로 고정 좌표계 재구성 오차와 함께 해석",
        "- affine 수정과 최소 마스크 및 loss 안전장치를 함께 비교하는 실험이며 단일 요인의 인과 효과 추정과 구분",
        "", "## 단순 예측 기준선", "", "| 기준선 | MAE | RMSE | WAPE (%) |", "|---|---:|---:|---:|",
    ]
    for name, metric in baseline_metrics.items():
        lines.append(f"| {name} | {metric['mae']:.6f} | {metric['rmse']:.6f} | {metric['wape_pct']} |")
    lines += ["", "## 데이터 분포", ""]
    for split in ["train", "val", "test"]:
        y = data[f"{split}_y"]
        lines.append(f"- {split}: {len(y)}개 표본, 평균 수요 {y.mean():.6f}, 0 수요 비율 {(y == 0).mean()*100:.3f}%")
    lines += ["", "## 증적", "",
              "- comparison.csv, test_predictions.csv: 시드별 지표와 원 예측",
              "- aggregate.json: 원 지표, 음수 절단 지표, paired bootstrap 구간",
              "- 각 실행 폴더의 summary.json: 실제 코드 커밋, 입력 hash, 설정, 실행 시간",
              "- comparison.png, forecast_examples.png: loss 및 예측 비교",
              ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"mean_metrics": mean_metrics, "clipped_metrics": clipped_metrics, "wape_delta_pp": delta_wape, "ci": ci, "baselines": baseline_metrics}, indent=2))


if __name__ == "__main__":
    main()
