"""저장된 CSV·배열·원천에서 주요 결과를 독립 재계산하는 사후 검증 도구."""

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def audit(output):
    """실험의 지표 함수를 재사용하지 않고 CSV에서 오차 합계를 직접 계산."""
    summary = json.loads((output / "summary.json").read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        assert digest(output / name) == expected, name
    assert digest(output / "experiment.json") == summary["experiment_sha256"]
    assert digest(output / "adapter.npz") == summary["adapter_sha256"]
    assert digest(Path(experiment["source"])) == summary["source_sha256"]
    csv_rows = list(csv.DictReader((output / "test_forecast.csv").open(newline="")))
    expected_count = summary["queries"]["test"] * 27
    assert len(csv_rows) == expected_count
    assert len({(row["oper_part_no"], row["origin"], row["relative_week"]) for row in csv_rows}) == expected_count
    arrays = np.load(output / "test_comparison.npz", allow_pickle=False)
    original = np.load(output / "test_arrays.npz", allow_pickle=False)
    for name, column in (("target", "target"), ("baseline", "chronos2_p50"), ("corrected", "mora_point")):
        values = np.asarray([float(row[column]) for row in csv_rows]).reshape(-1, 27, 1)
        np.testing.assert_array_equal(values, arrays[name])
    np.testing.assert_array_equal(arrays["baseline"][:, :, 0], original["original_quantiles"][:, :, 1])
    np.testing.assert_array_equal(arrays["corrected"], arrays["baseline"] + arrays["correction"])
    np.testing.assert_array_equal(arrays["baseline"][:, 0], arrays["corrected"][:, 0])
    actual_total = math.fsum(float(row["target"]) for row in csv_rows)
    independently_computed = {}
    for method, column in (("baseline", "chronos2_p50"), ("mora", "mora_point")):
        differences = [float(row[column]) - float(row["target"]) for row in csv_rows]
        absolute_sum = math.fsum(abs(error) for error in differences)
        mse = math.fsum(error * error for error in differences) / expected_count
        values = {"mae": absolute_sum / expected_count, "mse": mse, "rmse": math.sqrt(mse),
                  "wape": absolute_sum / actual_total if actual_total else None}
        for key, value in values.items():
            expected = summary["comparison"][method][key]
            assert expected is None and value is None or math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10), (method, key)
        independently_computed[method] = values
    # 시험 정답을 CSV 기록과 별개로 실제 parquet에서 다시 대조.
    ids = original["series_ids"].tolist()
    first = min(int(row["forecast_week"]) for row in csv_rows)
    last = max(int(row["forecast_week"]) for row in csv_rows)
    source = pl.scan_parquet(experiment["source"]).filter(pl.col("oper_part_no").is_in(ids),
        pl.col("demand_dt").is_between(first, last)).select("oper_part_no", "demand_dt", "demand_qty").collect()
    lookup = {(part, int(week)): quantity for part, week, quantity in source.iter_rows()}
    for row in csv_rows:
        assert float(row["target"]) == lookup[(row["oper_part_no"], int(row["forecast_week"]))]
        origin = int(row["origin"])
        monday = date.fromisocalendar(origin // 100, origin % 100, 1) + timedelta(weeks=int(row["relative_week"]) - 1)
        year, week, _ = monday.isocalendar()
        assert int(row["forecast_week"]) == year * 100 + week
    manifests = pl.read_csv(output / "window_manifest.csv")
    groups = {name: manifests.filter(pl.col("split") == name) for name in ("train", "validation", "test")}
    ids_by_split = {name: set(frame["oper_part_no"].to_list()) for name, frame in groups.items()}
    assert not (ids_by_split["train"] & ids_by_split["validation"] or ids_by_split["train"] & ids_by_split["test"] or ids_by_split["validation"] & ids_by_split["test"])
    assert groups["train"]["target_end"].max() < groups["validation"]["origin"].min()
    assert groups["validation"]["target_end"].max() < groups["test"]["origin"].min()
    part_before = np.mean(np.abs(arrays["baseline"] - arrays["target"]), axis=(1, 2))
    part_after = np.mean(np.abs(arrays["corrected"] - arrays["target"]), axis=(1, 2))
    difference = part_after - part_before
    outcomes = summary["comparison"]["part_mae"]
    assert int((difference < -outcomes["tolerance"]).sum()) == outcomes["improved"]
    assert int((difference > outcomes["tolerance"]).sum()) == outcomes["worsened"]
    assert len(difference) - outcomes["improved"] - outcomes["worsened"] == outcomes["unchanged"]
    root = Path(__file__).resolve().parents[3]
    for relative, expected in experiment["model_sources_before"].items():
        assert digest(root / relative) == expected, relative
    sealed = json.loads((output / "fit_complete_before_test.json").read_text())
    assert not sealed["test_labels_loaded_for_evaluation"] and sealed["adapter_sha256"] == summary["adapter_sha256"]
    assert sealed["completed_utc"] < summary["completed_utc"]
    result = {"status": "passed", "independent_csv_metrics": independently_computed,
              "forecast_rows": expected_count, "parts": len(ids), "artifact_hashes_verified": len(hashes),
              "csv_npz_original_quantiles_agree": True, "raw_parquet_targets_all_matched": True,
              "calendar_all_matched": True, "part_and_target_period_disjoint": True,
              "adapter_sealed_before_test": True, "first_week_unchanged": True,
              "part_win_loss_all_matched": True, "model_source_hashes_all_matched": True}
    (output / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def figures(output):
    """시험 결과를 숨김 없이 표시. 예시 부품은 성능이 아닌 ID 해시로 선택."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    summary = json.loads((output / "summary.json").read_text())
    arrays = np.load(output / "test_comparison.npz", allow_pickle=False)
    original = np.load(output / "test_arrays.npz", allow_pickle=False)
    target, baseline, corrected = arrays["target"], arrays["baseline"], arrays["corrected"]
    colors = {"baseline": "#355C8A", "mora": "#C4562D", "observed": "#262B33"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    weeks = np.arange(1, 28)
    for key, label, values, style in (("baseline", "Frozen Chronos2 P50", baseline, "-"), ("mora", "Chronos2 + MORA point", corrected, "--")):
        ax.plot(weeks, np.mean(np.abs(values - target), axis=(0, 2)), style, color=colors[key], label=label, linewidth=2)
    ax.set(xlabel="Forecast horizon (weeks)", ylabel="MAE (demand units)", xticks=[1, 5, 9, 13, 17, 21, 25, 27],
           title="Held-out 1,050 parts | origin 2025-W18 | L52 / H27")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(output / "horizon_mae.png")
    plt.close(fig)
    before = np.mean(np.abs(baseline - target), axis=(1, 2))
    after = np.mean(np.abs(corrected - target), axis=(1, 2))
    limit = max(before.max(), after.max()) * 1.1
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.scatter(before, after, s=15, alpha=.45, color=colors["mora"], edgecolor="none")
    ax.plot([0, limit], [0, limit], color=colors["observed"], linewidth=1, linestyle=":", label="Equal MAE")
    ax.set(xscale="symlog", yscale="symlog", xlim=(0, limit), ylim=(0, limit),
           xlabel="Frozen Chronos2 part MAE (symlog scale)", ylabel="MORA part MAE (symlog scale)",
           title="Below diagonal = lower error after correction")
    ax.legend(frameon=False)
    fig.savefig(output / "part_mae.png")
    plt.close(fig)
    selected = sorted(range(len(target)), key=lambda i: hashlib.sha256(original["series_ids"][i].encode()).hexdigest())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, index in zip(axes.ravel(), selected):
        origin = int(original["origin_weeks"][index])
        monday = date.fromisocalendar(origin // 100, origin % 100, 1)
        past_dates = [monday + timedelta(weeks=offset) for offset in range(-52, 0)]
        future_dates = [monday + timedelta(weeks=offset) for offset in range(27)]
        ax.plot(past_dates, original["history"][index, :, 0], color=colors["observed"], alpha=.55, label="History")
        ax.plot(future_dates, target[index, :, 0], color=colors["observed"], label="Actual")
        ax.plot(future_dates, baseline[index, :, 0], color=colors["baseline"], label="Chronos2 P50")
        ax.plot(future_dates, corrected[index, :, 0], color=colors["mora"], linestyle="--", label="MORA point")
        ax.axvline(monday, color=colors["observed"], linewidth=1, linestyle=":")
        ax.set(title=str(original["series_ids"][index]), ylabel="Demand units")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        ax.set_xticks([past_dates[0], past_dates[26], future_dates[0], future_dates[-1]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Four ID-hash-selected test examples (not selected by performance)", fontweight="bold")
    fig.savefig(output / "forecast_examples.png")
    plt.close(fig)
    return ["horizon_mae.png", "part_mae.png", "forecast_examples.png"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir), ensure_ascii=False, indent=2))
    print(figures(args.output_dir))
