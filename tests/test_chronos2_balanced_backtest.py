"""시간 누수·유형 균형·같은 27주 평가·실패 상태를 검증하는 회귀 테스트."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from model_test.chronos2 import run_balanced_backtest as runner
from model_test.chronos2.audit_balanced_backtest import audit
from model_test.chronos2.backtest_metrics import aggregate, cluster_bootstrap, pair_metrics, paired_comparison, panel_summary
from model_test.chronos2.balanced_sampling import (CONTEXT, DEFAULT_PANELS, HORIZON, PATTERNS, Panel,
    choose_cohort, last_value_naive, load_selected_window, profile_history, week_sequence)


def pattern_values(pattern):
    values = np.zeros(CONTEXT, dtype=float)
    if pattern == "smooth":
        values[:] = 10
    elif pattern == "erratic":
        values[:] = 1
        values[-1] = 200
    elif pattern == "intermittent":
        values[::5] = 10
    elif pattern == "lumpy":
        values[::5] = 1
        values[0] = 100
    return values


def source_rows(copies=1, origin=202601):
    rows = []
    weeks = week_sequence(origin, -CONTEXT, CONTEXT + HORIZON)
    for pattern in PATTERNS:
        for i in range(copies):
            values = np.concatenate([pattern_values(pattern), np.full(HORIZON, 17.0)])
            rows.extend({"oper_part_no": f"{pattern}_{i:03d}", "demand_dt": week, "demand_qty": float(value)}
                        for week, value in zip(weeks, values))
    return rows


def test_iso_calendar_and_nonoverlapping_long_panel():
    assert week_sequence(202101, -1, 2) == [202053, 202101]
    sets = [set(week_sequence(origin, 0, HORIZON)) for origin in DEFAULT_PANELS[0].origins]
    assert all(len(weeks) == 27 for weeks in sets)
    assert sum(map(len, sets)) == len(set.union(*sets))


def test_history_classification_and_quality_exclusions(tmp_path):
    weeks = week_sequence(202601, -52, 52)
    rows = source_rows()
    extra = {"zero": [0.0] * 52, "single": [1.0] + [0.0] * 51,
             "negative": [-1.0] + [10.0] * 51, "nan": [float("nan")] + [10.0] * 51,
             "null": [None] + [10.0] * 51, "short": [1.0] * 51, "duplicate": [1.0] * 52}
    for part, values in extra.items():
        rows.extend({"oper_part_no": part, "demand_dt": week, "demand_qty": value}
                    for week, value in zip(weeks, values))
    rows.append({"oper_part_no": "duplicate", "demand_dt": weeks[0], "demand_qty": 1.0})
    rows.append({"oper_part_no": None, "demand_dt": weeks[0], "demand_qty": 1.0})
    path = tmp_path / "source.parquet"
    pl.DataFrame(rows).write_parquet(path)
    result = {row["oper_part_no"]: row for row in profile_history(path, 202601).to_dicts()}
    for pattern in PATTERNS:
        row = result[f"{pattern}_000"]
        assert row["pattern"] == pattern and row["eligibility"] == "eligible"
        positive = pattern_values(pattern)[pattern_values(pattern) > 0]
        assert row["cv2"] == pytest.approx(positive.var(ddof=1) / positive.mean() ** 2)
    for part, reason in (("zero", "no_demand"), ("single", "single_event"), ("short", "incomplete_history"),
                         ("negative", "invalid_quantity"), ("nan", "invalid_quantity"), ("null", "invalid_quantity"),
                         ("duplicate", "duplicate_week"), (None, "null_part_id")):
        assert result[part]["eligibility"] == reason
        assert result[part]["pattern"] is None
    assert result["zero"]["adi"] is None


def test_balanced_sampling_is_seeded_not_first_ids_and_future_invariant(tmp_path):
    path = tmp_path / "source.parquet"
    rows = source_rows(20)
    pl.DataFrame(rows).write_parquet(path)
    panel = Panel("test", (202601,), PATTERNS)
    profile = profile_history(path, 202601)
    selected, audit = choose_cohort(profile, panel, 202601, 5, 42)
    assert len(selected) == 20 and audit["actual_per_pattern"] == 5
    assert all(sum(row["pattern"] == p for row in selected) == 5 for p in PATTERNS)
    assert len({row["oper_part_no"] for row in selected}) == 20
    assert choose_cohort(profile.reverse(), panel, 202601, 5, 42)[0] == selected
    assert choose_cohort(profile, panel, 202601, 5, 43)[0] != selected
    assert [row["oper_part_no"] for row in selected if row["pattern"] == "smooth"] != [f"smooth_{i:03d}" for i in range(5)]
    # 미래 수요 변경과 미래에만 등장하는 부품은 표본에 영향을 주면 안 됨.
    for row in rows:
        if row["demand_dt"] >= 202601:
            row["demand_qty"] = 99999.0
    rows.append({"oper_part_no": "future_only", "demand_dt": 202601, "demand_qty": 1.0})
    pl.DataFrame(rows).reverse().write_parquet(path)
    assert choose_cohort(profile_history(path, 202601), panel, 202601, 5, 42) == (selected, audit)


def test_quota_is_per_origin_and_missing_pattern_skips_whole_origin(tmp_path):
    path = tmp_path / "source.parquet"
    pl.DataFrame(source_rows(3)).write_parquet(path)
    profile = profile_history(path, 202601)
    panel = Panel("test", (202601,), PATTERNS)
    selected, audit = choose_cohort(profile, panel, 202601, 50, 42)
    assert len(selected) == 12 and audit["actual_per_pattern"] == 3
    selected, audit = choose_cohort(profile.filter(pl.col("pattern") != "lumpy"), panel, 202601, 50, 42)
    assert selected == [] and audit["status"] == "skipped_missing_pattern"


@pytest.mark.parametrize("quota", [0, -1, 201, True])
def test_invalid_quota_rejected(quota):
    with pytest.raises(ValueError, match="per_pattern"):
        choose_cohort(pl.DataFrame(), Panel("test", (202601,), PATTERNS), 202601, quota, 42)


def test_selected_future_is_separate_and_missing_actual_fails(tmp_path):
    path = tmp_path / "source.parquet"
    rows = source_rows()
    pl.DataFrame(rows).write_parquet(path)
    ids = ["smooth_000", "intermittent_000"]
    history = load_selected_window(path, ids, 202601, future=False)
    actual = load_selected_window(path, ids, 202601, future=True)
    assert history.shape == (2, 52) and actual.shape == (2, 27)
    assert not np.shares_memory(history, actual) and np.all(actual == 17)
    pl.DataFrame(rows).filter(~((pl.col("oper_part_no") == ids[0]) & (pl.col("demand_dt") == 202601))).write_parquet(path)
    with pytest.raises(ValueError, match="실적 누락"):
        load_selected_window(path, ids, 202601, future=True)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -1.0])
def test_invalid_selected_actual_is_not_silently_dropped(tmp_path, bad):
    path = tmp_path / "source.parquet"
    rows = source_rows()
    next(row for row in rows if row["oper_part_no"] == "smooth_000" and row["demand_dt"] == 202601)["demand_qty"] = bad
    pl.DataFrame(rows).write_parquet(path)
    with pytest.raises(ValueError, match="잘못된 수량"):
        load_selected_window(path, ["smooth_000"], 202601, future=True)


def metric_fixture():
    cohort = [{"panel": "test", "origin": 202601, "oper_part_no": p, "pattern": "smooth"} for p in ("A", "B")]
    actual = np.array([[0.0] * 27, [10.0] * 27])
    return cohort, actual


def test_naive_uses_last_observation_exactly_27_times():
    history = np.array([np.arange(52), np.zeros(52)], dtype=float)
    pred = last_value_naive(history)
    assert pred.shape == (2, 27) and (pred[0] == 51).all() and (pred[1] == 0).all()
    with pytest.raises(ValueError):
        last_value_naive(history[:, :-1])


def test_zero_actual_pairs_remain_in_mae_and_wape_numerator():
    cohort, actual = metric_fixture()
    records = pl.DataFrame(pair_metrics(cohort, actual, np.ones((2, 27)), "chronos2"))
    result = aggregate(records, ["model"])[0]
    assert result["mae"] == 5.0 and result["wape_pct"] == 100.0
    assert result["zero_actual_pairs"] == 1 and result["points"] == 54
    zero = aggregate(records.filter(pl.col("oper_part_no") == "A"), ["model"])[0]
    assert zero["wape_pct"] is None and zero["mae"] == 1.0


def test_paired_wins_bootstrap_and_macro_are_reproducible():
    cohort, actual = metric_fixture()
    chrono = pair_metrics(cohort, actual, actual.copy(), "chronos2")
    naive = pair_metrics(cohort, actual, actual + 1, "naive")
    records = pl.DataFrame(chrono + naive)
    assert paired_comparison(records)["chronos_wins"] == 2
    result = panel_summary(records, 42, 200)
    assert result["macro"]["chronos2"]["equal_origin_pattern_wape_pct"] == 0
    assert result["bootstrap"] == cluster_bootstrap(records, 42, 200)
    assert result["bootstrap"]["delta_pooled_wape_pp_95pct"][1] < 0
    identical = pl.DataFrame(chrono + [{**row, "model": "naive"} for row in chrono])
    assert cluster_bootstrap(identical, 42, 200)["delta_pooled_wape_pp_95pct"] == [0.0, 0.0]
    assert paired_comparison(identical)["ties"] == 2


def test_different_cohorts_cannot_be_compared():
    cohort, actual = metric_fixture()
    records = pair_metrics(cohort, actual, actual, "chronos2") + pair_metrics(cohort[:1], actual[:1], actual[:1], "naive")
    with pytest.raises(ValueError, match="표본"):
        paired_comparison(pl.DataFrame(records))


def test_macro_is_equal_stratum_weight_not_pooled_wape():
    cohort = [{"panel": "test", "origin": 202601, "oper_part_no": "A", "pattern": "smooth"},
              {"panel": "test", "origin": 202601, "oper_part_no": "B", "pattern": "lumpy"}]
    actual = np.array([[1.0] * 27, [100.0] * 27])
    prediction = np.array([[2.0] * 27, [100.0] * 27])
    records = pl.DataFrame(pair_metrics(cohort, actual, prediction, "chronos2")
                           + pair_metrics(cohort, actual, prediction, "naive"))
    result = panel_summary(records, 42, 100)
    assert result["macro"]["chronos2"]["equal_origin_pattern_wape_pct"] == 50.0
    assert result["pooled"][0]["wape_pct"] == pytest.approx(100 / 101)


def test_all_zero_denominator_is_reported_not_hidden():
    cohort, actual = metric_fixture()
    actual[:] = 0
    records = pl.DataFrame(pair_metrics(cohort, actual, actual + 1, "chronos2")
                           + pair_metrics(cohort, actual, actual, "naive"))
    result = panel_summary(records, 42, 100)
    assert result["pooled"][0]["mae"] == 1.0 and result["pooled"][0]["wape_pct"] is None
    assert result["macro"]["chronos2"]["undefined_zero_demand_buckets"] == 1
    assert result["bootstrap"]["valid_repetitions"] == 0
    assert result["bootstrap"]["delta_pooled_wape_pp_95pct"] == [None, None]


@pytest.mark.parametrize("bad_shape", [(2, 26), (1, 27)])
def test_metric_contract_requires_identical_h27(bad_shape):
    cohort, actual = metric_fixture()
    with pytest.raises(ValueError, match="27"):
        pair_metrics(cohort, actual, np.zeros(bad_shape), "chronos2")


def install_fake_model(monkeypatch):
    class FakeModel:
        def __init__(self, config):
            self.backbone = SimpleNamespace(provenance=lambda: {"mock": True})

        def load(self):
            return self

        def predict(self, x):
            assert x.shape[1:] == (52, 1) and (x < 999).all()
            return SimpleNamespace(point=torch.from_numpy(last_value_naive(x[:, :, 0])[:, :, None]))

        def parameters(self):
            return []

    monkeypatch.setattr(runner, "Chronos2Model", FakeModel)
    monkeypatch.setattr(runner, "DEFAULT_PANELS", (Panel("test", (202601,), PATTERNS),))
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "test")


def test_runner_writes_manifest_before_inference_and_all_outputs(tmp_path, monkeypatch):
    install_fake_model(monkeypatch)
    source = tmp_path / "source.parquet"
    rows = source_rows(3)
    for row in rows:
        if row["demand_dt"] >= 202601:
            row["demand_qty"] = 999.0
    pl.DataFrame(rows).write_parquet(source)
    output = tmp_path / "run"
    args = runner.build_parser().parse_args(["--source-parquet", str(source), "--model-path", str(tmp_path),
            "--output-dir", str(output), "--per-pattern", "2", "--bootstrap-repetitions", "50"])
    result = runner.run(args)
    assert result["status"] == "complete" and result["source_unchanged"]
    assert result["panels"]["test"]["paired_mae"]["ties"] == 8
    forecast = pl.read_csv(output / "forecast_27_weeks.csv")
    assert forecast.height == 8 * 27 * 2 and (forecast["actual"] == 999).all()
    assert (output / "selection_manifest.json").exists() and (output / "report.md").exists()
    assert not (output / "failure.json").exists()
    # 실제 가중치 검증은 별도 실물 테스트 대상. 여기서는 모의 모델의 증적 재집계를 검증.
    import json
    summary = json.loads((output / "summary.json").read_text())
    summary["provenance"].update(weights_verified=True, config_verified=True)
    (output / "summary.json").write_text(json.dumps(summary))
    assert audit(output)["status"] == "passed"
    (output / "forecast_27_weeks.csv").write_text("tampered")
    with pytest.raises(ValueError, match="원 예측 변경"):
        audit(output)
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_runner_records_failure_without_resampling(tmp_path, monkeypatch):
    install_fake_model(monkeypatch)
    source = tmp_path / "source.parquet"
    rows = [row for row in source_rows() if row["demand_dt"] != 202601]
    pl.DataFrame(rows).write_parquet(source)
    output = tmp_path / "failed"
    args = runner.build_parser().parse_args(["--source-parquet", str(source), "--model-path", str(tmp_path),
                                            "--output-dir", str(output), "--per-pattern", "1"])
    with pytest.raises(ValueError, match="실적 누락"):
        runner.run(args)
    assert pl.read_csv(output / "cohort_manifest.csv").height == 4
    assert (output / "failure.json").exists() and not (output / "summary.json").exists()
