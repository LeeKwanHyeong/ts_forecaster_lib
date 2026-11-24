"""Fit and evaluate isolated, non-production TPP checkpoints on weekly demand."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
import torch

from modeling_module.api.event import (
    EventHistory,
    EventModelConfig,
    build_event_model,
    forecast_event_paths,
    list_event_model_keys,
    load_event_checkpoint,
    next_event_loss,
    save_event_checkpoint,
)

HORIZON = 26
LOOKBACK = 104
MAX_EVENTS = 32
MIN_EVENTS = 4
SEED = 42
EPOCHS = 4
PATHS = 16


def week_index(code: int) -> int:
    return date.fromisocalendar(code // 100, code % 100, 1).toordinal() // 7


def week_code(index: int) -> int:
    y, w, _ = date.fromordinal(index * 7 + 1).isocalendar()
    return y * 100 + w


def read_panel(path: Path):
    frame = pl.read_parquet(path, columns=["oper_part_no", "demand_dt", "demand_qty"])
    y = pl.col("demand_qty")
    if frame.filter(y.is_null() | ~y.is_finite() | (y < 0)).height:
        raise ValueError("Pilot requires finite, nonnegative, observed weekly quantities")
    latest = week_index(frame["demand_dt"].max())
    test_start = latest - 51
    val_start = test_start - HORIZON
    stats = (
        frame.filter(pl.col("demand_dt") < week_code(val_start))
        .group_by("oper_part_no")
        .agg(
            pl.len().alias("weeks"),
            (y > 0).sum().alias("events"),
            y.filter(y > 0).mean().alias("mean"),
            y.filter(y > 0).var(ddof=0).alias("var"),
        )
        .filter((pl.col("events") >= 10) & (pl.col("weeks") / pl.col("events") >= 1.32))
        .with_columns((pl.col("var") / pl.col("mean") ** 2 >= 0.49).alias("lumpy"))
    )
    cohorts: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    lumpy: set[str] = set()
    for is_lumpy in (False, True):
        ids = stats.filter(pl.col("lumpy") == is_lumpy)["oper_part_no"].to_list()
        ids.sort(key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode()).digest())
        if is_lumpy:
            lumpy.update(ids)
        n_train, n_validation = int(len(ids) * 0.70), int(len(ids) * 0.85)
        cohorts["train"].extend(ids[:n_train])
        cohorts["validation"].extend(ids[n_train:n_validation])
        cohorts["test"].extend(ids[n_validation:])

    weeks = {code: week_index(code) for code in frame["demand_dt"].unique()}
    grouped = (
        frame.filter(pl.col("oper_part_no").is_in(stats["oper_part_no"].to_list()))
        .sort(["oper_part_no", "demand_dt"])
        .group_by("oper_part_no")
        .agg("demand_dt", "demand_qty")
    )
    panel = {}
    for key, labels, qty in grouped.iter_rows():
        first, last = weeks[labels[0]], weeks[labels[-1]]
        if any(weeks[label] != first + i for i, label in enumerate(labels)):
            continue
        panel[key] = (first, np.asarray(qty, dtype=np.float64))
        assert first + len(qty) - 1 == last
    return frame.height, frame["oper_part_no"].n_unique(), panel, cohorts, lumpy, val_start, test_start, latest


def events(context: np.ndarray):
    indices = np.flatnonzero(context > 0)
    if len(indices) < MIN_EVENTS:
        return None
    gaps = np.concatenate(([0], np.diff(indices)))[-MAX_EVENTS:]
    return gaps, context[indices][-MAX_EVENTS:], len(context) - int(indices[-1]) - 1


def pack(samples, device="cpu") -> EventHistory:
    width = max(len(gaps) for gaps, _, _ in samples)
    gaps = np.zeros((len(samples), width), dtype=np.int64)
    qty = np.zeros((len(samples), width), dtype=np.float32)
    mask = np.zeros((len(samples), width), dtype=bool)
    for index, (g, q, _) in enumerate(samples):
        gaps[index, :len(g)] = g
        qty[index, :len(g)] = q
        mask[index, :len(g)] = True
    return EventHistory(
        torch.from_numpy(gaps).to(device),
        torch.from_numpy(qty).to(device),
        torch.from_numpy(mask).to(device),
        "week",
    )


def training_examples(panel, keys, cutoff):
    records = []
    for key in keys:
        if key not in panel:
            continue
        first, quantities = panel[key]
        stop = cutoff - first
        options = np.flatnonzero(quantities[:max(0, stop)] > 0)
        options = options[options >= LOOKBACK]
        for index in np.unique(np.linspace(0, len(options) - 1, min(10, len(options)), dtype=int)):
            target = options[index]
            history = events(quantities[target - LOOKBACK:target])
            if history is not None:
                records.append((history, history[2] + 1, quantities[target]))
    if not records:
        raise ValueError("No eligible training examples")
    return records


def holdout(panel, keys, origins, lumpy):
    rows = []
    for origin in origins:
        for key in keys:
            if key not in panel:
                continue
            first, values = panel[key]
            stop = origin - first
            if stop < LOOKBACK or stop + HORIZON > len(values):
                continue
            history = events(values[stop - LOOKBACK:stop])
            if history is not None:
                rows.append((history, values[stop:stop + HORIZON], values[stop - LOOKBACK:stop], key in lumpy))
    if not rows:
        raise ValueError("No holdout rows")
    return rows


def metrics(actual, quantity, probability) -> dict:
    total = actual.sum()
    return {
        "weekly_wape": float(np.abs(actual - quantity).sum() / total) if total else None,
        "h26_wape": float(np.abs(actual.sum(axis=1) - quantity.sum(axis=1)).sum() / total) if total else None,
        "bias": float((quantity.sum() - total) / total) if total else None,
        "occurrence_brier": float(np.mean(((actual > 0) - probability) ** 2)),
        "part_origins": len(actual),
    }


def predict_rows(model, rows, *, paths: int, seed: int):
    started = time.perf_counter()
    quantity = np.zeros((len(rows), HORIZON))
    probability = np.zeros_like(quantity)
    device = next(model.parameters()).device
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        history = [row[0] for row in batch]
        ids = tuple(str(index) for index in range(start, start + len(batch)))
        result = forecast_event_paths(
            model, pack(history, device), series_ids=ids,
            elapsed_steps=tuple(item[2] for item in history), horizon_steps=HORIZON,
            num_paths=paths, seed=seed + start,
        )
        if tuple(series.series_id for series in result.series) != ids:
            raise RuntimeError("Forecast result lost series alignment")
        for index, series in enumerate(result.series, start=start):
            for path in series.paths:
                for event in path:
                    quantity[index, event.offset] += event.quantity / paths
                    probability[index, event.offset] += 1 / paths
    return quantity, probability, round(time.perf_counter() - started, 3)


def baseline(rows, model_key):
    qty, prob = [], []
    for _, _, context, _ in rows:
        if model_key == "mean52":
            qty.append(context[-52:].mean())
            prob.append(np.mean(context[-52:] > 0))
        else:
            first = int(np.flatnonzero(context > 0)[0])
            p, z = 1 / (first + 1), context[first]
            for observed in context[first + 1:]:
                p += 0.10 * ((observed > 0) - p)
                if observed > 0:
                    z += 0.10 * (observed - z)
            qty.append(p * z)
            prob.append(p)
    return np.repeat(np.asarray(qty)[:, None], HORIZON, axis=1), np.repeat(np.asarray(prob)[:, None], HORIZON, axis=1)


def train_and_score(path: Path, out: Path) -> None:
    n_rows, n_parts, panel, cohorts, lumpy, val, test, final = read_panel(path)
    train = training_examples(panel, cohorts["train"], val)
    validation = holdout(panel, cohorts["validation"], (val,), lumpy)
    origins = (test, test + 13, test + 26)
    hold = holdout(panel, cohorts["test"], origins, lumpy)
    val_actual = np.stack([row[1] for row in validation])
    actual = np.stack([row[1] for row in hold])
    report = {
        "wheel_version": version("modeling-module"), "python": platform.python_version(),
        "torch": torch.__version__, "dataset": path.name, "rows": n_rows, "parts": n_parts,
        "validation_w0": week_code(val), "test_w0": [week_code(w) for w in origins],
        "last_week": week_code(final), "cohorts": {k: len(v) for k, v in cohorts.items()},
        "lumpy_parts": len(lumpy), "lookback": LOOKBACK, "horizon": HORIZON,
        "train_windows": len(train), "test_part_origins": len(hold),
        "validation_part_origins": len(validation), "paths": PATHS, "seed": SEED, "models": {},
    }
    print(json.dumps({k: v for k, v in report.items() if k != "models"}), flush=True)
    log_gaps = np.log([row[1] for row in train])
    config = EventModelConfig(
        "week", MAX_EVENTS, float(np.log1p([row[2] for row in train]).mean()),
        time_initial_location=float(log_gaps.mean()),
        time_initial_scale=max(float(log_gaps.std()), 0.2), min_history_events=MIN_EVENTS,
    )
    for key in ("mean52", "tsb_01"):
        report["models"][key] = metrics(actual, *baseline(hold, key))
    for key in list_event_model_keys():
        torch.manual_seed(SEED)
        model = build_event_model(key, config)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        rng = np.random.default_rng(SEED)
        best, best_epoch = math.inf, 0
        checkpoint = out / f"{key}.pt"
        for epoch in range(1, EPOCHS + 1):
            model.train()
            order = rng.permutation(len(train))
            losses = []
            for start in range(0, len(order), 64):
                rows = [train[index] for index in order[start:start + 64]]
                loss = next_event_loss(
                    model, pack([row[0] for row in rows]),
                    target_gap=torch.tensor([row[1] for row in rows]),
                    target_quantity=torch.tensor([row[2] for row in rows], dtype=torch.float32),
                )
                cost = loss.time_nll.mean() + loss.quantity_log_mse.mean()
                if not bool(torch.isfinite(cost)):
                    raise ValueError(f"Nonfinite loss for {key}")
                opt.zero_grad(set_to_none=True)
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(cost.item())
            val_qty, val_prob, _ = predict_rows(model.eval(), validation, paths=8, seed=52)
            value = metrics(val_actual, val_qty, val_prob)["h26_wape"]
            if value < best:
                best, best_epoch = value, epoch
                save_event_checkpoint(model, checkpoint)
            print(f"{key} epoch={epoch} loss={np.mean(losses):.5f} val_h26_wape={value:.5f}", flush=True)
        restored = load_event_checkpoint(checkpoint, expected_model_key=key)
        qty, prob, seconds = predict_rows(restored, hold, paths=PATHS, seed=62)
        result = metrics(actual, qty, prob)
        lumpy_rows = np.asarray([row[3] for row in hold])
        if lumpy_rows.any():
            result["lumpy"] = metrics(actual[lumpy_rows], qty[lumpy_rows], prob[lumpy_rows])
        result.update(best_epoch=best_epoch, validation_h26_wape=best, inference_seconds=seconds)
        report["models"][key] = result
        print(f"{key} test={json.dumps(result)}", flush=True)
        (out / "weekly_pilot.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def check_cuda(out: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available; no CPU fallback")
    samples = [
        (np.array([0, 3, 7, 1]), np.array([2.0, 8.0, 3.0, 12.0]), 0),
        (np.array([0, 8, 2, 3, 4]), np.array([1.0, 3.0, 2.0, 7.0, 3.0]), 11),
    ]
    info = {
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(), "wheel_version": version("modeling-module"), "models": {},
    }
    for key in list_event_model_keys():
        kwargs = dict(series_ids=("fixture-a", "fixture-b"), elapsed_steps=(0, 11),
                      horizon_steps=HORIZON, num_paths=8, seed=42)
        model = load_event_checkpoint(out / f"{key}.pt", expected_model_key=key)
        expected = forecast_event_paths(model, pack(samples), **kwargs)
        model.to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = forecast_event_paths(model, pack(samples, "cuda"), **kwargs)
        torch.cuda.synchronize()
        quantities = []
        for cpu_series, gpu_series in zip(expected.series, result.series, strict=True):
            assert cpu_series.series_id == gpu_series.series_id
            for cpu_path, gpu_path in zip(cpu_series.paths, gpu_series.paths, strict=True):
                assert [event.offset for event in cpu_path] == [event.offset for event in gpu_path]
                for a, b in zip(cpu_path, gpu_path, strict=True):
                    assert math.isclose(a.quantity, b.quantity, rel_tol=5e-4, abs_tol=1e-5)
                    quantities.append(abs(a.quantity - b.quantity))
        json.dumps(result.as_dict(), allow_nan=False)
        if not quantities:
            raise RuntimeError(f"CUDA fixture did not generate any event: {key}")
        info["models"][key] = dict(
            event_count=len(quantities), max_quantity_delta=max(quantities),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    print(json.dumps(info, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.cuda_check:
        check_cuda(args.output)
    elif args.parquet is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        train_and_score(args.parquet, args.output)
    else:
        parser.error("Specify --parquet or --cuda-check")


if __name__ == "__main__":
    main()
