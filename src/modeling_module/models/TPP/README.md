# Event Forecasting Models

The event-model registry is available through `modeling_module.api.event`.
It is separate from the regular-window `train()`, `forecast()`, and `load_predictor()`
paths and does not admit a model for Demand Engine execution.

| Key | Backbone |
| --- | --- |
| `rmtpp_matched_head` | GRU event encoder |
| `phen_tpp` | Frozen research baseline: shared causal encoder, Shared15 history correction, static prototype retrieval |
| `s2p2_matched_head` | Two backward-ZOH state-space layers with the research affine scan |

All three use the same rounded log-normal time law and log1p-quantity point
head. RMTPP and S2P2 are matched-head adaptations, not full reproductions of
the respective paper likelihoods. Do not infer upstream scan complexity or
throughput from the local S2P2 implementation.

## Contract

- A batch is `EventHistory(gaps, quantities, mask, time_unit)`, with matching
  right-padded `[batch, events]` tensors and a boolean mask.
- Observed quantities are finite and positive. Observed gaps use positive
  integer steps in the configured unit; a zero at the first position can mark
  an unknown predecessor boundary. Padding is not a demand observation.
- `EventModelConfig` records the event-count bounds, unit, and training-only
  initialization statistics. The model does not impose a 52-week window.
- `predict_next_event()` returns parameters for a rounded positive log-normal
  duration, plus a single quantity point. The duration is measured from the
  last event, not the forecast anchor. It exposes log mass for a discrete gap
  and log survival for a bound of fully elapsed steps. It does not itself
  simulate future events.
- `next_event_loss()` returns observed-target time NLL and squared error on
  log1p quantity. The target is not appended to the encoder's input.
- `save_event_checkpoint()` and `load_event_checkpoint()` persist and validate
  version, model key, config and strict `state_dict`; restore is CPU/eval by
  default. These files do not resume the optimizer or an experiment runner.

```python
import torch
from modeling_module.api.event import (
    EventHistory, EventModelConfig, build_event_model, predict_next_event,
)

config = EventModelConfig("week", max_history_events=32, train_log_mean=1.2)
model = build_event_model("phen_tpp", config).eval()
history = EventHistory(
    gaps=torch.tensor([[0, 3, 8]]),
    quantities=torch.tensor([[4.0, 7.0, 3.0]]),
    mask=torch.tensor([[True, True, True]]),
    time_unit="week",
)
prediction = predict_next_event(model, history)  # Fresh weights for API example only.
```

In a training run, calculate the initialization values from Train only.
Load a fitted checkpoint, not the random example weights, for forecasting.
Time probability calculations use float64. CPU was exercised; CUDA still needs
validation in the intended Runtime, and the API rejects MPS devices.

## Forecast Path Contract

`forecast_event_paths()` uses an evaluated model and the same `EventHistory`.
Pass one unique `series_id` and one integer `elapsed_steps` per history row,
plus the horizon, path count, and seed. Elapsed steps count the complete
no-event buckets **after** the last event and **before** the forecast origin.
For example, if the last event was W-3 and W-2 and W-1 are observed zero,
`elapsed_steps=2`; a sampled gap of 3 becomes offset 0 (W0).

```python
import json
from modeling_module.api.event import forecast_event_paths, load_event_checkpoint

model = load_event_checkpoint("fitted_phen_weekly.pt", expected_model_key="phen_tpp")
result = forecast_event_paths(
    model, history, series_ids=("PART-A",), elapsed_steps=(2,),
    horizon_steps=26, num_paths=32, seed=42,
)
message = json.dumps(result.as_dict(), allow_nan=False)
```

- Contract: `modeling-module.event-paths.v1`. Result metadata contains the
  model key, time unit, horizon, and seed. `series` follows the input order;
  each entry has its `series_id` and the requested number of `paths`.
- A path is an ordered list of `{offset, quantity}` events. Offsets are
  integers in `[0, horizon_steps)`. An empty list is one valid no-event
  scenario, not a missing series or a failed forecast.
- The first gap is sampled conditional on `D > elapsed_steps`. Probability
  beyond the horizon is kept. Later gaps use the simulated history from each
  path, retaining at most the configured number of recent events.
- Quantities are point predictions at sampled arrival times; paths do not
  represent independently sampled quantities or calibrated quantity intervals.
  ISO week conversion, zero filling for output buckets, and aggregation belong
  to the caller. This API does not train on right-censored targets.
- Path sampling requires an uncapped-duration checkpoint (`top_code=None`).
  It does not place an entire top-coded tail at the final code. The one-step API
  still supports top-coded checkpoints. Seeded replay targets the same inputs,
  checkpoint, device, and software versions.

## Source Revisions

The source reference is `paper_research` commit
`58e187c4f3c353d29f2862b16ad7e850640624e8`, package `phen`. Backbones, common
heads, and time likelihood were taken from that package, with local imports
adapted and the duration parameters exposed to the public event API.
The S2P2 vendor snapshot is from `UCIDataLab/state_space_point_process`
revision `c3933240` as recorded by the research manifest. The source copies
and their `LICENCE` and `NOTICE` under `_vendor/s2p2/` remain unchanged.

`tests/test_tpp_research_parity.py` runs against the separate research source
when `TPP_RESEARCH_ROOT` is provided. This checks initialization, outputs and
gradients on synthetic events, not the original training runs or datasets.

Local synthetic benchmark, arm64 CPU, PyTorch 2.9.1, one thread, batch 16:
at 64 events per series, median next-event latency was 1.50 ms (RMTPP),
5.66 ms (PHEN) and 4.20 ms (S2P2), over 25 calls after five warmups.
This does not measure GPU latency or a 26-week rollout.
