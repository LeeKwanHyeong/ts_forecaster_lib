# ExoTST public baseline

이 문서는 ExoTST의 public training, checkpoint, load, prediction 기준선과 지원 경계를
기록합니다. 상세 입출력 계약은 [ExoTSTInfo.md](ExoTSTInfo.md)를 기준으로 합니다.

## Baseline decision

`exotst_base`는 past와 future continuous exogenous input을 모두 요구하는 dedicated exogenous
forecasting artifact입니다. Endogenous-only fallback이나 categorical 입력은 지원하지 않습니다.

지원 output mode는 point, Normal, StudentT입니다. Quantile head와 SSL은 지원하지 않습니다.

## Source boundary

현재 코드와 checkpoint metadata에는 검증 가능한 upstream repository와 commit이 고정되어 있지
않습니다. 구현 주석의 `paper-aligned`는 구조적 의도를 나타내며 exact upstream parity 또는 논문
benchmark 재현을 증명하지 않습니다. 성능 비교 시 repository 구현 결과만 사용합니다.

## Frozen public contract

| Item | Baseline |
|---|---|
| Artifact key | `exotst_base` |
| Family request | `exotst` |
| Target input | `x: [B, lookback, 1]` |
| Past exogenous | `past_exo_cont: [B, lookback, E_p]`, required |
| Future exogenous | `future_exo: [B, horizon, E_f]`, required |
| Point output | model `[B, horizon]`, predictor `{"point": ...}` |
| Distribution output | Normal or StudentT parameter tensor; predictor returns location as `point` |
| Categorical exogenous | unsupported |
| Quantile / SSL | unsupported |
| Checkpoint | `modeling_module.ckpt.v3`, strict-load supported |

The model validates batch, time, and feature widths for both exogenous tensors. `exo_nan_policy` is
applied before finite-value enforcement:

- `zero`: replace NaN and infinite values with zero
- `zero+indicator`: replace invalid values and append a NaN indicator channel

Finite-input calculations are unchanged by this validation boundary.

## Characterization baseline

The tiny point configuration in `tests/test_exotst_contract.py` uses lookback 6, horizon 2, one past and
one future feature, `d_model=4`, and one encoder/fusion/decoder layer.

- Parameter count: `1,246`
- State-dict entries: `79`
- Gradients reach target history, past exogenous input, future exogenous input, and model parameters
- Both NaN policies return finite point forecasts

These counts characterize the tiny test architecture only.

## Public lifecycle evidence

The verified CPU paths include:

- point train, checkpoint, strict load, repeated predict
- future-exogenous presence, shape, and sensitivity checks
- Normal and StudentT train, checkpoint contract, exact state restoration, and prediction
- supported legacy checkpoint fixture behavior
- invalid training request rejection before model construction

The RTX 5090 CUDA smoke covers point, Normal, and StudentT public training, checkpoint creation,
strict CUDA restore, and repeated prediction. It was verified with PyTorch `2.11.0+cu130` and CUDA
runtime `13.0`.

The server's packaged cu13 NVRTC libraries require their directory in `LD_LIBRARY_PATH` for
`torch.lgamma`, which is used by StudentT log probability. Point and Normal do not exercise that kernel,
but the supported runtime contract configures the path for all modes.

Relevant regression coverage:

- `tests/test_exotst_contract.py`
- `tests/test_nhits_exotst_cuda_smoke.py`
- `tests/test_model_future_exo_contract.py`
- `tests/test_public_point_training_smoke.py`
- `tests/test_distribution_checkpoint_restore.py`
- `tests/test_legacy_checkpoint_fixtures.py`
- `tests/test_public_train_validation.py`

## Reopen criteria

Architecture changes require a measured accuracy or compatibility problem, an output/state baseline,
and repeated public lifecycle verification. Quantile support, categorical features, optional one-sided
exogenous input, or exact-upstream parity are separate capability changes rather than silent extensions
of `exotst_base`.
