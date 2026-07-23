# N-HiTS public baseline

이 문서는 repository의 N-HiTS 구현이 public training/inference API에 연결된 시점의 동작
기준선을 기록합니다. 사용 계약은 [NHITSInfo.md](NHITSInfo.md)를 기준으로 합니다.

## Baseline decision

`nhits_base`는 single-target endogenous point forecasting 전용 artifact입니다. 기존
`NHITS/backbone.py`의 additive residual decomposition과 interpolation 계산을 변경하지 않고,
공용 tensor API를 위한 `NHITSModel` wrapper가 `Backbone.forecast`를 직접 호출합니다.

기존 `NHITS` class는 dict batch를 받는 legacy wrapper로 보존합니다. Public registry와 새
checkpoint는 `NHITSModel`만 사용합니다.

## Source boundary

- Legacy N-HiTS source snapshot은 repository commit `f5fa39f`에서 추가되었습니다.
- 현재 repository에는 해당 snapshot의 upstream repository와 commit을 증명하는 metadata가
  없습니다.
- 따라서 이 구현은 N-HiTS 계열 계산 구조를 사용하지만, audited exact upstream port 또는 논문
  benchmark 재현본으로 표현하지 않습니다.
- 향후 exact parity가 필요하면 upstream source를 별도 artifact로 고정하고 현재
  `nhits_base` checkpoint schema를 변경하지 않습니다.

## Frozen public contract

| Item | Baseline |
|---|---|
| Artifact key | `nhits_base` |
| Family request | `nhits` |
| Input | `x: [B, lookback, 1]` |
| Model output | point tensor `[B, horizon, 1]` |
| Public prediction | `{"point": ...}` |
| Exogenous inputs | unsupported |
| Distribution output | unsupported |
| Quantile output | unsupported |
| SSL | unsupported |
| Checkpoint | `modeling_module.ckpt.v3`, strict-load supported |

The public artifact rejects non-empty past, future, or categorical exogenous inputs. It also rejects
distribution loss before data materialization. Multi-target input is rejected until a separately tested
channel policy is introduced.

## Characterization baseline

The tiny contract configuration in `tests/test_nhits_contract.py` uses one identity stack, one block,
two hidden layers of width 8, lookback 6, and horizon 2.

- Parameter count: `176`
- State-dict entries: `6`
- Wrapper output matches direct `Backbone.forecast` exactly with `rtol=0`, `atol=0`
- Forward, parameter gradient, config mapping, invalid shape, and exogenous rejection are fixed by tests

These values characterize the tiny test architecture only. Production parameter count depends on the
architecture overrides stored in each checkpoint.

## Public lifecycle evidence

The following path is exercised on CPU and RTX 5090 CUDA using the real public API:

1. `train(TrainRequest(..., models=["nhits_base"]))`
2. one supervised point-training stage through `CommonTrainer`
3. `modeling_module.ckpt.v3` checkpoint and training manifest creation
4. `load_predictor(path, strict=True)`
5. exact state-dict restoration
6. repeated deterministic `predict(...)` calls with finite output

Relevant regression coverage:

- `tests/test_nhits_contract.py`
- `tests/test_nhits_exotst_cuda_smoke.py`
- `tests/test_public_point_training_smoke.py`
- `tests/test_public_train_validation.py`
- `tests/test_model_registry.py`
- `tests/test_public_api_contract.py`

The CUDA baseline was verified with PyTorch `2.11.0+cu130`, CUDA runtime `13.0`, and NVIDIA GeForce
RTX 5090. The CUDA smoke trains, saves, strictly restores, and repeatedly predicts with the model on
`cuda:0`.

## Reopen criteria

Change the N-HiTS calculation path only when an accuracy, compatibility, or capability requirement is
defined in advance. Exogenous, probabilistic, multivariate, or exact-upstream variants require separate
artifact responsibilities, state-dict baselines, and public lifecycle tests.
