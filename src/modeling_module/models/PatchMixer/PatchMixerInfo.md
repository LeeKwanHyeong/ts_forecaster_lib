# PatchMixer implementation notes

이 문서는 논문 일반론이 아니라 현재 repository 구현과 public API 계약을 설명합니다. 최종 지원
범위는 repository root의 `README.md`와 `README.package.md` 표를 기준으로 합니다.

## Registered artifacts

| Artifact key | Output mode | Checkpoint-safe loss |
|---|---|---|
| `patchmixer_base` | point 또는 distribution | point loss, `Normal`, `StudentT` |
| `patchmixer_exogenous` | exogenous point | point loss only |
| `patchmixer_original` | endogenous point | point loss only |
| `patchmixer_quantile` | q10/q50/q90 | quantile loss |
| `patchmixer_quantile_exogenous` | exogenous q10/q50/q90 | quantile loss |

`patchmixer_base`는 point 전용 모델이 아니라 `out_mul`에 따라 point와 distribution parameter를
출력하는 unified model입니다. Quantile은 별도 artifact입니다. Distribution checkpoint의 public
prediction 결과는 현재 location을 `{"point": ...}`로 노출합니다.

기본 선택 전략은 기능별로 구분합니다. 내생변수 point 예측은 RTX 5090 3-seed 검증에서 승격된
`patchmixer_original`, 외생변수 point는 `patchmixer_exogenous`, distribution은
`patchmixer_base`, quantile은 입력 계약에 따라 `patchmixer_quantile` 또는
`patchmixer_quantile_exogenous`를 사용합니다. 코드는 `get_patchmixer_default_model_key`로
조회합니다. 기존 `patchmixer` family 확장 순서와 checkpoint alias는 호환성을 위해 변경하지
않습니다.

`patchmixer_exogenous`는 외생 입력을 요구하는 호출의 명시적 capability route입니다. 현재
`z_gate + future_shift` 구현은 RTX 5090 3-seed ablation에서 Endogenous보다 정확도가 낮았으므로
accuracy promotion 상태가 아닙니다. 상세 수치와 제약은
[PatchMixerBaseline.md](PatchMixerBaseline.md)에 고정합니다.

## Data contract

- target input: `(B, lookback, 1)`
- past continuous exogenous: 선택 사항; latent gate 경로로 주입
- future continuous exogenous: 선택 사항; horizon별 output shift로 주입
- categorical exogenous: public training과 prediction API에서 fail-fast

Future width가 설정된 checkpoint는 public prediction 시 batch 공용 `(horizon, future_exo_dim)`
또는 `(B, horizon, future_exo_dim)`을 요구합니다. 누락과 차원 오류는 즉시 실패하며, future
width가 0인 모델은 non-empty future input을 거부합니다.

## Current base path

1. target을 RevIN으로 정규화합니다.
2. `PatchMixerBackbone`이 patch 기반 latent vector를 만듭니다.
3. 구성된 past continuous feature와 optional part embedding을 latent에 결합합니다.
4. `TemporalExpander`와 MLP head가 horizon별 output을 만듭니다.
5. point branch는 마지막 관측값을 level anchor로 더하고, target scale 복원 후 optional future
   shift와 nonnegative transform을 적용합니다.
6. distribution branch는 location에 level anchor와 depthwise refinement를 적용하고 location/scale을
   target scale에 맞춘 뒤 parameter tensor를 다시 조립합니다.

과거 문서의 point dual-head와 point depthwise-refinement 설명은 현재 구현과 다르므로 더 이상
계약으로 사용하지 않습니다.

## Quantile path

`PatchMixerQuantileModel`은 q10/q50/q90을 출력하는 별도 artifact입니다. Base distribution mode와
혼용하지 않으며 public family `patchmixer` 요청은 base 다음 quantile 순서로 확장됩니다.

## Code map

- [model and exogenous paths](PatchMixer.py)
- [configuration](common/configs.py)
- [trainer integration](../../training/model_trainers/patchmixer_train.py)

지원 경계는 registry, public validation, CPU point smoke, distribution restore, future-exogenous
sensitivity 회귀 테스트로 고정합니다.
