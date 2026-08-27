# PatchMixer implementation contract

이 문서는 현재 repository에서 지원하는 PatchMixer의 public 모델, 입력, 출력, 학습 및
checkpoint 계약을 설명합니다.

## Active models

| Registry key | Public model | Public config | Capability |
|---|---|---|---|
| `patchmixer` | `PatchMixerModel` | `PatchMixerConfig` | endogenous point |
| `patchmixer_exo` | `PatchMixerExogenousModel` | `PatchMixerExogenousConfig` | exogenous point |

`patchmixer`는 Zeying-Gong/PatchMixer의 고정 upstream commit을 따르는 논문 기반 모델입니다.
과거의 `patchmixer_original`은 같은 모델을 가리키는 legacy alias이며 새 artifact key가
아닙니다. `patchmixer_exo`는 project gated-fusion 계보로, 논문 parity 경계 밖의 별도 모델입니다.

PatchMixer family 요청은 `patchmixer` 하나로 확장됩니다. Distribution과 quantile은 신규 학습
capability가 아니며 `list_available_model_keys()`에도 나타나지 않습니다.

## Tensor contract

### Endogenous

- input: finite floating tensor `[B, lookback, enc_in]`
- output: point tensor `[B, horizon, enc_in]`
- non-empty past/future exogenous input: fail-fast
- channel handling: upstream과 같은 channel-independent 계산

### Exogenous

- target input: finite floating tensor `[B, lookback, 1]`
- output: point tensor `[B, horizon]`
- past continuous: configured width가 양수이면 `[B, lookback, E_p]` 필수
- past categorical: model 내부 계약은 지원하지만 public data API는 현재 fail-fast
- future continuous: configured width가 양수이면 `[B, horizon, E_f]` 필수
- configured exogenous width: past 또는 future 중 하나 이상 필수
- output multiplier: 정확히 `1`; distribution/quantile 요청은 생성 전에 거부

`future_exo_shift_space`는 외생 입력의 정규화 여부가 아니라 target residual의 좌표계입니다.

- `output`: target denormalization 뒤 raw output에 residual을 더합니다.
- `normalized`: target RevIN 공간에서 residual을 더한 뒤 denormalize합니다.
- 기본값은 checkpoint 및 RTX 5090 정확도 근거에 따라 `output`입니다.

## Architecture ownership

`PatchMixerModel`은 upstream 수식을 그대로 소유합니다.

1. RevIN
2. patch unfold와 projection
3. separable convolution PatchMixer block
4. linear/nonlinear dual forecasting head
5. channel-independent reshape와 RevIN inverse

`PatchMixerExogenousModel`은 retired Enhanced identity를 상속하지 않습니다. Past feature는 pooled
latent gated residual로, future feature는 horizon별 target residual로 결합합니다. 두 모델은 config,
state-dict와 출력 shape가 서로 다르므로 checkpoint를 교차 로드하지 않습니다.

## Checkpoint policy

| Artifact | New training | Public registry | Restore policy |
|---|---:|---:|---|
| `patchmixer` | yes | active | v3 strict restore |
| `patchmixer_exo` | yes | active | v3 strict restore |
| `patchmixer_base` | no | hidden load-only | supported v1/v2/v3 schema only |
| `patchmixer_quantile` | no | hidden load-only | supported v3 schema only |
| `patchmixer_quantile_exogenous` | no | hidden load-only | supported v3 schema only |

과거 Enhanced, distribution 및 quantile key는 checkpoint 식별과 복원만 위해 registry의 legacy
영역에 남습니다. Public training 요청은 즉시 거부됩니다. 새 코드에서 해당 builder/class를 직접
사용하는 것은 지원 계약이 아닙니다.

포맷 정보가 없고 `model_class`가 `BaseModel` 또는 `QuantileModel`인 checkpoint는 exact
state-dict 복원이 성공할 때만 허용합니다. 현재 repository와 5090에서 발견된 2026-01-19 파일은
당시 source schema를 재구성할 수 없어 fail-closed 대상입니다. 기본 `strict=False` 호출에서도
부분 가중치 로드로 넘어가지 않습니다.

## Legacy names

- `patchmixer_original` -> `patchmixer`
- `patchmixer_exogenous` -> `patchmixer_exo`
- `PatchMixerOriginalModel` 및 `PatchMixerOriginalConfig`: Python pickle/checkpoint 호환용 hidden alias
- `patchmixer_base`, `patchmixer_quantile`, `patchmixer_quantile_exogenous`: load-only artifact key

Legacy alias는 기존 artifact를 읽기 위한 정책이며 신규 training result에 기록할 이름이 아닙니다.

## Code map

- `PatchMixer.py`: paper wrapper, project core 및 load-only identity
- `backbone.py`: paper backbone과 private legacy backbone
- `variants.py`: active exogenous model boundary
- `common/configs.py`: active configs와 hidden config aliases
- `models/registry.py`: active와 load-only registry 분리
- `model_builder.py`: active builders와 private load-only builders
- `training/model_trainers/patchmixer_train.py`: point-only active trainer

수치 기준선과 RTX 검증은 [PatchMixerBaseline.md](PatchMixerBaseline.md), 재개 조건은
[PatchMixerImprovementNeeds.md](PatchMixerImprovementNeeds.md)를 따릅니다.
