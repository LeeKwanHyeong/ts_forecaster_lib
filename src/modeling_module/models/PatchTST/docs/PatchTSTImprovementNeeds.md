# PatchTST freeze and improvement backlog

이 문서는 현재 PatchTST 구현을 안정 기준선으로 동결하고, 이후 다시 작업할 조건과 개선
필요 내역을 기록합니다. 구현 및 구조 기준선은 [PatchTSTBaseline.md](PatchTSTBaseline.md),
지원 기능과 API 계약은 [PatchTSTInfo.md](PatchTSTInfo.md)를 기준으로 합니다.

## Current decision

현재 PatchTST family는 source identity, public artifact 책임, parameter와 state-dict schema,
checkpoint 복원, supervised/SSL 통합 계약이 고정되어 있습니다. 따라서 더 이상 활성
refactoring 대상으로 두지 않고 PatchTST-derived production baseline으로 동결합니다.
NHITS와 ExoTST 실행 계약을 검증하는 동안 PatchTST 계산 그래프는 변경하지 않습니다.

동결된 capability routing은 다음과 같습니다.

| Capability | Frozen artifact |
|---|---|
| Endogenous point | `patchtst_base` |
| Exogenous point | `patchtst_exogenous` |
| Endogenous distribution | `patchtst_base` |
| Exogenous distribution | `patchtst_exogenous` |
| Endogenous quantile | `patchtst_quantile` |
| Exogenous quantile | `patchtst_quantile_exogenous` |

`patchtst_base`와 `patchtst_quantile`에 남아 있는 legacy exogenous routing은 checkpoint
호환 경계이며 신규 학습의 권장 경로가 아닙니다.

## Improvement needs

### P1. Cross-family accuracy position

현재 기준선은 구조와 호환성을 고정하지만 정확도 승격을 의미하지 않습니다. PatchTST는 아직
PatchMixer와 동일한 Walmart weekly, seed 11/22/33, loss, split, rolling, last-origin
프로토콜로 비교되지 않았습니다.

- 입력 계약이 허용하는 범위에서 PatchTST, PatchMixer, NHITS, ExoTST와 단순 통계 기준선을
  하나의 통제된 프로토콜로 비교합니다.
- rolling과 last-origin을 분리하고 MAE, RMSE, sMAPE 또는 WAPE를 함께 기록합니다.
- 최소 하나의 추가 domain dataset에서 모델 순위가 유지되는지 확인합니다.
- 이 근거가 생기기 전에는 PatchTST를 기본 또는 우위 family로 표현하지 않습니다.

### P1. Upstream parity boundary

현재 모델은 PatchTST의 patch-token Transformer 계보를 따르지만 공식 구현의 exact port는
아닙니다. Mean-pooled head, distribution/quantile 출력, exogenous fusion, multi-channel
projection 동작은 repository 확장입니다.

- 논문과 library의 직접 비교가 필요할 때만 exact upstream variant를 추가합니다.
- exact variant가 필요하면 `patchtst_base` 수식을 변경하지 않고 별도 artifact key를 사용합니다.
- 비교 전 upstream repository commit, 수식, 출력, gradient, channel independence, parameter 수,
  state-dict schema를 고정합니다.
- 공식 PatchTST benchmark 수치를 현재 derived 모델의 근거로 사용하지 않습니다.

### P2. Exogenous effectiveness

명시적 exogenous artifact에는 past patch concatenation과 future cross-attention 경로가 있고,
shape, gradient, checkpoint, input sensitivity 계약도 존재합니다. 다만 endogenous artifact보다
정확도가 일관되게 개선된다는 근거는 없습니다.

- 같은 초기화와 split에서 past-only, future-only, combined exogenous ablation을 비교합니다.
- 구조 조정 전에 forecast origin 시점의 feature availability와 target leakage를 검증합니다.
- attention과 gate saturation을 series별, horizon별 오차 변화와 함께 측정합니다.
- explicit exogenous artifact는 유지하고 endogenous 모델에 공통 adapter를 삽입하지 않습니다.

### P2. SSL value

`ssl_only`와 `full`은 public API에서 실행되지만 pretraining과 checkpoint transfer 성공만으로
downstream 정확도 개선이 입증되지는 않습니다.

- 동일 초기화, 데이터 양, fine-tuning budget으로 supervised-only와 SSL-finetuned를 비교합니다.
- full-data 예측과 low-data 또는 cold-start 조건을 분리해 평가합니다.
- 여러 seed와 dataset에서 개선이 일관되기 전까지 SSL은 opt-in으로 유지합니다.

### P3. Output and gradient characterization

현재 동결 테스트는 source identity, public import, parameter 수와 state schema를 고정합니다.
모든 artifact의 exact forward output과 전체 gradient signature는 고정하지 않았습니다.

- 해당 계산 경로를 실제로 변경하기 직전에만 output과 gradient characterization을 추가합니다.
- 변경 범위에 따라 RevIN, future cross-attention, distribution location denormalization,
  quantile ordering을 검증합니다.
- 계산 변경 계획이 없는 동결 기간에는 광범위한 golden test를 선행 작업으로 추가하지 않습니다.

### P3. Probabilistic calibration

Normal, StudentT, q10/q50/q90 artifact는 복원 가능하지만 calibration 품질은 현재 승격 근거에
포함되지 않습니다.

- Distribution은 NLL, CRPS와 interval coverage를 평가합니다.
- Quantile은 pinball loss, coverage, interval width와 crossing 빈도를 평가합니다.
- point 정확도와 probabilistic calibration을 별도 promotion gate로 취급합니다.
- Distribution scale과 degrees of freedom의 predictor 노출은 versioned public output 계약으로만
  다시 검토합니다.

## Deferred changes

- 측정된 결함 없이 현재 PatchTST 계산 그래프를 refactor하는 작업
- 기존 artifact key에서 legacy exogenous config routing을 제거하는 작업
- paired endogenous 정확도 근거 없이 exogenous artifact를 기본값으로 승격하는 작업
- 구조 characterization을 정확도 결과로 해석하는 작업
- 구체적인 계산 변경 전에 output 또는 gradient golden test를 추가하는 작업
- cross-family 근거 없이 family expansion이나 기본 artifact routing을 변경하는 작업

## Reopen criteria

PatchTST 모델 작업은 다음 중 적용 가능한 조건을 모두 만족할 때 다시 시작합니다.

1. 해결할 accuracy, capability, compatibility 또는 운영 문제를 수치로 정의합니다.
2. test 결과를 보기 전에 validation protocol을 확정합니다.
3. 현재 source, state-dict, parameter, import와 checkpoint 기준선을 보존합니다.
4. 변경할 계산 경로에만 집중된 output과 gradient guard를 추가합니다.
5. 동일 데이터, loss, seed와 training budget으로 후보를 비교합니다.
6. 여러 seed에서 rolling과 last-origin을 모두 평가합니다.
7. 개선이 일관되지 않으면 opt-in으로 유지하거나 기각합니다.

## Maintenance boundary

재개 조건을 만족하기 전까지 PatchTST에는 regression 수정, 의존성 또는 보안 대응, 명시적으로
승인된 비교 실험만 반영합니다. 다음 모델링 작업은 NHITS와 ExoTST의 public 실행 계약을 완전히
확인하는 것입니다. 두 family의 지원 및 미지원 경계가 문서화된 뒤 TimeMixer 원본 계보 작업을
시작합니다.
