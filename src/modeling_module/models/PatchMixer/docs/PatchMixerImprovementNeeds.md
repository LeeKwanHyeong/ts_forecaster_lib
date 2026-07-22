# PatchMixer freeze and improvement backlog

이 문서는 현재 PatchMixer 구현을 안정 기준선으로 동결하고, 이후 다시 작업할 조건과 개선
필요 내역을 기록합니다. 구현 및 수치 기준선은 [PatchMixerBaseline.md](PatchMixerBaseline.md),
지원 기능과 API 계약은 [PatchMixerInfo.md](PatchMixerInfo.md)를 기준으로 합니다.

## Current decision

PatchMixer는 더 이상 기본 구조를 복구하거나 평균 수준으로 끌어올리기 위한 보수 대상이
아닙니다. 논문 계보를 따르는 Original point 모델, 기존 checkpoint를 지원하는 Enhanced
계열, 모델별 exogenous variant, public builder/registry/checkpoint 계약과 RTX 5090 재현성
검증이 완료되었습니다. 따라서 현재 상태를 production/research baseline으로 동결합니다.

동결된 기본 전략은 다음과 같습니다.

| Capability | Frozen strategy |
|---|---|
| Endogenous point | `patchmixer_original` |
| Exogenous point | `patchmixer_exogenous`, 명시적 capability route |
| Distribution | `patchmixer_base` |
| Quantile | `patchmixer_quantile` |
| Exogenous quantile | `patchmixer_quantile_exogenous` |
| Future shift coordinate | `output` 기본, `normalized` opt-in |
| Normalized residual bound | 기본 `None`; `0.15` 후보는 accuracy promotion 기각 |

`patchmixer_exogenous`는 외생 입력을 처리할 수 있다는 의미이지 Endogenous보다 정확하다는
의미가 아닙니다. 미래 외생변수가 필수 요구사항이 아니면 Endogenous 경로를 우선합니다.

## Improvement needs

### P1. Cross-family accuracy position

현재 정확도 근거는 Walmart weekly 45개 series와 seed 11/22/33에 집중되어 있습니다.
PatchMixer 자체의 안정성은 확인했지만 전체 시계열 모델군에서의 상대 순위는 확정하지
않았습니다.

- PatchTST, PatchMixer, TimeMixer와 단순 통계 기준선을 같은 split/loss/seed로 비교합니다.
- rolling과 last-origin을 함께 보고 MAE, RMSE, sMAPE 또는 WAPE를 분리합니다.
- 최소 하나의 추가 domain dataset에서 순위가 유지되는지 확인합니다.
- 이 비교 전에는 PatchMixer를 SOTA 또는 전체 모델군의 우위 모델로 표현하지 않습니다.

### P2. Exogenous architecture

현재 past `z_gate`는 activation의 약 78.62%가 0.05 미만 또는 0.95 초과로 포화되었고,
Full fusion은 Endogenous보다 rolling/last-origin MAE가 일관되게 좋아지지 않았습니다. Future
shift도 현재 데이터의 target scale에서 효과가 작거나 seed별로 불안정합니다.

- 외생변수 사용이 실제 제품 요구사항일 때만 구조를 다시 엽니다.
- 입력 availability, forecast-origin 시점의 누수, upstream exogenous forecast 오차를 먼저
  계약으로 고정합니다.
- pooling이나 gate를 재설계할 경우 target history뿐 아니라 backbone forecast state를
  사용하고, Endogenous parity를 별도 guardrail로 둡니다.
- 외생 입력이 없는 모델에 공통 adapter를 삽입하는 방식은 채택하지 않습니다.

### P2. Output/normalized ensemble hypothesis

두 개의 독립 학습 모델을 약 0.5로 혼합한 validation 분석은 output 대비 rolling MAE를
seed 11/22/33에서 각각 4.471%, 1.713%, 4.553% 개선했습니다. 그러나 last-origin seed 11은
MAE 3.278%, MSE 18.058% 회귀했고 모델을 두 번 실행해야 합니다.

- 내부 residual gate와 별개의 dual-model ensemble 실험으로만 취급합니다.
- 고정 `g=0.5`를 test 전에 선언하고 held-out rolling/last-origin에서 검증합니다.
- 두 모델의 latency, VRAM, checkpoint 배포 비용을 함께 측정합니다.
- 검증을 통과하더라도 shared-backbone dual-coordinate head가 같은 이득을 유지하는지 확인한
  후 production 후보로 판단합니다.

### P3. Normalized residual reliability

Target-aware oracle은 normalized residual의 scalar gate에서 rolling MAE 6.596%, horizon-wise
gate에서 22.855%의 이론적 개선 여지를 보였습니다. 반면 9개 history summary를 사용한 nested
series-OOF ridge/KNN은 oracle gain을 회수하지 못하고 ungated normalized 모델보다
악화되었습니다.

- 현재 history-conditioned residual gate는 구현하지 않습니다.
- 다시 검토하려면 richer forecast-state representation과 genuine outer-fold model training을
  사용합니다.
- `normalized` 또는 soft bound를 일반 기본값으로 승격하지 않습니다.

### P3. Probabilistic evaluation

Distribution과 Quantile의 checkpoint 및 출력 계약은 고정되어 있지만 calibration 우위까지
입증한 것은 아닙니다.

- Distribution은 NLL 외에 CRPS와 interval coverage를 평가합니다.
- Quantile은 pinball loss, coverage, interval width, crossing 발생 여부를 비교합니다.
- point 정확도와 probabilistic calibration을 서로 대체 가능한 지표로 해석하지 않습니다.

## Rejected or deferred changes

- `future_exo_shift_space="normalized"`를 기본값으로 변경
- normalized residual soft bound `0.15`를 accuracy 기본 전략으로 사용
- 현재 9개 history feature 기반 residual gate 구현
- 공통 Exogenous Adapter를 모든 endogenous 모델에 삽입
- 검증 근거 없이 Original 수식이나 state-dict schema 변경
- 삭제된 `PatchMixer.original` compatibility import 경로 복원

## Reopen criteria

PatchMixer 구현을 다시 변경하려면 다음 조건을 순서대로 충족해야 합니다.

1. 해결하려는 accuracy, capability 또는 운영 문제를 수치로 정의합니다.
2. 현재 baseline의 출력, state dict, parameter count, public import와 checkpoint 계약을
   고정합니다.
3. 후보 선택은 validation에서 끝내고 test 결과로 hyperparameter를 재선택하지 않습니다.
4. 동일 데이터/loss/seed의 multi-seed rolling 및 last-origin 결과를 모두 비교합니다.
5. 공용 경로를 변경하면 latency, VRAM, save/load/predict와 전체 회귀 테스트를 포함합니다.
6. 일관된 개선이 없으면 후보는 opt-in 실험으로 남기거나 제거하고 기본 전략은 유지합니다.

## Maintenance boundary

현재부터 PatchMixer에는 회귀 수정, 보안/의존성 대응, 명시적으로 승인된 비교 실험만
반영합니다. 이름 변경, helper 정리, 추가 abstraction 같은 유지보수성 작업은 실제 문제나
새 기능이 발생하기 전까지 진행하지 않습니다. 다음 주력 모델링 작업은 동일 프로토콜의
cross-family benchmark와 TimeMixer 기준선 구축을 우선합니다.
