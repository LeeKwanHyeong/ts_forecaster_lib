# PatchMixer freeze and improvement backlog

PatchMixer는 두 활성 point 모델의 계약을 기준으로 동결합니다. 구현 및 수치 기준선은
[PatchMixerBaseline.md](PatchMixerBaseline.md), API 계약은
[PatchMixerInfo.md](PatchMixerInfo.md)를 기준으로 합니다.

## Frozen decision

| Capability | Strategy |
|---|---|
| Endogenous point | `patchmixer` |
| Exogenous point | `patchmixer_exo` |
| Future residual coordinate | `output` default, `normalized` opt-in |
| Distribution/quantile | 신규 학습 미지원; 과거 지원 schema만 load-only |

`patchmixer`는 논문 기반 기본 endogenous 모델입니다. `patchmixer_exo`는 외생 입력이 실제로
필요할 때 선택하는 capability route이며 endogenous보다 정확하다는 의미가 아닙니다.

## Improvement backlog

### P1. Cross-family accuracy

현재 승격 근거는 Walmart weekly 45개 series와 seed 11/22/33에 집중되어 있습니다.

- PatchTST, PatchMixer, NHITS와 향후 TimeMixer를 동일 split/loss/seed로 비교합니다.
- rolling과 last-origin의 MAE, RMSE, sMAPE 또는 WAPE를 분리합니다.
- 최소 한 개의 추가 domain dataset에서 순위가 유지되는지 확인합니다.
- 이 검증 전에는 PatchMixer를 전체 모델군의 우위 모델로 표현하지 않습니다.

### P1. Exogenous gate redesign

기존 past gate activation의 약 78.62%가 0.05 미만 또는 0.95 초과로 포화되었고, full fusion은
endogenous 대비 일관된 multi-seed 개선을 만들지 못했습니다.

- feature availability와 forecast-origin 누수 경계를 먼저 고정합니다.
- upstream exogenous forecast 오차를 포함한 평가를 추가합니다.
- pooling/gate를 재설계하면 backbone forecast state를 조건으로 사용합니다.
- exogenous feature를 제거했을 때의 endogenous parity를 promotion guardrail로 둡니다.
- 공통 Exogenous Adapter를 endogenous 모델에 삽입하는 방식은 사용하지 않습니다.

### P2. Normalized residual reliability

`normalized`는 일부 rolling window에서 가능성을 보였지만 seed 33과 last-origin에서 불안정했고,
history summary 기반 gate도 oracle gain을 회수하지 못했습니다.

- `normalized` 또는 residual limit을 기본값으로 승격하지 않습니다.
- 재검토 시 series, horizon, history scale별 failure slice를 먼저 정의합니다.
- 후보 선택은 validation에서 끝내고 test 결과로 gate를 다시 선택하지 않습니다.

### P2. Probabilistic replacement

retired distribution/quantile 구현을 다시 public training에 연결하지 않습니다. 필요해지면 현재
point 모델에 조건문을 복원하는 대신 별도 모델 책임과 다음 평가 계약을 먼저 설계합니다.

- pinball loss, CRPS, interval coverage와 width
- quantile crossing
- v3 checkpoint metadata와 exact restore
- point baseline의 public API 비회귀

### P3. Pre-version checkpoint migration

발견된 `BaseModel`/`QuantileModel` 파일은 저장 당시 source schema가 repository history에 없어
현재 코드에서 정확히 복원할 수 없습니다.

- historical environment 또는 배포 image를 찾습니다.
- 원래 class/state schema로 strict load합니다.
- 예측 golden sample을 만든 뒤 v3 metadata와 함께 다시 저장합니다.
- migration 전에는 부분 로드 결과를 운영 예측으로 사용하지 않습니다.

## Reopen criteria

1. 해결하려는 accuracy, capability 또는 운영 문제를 수치로 정의합니다.
2. 현재 output, gradient, state-dict, parameter count와 public import를 고정합니다.
3. 동일 데이터/loss/seed의 multi-seed rolling 및 last-origin 결과를 비교합니다.
4. save/load/predict, latency, VRAM과 전체 CPU 회귀를 함께 검증합니다.
5. 일관된 개선이 없으면 후보를 opt-in 연구로 남기거나 제거합니다.

다음 주력 모델링 순서는 NHITS/ExoTST 실행 기준선 이후 TimeMixer 논문 계보 구축입니다.
