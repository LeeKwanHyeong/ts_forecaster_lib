# DSIO V100 H26 ExoTST Negative-output Penalty Pilot

## 결론

ExoTST의 음수 출력 penalty는 구현 및 검증을 완료했지만, seed 42 pilot의 세 후보가
모두 승격 기준을 통과하지 못했다. 따라서 seed 11/22/33 다중 검증으로 확장하지
않으며 운영 `clip_zero`, Production checkpoint, Demand Engine registry는 변경하지
않는다.

기본값 `negative_output_penalty_weight=0.0`은 penalty hook을 생성하지 않으므로 기존
forward, point loss, state dict 및 checkpoint 동작과 동일하다. 양수 penalty는
RevIN 역변환 이후 실제 수요 좌표의 point 예측에만 학습 regularizer로 적용된다.
Validation loss에는 penalty를 더하지 않아 기존 point loss로 best checkpoint를
선택한다.

## 실험 조건

- 모델: `exotst_base`
- Lookback / Horizon: L52 / H26
- 학습 상한: 202435
- 검증 구간: 202436~202509
- Validation series: 6,952개
- Seed / epoch: 42 / 40
- Lambda: 0.01, 0.1, 1.0
- State selection: 기존 validation point loss 기준 best state

승격 조건은 clip 이후 MAE 1% 이상 개선, raw 음수율 50% 이상 감소, 절대
normalized bias 악화 1%p 이하를 모두 만족하는 것이다.

## 결과

| Lambda | Clip MAE | MAE 개선 | Raw 음수율 | 음수율 감소 | 절대 bias 악화 | 판정 |
|---:|---:|---:|---:|---:|---:|---|
| Control | 1.25480 | - | 23.95% | - | - | 기준선 |
| 0.01 | 1.23986 | +1.19% | 23.97% | -0.06% | +2.08%p | 탈락 |
| 0.1 | 1.33132 | -6.10% | 25.82% | -7.80% | -0.01%p | 탈락 |
| 1.0 | 1.39183 | -10.92% | 22.21% | +7.27% | +3.40%p | 탈락 |

`lambda=0.01`은 clip MAE만 개선하고 음수율과 bias 조건을 통과하지 못했다.
`lambda=0.1`은 MAE와 음수율이 모두 악화됐다. `lambda=1.0`은 음수율을 일부
낮췄지만 목표 50%에 크게 못 미쳤고 MAE와 bias가 함께 악화됐다. 이 결과는 단순
제곱 penalty의 강도를 높여도 음수 발생이 단조롭게 감소하지 않으며, 전체 예측
분포와 정확도의 균형도 악화될 수 있음을 보여준다.

## 검증 범위

- penalty 0 출력·loss·state dict parity
- 음수 point 예측에 대한 양의 gradient와 양수 예측의 zero gradient
- public train, save, strict load, predict 통합 경로
- qualification runner의 config 및 checkpoint metadata 일치
- Demand Engine `clip_zero` Canary 회귀
- RTX 5090 세 후보의 checkpoint SHA-256, finite prediction 및 평가 receipt

세 후보의 학습 시간은 각각 약 20분이며 VRAM은 약 4.2GB로 유지됐다. 후보별
checkpoint는 연구 artifact 디렉터리에만 저장됐고 운영 checkpoint를 덮어쓰지
않았다.

구조화된 결과와 provenance는
`DSIOV100H26ExoTSTNegativePenaltyPilot.json`에 고정한다. 원격 pilot receipt의
파일 SHA-256은
`5922f2c205d9402000174c549449e2a529ae78f9e60e65a73469d9284a3bce1d`이다.

## 후속 판단

현재 제곱 penalty는 다중 seed 실험 대상에서 제외한다. 음수 출력 자체를 더 줄여야
한다면 다음 후보는 동일 penalty의 세부 lambda 탐색보다 target scale에 따른
정규화 penalty, zero-demand 전용 gate, 또는 양수 분포 head처럼 loss와 출력 계약을
함께 설계하는 방향이 더 적절하다. 이 후보들은 별도 실험 승인 전에는 구현하지
않는다.
