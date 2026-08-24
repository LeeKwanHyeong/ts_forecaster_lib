# DSIO V100 H26 Nonnegative Output Experiment Decision

## 결정

별도 **학습 실험은 진행할 가치가 있다**. 다만 이미 검증된 `clip_zero`와 다른
단순 후처리를 비교하는 실험이 아니라, 현재 선형 point head가 음수 영역에 사용하는
학습 용량을 줄였을 때 `clip_zero` 적용 후 정확도까지 개선되는지를 확인하는
연구 실험으로 한정한다.

- 운영 `clip_zero`: 유지
- Production checkpoint 및 registry: 변경하지 않음
- 1차 실험 모델: `exotst_base`
- 2차 확장 후보: `patchtst_exogenous`
- 3차 확장 후보: `timexer_base`

## 이미 확인된 내용

정답 수요가 0 이상일 때 `max(0, raw)`는 raw 음수를 그대로 사용하는 것보다
개별 표본의 절대오차가 커질 수 없다. 따라서 MAE와 WAPE 기준으로 raw 대
`clip_zero`를 다시 비교할 필요는 없다. 기존 L52/H26 Qualification의 seed
11·22·33·42에서도 12개 모델-seed 조합 모두 동일하게 개선됐다.

| Model | 4-seed 평균 raw 음수율 | `clip_zero` 평균 MAE 개선 | 개선 범위 | 평균 sMAPE 개선 |
|---|---:|---:|---:|---:|
| `exotst_base` | 24.13% | 11.39% | 9.49-13.38% | 41.63% |
| `patchtst_exogenous` | 18.03% | 5.67% | 2.85-8.61% | 36.69% |
| `timexer_base` | 9.01% | 3.67% | 2.66-5.18% | 18.15% |

202545 Production Canary에서는 raw 음수율이 각각 38.07%, 30.56%, 22.56%로
Qualification보다 높았다. 이 Canary에는 미래 정답이 없으므로 정확도 근거로
사용하지 않고, 입력 분포가 바뀌었을 때 음수 출력이 증가할 수 있다는 위험 신호로만
해석한다.

## 왜 학습 실험이 필요한가

`clip_zero`가 raw보다 좋다는 사실은 이미 확정됐지만, 다음 질문에는 답하지 못한다.

> 모델이 처음부터 음수 영역에 덜 진입하도록 학습하면, 기존 모델에
> `clip_zero`를 적용한 결과보다도 정확해지는가?

ExoTST는 모든 seed에서 음수율이 22% 이상이고 `clip_zero`의 MAE 개선도 가장
크다. 따라서 학습 손실 변경의 효과를 관찰하기 가장 좋은 1차 대상이다.
PatchTSTExogenous는 효과가 중간 수준이며 ExoTST pilot이 통과하면 같은 계약으로
확장한다. TimeXer는 Qualification 음수율과 개선폭이 가장 작으므로 우선순위를
낮추되, Production Canary의 all-zero 자재 문제가 있어 최종 제외하지는 않는다.

## 실험 방식

Control은 현재 모델과 동일한 선형 point head, point loss, `clip_zero` 평가다.
Candidate는 모델의 output-space 예측에 다음 penalty를 추가한다.

```text
total_loss = point_loss + lambda * mean(relu(-raw_output_space_point)^2)
lambda in {0.01, 0.1, 1.0}
```

Penalty는 RevIN 정규화 좌표가 아니라 역변환이 끝난 실제 수요 좌표에서 계산한다.
모델 head와 checkpoint schema는 유지하며, 운영 평가에서는 Candidate에도
`clip_zero`를 계속 적용한다. 즉, penalty는 모델 품질 개선 후보이고 clip은 최종
방어 계약이다.

Hard ReLU output head는 음수 pre-activation에서 gradient가 끊기므로 제외한다.
`abs(raw)`는 음수를 근거 없는 양수 수요로 바꾸므로 제외한다. 학습 없이 Softplus만
붙이는 방식도 모든 양수 예측을 바꾸고 현재 checkpoint 계약과 달라지므로 제외한다.

## 실행 및 승격 기준

첫 단계는 기존과 같은 데이터 분할, loss, capacity, 최대 40 epoch를 사용한
ExoTST seed 42 pilot이다. 다음 조건을 모두 만족할 때만 seed 11·22·33으로 확장한다.

- `clip_zero` 적용 MAE 1% 이상 개선
- raw 음수율 50% 이상 감소
- 절대 normalized bias 악화 1%p 이하

다중 seed에서는 평균 MAE 1% 이상 개선, 최악 seed MAE 악화 1% 이하, 평균 음수율
50% 이상 감소, 각 horizon 구간의 MAE 악화 2% 이하를 모두 요구한다. 통과 후에도
Production refit, checkpoint 교체, Demand Engine registry 변경은 별도 승인 단계다.

## 증적

구조화된 수치, seed별 원본 지표와 CSV SHA-256은
`DSIOV100H26NonnegativeOutputExperimentDecision.json`에 고정했다. 기존 기준선은
`DSIOV100H26ExogenousValidation.md`이며 이번 결정은 해당 정확도 결과나 운영
`clip_zero` 계약을 변경하지 않는다.
