# LTB Lifecycle Model Baseline

## 현재 결정

LTB Lifecycle 예측은 **초기 12개월 관측으로 이후 72개월을 예측**하는
별도 문제로 관리한다. 일반 시계열 모델의 장기 horizon을 늘려 해결하지
않고, 완료된 Lifecycle을 학습하는 통계 모델과 시간 순서를 지킨 장기 Tail
보정을 함께 사용한다.

현재는 단일 모델을 운영 기본값으로 확정하지 않는다. 다음 두 기준선을
동시에 유지한다.

- **CGMM + cohort/tail correction**: 확률분포와 불확실성을 제공하는 주 후보
- **Similar Lifecycle + cohort/tail correction**: 최신 cohort 변화에 대한
  견고성과 설명 가능성을 확인하는 비교 기준선

Production registry는 변경하지 않는다.

## 2026-08-20 M0 및 Tail 개선 기준선

현재 DSDM 시간순 8:2 데이터에서는 기존 모델 용량을 유지한 채 M0 ordinal,
cohort strength, Tail half-life, scale gate를 분리해 비교했다.

- Dataset fingerprint:
  `12a4eb4d02e3c7a4222cc02c59de7dd4e55922005460e627129d2b5de36b6b22`
- Train: M0 `2020-03` 이하, 5,620개
- Validation: M0 `2021-01~2023-03`, 1,380개
- 설정 선택: Train 5,620개 내부의 시간순 rolling 구간만 사용
- 입력 경계: M0 ordinal은 사용하고 실제 미래 감소 속도는 사용하지 않음

Train 내부 단일 요인 비교 후 선택된 값을 조합한 결과는 다음과 같다.

- CGMM: `static_observed_m0_v1`, cohort strength `0.25`, Tail half-life
  `48`, scale gate quantile `0.25`
- Similar Lifecycle 후보: `static_observed_m0_v1`, cohort strength `0.0`,
  Tail half-life `48`, scale gate quantile `0.25`

고정된 Validation에서 기존 기준선과 비교한 결과는 다음과 같다.

| 모델 | 설정 | WAPE | 수량 편향 | 90% coverage | Interval score |
|---|---|---:|---:|---:|---:|
| CGMM | 기존 | 0.7370 | +64.80% | 0.9088 | 59.0856 |
| CGMM | M0 및 Tail 후보 | **0.5962** | **+48.50%** | 0.8836 | **54.2962** |
| Similar Lifecycle | 기존 | **0.7632** | **+66.58%** | **0.7354** | **76.0036** |
| Similar Lifecycle | M0 및 Tail 후보 | 0.7714 | +68.68% | 0.6691 | 79.2129 |

CGMM 후보는 WAPE를 `19.11%` 줄이고 절대 편향을 `25.15%` 줄였다. 특히
M37~M72 WAPE가 `1.0869`에서 `0.6345`로 감소했다. 다만 M1~M12 WAPE는
`0.3255`에서 `0.3880`으로 증가했으므로 단기 구간은 후속 개선 대상이다.

Similar Lifecycle 후보는 장기 WAPE는 줄였지만 전체 WAPE와 편향 및 구간
품질이 악화됐다. 따라서 현재 개발 기준선은 **개선된 CGMM을 채택 후보로
고정하고 Similar Lifecycle은 기존 설정을 유지**한다. 이 결정은 production
registry, wheel, 5090 Runtime을 변경하지 않는다.

Train 선택 report SHA-256:
`182a81ab9b447b01ae7c43e40fcb98b51cbd594c74d0d314e6912db9b979197c`

Validation 비교 report SHA-256:
`9922dce0c76899076ae1326892592c722621eac1828c2312b7128dcc97fa9119`

## 검증 데이터와 경계

- Dataset fingerprint:
  `bc42c334a9f2bf7e2b3db7ba797400cb1df50caa72301e2284509cca0433a2be`
- Train: 4,978 Lifecycle
- Validation: 1,002 Lifecycle
- Test: 1,020 Lifecycle
- 입력: M0~M11 12개월
- 출력: M12~M83 72개월
- 모델과 보정 정책 선택: forward-only rolling Validation
- 최종 refit: Train + Validation
- 기존 Test: 선택이 끝난 후 평가했으며 현재는 이미 결과를 확인한 구간

검증 근거는 `DSIODemandEngine/reports/ltb_cgmm/`
`v100_ltb_model_suite_bc42c334a9f2.json`에 고정했다.

- Report SHA-256:
  `b60fd91a9fd250b10ef5a2977c9d8b90c9fa0f83370f6e662df0a32f90d6c699`

## 정확도 결과

| 모델 | Rolling Validation WAPE | Final Test WAPE | Final Test MAE |
|---|---:|---:|---:|
| Similar Lifecycle | 0.3537 | 0.9625 | 17.3301 |
| Similar Lifecycle + correction | **0.3026** | **0.5718** | **10.2950** |
| CGMM | 0.3591 | 0.9675 ± 0.0086 | 17.4200 수준 |
| CGMM + correction | **0.2965 ± 0.0004** | **0.6304 ± 0.0067** | **11.3509** |

Rolling Validation에서는 CGMM 보정 모델이 약 2% 우세했다. 반면 Final
Test에서는 Similar Lifecycle 보정 모델의 WAPE가 CGMM보다 약 9% 낮았다.
이 순위 역전 때문에 현재 Test 결과만 이용해 CGMM을 버리거나 Similar
Lifecycle을 새 운영 기본값으로 승격하지 않는다.

현재 역할은 다음과 같이 고정한다.

1. **CGMM 보정 모델**은 rolling Validation에서 선택된 확률적 주 후보다.
2. **Similar Lifecycle 보정 모델**은 Final Test에서 더 강한 견고성 기준선이다.
3. 이후 모델은 두 기준선을 모두 넘어야 하며, 한쪽만 비교해서는 안 된다.

## 불확실성 결과

두 모델 모두 nominal 90% 구간으로 비교했다.

| 모델 | Test coverage | 평균 구간 폭 | 평균 interval score |
|---|---:|---:|---:|
| Similar Lifecycle + correction | 0.7848 | 29.6368 | 52.7692 |
| CGMM + correction | 0.9293 ± 0.0014 | 40.5257 | 45.3347 |

Similar Lifecycle 구간은 이웃 곡선의 가중 분산을 정규분포로 근사한
구간이므로 과소포착 문제가 있다. CGMM은 구간이 더 넓지만 목표 coverage에
가깝고 interval score도 더 좋다. 불확실성이 필요한 API에서는 CGMM을
기준으로 유지한다.

## 2023 Chronological Qualification

기존 Test 전체를 이미 확인했기 때문에 해당 구간을 새 모델 선택에 다시 쓰지
않는다. 현재 스냅샷에는 2023-03 이후의 완료 Lifecycle이 없으므로, 가장 늦은
Lifecycle 시작 연도인 2023년 cohort를 **고정 정책 시간 스트레스 검증**으로
분리했다.

- 저장소와 보정 근거: 2018-01~2022-03, 6,540개
- Qualification: 2023-01~2023-03, 460개
- Qualification fingerprint:
  `f2127c5c887b4d7db84d44795328d2d3e8a8cdf4267ca750c16ea14803aa3b47`
- Report SHA-256:
  `fb6daae95b4d0c9dee9a12bdfcaad4c7b043eddbdb20f858e7b183fe4b0d3aaf`
- 정책: CGMM 2 components/PCA 2, seed 11·22·33, Similar K=15,
  `cohort_half_plus_tail_72`

| 모델 | WAPE | MAE | 편향 | 90% coverage |
|---|---:|---:|---:|---:|
| Similar Lifecycle raw | 1.5282 | 17.7043 | +151.7% | 0.6560 |
| **Similar Lifecycle + correction** | **0.8064** | **9.3429** | **+78.8%** | 0.8042 |
| CGMM + correction | 0.8880 ± 0.0313 | seed별 산출 | +82.3~89.3% | **0.9539 ± 0.0031** |

점 예측에서는 Similar Lifecycle이 다시 우세했고, 구간 coverage에서는 CGMM이
우세했다. 두 모델 모두 최근 cohort를 여전히 크게 과대예측하므로 어느 한쪽도
운영 기본값으로 승격하지 않는다.

이 구간은 기존 Test 보고서에서 레이블을 이미 확인했다. 따라서
`pristine_holdout=false`, `model_selection_allowed=false`이며 결과는 고정된
정책의 시간 안정성 확인에만 사용한다. 현재 pristine qualification 상태는
`blocked_no_unseen_completed_lifecycle_cohort`이며, 2023-03보다 늦고 한 번도
평가하지 않은 완료 Lifecycle을 확보하기 전까지 rolling validation만 개발
참고 자료로 사용한다.

## Artifact 계약

- CGMM은 전처리, mixture 파라미터, correction state를 하나의 checksum
  artifact로 저장한다. seed 11/22/33 모두 strict load 후 전체 분포 출력이
  bitwise 동일했다.
- Similar Lifecycle도 전처리, 이웃 저장소와 correction state를 하나의
  checksum artifact로 저장한다. strict load 후 평균, 표준편차, 구간, 이웃
  ID·가중치·거리가 모두 bitwise 동일했다.

Similar Lifecycle 모델과 공개 fit/forecast API 이관은 완료됐다. 다만 정확도
기준을 통과하지 않았으므로 Demand Engine production registry에는 연결하지
않고 비교 기준선으로 유지한다.

## 장기 Tail 보정 범위

두 모델 모두 Validation에서 `cohort_half_plus_tail_72`가 선택됐다.

- cohort strength: 0.5
- tail 시작: 미래 37번째 월
- tail half-life: 72개월
- scale gate: 사용하지 않음
- correction 범위: 0.20~1.50

Tail 보정은 독립적인 예측 모델이 아니라 **시간 순서를 지켜 학습한 사후
보정 계층**으로 사용한다. CGMM에서는 후보 곡선, 평균, 표준편차, 하한과
상한에 동일한 양수 계수를 적용해 분포 계약을 보존한다.

Weibull, Log-logistic, Piecewise Exponential은 현재 운영 경로에 추가하지
않는다. 이 방식들은 CGMM과 Similar Lifecycle이 모두 실패하는 분포 이탈
표본의 fallback 실험에서만 다시 검토한다.

## CVAE 도입 범위

CVAE는 현재 **실험 후보**로만 유지하고 바로 구현하거나 registry에 등록하지
않는다. 약 5천 개의 Train Lifecycle은 작은 모델의 사전검증에는 사용할 수
있지만, 이번 실패의 주원인은 모델 표현력보다 최근 cohort의 장기 수요 감소다.
CVAE 자체는 이 시간 변화 문제를 자동으로 해결하지 않는다.

도입 시 계약은 다음과 같다.

- 위치: `src/modeling_module/models/CVAE`
- 입력: 기존 `LifecycleSample`의 M0~M11과 동일 조건 특성
- 출력: 72개월 후보 곡선, 평균, 표준편차 또는 quantile
- 학습: 완료 Lifecycle만 사용하고 Validation/Test 미래값 누수 금지
- 용량: 작은 latent dimension부터 시작하고 seed 11/22/33 비교
- 저장: pickle 없는 checksum artifact와 strict load 동일성 보장

CVAE 구현 진입 조건은 다음과 같다.

1. 현재 열어본 Test를 재선택에 사용하지 않고 새로운 outer chronological
   fold를 확보한다.
2. 최소 3개 시간 fold 중 2개 이상에서 두 보정 기준선 대비 WAPE를 5% 이상
   개선한다.
3. 최악 fold WAPE가 기준선보다 5% 이상 악화되지 않는다.
4. 90% 구간 coverage가 0.85~0.95이고 interval score가 CGMM보다 개선된다.
5. seed별 WAPE 표준편차가 0.02 이하이며 artifact 복원이 완전히 동일하다.

## Wheel Release 기준

Demand Engine의 연구용 OOD·capacity·cohort 분석은 새 공개 API로 전환했고,
중복 전처리·artifact·보정 구현은 제거했다. 다음 release 검증은 아래 범위로
제한한다.

1. 저장소별 LTB 변경사항을 독립 커밋으로 고정한다.
2. 해당 `ts_forecaster_lib` commit에서 Wheel을 빌드한다.
3. 격리된 설치 환경에서 CGMM과 Similar Lifecycle artifact를 strict load하고
   공개 forecast 출력이 source 실행과 동일한지 확인한다.
4. Production registry 연결과 5090 배포는 별도 승인 전까지 진행하지 않는다.
