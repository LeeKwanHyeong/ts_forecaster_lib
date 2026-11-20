# 동결 Chronos2 + MORA 실제 데이터 비교

## 결과

202518 시작, MORA 적합에 쓰지 않은 부품 1,050개 × 27주 = 28,350개 정답의 동일 조건 비교.

MAE·WAPE는 소폭 악화, RMSE는 미세 개선된 결과로, 이번 조건에서 기본 Chronos2를 MORA 경로로 교체할 성능 근거는 확인되지 않은 상태.

| 지표 | 공식 Chronos2 P50 | Chronos2 + MORA 점예측 |
|---|---:|---:|
| MAE | 2.215364 | 2.216366 |
| RMSE | 5.606588 | 5.604311 |
| WAPE | 11.5083% | 11.5135% |
| 부품별 RMSE 평균 | 2.686577 | 2.687452 |
| 음수 점예측 수 | 6393 | 6892 |
| 27주 전체 0수요 부품의 MAE | 2.394199 | 2.395882 |

MAE 기준 0.0452% 증가. 부품별 개선 517개·동일 0개·악화 533개.
부품 단위 paired bootstrap 2,000회에서 `보정 MAE - 기준 MAE` 평균 차이의 95% percentile 구간: [0.000253, 0.001697].
한 시작 주차의 부품 표본 불확실성에 대한 구간이며 시점 변화에 대한 신뢰구간은 아님.
시험 부품 중 657개(62.57%)는 27주 실제 수요 합이 0인 부품. 개별 WAPE의 분모가 0인 집단은 WAPE 평균에서 제외하고 MAE를 별도 제시한 구성.

## 비교 조건

- 대상 저장소·브랜치: `ts_forecaster_lib` / `ts_forecaster_lib_v2`
- 원천: `raw_data/master/tb_master_target.parquet`, 2,455,508행·7,000부품·201801–202544, 사용자가 실제 관측된 0수요로 확인한 로컬 스냅샷
- 기존 부품·기간 분할 그대로 유지: 학습 4,900개/201801–202442, 검증 1,050개/202443–202517, 시험 1,050개/202518–202544
- 모든 배정 부품에서 L52/H27 조건 충족, 제외 0개
- 학습 가능한 1,080,909개 창 중 ID·주차·seed 42만으로 부품당 4개, 총 19,600개 선택. 수요량이나 예측 성능을 기준으로 선택하지 않은 제한 표집
- 검증·시험 각 1,050개 창 전부 포함. 정답은 검증에서 보정 강도 결정, 시험에서 최종 평가 용도로만 사용
- 공식 `amazon/chronos-2` 고정 revision, CPU FP32, 입력 52주, 출력 27주, 별도 외생변수 없음, cross-learning 비활성
- 양쪽에 동일한 원본 P50 사용. 반올림·음수 절단·부품별 사후 fallback 없음
- MORA만 적합. 학습 시 outer/inner 4블록의 같은 부품 raw 구간 중첩 제거
- `period=52`, `rank=16`, `ridge_strength=10`, 나머지 기본 설정 고정. 시험 결과 기반 설정 탐색 없음

### MORA 적용 범위

논문 수식의 독립 구현으로 저자 코드를 복사하지 않은 구성. `pooled_series`는 MORA 적합에 쓰지 않은 부품에도 공통 보정 규칙을 적용하는 명시적 연구 확장.
부품마다 수요 채널 1개이며 수천 부품의 동시 다채널 입력 방식이 아님.
검증에는 하나의 H27 시작점만 있으므로 강도 결정의 4개 그룹은 시간 블록이 아닌 부품 그룹.
따라서 논문 원본 실험을 그대로 재현한 결과로 설명하지 않을 조건.

## 불변·누수 방지 확인

- 공식 가중치 파일·설정·메모리 파라미터·모든 기존 모델 Python 소스의 전후 동일성 확인
- Chronos2 학습 가능 파라미터 0개, 평가 모드 유지
- 기존 분할 manifest와 원천 parquet 해시 유지
- 보정기 저장 완료 시각과 해시 기록 후 시험 정답 로드·평가
- 시험 평가 중 보정기 해시 불변, 저장·복원 후 점예측 완전 일치
- 첫 예측 주차는 MORA 기저 구조상 원본 값 그대로 유지
- CSV 독립 오차 합산, 배열·원본 P50 대조, 실제 parquet 정답 28,350개 및 달력 전체 대조 통과

## 제한과 판단 기준

단일 시작 주차이므로 다른 시점·법인에서도 같은 개선을 보장하지 않는 결과.
공식 Chronos2 사전학습 데이터와의 중복 여부는 미확인. 부품 분리는 이번 MORA 적합·검증·시험 데이터 내부에 대한 계약.
전체 오차와 부품별 개선·악화, 음수 예측을 함께 판단할 필요.
MORA 출력은 보정된 점예측이며 원본 P10/P50/P90 분위수의 보정 또는 예측구간 개선으로 해석 불가.
이번 시험은 이미 사용된 holdout이므로 이후 설정 탐색의 근거로 사용하면 새 최종 시험 구간 필요.

## 파일과 재현

- 결과: `artifacts/mora/chronos2_frozen_l52_h27_20261120_seed42`
- 셀별 설명: `src/modeling_module/forecast_adapters/MORA/notebook/mora_chronos2_real_comparison.ipynb`
- `experiment.json`: 수요량 확인 전 고정한 계획·분할·모델 소스 해시
- `window_manifest.csv`, `eligibility.json`, `excluded_parts.csv`: 창 선택·완전 관측 여부
- `train_arrays.npz`, `validation_arrays.npz`, `test_arrays.npz`: 원본 입력·정답·P50·분위수
- `adapter.npz`, `fit_complete_before_test.json`: 보정 상태와 시험 전 고정 증적
- `test_forecast.csv`, `test_comparison.npz`: 전체 짝지은 시험 결과
- `summary.json`, `independent_audit.json`, `artifact_hashes.json`: 지표·독립 검산·무결성

```bash
# 기존 결과를 덮어쓰지 않도록 새 출력 경로 지정 필요
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m model_test.mora.run_chronos2_comparison --output-dir artifacts/mora/NEW_RUN

# 저장된 결과 독립 검산과 그림 재생성
PYTHONPATH=src python -m model_test.mora.audit_chronos2_comparison --output-dir artifacts/mora/chronos2_frozen_l52_h27_20261120_seed42
```

실행 환경: chronos-forecasting 2.3.2, torch 2.9.1, NumPy 2.1.3, SciPy 1.15.2.
실험 소요 시간 354.8초, MORA 적합 3.6초, 시험 1,050창 보정 0.0038초.
최종 전체 회귀 테스트 1,100 passed / CUDA 6 skipped / 46 subtests passed. 설명 노트북 코드 셀 8개 전체 실행과 저장·복원 점예측 일치 확인.

### 식별 해시

- 원천 SHA-256: `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547`
- 분할 manifest SHA-256: `a7b16441549e128297797be87d1c0853690a89bc1dc50baa112ca5883fc72238`
- 공식 가중치 SHA-256: `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`
- 보정기 SHA-256: `49eb9898e6bfe40b60e3d739d05bfdc93fc197f31c5cf39160f6c89b94a2307b`

## 남은 작업

- **다음 작업**: 새 기간 또는 미사용 데이터로 다중 시작 주차 검증. 이미 확인한 시험 결과에 맞춘 재튜닝과 구분 필요
- **승인 필요**: 운영 적용·공용 Runtime 변경. 이번 비교에서는 미수행
- 커밋·Push 미수행. 기존 개발 변경·IDE 변경·stash 보존
