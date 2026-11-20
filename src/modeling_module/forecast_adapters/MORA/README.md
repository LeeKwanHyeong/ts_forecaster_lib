# MORA 출력 보정기

## 목적

기존 모델이 만든 점예측에 과거 이력으로 추정한 보정량을 더하는 독립 모듈.
기준 모델의 소스·가중치·공통 Trainer·모델 등록부는 변경하지 않는 구조.
논문 수식을 직접 구현한 버전이며 저자 저장소 코드는 복사하지 않은 상태.
출처·구현 선택·미포함 기능은 [SOURCE.md](SOURCE.md) 참고.

```text
과거 이력 + 기준 점예측
  -> 미래 구간 평균 추정
  -> 외삽 기반 1단계 보정
  -> 학습 평균 기반 2단계 보정
  -> 검증에서 정한 scale 적용
  -> 원본 점예측 + 보정량
```

## 구조

- `../contracts.py`: `ForecastQuery`, 적합용 `ForecastFitData`, `AdaptedForecast`
- `configs.py`: L/H·구간·rank·기간·정규화·OOF·scope 설정
- `adapter.py`: `MORAAdapter.fit/predict/save/load`
- `fitting.py`: raw 구간 중첩 제거, nested OOF, 적합 상태
- `native/`: NumPy·SciPy 기반의 실제 보정 연산, 주요 설명은 한글 주석 포함
- `checkpoints.py`: pickle 없는 NPZ, JSON metadata, SHA-256, shape·기저·gain 검사
- `notebook/`: 셀별 설명과 실행 결과

실행 결과가 포함된 [설명 노트북](notebook/mora_output_residual_adapter_walkthrough.ipynb) 제공.

외부 `mora` 패키지나 LLM·학습 모델 체크포인트는 필요하지 않는 구성.
NumPy·SciPy는 프로젝트 기본 의존성 사용. 프로젝트 자체의 기본 PyTorch 의존성은 변경하지 않은 상태.

## 데이터 계약

| 필드 | 의미 |
|---|---|
| `history` | `(N,L,C)`의 과거 관측 |
| `backbone_prediction` | `(N,H,C)`의 기준 점예측 |
| `origins` | 첫 예측 위치의 연속 정수 인덱스. ISO YYYYWW를 직접 연산하지 않는 계약 |
| `channel_names` | 고정된 C개 채널의 이름과 순서 |
| `series_ids` | 각 query의 시계열·부품 식별값 N개 |
| `backbone_id` | 모델·체크포인트를 식별하는 이름 또는 해시 |
| `scale_id` | 원 단위 또는 고정 정규화 방식을 식별하는 값 |

학습·검증·추론의 기준 모델 ID, 단위, 채널 순서, 길이가 다르면 오류.
모든 배열은 복사해 호출자의 입력을 변경하지 않는 계약.
0은 관측값이며 결측은 별도 전처리가 필요한 오류. 음수 자체는 온도 등 일반 시계열을 고려해 허용.
결과를 수요 0 이상으로 절단하는 업무 규칙은 이 구현에 자동 삽입하지 않는 기준.

기본 `single_series` 보정기를 다른 부품에 재사용하면 오류.
여러 부품을 공용 보정기로 적합하려면 `scope="pooled_series"`를 명시하고 채널의 업무상 의미를 동일하게 맞출 필요.
이는 연구 확장이며 어댑터 자체가 기존 Chronos2의 부품 분할을 자동으로 이관하지 않는 구조.
분할 복원과 실제 모델 호출은 아래 별도 실험 실행기에 구성.

## 적합과 누수 경계

- `ForecastFitData`만 정답을 보유하며 `predict(query)`에는 정답 인자가 없는 구조.
- 기준 예측 출처는 `frozen_pretrained`, `synthetic_history_only`, `blocked_oof` 중 하나로 선언.
- 학습형 기준 모델은 `blocked_oof` manifest SHA-256 필요. 선언은 생산자가 제공하는 증적이며 이 모듈이 원본 모델의 OOF 학습을 실행하거나 사실성을 인증하는 기능과 구분.
- 학습과 검증의 정답 기간은 중복 금지. 검증의 과거 입력이 학습 구간으로 들어가는 것은 허용.
- held 블록의 `[origin-L, origin+H-1]`과 겹치는 창은 같은 시계열의 OOF 학습에서 제거.
- 2단계 reliability 추정은 nested OOF 사용. 정규화·coarse estimator·매핑·gain은 outer held 정답 없이 적합.
- 최종 scale만 검증 정답으로 적합하며 테스트 정답은 적합에 미사용.
- 추론에는 calibration 정답 종료 이후의 query만 허용. 실제 재학습 계획과 별개로 파라미터는 고정.

## 최소 예제

저장소 루트에서 `PYTHONPATH=src`로 실행.

```python
from modeling_module.forecast_adapters.MORA import MORAAdapter, MORAConfig
from model_test.mora.synthetic import make_synthetic

config = MORAConfig(period=13, correction_rank=12)
train, validation, test = make_synthetic(config)
adapter = MORAAdapter(config).fit(train, validation)
result = adapter.predict(test.query)
assert result.point.shape == (24, 27, 2)
assert (result.point[:, 0] == result.backbone_point[:, 0]).all()

# 기존 파일을 덮어쓰지 않으므로 새 경로 사용.
adapter.save("artifacts/mora/my_run/adapter.npz")
restored = MORAAdapter.load("artifacts/mora/my_run/adapter.npz")
```

기존 Chronos2 연결 시 `history`와 `forecast.point`를 동일 길이·채널·단위의 NumPy 배열로 준비.
Torch tensor라면 호출자에서 `detach().cpu().numpy()`로 변환하고 원본 `Chronos2Forecast` 객체는 유지.
이 모듈에 모델 객체를 전달하지 않으므로 기준 모델의 가중치를 갱신하는 경로는 없음.

반환 `point`는 **보정 점예측**이며 원래 P50의 분위수 성질까지 검증한 결과는 아님.
원본 P10/P90과 보정 점예측을 원래와 동일한 확률 예측 묶음으로 바꿔 끼우지 않는 기준.
`quantiles_calibrated=False`를 metadata에 표시.
첫 예측 위치는 원본을 유지하며 마지막 관측값으로 강제로 맞추는 동작과 구분.

## 검증과 실행

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest -q tests/test_mora_contracts.py \
  tests/test_mora_numerics.py tests/test_mora_adapter.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m model_test.mora.run_synthetic \
  --output-dir artifacts/mora/my_synthetic_h27 --seed 42
```

합성 실험 출력: `adapter.npz`, `evaluation_arrays.npz`, `forecast.csv`, `summary.json`.
과거 평균에 의도적인 과소 예측을 넣은 baseline으로 계산·적합·재로딩을 점검하는 실험.
실제 부품 성능이나 논문 benchmark 개선율의 재현으로 해석하지 않는 조건.

저장 파일은 완성 후 공개하며 기존 파일 덮어쓰기 금지. `allow_pickle=False` 사용.
학습 상태는 C-order FP64로 고정해 현재 환경에서 저장 전후 예측 일치를 확인하는 방식.
SHA-256은 무결성 검사이며 공격자 인증·전자서명은 아님. 신뢰할 수 있는 로컬 산출물에 사용.

## 현재 검증과 다음 단계

2026-11-20 신규 테스트 53개 통과. 기존 모델을 포함한 전체 회귀는 1,090 passed / CUDA 6 skipped / 46 subtests passed.
Chronos·LLM 선택 의존성 없는 환경은 1,070 passed / 26 skipped / 46 subtests passed.
노트북 코드 셀 15개 전체 실행과 별도 wheel 설치본의 H27 재로딩 예측 일치 확인.

### 동결 Chronos2 실제 데이터 비교

후속 승인에 따라 기존 부품 70:15:15 분할과 L52/H27을 유지한 로컬 스냅샷 비교 완료.
학습 4,900부품·19,600창, 검증 1,050부품·1,050창, 시험 1,050부품·1,050창, 제외 0부품.
원본 P50은 보존하고 MORA만 `pooled_series`로 적합한 구성. 기존 모델 소스·파일 가중치·메모리 파라미터 불변 확인.

시험 MAE `2.215364 → 2.216366`으로 0.0452% 악화, RMSE `5.606588 → 5.604311`로 0.0406% 개선.
부품별 MAE 개선 517개·악화 533개로, 이번 조건에서 일관된 성능 개선은 확인되지 않은 결과.
전체 예측 28,350개의 CSV·배열·원천 정답·달력 독립 대조 통과.
최종 전체 회귀 1,100 passed / CUDA 6 skipped / 46 subtests passed. 실제 비교 노트북 코드 셀 8개 전체 실행 확인.

- [실제 비교 보고서](../../../../docs/Chronos2MORARealComparison20261120.md)
- [실행 결과·셀별 설명 노트북](notebook/mora_chronos2_real_comparison.ipynb)
- `src/model_test/mora/real_data.py`: 수요량 독립 표본 선택·달력·완전 관측 창 검사
- `src/model_test/mora/run_chronos2_comparison.py`: 원본 동결·MORA 적합·시험 순차 실행
- `src/model_test/mora/audit_chronos2_comparison.py`: 독립 검산·진단 그림

검증·시험 시작점은 각각 하나뿐이므로 검증 그룹은 시간 블록이 아닌 부품 그룹.
논문 원본 실험 재현이나 다중 시점 운영 성능 보장과는 구분되는 결과.
서비스 등록·공용 Runtime 배포·커밋·Push는 미수행. 다음 단계는 새 미사용 기간의 다중 시작 주차 검증.
