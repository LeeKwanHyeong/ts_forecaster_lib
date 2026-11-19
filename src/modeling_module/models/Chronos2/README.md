# Chronos-2 독립 zero-shot 어댑터

## 목적과 범위

공식 사전학습 가중치로 기본 52주 이력에서 27주 수요를 예측하는 독립 모델.
새 가중치 초기화나 fine-tuning으로 사전학습 모델을 대체하지 않는 구성.
기존 PatchTST, SELLM, TinyTimeMixer와 공통 학습·체크포인트 경로 변경 없음.

현재는 직접 import하는 연구·검증용 모델이며 기존 `train(models=[...])`, 모델 등록부,
`load_predictor`에 자동 등록하지 않은 상태. 운영 승격과 ReasonCast 학습은 별도 작업.

| 실행 방식 | 현재 연결 범위 |
|---|---|
| 공식 가중치 zero-shot | 구현 및 검증 완료 |
| Fine-tuning·추가 사전학습 | 공식 패키지의 학습 기능과 프로젝트 연결 미구현 |
| 새 가중치로 처음부터 사전학습 | 초기화·데이터 표집·학습·체크포인트 경로 미구현 |

`backbone.py`는 공식 구조를 재정의한 학습 모델이 아닌 동결 추론 어댑터.
공식 체크포인트의 최대 context는 8192이며 현재 기본 입력은 52주, 예측은 27주.
context 설정 변경만으로 가중치를 학습하거나 `2048 → 8192` 단계 학습을 수행하지 않는 동작.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `Chronos2.py` | 입력 검사와 예측 결과를 연결하는 PyTorch 인터페이스 |
| `backbone.py` | 공식 모델 로딩, 동결, 추론, 미래 패치 표현 추출 |
| `configs.py` | 고정 모델 revision, 해시, 장치, 길이, 분위수 설정 |
| `contracts.py` | 입력 shape·결측·외생변수 검증 및 결과 dataclass |
| `__init__.py` | 독립 패키지 공개 이름 |

```text
Chronos2Model.predict
  -> prepare_inputs: 과거 수요와 외생변수 검증
  -> Chronos2Backbone: 공식 고정 가중치 로딩
  -> Chronos2Pipeline.predict_quantiles
  -> Chronos2Forecast: P50 점 예측, P10/P50/P90, 실행 정보
```

## 공식 원본과 고정 버전

- 모델: `amazon/chronos-2`
- 모델 revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- `chronos-forecasting==2.3.2`, `transformers==4.57.6`
- 가중치 SHA-256: `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`
- 설정 SHA-256: `ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031`
- 공식 가중치 파일 크기: 477,930,472 bytes
- 코드·모델 라이선스: Apache-2.0

원본 링크:

- [공식 코드](https://github.com/amazon-science/chronos-forecasting)
- [고정 모델 snapshot](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c)
- [공식 패키지 2.3.2](https://pypi.org/project/chronos-forecasting/2.3.2/)

이 어댑터는 공식 소스 전체를 복사한 재구현이 아니라 버전을 고정한 공식 패키지의 사용 경계를 분리한 구성.
로컬 파일의 `declared_revision`은 지정값이며, 실제 일치 여부는 파일 SHA-256 검증 결과로 구분.
검증용이 아닌 다른 가중치로 교체하려면 별도 설정과 자격 검증 필요. 해시 검증을 해제한 실행은 공식 파일 검증 완료로 표시하지 않는 정책.

## 설치와 로딩

저장소 루트와 별도 Python 환경에서 선택 의존성 설치.

```bash
python -m pip install -e '.[chronos2,dev]'
```

라이브러리 import와 모델 생성만으로 다운로드하지 않는 동작.
기본 `local_files_only=True`이므로 사전 준비한 파일이나 캐시가 없으면 오류 발생.
온라인 가중치 확보는 `local_files_only=False` 또는 실행기의 `--allow-download`로 명시하는 별도 동작.
Hub 접근 시 익명 읽기만 사용하고 remote code 실행은 허용하지 않는 구성.

```python
import torch
from modeling_module.models.Chronos2 import Chronos2Config, Chronos2Model

cfg = Chronos2Config(
    model_path="artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c",
    context_length=52,
    horizon=27,
    device="cpu",
)
model = Chronos2Model(cfg)

# 실제 사용 시 아래 예시를 과거 관측 수요로 교체.
x = torch.ones(2, 52, 1)
forecast = model.predict(x)
assert forecast.point.shape == (2, 27, 1)
assert forecast.quantiles.shape == (2, 27, 3)

# forward는 점 예측 tensor만 반환.
point = model(x)
```

## 입력과 출력 계약

- 수요 입력: `(B,L,1)`의 단일 목표 채널. 이번 어댑터의 다변량 목표 예측 미지원
- L이 context_length보다 길면 가장 최근 구간만 사용
- NaN은 미관측 상태로 전달하며 0 수요로 변환하지 않는 정책
- 최근 context 전체가 결측인 부품, Inf, bool·복소수·문자열 입력 거부
- 원래 입력 배열을 수정하지 않고 CPU FP32 복사본으로 공식 전처리에 전달
- 과거 외생변수: 이름별 `(B,L)`의 연속형 수치 배열
- 미래 외생변수: 같은 이름의 과거 값과 `(B,H)`의 유한한 known-future 값 필요
- 범주형 문자열·정적 특성의 별도 인코딩 미지원. 범주 코드를 수치로 넘긴다고 범주형 인코딩이 되는 것은 아닌 범위
- 공식 batch_size는 수요·외생변수 채널 수를 합한 기준이며, 한 부품의 모든 채널을 담을 수 있는 크기 필요
- `cross_learning=False` 고정으로 서로 다른 부품 간 정보 공유 차단
- 반환값은 CPU FP32이며 점 예측은 P50, 즉 중앙값
- 버전 2.3.2 공식 API의 두 번째 반환값 이름이 mean이어도 내부 값은 median인 점에 주의. 어댑터는 명시적으로 0.5 분위수 선택
- 음수 예측 절단이나 분위수 재정렬을 자동 수행하지 않고 공식 원 출력 유지
- 분위수 간격은 불확실성의 한 지표이며 예측 엔트로피와 동일한 값이 아닌 구분

```python
past = {"sin_annual": torch.ones(2, 52)}
future = {"sin_annual": torch.ones(2, 27)}
forecast = model.predict(x, past_covariates=past, future_covariates=future)
```

공식 모델은 eval 상태와 `requires_grad=False`로 유지.
상위 모델의 `train()` 호출로 backbone의 dropout이나 학습 가능 상태를 켜지 않는 경로.
CUDA를 요청했는데 사용할 수 없으면 CPU로 조용히 바꾸지 않고 오류 발생.

## 미래 표현 관찰

```python
features = model.encode_future_tokens(x)
assert features.future_tokens.shape == (2, 2, 768)
```

공식 output patch 크기가 16이므로 27주는 `ceil(27/16)=2`개 미래 패치로 표현.
현재 메서드는 외생변수 없는 입력의 추론·관찰 전용이며, 미래 토큰과 역정규화용 위치·스케일을 반환.
이 표현은 실제 출력 head 직전 값인지 별도 실물 테스트로 검증하는 대상.
ReasonCast의 학습 가능한 보정 경로와 외생변수를 포함한 미래 표현 연결은 아직 구현하지 않은 범위.

## 실제 27주 실행

저장소 루트에서 합성 계절·추세·간헐 시계열 3개로 실행.
각각 미래 27주 정답은 평가용으로만 분리하며 모델 입력에는 미포함.

```bash
PYTHONPATH=src python -m model_test.chronos2.run_zero_shot \
  --model-path artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c \
  --output-dir artifacts/chronos2/my_zero_shot_run \
  --with-calendar
```

아직 캐시가 없다면 `--model-path` 대신 `--allow-download --cache-dir artifacts/chronos2/hf_cache` 지정.
기존 출력 폴더를 덮어쓰지 않으므로 재실행마다 새 경로 필요.

로컬 주간 parquet 점검은 다음과 같이 입력. 조회한 데이터를 DB로 적재하는 기능은 없는 범위.

```bash
PYTHONPATH=src python -m model_test.chronos2.run_zero_shot \
  --model-path artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c \
  --source-parquet /path/to/tb_master_target.parquet \
  --plan-week 202518 --max-parts 8 \
  --output-dir artifacts/chronos2/my_local_sample_run \
  --with-calendar
```

- parquet 컬럼: `oper_part_no`, `demand_dt`(ISO YYYYWW), `demand_qty`
- 과거 52개 실제 주차가 연속이고 관측 수요가 있는 부품을 과거 이력만으로 선택
- 최대 16개 부품의 점검만 허용하는 제한 실행
- 미래 실적은 없으면 NaN으로 유지하고 평가에서 미관측으로 분리
- 예측 주차는 W0~W26의 총 27개이며 연말·53주도 ISO 날짜 연산 사용
- 로컬 스냅샷의 운영/합성 출처, 공식 사전학습 데이터와의 비중복 여부는 별도 확인 대상

생성 파일:

- `forecast_27_weeks.csv`: 부품·주차별 P50, P10/P50/P90, 비교용 실적
- `arrays.npz`: 실제 context, 미래 정답, 원 예측 배열
- `summary.json`: 공식 파일 해시, 입력 hash, 예측 shape, 지표, 실행 시간, 버전

## 유형 균등 다중 시점 비교

위 `run_zero_shot`은 최대 16개의 실행 점검용 경로이며 성능 대표 표본이 아닌 구분.
유형별 균등 표본과 여러 시작 주차의 H27 비교는 아래 독립 실행기 사용.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m model_test.chronos2.run_balanced_backtest \
  --source-parquet /path/to/tb_master_target.parquet \
  --model-path artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c \
  --output-dir artifacts/chronos2/my_balanced_h27 \
  --per-pattern 50 --seed 42 --bootstrap-repetitions 2000

PYTHONPATH=src python -m model_test.chronos2.audit_balanced_backtest \
  --output-dir artifacts/chronos2/my_balanced_h27
```

- 과거 52주 ADI·CV² 기준의 안정형·고변동형·간헐형·간헐 고변동형 표본
- 장기 8개 시점의 2유형 패널과 최근 4유형 패널을 별도로 구성
- 해당 시점의 최소 후보 수에 맞춰 유형당 표본 축소, 특정 유형 부재 시 해당 시점 전체 제외
- seed 고정 무작위 비복원 추출, 미래 정답 확인 전 cohort manifest 저장
- 같은 입력과 실적으로 Chronos2 P50/마지막 관측값 Naive를 평가, 실제 정답 누락 시 오류
- pooled/macro WAPE와 MAE·RMSE·부품별 승패를 병기, 무수요·음수 예측도 숨기지 않는 평가
- `balanced_sampling.py`: 과거 이력 품질 검사·유형 분류·표본 선정
- `backtest_metrics.py`: H27 오차·동일 가중 지표·부품 단위 bootstrap
- `run_balanced_backtest.py`: 공식 모델 실행·입력/결과/설정 증적 저장
- `audit_balanced_backtest.py`: 원천·배열·CSV를 독립 계산으로 대조

시점은 `balanced_sampling.DEFAULT_PANELS`에서 명시적으로 관리하는 실험 설정.
다른 데이터 기간에서는 실행 전 해당 시점을 검토하고 새 결과 경로를 사용해야 하는 조건.
실제 확대 비교 결과 및 한계는 `docs/Chronos2BalancedBacktest20261119.md` 참고.

## 테스트

모의 모델로 입력·오류 경로를 확인하는 테스트와 공식 가중치 실물 테스트를 구분.
실물 테스트는 아래 환경변수가 없으면 다운로드 없이 명시적으로 건너뛰는 동작.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
TSF_CHRONOS2_MODEL_DIR="$PWD/artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c" \
python -m pytest -q tests/test_chronos2_contract.py \
  tests/test_chronos2_official_weights.py tests/test_chronos2_runner.py
```

실제 점검 결과와 제한사항은 `docs/Chronos2ZeroShotValidation.md`에 기록.
