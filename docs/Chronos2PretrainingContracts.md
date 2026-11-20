# Chronos2 자체 사전학습: 데이터 계약과 학습용 backbone

작성일: 2026-11-19
대상: `ts_forecaster_lib` 저장소 / `ts_forecaster_lib_v2` 브랜치
구현 범위: 계획의 1단계 데이터·평가 계약과 2단계 새 가중치 backbone

## 1. 현재 상태

공식 Chronos-2 연산 코드는 재사용하되 가중치는 새로 초기화하는 독립 경로.
현재 학습용 연산은 저장소 내부 `Chronos2/native/`에 포함한 공식 원본 소스 사용.
기존 공식 가중치 zero-shot의 `Chronos2.py`, `backbone.py`, `configs.py`, `contracts.py`는 변경 없음.
공통 `train()`·모델 등록부·기존 체크포인트 로더에 자동 연결하지 않은 상태.

| 항목 | 상태와 확인 범위 |
|---|---|
| 0수요 의미 | 사용자 확인 완료: 모두 실제 관측된 무수요 |
| 부품·정답 기간 분할 | 사용자 승인 기준으로 manifest 생성 및 독립 대조 완료 |
| 무작위 가중치 초기화 | 공식 생성자 사용, 체크포인트·Hub 접근 없이 생성 확인 |
| 학습 forward | 공식 미래 분위수 손실, 합성 데이터 역전파·1회 optimizer 갱신 확인 |
| 실제 수량 품질·유효 시간 창 | 다음 데이터셋 단계에서 검사할 항목 |
| 데이터 표집·trainer·저장·재시작 | 미구현 |
| 실데이터 학습·5090 실행·성능 비교 | 미실행 |

원천의 `provenance=user_provided`는 사용자 제공 데이터라는 선언.
운영 원천의 계보를 독립 확인했거나 공식 사전학습 데이터와 비중복임을 증명한 의미는 아닌 구분.

## 2. 데이터와 평가 계약

### 2.1 식별자와 0수요

- 원천: `raw_data/master/tb_master_target.parquet`
- 기본 식별자: 문자열 `oper_part_no`. 회사·법인·사이트 추가 시 `SeriesKey` 복합키 사용 가능
- 주차: `demand_dt`, 유효한 ISO `YYYYWW` 정수
- 목표: `demand_qty`, 관측된 유한한 0 이상 수량
- 관측 수요 0: 학습·손실·평가에 포함하는 실제 값
- NaN 또는 명시적 미관측 마스크: 0으로 대체하지 않고 손실에서 제외
- 중복 부품·주차: 계약 준비 시 오류. 주차 누락과 수량 이상은 후속 창 구성 단계에서 검사

새 원천에 대한 기본 선언은 `unverified`·`unconfirmed`.
실데이터 loader는 후속 구현에서 `assert_semantics_confirmed()`와 수량·창 품질 검사를 모두 통과한 뒤 배치를 구성해야 하는 계약.
현재 backbone은 이미 구성한 텐서를 받으므로 원천 파일의 계보나 승인 여부까지 판별하는 실행기가 아닌 범위.

### 2.2 승인된 분할 기준

부품 배정과 정답 기간을 모두 분리하는 엄격한 평가 설계.
새 부품에 대한 일반화와 미래 시점 예측을 함께 확인하려는 기준이며, 부품군 단위 분리 검증은 메타데이터 확보 후 별도 대상.

| 분할 | 부품 수 | 비율 | 허용 정답 주차 |
|---|---:|---:|---|
| 학습 | 4,900 | 70% | 201801~202442 |
| 검증 | 1,050 | 15% | 202443~202517, 27주 |
| 최종 시험 | 1,050 | 15% | 202518~202544, 27주 |

- seed: `42`. 부품 ID와 역할명을 함께 해시한 순위로 배정하며 수요량·패턴·오차는 미사용
- 검증·시험 수는 각각 전체의 15%를 내림하고 나머지를 학습에 배정
- 기존 Chronos2 확대 비교·8개 표본 점검·PatchTST loss 비교의 합집합 1,177개 부품을 새 시험 후보에서 제외
- 시험 제외 부품은 학습·검증에는 배정 가능. 세 분할 사이 ID 중복과 이전 평가 부품의 시험군 유입은 0건
- 정답은 시작 주차 포함 W0~W26. 마지막 정답까지 해당 분할의 cutoff 안에 있어야 하는 조건
- 입력 context는 origin 직전 52주이며 자기 분할의 정답 기간보다 앞설 수 있는 구분
- 검증·시험 정답 구간은 각각 정확히 27주이므로 현재 계약의 완전한 H27 origin은 각 1개. 다중 origin 최종 시험은 계약을 먼저 확장할 별도 작업
- 표의 수는 배정된 부품 수. 실제로 L52/H27 창을 구성할 수 있는 유효 부품·배치 수는 아직 미산정

분할 이후 정규화와 패턴 분류에는 각 과거 context만 사용하도록 계약에 기록.
현재는 수량값을 읽어 이 정책을 적용하는 sampler까지 구현한 상태가 아닌 구분.

### 2.3 현재 스냅샷의 증적

원천 2,455,508행 / 7,000개 부품 / 201801~202544.
원천 schema와 식별자·주차를 점검하되 수요량은 값으로 materialize하지 않은 실행.
파일 SHA-256 계산은 바이너리 무결성 확인이며 수요량을 분할 기준으로 사용하는 작업과 별개.

```text
artifacts/chronos2/pretraining_contract_20261119_seed42/
  source_contract.json  # 원천 해시·schema·0수요 선언
  split_manifest.json   # 전체 부품 배정·cutoff·이전 평가 제외 목록
  summary.json          # 배정 집계와 미완료 항목
```

원천 SHA-256:
`328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547`

분할 manifest SHA-256:
`a7b16441549e128297797be87d1c0853690a89bc1dc50baa112ca5883fc72238`

원천 해시 재확인, 7,000개 ID 전체 배정, 분할 간 교집합, 기존 평가와 시험군의 교집합,
ISO 날짜 기준 두 평가 구간 길이, manifest 해시를 별도 계산으로 대조한 결과 모두 일치.
원천 수정·DB 적재·실제 수요 학습은 없는 실행.

## 3. 코드 책임과 학습 동작

| 파일 | 책임 |
|---|---|
| `Chronos2/training/configs.py` | 새 모델 크기·길이·분위수·seed 설정 |
| `Chronos2/training/contracts.py` | 원천 의미·복합 식별자·분할·정답 cutoff·manifest |
| `Chronos2/training/batch.py` | 텐서·마스크·독립 group 검사, 원 단위 출력 계약 |
| `Chronos2/trainable_backbone.py` | 공식 생성자로 무작위 초기화, 학습 가능한 native forward |
| `Chronos2/native/` | 실제 모델·Attention·정규화·패치·손실 연산과 원본 출처·라이선스 |
| `model_test/chronos2/prepare_pretraining_contract.py` | 실제 수량 학습 없이 ID·주차로 계약 파일 생성 |

모델 경로의 기준 디렉토리는 `src/modeling_module/models/`, 실험 실행기의 기준은 `src/`.

### 3.1 생성과 가중치 출처

고정 원본: 공식 `v2.3.2`, 커밋 `0aba28e9360b6355d5a919596c90b16c25939f02`.
학습용 선택 의존성: `chronos2-native`. 외부 `chronos-forecasting`은 불필요하며
`transformers==4.57.6`, PyTorch·einops·packaging은 계속 사용하는 기반 의존성.
내부 `native.config.Chronos2CoreConfig`와 `native.model.Chronos2Model` 생성자 직접 호출.
`from_pretrained()`, 기존 `state_dict` 복사, 공식 모델 경로, Hub 다운로드는 미사용.

기본 구조는 `d_model=128`, `d_ff=256`, `num_layers=2`, `num_heads=4`, patch 16.
CPU 계약 검증용 소형 구조이며 공식 대형 모델과 같은 크기이거나 최적 모델이라는 의미는 아님.
생성 시 초기 파라미터 해시·구조 해시·seed·패키지 버전을 추적하며 원래 CPU 난수 상태는 복원.
재현성은 같은 설정과 검증 환경을 기준으로 확인한 범위.

`native/`는 공식 코드의 내부 관리본이며 새로 작성한 독자 모델로 표시하지 않는 원칙.
한글 주석과 정규화·패치 import 변경 외 연산 AST 보존.
LICENSE·NOTICE와 원본/내부 파일 해시를 `native/upstream_manifest.json`에 기록.
기존 zero-shot은 외부 패키지 경로를 유지하며 공식 Pipeline·dataset·trainer 전체를 내부화한 상태는 아닌 구분.

### 3.2 입력과 손실

- `context`: `(N,52)`, `future_target`: `(N,27)` 실수 Tensor
- 마스크: 같은 shape의 bool Tensor. True가 실제 관측이며 미지정 시 NaN으로 판별
- 각 행에 context와 정답이 각각 하나 이상 관측돼 있어야 하는 조건
- Inf·FP32 범위 초과·관측된 음수·관측된 NaN·빈 정답 마스크는 오류
- 같은 부품 작업의 외생변수 그룹은 아직 미지원. 현재는 서로 다른 group ID를 가진 독립 단변량 행만 허용
- native 정규화가 관측 마스크 적용보다 먼저 수행되는 점을 고려해, 숨긴 값 자체를 NaN으로 치환한 뒤 전달
- 학습 목표는 미래 수요의 분위수 예측. 과거 복원 MSE나 원 단위 MAE가 아닌 공식 masked quantile loss
- patch 16의 2개 출력으로 32개 위치 생성. 남는 5개는 손실 마스크 0, 반환은 정확히 27주
- native 손실은 마스크를 적용한 뒤 패딩을 포함한 출력 길이로 평균하는 구조. 유효 정답 개수로 다시 나눈 손실로 바꾸지 않은 구현
- 기본 분위수 21개, P50 점 예측 `(N,27,1)`, 전체 분위수 `(N,27,21)`
- 반환 예측은 원 수량 단위이며 음수 절단·분위수 재정렬은 미적용

공식 초기화와 손실 구현 근거:
[Chronos-2 v2.3.2 model.py](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.2/src/chronos/chronos2/model.py).

### 3.3 최소 합성 배치 예제

저장소 루트에서 `PYTHONPATH=src`로 실행하는 예제.
실제 parquet loader나 지속 학습 trainer를 제공하는 예제는 아닌 구분.

```python
import torch
from modeling_module.models.Chronos2.trainable_backbone import Chronos2TrainableBackbone
from modeling_module.models.Chronos2.training import Chronos2TrainableConfig, Chronos2TrainingBatch

model = Chronos2TrainableBackbone(Chronos2TrainableConfig())
batch = Chronos2TrainingBatch(
    context=torch.linspace(3, 20, 52).repeat(2, 1),
    future_target=torch.linspace(22, 30, 27).repeat(2, 1),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()
result = model(batch)
result.loss.backward()
optimizer.step()
assert result.point.shape == (2, 27, 1)
print(result.loss.item(), model.provenance())
```

## 4. 계약 준비와 검증

현재 승인된 원천으로 새 계약을 준비하는 명령. 이미 존재하는 출력 폴더는 덮어쓰지 않는 정책.
이 명령에는 학습이나 가중치 갱신 동작 없음.

```bash
PYTHONPATH=src python -m model_test.chronos2.prepare_pretraining_contract \
  --source-parquet raw_data/master/tb_master_target.parquet \
  --output-dir artifacts/chronos2/my_pretraining_contract \
  --zero-semantics observed_zero --provenance user_provided \
  --exclude-evaluated-cohort artifacts/chronos2/balanced_h27_20261119_seed42/cohort_manifest.csv \
  --exclude-evaluated-cohort src/model_test/patchtst_loss_comparison/results/data/cohort.csv \
  --exclude-smoke-summary artifacts/chronos2/local_snapshot_zero_shot_27_20261118_verified/summary.json \
  --seed 42 --context-length 52 --horizon 27
```

신규 계약·backbone 테스트:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pytest -q tests/test_chronos2_pretraining_contracts.py \
  tests/test_chronos2_pretraining_backbone.py
```

2026-11-19 초기 1·2단계 검증 기록:

- 신규 테스트 55개 통과: ID 전용 배정, ISO 53주, cutoff·마스크·손실·초기화·합성 역전파 포함
- 공식 가중치 로딩과 Hub 함수를 차단한 상태에서도 새 모델 생성 성공
- 같은 seed 초기화 일치, 다른 seed 가중치 차이, 합성 AdamW 1회 갱신 확인
- 숨긴 context 값 변경·미래 정답 변경·다른 독립 행 변경이 해당 예측에 유입되지 않는 검사 통과
- 공식 가중치 실물 검사를 포함한 전체 회귀: 1,028 passed / 6 CUDA skipped / 46 subtests passed
- Chronos 선택 의존성 없는 환경: 1,016 passed / 18 skipped / 46 subtests passed
- 기존 nested tensor·MPS pin-memory·Titan deprecation 경고 72개. 실패로 처리된 검사 없음

내부 Raw source 전환 후 최신 검증: 관련 64개 통과, 공식 원본과 출력·손실·gradient 일치.
전체 회귀는 1,037 passed / CUDA 6 skipped / 46 subtests passed.
외부 chronos 미설치 환경은 1,025 passed / 18 skipped / 46 subtests passed.
노트북 코드 21개 전체 실행과 별도 wheel 설치본의 H27 forward/backward 확인.
출처·경계는 `src/modeling_module/models/Chronos2/native/README.md`,
최신 증적은 `docs/evidence/Chronos2NativeSource20261119.json` 참고.

## 5. 남은 작업 순서

**시간 창과 패턴별 표집 구성 · 다음 작업**

고정된 분할을 유지하면서 실제 수량 품질과 L52/H27 이력 가용성을 검사하는 단계.
유효 창 수·제외 사유를 기록하고 학습 context만으로 패턴별 균형 표집을 구성하는 완료 기준.

**Trainer와 자체 체크포인트 연결 · 다음 작업**

앞 단계의 배치 계약에 학습 반복·검증·중단·재시작·저장·자체 가중치 추론을 연결하는 단계.
저장 전후 H27 예측 일치와 통제된 중단·재개 재현을 확인하는 완료 기준.

**제한된 실데이터 GPU 실험 · 승인 필요**

위 구현과 합성 통합 검증 후 대상 장치·데이터 범위·step·시간 예산을 확정할 단계.
현재 실데이터 학습, 공용 Runtime 변경, 커밋·Push·main 병합은 미실행.
