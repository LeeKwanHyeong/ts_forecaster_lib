# PatchTST Future Categorical RTX 5090 검증

검증일: 2026-07-24

## 결론

PatchTST의 미래 카테고리 입력은 학습, checkpoint 복원, 운영 추론까지
정상 연결됩니다.

- 학습 사전에 없는 운영 값은 재학습 없이 UNK ID `0`으로 변환됩니다.
- UNK가 포함된 요청도 예측을 정상 완료하며 checkpoint vocabulary는 바뀌지
  않습니다.
- 요청에서 카테고리 컬럼이 빠지거나, 설정된 카테고리 순서가 바뀌거나,
  과거/미래 역할이 checkpoint schema와 달라지면 모델 실행 전에 명확한
  `ValueError`로 차단됩니다.
- RTX 5090 합성 데이터 실험에서 카테고리를 포함한 모델의 평균 MAE는
  `1.04393`에서 `0.77048`로 `26.19%` 개선됐습니다.
- 카테고리 경로의 추가 메모리는 학습 peak 기준 `0.189 MiB`, 추론 peak
  기준 `0.246 MiB`였습니다.
- 추론 시간의 seed 중앙값은 `1.282 ms`에서 `1.411 ms`로 `10.07%`
  증가했습니다.

이 정확도 수치는 카테고리 신호가 정답에 영향을 주도록 만든 합성 데이터의
기능 검증 결과입니다. 실제 수요 데이터에서 같은 개선 폭을 보장하지 않으며,
운영 모델 선정에는 별도의 qualification이 필요합니다.

## 운영 입력 계약

### 처음 등장한 카테고리

학습 vocabulary에 없는 값은 `CategoricalVocabulary.id_of()`와 추론
Dataset에서 모두 UNK ID `0`으로 변환됩니다. 운영 요청으로 새 값
`emergency`를 전달한 통합 테스트에서 다음 계약을 확인했습니다.

1. 인코딩된 `future_cat` tensor의 모든 값이 `0`입니다.
2. Point와 Quantile 예측이 모두 유한한 값을 반환합니다.
3. 예측 전후 checkpoint vocabulary fingerprint가 동일합니다.

### 잘못된 요청

다음 입력은 모델 forward 전에 차단됩니다.

| 오류 | 검증 기준 |
|---|---|
| 카테고리 컬럼 누락 | 설정에 있는 컬럼이 입력 DataFrame에 없으면 누락 컬럼 이름을 표시 |
| 카테고리 설정 순서 변경 | checkpoint vocabulary 순서와 요청의 `future_exo_cat_cols` 순서를 비교 |
| 과거/미래 역할 변경 | 요청 schema와 checkpoint schema의 `past_cat_names`, `future_cat_names`를 비교 |
| cardinality 불일치 | 요청에 연결된 vocabulary와 checkpoint cardinality를 비교 |

DataFrame의 물리적인 컬럼 배치는 이름 기반으로 처리합니다. 여기서 순서
계약은 모델 tensor 축을 결정하는 `past_exo_cat_cols`와
`future_exo_cat_cols`의 설정 순서를 뜻합니다.

## RTX 5090 측정 조건

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| 정밀도 | BF16 |
| seed | 11, 22, 33 |
| lookback / horizon | 52 / 27 |
| 학습 | 준비 100 step, 측정 1000 step, batch 128 |
| 추론 | 준비 100회, 측정 200회, batch 128 |
| loss / optimizer | MAE / AdamW |
| 학습률 | 0.001 |
| 데이터 | 미래 연속형 2개와 미래 카테고리 2개가 포함된 고정 합성 데이터 |
| 비교 모델 | 연속형 전용, 동일 연속형 + 미래 카테고리 |

두 모델은 PatchTST backbone, 연속형 입력, 데이터, loss, seed를 동일하게
사용합니다. 카테고리 모델에만 cardinality `(5, 4)`와 embedding 크기
`8`을 추가했습니다. 데이터 로딩과 host-to-device 전송 시간은 측정에서
제외했습니다.

실행 소스는 `exogenous-models`의 HEAD
`846d3e2a1adfb453931a0e79850a9a9cd3b36865`을 기반으로 한 dirty
snapshot입니다. Python source snapshot SHA-256은
`1fd357f1b0bb9a9b97d54548d8bde5fbab85ec64f9e90fedb0ae02d6eedd0f9c`입니다.
5090의 기존 저장소는 수정하지 않고 `/tmp`의 격리 snapshot에서
실행했습니다.

## 정확도

3개 seed 평균입니다.

| 모델 | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| 연속형 전용 | 1.04393 | 1.27631 | 20.1276% |
| 연속형 + 카테고리 | 0.77048 | 1.08293 | 14.9662% |
| 변화 | -26.19% | -15.15% | -25.69% |

seed별 MAE도 같은 방향입니다.

| seed | 연속형 전용 | 연속형 + 카테고리 |
|---:|---:|---:|
| 11 | 1.03713 | 0.78337 |
| 22 | 1.05522 | 0.76457 |
| 33 | 1.03944 | 0.76350 |

200-step 사전 실험에서는 개선 폭이 `0.08%`에 불과했습니다. 카테고리
경로가 신호를 학습하기 전에 실험을 종료한 결과였으며, 1000-step에서
세 seed 모두 안정적으로 차이가 나타났습니다.

## 속도와 VRAM

시간은 seed별 평균 step 시간의 중앙값을 사용합니다. 같은 shape의
반복 측정 중 일시적인 서버 지연이 관측돼 단순 평균보다 중앙값이
재현성을 더 잘 나타냈습니다.

| 지표 | 연속형 전용 | 연속형 + 카테고리 | 변화 |
|---|---:|---:|---:|
| 파라미터 수 | 606,621 | 608,741 | +2,120 |
| 학습 step 중앙값 | 15.022 ms | 10.520 ms | -29.97% |
| 추론 step 중앙값 | 1.282 ms | 1.411 ms | +10.07% |
| 학습 peak allocated | 82.875 MiB | 83.063 MiB | +0.189 MiB |
| 추론 peak allocated | 28.379 MiB | 28.625 MiB | +0.246 MiB |

카테고리 모델의 학습 step이 더 빠른 현상은 두 차례 순서 변경 측정에서도
같은 방향이었습니다. 작은 입력 폭의 연속형 projection과 더 넓은
카테고리 결합 projection이 서로 다른 CUDA kernel 효율을 보인 결과로
추정합니다. 이는 이 shape와 RTX 5090 환경의 관측값이며 다른 batch,
horizon, GPU에서도 빨라진다는 보장은 아닙니다.

canonical run에서 카테고리 추론 한 건이 `12.583 ms`로 튀었고, 동일
조건 재측정에서는 `1.422 ms`로 정상화됐습니다. 다른 재측정에서는
연속형 모델 한 건이 `16.952 ms`로 튀어 특정 모델 경로의 구조적인
지연이 아니라 순간적인 서버 지연임을 확인했습니다.

## 원본 결과

| 파일 | SHA-256 |
|---|---|
| `PatchTSTFutureCategoricalRTX5090Benchmark-20260724.json` | `7217f2d717a23d3e7418ba18e698495210f0ea884413449976ed27fa46711402` |
| `PatchTSTFutureCategoricalRTX5090Seed22Retry-20260724.json` | `d0eef592a0e2b33635de548a9cbb3aca316ba421cfa9b8deec6376152dc6eeb4` |
| `PatchTSTFutureCategoricalRTX5090Seed33Retry-20260724.json` | `d33fe92a0547a8b18a116c2373e46716071918f3f4e3b2c362ad71ce5933659f` |

재실행 도구는 저장소의
`tools/benchmark_patchtst_future_categorical_5090.py`입니다.
