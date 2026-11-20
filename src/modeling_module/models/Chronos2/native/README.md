# Chronos2 내부 연산 소스

## 범위와 출처

공식 Chronos-2 연산 코드를 저장소 내부에서 읽고 수정할 수 있도록 포함한 구성.
새 알고리즘의 독자 재구현이 아니라 버전을 고정한 공개 원본의 내부 관리 방식.

- 원본: https://github.com/amazon-science/chronos-forecasting
- 기준 태그: `v2.3.2`
- 고정 커밋: `0aba28e9360b6355d5a919596c90b16c25939f02`
- 라이선스·저작권: `LICENSE.upstream`, `NOTICE.upstream` 및 각 Python 파일 헤더 보존
- 원본 URL·SHA-256·내부 파일 해시·수정 내역: `upstream_manifest.json`

공식 커밋의 파일과 기존 검증 환경에 설치했던 패키지 파일이 바이트 단위로 동일한지 확인 후 포함.
추가한 한글 주석과 `model.py`의 정규화·패치 import 변경을 제외하면 연산 AST가 원본과 동일한 상태.
원본 갱신 시에는 manifest 해시와 원본 대조 테스트를 함께 갱신할 대상.

## 읽는 순서

| 파일 | 주요 클래스·메서드 | 역할 |
|---|---|---|
| `config.py` | `Chronos2CoreConfig`, `Chronos2ForecastingConfig` | 층·head·차원·patch·분위수 설정 |
| `preprocessing.py` | `Patch`, `InstanceNorm` | NaN 왼쪽 패딩, 패치 분할, 정규화·arcsinh·역변환 |
| `layers.py` | `MHA`, `TimeSelfAttention`, `GroupSelfAttention` | 실제 Attention, RoPE, 잔차 MLP |
| `model.py` | `Chronos2EncoderBlock` | Time → Group Attention → Feed Forward |
| `model.py` | `_prepare_patched_context`, `_prepare_patched_future` | 입력·미래 토큰과 관측 마스크 준비 |
| `model.py` | `_compute_loss`, `forward` | 분위수 손실, 출력 head, 원 단위 예측 |

`Patch`와 `InstanceNorm`은 공식 `chronos_bolt.py`에서 필요한 클래스만 추출한 구성.
Chronos-Bolt 모델 전체나 Chronos2의 공식 Pipeline·dataset·trainer를 복사한 범위는 아님.

```text
Chronos2TrainableBackbone
  -> native.config.Chronos2CoreConfig
  -> native.model.Chronos2Model
     -> native.preprocessing.Patch / InstanceNorm
     -> native.layers.TimeSelfAttention / GroupSelfAttention / FeedForward
     -> native.model._compute_loss
```

## 외부 의존성 경계

```bash
python -m pip install -e '.[chronos2-native]'
```

학습용 backbone은 외부 `chronos-forecasting` 패키지를 import하거나 설치 여부를 검사하지 않는 경로.
PyTorch·Transformers·einops·packaging은 기반 라이브러리로 계속 사용.
`huggingface-hub`는 Transformers의 기반 의존성이지만 랜덤 초기화 과정에서 다운로드를 수행하지 않는 계약.

기존 공식 가중치 zero-shot의 `../backbone.py`는 변경하지 않았으므로 그 경로에만
기존 `chronos2` extra와 `chronos-forecasting==2.3.2` 유지.
전체 저장소에서 모든 외부 의존성을 없앴다는 의미와 구분.

`provenance()`의 `implementation_source=vendored`, `implementation_module`, `upstream_commit`으로
실제 실행 위치와 원본을 추적. 기존 `chronos_forecasting` 필드는 호환을 위한 원본 버전 표기이며
외부 패키지 설치·호출을 의미하지 않는 구분.

## 실행과 검증

일반 사용은 직접 raw 모델보다 `../trainable_backbone.py`와 `../training/batch.py` 경유 권장.
raw 클래스는 공식의 낮은 수준 API를 유지하므로 프로젝트의 수량·독립 group 검사를 대신하지 않는 경계.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m pytest -q tests/test_chronos2_native_source.py \
  tests/test_chronos2_pretraining_backbone.py tests/test_chronos2_pretraining_contracts.py
```

원본 대조 검사는 별도 검증 환경에 `chronos-forecasting==2.3.2`가 있을 때만 수행.
앱 실행용으로 설치할 필요는 없는 원본 대조 도구.
검사 대상: 파일 해시·라이선스·AST, 공식/내부 초기화·state_dict·출력·손실·gradient,
eager/SDPA, L52/L53, H13/H27/H33, 정규화·patch, 외부 chronos import 차단.

2026-11-19 검증: 관련 64개 통과. 공식 패키지가 있는 전체 환경은 1,037 passed / CUDA 6 skipped,
외부 chronos 미설치 환경은 1,025 passed / 18 skipped. 두 환경 모두 46 subtests 통과.
별도 wheel 설치본에서도 소스·라이선스 리소스 해시와 H27 forward/backward 확인.
노트북은 외부 chronos 미설치 환경에서 코드 셀 21개 전체 실행 완료.

셀별 해설과 실제 소스 관찰은
[`../notebook/chronos2_pretraining_contracts_and_backbone.ipynb`](../notebook/chronos2_pretraining_contracts_and_backbone.ipynb) 참고.
실데이터 학습·공용 GPU 실행·DB 변경·배포는 이 소스 포함 작업에서 제외.
