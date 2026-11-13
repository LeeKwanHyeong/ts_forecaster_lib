# TinyTimeMixer 자체 학습 구현

## 구현 범위

TTM 논문의 패치 기반 MLP Mixer 구조를 참고해, 모델의 연산과 정적 특성
결합 과정을 읽기 쉽게 분리한 독립 PyTorch 구현이다. 모든 모델은 무작위
초기화되며 공개 가중치 다운로드나 `granite-tsfm`, `transformers` 호출이 없다.

IBM 코드의 전체 복제나 수치적으로 동일한 재현 구현은 아니다. 공식
체크포인트와 이름·가중치 호환성을 보장하지 않으며, 예측 품질 검증도 별도다.

| 파일 | 역할 |
| --- | --- |
| `configs.py` | 입력 길이, 패치, Mixer, 정적 사전 설정 및 검증 |
| `layers.py` | 채널·패치·은닉 축의 MLP, 게이트, 적응형 패치 계층 |
| `backbone.py` | 관측값 정규화, 패치화, 주기 토큰, 채널 독립 표현 |
| `decoder.py` | 정적 코드 임베딩, 원시 패치 잔차, 채널 결합 |
| `TinyTimeMixer.py` | 미래 예측 헤드와 역정규화, 백본 동결 |
| `pretraining.py` | 자체 마스킹 복원 목적과 예측 모델로의 가중치 전이 |
| `examples/train_from_scratch.py` | 합성 수요의 짧은 사전학습·미세조정 예제 |

기존 모델과 같은 `from_config()` 생성 방식을 제공하지만, 공용 `api.train`,
`api.infer`, 모델 Registry에는 아직 등록하지 않은 실험 모델이다. 기존
DataLoader도 정적 입력을 자동 전달하지 않는다. 아래처럼 직접 호출한다.

## 입력과 예측 흐름

```text
수요 x (B,L,C)
  → 관측값만으로 정규화
  → 패치화 (B,C,P,patch_length)
  → 선형 투영 (B,C,P,D)
  → 선택적 주기 토큰과 적응형 패치 Mixer
  → 백본 표현
                         정적 코드 (B,S)
                           → 변수별 임베딩
                           → 패치 축으로 펼침 (B,S,P,D_dec)
  → 디코더 차원 투영 ────────────┘
  → 채널 축 결합 (B,C+S,P,D_dec)
  → 채널·패치·은닉 Mixer
  → 정적 출력 채널 제거
  → 미래 예측 헤드 (B,H,C)
  → 같은 호출의 통계로 역정규화
```

- B는 배치, L은 과거 길이, H는 미래 길이, C는 수요 채널, S는 정적 변수 수다.
- 일반적인 부품별 예측에서는 여러 부품을 B에 넣고 C=1로 구성한다.
- 법인·부품군 ID를 L번, H번 복사해서 넣지 않는다. `static_categorical_values`
  한 번으로 전달하고 내부에서 임베딩을 패치 축으로 펼친다.
- 정적 분기는 디코더 전용이다. 같은 수요 입력의 백본 출력은 정적 코드만
  바꿔도 동일하지만, 디코더 이후 결과는 달라질 수 있다.
- 백본은 채널 간 가중치를 공유하므로 정적 분기 없이도 서로 다른 시계열을
  공통 학습할 수 있다. 정적 분기가 있으면 디코더의 채널 혼합을 활성화해야 한다.
- `adaptive_patching_levels=2`는 2P·D/2 단계와 P·D 단계를 차례로 적용한다.
- `resolution_vocab_size>0`이면 `freq_token`에 (B,) 정수 코드를 전달한다.
  주기 코드의 의미와 사전은 사용자가 일관되게 정의한다.

```python
import torch
from modeling_module.models.TinyTimeMixer import TinyTimeMixerConfig, TinyTimeMixerModel

cfg = TinyTimeMixerConfig(
    lookback=52, horizon=26, num_input_channels=1,
    categorical_vocab_size_list=(10, 30),
    decoder_mode="mix_channel",
)
model = TinyTimeMixerModel.from_config(cfg)
x = torch.randn(2, 52, 1)
static_ids = torch.tensor([[1, 4], [2, 7]], dtype=torch.long)
forecast = model(x, static_categorical_values=static_ids)  # (2,26,1)
```

범주 수에는 UNK가 있다면 그 자리까지 포함한다. 문자열→정수 사전은 학습
데이터에서 정의하고 전이·추론 때 그대로 유지한다. 범위를 벗어난 코드,
실수형 코드, 필요한 정적 입력의 누락은 오류로 처리한다. 정적 연속형 입력과
과거·미래 동적 외생변수 경로는 이 버전에 포함하지 않는다. 다른 인자를
전달하면 묵살하지 않고 호출 오류를 발생시킨다.

## 두 가지 공통 학습 방식

### 미래 예측을 바로 공통 학습

여러 부품을 묶은 `TinyTimeMixerModel`로 과거→미래 예측 오차를 학습한다.
이 모델을 법인별로 추가 학습하면 예측 목적의 전이학습이다. 처음부터
미래 예측 헤드까지 학습하므로 아래 복원 사전학습과 구분한다.

### 마스킹 복원 후 미래 예측 학습

`TinyTimeMixerPretrainModel`은 이 저장소의 자체 확장이다. IBM의 공식 사전학습
방식을 재현한다고 해석하지 않는다. 과거 패치를 가리고 정규화된 MSE 또는
MAE로 복원하며, 실제 관측된 가림 위치만 손실에 포함한다.

숨긴 값은 정규화 평균·분산, 패치 투영 입력, 디코더 원시 잔차에서 모두 제외한다.
겹친 패치로 같은 값이 노출되는 것을 막으려고 복원 학습은 다음 조건을 요구한다.

- `patch_stride == patch_length`
- `lookback % patch_length == 0`
- 패치가 2개 이상이며 각 채널에 보이는 관측값과 숨긴 패치가 모두 존재

```python
from modeling_module.models.TinyTimeMixer import TinyTimeMixerPretrainModel

pretrainer = TinyTimeMixerPretrainModel(cfg)
result = pretrainer(x, static_categorical_values=static_ids, mask_ratio=0.3)
result["loss"].backward()

forecaster = TinyTimeMixerModel(cfg)
pretrainer.transfer_to(forecaster, freeze_backbone=False)
```

실제 학습에는 옵티마이저의 `zero_grad()`와 `step()`이 추가로 필요하다.
전이 대상은 백본·디코더·정적 임베딩이며, 복원 헤드를 미래 예측 헤드로
변환하지 않는다. 예측 헤드는 새로 학습해야 하므로 복원 사전학습 직후의
모델을 제로샷 예측 준비가 끝난 모델로 간주하지 않는다.

전이 시 `horizon`만 달라도 되며 나머지 설정은 같아야 한다. 백본을 동결하면
정적 임베딩·디코더·예측 헤드만 갱신한다. 동결은 `requires_grad`만 바꾸므로
드롭아웃 사용 여부는 `train()`·`eval()`로 별도 제어한다.

8개 패턴의 균형 샘플링, 법인별 데이터 선택, 부품 분할과 시점별 검증은
모델 외부 학습 파이프라인의 책임이다. 평가 시점 이후의 수요로 사전학습하거나
전체 생애를 보고 만든 패턴 라벨을 과거 예측 입력으로 넣으면 누수가 발생한다.

## 실행과 검증

저장소 루트에서 데이터 파일과 네트워크 없이 실행한다.

```bash
PYTHONPATH=src python -m modeling_module.models.TinyTimeMixer.examples.train_from_scratch
PYTHONPATH=src python -m unittest discover -s tests -p 'test_tinytimemixer.py' -v
```

예제는 CPU에서 복원 학습 3회, 미래 예측 학습 3회만 수행하며 파일을 만들지
않는다. 실제 부품 데이터 학습이나 예측 정확도 평가는 실행하지 않는다.

가중치를 저장할 때는 `model.state_dict()`와 `dataclasses.asdict(cfg)`를 함께
관리한다. 다시 초기화한 모델에 같은 설정과 `load_state_dict(..., strict=True)`를
사용한다. 외부 체크포인트를 임의로 읽는 로더는 제공하지 않는다.

## 참고와 미구현 범위

- 논문: https://arxiv.org/abs/2401.03955
- 구조 참고: https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer
- 반영한 개념: 채널 독립 백본, 패치·은닉·채널 MLP, 축별 게이트,
  적응형 패치, 선택적 주기 토큰, 정적 디코더, 원시 패치 잔차
- 별도 구현한 부분: 관측값 전용 정규화, 누수 방지 마스킹 복원,
  엄격한 입력 검증, 복원→예측 가중치 전이
- 미구현: 공식 다양한 해상도 데이터 샘플러, 예측 후 FCM 보정,
  동적 외생변수, 분위수·분포 출력, 공식 체크포인트 호환, 공용 학습 API 연결
