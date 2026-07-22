# PatchTST implementation notes

이 문서는 현재 repository의 PatchTST 구현과 public API 책임을 설명합니다. 수치 및 state
schema 기준선은 [PatchTSTBaseline.md](PatchTSTBaseline.md), 사용자에게 노출되는 최종 지원
범위는 repository root의 `README.md`와 `README.package.md`를 기준으로 합니다.

## Registered artifacts

| Artifact key | Output mode | Exogenous policy | Family default |
|---|---|---|---|
| `patchtst_base` | point, `Normal`, `StudentT` | endogenous 기본, legacy config routing | 포함 |
| `patchtst_exogenous` | point, `Normal`, `StudentT` | past/future 중 하나 이상 필수 | 제외 |
| `patchtst_quantile` | q10/q50/q90 | endogenous 기본, legacy config routing | 포함 |
| `patchtst_quantile_exogenous` | q10/q50/q90 | past/future 중 하나 이상 필수 | 제외 |

신규 exogenous 학습은 명시적인 exogenous key를 사용합니다. `patchtst_base`와
`patchtst_quantile`에 exogenous width가 들어오면 대응 exogenous subclass를 선택하는 동작은
기존 config와 checkpoint 호환을 위해 유지합니다. `patchtst` family는 base와 quantile만
확장하며 exogenous artifact를 자동 추가하지 않습니다.

Capability routing은 `get_patchtst_default_model_key`로 조회합니다. 이 함수는 모델 간 정확도
우위를 뜻하지 않고 입력 및 출력 계약에 맞는 artifact 책임만 반환합니다.

## Model responsibilities

| Module | Responsibility |
|---|---|
| `common/configs.py` | Patch, encoder, output 및 exogenous width 직렬화 |
| `common/backbone_base.py` | Target/past-exogenous patch 생성과 input projection |
| `supervised/backbone.py` | Patch token Transformer encoding |
| `supervised/PatchTST.py` | RevIN, future cross-attention, point/distribution/quantile 계산 그래프 |
| `supervised/variants.py` | Endogenous/exogenous config 및 forward 입력 계약 |
| `heads/*.py` | 계산 그래프 내부의 output head 구현 |
| `model_builder.py` | Legacy config routing과 명시적 variant 생성 |
| `registry.py` | Artifact identity, family expansion, checkpoint alias, capability routing |
| `patchtst_train.py` | 학습 데이터에 맞춘 future path/head 재구성과 trainer 연결 |
| `self_supervised/*` | Target-only masked patch reconstruction |

`PatchTSTModel`과 `PatchTSTQuantileModel`은 checkpoint 호환을 위한 공통 계산 구현입니다. 신규
호출자가 직접 선택할 public artifact는 strict variant와 registry key이며, head 구현은 독립
artifact가 아닙니다.

## Data contract

Public training 및 prediction 계약은 다음과 같습니다.

- target: `(B, lookback, 1)`
- past continuous exogenous: exogenous artifact에서 선택 가능
- future continuous exogenous: exogenous artifact에서 선택 가능
- categorical exogenous: public API에서 fail-fast

내부 모델은 `past_exo_cat_dim`, cardinality와 embedding을 직접 구성하면 categorical tensor를
계산할 수 있지만 이는 public 지원 범위가 아닙니다. Public pipeline에서는 categorical feature를
continuous feature로 인코딩한 뒤 전달합니다.

구성된 exogenous width는 해당 입력을 필수로 만듭니다. Future width가 0보다 크면
`(B, horizon, future_exo_dim)`이 필요하며 batch, horizon, width 불일치는 모델 경계에서 즉시
실패합니다. Width 0 모델은 non-empty future tensor를 받지 않습니다.

## Endogenous flow

1. Target를 RevIN으로 정규화합니다.
2. Target patch를 `d_model` token으로 투영합니다.
3. `SupervisedBackbone`이 patch sequence를 인코딩합니다.
4. Point/distribution/quantile head가 horizon 출력을 만듭니다.
5. Location 또는 point/quantile을 target scale로 복원합니다.

Endogenous variant는 exogenous width가 모두 0이어야 하며 forward signature도 target-only로
제한합니다.

## Exogenous flow

1. Past continuous feature는 target과 같은 window로 patching한 뒤 patch vector에 concat합니다.
2. Future continuous feature는 horizon token으로 투영합니다.
3. Backbone patch token이 future token sequence를 cross-attention으로 읽습니다.
4. Sigmoid gate가 적용된 attention/FFN residual을 거쳐 기존 output head로 전달합니다.

현재 head의 `d_future`는 0으로 유지되므로 future feature를 head에서 다시 flatten-concat하지
않습니다. Future 정보의 활성 경로는 `FutureExoTokenFusion` 하나입니다.

## Output contract

- Point: `(B, horizon)` tensor
- Distribution: `(B, horizon, outputsize_multiplier)` packed tensor
- Quantile: `{"q": (B, horizon, Q)}`

Public predictor는 point/distribution checkpoint에서 location을 `{"point": ...}`로 노출합니다.
Quantile checkpoint는 q10/q50/q90과 `point=q50`을 노출합니다. Distribution sample 또는 interval
생성은 현재 predictor 계약에 포함되지 않습니다.

## SSL contract

`SSLConfig.mode`는 `sl_only`, `ssl_only`, `full`을 지원하며 `off`는 `sl_only` alias입니다.

- `sl_only`: supervised training만 실행
- `ssl_only`: target-only pretraining 후 supervised artifact 없이 종료
- `full`: target-only pretraining 후 선택한 PatchTST artifact를 fine-tune

Pretraining은 모든 exogenous width를 0으로 강제합니다. Exogenous supervised fine-tuning은
pretrained target projection과 encoder에서 shape가 일치하는 파라미터만 복원하고 exogenous
projection/fusion은 supervised 단계에서 학습합니다.

## Lineage boundary

현재 모델은 PatchTST 논문의 patch-token Transformer 아이디어에서 출발했지만 upstream 코드를
수식 그대로 보존한 Original implementation은 아닙니다. 특히 현재 public target는 단변량이며,
`c_in > 1`이면 채널별 독립 처리 대신 한 patch projection에서 채널을 결합합니다. Mean-pooling
head, causal attention 설정, distribution/quantile head와 exogenous fusion도 repository 확장입니다.

따라서 이 모델은 `PatchTST-derived production implementation`으로 관리하고, 공식 논문 수치와의
직접 parity를 주장하지 않습니다.

## Code map

- [configuration](../common/configs.py)
- [core supervised models](../supervised/PatchTST.py)
- [strict variants](../supervised/variants.py)
- [backbone](../supervised/backbone.py)
- [self-supervised path](../self_supervised/PatchTST.py)
- [trainer integration](../../../training/model_trainers/patchtst_train.py)
