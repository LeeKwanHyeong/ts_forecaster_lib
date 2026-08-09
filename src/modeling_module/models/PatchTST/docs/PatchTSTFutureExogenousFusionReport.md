# PatchTST Future Exogenous Fusion 개선 보고

## 1. 배경

PatchTST에 미래 외생변수 Future Exogenous를 넣는 목적은 명확하다.

수요 예측에서 과거 판매량만으로는 설명하기 어려운 미래 조건들이 있다.

예를 들면:

- 보증 종료 시점
- 수요 발생 후 경과 시간
- 누적 주문량 변화
- warranty 관련 future-known 정보
- calendar / lifecycle / campaign / supply 관련 미래 시점 feature

이런 값들은 예측 시점에 이미 알 수 있거나 계획 가능한 정보이므로, 모델이 horizon별 예측을 할 때 활용할 수 있어야 한다.

기존 PatchTST에는 future exogenous를 넣는 경로가 있었지만, 실제 AB 테스트 결과 특정 데이터셋에서 매우 불안정한 예측 패턴이 나타났다.

특히 기존 방식에서는:

- 특정 horizon에서 예측이 과도하게 튐
- 어떤 horizon에서는 거의 0에 가까워짐
- part별 차이를 충분히 반영하지 못함
- aggregate forecast가 불안정하게 흔들림

이 문제가 관찰되었다.

따라서 기존 future exogenous fusion 방식을 재검토하고, 새롭게 token cross-attention 기반 방식을 추가하여 비교 검증했다.

## 2. 기존 PatchTST Future Exogenous 방식

### 2.1 작동 원리

기존 방식은 `head_flatten` 방식이다.

입력 future exogenous의 shape는 일반적으로 다음과 같다.

```text
future_exo: (B, H, E)
```

여기서:

- `B`: batch size
- `H`: forecast horizon
- `E`: future exogenous feature dimension

기존 방식은 이 future exogenous를 horizon 축과 feature 축을 모두 펼쳐서 하나의 벡터로 만든다.

```text
(B, H, E) -> (B, H * E)
```

이후 linear projection을 통해 모델 내부 차원으로 변환한다.

```text
future_exo_flat = flatten(future_exo)
future_exo_emb = Linear(H * E -> d_model)(future_exo_flat)
```

그리고 PatchTST backbone에서 나온 representation과 head 단계에서 결합한다.

전체 흐름은 다음과 같다.

```text
past target + past exo
        |
        v
PatchTST Backbone
        |
        v
backbone representation
        |
        +---- concat / projection ---- future_exo_flatten_embedding
        |
        v
Prediction Head
        |
        v
forecast: (B, H)
```

즉 기존 방식에서는 future exogenous가 backbone 내부의 patch token들과 상호작용하지 않고, 마지막 prediction head 근처에서 한 번에 주입된다.

## 3. 기존 방식의 문제점

### 3.1 Horizon 구조가 손실됨

future exogenous는 원래 horizon별 정보다.

예를 들어:

```text
week 1 feature
week 2 feature
week 3 feature
...
week H feature
```

처럼 각 미래 시점마다 다른 의미를 갖는다.

하지만 기존 방식은 `(H, E)`를 `H * E`로 flatten한다.

그 결과 모델 입장에서는:

```text
horizon별 token sequence
```

가 아니라

```text
큰 feature vector 하나
```

로 보게 된다.

즉 `horizon 1의 외생변수`, `horizon 13의 외생변수`, `horizon 27의 외생변수`가 시간 구조를 가진 sequence로 처리되지 않는다.

이 때문에 특정 horizon의 정보가 전체 forecast vector에 과하게 영향을 주거나, horizon 간 영향이 뒤섞일 수 있다.

### 3.2 Future exo가 item-specific signal을 덮을 수 있음

PatchTST backbone은 과거 target series와 past exogenous를 통해 part/item별 패턴을 학습한다.

그런데 기존 방식에서는 future exogenous가 head에서 큰 벡터로 주입된다.

이 구조에서는 모델이 다음 중 더 쉬운 경로를 선택할 수 있다.

```text
part별 과거 패턴을 세밀하게 활용하기
```

보다

```text
future_exo_flat vector를 보고 horizon별 공통 패턴을 강하게 출력하기
```

가 더 쉬워질 수 있다.

특히 future exogenous가 part 간에 비슷하거나 공통적인 feature를 많이 포함하면, 모델이 part별 차이를 무시하고 horizon별 공통값처럼 예측하는 collapse가 발생할 수 있다.

실험 중 관찰된 증상도 이와 일치했다.

- 같은 horizon에서 여러 part가 유사한 예측값을 가짐
- 일부 horizon에서 모든 part가 비정상적으로 낮거나 높게 예측됨
- aggregate plot에서 큰 spike / drop 발생

### 3.3 Head에 너무 많은 역할이 몰림

기존 방식에서는 future exogenous를 backbone 단계에서 해석하지 않는다.

대신 마지막 head가 아래를 동시에 처리해야 한다.

- backbone representation 해석
- flattened future exo 해석
- part별 level 조정
- horizon별 output 생성
- future exo와 과거 패턴의 상호작용 학습

이 부담이 head에 집중된다.

특히 `H * E`가 커질수록 head가 받아야 하는 future exo vector도 커진다.

예를 들어:

```text
H = 27
E = 17
H * E = 459
```

이면 future exogenous만 459차원으로 flatten된다.

이 큰 벡터가 마지막 head에서 한 번에 들어오면, 안정적인 시계열 representation과 균형 있게 결합되기 어렵다.

### 3.4 실험상 불안정성이 컸음

Walmart sanity check와 future exo AB test에서 기존 `head_flatten` 방식은 다음과 같은 특징을 보였다.

- MAE / RMSE / WAPE가 크게 악화
- aggregate forecast에서 spike가 큼
- 일부 horizon에서 예측이 비정상적으로 튐
- `patchtst_no_future`보다 훨씬 나쁜 결과
- `token_cross_attn`보다 훨씬 나쁜 결과

특히 `head_flatten` 방식은 future exogenous를 넣었음에도 성능이 개선되는 것이 아니라, 오히려 모델을 불안정하게 만드는 방향으로 작동했다.

## 4. Token Cross-Attention 방식을 채택한 이유

기존 방식의 핵심 문제는 future exogenous를 시간축 구조 없이 flatten해서 head에 직접 주입한다는 점이다.

따라서 새 방식의 목표는 다음과 같았다.

1. Future exogenous의 horizon-wise 구조를 보존한다.
2. Future exogenous를 head가 아니라 token representation 단계에서 결합한다.
3. 과거 시계열 patch token과 future exogenous token이 attention을 통해 상호작용하게 한다.
4. Future exogenous가 item-specific backbone representation을 덮지 않고, 보조 정보로 작동하게 한다.
5. 기존 PatchTST backbone 구조는 최대한 유지한다.

이를 위해 token cross-attention 방식을 채택했다.

## 5. Token Cross-Attention Future Exo Fusion 구조

### 5.1 입력 구조

새 방식에서도 future exogenous 입력은 동일하다.

```text
future_exo: (B, H, E)
```

하지만 이를 flatten하지 않는다.

대신 각 horizon step을 하나의 token으로 본다.

```text
future_exo tokens:
(B, H, E)
```

이를 model dimension으로 projection한다.

```text
future_tokens = Linear(E -> d_model)(future_exo)
```

결과는:

```text
future_tokens: (B, H, d_model)
```

### 5.2 PatchTST backbone token

PatchTST backbone은 과거 target 및 past exogenous를 patch 단위 token representation으로 만든다.

개념적으로:

```text
backbone_tokens: (B, N, d_model)
```

여기서:

- `N`: patch token 수
- `d_model`: hidden dimension

### 5.3 Cross-Attention Fusion

새 방식에서는 backbone token이 future exogenous token을 attention으로 참조한다.

개념적으로는 다음과 같다.

```text
Query  = backbone_tokens
Key    = future_tokens
Value  = future_tokens
```

즉 과거 시계열에서 만들어진 patch representation이, 미래 외생변수 token들을 선택적으로 참고한다.

```text
attended_future = CrossAttention(
    query = backbone_tokens,
    key   = future_tokens,
    value = future_tokens
)
```

이후 residual connection과 normalization을 통해 backbone representation에 future information을 더한다.

```text
fused_tokens = Norm(backbone_tokens + Dropout(attended_future))
```

전체 흐름은 다음과 같다.

```text
past target + past exo
        |
        v
PatchTST Backbone
        |
        v
backbone tokens: (B, N, D)

future exo: (B, H, E)
        |
        v
future tokens: (B, H, D)

backbone tokens -- Query
future tokens   -- Key / Value
        |
        v
Cross-Attention Fusion
        |
        v
fused backbone tokens
        |
        v
Prediction Head
        |
        v
forecast: (B, H)
```

## 6. 기존 방식과 신규 방식의 핵심 차이

| 항목 | 기존 head_flatten | 신규 token_cross_attn |
|---|---|---|
| future exo 처리 | `(B,H,E)`를 `(B,H*E)`로 flatten | `(B,H,E)`를 horizon token으로 유지 |
| 결합 위치 | Prediction head 근처 | Backbone 이후 token representation 단계 |
| horizon 구조 | 손실됨 | 유지됨 |
| 과거 패턴과 상호작용 | 제한적 | attention으로 직접 상호작용 |
| collapse 위험 | 높음 | 낮음 |
| head 부담 | 큼 | 상대적으로 작음 |
| part별 signal 보존 | 약함 | 더 강함 |
| 실험 안정성 | 낮음 | 높음 |

## 7. AB Test 결과

### 7.1 Walmart sanity check 결과

Walmart 데이터에서 기존 `head_flatten` 방식은 매우 불안정했다.

대표 결과:

```text
patchtst_no_future         WAPE ≈ 0.5776
patchtst_token_cross_attn  WAPE ≈ 0.5850
timexer                    WAPE ≈ 0.6200
exotst                     WAPE ≈ 0.6368
patchtst_head_flatten      WAPE ≈ 1.2889
```

해석:

- `patchtst_head_flatten`은 압도적으로 나빴다.
- `patchtst_token_cross_attn`은 `head_flatten` 대비 큰 폭으로 개선됐다.
- `token_cross_attn`은 `patchtst_no_future`와 거의 비슷한 수준까지 안정화됐다.
- 다만 Walmart에서는 future exo를 넣은 token cross-attn이 no-future를 넘어서지는 못했다.

즉 Walmart 기준 결론은:

```text
future exo 자체가 항상 이득인 것은 아니지만,
future exo를 넣는다면 head_flatten 방식은 부적절하고,
token_cross_attn 방식이 훨씬 안정적이다.
```

### 7.2 DSIO / GCS 계열 결과

DSIO 계열 데이터에서는 네 모델이 대체로 비슷한 수준에 수렴했다.

대표 결과:

```text
timexer                    WAPE ≈ 2.338
patchtst_no_future         WAPE ≈ 2.374
exotst                     WAPE ≈ 2.389
patchtst_token_cross_attn  WAPE ≈ 2.441
```

해석:

- 모델 간 차이가 크지 않았다.
- 모든 모델이 GT보다 높은 level로 예측하는 공통 bias가 있었다.
- 이는 future exo fusion 방식보다 데이터 표현, cohort 구성, calibration 문제가 더 큰 병목임을 시사한다.
- token cross-attn이 구조적으로 안정화된 방식이긴 하지만, DSIO에서는 future exo 정보 자체가 성능 개선으로 강하게 이어지지는 않았다.

즉 DSIO 기준 결론은:

```text
PatchTST future exo fusion 구조 문제는 token_cross_attn으로 완화됐지만,
현재 DSIO 문제에서는 모델 구조보다 데이터 상태 표현 / cohort-aware 학습 / calibration이 더 큰 개선 포인트다.
```

### 7.3 직접 AB 테스트 결과

`future_exo_ab_test2.py`에서는 다음 case를 비교하도록 구성했다.

```text
past_o_future_o_head_flatten
past_o_future_o_token_cross_attn
past_o_future_x
past_x_future_o_head_flatten
past_x_future_o_token_cross_attn
past_x_future_x
```

또한 quantile 모델은 기본적으로 끄고, base model만 비교하도록 했다.

```python
RUN_PATCHTST_QUANTILE = False
```

비교 결과를 보기 위해 global summary plot도 추가했다.

생성되는 주요 결과물:

```text
global_summary/fusion_pair_delta.parquet
global_summary/fusion_far_by_batch.parquet
global_summary/fusion_far_by_target_week.parquet
global_summary/fusion_forecast_sum_by_target_week.parquet

global_plots/{plant}_{base_case}_fusion_far_by_batch.png
global_plots/{plant}_{base_case}_fusion_far_by_target_week.png
global_plots/{plant}_{base_case}_fusion_aggregate_by_target_week.png
global_plots/{plant}_fusion_pair_improvement_by_batch.png
```

이 결과물들은 `head_flatten`과 `token_cross_attn`을 같은 plant / case / batch / seed 조건에서 직접 비교하기 위한 것이다.

## 8. 결론

기존 PatchTST future exogenous 방식은 `head_flatten` 방식이었다.

이 방식은 구현이 단순하지만, 다음 문제가 있었다.

- horizon-wise future exo 구조를 잃음
- future exo가 하나의 큰 vector로 head에 주입됨
- item-specific backbone signal을 덮을 위험이 있음
- head에 지나치게 많은 역할이 몰림
- 실제 실험에서 forecast spike / collapse / 큰 성능 저하가 관찰됨

이에 따라 future exogenous를 horizon token으로 유지하고, backbone token과 cross-attention으로 결합하는 `token_cross_attn` 방식을 도입했다.

새 방식은:

- future exo의 시간 구조를 보존하고
- 과거 patch representation과 future exo token을 attention으로 결합하며
- 기존 head_flatten 방식보다 안정적이고
- 실험상 큰 폭의 성능 개선을 보였다.

특히 Walmart sanity check에서는:

```text
head_flatten WAPE ≈ 1.2889
token_cross_attn WAPE ≈ 0.5850
```

으로, token cross-attention 방식이 기존 방식 대비 명확히 우수했다.

다만 DSIO/GCS 계열 데이터에서는 모든 모델이 비슷한 공통 bias를 보였고, 이 경우에는 future exo fusion 구조보다:

- cohort-aware training
- small / erratic / obsolete segment 대응
- lifecycle state representation
- calibration

이 더 중요한 개선 포인트로 보인다.

최종적으로 정리하면:

```text
PatchTST에 future exogenous를 넣는 방식으로 head_flatten은 폐기 또는 비교용으로만 유지하는 것이 적절하다.
실사용 기준 future exogenous fusion은 token_cross_attn 방식이 더 안정적이고 구조적으로 타당하다.
```
