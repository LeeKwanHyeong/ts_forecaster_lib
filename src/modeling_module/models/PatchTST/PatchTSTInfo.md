# PatchTST Implementation Notes

> 기반 논문: *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*  
> 이 문서는 현재 레포의 PatchTST 구현이 원 논문에서 어떻게 확장되었는지, 특히 외생변수를 어떤 경로로 주입하는지 설명합니다.

## 요약

현재 구현의 핵심은 아래 두 줄입니다.

- `past exogenous`는 패치 임베딩 단계에서 target patch와 함께 backbone 입력으로 들어갑니다.
- `future exogenous`는 horizon token 시퀀스로 투영된 뒤, backbone output token이 cross-attention으로 받아들입니다.

즉, 미래 외생변수를 예측 head에서 한 번에 납작하게(flatten) 눌러 쓰지 않습니다.  
PatchTST는 이제 `future exo -> token sequence -> cross attention` 경로를 기본 경로로 사용합니다.

## 전체 흐름

| 단계 | 역할 | 대표 shape |
| :--- | :--- | :--- |
| Input | target, past exo, future exo 입력 | `x: (B, L, C)` / `future_exo: (B, H, E)` |
| RevIN | target 정규화 | `(B, L, C)` |
| Patchify | target + past exo를 patch token으로 변환 | `(B, N, D)` |
| Backbone | self-attention encoder | `(B, N, D)` |
| Future Fusion | future exo token과 cross-attention 결합 | `(B, N, D)` |
| Head | point / quantile / distribution output | `(B, H)` 또는 `(B, H, Q)` |
| RevIN Denorm | target scale 복원 | output |

## Past Exogenous

과거 외생변수는 backbone 앞단에서 함께 패치화됩니다.

- `past_exo_cont`
- `past_exo_cat`

이 값들은 target patch와 concat된 뒤 `d_model` 차원으로 projection됩니다.  
즉 PatchTST backbone은 이미 과거 외생 신호를 포함한 token sequence를 인코딩합니다.

의미적으로는:

```text
patch_token = proj([target_patch, past_cont_patch, past_cat_emb_patch])
```

## Future Exogenous

미래 외생변수는 `FutureExoTokenFusion` 모듈에서 처리됩니다.

흐름은 다음과 같습니다.

1. `future_exo: (B, H, E)`를 horizon token sequence로 투영
2. learnable future positional embedding 추가
3. backbone token을 query로, future token을 key/value로 사용
4. residual + FFN으로 안정적으로 결합

개념적으로는:

```text
future_tokens = Linear(future_exo) + future_pos
z = CrossAttention(query=backbone_tokens, key=future_tokens, value=future_tokens)
```

이 방식의 장점:

- horizon별 future exo 영향이 분리됩니다
- part-specific backbone signal이 유지됩니다
- future exo가 공통 패턴일 때도 head가 쉽게 collapse하지 않습니다

## Head

future exo는 head에서 직접 다시 받지 않습니다.  
head는 미래 외생 신호가 이미 fusion된 backbone token만 사용합니다.

지원 head:

- `PointHeadWithExo`
- `QuantileHeadWithExo`
- `DistHeadWithExo`

현재는 head의 `d_future`를 `0`으로 두고, output head는 순수하게 fused token representation만 읽습니다.

## Point / Quantile / Distribution

### Point model

- output: `(B, H)`
- loss 예시: `MAE`, `MSE`, `Huber`

### Quantile model

- output: `(B, H, Q)`
- `monotonic_quantiles=True`일 때 분위수 정렬

### Distribution model

- output: `(B, H, P)`
- location / scale 계열 파라미터를 출력

## 현재 구현의 실무적 의미

이 구현은 아래 상황에서 유리합니다.

- 미래 calendar / promo / weather feature가 horizon별로 다르게 작동할 때
- part별 과거 패턴과 future-known feature를 함께 써야 할 때
- 미래 외생변수를 단일 벡터로 압축했을 때 발생하는 불안정성을 피하고 싶을 때

반대로, future exo가 거의 공통 패턴만 제공하고 개별 품목 차이를 충분히 못 만들면:

- `PatchTST + no_future`
- 또는 `ExoTST`

가 더 적절할 수 있습니다.  
즉 future exo를 “넣을 수 있느냐”보다 “미래 신호가 item-specific 정보로 충분한가”가 더 중요합니다.

## 코드 위치

- config: [configs.py](/Users/igwanhyeong/PycharmProjects/ts_forecaster_lib/src/modeling_module/models/PatchTST/common/configs.py)
- supervised model: [PatchTST.py](/Users/igwanhyeong/PycharmProjects/ts_forecaster_lib/src/modeling_module/models/PatchTST/supervised/PatchTST.py)
- backbone: [backbone.py](/Users/igwanhyeong/PycharmProjects/ts_forecaster_lib/src/modeling_module/models/PatchTST/supervised/backbone.py)
- trainer glue: [patchtst_train.py](/Users/igwanhyeong/PycharmProjects/ts_forecaster_lib/src/modeling_module/training/model_trainers/patchtst_train.py)

## 권장 해석

현재 레포 기준으로는 PatchTST를 이렇게 이해하는 게 가장 정확합니다.

- PatchTST는 `past-only`로도 강한 baseline이 된다
- future exo가 유효하면 token cross-attention 경로로 추가 이득을 볼 수 있다
- future exo 품질이 낮거나 공통 패턴 위주면 `no_future`가 더 안정적일 수 있다
