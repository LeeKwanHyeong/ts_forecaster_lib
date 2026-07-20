# PatchTST implementation notes

이 문서는 현재 repository의 PatchTST 구현 계약을 설명합니다. 사용자에게 노출되는 최종 지원
범위는 repository root의 `README.md`와 `README.package.md` 표를 기준으로 합니다.

## Registered artifacts

| Artifact key | Output mode | Checkpoint-safe loss |
|---|---|---|
| `patchtst_base` | point 또는 distribution | point loss, `Normal`, `StudentT` |
| `patchtst_quantile` | q10/q50/q90 | quantile loss |

Base와 quantile은 별도 artifact입니다. Distribution checkpoint를 public predictor로 불러오면 현재
예측 API는 location을 `{"point": ...}`로 반환하며 sample/interval을 만들지는 않습니다.

## Data contract

- target input: `(B, lookback, 1)`
- past continuous exogenous: 선택 사항
- future continuous exogenous: 선택 사항
- categorical exogenous: public training과 prediction API에서 fail-fast

학습 시 `future_exo_dim > 0`으로 구성했다면 public prediction에서 batch 공용 `(horizon,
future_exo_dim)` 또는 `(B, horizon, future_exo_dim)`을 전달해야 합니다. 누락,
batch/horizon/width 불일치, 또는 width 0 모델에 non-empty future tensor를 전달하는 경우 모두
즉시 실패합니다.

내부 모듈에 categorical tensor를 받을 수 있는 인자가 남아 있어도 public 지원을 뜻하지 않습니다.

## Current data flow

1. target을 RevIN으로 정규화합니다.
2. target과 구성된 past continuous feature를 patch token으로 만듭니다.
3. PatchTST backbone이 patch sequence를 인코딩합니다.
4. `future_exo_dim > 0`이면 `FutureExoTokenFusion`이 horizon별 future token을 만들고,
   backbone token이 이를 cross-attention으로 읽습니다.
5. base 또는 quantile head를 거쳐 target scale로 복원합니다.

Future fusion은 future width가 설정된 경우에만 존재합니다. Future exogenous를 사용하지 않는
모델은 이 경로를 만들지 않습니다.

## SSL contract

`SSLConfig.mode`는 `sl_only`, `ssl_only`, `full`을 지원하며 `off`는 `sl_only` alias입니다.

- `sl_only`: supervised training만 실행
- `ssl_only`: PatchTST y-only pretraining 후 supervised artifact 없이 종료
- `full`: y-only pretraining 후 선택한 PatchTST base/quantile artifact를 fine-tune

`ssl_only`와 `full`은 request에 PatchTST artifact가 최소 하나 있어야 하고 artifact `save_dir`도
필수입니다. PatchMixer, Titan, ExoTST, TimeXer-only request에는 사용할 수 없습니다. SSL
pretraining에서는 exogenous dimension을 0으로 두고 target history만 사용합니다.

## Code map

- [configuration](common/configs.py)
- [supervised models and future fusion](supervised/PatchTST.py)
- [backbone](supervised/backbone.py)
- [trainer integration](../../training/model_trainers/patchtst_train.py)

지원 경계는 registry, public validation, checkpoint restore, future-exogenous sensitivity 회귀 테스트로
고정합니다.
