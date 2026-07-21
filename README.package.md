# modeling-module

`modeling_module` 는 시계열 forecasting 학습과 추론을 위한 Python library입니다.

설치 후에는 아래처럼 사용합니다.

```python
from modeling_module import DistributionLoss, forecast, load_predictor, predict, train
```

중요한 이름은 아래와 같습니다.

- package metadata 이름: `modeling-module`
- Python import 이름: `modeling_module`

## What It Provides

- 공통 training API
- 모델 family 단위 / artifact 단위 학습
- checkpoint 기반 inference API
- one-table exogenous data loading
- dataclass 기반 request object

공식 public API는 아래를 기준으로 사용하면 됩니다.

- 함수: `train`, `forecast`, `load_predictor`, `predict`, `build_dataset`, `build_dataloader`
- 타입: `TrainRequest`, `TrainResult`, `ForecastRequest`, `ForecastResult`,
  `ForecastRuntimeConfig`, `DataRequest`, `DistributionLoss`
- nested config:
  `TrainerConfig`, `SSLConfig`, `RuntimeConfig`, `ArtifactConfig`,
  `DataWindowConfig`, `DataColumnConfig`, `ExogenousConfig`, `LoaderConfig`,
  `ArchitectureConfig`, `PatchTSTArchitectureConfig`, `TitanArchitectureConfig`,
  `PatchMixerArchitectureConfig`, `ExoTSTArchitectureConfig`,
  `TimexerArchitectureConfig`

## Installation

Python 3.10 이상이 필요합니다. 새 환경에서는 dependency를 포함한 일반 설치를 권장합니다.

기본 설치:

```bash
pip install modeling-module
```

notebook 환경까지 같이 쓰려면:

```bash
pip install "modeling-module[notebook]"
```

추가 extra는 `[plot]`, `[survival]`, 그리고 이를 합친 `[all]`입니다.

`--no-deps`는 wheel이 요구하는 dependency를 대상 환경이 이미 모두 충족하는 경우에만 사용합니다.
설치 후에는 반드시 `pip check`로 확인합니다.

```bash
pip install --no-deps --force-reinstall /path/to/modeling_module-0.2.0-py3-none-any.whl
pip check
```

즉:

- 새 가상환경: 일반 `pip install ...`
- dependency를 외부에서 고정한 GPU 환경: 요구사항을 먼저 확인한 뒤에만 `--no-deps`

개발 환경이나 editable install은 repository의 개발자 문서를 참고하면 됩니다.

## Supported Models

아래 표는 public registry에 연결된 구현 범위입니다.

| Family request | Canonical artifact | Status | 구현된 학습/checkpoint mode | Continuous exogenous | PatchTST SSL |
|---|---|---|---|---|---|
| `patchtst` | `patchtst_base` | 지원 | point, Normal, StudentT | past/future optional | `full`, `ssl_only` |
| `patchtst` | `patchtst_quantile` | 지원 | q10/q50/q90 | past/future optional | `full`, `ssl_only` |
| `patchmixer` | `patchmixer_base` | 지원 | point, Normal, StudentT | past/future optional | 미지원 |
| `patchmixer` | `patchmixer_quantile` | 지원 | q10/q50/q90 | past/future optional | 미지원 |
| explicit only | `patchmixer_original` | 지원 | point only | 미지원 | 미지원 |
| `titan` | `titan_base` | Deprecated | point, Normal, StudentT | past/future optional | 미지원 |
| `titan` | `titan_lmm` | Deprecated | point, Normal, StudentT | past/future optional | 미지원 |
| `titan` | `titan_seq2seq` | Deprecated | point, Normal, StudentT | past/future optional | 미지원 |
| `exotst` | `exotst_base` | 지원 | point, Normal, StudentT | past/future 모두 필수 | 미지원 |
| `timexer` | `timexer_base` | 지원 | point only | past 필수, future 금지 | 미지원 |

동작 규칙:

- family key는 표에 나열된 canonical artifact로 확장됩니다.
- artifact key를 직접 주면 해당 artifact만 학습합니다.
- `patchmixer_original`은 upstream 비교용 endogenous baseline이므로 `models`에 직접 지정해야 하며,
  `patchmixer` family 요청에는 자동으로 포함되지 않습니다.
- `models`를 생략하거나 빈 목록을 주면 `patchtst` family가 기본값이며
  `patchtst_base`, `patchtst_quantile` 순서로 확장됩니다.
- categorical past exogenous 입력은 현재 모든 public family에서 fail-fast합니다. 먼저 continuous
  feature로 인코딩해야 합니다.
- public distribution checkpoint가 지원하는 분포는 `Normal`, `StudentT`입니다.
  `Poisson`, `Bernoulli`, `NegativeBinomial`, `Tweedie`는 data materialization 전에 거부합니다.
- Titan은 신규 검증 matrix와 DSIO default에서 제외합니다. Registry key, 직접 학습과 지원
  checkpoint load는 deprecation 기간 동안 유지하며 public 학습 요청은 `FutureWarning`을 냅니다.
- point/distribution checkpoint prediction은 현재 `{"point": ...}`를 반환합니다. Quantile
  checkpoint는 `q10`, `q50`, `q90`과 `point=q50`을 반환합니다.
- `DistributionLoss`는 top-level stable selector입니다. Public training/checkpoint 계약은
  `Normal`, `StudentT`만 지원합니다.
- `ssl.mode="full"`과 `ssl.mode="ssl_only"`는 request에 PatchTST artifact와 artifact `save_dir`가
  필요합니다. Mixed request에서는 SSL이 PatchTST에만 적용되고 다른 family는 supervised로
  실행됩니다. `ssl_only`는 PatchTST supervised checkpoint 없이 pretraining checkpoint만 만듭니다.

Distribution 학습은 다음처럼 선택합니다.

```python
from modeling_module import DistributionLoss, TrainerConfig

trainer = TrainerConfig(
    loss=DistributionLoss(distribution="Normal"),
)
```

## Checkpoint Compatibility

- 현재 포맷은 `modeling_module.ckpt.v3`입니다. Output mode, distribution, parameter order,
  output multiplier와 loss 설정을 함께 저장하며 현재 artifact 검증에는 `strict=True`를 권장합니다.
- `Normal`/`StudentT`는 저장 전후 head·loss·parameter names·state shape가 동일해야 합니다.
- 실제 fixture 기준 legacy v2 PatchTST/PatchMixer distribution artifact만 구조적으로 복원되며,
  유실된 historical loss option은 warning과 함께 기본값을 사용합니다. 복원 즉시 v3로 다시 저장하는
  것을 권장합니다.
- legacy v1과 Titan/ExoTST legacy distribution fixture는 필요한 identity/구조를 안전하게 추론할
  수 없어 거부합니다. 저장된 head와 metadata가 충돌하면 `strict=False`에서도 point model로 부분
  복원하지 않고 fail-closed합니다.

## Data Format

기본 입력은 long-table 형태입니다.

필수 컬럼:

- id column
- date column
- target column

권장 date 정수 포맷:

- `weekly`: `YYYYWW`
- `monthly`: `YYYYMM`
- `daily`: `YYYYMMDD`
- `hourly`: `YYYYMMDDHH`

## One-Table Exogenous

exogenous 사용 시에도 한 테이블로 학습할 수 있습니다.

- 과거 구간에서는 `y` 와 `past_exo_*` 가 사용됩니다.
- 미래 구간에서는 `future_exo_cont_cols` 가 known future covariate로 사용됩니다.
- 미래 row에서는 `y` 가 `null` 이어도 됩니다.
- future-exogenous width가 설정된 predictor에는 batch 공용 `(horizon, width)` 또는
  `(batch, horizon, width)`를 전달할 수 있고 `future_exo_cb`도 지원합니다. 누락·unexpected input과
  batch·horizon·width 오류는 model forward 전에 거부합니다.
- categorical exogenous column은 현재 public training 계약에서 지원하지 않습니다.

예시:

| oper_part_no | demand_dt | demand_qty | weather_index | promo_flag |
|---|---:|---:|---:|---:|
| A | 202401 | 12.0 | 0.20 | 0 |
| A | 202402 | 15.0 | 0.31 | 1 |
| A | 202403 | null | 0.40 | 1 |

## Quick Start

### Endogenous Only

```python
from modeling_module import (
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    LoaderConfig,
    RuntimeConfig,
    TrainRequest,
    TrainerConfig,
    train,
)

req = TrainRequest(
    data=DataRequest(
        df=target_df,
        window=DataWindowConfig(lookback=52, horizon=27, freq="weekly"),
        columns=DataColumnConfig(
            id_col="oper_part_no",
            date_col="demand_dt",
            y_col="demand_qty",
        ),
        loader=LoaderConfig(batch_size=16),
    ),
    models=["patchtst_base"],
    trainer=TrainerConfig(epochs=5, lr=1e-3),
    runtime=RuntimeConfig(device="cpu"),
    artifacts=ArtifactConfig(
        save_dir="./artifacts/endo_only",
        auto_save_dir=False,
    ),
)

result = train(req)
print(result.ckpt_paths)
```

### Model Architecture Override

```python
from modeling_module import (
    ArchitectureConfig,
    PatchMixerArchitectureConfig,
    PatchTSTArchitectureConfig,
)

req = TrainRequest(
    data=DataRequest(
        df=target_df,
        window=DataWindowConfig(lookback=104, horizon=27, freq="weekly"),
    ),
    models=["patchtst_base", "patchmixer_base"],
    architecture=ArchitectureConfig(
        patchtst=PatchTSTArchitectureConfig(
            patch_len=13,
            stride=6,
            d_model=384,
            n_layers=5,
        ),
        patchmixer=PatchMixerArchitectureConfig(
            patch_len=13,
            stride=6,
            d_model=192,
            e_layers=6,
        ),
    ),
    trainer=TrainerConfig(warmup_epochs=3, spike_epochs=2, base_lr=1e-3),
)
```

### Endogenous + Exogenous

```python
from modeling_module import (
    ArtifactConfig,
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExogenousConfig,
    LoaderConfig,
    RuntimeConfig,
    TrainRequest,
    TrainerConfig,
    train,
)

req = TrainRequest(
    data=DataRequest(
        df=exo_one_table_df,
        window=DataWindowConfig(lookback=52, horizon=27, freq="weekly"),
        columns=DataColumnConfig(
            id_col="oper_part_no",
            date_col="demand_dt",
            y_col="demand_qty",
        ),
        exogenous=ExogenousConfig(
            use_exogenous_mode=True,
            past_exo_cont_cols=["weather_index", "macro_index"],
            future_exo_cont_cols=["weather_index", "macro_index", "promo_flag"],
        ),
        loader=LoaderConfig(batch_size=16),
    ),
    models=["patchtst_base"],
    trainer=TrainerConfig(epochs=5, lr=1e-3),
    runtime=RuntimeConfig(device="cpu"),
    artifacts=ArtifactConfig(
        save_dir="./artifacts/endo_plus_exo",
        auto_save_dir=False,
    ),
)

result = train(req)
```

## Inference

Part/series 단위 anchored inference에는 결과를 반환하는 공개 `forecast()` API를 사용합니다.
Weekly origin은 ISO W0, monthly origin은 M0이며 `horizon_step=0`이 origin 자체입니다.
상세한 request, 결과 schema, series 선택 정책은 repository의
`docs/public_forecast_api.md`를 참고합니다.

```python
from modeling_module import (
    DataColumnConfig,
    DataRequest,
    DataWindowConfig,
    ExogenousConfig,
    ForecastRequest,
    ForecastRuntimeConfig,
    forecast,
)

result = forecast(
    ForecastRequest(
        checkpoint_path="./checkpoints/patchtst.pt",
        expected_model_key="patchtst_base",
        data=DataRequest(
            df=source_df,
            backend="exo",
            window=DataWindowConfig(lookback=52, horizon=27, freq="weekly"),
            columns=DataColumnConfig(id_col="series", date_col="period", y_col="target"),
            exogenous=ExogenousConfig(fill_missing="zero"),
        ),
        series_ids=["series-a", "series-b"],
        forecast_origin=202601,
        runtime=ForecastRuntimeConfig(batch_size=32, device="cpu"),
    )
)

predictions = result.predictions
```

`forecast()`는 파일을 쓰지 않습니다. Database 접근, `.env`, checkpoint/data 경로 결정,
Parquet 저장 정책은 Consumer가 소유합니다.

checkpoint를 재사용할 계획이면 predictor를 먼저 로드하는 것이 좋습니다.

```python
from modeling_module import load_predictor

predictor = load_predictor(result.primary_ckpt_path, device="cpu")
pred = predictor(batch, horizon=27)
```

한 번만 예측하면 helper도 사용할 수 있습니다.

```python
from modeling_module import predict

pred = predict(result.primary_ckpt_path, batch, device="cpu", horizon=27)
```

최종 supervised checkpoint가 정확히 하나 생성된 경우에만 `primary_ckpt_path`가 채워집니다.
그 외에는 `ckpt_paths`에서 선택하고, SSL-only 결과는 `pretrain_ckpt_paths`를 사용합니다.

## Notes

- 새 코드는 flat dict보다 dataclass 스타일을 권장합니다.
- `ExoTST`는 past/future continuous exogenous가 모두 필요합니다.
- `TimeXer` v1은 past continuous exogenous만 사용하며 future/categorical exogenous와
  quantile/distribution output을 거부합니다.
- Patch 기반 모델은 frequency별 `patch_len` 이상 `lookback` 이 필요합니다.
- `ssl.mode="full"`/`"ssl_only"`는 request에 PatchTST target과 artifact directory가 필요하며,
  mixed request에서는 PatchTST stage에만 적용됩니다.
- 일부 배포 환경에서는 internal module이 sourceless compiled artifact 형태로 제공될 수 있습니다.
