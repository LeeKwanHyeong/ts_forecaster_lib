# Public Anchored Forecast API

`modeling-module` 0.2.0부터 Consumer는 concrete DataModule이나 model builder를 직접
조합하지 않고 `modeling_module` 또는 `modeling_module.api`의 공개 API만으로 anchored
inference를 실행할 수 있습니다.

동결된 machine-readable contract는
[`contracts/public_forecast_contract.v1.json`](contracts/public_forecast_contract.v1.json)이며,
contract identity는 다음과 같습니다.

- Contract ID: `modeling-module.public-anchored-forecast`
- Contract version: `1.0.0`
- Contract SHA-256: `07e8d2d825929bd9882d413c32faf76108b3f5e0d147d6a628575e0ebda563bd`

## Public surface

아래 두 import surface는 동일한 API를 제공합니다.

```python
from modeling_module import ForecastRequest, ForecastResult, ForecastRuntimeConfig, forecast
from modeling_module.api import ForecastRequest, ForecastResult, ForecastRuntimeConfig, forecast
```

최종 signature는 다음과 같습니다.

```python
ForecastRuntimeConfig(
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
)

ForecastRequest(
    checkpoint_path: str | Path,
    expected_model_key: str | None,
    data: DataRequest,
    series_ids: Sequence[str] | None,
    forecast_origin: date | datetime | int,
    runtime: ForecastRuntimeConfig = <factory>,
    unknown_series_policy: Literal["error", "ignore"] = "error",
)

ForecastResult(
    predictions: pl.DataFrame,
    model_key: str,
    forecast_origin: int,
)

forecast(request: ForecastRequest) -> ForecastResult
```

`DataRequest`가 lookback, horizon, frequency, input column, missing-value policy와
exogenous configuration을 소유합니다. `ForecastRuntimeConfig`는 inference 실행 시의 batch,
worker와 device 설정만 소유합니다.

## Example

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

request = ForecastRequest(
    checkpoint_path="./checkpoints/patchtst_base.pt",
    expected_model_key="patchtst_base",
    data=DataRequest(
        df=source_df,
        backend="exo",
        window=DataWindowConfig(lookback=52, horizon=27, freq="weekly"),
        columns=DataColumnConfig(
            id_col="series",
            date_col="period",
            y_col="target",
        ),
        exogenous=ExogenousConfig(
            use_exogenous_mode=True,
            past_exo_cont_cols=["past_feature"],
            future_exo_cont_cols=["known_future_feature"],
            fill_missing="zero",
        ),
    ),
    series_ids=["series-a", "series-b"],
    forecast_origin=202601,
    runtime=ForecastRuntimeConfig(
        batch_size=32,
        num_workers=0,
        device="cpu",
        pin_memory=False,
    ),
)

result = forecast(request)
predictions = result.predictions
```

기존 `fill_missing="ffill"` 기본값은 바뀌지 않았습니다. Zero fill이 필요한 Consumer는
예시처럼 `fill_missing="zero"`를 명시해야 합니다.

## Ordered result schema

`ForecastResult.predictions`는 아래 순서와 dtype을 갖는 Polars DataFrame입니다.

| 순서 | Column | Polars dtype | Nullable | 의미 |
|---:|---|---|---|---|
| 1 | `series_id` | `pl.String` | No | 범용 series 식별자 |
| 2 | `model_key` | `pl.String` | No | checkpoint에서 검증한 artifact key |
| 3 | `forecast_origin` | `pl.Int64` | No | frequency별 canonical origin |
| 4 | `horizon_step` | `pl.Int32` | No | origin부터 시작하는 0-based step |
| 5 | `point` | `pl.Float64` | No | point forecast |
| 6 | `q10` | `pl.Float64` | Yes | 10th percentile |
| 7 | `q50` | `pl.Float64` | Yes | median |
| 8 | `q90` | `pl.Float64` | Yes | 90th percentile |

Point-only model도 quantile column을 유지하며 그 값은 null입니다. Quantile model이 별도의
point output을 제공하지 않으면 `q50`을 `point`로 사용합니다. 결과는 resolved series 순서,
그다음 `horizon_step` 순으로 정렬되며 `batch_size`가 달라도 row identity와 순서는 같습니다.

## Temporal semantics

| Frequency | Canonical key | 허용 입력 |
|---|---|---|
| `weekly` | ISO `YYYYWW` | `pl.Date`, `pl.Datetime`, valid `YYYYWW` integer |
| `monthly` | `YYYYMM` | `pl.Date`, `pl.Datetime`, valid `YYYYMM` integer |
| `daily` | `YYYYMMDD` | `pl.Date`, `pl.Datetime`, valid `YYYYMMDD` integer |
| `hourly` | `YYYYMMDDHH` | `pl.Datetime`, valid `YYYYMMDDHH` integer |

- Weekly year는 Gregorian year가 아니라 ISO week-year입니다.
- ISO Week 53은 실제로 존재하는 ISO year에서만 유효합니다.
- Weekly origin은 W0, monthly origin은 M0입니다.
- `horizon_step=0`은 origin 자체이고 첫 예측 기간입니다.
- Lookback L은 origin 직전의 canonical L개 기간이며 오래된 기간부터 정렬됩니다.
- timezone-aware datetime은 먼저 UTC로 변환하지 않고 표현된 timezone의 calendar period를
  사용합니다.

## Series selection

- `series_ids=None`: 사용 가능한 모든 series를 canonical string 순서로 선택합니다.
- 빈 sequence: `ValueError`를 발생시킵니다.
- 중복 ID: 첫 등장을 보존해 de-duplicate합니다.
- 명시적 selection: request 순서를 보존합니다.
- unknown ID: 기본 `unknown_series_policy="error"`에서 `ValueError`를 발생시킵니다.
- `unknown_series_policy="ignore"`: unknown ID를 제외하지만 known ID가 하나도 남지 않으면
  `ValueError`를 발생시킵니다.

## Compatibility and ownership

- `load_predictor()`, `predict()`, `build_dataset()`, `build_dataloader()`는 그대로 유지됩니다.
- `modeling_module.data_loader.MultiPartExoDataModule`은 lowercase authority를 re-export하는
  legacy compatibility module입니다. 신규 Consumer가 사용할 공개 경로는 아닙니다.
- `forecast_to_parquet()`는 stable public surface가 아닙니다. 신규 integration은 `forecast()`의
  반환값을 Consumer의 저장 정책에 따라 직접 저장해야 합니다.
- `forecast()`는 canonical exogenous data backend만 지원하며 파일을 쓰지 않습니다.
- Database, `.env`, Consumer path, Parquet naming/partition/retention은 Consumer 책임입니다.

DSIODemandEngine에서는 다음 private imports를 제거해야 합니다.

- `modeling_module.data_loader.multi_part_exo_data_module.MultiPartExoDataModule`
- `modeling_module.models`와 `modeling_module.models.model_builder`의 model builder
- `modeling_module.training.forecater.forecast_to_parquet`
- `modeling_module.utils.checkpoint.load_model_dict`
- `modeling_module.utils.exogenous_utils.compose_exo_calendar_cb`

대신 `ForecastRequest`, `ForecastRuntimeConfig`, `DataRequest`, `DataWindowConfig`,
`DataColumnConfig`, `ExogenousConfig`, `forecast`만 public surface에서 import합니다.

## Known limitations

- High-level `forecast()`는 contract 우회를 막기 위해 legacy `backend="simple"`을 거부합니다.
- 공개 quantile schema는 현재 `q10`, `q50`, `q90`으로 고정되어 있습니다.
- Checkpoint가 생성된 model configuration과 request의 lookback/horizon/exogenous shape은 서로
  호환되어야 합니다.
- `forecast()`는 persistence나 여러 checkpoint의 orchestration을 수행하지 않습니다.
