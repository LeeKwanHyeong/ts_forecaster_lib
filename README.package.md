# modeling-module

`modeling_module` 는 시계열 forecasting 학습과 추론을 위한 Python library입니다.

설치 후에는 아래처럼 사용합니다.

```python
from modeling_module import train, load_predictor, predict, build_dataloader
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

- 함수: `train`, `load_predictor`, `predict`, `build_dataset`, `build_dataloader`
- 타입: `TrainRequest`, `TrainResult`, `DataRequest`
- nested config:
  `TrainerConfig`, `SSLConfig`, `RuntimeConfig`, `ArtifactConfig`,
  `DataWindowConfig`, `DataColumnConfig`, `ExogenousConfig`, `LoaderConfig`

## Installation

기본 설치:

```bash
pip install modeling-module
```

notebook 환경까지 같이 쓰려면:

```bash
pip install "modeling-module[notebook]"
```

개발 환경이나 editable install은 repository의 개발자 문서를 참고하면 됩니다.

## Supported Models

현재 public registry 기준 canonical model key는 아래와 같습니다.

- `patchtst_base`
- `patchtst_quantile`
- `patchmixer_base`
- `patchmixer_quantile`
- `titan_base`
- `titan_lmm`
- `titan_seq2seq`
- `exotst_base`

family 이름으로도 요청할 수 있습니다.

- `patchtst`
- `patchmixer`
- `titan`
- `exotst`

동작 규칙:

- `models=["patchtst"]` 처럼 family를 주면 해당 family의 기본 artifact들이 학습됩니다.
- `models=["patchtst_quantile"]` 처럼 artifact key를 직접 주면 그 모델만 학습됩니다.
- `models=["titan"]` 는 현재 `titan_base`, `titan_lmm`, `titan_seq2seq` 를 함께 학습합니다.

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

single-model run이면 `primary_ckpt_path` 가 채워집니다.
family run이나 multi-model run이면 `ckpt_paths` 에서 원하는 모델 checkpoint를 선택해야 합니다.

## Notes

- 새 코드는 flat dict보다 dataclass 스타일을 권장합니다.
- `ExoTST`는 `past_exo_cont_cols` 와 `future_exo_cont_cols` 가 모두 필요합니다.
- Patch 기반 모델은 frequency별 `patch_len` 이상 `lookback` 이 필요합니다.
- `ssl.mode="full"` 경로는 현재 PatchTST family에서 가장 의미가 큽니다.
- 일부 배포 환경에서는 internal module이 sourceless compiled artifact 형태로 제공될 수 있습니다.
