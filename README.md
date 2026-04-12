# ts_forecaster_lib

이 문서는 repository 개발자용 README입니다.
패키지 사용자용 문서는 [README.package.md](README.package.md) 를 기준으로 보면 됩니다.

시계열 forecasting 학습과 추론을 library 형태로 사용할 수 있도록 정리한 프로젝트입니다.
현재 public API의 중심은 `modeling_module` 패키지입니다.

## Package Names

- repo 이름: `ts_forecaster_lib`
- Python import 이름: `modeling_module`
- package metadata 이름: `modeling-module`

즉, 설치 후 실제 코드에서는 아래처럼 import 합니다.

```python
from modeling_module import train, load_predictor, build_dataloader
```

## What This Library Provides

- 공통 training API
- 모델 family 단위 / artifact 단위 학습
- checkpoint 기반 inference API
- one-table exogenous data loading
- dataclass 기반 request object
- training manifest / checkpoint artifact 정리

현재 public API에서 직접 다루는 주요 함수와 타입은 아래입니다.

- `train`
- `load_predictor`
- `predict`
- `build_dataset`
- `build_dataloader`
- `TrainRequest`
- `DataRequest`
- `TrainerConfig`
- `SSLConfig`
- `RuntimeConfig`
- `ArtifactConfig`
- `DataWindowConfig`
- `DataColumnConfig`
- `ExogenousConfig`
- `LoaderConfig`

## Official Public API

현재 library에서 안정적으로 사용하기를 권장하는 표면은 아래입니다.

- 함수: `train`, `load_predictor`, `predict`, `build_dataset`, `build_dataloader`
- request/result 타입: `TrainRequest`, `TrainResult`, `DataRequest`
- nested config 타입:
  `TrainerConfig`, `SSLConfig`, `RuntimeConfig`, `ArtifactConfig`,
  `DataWindowConfig`, `DataColumnConfig`, `ExogenousConfig`, `LoaderConfig`

권장 사용 방식은 dataclass 기반입니다.

- training: `train(TrainRequest(...))`
- data: `build_dataloader(DataRequest(...))`
- inference: `predict(...)` 또는 `load_predictor(...)`

flat dict style도 아직 지원하지만, 호환성 목적에 가깝습니다.

## Public / Private Boundary

현재 구조는 아래 경계를 기준으로 정리하고 있습니다.

- public: `modeling_module`, `modeling_module.api`
- private: `modeling_module._internal`

즉, 사용자 코드는 public API만 직접 사용하고, 내부 엔진은 `_internal` 경계 뒤에 두는 방향입니다.
이 구조를 잡아두면 이후 packaging 단계에서 `_internal` 만 compiled/private artifact로 교체해도
public API 시그니처는 그대로 유지할 수 있습니다.

중요:

- `modeling_module._internal` 은 안정 API가 아닙니다.
- `models`, `training`, `utils`, `data_loader` 역시 구현 디테일로 간주하는 것이 맞습니다.
- 외부 사용 코드는 `from modeling_module import ...` 형태를 기준으로 작성하는 것을 권장합니다.

## Supported Training Targets

현재 registry 기준 canonical model key는 아래와 같습니다.

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

예를 들어:

- `models=["patchtst"]` 는 family default artifact로 확장됩니다.
- `models=["patchtst_base"]` 는 단일 artifact만 학습합니다.

## Current Model Status

아래 표는 현재 `modeling_module` public API 기준으로 실제 학습/체크포인트 복원 경로에 연결된 모델들입니다.

| Family | Canonical Key | 역할 | 예측 타입 | Exogenous | 비고 |
|---|---|---|---|---|---|
| PatchTST | `patchtst_base` | 기본 supervised 모델 | point / distribution | optional | public API의 기본 fallback family |
| PatchTST | `patchtst_quantile` | 분위수 예측 | quantile | optional | `models=["patchtst"]` 에 포함 |
| PatchMixer | `patchmixer_base` | 기본 supervised 모델 | point / distribution | optional | public API 연결됨 |
| PatchMixer | `patchmixer_quantile` | 분위수 예측 | quantile | optional | `models=["patchmixer"]` 에 포함 |
| Titan | `titan_base` | family default Titan variant | point / distribution | optional | `models=["titan"]` 에 포함 |
| Titan | `titan_lmm` | family default Titan variant | point / distribution | optional | `models=["titan"]` 에 포함 |
| Titan | `titan_seq2seq` | family default Titan variant | point / distribution | optional | `models=["titan"]` 에 포함 |
| ExoTST | `exotst_base` | exogenous 중심 모델 | point / distribution | required | past continuous exo + future exo 필요 |

추가 메모:

- `PatchTST` family는 현재 SSL pretrain + finetune (`ssl.mode="full"`) 경로가 가장 잘 연결되어 있습니다.
- `ExoTST`는 `use_exogenous_mode=True` 여야 하고, `past_exo_cont_cols` 와 `future_exo_cont_cols` 가 모두 있어야 안전합니다.
- `models=["titan"]` 는 현재 `titan_base`, `titan_lmm`, `titan_seq2seq` 를 함께 학습합니다.
- artifact key를 직접 주면 family 전체가 아니라 그 모델만 학습됩니다.
  예: `models=["titan_lmm"]`, `models=["titan_seq2seq"]`, `models=["titan_base"]`, `models=["patchmixer_quantile"]`, `models=["patchtst_quantile"]`
- repo 안에는 `NHITS`, `Transformer` 디렉토리도 있지만, 현재 README의 이 섹션은 "public training/inference registry에 연결된 모델" 기준입니다.

## Installation

개발용으로는 editable install이 가장 편합니다.

```bash
git clone <REPO_URL>
cd ts_forecaster_lib
pip install -e .[dev]
```

notebook/manual check까지 같이 쓰려면:

```bash
pip install -e .[notebook]
```

이미 `torch`, CUDA, `sktime`, `gluonts`, `datasets` 등이 깔린 기존 환경에서
wheel만 다시 설치할 때는 dependency 재해결을 피하는 편이 안전합니다.

```bash
pip install --no-deps --force-reinstall /path/to/modeling_module-0.1.1-py3-none-any.whl
```

최소 의존성만 수동 설치하려면:

```bash
pip install -r requirements.core.txt
pip install -r requirements.dev.txt
```

notebook 전용 의존성은 아래 파일로도 설치할 수 있습니다.

```bash
pip install -r requirements.notebook.txt
```

배포 artifact를 직접 만들려면:

```bash
python3 -m build --sdist --wheel
```

오프라인이나 제한된 환경에서는:

```bash
python3 -m build --sdist --wheel --no-isolation
```

## Private Wheel Build

`api` 소스만 공개하고 나머지 internal module은 sourceless compiled artifact로 배포하려면 아래 스크립트를 사용합니다.

```bash
python3 tools/build_private_wheel.py
```

기본 동작:

- 먼저 public wheel을 build
- wheel을 unpack
- `modeling_module/__init__.py` 와 `modeling_module/api/**/*.py` 만 source로 유지
- 나머지 `modeling_module/**/*.py` 는 `.pyc` 로 컴파일 후 source 제거
- `dist/private/` 아래에 private wheel 생성

원하는 public wheel이 이미 있으면 직접 넘길 수도 있습니다.

```bash
python3 tools/build_private_wheel.py --wheel dist/modeling_module-<version>-py3-none-any.whl
```

## Data Expectations

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

## One-Table Exogenous Design

현재 library는 one-table exogenous 경로를 지원합니다.

의미는 아래와 같습니다.

- 과거 구간의 `y`와 `past_exo_*` 를 lookback window에서 slice
- 미래 구간의 `future_exo_cont_cols` 를 horizon window에서 known future covariate로 slice
- `tb_master_exo` 같은 exogenous table에 `y`가 없으면, 외부에서 target table과 join해서 one-table 형태로 맞춘 뒤 학습

예시:

| oper_part_no | demand_dt | demand_qty | weather_index | promo_flag |
|---|---:|---:|---:|---:|
| A | 202401 | 12.0 | 0.20 | 0 |
| A | 202402 | 15.0 | 0.31 | 1 |
| A | 202403 | null | 0.40 | 1 |

이 경우:

- `demand_qty` 는 미래 row에서 비어 있어도 됨
- `promo_flag` 같은 known future covariate는 미래 row까지 채워둘 수 있음

## Quick Start: Endogenous Only

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
        window=DataWindowConfig(
            lookback=52,
            horizon=27,
            freq="weekly",
        ),
        columns=DataColumnConfig(
            id_col="oper_part_no",
            date_col="demand_dt",
            y_col="demand_qty",
        ),
        loader=LoaderConfig(
            batch_size=16,
        ),
    ),
    models=["patchtst_base"],
    trainer=TrainerConfig(
        epochs=5,
        lr=1e-3,
    ),
    runtime=RuntimeConfig(
        device="cpu",
    ),
    artifacts=ArtifactConfig(
        save_dir="./artifacts/endo_only",
        auto_save_dir=False,
    ),
)

result = train(req)
print(result.ckpt_paths)
print(result.manifest_path)
```

## Quick Start: Endogenous + Exogenous

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
        window=DataWindowConfig(
            lookback=52,
            horizon=27,
            freq="weekly",
        ),
        columns=DataColumnConfig(
            id_col="oper_part_no",
            date_col="demand_dt",
            y_col="demand_qty",
        ),
        exogenous=ExogenousConfig(
            use_exogenous_mode=True,
            past_exo_cont_cols=[
                "sin_annual",
                "cos_annual",
                "weather_index",
                "macro_index",
            ],
            future_exo_cont_cols=[
                "sin_annual",
                "cos_annual",
                "weather_index",
                "macro_index",
                "promo_flag",
            ],
        ),
        loader=LoaderConfig(
            batch_size=16,
        ),
    ),
    models=["patchtst_base"],
    trainer=TrainerConfig(
        epochs=5,
        lr=1e-3,
    ),
    runtime=RuntimeConfig(
        device="cpu",
    ),
    artifacts=ArtifactConfig(
        save_dir="./artifacts/endo_plus_exo",
        auto_save_dir=False,
    ),
)

result = train(req)
```

## SSL Training

SSL option은 `SSLConfig` 로 전달합니다.

```python
from modeling_module import SSLConfig

req.ssl = SSLConfig(
    mode="full",
    pretrain_epochs=5,
    mask_ratio=0.3,
)
```

현재 기준으로 `full` / `ssl_only` 경로는 PatchTST family에서 가장 의미가 큽니다.

## Inference

checkpoint만 있으면 predictor를 따로 로드해서 inference 할 수 있습니다.

```python
from modeling_module import load_predictor

predictor = load_predictor(result.primary_ckpt_path, device="cpu")
pred = predictor(batch, horizon=27)

print(pred.keys())
```

위 예시는 single-model training 기준입니다.
multi-model 또는 family training이면 `result.primary_ckpt_path` 가 비어 있을 수 있으므로,
그때는 `result.ckpt_paths["patchtst_base"]` 처럼 명시적으로 checkpoint를 선택하는 것이 맞습니다.

직접 helper를 써도 됩니다.

```python
from modeling_module import predict

pred = predict(result.primary_ckpt_path, batch, device="cpu", horizon=27)
```

## Data Loader Only

학습 전에 loader/batch shape만 보고 싶으면 data API만 따로 사용할 수 있습니다.

```python
from modeling_module import (
    DataRequest,
    DataWindowConfig,
    DataColumnConfig,
    LoaderConfig,
    build_dataset,
    build_dataloader,
)

data_req = DataRequest(
    df=target_df,
    window=DataWindowConfig(lookback=52, horizon=27, freq="weekly"),
    columns=DataColumnConfig(
        id_col="oper_part_no",
        date_col="demand_dt",
        y_col="demand_qty",
    ),
    loader=LoaderConfig(stage="train", batch_size=16),
)

dataset = build_dataset(data_req)
loader = build_dataloader(data_req)
batch = next(iter(loader))
```

## Device Behavior

device를 생략하면 library가 자동으로 usable accelerator를 탐색합니다.

- usable `cuda` 가 있으면 `cuda`
- 아니면 usable `mps`
- 둘 다 아니면 `cpu`

중요한 점:

- 단순히 `torch.cuda.is_available()` 만 보는 것이 아니라, 실제 작은 연산까지 가능한지 probe 후 선택합니다.
- `device="cuda"` 를 명시했는데 현재 환경에서 CUDA runtime이 실제로 unusable 하면 초반에 명시적인 에러를 냅니다.

## Training Outputs

`train(...)` 결과에는 아래 정보가 포함됩니다.

- `results`
- `requested_models`
- `save_dir`
- `ckpt_paths`
- `pretrain_ckpt_paths`
- `manifest_path`
- `primary_ckpt_path`
- `best_ckpt_path`

중요한 계약:

- single-model run이면 `primary_ckpt_path` 와 `best_ckpt_path` 가 채워집니다.
- multi-model / family run이면 이 두 convenience field는 `None` 이고, 실제 결과는 `ckpt_paths` 를 사용해야 합니다.

artifact directory에는 보통 아래가 생성됩니다.

- supervised checkpoint
- optional SSL pretrain checkpoint
- `training_manifest.json`

## Manual Notebook

직접 손으로 확인하기 위한 notebook은 여기 있습니다.

- `src/model_test/library_manual_checks/library_api_manual_check.ipynb`

이 notebook은 현재:

- `tb_master_target` 기반 endogenous-only 테스트
- `tb_master_exo` 기반 one-table exogenous 테스트
- dataclass 기반 `TrainRequest` / `DataRequest` 예제

를 포함합니다.

## Tests

대표 smoke test는 아래로 실행할 수 있습니다.

```bash
python3 -m pytest -q
```

현재 public API 관련 검증은 주로 아래 범위를 포함합니다.

- import smoke
- data API smoke
- checkpoint / infer smoke
- compact dataclass config smoke
- public train validation

## Current Notes

- backward-compatible flat dict style도 아직 동작하지만, 새 코드는 dataclass 스타일을 권장합니다.
- ExoTST는 past continuous exogenous와 future exogenous가 모두 있어야 안전합니다.
- Patch 기반 모델은 frequency별 `patch_len` 이상 `lookback` 이 필요합니다.
