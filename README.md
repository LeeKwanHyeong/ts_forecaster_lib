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
from modeling_module import DistributionLoss, forecast, load_predictor, train
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
- `forecast`
- `load_predictor`
- `predict`
- `build_dataset`
- `build_dataloader`
- `TrainRequest`
- `ForecastRequest`
- `ForecastResult`
- `ForecastRuntimeConfig`
- `DataRequest`
- `TrainerConfig`
- `DistributionLoss`
- `SSLConfig`
- `RuntimeConfig`
- `ArtifactConfig`
- `DataWindowConfig`
- `DataColumnConfig`
- `ExogenousConfig`
- `LoaderConfig`

## Official Public API

현재 library에서 안정적으로 사용하기를 권장하는 표면은 아래입니다.

- 함수: `train`, `forecast`, `load_predictor`, `predict`, `build_dataset`, `build_dataloader`
- request/result 타입: `TrainRequest`, `TrainResult`, `ForecastRequest`, `ForecastResult`,
  `ForecastRuntimeConfig`, `DataRequest`
- nested config 타입:
  `TrainerConfig`, `SSLConfig`, `RuntimeConfig`, `ArtifactConfig`,
  `DataWindowConfig`, `DataColumnConfig`, `ExogenousConfig`, `LoaderConfig`,
  `ArchitectureConfig`, `PatchTSTArchitectureConfig`, `PatchMixerArchitectureConfig`,
  `TitanArchitectureConfig`, `ExoTSTArchitectureConfig`, `NHITSArchitectureConfig`,
  `TimexerArchitectureConfig`
- loss selector: `DistributionLoss` (`Normal`, `StudentT`)

권장 사용 방식은 dataclass 기반입니다.

- training: `train(TrainRequest(...))`
- data: `build_dataloader(DataRequest(...))`
- anchored inference: `forecast(ForecastRequest(...))`
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
- `patchmixer`
- `patchmixer_exo`
- `titan_base`
- `titan_lmm`
- `titan_seq2seq`
- `exotst_base`
- `nhits_base`
- `timexer_base`

`titan_base`, `titan_lmm`, `titan_seq2seq`는 deprecation 기간의 학습/checkpoint 호환을 위해
registry에 남아 있습니다. 신규 운영 학습 대상으로는 권장하지 않습니다.

family 이름으로도 요청할 수 있습니다.

- `patchtst`
- `patchmixer`
- `titan`
- `exotst`
- `nhits`
- `timexer`

예를 들어:

- `models=["patchtst"]` 는 family default artifact로 확장됩니다.
- `models=["patchtst_base"]` 는 단일 artifact만 학습합니다.

## Current Model Status

아래 표는 public registry에 연결된 구현 범위입니다.

| Family | Canonical Key | Status | 구현된 학습/checkpoint mode | Continuous exogenous | SSL |
|---|---|---|---|---|---|
| PatchTST | `patchtst_base` | 지원 | point, Normal, StudentT | endogenous 기본, legacy exogenous 호환 | `full`, `ssl_only` |
| PatchTST | `patchtst_exogenous` | 지원 | point, Normal, StudentT | past/future 중 하나 이상 필수 | `full`, `ssl_only` |
| PatchTST | `patchtst_quantile` | 지원 | q10/q50/q90 | endogenous 기본, legacy exogenous 호환 | `full`, `ssl_only` |
| PatchTST | `patchtst_quantile_exogenous` | 지원 | q10/q50/q90 | past/future 중 하나 이상 필수 | `full`, `ssl_only` |
| PatchMixer | `patchmixer` | 지원 | point only | 미지원 | 미지원 |
| PatchMixer | `patchmixer_exo` | 지원 | point only | past/future 중 하나 이상 필수 | 미지원 |
| Titan | `titan_base` | Deprecated | point, Normal, StudentT | past/future optional | 미지원 |
| Titan | `titan_lmm` | Deprecated | point, Normal, StudentT | past/future optional | 미지원 |
| Titan | `titan_seq2seq` | Deprecated | point, Normal, StudentT | past/future optional | 미지원 |
| ExoTST | `exotst_base` | 지원 | point, Normal, StudentT | past/future 모두 필수 | 미지원 |
| N-HiTS | `nhits_base` | 지원 | point only | 미지원 | 미지원 |
| TimeXer | `timexer_base` | 지원 | point only | past 필수, future 금지 | 미지원 |

추가 메모:

- `PatchTST`의 `full`/`ssl_only`는 artifact `save_dir`가 필수이며 다른 family-only request에는 사용할 수 없습니다.
- 신규 exogenous 학습은 `patchtst_exogenous`, `patchtst_quantile_exogenous`,
  `patchmixer_exo`를 직접 요청합니다. 이 키들은 기존 `patchtst`/`patchmixer` family 기본
  확장에는 포함되지 않습니다.
- `patchmixer`는 논문 기반 endogenous point 모델입니다. `patchmixer_original`은 같은 모델의
  legacy alias이며, 과거 `patchmixer_base`와 quantile key는 지원 schema의 checkpoint
  복원에만 사용하는 load-only key입니다.
- mixed request에서는 SSL이 PatchTST stage에만 적용되고 다른 family는 supervised로 실행됩니다.
  `ssl_only`는 PatchTST supervised checkpoint 없이 pretraining checkpoint만 만듭니다.
- `ExoTST`는 `use_exogenous_mode=True`와 past/future continuous exogenous가 모두 필요합니다.
- `N-HiTS` public artifact는 single-target endogenous point 전용이며 exogenous,
  distribution, quantile output을 거부합니다.
- `TimeXer` v1은 past continuous exogenous만 사용하며 future/categorical exogenous와
  quantile/distribution output을 거부합니다.
- categorical past exogenous는 현재 모든 public family에서 fail-fast합니다.
- checkpoint-safe distribution은 `Normal`, `StudentT`입니다. `Poisson`, `Bernoulli`,
  `NegativeBinomial`, `Tweedie`는 data materialization 전에 거부합니다.
- Titan은 신규 검증 matrix와 DSIO default에서 제외합니다. 기존 regression은 deprecation 기간의
  지원 checkpoint 호환을 깨지 않기 위한 안전망으로만 유지합니다.
- point/distribution predictor는 현재 location을 `point`로 반환하며, quantile predictor는
  `q10`, `q50`, `q90`과 `point=q50`을 반환합니다.
- `DistributionLoss(distribution="Normal")` 또는 `DistributionLoss(distribution="StudentT")`를
  top-level public selector로 사용합니다. 다른 분포는 data materialization 전에 거부합니다.
- `models` 생략 또는 빈 목록은 `patchtst_base`, `patchtst_quantile`로 확장됩니다.
- `models=["titan"]`은 호환상 세 Titan artifact로 계속 확장되지만 `FutureWarning`을 냅니다.
- artifact key를 직접 주면 family 전체가 아니라 그 모델만 학습됩니다.
  예: `models=["titan_lmm"]`, `models=["titan_seq2seq"]`, `models=["titan_base"]`, `models=["patchmixer_exo"]`, `models=["patchtst_quantile"]`
- repo 안의 `Transformer` 디렉토리는 아직 public training/inference registry에 연결되지 않았습니다.

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
pip install --no-deps --force-reinstall /path/to/modeling_module-0.2.0-py3-none-any.whl
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

- 현재 `src/modeling_module` 만 clean staging한 뒤 public wheel을 build
- private distribution profile은 `non-sellm`이며 SELLM model, trainer, public export,
  registry entry와 LLM extra dependency metadata를 제외
- wheel을 unpack
- `modeling_module/__init__.py` 와 `modeling_module/api/**/*.py` 만 source로 유지
- 나머지 `modeling_module/**/*.py` 는 `.pyc` 로 컴파일 후 source 제거
- `dist/private/` 아래에 private wheel 생성
- 빈 임시 venv에 `--no-deps --no-index` 로 wheel만 설치해 metadata와 설치 경로를 검사
- repository 밖의 격리 모드 Python에서 그 venv의 wheel과 기존 ML 의존성을 사용해 public import와
  `build_dataset` smoke를 실행

일반 source checkout과 `python3 -m build`로 생성한 public wheel의 SELLM 지원은 유지됩니다.
`tools/build_private_wheel.py`로 생성하는 core private wheel만 non-SELLM 경계를 적용합니다.

Private wheel filename/ABI 정책은 다음과 같습니다.

- 기본 filename: `modeling_module-<version>-1private-cp<major><minor>-none-any.whl`
  (CPython 3.12에서 빌드하면 `...-1private-cp312-none-any.whl`)
- wheel build tag는 숫자로 시작해야 하며 기본값은 `1private`입니다. `private1`은 허용하지 않습니다.
- internal `.pyc` 때문에 Python tag는 빌드한 CPython minor에 고정합니다. 정확한 bytecode magic도
  `PRIVATE-BUILD.json`에 기록하고 모든 `.pyc`를 설치 전에 검사합니다.
- native extension을 포함하지 않으므로 ABI tag는 `none`, platform tag는 `any`입니다. `.so`, `.pyd`,
  `.dll`, `.dylib`가 발견되면 `any` wheel 생성을 거부합니다.
- release artifact에서는 기본 install gate를 유지합니다. `--skip-install-check`는 로컬 진단용 우회 옵션입니다.
- RTX 5090 non-SELLM 환경은 [`docs/5090_non_sellm_bootstrap.md`](docs/5090_non_sellm_bootstrap.md)의
  pinned overlay와 provenance/CUDA gate를 사용합니다.

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

`full`/`ssl_only`는 PatchTST artifact와 artifact `save_dir`가 모두 필요합니다. PatchTST가 없는
request에는 `sl_only` 또는 `off`를 사용합니다.

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

- 최종 supervised checkpoint가 정확히 하나 생성되면 `primary_ckpt_path`와 `best_ckpt_path`가
  채워집니다.
- 그 외에는 `ckpt_paths`를 사용하고, SSL-only 결과는 `pretrain_ckpt_paths`를 사용합니다.

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

## DSIO Total Train Script

리눅스 서버에서는 notebook 대신
[DSIO total-train runner guide](src/model_test/total_train/README.md)를 기준으로 실행합니다.

현재 executable default는 `lookback=104`, `horizon=27`, endogenous/exogenous batch
`1024/512`, `ssl_mode="sl_only"`입니다. 기본 model group은 다음 scenario로 나뉩니다.

- `endo_only`: PatchTST, PatchMixer
- `exo_future`: ExoTST
- `exo_past_only`: TimeXer

작은 preflight 예시:

```bash
PYTHON_BIN=/path/to/project/python \
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=exo \
EXO_MODELS="exotst_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --exo-batch-size 32 \
  --warmup-epochs 1 --spike-epochs 0
```

기본 artifact root는 `artifacts/total_train`이며 scenario directory는 `endo_only`,
`exo_future`, `exo_past_only`입니다. PatchTST full SSL은 PatchTST-only stage에
`SSL_MODE=full`을 명시해 별도로 실행합니다. 5090 운영 실행에서는 `--device cuda`로 CPU fallback을
막고, smoke root를 운영 artifact root와 분리합니다.

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
- ExoTST는 past/future continuous exogenous가 모두 필요합니다.
- TimeXer v1은 past continuous exogenous-only 모델입니다.
- categorical exogenous는 continuous channel로 인코딩하기 전에는 public API가 거부합니다.
- Patch 기반 모델은 frequency별 `patch_len` 이상 `lookback` 이 필요합니다.
