# DSIO total-train runner

`dsio_total_running.py`는 weekly DSIO 데이터를 작은 smoke부터 전체 학습까지 같은 CLI로
실행하기 위한 운영용 runner입니다. Notebook 대신 새 process에서 실행하며, 결과는 scenario별
artifact directory와 `training_manifest.json`으로 분리합니다.

## Preconditions

- 기존 `DSIODemandEngine/modeling_module` 사본을 덮어쓰지 말고 `ts_forecaster_lib` 전용 checkout을
  사용합니다.
- 기본 data root는 `<repo>/raw_data/master`입니다.
- target data: `tb_master_target.parquet`
- exogenous data: `tb_master_target_exo.parquet` 우선, 없으면 `tb_master_exo.parquet`를 target과
  `(oper_part_no, demand_dt)`로 join합니다.
- 기본 column은 `oper_part_no`, `demand_dt`, `demand_qty`입니다. `exo_future`는 runner 상단의
  past/future continuous column이 모두 필요하고, `exo_past_only`는 past continuous column만
  필요합니다. 두 source를 join할 때 ID와 join-key dtype도 일치해야 합니다.
- 학습 가능한 window에는 최소 `lookback + horizon`개의 row와 해당 구간 전체의 finite target이
  필요합니다.
- categorical exogenous 목록은 비어 있습니다. Public API는 categorical 입력을 fail-fast하므로
  필요한 범주는 먼저 continuous feature로 인코딩합니다.

## Executable defaults

| 항목 | Python runner | Linux wrapper |
|---|---:|---:|
| mode | `both` | `exo` |
| endogenous models | `patchtst patchmixer` | 동일 |
| exogenous models | `exotst timexer` | 동일 |
| lookback / horizon | `104 / 27` | 동일 |
| endogenous / exogenous batch | `1024 / 512` | 동일 |
| warmup / spike epochs | `3 / 2` | 동일 |
| SSL mode | `sl_only` | `sl_only` |
| workers / prefetch | `8 / 4` | 동일 |
| device | `auto` | 동일 |
| artifact root | `<repo>/artifacts/total_train` | 동일 |

Family request는 public registry artifact로 확장됩니다.

- `endo_only`: `patchtst_base` → `patchtst_quantile` → `patchmixer`
- `exo_future`: `exotst_base`; past와 future continuous exogenous가 모두 필요
- `exo_past_only`: `timexer_base`; past continuous exogenous만 사용하고 future exogenous를 거부

CLI `--mode` 값은 `endo`, `exo`, `both`뿐입니다. 위 scenario 이름은 output directory이며
`--mode` 값이 아닙니다. `both`는 endogenous를 먼저 실행하고, 이어서 `exo_future`,
`exo_past_only` 순서로 실행합니다.

Titan은 deprecated이므로 기본 group과 5090 promotion 대상에서 제외합니다. 기존 registry key와
지원 checkpoint load는 유지되며, Titan을 명시적으로 학습 요청하면 `FutureWarning`이 발생합니다.

## Smoke-first commands

Linux에서는 wrapper가 실행 command와 환경을 기록하고 기본적으로
`logs/dsio_total_train_<timestamp>.log`에 stdout/stderr를 저장합니다.

```bash
chmod +x src/model_test/total_train/run_dsio_total_running_linux.sh

# 1. 가장 작은 endogenous point smoke
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=endo \
ENDO_MODELS="patchtst_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --endo-batch-size 32 \
  --warmup-epochs 1 --spike-epochs 0

# 2. future-exogenous smoke
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=exo \
EXO_MODELS="exotst_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --exo-batch-size 32 \
  --warmup-epochs 1 --spike-epochs 0

# 3. TimeXer past-only smoke
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=exo \
EXO_MODELS="timexer_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --exo-batch-size 32 \
  --warmup-epochs 1 --spike-epochs 0
```

PatchTST full SSL은 PatchTST만 포함한 별도 stage로 실행합니다. `full`/`ssl_only`를
ExoTST·TimeXer-only request에 전달하면 public API가 학습 전에 거부합니다.

```bash
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=endo \
ENDO_MODELS="patchtst_base" \
SSL_MODE=full \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --endo-batch-size 32 \
  --ssl-pretrain-epochs 1 --warmup-epochs 1 --spike-epochs 0
```

Python CLI를 직접 사용할 수도 있습니다.

```bash
python src/model_test/total_train/dsio_total_running.py \
  --mode both \
  --artifact-root "$PWD/artifacts/total_train_smoke" \
  --ssl-mode sl_only \
  --device cpu \
  --endo-batch-size 32 \
  --exo-batch-size 32 \
  --sample-part-count 8 \
  --warmup-epochs 1 \
  --spike-epochs 0
```

## Wrapper environment

| Variable | 의미 |
|---|---|
| `PYTHON_BIN` | 실행할 Python; 5090에서는 승인된 project environment를 명시 |
| `TS_FORECASTER_REPO_ROOT` | 전용 checkout root override |
| `ARTIFACT_ROOT` | artifact root override |
| `MODE` | `endo`, `exo`, `both` |
| `ENDO_MODELS` / `EXO_MODELS` | 공백으로 구분한 family 또는 canonical artifact key |
| `SSL_MODE` | `sl_only`, `off`, `full`; `full`은 PatchTST request 전용 |
| `SAMPLE_PART_COUNT` | 관측 target이 충분한 part 중 deterministic sample 수 |
| `CLEAN_OUTPUT` | `1`이면 이번 scenario output directory를 먼저 삭제 |
| `LOG_TO_FILE` / `LOG_DIR` / `RUN_TAG` | log 생성 제어 |

Wrapper 뒤에 전달한 인수는 Python CLI로 그대로 넘어갑니다.

## 5090 preflight

5090에서는 CUDA 실행이 검증된 project environment를 `PYTHON_BIN`으로 명시합니다. `auto`는
CUDA probe 실패 시 CPU로 fallback하므로 운영 학습은 `--device cuda`로 실패 경계를 고정합니다.
Non-SELLM private wheel 환경은 repository root의
[`docs/5090_non_sellm_bootstrap.md`](../../../docs/5090_non_sellm_bootstrap.md) 절차로 생성합니다.
단, 이 runner는 checkout의 `src`를 import path 앞에 추가하므로 `PYTHON_BIN`의 설치 wheel을 직접
검증하지 않습니다. 반드시 승인된 clean non-SELLM checkout에서 실행하고, private-wheel install
gate와 runner smoke 결과를 서로 다른 검증 증거로 기록합니다.

```bash
PYTHON_BIN=/path/to/project/python
"$PYTHON_BIN" -c \
'import torch; from modeling_module.utils.device import probe_device; print(torch.__version__, torch.version.cuda, probe_device("cuda"))'

PYTHON_BIN="$PYTHON_BIN" \
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=exo \
EXO_MODELS="exotst_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --exo-batch-size 32 \
  --warmup-epochs 1 --spike-epochs 0
```

Wrapper는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 기본 설정합니다. 자동 batch 축소나
OOM retry는 없으므로 작은 batch부터 올립니다. smoke artifact root는 운영 root와 분리합니다.

## Artifact and restart behavior

기본 output은 다음과 같습니다.

- `artifacts/total_train/endo_only`
- `artifacts/total_train/exo_future`
- `artifacts/total_train/exo_past_only`

각 directory에는 model checkpoint와 `training_manifest.json`이 생성됩니다. Runner에는 epoch-level
resume 기능이 없습니다. `CLEAN_OUTPUT=0`은 기존 directory를 보존할 뿐 자동 resume을 의미하지
않으며, 같은 이름의 checkpoint는 덮어쓰고 `training_manifest.json`도 현재 호출 결과로 교체합니다.
`CLEAN_OUTPUT=1`은 선택한 scenario directory 전체를 삭제하므로 경로와 sample 설정을 확인한 뒤
사용합니다.

기본 checkpoint basename은 다음과 같습니다.

| Artifact key | Scenario / checkpoint |
|---|---|
| `patchtst_base` | `endo_only/weekly_PatchTST_L104_H27.pt` |
| `patchtst_quantile` | `endo_only/weekly_PatchTSTQuantile_L104_H27.pt` |
| `patchmixer` | `endo_only/weekly_PatchMixer_L104_H27.pt` |
| `exotst_base` | `exo_future/weekly_ExoTSTBase_L104_H27.pt` |
| `timexer_base` | `exo_past_only/weekly_TimeXerBase_L104_H27.pt` |

PatchTST full SSL은 추가로 `endo_only/pretrain/patchtst_pretrain_best.pt`를 생성합니다.

하나의 checkpoint만 다시 만들 때는 canonical key를 지정하고 `CLEAN_OUTPUT=0`을 사용합니다. 이
방식도 학습 resume은 아니며 scenario manifest를 덮어씁니다. aggregate manifest 보존이 중요하면
별도 `ARTIFACT_ROOT`를 사용합니다. Runner는 pretrained checkpoint 인수를 노출하지 않으므로
PatchTST `full` 재실행은 SSL pretraining도 다시 수행합니다.

```bash
MODE=endo \
ENDO_MODELS="patchmixer" \
SSL_MODE=sl_only \
CLEAN_OUTPUT=0 \
src/model_test/total_train/run_dsio_total_running_linux.sh --device cuda
```

`total_running.ipynb`는 과거 수동 prototype으로, 현재 model group·batch·artifact 경로의 기준이
아닙니다. 실행 계약은 이 문서와 `dsio_total_running.py`를 기준으로 합니다.

## Promotion order

1. local `pytest`와 CPU artifact smoke
2. 5090 전용 checkout에서 `SAMPLE_PART_COUNT=8` preflight
3. PatchTST full SSL
4. point family 순서: PatchTST → PatchMixer → ExoTST → TimeXer
5. distribution/legacy restore와 fresh-process prediction
6. 모든 sampled gate가 통과한 뒤에만 전체 part/model run

모든 run에서 commit, Python/Torch/CUDA version, command, log, artifact path를 함께 보존합니다.
