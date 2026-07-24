# DSIO total-train runner

`dsio_total_running.py`는 weekly DSIO 데이터를 작은 smoke부터 전체 학습까지 같은 CLI로
실행하기 위한 운영용 runner입니다. Notebook 대신 새 process에서 실행하며, 결과는 scenario별
artifact directory와 `training_manifest.json`으로 분리합니다.

## Preconditions

- 기존 `DSIODemandEngine/modeling_module` 사본을 덮어쓰지 말고 `ts_forecaster_lib` 전용 checkout을
  사용합니다.
- 기본 data root는 `<repo>/raw_data/master`입니다.
- target data: `tb_master_target.parquet`
- endogenous canonical schema는 `oper_part_no`, `demand_dt`(ISO `YYYYWW`),
  `demand_qty`이며 `seq` 같은 추가 열은 학습 입력에서 제외합니다.
- endogenous 기본 계약은 source 상한 `202544`, 실제 예측 원점 `202545`, 검증 원점
  `202518`입니다. 검증 target은 `202518..202544`의 27주이고 학습 target은 최대
  `202517`까지만 사용합니다.
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
| mode | `both` | `endo` |
| endogenous models | `patchtst patchmixer nhits timemixer` | 동일 |
| exogenous models | `exotst timexer` | 동일 |
| lookback / horizon | `52 / 27` | 동일 |
| train end / forecast origin | `202544 / 202545` | 동일 |
| validation origin / window stride | `202518 / 4` | 동일 |
| endogenous loader | `indexed_temporal` | 동일 |
| endogenous / exogenous batch | `1024 / 512` | 동일 |
| warmup / spike epochs | `30 / 0` | 동일 |
| SSL mode | `sl_only` | `sl_only` |
| workers / prefetch | `8 / 4` | 동일 |
| PatchTST default capacity | Small: `128 / 2 / 512` | 동일 |
| device | `auto` | 동일 |
| artifact root | `<repo>/artifacts/total_train` | 동일 |

Family request는 public registry artifact로 확장됩니다.

- `endo_only`: `patchtst_base` → `patchtst_quantile` → `patchmixer` →
  `nhits_base` → `timemixer`
- `exo_future`: `exotst_base`; past와 future continuous exogenous가 모두 필요
- `exo_past_only`: `timexer_base`; past continuous exogenous만 사용하고 future exogenous를 거부

CLI `--mode` 값은 `endo`, `exo`, `both`뿐입니다. 위 scenario 이름은 output directory이며
`--mode` 값이 아닙니다. `both`는 endogenous를 먼저 실행하고, 이어서 `exo_future`,
`exo_past_only` 순서로 실행합니다.

Titan은 deprecated이므로 기본 group과 5090 promotion 대상에서 제외합니다. 기존 registry key와
지원 checkpoint load는 유지되며, Titan을 명시적으로 학습 요청하면 `FutureWarning`이 발생합니다.
SELLM도 이번 endogenous batch에서 제외합니다.

기본 `indexed_temporal` loader는 시리즈 값을 한 번만 보관하고 윈도우를 산술 인덱스로 계산합니다.
랜덤 window split을 사용하지 않으며, 각 part의 마지막 27주를 동일한 last-origin 검증 구간으로
고정합니다. `window_stride=4`는 최근 학습 window가 항상 `202517`에 끝나도록 역방향 정렬됩니다.
`legacy` backend는 호환 확인용일 뿐 202545 운영 baseline에는 사용하지 않습니다.

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

# 2. 전체 source의 schema/calendar/window만 검사하고 종료
ARTIFACT_ROOT="$PWD/artifacts/total_train_preflight" \
MODE=endo \
PREFLIGHT_ONLY=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda

# 3. future-exogenous smoke
ARTIFACT_ROOT="$PWD/artifacts/total_train_smoke" \
MODE=exo \
EXO_MODELS="exotst_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --exo-batch-size 32 \
  --warmup-epochs 1 --spike-epochs 0

# 4. TimeXer past-only smoke
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
  --ssl-pretrain-epochs 1 \
  --ssl-pretrain-stride 13 \
  --ssl-mask-ratio 0.4 \
  --warmup-epochs 1 --spike-epochs 0
```

Python CLI를 직접 사용할 수도 있습니다.

```bash
python src/model_test/total_train/dsio_total_running.py \
  --mode endo \
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
| `TARGET_SOURCE` | canonical target Parquet 경로 |
| `MODE` | `endo`, `exo`, `both` |
| `ENDO_MODELS` / `EXO_MODELS` | 공백으로 구분한 family 또는 canonical artifact key |
| `SSL_MODE` | `sl_only`, `off`, `full`; `full`은 PatchTST request 전용 |
| `SSL_PRETRAIN_EPOCHS` | PatchTST SSL Pretrain epoch 수 |
| `SSL_PRETRAIN_STRIDE` | SSL 전용 patch stride; 비우면 supervised stride 재사용 |
| `SSL_MASK_RATIO` | SSL patch mask 비율 |
| `LOOKBACK` / `HORIZON` | endogenous/exogenous window 길이 |
| `TRAIN_END_WEEK` / `FORECAST_ORIGIN` | source 상한과 실제 예측 원점 |
| `VALIDATION_ORIGIN` / `WINDOW_STRIDE` | last-origin 검증 시작과 학습 stride |
| `WARMUP_EPOCHS` / `SPIKE_EPOCHS` | supervised stage epoch 수 |
| `SAMPLE_PART_COUNT` | 관측 target이 충분한 part 중 deterministic sample 수 |
| `PREFLIGHT_ONLY` | `1`이면 data manifest와 batch 검증 후 학습 없이 종료 |
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
MODE=endo \
ENDO_MODELS="patchtst_base" \
SAMPLE_PART_COUNT=8 \
CLEAN_OUTPUT=1 \
src/model_test/total_train/run_dsio_total_running_linux.sh \
  --device cuda --endo-batch-size 32 \
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
| `patchtst_base` | `endo_only/weekly_PatchTST_L52_H27.pt` |
| `patchtst_quantile` | `endo_only/weekly_PatchTSTQuantile_L52_H27.pt` |
| `patchmixer` | `endo_only/weekly_PatchMixer_L52_H27.pt` |
| `nhits_base` | `endo_only/weekly_NHITSBase_L52_H27.pt` |
| `timemixer` | `endo_only/weekly_TimeMixer_L52_H27.pt` |
| `exotst_base` | `exo_future/weekly_ExoTSTBase_L52_H27.pt` |
| `timexer_base` | `exo_past_only/weekly_TimeXerBase_L52_H27.pt` |

PatchTST full SSL은 추가로 `endo_only/pretrain/patchtst_pretrain_best.pt`를 생성합니다.
Endogenous run은 별도로 `endo_only/data_manifest.json`을 기록하며 source SHA-256, row/series 수,
기간, temporal split, 윈도우 수와 요청 artifact 목록을 보존합니다.

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

## Qualification evaluation and production-refit epochs

Qualification checkpoint는 public `load_predictor(..., strict=True)` 경로로 다시 로드한 뒤
`202518..202544`의 last-origin holdout 전체를 예측해 비교합니다. Point 모델은 `point`,
PatchTST Quantile은 `q50`을 point forecast로 사용합니다.

```bash
python tools/evaluate_dsio_qualification.py \
  --artifact-dir "$PWD/artifacts/<qualification-run>/endo_only" \
  --training-log "$PWD/logs/<qualification-run>.log" \
  --target-source "$PWD/raw_data/master/tb_master_target.parquet" \
  --baseline-max-epoch 30 \
  --device cuda \
  --batch-size 1024 \
  --num-workers 4
```

Epoch 상한이나 학습 길이를 비교하는 qualification은 canonical artifact 하나당 새 Python
process를 사용합니다. 여러 artifact를 한 process에서 순차 학습하면 앞선 모델이 소비한
dropout·shuffle RNG 횟수가 후속 모델의 초기화와 batch 순서에 영향을 주므로, 앞 모델의 epoch
수를 바꾸는 실험이 후속 모델의 seed 궤적까지 바꾸게 됩니다. 동일 `--seed` 값만으로는 이
순서 결합이 제거되지 않습니다.

평가 metric은 모든 `(series, horizon)` 관측치를 합친 micro 기준입니다.

- MAE: `mean(abs(prediction - actual))`
- WAPE: `sum(abs(prediction - actual)) / sum(abs(actual))`
- sMAPE: `mean(2 * abs(prediction - actual) / (abs(actual) + abs(prediction)))`

분모에는 zero-safe epsilon `1e-8`을 더합니다. CSV의 `wape`, `smape`는 ratio이고
`wape_pct`, `smape_pct`는 백분율입니다. 모델별 production refit epoch는 세 metric의 순위로
고르지 않습니다. 각 모델이 학습에 사용한 고유 validation objective의 최소 epoch를 사용하며,
평가기에서 log의 최소값과 `training_manifest.json`의 `best_val_loss`가 일치해야만 확정됩니다.
Quantile validation loss와 point loss의 숫자는 서로 직접 비교하지 않습니다.

기본 결과 위치는 `<artifact-dir>/qualification_evaluation`입니다.

- `qualification_metrics.csv`: 모델별 micro metric, 순위, checkpoint hash와 추론 정보
- `qualification_predictions.parquet`: 전체 point 예측과 오차
- `qualification_metrics_by_series.parquet`: 모델·부품별 metric
- `qualification_metrics_by_horizon.parquet`: 모델·horizon별 metric
- `production_refit_epochs.json`: 모델별 고정 refit epoch와 선택 근거
- `epoch_extension_analysis.json`: `--baseline-max-epoch` 전후의 best loss와 갱신 여부
- `qualification_summary.json`: metric/epoch 계약, 전체 학습 history와 산출물 manifest

Production refit은 선정 epoch만큼 새로 학습하되 target `202544`까지 모두 학습에 편입합니다.
이미 모델 선택에 사용한 qualification holdout으로 early stopping을 다시 수행하지 않습니다.
`production_refit` mode는 validation loader를 생성하거나 전달하지 않으며, validation loop,
early stopping, best-state 복원을 모두 비활성화하고 마지막 epoch state를 저장합니다.

PatchTST Small production artifact는 다음처럼 독립 artifact root에서 생성합니다.

```bash
MODE=endo \
TRAINING_MODE=production_refit \
ENDO_MODELS=patchtst_base \
WARMUP_EPOCHS=8 \
SPIKE_EPOCHS=0 \
SEED=42 \
SSL_MODE=sl_only \
ARTIFACT_ROOT="$PWD/artifacts/dsio_202545_patchtst_small_production_refit" \
src/model_test/total_train/run_dsio_total_running_linux.sh --device cuda
```

생성 checkpoint의 `meta`에는 `training_mode=production_refit`,
`validation_enabled=false`, `state_selection=final_epoch`, `configured_epochs=8`,
`completed_epochs=8`, `random_seed=42`가 기록되어야 합니다. Strict restore와 운영 origin
예측 검증은 다음 명령으로 수행합니다.

```bash
python tools/verify_dsio_production_refit.py \
  --checkpoint "$PWD/artifacts/dsio_202545_patchtst_small_production_refit/endo_only/weekly_PatchTST_L52_H27.pt" \
  --target-source "$PWD/raw_data/master/tb_master_target.parquet" \
  --output-dir "$PWD/artifacts/dsio_202545_patchtst_small_production_refit/verification" \
  --device cuda
```

202545 qualification 확정 결과와 checkpoint identity는
[`docs/DSIO202545QualificationBaseline.md`](docs/DSIO202545QualificationBaseline.md)에
고정합니다.

PatchTST의 Small·Medium·Current seed-42 capacity 비교는 기존 Current
checkpoint를 변경하지 않는 별도 실험입니다. 정확한 capacity, 수렴, checkpoint identity와
승격 경계는
[`docs/DSIO202545PatchTSTCapacitySweep.md`](docs/DSIO202545PatchTSTCapacitySweep.md)에
기록합니다.

Seed `11 / 22 / 33 / 42`의 Small·Current 격리 검증 결과, DSIO PatchTST
기본 capacity는 Small (`d_model=128`, `n_layers=2`, `d_ff=512`)이고 고정
production-refit epoch는 `8`입니다. Dense·intermittent cohort와 horizon별 적용 경계,
Current 재현 인수 및 refit 계약은
[`docs/DSIO202545PatchTSTMultiSeedDecision.md`](docs/DSIO202545PatchTSTMultiSeedDecision.md)에
고정합니다.

실제 Small·seed 42·8 epoch production checkpoint의 source/checkpoint hash, final-state
metadata, strict restore 결과와 Demand Engine 비음수 후처리 경계는
[`docs/DSIO202545PatchTSTProductionRefit.md`](docs/DSIO202545PatchTSTProductionRefit.md)에
고정합니다.

PatchTST Quantile·PatchMixer·N-HiTS·TimeMixer까지 포함한 다섯 endogenous
production-refit artifact의 고정 epoch, checkpoint SHA-256, parameter count와 RTX 5090
strict forecast 결과는
[`docs/DSIO202545EndogenousProductionRefit.md`](docs/DSIO202545EndogenousProductionRefit.md)에
고정합니다.

## Promotion order

1. local `pytest`와 전체 canonical source `PREFLIGHT_ONLY=1`
2. 5090 전용 checkout에서 전체 source preflight
3. 5090에서 각 endogenous artifact의 `SAMPLE_PART_COUNT` 1-epoch smoke
4. qualification 순서: PatchTST base/quantile → PatchMixer → N-HiTS → TimeMixer
5. checkpoint fresh-process load/predict
6. qualification 결과로 epoch/model을 확정한 뒤 `202544` 전체 refit artifact 생성

모든 run에서 commit, Python/Torch/CUDA version, command, log, artifact path를 함께 보존합니다.
