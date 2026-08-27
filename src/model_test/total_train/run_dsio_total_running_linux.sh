#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export TS_FORECASTER_REPO_ROOT="${TS_FORECASTER_REPO_ROOT:-$REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/total_train}"
TARGET_SOURCE="${TARGET_SOURCE:-$REPO_ROOT/raw_data/master/tb_master_target.parquet}"
MODE="${MODE:-endo}"
TRAINING_MODE="${TRAINING_MODE:-qualification}"
ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer nhits timemixer}"
EXO_MODELS="${EXO_MODELS:-exotst timexer}"
SSL_MODE="${SSL_MODE:-sl_only}"
SSL_PRETRAIN_EPOCHS="${SSL_PRETRAIN_EPOCHS:-2}"
SSL_PRETRAIN_STRIDE="${SSL_PRETRAIN_STRIDE:-}"
SSL_MASK_RATIO="${SSL_MASK_RATIO:-0.3}"
LOOKBACK="${LOOKBACK:-52}"
HORIZON="${HORIZON:-27}"
TRAIN_END_WEEK="${TRAIN_END_WEEK:-202544}"
FORECAST_ORIGIN="${FORECAST_ORIGIN:-202545}"
VALIDATION_ORIGIN="${VALIDATION_ORIGIN:-202518}"
WINDOW_STRIDE="${WINDOW_STRIDE:-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-30}"
SPIKE_EPOCHS="${SPIKE_EPOCHS:-0}"
SEED="${SEED:-42}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SAMPLE_PART_COUNT="${SAMPLE_PART_COUNT:-}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/dsio_total_train_${RUN_TAG}.log}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a ENDO_MODELS_ARR <<< "$ENDO_MODELS"
read -r -a EXO_MODELS_ARR <<< "$EXO_MODELS"

CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/src/model_test/total_train/dsio_total_running.py"
  --mode "$MODE"
  --training-mode "$TRAINING_MODE"
  --artifact-root "$ARTIFACT_ROOT"
  --target-source "$TARGET_SOURCE"
  --ssl-mode "$SSL_MODE"
  --ssl-pretrain-epochs "$SSL_PRETRAIN_EPOCHS"
  --ssl-mask-ratio "$SSL_MASK_RATIO"
  --lookback "$LOOKBACK"
  --horizon "$HORIZON"
  --train-end-week "$TRAIN_END_WEEK"
  --forecast-origin "$FORECAST_ORIGIN"
  --validation-origin "$VALIDATION_ORIGIN"
  --window-stride "$WINDOW_STRIDE"
  --warmup-epochs "$WARMUP_EPOCHS"
  --spike-epochs "$SPIKE_EPOCHS"
  --seed "$SEED"
)

if [[ -n "$SSL_PRETRAIN_STRIDE" ]]; then
  CMD+=(--ssl-pretrain-stride "$SSL_PRETRAIN_STRIDE")
fi

if [[ ${#ENDO_MODELS_ARR[@]} -gt 0 ]]; then
  CMD+=(--endo-models "${ENDO_MODELS_ARR[@]}")
fi

if [[ ${#EXO_MODELS_ARR[@]} -gt 0 ]]; then
  CMD+=(--exo-models "${EXO_MODELS_ARR[@]}")
fi

if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  CMD+=(--clean-output)
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  CMD+=(--preflight-only)
fi

if [[ -n "$SAMPLE_PART_COUNT" ]]; then
  CMD+=(--sample-part-count "$SAMPLE_PART_COUNT")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run] REPO_ROOT=$REPO_ROOT"
echo "[run] PYTHON_BIN=$PYTHON_BIN"
echo "[run] ARTIFACT_ROOT=$ARTIFACT_ROOT"
echo "[run] TARGET_SOURCE=$TARGET_SOURCE"
echo "[run] MODE=$MODE"
echo "[run] TRAINING_MODE=$TRAINING_MODE"
echo "[run] ENDO_MODELS=$ENDO_MODELS"
echo "[run] EXO_MODELS=$EXO_MODELS"
echo "[run] SSL_MODE=$SSL_MODE"
echo "[run] SSL_PRETRAIN_EPOCHS=$SSL_PRETRAIN_EPOCHS"
echo "[run] SSL_PRETRAIN_STRIDE=$SSL_PRETRAIN_STRIDE"
echo "[run] SSL_MASK_RATIO=$SSL_MASK_RATIO"
echo "[run] LOOKBACK=$LOOKBACK"
echo "[run] HORIZON=$HORIZON"
echo "[run] TRAIN_END_WEEK=$TRAIN_END_WEEK"
echo "[run] FORECAST_ORIGIN=$FORECAST_ORIGIN"
echo "[run] VALIDATION_ORIGIN=$VALIDATION_ORIGIN"
echo "[run] WINDOW_STRIDE=$WINDOW_STRIDE"
echo "[run] WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "[run] SPIKE_EPOCHS=$SPIKE_EPOCHS"
echo "[run] SEED=$SEED"
echo "[run] CLEAN_OUTPUT=$CLEAN_OUTPUT"
echo "[run] PREFLIGHT_ONLY=$PREFLIGHT_ONLY"
echo "[run] SAMPLE_PART_COUNT=$SAMPLE_PART_COUNT"
echo "[run] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
if [[ "$LOG_TO_FILE" == "1" ]]; then
  mkdir -p "$LOG_DIR"
  echo "[run] LOG_FILE=$LOG_FILE"
fi

if [[ "$LOG_TO_FILE" == "1" ]]; then
  {
    printf '[run] CMD='
    printf '%q ' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
  } 2>&1 | tee -a "$LOG_FILE"
else
  exec "${CMD[@]}"
fi
