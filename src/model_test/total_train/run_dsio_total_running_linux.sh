#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export TS_FORECASTER_REPO_ROOT="${TS_FORECASTER_REPO_ROOT:-$REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/total_train}"
MODE="${MODE:-exo}"
ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer titan}"
EXO_MODELS="${EXO_MODELS:-exotst timexer}"
SSL_MODE="${SSL_MODE:-sl_only}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
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
  --artifact-root "$ARTIFACT_ROOT"
  --ssl-mode "$SSL_MODE"
)

if [[ ${#ENDO_MODELS_ARR[@]} -gt 0 ]]; then
  CMD+=(--endo-models "${ENDO_MODELS_ARR[@]}")
fi

if [[ ${#EXO_MODELS_ARR[@]} -gt 0 ]]; then
  CMD+=(--exo-models "${EXO_MODELS_ARR[@]}")
fi

if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  CMD+=(--clean-output)
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
echo "[run] MODE=$MODE"
echo "[run] ENDO_MODELS=$ENDO_MODELS"
echo "[run] EXO_MODELS=$EXO_MODELS"
echo "[run] SSL_MODE=$SSL_MODE"
echo "[run] CLEAN_OUTPUT=$CLEAN_OUTPUT"
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
