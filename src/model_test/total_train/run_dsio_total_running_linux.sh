#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export TS_FORECASTER_REPO_ROOT="${TS_FORECASTER_REPO_ROOT:-$REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/total_train}"
MODE="${MODE:-both}"
ENDO_MODELS="${ENDO_MODELS:-patchtst patchmixer titan}"
EXO_MODELS="${EXO_MODELS:-patchtst patchmixer titan exotst}"

echo "[run] REPO_ROOT=$REPO_ROOT"
echo "[run] PYTHON_BIN=$PYTHON_BIN"
echo "[run] ARTIFACT_ROOT=$ARTIFACT_ROOT"
echo "[run] MODE=$MODE"
echo "[run] ENDO_MODELS=$ENDO_MODELS"
echo "[run] EXO_MODELS=$EXO_MODELS"

exec "$PYTHON_BIN" "$REPO_ROOT/src/model_test/total_train/dsio_total_running.py" \
  --mode "$MODE" \
  --artifact-root "$ARTIFACT_ROOT" \
  --endo-models $ENDO_MODELS \
  --exo-models $EXO_MODELS \
  --clean-output \
  "$@"
