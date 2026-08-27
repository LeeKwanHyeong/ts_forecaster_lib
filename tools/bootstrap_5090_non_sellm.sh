#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONSTRAINTS_FILE="$REPO_ROOT/requirements.5090-non-sellm.txt"
WHEEL_VERIFIER="$REPO_ROOT/tools/verify_5090_private_wheel.py"
BASE_PYTHON="${BASE_PYTHON:-/opt/miniconda3/envs/ai_env/bin/python}"
REUSE_VENV="${REUSE_VENV:-0}"
EXPECTED_BUILDER_COMMIT="${EXPECTED_BUILDER_COMMIT:-}"
EXPECTED_WHEEL_SHA256="${EXPECTED_WHEEL_SHA256:-}"

EXPECTED_PYTHON_VERSION="3.12.13"
EXPECTED_TORCH_VERSION="2.11.0+cu130"
EXPECTED_CUDA_RUNTIME="13.0"
EXPECTED_DEVICE_NAME="NVIDIA GeForce RTX 5090"
EXPECTED_SCIKIT_LEARN_VERSION="1.7.2"
RECEIPT_FILENAME="TS_FORECASTER_5090_BOOTSTRAP.json"

usage() {
  cat <<'EOF'
Usage:
  EXPECTED_BUILDER_COMMIT=<40-char-sha> \
  EXPECTED_WHEEL_SHA256=<64-char-sha256> \
    tools/bootstrap_5090_non_sellm.sh WHEEL_PATH VENV_DIR

Required:
  WHEEL_PATH                 cp312 private wheel built from the approved clean checkout
  VENV_DIR                   new absolute venv path outside the repository
  EXPECTED_BUILDER_COMMIT    independently approved full commit in PRIVATE-BUILD.json
  EXPECTED_WHEEL_SHA256      lowercase SHA-256 checked before any venv mutation

Optional:
  BASE_PYTHON                approved CUDA environment Python
                             (default: /opt/miniconda3/envs/ai_env/bin/python)
  REUSE_VENV=1               verification-only reuse with a matching receipt
EOF
}

fail() {
  echo "[bootstrap-5090][error] $*" >&2
  exit 1
}

resolve_path() {
  "$BASE_PYTHON" -I -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1"
}

if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 2
fi

WHEEL_PATH="$1"
VENV_DIR="$2"

[[ -x "$BASE_PYTHON" ]] || fail "BASE_PYTHON is not executable: $BASE_PYTHON"
[[ -f "$CONSTRAINTS_FILE" ]] || fail "missing overlay requirements: $CONSTRAINTS_FILE"
[[ -f "$WHEEL_VERIFIER" ]] || fail "missing private-wheel verifier: $WHEEL_VERIFIER"
[[ -f "$WHEEL_PATH" ]] || fail "private wheel not found: $WHEEL_PATH"
[[ "$VENV_DIR" = /* ]] || fail "VENV_DIR must be an absolute path: $VENV_DIR"
[[ "$REUSE_VENV" == "0" || "$REUSE_VENV" == "1" ]] || \
  fail "REUSE_VENV must be 0 or 1"
[[ "$EXPECTED_BUILDER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
  fail "EXPECTED_BUILDER_COMMIT must be a lowercase 40-character Git SHA"
[[ "$EXPECTED_WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  fail "EXPECTED_WHEEL_SHA256 must be a lowercase 64-character SHA-256"

REPO_REALPATH="$(resolve_path "$REPO_ROOT")"
VENV_REALPATH="$(resolve_path "$VENV_DIR")"
HOME_REALPATH="$(resolve_path "${HOME:-/nonexistent-home}")"
[[ "$VENV_REALPATH" != "/" ]] || fail "VENV_DIR cannot resolve to /"
[[ "$VENV_REALPATH" != "$HOME_REALPATH" ]] || fail "VENV_DIR cannot resolve to HOME"
case "$VENV_REALPATH/" in
  "$REPO_REALPATH/"*) fail "VENV_DIR must resolve outside the repository: $VENV_DIR" ;;
esac
VENV_DIR="$VENV_REALPATH"
ORIGINAL_WHEEL_PATH="$(resolve_path "$WHEEL_PATH")"

# Freeze the caller-owned wheel into a private file. Only this copy is verified and installed,
# so replacing the original path after preflight cannot change the installed bytes.
STAGED_WHEEL_DIR="$(mktemp -d /tmp/tsf-5090-private-wheel.XXXXXX)"
STAGED_WHEEL_PATH="$STAGED_WHEEL_DIR/$(basename "$ORIGINAL_WHEEL_PATH")"
cleanup_staged_wheel() {
  "$BASE_PYTHON" -I -c '
from pathlib import Path
import sys
wheel = Path(sys.argv[1])
wheel.unlink(missing_ok=True)
wheel.parent.rmdir()
' "$STAGED_WHEEL_PATH" >/dev/null 2>&1 || true
}
trap cleanup_staged_wheel EXIT
"$BASE_PYTHON" -I -c '
from pathlib import Path
import shutil
import sys
source, destination = map(Path, sys.argv[1:])
with source.open("rb") as input_stream, destination.open("xb") as output_stream:
    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
destination.chmod(0o400)
' "$ORIGINAL_WHEEL_PATH" "$STAGED_WHEEL_PATH"

# This archive-only gate intentionally runs before venv creation or any pip command.
"$BASE_PYTHON" -I "$WHEEL_VERIFIER" \
  "$STAGED_WHEEL_PATH" "$EXPECTED_BUILDER_COMMIT" "$EXPECTED_WHEEL_SHA256"

BASE_PREFIX="$("$BASE_PYTHON" -I -c \
  'from pathlib import Path; import sys; print(Path(sys.prefix).resolve())')"
CONSTRAINTS_SHA256="$("$BASE_PYTHON" -I -c \
  'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "$CONSTRAINTS_FILE")"

EXPECTED_BASE_PREFIX="$BASE_PREFIX" \
EXPECTED_PYTHON_VERSION="$EXPECTED_PYTHON_VERSION" \
EXPECTED_TORCH_VERSION="$EXPECTED_TORCH_VERSION" \
EXPECTED_CUDA_RUNTIME="$EXPECTED_CUDA_RUNTIME" \
EXPECTED_DEVICE_NAME="$EXPECTED_DEVICE_NAME" \
"$BASE_PYTHON" -I - <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

base_prefix = Path(os.environ["EXPECTED_BASE_PREFIX"]).resolve()
torch_path = Path(torch.__file__).resolve()
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None

assert sys.implementation.name == "cpython", sys.implementation.name
assert sys.version.split()[0] == os.environ["EXPECTED_PYTHON_VERSION"], sys.version
assert Path(sys.prefix).resolve() == base_prefix, (sys.prefix, base_prefix)
assert torch_path.is_relative_to(base_prefix), (torch_path, base_prefix)
assert torch.__version__ == os.environ["EXPECTED_TORCH_VERSION"], torch.__version__
assert torch.version.cuda == os.environ["EXPECTED_CUDA_RUNTIME"], torch.version.cuda
assert torch.cuda.is_available()
assert device_name == os.environ["EXPECTED_DEVICE_NAME"], device_name
assert capability == (12, 0), capability
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
print("BASELINE_RESULT=" + json.dumps({
    "python": sys.version.split()[0],
    "base_prefix": str(base_prefix),
    "torch": torch.__version__,
    "torch_path": str(torch_path),
    "cuda_runtime": torch.version.cuda,
    "device": device_name,
    "capability": list(capability),
}, sort_keys=True))
PY

BOOTSTRAP_MODE="new"
if [[ -e "$VENV_DIR" ]]; then
  [[ -d "$VENV_DIR" ]] || fail "existing VENV_DIR is not a directory: $VENV_DIR"
  [[ "$REUSE_VENV" == "1" ]] || \
    fail "VENV_DIR already exists; choose a new path or set REUSE_VENV=1: $VENV_DIR"
  [[ -x "$VENV_DIR/bin/python" ]] || fail "existing VENV_DIR has no executable Python"
  BOOTSTRAP_MODE="reuse"
else
  [[ "$REUSE_VENV" == "0" ]] || \
    fail "REUSE_VENV=1 requires an existing receipt-bearing VENV_DIR: $VENV_DIR"
  mkdir -p "$(dirname "$VENV_DIR")"
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
RECEIPT_PATH="$VENV_DIR/$RECEIPT_FILENAME"
[[ -f "$VENV_DIR/pyvenv.cfg" ]] || fail "VENV_DIR is missing pyvenv.cfg"
grep -Eq '^include-system-site-packages = true$' "$VENV_DIR/pyvenv.cfg" || \
  fail "VENV_DIR must inherit the approved GPU dependency baseline"

# Prove that both a new and reused venv derive from the approved ai_env before pip.
EXPECTED_BASE_PREFIX="$BASE_PREFIX" \
EXPECTED_PYTHON_VERSION="$EXPECTED_PYTHON_VERSION" \
EXPECTED_TORCH_VERSION="$EXPECTED_TORCH_VERSION" \
EXPECTED_CUDA_RUNTIME="$EXPECTED_CUDA_RUNTIME" \
EXPECTED_DEVICE_NAME="$EXPECTED_DEVICE_NAME" \
"$VENV_PYTHON" -I - <<'PY'
import os
import sys
from pathlib import Path

import torch

base_prefix = Path(os.environ["EXPECTED_BASE_PREFIX"]).resolve()
torch_path = Path(torch.__file__).resolve()
assert sys.version.split()[0] == os.environ["EXPECTED_PYTHON_VERSION"], sys.version
assert Path(sys.base_prefix).resolve() == base_prefix, (sys.base_prefix, base_prefix)
assert torch_path.is_relative_to(base_prefix), (torch_path, base_prefix)
assert torch.__version__ == os.environ["EXPECTED_TORCH_VERSION"], torch.__version__
assert torch.version.cuda == os.environ["EXPECTED_CUDA_RUNTIME"], torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == os.environ["EXPECTED_DEVICE_NAME"]
assert torch.cuda.get_device_capability(0) == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
PY

unset PIP_TARGET PIP_PREFIX PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null

if [[ "$BOOTSTRAP_MODE" == "new" ]]; then
  "$VENV_PYTHON" -m pip --isolated install \
    --require-hashes --only-binary=:all: \
    --no-cache-dir --ignore-installed --no-deps --requirement "$CONSTRAINTS_FILE"
  "$VENV_PYTHON" -m pip --isolated install \
    --no-cache-dir --ignore-installed --no-deps --no-index "$STAGED_WHEEL_PATH"
else
  [[ -f "$RECEIPT_PATH" ]] || \
    fail "REUSE_VENV=1 requires a completed bootstrap receipt: $RECEIPT_PATH"
fi

"$VENV_PYTHON" -m pip --isolated check

BOOTSTRAP_MODE="$BOOTSTRAP_MODE" \
RECEIPT_PATH="$RECEIPT_PATH" \
EXPECTED_BASE_PREFIX="$BASE_PREFIX" \
EXPECTED_BUILDER_COMMIT="$EXPECTED_BUILDER_COMMIT" \
EXPECTED_WHEEL_SHA256="$EXPECTED_WHEEL_SHA256" \
EXPECTED_WHEEL_PATH="$STAGED_WHEEL_PATH" \
EXPECTED_CONSTRAINTS_SHA256="$CONSTRAINTS_SHA256" \
EXPECTED_PYTHON_VERSION="$EXPECTED_PYTHON_VERSION" \
EXPECTED_SCIKIT_LEARN_VERSION="$EXPECTED_SCIKIT_LEARN_VERSION" \
EXPECTED_TORCH_VERSION="$EXPECTED_TORCH_VERSION" \
EXPECTED_CUDA_RUNTIME="$EXPECTED_CUDA_RUNTIME" \
EXPECTED_DEVICE_NAME="$EXPECTED_DEVICE_NAME" \
"$VENV_PYTHON" -I - "$VENV_DIR" <<'PY'
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

venv_dir = Path(sys.argv[1]).resolve()
receipt_path = Path(os.environ["RECEIPT_PATH"]).resolve()
base_prefix = Path(os.environ["EXPECTED_BASE_PREFIX"]).resolve()
expected_receipt = {
    "schema_version": 1,
    "wheel_sha256": os.environ["EXPECTED_WHEEL_SHA256"],
    "builder_commit": os.environ["EXPECTED_BUILDER_COMMIT"],
    "constraints_sha256": os.environ["EXPECTED_CONSTRAINTS_SHA256"],
    "base_prefix": str(base_prefix),
    "python": os.environ["EXPECTED_PYTHON_VERSION"],
    "scikit_learn": os.environ["EXPECTED_SCIKIT_LEARN_VERSION"],
    "torch": os.environ["EXPECTED_TORCH_VERSION"],
    "cuda_runtime": os.environ["EXPECTED_CUDA_RUNTIME"],
    "device": os.environ["EXPECTED_DEVICE_NAME"],
    "capability": [12, 0],
}
if os.environ["BOOTSTRAP_MODE"] == "reuse":
    actual_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert actual_receipt == expected_receipt, (actual_receipt, expected_receipt)

import sklearn
import torch

import modeling_module
import modeling_module.models.registry as registry
from modeling_module import DistributionLoss, TrainRequest
from modeling_module.utils.device import probe_device

sellm_dependency_roots = {"accelerate", "safetensors", "tokenizers", "transformers"}
assert sellm_dependency_roots.isdisjoint(sys.modules), sorted(
    sellm_dependency_roots.intersection(sys.modules)
)

distribution = metadata.distribution("modeling-module")
sklearn_distribution = metadata.distribution("scikit-learn")
manifest = json.loads(distribution.read_text("PRIVATE-BUILD.json"))
package_path = Path(modeling_module.__file__).resolve()
registry_path = Path(registry.__file__).resolve()
sklearn_path = Path(sklearn.__file__).resolve()
torch_path = Path(torch.__file__).resolve()
site_packages_path = Path(distribution.locate_file("")).resolve()
sklearn_site_packages_path = Path(sklearn_distribution.locate_file("")).resolve()
distribution_files = [PurePosixPath(str(path)) for path in (distribution.files or ())]


def verify_record(distribution_metadata):
    record = distribution_metadata.read_text("RECORD")
    assert record is not None
    verified = 0
    for relative_path, digest_spec, size_text in csv.reader(io.StringIO(record)):
        if not digest_spec:
            continue
        algorithm, expected_digest = digest_spec.split("=", 1)
        installed_path = Path(distribution_metadata.locate_file(relative_path)).resolve()
        assert installed_path.is_file(), installed_path
        content = installed_path.read_bytes()
        actual_digest = base64.urlsafe_b64encode(
            hashlib.new(algorithm, content).digest()
        ).rstrip(b"=").decode("ascii")
        assert actual_digest == expected_digest, installed_path
        if size_text:
            assert len(content) == int(size_text), installed_path
        verified += 1
    assert verified > 0, distribution_metadata
    return verified


modeling_record_count = verify_record(distribution)
sklearn_record_count = verify_record(sklearn_distribution)

installed_wheel_members = 0
with ZipFile(os.environ["EXPECTED_WHEEL_PATH"]) as wheel_archive:
    for member in wheel_archive.infolist():
        if member.is_dir() or member.filename.endswith(".dist-info/RECORD"):
            continue
        installed_path = (site_packages_path / member.filename).resolve()
        assert installed_path.is_relative_to(site_packages_path), installed_path
        assert installed_path.is_file(), installed_path
        assert installed_path.read_bytes() == wheel_archive.read(member), installed_path
        installed_wheel_members += 1
assert installed_wheel_members > 0

assert sys.version.split()[0] == os.environ["EXPECTED_PYTHON_VERSION"], sys.version
assert Path(sys.base_prefix).resolve() == base_prefix, (sys.base_prefix, base_prefix)
assert venv_dir in site_packages_path.parents, (venv_dir, site_packages_path)
assert venv_dir in sklearn_site_packages_path.parents, (
    venv_dir,
    sklearn_site_packages_path,
)
assert venv_dir in package_path.parents, (venv_dir, package_path)
assert venv_dir in sklearn_path.parents, (venv_dir, sklearn_path)
assert torch_path.is_relative_to(base_prefix), (torch_path, base_prefix)
assert registry_path.suffix == ".pyc", registry_path
assert manifest["builder_commit"] == os.environ["EXPECTED_BUILDER_COMMIT"], manifest
assert manifest["builder_worktree_dirty"] is False, manifest
assert manifest["distribution_profile"] == "non-sellm", manifest
assert manifest["build_tag"] == "1private", manifest
assert manifest["python_tag"] == "cp312", manifest
assert manifest["abi_tag"] == "none", manifest
assert manifest["platform_tag"] == "any", manifest
assert manifest["python_version"] == os.environ["EXPECTED_PYTHON_VERSION"], manifest
assert manifest["python_cache_tag"] == sys.implementation.cache_tag, manifest
assert manifest["bytecode_magic_hex"] == importlib.util.MAGIC_NUMBER.hex(), manifest
assert re.fullmatch(r"[0-9a-f]{64}", manifest["source_wheel_sha256"]), manifest
assert not any(
    "sellm" in part.casefold() for path in distribution_files for part in path.parts
), distribution_files
assert not hasattr(modeling_module, "SELLMArchitectureConfig")
assert not any("sellm" in key.casefold() for key in registry.MODEL_SPECS)
assert not any("sellm" in family.casefold() for family in registry.TRAINING_FAMILY_DEFAULTS)
assert sklearn.__version__ == os.environ["EXPECTED_SCIKIT_LEARN_VERSION"], sklearn.__version__
assert sklearn_distribution.version == os.environ["EXPECTED_SCIKIT_LEARN_VERSION"], (
    sklearn_distribution.version
)
assert torch.__version__ == os.environ["EXPECTED_TORCH_VERSION"], torch.__version__
assert torch.version.cuda == os.environ["EXPECTED_CUDA_RUNTIME"], torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == os.environ["EXPECTED_DEVICE_NAME"]
assert torch.cuda.get_device_capability(0) == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
assert DistributionLoss(distribution="Normal").param_names == ["-loc", "-scale"]
assert TrainRequest(models=["patchtst_base"]).models == ["patchtst_base"]
assert probe_device("cuda") == (True, None)

x = torch.arange(4096, device="cuda", dtype=torch.float32)
tensor_sum = (x * x).sum().item()
torch.cuda.synchronize()

if os.environ["BOOTSTRAP_MODE"] == "new":
    temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary_receipt.write_text(
        json.dumps(expected_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_receipt, receipt_path)

print("BOOTSTRAP_RESULT=" + json.dumps({
    "mode": os.environ["BOOTSTRAP_MODE"],
    "package_version": distribution.version,
    "package_path": str(package_path),
    "registry_path": str(registry_path),
    "builder_commit": manifest["builder_commit"],
    "wheel_sha256": os.environ["EXPECTED_WHEEL_SHA256"],
    "receipt_path": str(receipt_path),
    "python": sys.version.split()[0],
    "scikit_learn": sklearn.__version__,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "tensor_sum": tensor_sum,
    "installed_wheel_members": installed_wheel_members,
    "modeling_record_hashes": modeling_record_count,
    "sklearn_record_hashes": sklearn_record_count,
    "sellm_dependencies_loaded": False,
    "non_sellm": True,
}, sort_keys=True))
PY

echo "PYTHON_BIN=$VENV_PYTHON"
