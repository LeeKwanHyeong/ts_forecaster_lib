from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "tools" / "bootstrap_5090_non_sellm.sh"
OVERLAY_REQUIREMENTS = ROOT / "requirements.5090-non-sellm.txt"
PYPROJECT = ROOT / "pyproject.toml"
WHEEL_VERIFIER = ROOT / "tools" / "verify_5090_private_wheel.py"
APPROVED_COMMIT = "b015f9a5d0a282144738cae439755b3f5e409d4b"


def _load_wheel_verifier():
    spec = importlib.util.spec_from_file_location(
        "_ts_forecaster_5090_wheel_verifier", WHEEL_VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wheel_verifier = _load_wheel_verifier()


def _write_candidate_wheel(
    directory: Path,
    *,
    manifest_overrides: dict[str, object] | None = None,
    extra_members: dict[str, bytes] | None = None,
) -> tuple[Path, str]:
    wheel = directory / "modeling_module-0.1.1-1private-cp312-none-any.whl"
    manifest = {
        "build_tag": "1private",
        "python_tag": "cp312",
        "abi_tag": "none",
        "platform_tag": "any",
        "python_version": sys.version.split()[0],
        "python_cache_tag": "cpython-312",
        "bytecode_magic_hex": importlib.util.MAGIC_NUMBER.hex(),
        "builder_commit": APPROVED_COMMIT,
        "builder_worktree_dirty": False,
        "source_wheel_sha256": "a" * 64,
    }
    manifest.update(manifest_overrides or {})
    dist_info = "modeling_module-0.1.1.dist-info"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("modeling_module/__init__.py", "")
        archive.writestr("modeling_module/api/__init__.py", "")
        archive.writestr(
            "modeling_module/models/registry.pyc",
            importlib.util.MAGIC_NUMBER + (b"\x00" * 12),
        )
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: modeling-module\nVersion: 0.1.1\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nBuild: 1private\nTag: cp312-none-any\n",
        )
        archive.writestr(
            f"{dist_info}/PRIVATE-BUILD.json",
            json.dumps(manifest),
        )
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
    return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def _active_requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_5090_overlay_pins_a_project_compatible_scikit_learn():
    assert _active_requirements(OVERLAY_REQUIREMENTS) == [
        "scikit-learn==1.7.2 "
        "--hash=sha256:e5bf3d930aee75a65478df91ac1225ff89cd28e9ac7bd1196853a9229b6adb0b"
    ]

    project_text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'"(scikit-learn[^"]+)"', project_text)
    assert match is not None
    project_requirement = Requirement(match.group(1))

    assert Version("1.7.2") in project_requirement.specifier
    assert Version("1.8.0") not in project_requirement.specifier


def test_5090_bootstrap_has_fail_closed_environment_and_private_wheel_gates():
    script = BOOTSTRAP.read_text(encoding="utf-8")

    required_contracts = (
        "set -euo pipefail",
        "/opt/miniconda3/envs/ai_env/bin/python",
        'EXPECTED_PYTHON_VERSION="3.12.13"',
        'EXPECTED_DEVICE_NAME="NVIDIA GeForce RTX 5090"',
        "--system-site-packages",
        "EXPECTED_BUILDER_COMMIT",
        "EXPECTED_WHEEL_SHA256",
        "verify_5090_private_wheel.py",
        "STAGED_WHEEL_PATH",
        "--require-hashes --only-binary=:all:",
        "--ignore-installed --no-deps --requirement",
        "-m pip --isolated check",
        'manifest["builder_worktree_dirty"] is False',
        'manifest["build_tag"] == "1private"',
        'registry_path.suffix == ".pyc"',
        'probe_device("cuda") == (True, None)',
        '"sm_120" in torch.cuda.get_arch_list()',
        'torch.cuda.get_device_capability(0) == (12, 0)',
        "Path(sys.base_prefix).resolve() == base_prefix",
        "torch_path.is_relative_to(base_prefix)",
        'not hasattr(modeling_module, "SELLMArchitectureConfig")',
        "sellm_dependency_roots.isdisjoint(sys.modules)",
        "TS_FORECASTER_5090_BOOTSTRAP.json",
        "installed_wheel_members",
        "verify_record(distribution)",
        "verify_record(sklearn_distribution)",
        'if [[ "$BOOTSTRAP_MODE" == "new" ]]',
        "REUSE_VENV=1",
    )
    for contract in required_contracts:
        assert contract in script

    assert "rm -rf" not in script
    for sellm_dependency in ("accelerate", "safetensors", "tokenizers", "transformers"):
        assert f'"{sellm_dependency}"' in script.casefold()
        assert f"pip install {sellm_dependency}" not in script.casefold()


def test_5090_bootstrap_orders_all_preflight_gates_before_mutation():
    script = BOOTSTRAP.read_text(encoding="utf-8")

    wheel_gate = script.index('"$BASE_PYTHON" -I "$WHEEL_VERIFIER"')
    wheel_freeze = script.index('destination.open("xb")')
    venv_create = script.index('"$BASE_PYTHON" -m venv')
    venv_origin_gate = script.index("Path(sys.base_prefix).resolve() == base_prefix")
    first_pip_install = script.index('"$VENV_PYTHON" -m pip --isolated install')

    assert wheel_freeze < wheel_gate < venv_create < venv_origin_gate < first_pip_install
    assert script.count('"$VENV_PYTHON" -m pip --isolated install') == 2
    assert script.index('if [[ "$BOOTSTRAP_MODE" == "new" ]]') < first_pip_install
    assert first_pip_install < script.index("else\n  [[ -f \"$RECEIPT_PATH\" ]]")
    assert 'case "$VENV_REALPATH/" in' in script
    assert '"$REPO_REALPATH/"*' in script
    assert "unset PIP_TARGET PIP_PREFIX PYTHONPATH" in script
    assert '"$STAGED_WHEEL_PATH" "$EXPECTED_BUILDER_COMMIT"' in script
    assert '--no-index "$STAGED_WHEEL_PATH"' in script
    assert '--no-index "$ORIGINAL_WHEEL_PATH"' not in script


@pytest.mark.skipif(
    sys.implementation.cache_tag != "cpython-312",
    reason="the deployment verifier intentionally accepts CPython 3.12 only",
)
def test_5090_private_wheel_preflight_accepts_the_exact_non_sellm_contract(tmp_path):
    wheel, wheel_sha = _write_candidate_wheel(tmp_path)

    result = wheel_verifier.verify_5090_private_wheel(
        wheel,
        expected_commit=APPROVED_COMMIT,
        expected_sha256=wheel_sha,
        expected_builder_python=sys.version.split()[0],
    )

    assert result["non_sellm"] is True
    assert result["wheel_sha256"] == wheel_sha
    assert result["builder_commit"] == APPROVED_COMMIT


@pytest.mark.skipif(
    sys.implementation.cache_tag != "cpython-312",
    reason="the deployment verifier intentionally accepts CPython 3.12 only",
)
@pytest.mark.parametrize(
    ("manifest_overrides", "extra_members", "expected_message"),
    [
        ({"builder_worktree_dirty": True}, None, "builder_worktree_dirty"),
        ({"builder_commit": "c" * 40}, None, "builder_commit"),
        (None, {"modeling_module/models/SELLM/model.pyc": b"bad"}, "SELLM members"),
        (None, {"modeling_module/native.so": b"bad"}, "native payloads"),
    ],
)
def test_5090_private_wheel_preflight_rejects_unapproved_artifacts(
    tmp_path,
    manifest_overrides,
    extra_members,
    expected_message,
):
    wheel, wheel_sha = _write_candidate_wheel(
        tmp_path,
        manifest_overrides=manifest_overrides,
        extra_members=extra_members,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        wheel_verifier.verify_5090_private_wheel(
            wheel,
            expected_commit=APPROVED_COMMIT,
            expected_sha256=wheel_sha,
            expected_builder_python=sys.version.split()[0],
        )


@pytest.mark.skipif(
    sys.implementation.cache_tag != "cpython-312",
    reason="the deployment verifier intentionally accepts CPython 3.12 only",
)
@pytest.mark.parametrize(
    ("member_name", "expected_message"),
    [
        ("../escape.py", "unsafe paths"),
        ("modeling_module/__init__.py", "duplicate archive members"),
    ],
)
def test_5090_private_wheel_preflight_rejects_unsafe_or_duplicate_members(
    tmp_path,
    member_name,
    expected_message,
):
    wheel, _wheel_sha = _write_candidate_wheel(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(wheel, "a") as archive:
            archive.writestr(member_name, b"bad")
    wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match=expected_message):
        wheel_verifier.verify_5090_private_wheel(
            wheel,
            expected_commit=APPROVED_COMMIT,
            expected_sha256=wheel_sha,
            expected_builder_python=sys.version.split()[0],
        )


def test_5090_bad_wheel_hash_cannot_create_or_mutate_a_venv(tmp_path):
    wheel, _wheel_sha = _write_candidate_wheel(tmp_path)
    venv_dir = tmp_path / "must-not-exist"
    env = os.environ.copy()
    env.update(
        {
            "BASE_PYTHON": sys.executable,
            "EXPECTED_BUILDER_COMMIT": APPROVED_COMMIT,
            "EXPECTED_WHEEL_SHA256": "0" * 64,
        }
    )

    completed = subprocess.run(
        ["bash", str(BOOTSTRAP), str(wheel), str(venv_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert "wheel SHA-256 mismatch" in completed.stderr
    assert not venv_dir.exists()


def test_5090_bootstrap_shell_syntax_and_missing_argument_boundary():
    subprocess.run(["bash", "-n", str(BOOTSTRAP)], check=True)
    completed = subprocess.run(
        ["bash", str(BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "EXPECTED_BUILDER_COMMIT" in completed.stderr
