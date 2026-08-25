from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


_PRIVATE_WHEEL_RE = re.compile(
    r"^modeling_module-[^-]+-1private-cp312-none-any\.whl$"
)
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
_DISTRIBUTION_PROFILES = frozenset({"non-sellm", "sellm"})
_SELLM_DEPENDENCIES = frozenset(
    canonicalize_name(name)
    for name in ("accelerate", "safetensors", "tokenizers", "transformers")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {suffix}, found {len(matches)}")
    return matches[0]


def _is_sellm_member(name: str) -> bool:
    return any("sellm" in part.casefold() for part in PurePosixPath(name).parts)


def verify_5090_private_wheel(
    wheel_path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str,
    expected_builder_python: str = "3.12.13",
    distribution_profile: str = "non-sellm",
) -> dict[str, Any]:
    profile = str(distribution_profile).strip().casefold()
    if profile not in _DISTRIBUTION_PROFILES:
        raise RuntimeError(f"unsupported distribution profile: {distribution_profile!r}")
    wheel = Path(wheel_path).resolve()
    if not wheel.is_file():
        raise RuntimeError(f"private wheel not found: {wheel}")
    if not _PRIVATE_WHEEL_RE.fullmatch(wheel.name):
        raise RuntimeError(f"unexpected private wheel filename/tag: {wheel.name}")
    if not _HEX_40_RE.fullmatch(expected_commit):
        raise RuntimeError("expected commit must be a lowercase 40-character Git SHA")
    if not _HEX_64_RE.fullmatch(expected_sha256):
        raise RuntimeError("expected wheel SHA must be a lowercase 64-character SHA-256")

    actual_sha256 = _sha256(wheel)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"wheel SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if sys.implementation.name != "cpython":
        raise RuntimeError(f"wheel verifier requires CPython, got {sys.implementation.name}")
    if sys.version.split()[0] != expected_builder_python:
        raise RuntimeError(
            "wheel verifier must run with the builder Python: "
            f"expected {expected_builder_python}, got {sys.version.split()[0]}"
        )
    if sys.implementation.cache_tag != "cpython-312":
        raise RuntimeError(
            f"wheel verifier requires cpython-312, got {sys.implementation.cache_tag}"
        )

    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("private wheel contains duplicate archive members")
        unsafe_members = sorted(
            name
            for name in names
            if "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
        )
        if unsafe_members:
            raise RuntimeError(f"private wheel contains unsafe paths: {unsafe_members[:5]}")
        native_members = sorted(
            name for name in names if name.casefold().endswith(_NATIVE_SUFFIXES)
        )
        if native_members:
            raise RuntimeError(
                f"platform-independent private wheel contains native payloads: "
                f"{native_members[:5]}"
            )
        sellm_members = sorted(name for name in names if _is_sellm_member(name))
        if profile == "non-sellm" and sellm_members:
            raise RuntimeError(f"non-SELLM wheel contains SELLM members: {sellm_members[:5]}")
        if profile == "sellm" and not sellm_members:
            raise RuntimeError("SELLM wheel contains no SELLM members")

        manifest_name = _single_member(names, ".dist-info/PRIVATE-BUILD.json")
        metadata_name = _single_member(names, ".dist-info/METADATA")
        wheel_metadata_name = _single_member(names, ".dist-info/WHEEL")
        manifest = json.loads(archive.read(manifest_name))
        package_metadata = Parser().parsestr(
            archive.read(metadata_name).decode("utf-8")
        )
        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")

        dependency_metadata = [
            *package_metadata.get_all("Provides-Extra", []),
            *package_metadata.get_all("Requires-Dist", []),
        ]
        if profile == "non-sellm" and any(
            "sellm" in value.casefold() for value in dependency_metadata
        ):
            raise RuntimeError("non-SELLM wheel metadata exposes SELLM dependencies")
        if profile == "sellm":
            dependency_names = {
                canonicalize_name(Requirement(value).name)
                for value in package_metadata.get_all("Requires-Dist", [])
            }
            missing = sorted(_SELLM_DEPENDENCIES - dependency_names)
            if missing:
                raise RuntimeError(f"SELLM wheel metadata is missing dependencies: {missing}")

        wheel_lines = set(wheel_metadata.splitlines())
        if "Build: 1private" not in wheel_lines:
            raise RuntimeError("private wheel metadata is missing Build: 1private")
        if "Tag: cp312-none-any" not in wheel_lines:
            raise RuntimeError("private wheel metadata is missing Tag: cp312-none-any")

        public_sources = sorted(
            name
            for name in names
            if name.startswith("modeling_module/") and name.endswith(".py")
        )
        unexpected_sources = [
            name
            for name in public_sources
            if name != "modeling_module/__init__.py"
            and not name.startswith("modeling_module/api/")
        ]
        if unexpected_sources:
            raise RuntimeError(
                f"private wheel exposes internal Python sources: {unexpected_sources[:5]}"
            )

        if "modeling_module/models/registry.pyc" not in names:
            raise RuntimeError("private wheel is missing sourceless model registry")
        if "modeling_module/models/registry.py" in names:
            raise RuntimeError("private wheel exposes model registry source")

        bytecode_names = sorted(
            name
            for name in names
            if name.startswith("modeling_module/") and name.endswith(".pyc")
        )
        if not bytecode_names:
            raise RuntimeError("private wheel contains no internal bytecode")
        bad_magic = [
            name
            for name in bytecode_names
            if archive.read(name)[:4] != importlib.util.MAGIC_NUMBER
        ]
        if bad_magic:
            raise RuntimeError(f"private wheel contains incompatible bytecode: {bad_magic[:5]}")

    expected_manifest = {
        "distribution_profile": profile,
        "build_tag": "1private",
        "python_tag": "cp312",
        "abi_tag": "none",
        "platform_tag": "any",
        "python_version": expected_builder_python,
        "python_cache_tag": "cpython-312",
        "bytecode_magic_hex": importlib.util.MAGIC_NUMBER.hex(),
        "builder_commit": expected_commit,
        "builder_worktree_dirty": False,
    }
    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            raise RuntimeError(
                f"private wheel manifest mismatch for {key}: "
                f"expected {expected_value!r}, got {manifest.get(key)!r}"
            )
    if not _HEX_64_RE.fullmatch(str(manifest.get("source_wheel_sha256", ""))):
        raise RuntimeError("private wheel manifest has an invalid source wheel SHA-256")

    return {
        "wheel": str(wheel),
        "wheel_sha256": actual_sha256,
        "builder_commit": manifest["builder_commit"],
        "builder_python": manifest["python_version"],
        "python_tag": manifest["python_tag"],
        "bytecode_count": len(bytecode_names),
        "public_source_count": len(public_sources),
        "distribution_profile": profile,
        "non_sellm": profile == "non-sellm",
    }


def verify_sellm_checkpoint_from_wheel(
    wheel_path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    wheel = Path(wheel_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"SELLM checkpoint not found: {checkpoint}")

    with tempfile.TemporaryDirectory(prefix="sellm-wheel-strict-load-") as tmpdir:
        target = Path(tmpdir) / "site"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(target),
                str(wheel),
            ],
            cwd=tmpdir,
            env=env,
            check=True,
        )
        code = """
import json
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(target))

import modeling_module
from modeling_module import load_predictor

assert target in Path(modeling_module.__file__).resolve().parents
assert hasattr(modeling_module, "SELLMArchitectureConfig")
predictor = load_predictor(str(checkpoint), device="cpu", strict=True)
config = predictor.config
get_value = config.get if isinstance(config, dict) else lambda key: getattr(config, key)
assert predictor.model_key == "sellm_base", predictor.model_key
assert int(get_value("lookback")) == 52
assert int(get_value("horizon")) == 26
print(json.dumps({
    "checkpoint": str(checkpoint),
    "model_key": predictor.model_key,
    "lookback": int(get_value("lookback")),
    "horizon": int(get_value("horizon")),
    "strict_load": True,
}, sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code, str(target), str(checkpoint)],
            cwd=tmpdir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_path", type=Path)
    parser.add_argument("expected_commit")
    parser.add_argument("expected_sha256")
    parser.add_argument(
        "--distribution-profile",
        choices=sorted(_DISTRIBUTION_PROFILES),
        default="non-sellm",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = verify_5090_private_wheel(
        args.wheel_path,
        expected_commit=args.expected_commit,
        expected_sha256=args.expected_sha256,
        distribution_profile=args.distribution_profile,
    )
    if args.checkpoint is not None:
        if args.distribution_profile != "sellm":
            raise RuntimeError("--checkpoint requires --distribution-profile sellm")
        result["checkpoint"] = verify_sellm_checkpoint_from_wheel(
            args.wheel_path,
            checkpoint_path=args.checkpoint,
        )
    print("WHEEL_PREFLIGHT_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
