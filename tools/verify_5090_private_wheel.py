from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile


_PRIVATE_WHEEL_RE = re.compile(
    r"^modeling_module-[^-]+-1private-cp312-none-any\.whl$"
)
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")


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
) -> dict[str, Any]:
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
        if sellm_members:
            raise RuntimeError(f"non-SELLM wheel contains SELLM members: {sellm_members[:5]}")

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
        if any("sellm" in value.casefold() for value in dependency_metadata):
            raise RuntimeError("non-SELLM wheel metadata exposes SELLM dependencies")

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
        "non_sellm": True,
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: verify_5090_private_wheel.py WHEEL_PATH EXPECTED_COMMIT EXPECTED_SHA256"
        )
    result = verify_5090_private_wheel(
        sys.argv[1],
        expected_commit=sys.argv[2],
        expected_sha256=sys.argv[3],
    )
    print("WHEEL_PREFLIGHT_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
