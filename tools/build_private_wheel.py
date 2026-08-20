from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
from email.message import Message
from email.parser import Parser
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"
PRIVATE_DIST_DIR = DIST_DIR / "private"
DEFAULT_PRIVATE_BUILD_TAG = "1private"
PRIVATE_BUILD_MANIFEST = "PRIVATE-BUILD.json"
PRIVATE_DISTRIBUTION_PROFILE = "non-sellm"
_BUILD_TAG_RE = re.compile(r"^[0-9][0-9A-Za-z_]*$")
_NATIVE_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})
_SELLM_DEPENDENCIES = frozenset(
    canonicalize_name(name)
    for name in ("accelerate", "safetensors", "tokenizers", "transformers")
)
_WHEEL_BUILD_FILES = (
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "README.package.md",
    "pyproject.toml",
)
PUBLIC_SOURCE_ROOTS = (
    "modeling_module/api/",
)
PUBLIC_SOURCE_FILES = {
    "modeling_module/__init__.py",
}


def is_sellm_payload(rel_path: str) -> bool:
    path = PurePosixPath(rel_path.replace("\\", "/"))
    return (
        any(part.casefold() == "sellm" for part in path.parts)
        or path.name.casefold() in {"sellm_train.py", "sellm_train.pyc"}
    )


def is_public_source(rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    if rel_path in PUBLIC_SOURCE_FILES:
        return True
    return any(rel_path.startswith(prefix) for prefix in PUBLIC_SOURCE_ROOTS)


def iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        rel_path = path.relative_to(root).as_posix()
        if ".dist-info/" in rel_path or ".egg-info/" in rel_path:
            continue
        yield path, rel_path


def convert_internal_sources_to_sourceless(root: Path) -> list[str]:
    converted: list[str] = []

    for source_path, rel_path in iter_python_sources(root):
        if is_public_source(rel_path):
            continue

        pyc_path = source_path.with_suffix(".pyc")
        py_compile.compile(
            str(source_path),
            cfile=str(pyc_path),
            dfile=rel_path,
            doraise=True,
            optimize=0,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        source_path.unlink()
        converted.append(rel_path)

    return converted


def assert_only_public_python_sources(root: Path) -> None:
    unexpected = [
        rel_path
        for _path, rel_path in iter_python_sources(root)
        if not is_public_source(rel_path)
    ]
    if unexpected:
        joined = ", ".join(sorted(unexpected)[:10])
        raise RuntimeError(
            f"Unexpected internal source files remained in private wheel layout: {joined}"
        )


def validate_private_build_tag(build_tag: str) -> str:
    normalized = str(build_tag).strip()
    if not _BUILD_TAG_RE.fullmatch(normalized):
        raise ValueError(
            "Private wheel build tag must start with a digit and contain only ASCII letters, "
            f"digits, or underscores; got {build_tag!r}. Example: {DEFAULT_PRIVATE_BUILD_TAG!r}."
        )
    return normalized


def private_python_tag() -> str:
    if sys.implementation.name != "cpython":
        raise RuntimeError(
            "Sourceless private wheels are currently supported only for CPython because their "
            "internal modules use interpreter-specific bytecode."
        )
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def private_compatibility_tag() -> str:
    return f"{private_python_tag()}-none-any"


def assert_platform_independent_layout(root: Path) -> None:
    native_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in _NATIVE_SUFFIXES
    )
    if native_files:
        joined = ", ".join(native_files[:10])
        raise RuntimeError(
            "Private wheel cannot use platform tag 'any' while native artifacts are present: "
            f"{joined}"
        )


def _single_dist_info_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.glob("*.dist-info") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one .dist-info directory, found {len(candidates)}")
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _builder_git_provenance() -> tuple[str | None, bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    commit_value = commit.stdout.strip() if commit.returncode == 0 else None
    dirty_value = bool(status.stdout.strip()) if status.returncode == 0 else None
    return commit_value, dirty_value


def write_private_build_manifest(
    unpacked_root: Path,
    *,
    source_wheel: Path,
    build_tag: str,
) -> Path:
    commit, dirty = _builder_git_provenance()
    manifest = {
        "format_version": 1,
        "distribution_profile": PRIVATE_DISTRIBUTION_PROFILE,
        "build_tag": validate_private_build_tag(build_tag),
        "python_tag": private_python_tag(),
        "abi_tag": "none",
        "platform_tag": "any",
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag,
        "bytecode_magic_hex": importlib.util.MAGIC_NUMBER.hex(),
        "source_wheel": source_wheel.name,
        "source_wheel_sha256": _sha256(source_wheel),
        "builder_commit": commit,
        "builder_worktree_dirty": dirty,
    }
    manifest_path = _single_dist_info_dir(unpacked_root) / PRIVATE_BUILD_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def stage_clean_wheel_source(stage_root: Path) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    for filename in _WHEEL_BUILD_FILES:
        source = REPO_ROOT / filename
        if source.is_file():
            shutil.copy2(source, stage_root / filename)

    package_source = REPO_ROOT / "src" / "modeling_module"
    if not package_source.is_dir():
        raise FileNotFoundError(f"Package source directory does not exist: {package_source}")

    shutil.copytree(
        package_source,
        stage_root / "src" / "modeling_module",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*.egg-info",
            "SELLM",
            "sellm_train.py",
        ),
    )
    return stage_root


def remove_sellm_payload(unpacked_root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(
        unpacked_root.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        rel_path = path.relative_to(unpacked_root).as_posix()
        if not is_sellm_payload(rel_path):
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
        removed.append(rel_path)
    return sorted(removed)


def sanitize_non_sellm_metadata(unpacked_root: Path) -> Path:
    metadata_path = _single_dist_info_dir(unpacked_root) / "METADATA"
    message = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    sanitized = Message()
    for name, value in message.raw_items():
        normalized_name = name.casefold()
        if normalized_name == "provides-extra" and value.strip().casefold() == "sellm":
            continue
        if normalized_name == "requires-dist":
            requirement = Requirement(value)
            if (
                canonicalize_name(requirement.name) in _SELLM_DEPENDENCIES
                or "sellm" in value.casefold()
            ):
                continue
        sanitized[name] = value
    sanitized.set_payload(message.get_payload())
    metadata_path.write_text(sanitized.as_string(), encoding="utf-8")
    return metadata_path


def build_public_wheel(*, no_isolation: bool, dest_dir: Path | None = None) -> Path:
    output_dir = (dest_dir or DIST_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="public-wheel-source-") as source_tmp,
        tempfile.TemporaryDirectory(prefix="public-wheel-output-") as output_tmp,
    ):
        source_root = stage_clean_wheel_source(Path(source_tmp))
        staged_output = Path(output_tmp)
        cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(staged_output)]
        if no_isolation:
            cmd.append("--no-isolation")

        subprocess.run(cmd, cwd=source_root, check=True)
        wheels = sorted(staged_output.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Public wheel build produced {len(wheels)} wheels; expected one.")
        public_wheel = output_dir / wheels[0].name
        shutil.copy2(wheels[0], public_wheel)

    return public_wheel


def unpack_wheel(wheel_path: Path, dest_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "wheel", "unpack", str(wheel_path), "--dest", str(dest_dir)],
        check=True,
    )

    unpacked = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(unpacked) != 1:
        raise RuntimeError(f"Expected exactly one unpacked wheel directory, found {len(unpacked)}")
    return unpacked[0]


def pack_private_wheel(unpacked_root: Path, *, dest_dir: Path, build_number: str) -> Path:
    build_tag = validate_private_build_tag(build_number)
    python_tag = private_python_tag()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="private-wheel-pack-") as pack_tmp:
        pack_dir = Path(pack_tmp)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(unpacked_root),
                "--dest-dir",
                str(pack_dir),
                "--build-number",
                build_tag,
            ],
            check=True,
        )

        packed = sorted(pack_dir.glob("*.whl"))
        if len(packed) != 1:
            raise RuntimeError(
                f"Private wheel pack step produced {len(packed)} wheels; expected one."
            )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "tags",
                "--python-tag",
                python_tag,
                "--abi-tag",
                "none",
                "--platform-tag",
                "any",
                "--build",
                build_tag,
                "--remove",
                str(packed[0]),
            ],
            check=True,
        )

        tagged = sorted(pack_dir.glob("*.whl"))
        if len(tagged) != 1:
            raise RuntimeError(
                f"Private wheel retag step produced {len(tagged)} wheels; expected one."
            )

        private_wheel = dest_dir / tagged[0].name
        shutil.copy2(tagged[0], private_wheel)

    return private_wheel


def read_private_build_manifest(wheel_path: Path) -> dict[str, Any]:
    with ZipFile(wheel_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith(f".dist-info/{PRIVATE_BUILD_MANIFEST}")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one {PRIVATE_BUILD_MANIFEST} in private wheel, "
                f"found {len(candidates)}."
            )
        return json.loads(archive.read(candidates[0]).decode("utf-8"))


def assert_compatible_bytecode(archive: ZipFile) -> None:
    bytecode_files = sorted(
        name
        for name in archive.namelist()
        if name.startswith("modeling_module/") and name.endswith(".pyc")
    )
    if not bytecode_files:
        raise RuntimeError("Private wheel contains no internal bytecode modules.")

    incompatible = [
        name
        for name in bytecode_files
        if archive.read(name)[:4] != importlib.util.MAGIC_NUMBER
    ]
    if incompatible:
        joined = ", ".join(incompatible[:10])
        raise RuntimeError(
            "Private wheel contains bytecode incompatible with this interpreter "
            f"(expected magic {importlib.util.MAGIC_NUMBER.hex()}): {joined}"
        )


def assert_private_wheel_metadata(wheel_path: Path, *, build_tag: str) -> None:
    expected_build = validate_private_build_tag(build_tag)
    expected_tag = private_compatibility_tag()
    expected_suffix = f"-{expected_build}-{expected_tag}.whl"
    if not wheel_path.name.endswith(expected_suffix):
        raise RuntimeError(
            f"Private wheel filename must end with {expected_suffix!r}; got {wheel_path.name!r}."
        )

    with ZipFile(wheel_path) as archive:
        sellm_members = sorted(
            name for name in archive.namelist() if is_sellm_payload(name)
        )
        if sellm_members:
            joined = ", ".join(sellm_members[:10])
            raise RuntimeError(f"Private wheel contains SELLM payloads: {joined}")
        unexpected_sources = sorted(
            name
            for name in archive.namelist()
            if name.startswith("modeling_module/")
            and name.endswith(".py")
            and not is_public_source(name)
        )
        if unexpected_sources:
            joined = ", ".join(unexpected_sources[:10])
            raise RuntimeError(f"Private wheel still contains internal Python sources: {joined}")
        assert_compatible_bytecode(archive)

        wheel_metadata = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(wheel_metadata) != 1:
            raise RuntimeError(
                f"Expected exactly one WHEEL metadata file, found {len(wheel_metadata)}."
            )
        metadata_text = archive.read(wheel_metadata[0]).decode("utf-8")
        if f"Tag: {expected_tag}" not in metadata_text:
            raise RuntimeError(
                f"Private wheel WHEEL metadata does not declare Tag: {expected_tag}."
            )
        if f"Build: {expected_build}" not in metadata_text:
            raise RuntimeError(
                f"Private wheel WHEEL metadata does not declare Build: {expected_build}."
            )

        package_metadata_names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(package_metadata_names) != 1:
            raise RuntimeError(
                "Expected exactly one package METADATA file, "
                f"found {len(package_metadata_names)}."
            )
        package_metadata = Parser().parsestr(
            archive.read(package_metadata_names[0]).decode("utf-8")
        )
        exposed_sellm_metadata = [
            value
            for header in ("Provides-Extra", "Requires-Dist")
            for value in package_metadata.get_all(header, [])
            if "sellm" in value.casefold()
            or (
                header == "Requires-Dist"
                and canonicalize_name(Requirement(value).name) in _SELLM_DEPENDENCIES
            )
        ]
        if exposed_sellm_metadata:
            raise RuntimeError(
                "Private wheel metadata exposes SELLM dependencies: "
                f"{exposed_sellm_metadata[:5]}"
            )

    manifest = read_private_build_manifest(wheel_path)
    expected_manifest = {
        "distribution_profile": PRIVATE_DISTRIBUTION_PROFILE,
        "build_tag": expected_build,
        "python_tag": private_python_tag(),
        "abi_tag": "none",
        "platform_tag": "any",
        "python_cache_tag": sys.implementation.cache_tag,
        "bytecode_magic_hex": importlib.util.MAGIC_NUMBER.hex(),
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Private wheel build manifest mismatch: {mismatches}")


def verify_private_wheel_install(wheel_path: Path) -> dict[str, Any]:
    manifest = read_private_build_manifest(wheel_path)
    assert_private_wheel_metadata(wheel_path, build_tag=manifest["build_tag"])

    with tempfile.TemporaryDirectory(prefix="private-wheel-install-") as install_tmp:
        install_root = Path(install_tmp)
        venv_root = install_root / "venv"
        venv_python = venv_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_CONFIG_FILE"] = os.devnull
        env["PIP_NO_CACHE_DIR"] = "1"
        env["PIP_NO_INDEX"] = "1"

        subprocess.run(
            [sys.executable, "-I", "-m", "venv", str(venv_root)],
            cwd=install_root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(venv_python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                str(wheel_path),
            ],
            cwd=install_root,
            env=env,
            check=True,
        )

        site_dir = Path(
            subprocess.run(
                [
                    str(venv_python),
                    "-I",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                cwd=install_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        clean_install_code = """
import json
import sys
from importlib import metadata
from pathlib import Path

site_dir = Path(sys.argv[1]).resolve()
distribution = metadata.distribution("modeling-module")
package_path = Path(distribution.locate_file("modeling_module")).resolve()
assert site_dir in package_path.parents, (site_dir, package_path)
manifest = json.loads(distribution.read_text("PRIVATE-BUILD.json"))
print(json.dumps({
    "package_path": str(package_path),
    "manifest": manifest,
    "requires": sorted(distribution.requires or []),
}, sort_keys=True))
"""
        clean_install = subprocess.run(
            [str(venv_python), "-I", "-c", clean_install_code, str(site_dir)],
            cwd=install_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        smoke_code = """
import json
import sys
from importlib import metadata
from pathlib import Path

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))

import modeling_module
import modeling_module.models.registry as registry
import polars as pl
from modeling_module import DistributionLoss, TrainRequest, build_dataset, train

package_path = Path(modeling_module.__file__).resolve()
registry_path = Path(registry.__file__).resolve()
assert target in package_path.parents, (target, package_path)
assert target in registry_path.parents, (target, registry_path)
assert registry_path.suffix == ".pyc", registry_path
assert callable(train)
assert not hasattr(modeling_module, "SELLMArchitectureConfig")
assert not any("sellm" in key.casefold() for key in registry.MODEL_SPECS)
assert not any("sellm" in family.casefold() for family in registry.TRAINING_FAMILY_DEFAULTS)
assert TrainRequest(models=["patchtst_base"]).models == ["patchtst_base"]
assert DistributionLoss(distribution="Normal").param_names == ["-loc", "-scale"]

dataset = build_dataset({
    "df": pl.DataFrame({
        "unique_id": ["series-1"] * 5,
        "date": [202401, 202402, 202403, 202404, 202405],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
    }),
    "backend": "exo",
    "stage": "train",
    "lookback": 2,
    "horizon": 1,
    "freq": "monthly",
    "batch_size": 1,
    "val_ratio": 0.2,
    "shuffle": False,
})
assert len(dataset) > 0
sample = dataset[0]
assert tuple(sample[0].shape) == (2, 1)

distribution = metadata.distribution("modeling-module")
manifest = json.loads(distribution.read_text("PRIVATE-BUILD.json"))
print(json.dumps({
    "package_path": str(package_path),
    "registry_path": str(registry_path),
    "manifest": manifest,
    "non_sellm": True,
}, sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-c", smoke_code, str(site_dir)],
            cwd=install_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        result["clean_install"] = json.loads(
            clean_install.stdout.strip().splitlines()[-1]
        )
        return result


def build_private_wheel(
    *,
    wheel_path: Path | None = None,
    no_isolation: bool = True,
    build_number: str = DEFAULT_PRIVATE_BUILD_TAG,
    dest_dir: Path | None = None,
    verify_install: bool = True,
) -> Path:
    build_tag = validate_private_build_tag(build_number)
    source_wheel = wheel_path or build_public_wheel(no_isolation=no_isolation)
    output_dir = dest_dir or PRIVATE_DIST_DIR

    with tempfile.TemporaryDirectory(prefix="private-wheel-") as tmpdir:
        unpack_root = unpack_wheel(source_wheel, Path(tmpdir))
        remove_sellm_payload(unpack_root)
        sanitize_non_sellm_metadata(unpack_root)
        convert_internal_sources_to_sourceless(unpack_root)
        assert_only_public_python_sources(unpack_root)
        assert_platform_independent_layout(unpack_root)
        write_private_build_manifest(
            unpack_root,
            source_wheel=source_wheel,
            build_tag=build_tag,
        )
        private_wheel = pack_private_wheel(
            unpack_root,
            dest_dir=output_dir,
            build_number=build_tag,
        )

    assert_private_wheel_metadata(private_wheel, build_tag=build_tag)
    if verify_install:
        verify_private_wheel_install(private_wheel)
    return private_wheel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a private wheel that keeps only public API sources as .py and converts "
            "internal modules to .pyc."
        )
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Existing public wheel to transform. If omitted, a public wheel is built first.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=PRIVATE_DIST_DIR,
        help="Destination directory for the private wheel.",
    )
    parser.add_argument(
        "--build-number",
        type=str,
        default=DEFAULT_PRIVATE_BUILD_TAG,
        help=(
            "PEP-compliant wheel build tag. It must begin with a digit; "
            f"default: {DEFAULT_PRIVATE_BUILD_TAG}."
        ),
    )
    parser.add_argument(
        "--with-isolation",
        action="store_true",
        help="Use isolated PEP 517 build when creating the source wheel.",
    )
    parser.add_argument(
        "--skip-install-check",
        action="store_true",
        help="Build the artifact without the default isolated pip install/import smoke check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    private_wheel = build_private_wheel(
        wheel_path=args.wheel,
        no_isolation=not args.with_isolation,
        build_number=args.build_number,
        dest_dir=args.dest_dir,
        verify_install=not args.skip_install_check,
    )
    print(private_wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
