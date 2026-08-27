from __future__ import annotations

import importlib
import importlib.util
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.tags import Tag
from packaging.utils import parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
BUILD_TOOL_PATH = ROOT / "tools" / "build_private_wheel.py"
BUILD_TOOL_MODULE = "_ts_forecaster_private_wheel_builder"
# This deleted CamelCase module was previously leaked from a stale repository build/lib tree.
STALE_BUILD_LIB_SOURCE = "modeling_module/data_loader/MultiPartDataModule.py"


def _load_build_tool():
    spec = importlib.util.spec_from_file_location(BUILD_TOOL_MODULE, BUILD_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wheel_builder = _load_build_tool()
assert_only_public_python_sources = wheel_builder.assert_only_public_python_sources
assert_compatible_bytecode = wheel_builder.assert_compatible_bytecode
assert_platform_independent_layout = wheel_builder.assert_platform_independent_layout
build_private_wheel = wheel_builder.build_private_wheel
build_public_wheel = wheel_builder.build_public_wheel
convert_internal_sources_to_sourceless = wheel_builder.convert_internal_sources_to_sourceless
is_public_source = wheel_builder.is_public_source
is_sellm_payload = wheel_builder.is_sellm_payload
private_compatibility_tag = wheel_builder.private_compatibility_tag
read_private_build_manifest = wheel_builder.read_private_build_manifest
stage_clean_wheel_source = wheel_builder.stage_clean_wheel_source
validate_private_build_tag = wheel_builder.validate_private_build_tag
validate_distribution_profile = wheel_builder.validate_distribution_profile


def test_is_public_source_matches_api_boundary():
    assert is_public_source("modeling_module/__init__.py")
    assert is_public_source("modeling_module/api/train.py")
    assert is_public_source("modeling_module/api/data.py")
    assert not is_public_source("modeling_module/training/engine.py")
    assert not is_public_source("modeling_module/models/registry.py")


def test_private_wheel_build_and_compatibility_tags_are_pep_compliant():
    assert validate_private_build_tag("1private") == "1private"
    with pytest.raises(ValueError, match="must start with a digit"):
        validate_private_build_tag("private1")

    assert private_compatibility_tag() == (
        f"cp{sys.version_info.major}{sys.version_info.minor}-none-any"
    )


def test_distribution_profile_controls_sellm_source_payload(tmp_path):
    assert validate_distribution_profile("SELLM") == "sellm"
    with pytest.raises(ValueError, match="Distribution profile"):
        validate_distribution_profile("unknown")

    non_sellm = stage_clean_wheel_source(
        tmp_path / "non-sellm",
        distribution_profile="non-sellm",
    )
    sellm = stage_clean_wheel_source(
        tmp_path / "sellm",
        distribution_profile="sellm",
    )

    assert not (non_sellm / "src/modeling_module/models/SELLM").exists()
    assert not (
        non_sellm / "src/modeling_module/training/model_trainers/sellm_train.py"
    ).exists()
    assert (sellm / "src/modeling_module/models/SELLM/SELLM.py").is_file()
    assert (
        sellm / "src/modeling_module/training/model_trainers/sellm_train.py"
    ).is_file()


def test_private_wheel_bytecode_magic_is_checked_exhaustively(tmp_path):
    compatible = tmp_path / "compatible.zip"
    with ZipFile(compatible, "w") as archive:
        archive.writestr(
            "modeling_module/training/engine.pyc",
            importlib.util.MAGIC_NUMBER + (b"\x00" * 12),
        )
    with ZipFile(compatible) as archive:
        assert_compatible_bytecode(archive)

    incompatible = tmp_path / "incompatible.zip"
    with ZipFile(incompatible, "w") as archive:
        archive.writestr(
            "modeling_module/training/engine.pyc",
            importlib.util.MAGIC_NUMBER + (b"\x00" * 12),
        )
        archive.writestr(
            "modeling_module/training/bad.pyc",
            b"BAD!" + (b"\x00" * 12),
        )
    with ZipFile(incompatible) as archive:
        with pytest.raises(RuntimeError, match=r"bytecode incompatible.*bad\.pyc"):
            assert_compatible_bytecode(archive)


def test_platform_independent_tag_rejects_native_payloads(tmp_path):
    native_module = tmp_path / "modeling_module" / "native.cpython-312-darwin.so"
    native_module.parent.mkdir()
    native_module.write_bytes(b"not-a-real-extension")

    with pytest.raises(RuntimeError, match="cannot use platform tag 'any'"):
        assert_platform_independent_layout(tmp_path)


def test_convert_internal_sources_to_sourceless(tmp_path, monkeypatch):
    pkg_root = tmp_path / "modeling_module"
    api_dir = pkg_root / "api"
    training_dir = pkg_root / "training"
    utils_dir = pkg_root / "utils"

    api_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)

    (pkg_root / "__init__.py").write_text("from .api.train import public_value\n", encoding="utf-8")
    (api_dir / "__init__.py").write_text("", encoding="utf-8")
    (api_dir / "train.py").write_text(
        "from modeling_module.training.engine import value\npublic_value = value\n",
        encoding="utf-8",
    )
    (training_dir / "__init__.py").write_text("", encoding="utf-8")
    (training_dir / "engine.py").write_text("value = 123\n", encoding="utf-8")
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "helper.py").write_text("helper_value = 456\n", encoding="utf-8")

    converted = convert_internal_sources_to_sourceless(tmp_path)
    assert sorted(converted) == [
        "modeling_module/training/__init__.py",
        "modeling_module/training/engine.py",
        "modeling_module/utils/__init__.py",
        "modeling_module/utils/helper.py",
    ]

    assert (pkg_root / "__init__.py").exists()
    assert (api_dir / "train.py").exists()
    assert not (training_dir / "engine.py").exists()
    assert (training_dir / "engine.pyc").exists()
    assert not (utils_dir / "helper.py").exists()
    assert (utils_dir / "helper.pyc").exists()

    assert_only_public_python_sources(tmp_path)

    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "modeling_module" or name.startswith("modeling_module.")
    }

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        for name in list(sys.modules):
            if name == "modeling_module" or name.startswith("modeling_module."):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()

        module = importlib.import_module("modeling_module")
        assert module.public_value == 123
    finally:
        for name in list(sys.modules):
            if name == "modeling_module" or name.startswith("modeling_module."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def test_private_wheel_clean_build_and_isolated_install_gate(tmp_path, monkeypatch):
    public_wheel = build_public_wheel(
        no_isolation=True,
        dest_dir=tmp_path / "public",
    )

    package_root = ROOT / "src"
    expected_sources = {
        path.relative_to(package_root).as_posix()
        for path in (package_root / "modeling_module").rglob("*.py")
        if not is_sellm_payload(path.relative_to(package_root).as_posix())
    }
    with ZipFile(public_wheel) as archive:
        public_sources = {
            name
            for name in archive.namelist()
            if name.startswith("modeling_module/") and name.endswith(".py")
        }
    public_distribution, public_version, _public_build, _public_tags = parse_wheel_filename(
        public_wheel.name
    )
    assert public_sources == expected_sources
    assert STALE_BUILD_LIB_SOURCE not in public_sources

    private_dir = tmp_path / "private"
    private_dir.mkdir()
    stale_wheel = private_dir / "modeling_module-9.9.9-private1-py3-none-any.whl"
    stale_wheel.write_text("stale", encoding="utf-8")

    install_results = []
    real_install_check = wheel_builder.verify_private_wheel_install

    def capture_install_check(wheel_path, **kwargs):
        result = real_install_check(wheel_path, **kwargs)
        install_results.append(result)
        return result

    monkeypatch.setattr(wheel_builder, "verify_private_wheel_install", capture_install_check)
    private_wheel = build_private_wheel(
        wheel_path=public_wheel,
        dest_dir=private_dir,
    )

    assert private_wheel.is_file()
    assert private_wheel != stale_wheel
    distribution, version, build, tags = parse_wheel_filename(private_wheel.name)
    assert distribution == public_distribution
    assert version == public_version
    assert build == (1, "private")
    assert tags == {
        Tag(f"cp{sys.version_info.major}{sys.version_info.minor}", "none", "any")
    }

    expected_public_sources = {name for name in expected_sources if is_public_source(name)}
    expected_internal_bytecode = {
        f"{name[:-3]}.pyc" for name in expected_sources if not is_public_source(name)
    }
    with ZipFile(private_wheel) as archive:
        names = set(archive.namelist())
        private_sources = {
            name
            for name in names
            if name.startswith("modeling_module/") and name.endswith(".py")
        }
        private_bytecode = {
            name
            for name in names
            if name.startswith("modeling_module/") and name.endswith(".pyc")
        }
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))

    assert private_sources == expected_public_sources
    assert private_bytecode == expected_internal_bytecode
    assert not any(is_sellm_payload(name) for name in names)
    assert "sellm" not in [
        value.casefold() for value in metadata.get_all("Provides-Extra", [])
    ]
    assert not any(
        dependency in requirement.casefold()
        for requirement in metadata.get_all("Requires-Dist", [])
        for dependency in ("accelerate", "safetensors", "tokenizers", "transformers")
    )
    assert f"{STALE_BUILD_LIB_SOURCE[:-3]}.pyc" not in private_bytecode

    manifest = read_private_build_manifest(private_wheel)
    assert manifest["build_tag"] == "1private"
    assert manifest["distribution_profile"] == "non-sellm"
    assert manifest["python_tag"] == f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert manifest["abi_tag"] == "none"
    assert manifest["platform_tag"] == "any"
    assert manifest["python_cache_tag"] == sys.implementation.cache_tag
    assert len(manifest["bytecode_magic_hex"]) == 8
    assert len(manifest["source_wheel_sha256"]) == 64

    assert len(install_results) == 1
    assert install_results[0]["registry_path"].endswith("registry.pyc")
    assert install_results[0]["non_sellm"] is True
    assert install_results[0]["manifest"] == manifest
    clean_install = install_results[0]["clean_install"]
    assert clean_install["manifest"] == manifest
    clean_package_path = Path(clean_install["package_path"])
    assert clean_package_path.name == "modeling_module"
    assert "site-packages" in clean_package_path.parts
    assert any(requirement.startswith("polars") for requirement in clean_install["requires"])
