from __future__ import annotations

import importlib
import sys
from pathlib import Path

from tools.build_private_wheel import (
    assert_only_public_python_sources,
    convert_internal_sources_to_sourceless,
    is_public_source,
)


def test_is_public_source_matches_api_boundary():
    assert is_public_source("modeling_module/__init__.py")
    assert is_public_source("modeling_module/api/train.py")
    assert is_public_source("modeling_module/api/data.py")
    assert not is_public_source("modeling_module/training/engine.py")
    assert not is_public_source("modeling_module/models/registry.py")


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
    (api_dir / "train.py").write_text("from modeling_module.training.engine import value\npublic_value = value\n", encoding="utf-8")
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

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("modeling_module", None)
    sys.modules.pop("modeling_module.api", None)
    sys.modules.pop("modeling_module.api.train", None)
    sys.modules.pop("modeling_module.training", None)
    sys.modules.pop("modeling_module.training.engine", None)

    module = importlib.import_module("modeling_module")
    assert module.public_value == 123
