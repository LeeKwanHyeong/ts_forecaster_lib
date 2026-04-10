from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_expected_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])

    assert "torch" in dependencies
    assert "polars" in dependencies
    assert "pyarrow" in dependencies
    assert "PyYAML" in dependencies


def test_pyproject_uses_package_readme():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["readme"] == "README.package.md"
    assert (ROOT / "README.package.md").exists()


def test_pyproject_declares_dev_and_notebook_extras():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert "dev" in optional
    assert "notebook" in optional
    assert "build>=1" in optional["dev"]
    assert "pytest>=8" in optional["dev"]
    assert "jupyterlab" in optional["notebook"]
    assert "ipykernel" in optional["notebook"]
