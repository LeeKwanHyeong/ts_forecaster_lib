from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _dependency_name(requirement: str) -> str:
    match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, f"invalid dependency requirement: {requirement!r}"
    return match.group(0).casefold()


def test_pyproject_declares_expected_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        _dependency_name(requirement)
        for requirement in pyproject["project"]["dependencies"]
    }

    assert {"torch", "polars", "pyarrow", "pyyaml"} <= dependency_names


def test_pyproject_uses_package_readme():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["readme"] == "README.package.md"
    assert (ROOT / "README.package.md").exists()


def test_pyproject_declares_dev_and_notebook_extras():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert "dev" in optional
    assert "notebook" in optional
    dev_dependency_names = {
        _dependency_name(requirement)
        for requirement in optional["dev"]
    }
    requirements_dev_names = {
        _dependency_name(line)
        for line in (ROOT / "requirements.dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("-r ")
    }
    assert {"build", "packaging", "pytest", "wheel"} <= dev_dependency_names
    assert requirements_dev_names == dev_dependency_names
    assert "jupyterlab" in optional["notebook"]
    assert "ipykernel" in optional["notebook"]


def test_timemixer_upstream_fixtures_are_explicit_package_data():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    assert package_data["modeling_module.models.TimeMixer"] == [
        "LICENSE.upstream",
        "upstream_manifest.json",
    ]
