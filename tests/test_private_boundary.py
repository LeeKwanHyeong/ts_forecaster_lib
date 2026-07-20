from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_FILES = sorted((ROOT / "src" / "modeling_module" / "api").glob("*.py"))
FORBIDDEN_PREFIXES = (
    "modeling_module.training",
    "modeling_module.models",
    "modeling_module.utils",
    "modeling_module.data_loader",
)


def _import_targets(file_path: Path) -> set[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    return imports


def test_public_api_modules_only_depend_on_private_runtime_boundary():
    for file_path in API_FILES:
        imports = _import_targets(file_path)
        forbidden = sorted(
            target
            for target in imports
            if any(target.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        )
        assert not forbidden, f"{file_path.name} imports internal modules directly: {forbidden}"
