"""
Private runtime boundary for the public `modeling_module` API.

The public package surface is intentionally exposed through:
- `modeling_module`
- `modeling_module.api`

Everything under `modeling_module._internal` is considered private implementation detail.
This package exists so the public API can depend on a stable private boundary today, while
future packaging can replace these modules with compiled/private implementations without
changing the public API contract.
"""

