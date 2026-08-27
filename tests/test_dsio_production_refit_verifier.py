from __future__ import annotations

import pytest

from tools.verify_dsio_production_refit import (
    _normalize_contract_value,
    _parse_expected_config,
)


def test_expected_config_parser_supports_scalar_and_sequence_values():
    assert _parse_expected_config(
        [
            "d_model=384",
            'stack_types=["identity","trend"]',
            "use_norm=true",
        ]
    ) == {
        "d_model": 384,
        "stack_types": ["identity", "trend"],
        "use_norm": True,
    }


def test_expected_config_parser_rejects_non_json_values():
    with pytest.raises(ValueError, match="Invalid JSON value"):
        _parse_expected_config(["activation=gelu"])


def test_contract_value_normalization_treats_tuple_and_list_equally():
    assert _normalize_contract_value(("identity", (1, 2))) == [
        "identity",
        [1, 2],
    ]
