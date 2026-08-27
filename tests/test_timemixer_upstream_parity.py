from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from modeling_module.models.TimeMixer import TimeMixerConfig
from modeling_module.models.TimeMixer.backbone import TimeMixerBackbone


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "tests/fixtures/timemixer_upstream"
INTERMEDIATE_MODULES = (
    "pdm_blocks.0.decompsition",
    "pdm_blocks.0.mixing_multi_scale_season",
    "pdm_blocks.0.mixing_multi_scale_trend",
    "pdm_blocks.0",
    "enc_embedding",
    "predict_layers.0",
    "predict_layers.1",
    "predict_layers.2",
    "projection_layer",
)
EXPECTED_UNUSED_PARAMETERS = {
    "pdm_blocks.0.layer_norm.weight",
    "pdm_blocks.0.layer_norm.bias",
    "enc_embedding.temporal_embedding.embed.weight",
}
EXPECTED_STATE_SCHEMA_SHA256 = (
    "0608244ebd2ea1076bc17ea546cadbb39758a2c1e146c280d9334ccd202974b4"
)


def _load_upstream_module() -> ModuleType:
    source = REFERENCE_ROOT / "models/TimeMixer.py"
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            "_pinned_timemixer_upstream",
            source,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REFERENCE_ROOT))


UPSTREAM_MODULE = _load_upstream_module()


def _config() -> TimeMixerConfig:
    return TimeMixerConfig(
        lookback=16,
        horizon=4,
        y_dim=1,
        d_model=4,
        d_ff=8,
        e_layers=1,
        moving_avg=3,
        down_sampling_layers=2,
        down_sampling_window=2,
        dropout=0.0,
        embed="timeF",
        freq="h",
        use_norm=True,
    )


def _models(seed: int):
    config = _config()
    torch.manual_seed(seed)
    upstream = UPSTREAM_MODULE.Model(config)
    torch.manual_seed(seed)
    local = TimeMixerBackbone(config)
    return upstream, local


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    raise TypeError(f"Unsupported intermediate output type: {type(value)!r}.")


def _capture_intermediates(model):
    captures: dict[str, list[Any]] = {name: [] for name in INTERMEDIATE_MODULES}
    modules = dict(model.named_modules())
    handles = []
    for name in INTERMEDIATE_MODULES:
        assert name in modules

        def capture(_module, _inputs, output, *, key=name):
            captures[key].append(_clone_tree(output))

        handles.append(modules[name].register_forward_hook(capture))
    return captures, handles


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    assert type(actual) is type(expected)
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        _assert_tree_equal(actual_item, expected_item)


def _state_schema_hash(model) -> str:
    schema = "\n".join(
        f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
        for name, tensor in sorted(model.state_dict().items())
    )
    return hashlib.sha256(schema.encode("ascii")).hexdigest()


def test_timemixer_output_and_intermediates_match_pinned_upstream_exactly() -> None:
    upstream, local = _models(seed=20260723)
    upstream.eval()
    local.eval()
    upstream_captures, upstream_handles = _capture_intermediates(upstream)
    local_captures, local_handles = _capture_intermediates(local)
    torch.manual_seed(20260724)
    x = torch.randn(2, 16, 1)

    try:
        with torch.no_grad():
            expected = upstream(x, None, None, None)
            actual = local(x)
    finally:
        for handle in upstream_handles + local_handles:
            handle.remove()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert upstream_captures.keys() == local_captures.keys()
    for name in INTERMEDIATE_MODULES:
        _assert_tree_equal(local_captures[name], upstream_captures[name])


def test_timemixer_input_and_parameter_gradients_match_pinned_upstream() -> None:
    upstream, local = _models(seed=20260725)
    torch.manual_seed(20260726)
    base_input = torch.randn(2, 16, 1)
    upstream_input = base_input.clone().requires_grad_(True)
    local_input = base_input.clone().requires_grad_(True)
    probe = torch.linspace(-1.0, 1.0, 8).reshape(2, 4, 1)

    upstream_loss = (upstream(upstream_input, None, None, None) * probe).sum()
    local_loss = (local(local_input) * probe).sum()
    upstream_loss.backward()
    local_loss.backward()

    torch.testing.assert_close(
        local_input.grad,
        upstream_input.grad,
        rtol=0.0,
        atol=0.0,
    )
    assert local_input.grad is not None
    assert torch.isfinite(local_input.grad).all()
    assert torch.count_nonzero(local_input.grad) > 0

    upstream_parameters = dict(upstream.named_parameters())
    local_parameters = dict(local.named_parameters())
    assert local_parameters.keys() == upstream_parameters.keys()
    unused_parameters = set()
    for name, local_parameter in local_parameters.items():
        upstream_gradient = upstream_parameters[name].grad
        local_gradient = local_parameter.grad
        if upstream_gradient is None:
            assert local_gradient is None
            unused_parameters.add(name)
            continue
        assert local_gradient is not None
        torch.testing.assert_close(
            local_gradient,
            upstream_gradient,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.isfinite(local_gradient).all()
        assert torch.count_nonzero(local_gradient) > 0
    assert unused_parameters == EXPECTED_UNUSED_PARAMETERS


def test_timemixer_parameter_and_state_schema_match_frozen_baseline() -> None:
    upstream, local = _models(seed=20260727)
    upstream_state = upstream.state_dict()
    local_state = local.state_dict()

    assert sum(parameter.numel() for parameter in local.parameters()) == 1_039
    assert len(local_state) == 39
    assert _state_schema_hash(local) == EXPECTED_STATE_SCHEMA_SHA256
    assert local_state.keys() == upstream_state.keys()
    for name, local_tensor in local_state.items():
        torch.testing.assert_close(
            local_tensor,
            upstream_state[name],
            rtol=0.0,
            atol=0.0,
        )
