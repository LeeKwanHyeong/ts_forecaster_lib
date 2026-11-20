"""내부 연산 소스의 출처·외부 chronos 독립성·공식 원본 수치 동등성 검사."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from modeling_module.models.Chronos2.training import Chronos2TrainableConfig, Chronos2TrainingBatch


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/modeling_module/models/Chronos2/native"
HAS_NATIVE = all(importlib.util.find_spec(name) is not None for name in ("transformers", "einops"))
HAS_OFFICIAL = importlib.util.find_spec("chronos") is not None
native_required = pytest.mark.skipif(not HAS_NATIVE, reason="chronos2-native 선택 의존성 없음")
official_required = pytest.mark.skipif(not HAS_OFFICIAL, reason="원본 대조 전용 chronos 패키지 없음")


def test_vendored_manifest_checksums_and_license_are_preserved():
    manifest = json.loads((NATIVE / "upstream_manifest.json").read_text())
    assert manifest["commit"] == "0aba28e9360b6355d5a919596c90b16c25939f02"
    assert manifest["tag"] == "v2.3.2" and manifest["license"] == "Apache-2.0"
    for name, expected in manifest["local_files"].items():
        assert hashlib.sha256((NATIVE / name).read_bytes()).hexdigest() == expected, name
    assert "Apache License" in (NATIVE / "LICENSE.upstream").read_text()
    assert "Copyright Amazon.com" in (NATIVE / "NOTICE.upstream").read_text()
    for name in ("config.py", "layers.py", "model.py", "preprocessing.py"):
        source = (NATIVE / name).read_text()
        assert "SPDX-License-Identifier: Apache-2.0" in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("chronos")
            elif isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] == "chronos" for alias in node.names)


@official_required
def test_vendored_arithmetic_ast_matches_the_pinned_installed_source():
    source_root = Path(importlib.util.find_spec("chronos").origin).parent
    manifest = json.loads((NATIVE / "upstream_manifest.json").read_text())
    for short in ("config.py", "layers.py", "model.py"):
        original = (source_root / "chronos2" / short).read_text()
        expected = manifest["upstream_files"][f"src/chronos/chronos2/{short}"]["sha256"]
        assert hashlib.sha256(original.encode()).hexdigest() == expected
        original = original.replace("from chronos.chronos_bolt import InstanceNorm, Patch",
                                    "from .preprocessing import InstanceNorm, Patch")
        assert ast.dump(ast.parse(original)) == ast.dump(ast.parse((NATIVE / short).read_text()))
    original = ast.parse((source_root / "chronos_bolt.py").read_text())
    extracted = ast.parse((NATIVE / "preprocessing.py").read_text())
    for name in ("Patch", "InstanceNorm"):
        upstream_class = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == name)
        local_class = next(node for node in extracted.body if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.dump(upstream_class) == ast.dump(local_class)


@native_required
def test_scratch_path_never_imports_or_requires_external_chronos():
    source = '''
import importlib.abc, importlib.metadata, sys
class BlockChronos(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'chronos':
            raise AssertionError('external chronos import: ' + fullname)
sys.meta_path.insert(0, BlockChronos())
original_version = importlib.metadata.version
def guarded_version(name):
    if name == 'chronos-forecasting':
        raise AssertionError('external chronos metadata required')
    return original_version(name)
importlib.metadata.version = guarded_version
import huggingface_hub
def forbidden(*args, **kwargs):
    raise AssertionError('download requested')
huggingface_hub.snapshot_download = forbidden
huggingface_hub.hf_hub_download = forbidden
import torch
from modeling_module.models.Chronos2.trainable_backbone import Chronos2TrainableBackbone
from modeling_module.models.Chronos2.training import Chronos2TrainableConfig, Chronos2TrainingBatch
cfg = Chronos2TrainableConfig(d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8, dropout_rate=0)
model = Chronos2TrainableBackbone(cfg)
result = model(Chronos2TrainingBatch(torch.ones(2, 52), torch.ones(2, 27)))
result.loss.backward()
assert result.point.shape == (2, 27, 1)
assert model.provenance()['implementation_source'] == 'vendored'
assert model.provenance()['chronos_package_required'] is False
assert type(model.native_model).__module__ == 'modeling_module.models.Chronos2.native.model'
assert not any(name == 'chronos' or name.startswith('chronos.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@official_required
@pytest.mark.parametrize("context_length,horizon,attention,shared_group", [
    (52, 27, "sdpa", False), (53, 27, "eager", False),
    (52, 13, "sdpa", True), (52, 33, "eager", True),
])
def test_original_and_local_core_match_initialization_forward_loss_and_gradients(
    context_length, horizon, attention, shared_group,
):
    from chronos.chronos2.config import Chronos2CoreConfig as OriginalConfig
    from chronos.chronos2.model import Chronos2Model as OriginalModel
    from modeling_module.models.Chronos2.native.config import Chronos2CoreConfig as LocalConfig
    from modeling_module.models.Chronos2.native.model import Chronos2Model as LocalModel

    cfg = Chronos2TrainableConfig(context_length=context_length, horizon=horizon, d_model=16,
        d_ff=32, num_layers=1, num_heads=2, d_kv=8, dropout_rate=0, quantile_levels=(0.1, 0.5, 0.9))
    settings = cfg.native_kwargs()
    settings["attn_implementation"] = attention
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        original = OriginalModel(OriginalConfig(**settings)).eval()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        local = LocalModel(LocalConfig(**settings)).eval()
    assert original.state_dict().keys() == local.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, local.state_dict()[name], rtol=0, atol=0)
    local.load_state_dict(original.state_dict(), strict=True)
    context = torch.linspace(1, 30, context_length).repeat(2, 1)
    target = torch.linspace(20, 50, horizon).repeat(2, 1)
    context[0, 3] = torch.nan
    target[1, 2] = torch.nan
    inputs = Chronos2TrainingBatch(context, target).native_inputs(cfg, "cpu")
    if shared_group:
        inputs["group_ids"] = torch.tensor([7, 7])
    first = original(**inputs, num_output_patches=cfg.num_output_patches)
    second = local(**inputs, num_output_patches=cfg.num_output_patches)
    torch.testing.assert_close(first.quantile_preds, second.quantile_preds, rtol=0, atol=0)
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    first.loss.backward()
    second.loss.backward()
    for name, parameter in original.named_parameters():
        other = dict(local.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


@official_required
@pytest.mark.parametrize("use_arcsinh", [False, True])
def test_extracted_normalization_and_patching_match_original(use_arcsinh):
    from chronos.chronos_bolt import InstanceNorm as OriginalNorm, Patch as OriginalPatch
    from modeling_module.models.Chronos2.native.preprocessing import InstanceNorm, Patch
    values = torch.arange(106, dtype=torch.float32).reshape(2, 53)
    values[0, 3] = torch.nan
    values[1] = 0
    torch.testing.assert_close(OriginalPatch(16, 16)(values), Patch(16, 16)(values), rtol=0, atol=0, equal_nan=True)
    old, new = OriginalNorm(use_arcsinh=use_arcsinh), InstanceNorm(use_arcsinh=use_arcsinh)
    first, first_scale = old(values)
    second, second_scale = new(values)
    torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
    for left, right in zip(first_scale, second_scale):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(old.inverse(first, first_scale), new.inverse(second, second_scale),
                               rtol=0, atol=0, equal_nan=True)
