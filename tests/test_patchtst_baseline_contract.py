from __future__ import annotations

import hashlib

import pytest

from modeling_module.models import (
    get_patchtst_default_model_key as public_get_patchtst_default_model_key,
)
from modeling_module.models.PatchTST import (
    FutureExoTokenFusion,
    PatchTSTEndogenousModel,
    PatchTSTExogenousModel,
    PatchTSTModel,
    PatchTSTQuantileEndogenousModel,
    PatchTSTQuantileExogenousModel,
    PatchTSTQuantileModel,
    SupervisedBackbone,
)
from modeling_module.models.PatchTST import __all__ as patchtst_public_exports
from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
from modeling_module.models.PatchTST.provenance import (
    PATCHTST_BASELINE_BLOBS,
    PATCHTST_BASELINE_COMMIT,
)
from modeling_module.models.PatchTST.supervised import (
    FutureExoTokenFusion as CanonicalFutureExoTokenFusion,
    PatchTSTEndogenousModel as CanonicalEndogenousModel,
    PatchTSTExogenousModel as CanonicalExogenousModel,
    PatchTSTModel as CanonicalPatchTSTModel,
    PatchTSTQuantileEndogenousModel as CanonicalQuantileEndogenousModel,
    PatchTSTQuantileExogenousModel as CanonicalQuantileExogenousModel,
    PatchTSTQuantileModel as CanonicalQuantileModel,
    SupervisedBackbone as CanonicalSupervisedBackbone,
)
from modeling_module.models.model_builder import (
    build_patchTST,
    build_patchTST_exogenous,
    build_patchTST_quantile,
    build_patchTST_quantile_exogenous,
)
from modeling_module.models.registry import get_patchtst_default_model_key
from modeling_module.training.model_losses.loss_module import DistributionLoss


def _config(*, exogenous: bool = False, loss=None) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        padding_patch="end",
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=1,
        past_exo_cont_dim=1 if exogenous else 0,
        future_exo_dim=1 if exogenous else 0,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        loss=loss,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
            causal=False,
        ),
    )


def _state_schema_hash(model) -> str:
    schema = "\n".join(
        f"{name}|{tuple(tensor.shape)}|{tensor.dtype}"
        for name, tensor in sorted(model.state_dict().items())
    )
    return hashlib.sha256(schema.encode("ascii")).hexdigest()


def test_patchtst_public_exports_resolve_to_canonical_implementations() -> None:
    assert SupervisedBackbone is CanonicalSupervisedBackbone
    assert FutureExoTokenFusion is CanonicalFutureExoTokenFusion
    assert PatchTSTModel is CanonicalPatchTSTModel
    assert PatchTSTQuantileModel is CanonicalQuantileModel
    assert PatchTSTEndogenousModel is CanonicalEndogenousModel
    assert PatchTSTExogenousModel is CanonicalExogenousModel
    assert PatchTSTQuantileEndogenousModel is CanonicalQuantileEndogenousModel
    assert PatchTSTQuantileExogenousModel is CanonicalQuantileExogenousModel
    assert "PatchTSTPointModel" not in patchtst_public_exports
    assert "PointHead" not in patchtst_public_exports
    assert "QuantileHead" not in patchtst_public_exports
    assert public_get_patchtst_default_model_key is get_patchtst_default_model_key


def test_patchtst_source_identity_is_pinned() -> None:
    assert PATCHTST_BASELINE_COMMIT == "43f5ec8c9cbc89eaed2a28d7fb011d86b5303428"
    assert PATCHTST_BASELINE_BLOBS == {
        "supervised/PatchTST.py": "8fd033e32d2247f6af02442de5c1c4e68deefb8b",
        "supervised/backbone.py": "7104d734acd0f28d26cbbb09a9f129d908b51e44",
        "common/backbone_base.py": "5bb7fd4a42ecb707075cab5301e32e9a90f17a0a",
        "common/configs.py": "90c471a3760867377aa1fe1a4536f708310c8536",
        "supervised/variants.py": "6a580289c172d89957d93eae7371dcbbff869acc",
    }


@pytest.mark.parametrize(
    ("builder", "config", "model_class", "parameters", "state_keys", "schema_hash"),
    (
        (
            build_patchTST,
            _config(),
            PatchTSTEndogenousModel,
            706,
            29,
            "5117c80bdc1fd89f4801bfa7fda7440ed81f7f4e71c31f4fc9a6042bf6caae8d",
        ),
        (
            build_patchTST_exogenous,
            _config(exogenous=True),
            PatchTSTExogenousModel,
            1_676,
            50,
            "278bb77462e07d0d45262403a9b444fd1fe3b2caef012aebab47d56f5902ca4e",
        ),
        (
            build_patchTST_quantile,
            _config(),
            PatchTSTQuantileEndogenousModel,
            2_614,
            31,
            "dcfb96f61a4ccaa62b71c813404f98e89d483f7bac9a801377242ded55ce8253",
        ),
        (
            build_patchTST_quantile_exogenous,
            _config(exogenous=True),
            PatchTSTQuantileExogenousModel,
            3_584,
            52,
            "ae0305caebc0bbbbe4919986548f75cde57d32a2952b8522e7e2d4de4cb8d93e",
        ),
        (
            build_patchTST,
            _config(loss=DistributionLoss("Normal")),
            PatchTSTEndogenousModel,
            2_361,
            32,
            "06bf9aa6e29a0b2cd575ca8a88ca1fc6fdb57a31cce0ed5ca4e96f4bb64e9f32",
        ),
        (
            build_patchTST_exogenous,
            _config(exogenous=True, loss=DistributionLoss("Normal")),
            PatchTSTExogenousModel,
            3_331,
            53,
            "3acafa3c70d7cb1f833d95f9094defa29015132942115ef613e4834427e524be",
        ),
    ),
)
def test_patchtst_parameter_and_state_schema_baseline(
    builder,
    config,
    model_class,
    parameters,
    state_keys,
    schema_hash,
) -> None:
    model = builder(config)

    assert isinstance(model, model_class)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters
    assert len(model.state_dict()) == state_keys
    assert _state_schema_hash(model) == schema_hash
