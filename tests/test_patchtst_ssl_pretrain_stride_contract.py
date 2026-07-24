from __future__ import annotations

from pathlib import Path

import pytest
import torch

from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.PatchTST.self_supervised.PatchTST import (
    PatchTSTPretrainModel,
)
from modeling_module.models.model_builder import build_patchTST
from modeling_module.training.model_trainers.patchtst_finetune import (
    load_patchtst_pretrained_backbone,
)
from modeling_module.training.model_trainers.patchtst_pretrain import (
    build_patchtst_pretrain_contract,
)


def _config(
    *,
    patch_len: int,
    stride: int,
    input_channels: int = 1,
) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=52,
        horizon=4,
        patch_len=patch_len,
        stride=stride,
        padding_patch="end",
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=input_channels,
        use_revin=False,
        pe="sincos",
        learn_pe=True,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def _save_pretrain_checkpoint(
    path: Path,
    *,
    patch_len: int,
    pretrain_stride: int,
    supervised_stride: int,
    input_channels: int = 1,
) -> tuple[PatchTSTPretrainModel, dict]:
    pretrain = PatchTSTPretrainModel(
        _config(
            patch_len=patch_len,
            stride=pretrain_stride,
            input_channels=input_channels,
        )
    )
    contract = build_patchtst_pretrain_contract(
        pretrain,
        mask_ratio=0.4,
        loss_type="mse",
        supervised_stride=supervised_stride,
    )
    torch.save(
        {
            "state_dict": pretrain.state_dict(),
            "pretrain_contract": contract,
        },
        path,
    )
    return pretrain, contract


def test_candidate_stride_contract_restores_backbone_across_patch_counts(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "patchtst-pretrain-stride-13.pt"
    pretrain, contract = _save_pretrain_checkpoint(
        checkpoint_path,
        patch_len=13,
        pretrain_stride=13,
        supervised_stride=6,
    )
    pretrain.backbone.patch_embed.weight.data.fill_(0.125)
    pretrain.backbone.patch_embed.bias.data.fill_(-0.25)
    torch.save(
        {
            "state_dict": pretrain.state_dict(),
            "pretrain_contract": contract,
        },
        checkpoint_path,
    )

    supervised = build_patchTST(_config(patch_len=13, stride=6))
    head_before = {
        key: value.detach().clone()
        for key, value in supervised.state_dict().items()
        if key.startswith("head.")
    }

    report = load_patchtst_pretrained_backbone(
        supervised,
        str(checkpoint_path),
        load_strict=False,
    )

    assert contract["patching"] == {
        "lookback": 52,
        "patch_len": 13,
        "stride": 13,
        "input_channels": 1,
        "patch_count": 4,
        "padding": "none",
        "coverage_mode": "non_overlapping_contiguous",
        "uncovered_tail": 0,
    }
    assert contract["masking"]["mask_ratio"] == 0.4
    assert contract["transfer_target"] == {
        "patch_len": 13,
        "supervised_stride": 6,
    }
    assert report["target_patching"] == {
        "lookback": 52,
        "patch_len": 13,
        "stride": 6,
        "input_channels": 1,
        "patch_count": 8,
        "padding_patch": "end",
    }
    assert report["patching_compatibility"] == {
        "validation": "versioned_contract",
        "patch_len_match": True,
        "input_channels_match": True,
        "stride_match": False,
        "patch_count_match": False,
        "source_to_target_stride_change_allowed": True,
    }
    assert "backbone.pos_enc" in report["initialized_backbone_keys"]
    torch.testing.assert_close(
        supervised.state_dict()["backbone.input_proj.weight"],
        torch.full_like(
            supervised.state_dict()["backbone.input_proj.weight"],
            0.125,
        ),
    )
    torch.testing.assert_close(
        supervised.state_dict()["backbone.input_proj.bias"],
        torch.full_like(
            supervised.state_dict()["backbone.input_proj.bias"],
            -0.25,
        ),
    )
    for key, expected in head_before.items():
        torch.testing.assert_close(
            supervised.state_dict()[key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_versioned_pretrain_contract_rejects_patch_len_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "patchtst-pretrain-patch-12.pt"
    _save_pretrain_checkpoint(
        checkpoint_path,
        patch_len=12,
        pretrain_stride=12,
        supervised_stride=6,
    )
    supervised = build_patchTST(_config(patch_len=13, stride=6))

    with pytest.raises(
        ValueError,
        match="pretrain/supervised patch_len mismatch",
    ):
        load_patchtst_pretrained_backbone(
            supervised,
            str(checkpoint_path),
            load_strict=False,
        )


def test_versioned_pretrain_contract_rejects_input_channel_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "patchtst-pretrain-two-channel.pt"
    _save_pretrain_checkpoint(
        checkpoint_path,
        patch_len=13,
        pretrain_stride=13,
        supervised_stride=6,
        input_channels=2,
    )
    supervised = build_patchTST(_config(patch_len=13, stride=6))

    with pytest.raises(
        ValueError,
        match="input channel mismatch",
    ):
        load_patchtst_pretrained_backbone(
            supervised,
            str(checkpoint_path),
            load_strict=False,
        )


def test_versioned_pretrain_contract_rejects_inconsistent_patch_count(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "patchtst-pretrain-invalid-count.pt"
    pretrain, contract = _save_pretrain_checkpoint(
        checkpoint_path,
        patch_len=13,
        pretrain_stride=13,
        supervised_stride=6,
    )
    contract["patching"]["patch_count"] = 5
    torch.save(
        {
            "state_dict": pretrain.state_dict(),
            "pretrain_contract": contract,
        },
        checkpoint_path,
    )
    supervised = build_patchTST(_config(patch_len=13, stride=6))

    with pytest.raises(
        ValueError,
        match="patch_count is inconsistent",
    ):
        load_patchtst_pretrained_backbone(
            supervised,
            str(checkpoint_path),
            load_strict=False,
        )


def test_versioned_pretrain_contract_rejects_unintended_supervised_stride(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "patchtst-pretrain-target-stride-6.pt"
    _save_pretrain_checkpoint(
        checkpoint_path,
        patch_len=13,
        pretrain_stride=13,
        supervised_stride=6,
    )
    supervised = build_patchTST(_config(patch_len=13, stride=5))

    with pytest.raises(
        ValueError,
        match="created for supervised stride=6.*target stride=5",
    ):
        load_patchtst_pretrained_backbone(
            supervised,
            str(checkpoint_path),
            load_strict=False,
        )
