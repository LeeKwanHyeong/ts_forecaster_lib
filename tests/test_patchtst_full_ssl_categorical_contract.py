from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from modeling_module.api.data import DataRequest, ExogenousConfig
from modeling_module.api.forecast import (
    ForecastRequest,
    ForecastRuntimeConfig,
    forecast,
)
from modeling_module.api.infer import load_predictor
from modeling_module.api.train import (
    ArchitectureConfig,
    ArtifactConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TrainerConfig,
    TrainRequest,
    train,
)
from modeling_module.models.PatchTST.common.configs import (
    AttentionConfig,
    PatchTSTConfig,
)
from modeling_module.models.PatchTST.self_supervised.PatchTST import (
    PatchTSTPretrainModel,
)
from modeling_module.models.model_builder import build_patchTST_exogenous
from modeling_module.training.model_trainers.patchtst_finetune import (
    PATCHTST_SUPERVISED_ONLY_PREFIXES,
    load_patchtst_pretrained_backbone,
)
from modeling_module.training.model_trainers.patchtst_pretrain import (
    _eval_pretrain,
    _extract_x,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    ROOT
    / "src/modeling_module/models/PatchTST/docs"
    / "PatchTSTProductionSLOnlyBaseline.json"
)


def _tiny_config(
    *,
    future_cat_cardinalities: tuple[int, ...] = (),
) -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=4,
        horizon=2,
        patch_len=2,
        stride=1,
        padding_patch="end",
        d_model=8,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
        c_in=1,
        future_exo_cat_cardinalities=future_cat_cardinalities,
        future_exo_cat_embedding_dim=4,
        future_exo_fusion_dropout=0.0,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=2,
            d_model=8,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def _categorical_frame(length: int = 16) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["series-1"] * length,
            "date": [
                20240100 + index
                for index in range(1, length + 1)
            ],
            "y": [
                5.0 + 0.25 * index + float(index % 3)
                for index in range(1, length + 1)
            ],
            "event_type": [
                f"event-{index}"
                for index in range(1, length + 1)
            ],
        }
    )


def _full_training_request(
    *,
    frame: pl.DataFrame,
    save_dir: Path,
) -> TrainRequest:
    return TrainRequest(
        data=DataRequest(
            df=frame,
            lookback=4,
            horizon=2,
            freq="daily",
            batch_size=4,
            val_ratio=0.5,
            shuffle=False,
            seed=17,
            exogenous=ExogenousConfig(
                use_exogenous_mode=True,
                use_past_exogenous=False,
                use_future_exogenous=True,
                future_exo_cat_cols=["event_type"],
            ),
        ),
        models=["patchtst_exogenous"],
        trainer=TrainerConfig(
            epochs=1,
            lr=1e-3,
            use_intermittent=False,
            val_use_weights=False,
        ),
        ssl=SSLConfig(
            mode="full",
            pretrain_epochs=1,
            mask_ratio=1.0,
            loss_type="mse",
            freeze_encoder_before_ft=False,
        ),
        runtime=RuntimeConfig(device="cpu"),
        artifacts=ArtifactConfig(
            save_dir=str(save_dir),
            auto_save_dir=False,
        ),
        architecture=ArchitectureConfig(
            patchtst=PatchTSTArchitectureConfig(
                patch_len=2,
                stride=1,
                d_model=16,
                n_layers=1,
                d_ff=16,
                dropout=0.0,
                norm="LayerNorm",
                pre_norm=True,
                use_revin=False,
                pe="zeros",
                learn_pe=True,
                padding_patch="end",
                future_exo_cat_embedding_dim=4,
                future_exo_fusion_dropout=0.0,
            )
        ),
    )


def test_production_sl_only_baseline_manifest_is_frozen() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["status"] == "frozen"
    assert baseline["forecast_origin"] == 202545
    assert baseline["model_key"] == "patchtst_base"
    assert baseline["training_strategy"] == "sl_only"
    assert baseline["checkpoint"] == {
        "file_name": "weekly_PatchTST_L52_H27.pt",
        "sha256": (
            "2674a5b01a882a7d3bf36af598d787136"
            "d2c15181879307989a8206a43fa2d78"
        ),
        "format_version": "modeling_module.ckpt.v3",
        "parameters": 403099,
        "architecture_variant": "endogenous",
    }
    assert baseline["data"]["cutoff"] == 202544
    assert baseline["architecture"] == {
        "d_model": 128,
        "n_layers": 2,
        "d_ff": 512,
    }
    assert baseline["training"]["seed"] == 42
    assert baseline["training"]["epochs"] == 8
    assert baseline["protection"] == {
        "overwrite_during_full_ssl_qualification": False,
        "full_ssl_artifact_root_must_be_distinct": True,
    }


def test_pretrain_transfer_changes_only_supervised_backbone(
    tmp_path: Path,
) -> None:
    torch.manual_seed(701)
    supervised = build_patchTST_exogenous(
        _tiny_config(future_cat_cardinalities=(3,))
    )
    pretrain = PatchTSTPretrainModel(_tiny_config())
    pretrain_state = pretrain.state_dict()
    pretrain_state["backbone.patch_embed.weight"].fill_(0.125)
    pretrain_state["backbone.patch_embed.bias"].fill_(-0.25)

    supervised_state = supervised.state_dict()
    protected_before = {
        key: value.detach().clone()
        for key, value in supervised_state.items()
        if key.startswith(PATCHTST_SUPERVISED_ONLY_PREFIXES)
    }
    assert protected_before
    for key, value in protected_before.items():
        pretrain_state[key] = torch.full_like(value, 99)

    checkpoint_path = tmp_path / "pretrain-with-supervised-keys.pt"
    torch.save(
        {
            "state_dict": pretrain_state,
            "best_val": 0.25,
        },
        checkpoint_path,
    )

    report = load_patchtst_pretrained_backbone(
        supervised,
        str(checkpoint_path),
        load_strict=False,
    )

    assert report["transfer_scope"] == "backbone_only"
    assert report["transferred_key_count"] > 0
    assert all(
        key.startswith("backbone.")
        for key in report["transferred_keys"]
    )
    assert report["supervised_only_prefixes"] == list(
        PATCHTST_SUPERVISED_ONLY_PREFIXES
    )
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
    for key, expected in protected_before.items():
        torch.testing.assert_close(
            supervised.state_dict()[key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_pretrain_transfer_rejects_checkpoint_without_backbone(
    tmp_path: Path,
) -> None:
    supervised = build_patchTST_exogenous(
        _tiny_config(future_cat_cardinalities=(3,))
    )
    checkpoint_path = tmp_path / "head-only-pretrain.pt"
    torch.save(
        {
            "state_dict": {
                "head.proj.weight": torch.zeros_like(
                    supervised.state_dict()["head.proj.weight"]
                )
            }
        },
        checkpoint_path,
    )

    with pytest.raises(
        RuntimeError,
        match="no compatible PatchTST backbone weights",
    ):
        load_patchtst_pretrained_backbone(
            supervised,
            str(checkpoint_path),
        )


def test_ssl_pretrain_extracts_only_target_history_from_seven_tuple() -> None:
    x = torch.randn(2, 4, 1)
    batch = (
        x,
        torch.randn(2, 2),
        torch.randn(2, 4, 1),
        torch.randn(2, 2, 1),
        ["series-1", "series-2"],
        torch.ones(2, 4, 1, dtype=torch.long),
        torch.ones(2, 2, 1, dtype=torch.long),
    )

    assert _extract_x(batch) is x


def test_ssl_validation_mask_is_repeatable_and_rng_is_isolated() -> None:
    class RandomMaskLoss(torch.nn.Module):
        def forward(
            self,
            x,
            *,
            mask_ratio,
            return_loss,
            loss_type,
        ):
            del mask_ratio, return_loss, loss_type
            return {"loss": torch.rand((), device=x.device)}

    model = RandomMaskLoss()
    loader = [
        torch.ones(2, 4, 1),
        torch.ones(2, 4, 1),
    ]
    torch.manual_seed(703)
    rng_before = torch.random.get_rng_state()

    first = _eval_pretrain(
        model,
        loader,
        torch.device("cpu"),
        mask_ratio=0.3,
        loss_type="mse",
        eval_seed=91,
    )
    rng_after_evaluation = torch.rand(())

    torch.random.set_rng_state(rng_before)
    expected_next_random = torch.rand(())
    second = _eval_pretrain(
        model,
        loader,
        torch.device("cpu"),
        mask_ratio=0.3,
        loss_type="mse",
        eval_seed=91,
    )

    assert first == pytest.approx(second)
    torch.testing.assert_close(
        rng_after_evaluation,
        expected_next_random,
        rtol=0.0,
        atol=0.0,
    )


def test_public_full_ssl_categorical_train_save_load_forecast_has_no_leakage(
    tmp_path: Path,
) -> None:
    torch.manual_seed(702)
    np.random.seed(702)
    frame = _categorical_frame()
    result = train(
        _full_training_request(
            frame=frame,
            save_dir=tmp_path / "patchtst-full-categorical",
        )
    )

    assert result.primary_ckpt_path is not None
    assert Path(result.primary_ckpt_path).is_file()
    assert result.pretrain_ckpt_paths.keys() == {
        "patchtst_exogenous"
    }
    pretrain_path = Path(
        result.pretrain_ckpt_paths["patchtst_exogenous"]
    )
    assert pretrain_path.is_file()

    load_report = result.results["patchtst_exogenous"][
        "pretrain_load_report"
    ]
    assert load_report["transfer_scope"] == "backbone_only"
    assert all(
        key.startswith("backbone.")
        for key in load_report["transferred_keys"]
    )

    datamodule = result.datamodule
    full_dataset = datamodule._full_dataset
    train_positions = full_dataset.source_row_positions_for_windows(
        datamodule.train_dataset.indices
    )["series-1"]
    validation_positions = full_dataset.source_row_positions_for_windows(
        datamodule.val_dataset.indices
    )["series-1"]
    validation_only = set(validation_positions).difference(
        train_positions
    )
    assert validation_only

    vocabulary_artifact = (
        datamodule.categorical_vocabulary_artifact
    )
    vocabulary = vocabulary_artifact.vocabulary_for("event_type")
    assert set(vocabulary.known_values) == {
        frame["event_type"][position]
        for position in train_positions
    }
    assert all(
        vocabulary.id_of(frame["event_type"][position]) == 0
        for position in validation_only
    )

    pretrain_checkpoint = torch.load(
        pretrain_path,
        map_location="cpu",
        weights_only=False,
    )
    assert pretrain_checkpoint["best_epoch"] == 1
    assert len(pretrain_checkpoint["history"]) == 1
    assert (
        pretrain_checkpoint["history"][0]["validation_loss"]
        == pytest.approx(pretrain_checkpoint["best_val"])
    )
    assert isinstance(
        pretrain_checkpoint["validation_mask_seed"],
        int,
    )
    assert not any(
        key.startswith(
            (
                "future_cat_embedding.",
                "future_fuser.",
            )
        )
        for key in pretrain_checkpoint["state_dict"]
    )

    final_checkpoint = torch.load(
        result.primary_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    assert any(
        key.startswith("future_cat_embedding.")
        for key in final_checkpoint["state_dict"]
    )
    assert any(
        key.startswith("future_fuser.")
        for key in final_checkpoint["state_dict"]
    )
    assert (
        final_checkpoint["data_artifacts"][
            "categorical_vocabulary_fingerprint"
        ]
        == vocabulary_artifact.fingerprint
    )

    predictor = load_predictor(
        result.primary_ckpt_path,
        device="cpu",
        strict=True,
    )
    assert (
        predictor.categorical_vocabulary_fingerprint
        == vocabulary_artifact.fingerprint
    )

    known_event = vocabulary.known_values[0]
    future = pl.DataFrame(
        {
            "unique_id": ["series-1", "series-1"],
            "date": [20240117, 20240118],
            "y": [0.0, 0.0],
            "event_type": [known_event, "new-operation-event"],
        }
    )
    prediction = forecast(
        ForecastRequest(
            checkpoint_path=result.primary_ckpt_path,
            expected_model_key="patchtst_exogenous",
            data=DataRequest(
                df=pl.concat([frame, future]),
                lookback=4,
                horizon=2,
                freq="daily",
                exogenous=ExogenousConfig(
                    use_exogenous_mode=True,
                    use_future_exogenous=True,
                    future_exo_cat_cols=["event_type"],
                ),
            ),
            series_ids=["series-1"],
            forecast_origin=20240117,
            runtime=ForecastRuntimeConfig(
                batch_size=1,
                num_workers=0,
                device="cpu",
                pin_memory=False,
            ),
        )
    )

    assert prediction.predictions.height == 2
    assert prediction.predictions["point"].is_finite().all()
