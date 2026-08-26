from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from modeling_module.api import load_predictor
from modeling_module.api.icl import (
    ICLForecastRequest,
    ICLForecastRuntimeConfig,
    forecast_icl,
)
from modeling_module.data_loader import ICLEpisodeDataModule
from modeling_module.icl import (
    EndogenousICLBuilderConfig,
    EndogenousICLDatasetBuilder,
    ICLSplit,
    ICLTrainerConfig,
    write_icl_episode_artifact,
)
from modeling_module.models.AutoTimes import AutoTimesConfig, AutoTimesModel
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.training.model_trainers.autotimes_train import train_autotimes_icl
from modeling_module.training.model_trainers.sellm_train import (
    fit_sellm_validation_scalar_calibration,
    train_sellm_icl,
)
from modeling_module.utils.checkpoint import save_model


def _week(start: date, offset: int) -> int:
    iso = (start + timedelta(weeks=offset)).isocalendar()
    return int(iso.year) * 100 + int(iso.week)


def _bundle():
    start = date.fromisocalendar(2022, 1, 1)
    frame = pl.DataFrame(
        {
            "oper_part_no": ["part-1"] * 52,
            "demand_dt": [_week(start, offset) for offset in range(52)],
            "demand_qty": [float(20 + offset % 7) for offset in range(52)],
        }
    )
    builder = EndogenousICLDatasetBuilder(
        EndogenousICLBuilderConfig(
            lookback=8,
            horizon=4,
            seasonal_period=12,
            window_stride=4,
            validation_episodes_per_series=1,
            test_episodes_per_series=1,
        )
    )
    return builder.build(frame, source_revision="icl-execution-r1")


def _module():
    return ICLEpisodeDataModule(_bundle(), batch_size=2, seed=31)


def _autotimes_config(*, icl_enabled: bool = True) -> AutoTimesConfig:
    return AutoTimesConfig(
        lookback=8,
        horizon=4,
        y_dim=1,
        token_len=4,
        backbone_type="mock",
        hidden_size=8,
        mock_layers=1,
        mock_heads=2,
        mlp_hidden_dim=8,
        mlp_hidden_layers=1,
        dropout=0.0,
        mix_timestamp_embeddings=False,
        icl_enabled=icl_enabled,
        use_exogenous_mode=False,
        use_intermittent=False,
    )


def _sellm_config(*, icl_enabled: bool = True, **overrides) -> SELLMConfig:
    values = dict(
        lookback=8,
        horizon=4,
        y_dim=1,
        architecture_variant="paper_v1",
        token_len=4,
        d_model=8,
        n_heads=2,
        dropout=0.0,
        mlp_hidden_dim=8,
        semantic_vocab_size=8,
        semantic_top_k=2,
        tscc_latent_dim=2,
        tscc_hidden_dim=8,
        use_pretrained_llm=False,
        fallback_layers=1,
        d_ff=16,
        head_hidden_dim=8,
        use_norm=True,
        icl_enabled=icl_enabled,
        use_exogenous_mode=False,
        use_intermittent=False,
    )
    values.update(overrides)
    return SELLMConfig(**values)


def _save_checkpoint(model, path: Path) -> None:
    save_model(
        model,
        model.cfg,
        str(path),
        extra_meta={
            "model_key": model.model_key,
            "family_key": model.model_key.removesuffix("_base"),
        },
    )


def test_autotimes_icl_trainer_and_artifact_forecast_round_trip(tmp_path: Path):
    torch.manual_seed(37)
    module = _module()
    model = AutoTimesModel(_autotimes_config())
    result = train_autotimes_icl(
        model,
        module.loader(ICLSplit.TRAIN, shuffle=False),
        module.loader(ICLSplit.VALIDATION, shuffle=False),
        trainer_config=ICLTrainerConfig(epochs=1, lr=1e-3, device="cpu"),
    )
    assert result.epochs_completed == 1
    assert result.best_validation_loss is not None
    assert len(result.epoch_history) == 1
    assert result.epoch_history[0]["epoch"] == 1
    assert result.epoch_history[0]["validation_mae"] is not None
    assert result.epoch_history[0]["validation_wape"] is not None
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())

    artifact_dir = tmp_path / "episodes"
    checkpoint_path = tmp_path / "autotimes_icl.pt"
    receipt = write_icl_episode_artifact(module.bundle, artifact_dir)
    _save_checkpoint(result.model, checkpoint_path)
    forecast = forecast_icl(
        ICLForecastRequest(
            checkpoint_path=checkpoint_path,
            episode_artifact_dir=artifact_dir,
            expected_model_key="autotimes_base",
            runtime=ICLForecastRuntimeConfig(batch_size=1, device="cpu"),
        )
    )

    assert forecast.manifest_hash == receipt.manifest_hash
    assert forecast.predictions.height == 4
    assert forecast.predictions["horizon_step"].to_list() == [0, 1, 2, 3]
    assert forecast.predictions["point"].is_finite().all()


def test_sellm_icl_trainer_updates_semantic_encoder_and_forecasts(tmp_path: Path):
    torch.manual_seed(41)
    module = _module()
    model = SELLMModel(_sellm_config())
    before = {
        name: value.detach().clone()
        for name, value in model.icl_prompt_encoder.state_dict().items()
    }
    result = train_sellm_icl(
        model,
        module.loader(ICLSplit.TRAIN, shuffle=False),
        module.loader(ICLSplit.VALIDATION, shuffle=False),
        trainer_config=ICLTrainerConfig(epochs=1, lr=1e-3, device="cpu"),
    )
    after = result.model.icl_prompt_encoder.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)

    artifact_dir = tmp_path / "episodes"
    checkpoint_path = tmp_path / "sellm_icl.pt"
    write_icl_episode_artifact(module.bundle, artifact_dir)
    _save_checkpoint(result.model, checkpoint_path)
    forecast = forecast_icl(
        ICLForecastRequest(
            checkpoint_path=checkpoint_path,
            episode_artifact_dir=artifact_dir,
            expected_model_key="sellm_base",
            runtime=ICLForecastRuntimeConfig(batch_size=1, device="cpu"),
        )
    )

    assert forecast.predictions.height == 4
    assert forecast.predictions["model_key"].unique().to_list() == ["sellm_base"]
    assert forecast.predictions["point"].is_finite().all()


def test_sellm_validation_scalar_uses_validation_only_and_restores_strictly(
    tmp_path: Path,
):
    torch.manual_seed(43)
    module = _module()
    model = SELLMModel(
        _sellm_config(
            output_head_mode="softplus",
            output_head_softplus_beta=8.0,
            output_calibration_mode="validation_scalar",
            output_calibration_min_scale=1e-6,
            output_calibration_max_scale=100.0,
        )
    )

    with pytest.raises(ValueError, match="validation episodes only"):
        fit_sellm_validation_scalar_calibration(
            model,
            module.loader(ICLSplit.TEST, shuffle=False),
            device="cpu",
        )

    result = train_sellm_icl(
        model,
        module.loader(ICLSplit.TRAIN, shuffle=False),
        module.loader(ICLSplit.VALIDATION, shuffle=False),
        trainer_config=ICLTrainerConfig(epochs=1, lr=1e-3, device="cpu"),
    )
    contract = result.model.output_calibration_contract()
    stats = result.model.output_calibration_fit_stats
    assert contract["mode"] == "validation_scalar"
    assert contract["fitted"] is True
    assert contract["source_split"] == "validation"
    assert len(str(contract["source_fingerprint"])) == 64
    assert 0.5 <= float(contract["scale"]) <= 1.5
    assert stats["episode_count"] == 1
    assert stats["point_count"] == 4
    assert abs(float(stats["validation_bias_after"])) < 1e-6

    path = tmp_path / "sellm-validation-calibrated.pt"
    _save_checkpoint(result.model, path)
    predictor = load_predictor(str(path), device="cpu", strict=True)
    batch = next(iter(module.loader(ICLSplit.TEST, shuffle=False)))
    with torch.no_grad():
        expected = result.model.forward_icl(
            demonstration_contexts=batch.demonstration_contexts,
            demonstration_targets=batch.demonstration_targets,
            query_context=batch.query_context,
            prompt_mask=batch.prompt_mask,
        )
        restored = predictor.model.forward_icl(
            demonstration_contexts=batch.demonstration_contexts,
            demonstration_targets=batch.demonstration_targets,
            query_context=batch.query_context,
            prompt_mask=batch.prompt_mask,
        )
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("model", "call"),
    [
        (
            AutoTimesModel(_autotimes_config(icl_enabled=False)),
            lambda model, batch: model.forward_icl(
                torch.cat(
                    [
                        batch.demonstration_contexts.reshape(1, -1, 1),
                        batch.demonstration_targets.reshape(1, -1, 1),
                        batch.query_context,
                    ],
                    dim=1,
                ),
                prompt_mask=batch.prompt_mask,
            ),
        ),
        (
            SELLMModel(_sellm_config(icl_enabled=False)),
            lambda model, batch: model.forward_icl(
                demonstration_contexts=batch.demonstration_contexts,
                demonstration_targets=batch.demonstration_targets,
                query_context=batch.query_context,
                prompt_mask=batch.prompt_mask,
            ),
        ),
    ],
)
def test_icl_execution_rejects_non_icl_checkpoint(model, call):
    batch = next(iter(_module().loader(ICLSplit.TEST, shuffle=False)))
    with pytest.raises(RuntimeError, match="not configured for ICL"):
        call(model, batch)
