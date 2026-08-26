from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from modeling_module.api.infer import load_predictor
from modeling_module.api.icl import ICLForecastRequest, ICLForecastRuntimeConfig, forecast_icl
from modeling_module.api.train import (
    ArchitectureConfig,
    AutoTimesArchitectureConfig,
    SELLMArchitectureConfig,
    _normalize_model_architecture,
)
from modeling_module.data_loader import ICLEpisodeDataModule
from modeling_module.icl import (
    AutoTimesICLAdapter,
    EndogenousICLBuilderConfig,
    ExogenousICLBuilderConfig,
    ExogenousICLDatasetBuilder,
    ExogenousICLInferenceBuilder,
    ICLInferenceBuilderConfig,
    ICLSplit,
    ICLTrainerConfig,
    SELLMICLAdapter,
    read_icl_episode_artifact,
    write_icl_episode_artifact,
    save_icl_production_checkpoint,
)
from modeling_module.models.AutoTimes import AutoTimesConfig, AutoTimesModel
from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import SELLMConfig
from modeling_module.training.model_trainers.autotimes_train import train_autotimes_icl
from modeling_module.training.model_trainers.sellm_train import train_sellm_icl
from modeling_module.utils.checkpoint import save_model


def _week(start: date, offset: int) -> int:
    iso = (start + timedelta(weeks=offset)).isocalendar()
    return int(iso.year) * 100 + int(iso.week)


def _bundle():
    start = date.fromisocalendar(2022, 1, 1)
    offsets = list(range(52))
    frame = pl.DataFrame(
        {
            "oper_part_no": ["part-1"] * len(offsets),
            "demand_dt": [_week(start, offset) for offset in offsets],
            "demand_qty": [float(20 + offset % 7) for offset in offsets],
            "promo_observed": [float(offset % 3 == 0) for offset in offsets],
            "calendar_cycle": [float((offset % 12) / 11) for offset in offsets],
        }
    )
    return ExogenousICLDatasetBuilder(
        ExogenousICLBuilderConfig(
            episode=EndogenousICLBuilderConfig(
                lookback=8,
                horizon=4,
                seasonal_period=12,
                window_stride=4,
                validation_episodes_per_series=1,
                test_episodes_per_series=1,
            ),
            past_feature_cols=("promo_observed", "calendar_cycle"),
            future_feature_cols=("calendar_cycle",),
        )
    ).build(
        frame,
        source_revision="demand-r1",
        exogenous_source_revision="approved-exogenous-r1",
    )


def _save(model, path: Path) -> None:
    save_model(
        model,
        model.cfg,
        str(path),
        extra_meta={
            "model_key": model.model_key,
            "family_key": model.model_key.removesuffix("_base"),
        },
    )


def _autotimes_config(schema_hash: str) -> AutoTimesConfig:
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
        icl_enabled=True,
        icl_past_exogenous_dim=2,
        icl_future_exogenous_dim=1,
        icl_exogenous_schema_hash=schema_hash,
        use_exogenous_mode=False,
        use_intermittent=False,
    )


def _sellm_config(schema_hash: str) -> SELLMConfig:
    return SELLMConfig(
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
        icl_enabled=True,
        icl_past_exogenous_dim=2,
        icl_future_exogenous_dim=1,
        icl_exogenous_schema_hash=schema_hash,
        use_exogenous_mode=False,
        use_intermittent=False,
    )


def test_exogenous_episode_artifact_seals_role_specific_schema(tmp_path: Path):
    bundle = _bundle()
    schema = bundle.manifest.exogenous_schema
    assert schema is not None
    assert schema.past_feature_names == ("promo_observed", "calendar_cycle")
    assert schema.future_feature_names == ("calendar_cycle",)

    module = ICLEpisodeDataModule(bundle, batch_size=1)
    batch = next(iter(module.loader(ICLSplit.TEST, shuffle=False)))
    assert batch.query_context_exogenous.shape == (1, 8, 2)
    assert batch.query_target_exogenous.shape == (1, 4, 1)

    autotimes = AutoTimesICLAdapter().adapt(batch)
    sellm = SELLMICLAdapter().adapt(batch)
    assert autotimes.packed_exogenous is not None
    assert autotimes.packed_exogenous.shape == (1, 32, 3)
    assert torch.equal(
        autotimes.packed_exogenous[:, -8:, 2],
        torch.zeros(1, 8),
    )
    assert sellm.demonstration_context_exogenous.shape == (1, 2, 8, 2)
    assert sellm.demonstration_target_exogenous.shape == (1, 2, 4, 1)

    artifact_dir = tmp_path / "exogenous-episodes"
    receipt = write_icl_episode_artifact(bundle, artifact_dir)
    loaded, loaded_receipt = read_icl_episode_artifact(artifact_dir)
    assert loaded == bundle
    assert loaded_receipt.manifest_hash == receipt.manifest_hash


def test_public_architecture_contract_routes_icl_schema_to_each_model_family():
    schema_hash = "a" * 64
    normalized = _normalize_model_architecture(
        ArchitectureConfig(
            autotimes=AutoTimesArchitectureConfig(
                icl_enabled=True,
                icl_past_exogenous_dim=2,
                icl_future_exogenous_dim=1,
                icl_exogenous_schema_hash=schema_hash,
            ),
            sellm=SELLMArchitectureConfig(
                icl_enabled=True,
                icl_past_exogenous_dim=2,
                icl_future_exogenous_dim=1,
                icl_exogenous_schema_hash=schema_hash,
            ),
        )
    )

    assert normalized == {
        "autotimes": {
            "icl_enabled": True,
            "icl_past_exogenous_dim": 2,
            "icl_future_exogenous_dim": 1,
            "icl_exogenous_schema_hash": schema_hash,
        },
        "sellm": {
            "icl_enabled": True,
            "icl_past_exogenous_dim": 2,
            "icl_future_exogenous_dim": 1,
            "icl_exogenous_schema_hash": schema_hash,
        },
    }


def test_sellm_icl_exogenous_config_requires_enabled_sha256_contract():
    config = _sellm_config("a" * 64)

    with pytest.raises(ValueError, match="requires icl_enabled=True"):
        replace(config, icl_enabled=False)
    with pytest.raises(ValueError, match="lowercase SHA256"):
        replace(config, icl_exogenous_schema_hash="not-a-sha256")


@pytest.mark.parametrize("model_key", ["autotimes_base", "sellm_base"])
def test_exogenous_icl_trains_saves_loads_and_forecasts(model_key: str, tmp_path: Path):
    torch.manual_seed(73)
    bundle = _bundle()
    schema = bundle.manifest.exogenous_schema
    assert schema is not None
    module = ICLEpisodeDataModule(bundle, batch_size=2, seed=73)
    train_loader = module.loader(ICLSplit.TRAIN, shuffle=False)
    validation_loader = module.loader(ICLSplit.VALIDATION, shuffle=False)
    trainer_config = ICLTrainerConfig(epochs=1, lr=1e-3, device="cpu")

    if model_key == "autotimes_base":
        model = AutoTimesModel(_autotimes_config(schema.fingerprint))
        result = train_autotimes_icl(
            model,
            train_loader,
            validation_loader,
            trainer_config=trainer_config,
        )
    else:
        model = SELLMModel(_sellm_config(schema.fingerprint))
        result = train_sellm_icl(
            model,
            train_loader,
            validation_loader,
            trainer_config=trainer_config,
        )

    artifact_dir = tmp_path / f"{model_key}-episodes"
    checkpoint_path = tmp_path / f"{model_key}.pt"
    write_icl_episode_artifact(bundle, artifact_dir)
    _save(result.model, checkpoint_path)
    forecast = forecast_icl(
        ICLForecastRequest(
            checkpoint_path=checkpoint_path,
            episode_artifact_dir=artifact_dir,
            expected_model_key=model_key,
            runtime=ICLForecastRuntimeConfig(batch_size=1, device="cpu"),
        )
    )
    assert forecast.predictions.height == 4
    assert forecast.predictions["point"].is_finite().all()


def test_exogenous_icl_rejects_checkpoint_schema_mismatch():
    bundle = _bundle()
    module = ICLEpisodeDataModule(bundle, batch_size=2)
    model = AutoTimesModel(_autotimes_config("0" * 64))

    with pytest.raises(ValueError, match="schema hash differ"):
        train_autotimes_icl(
            model,
            module.loader(ICLSplit.TRAIN, shuffle=False),
            trainer_config=ICLTrainerConfig(epochs=1, device="cpu"),
        )


def test_autotimes_production_refit_saves_final_epoch_contract(tmp_path: Path):
    source = _bundle()
    train_only = replace(
        source,
        episodes=tuple(
            replace(item, split=ICLSplit.TRAIN)
            for item in source.episodes
        ),
        manifest=replace(
            source.manifest,
            split_counts={
                "train": len(source.episodes),
                "validation": 0,
                "test": 0,
            },
            episode_hashes=tuple(
                replace(item, split=ICLSplit.TRAIN).episode_hash
                for item in source.episodes
            ),
        ),
    )
    # Recreate the manifest because split identity is part of its seal.
    from modeling_module.icl.contracts import ICLManifest

    train_only = replace(
        train_only,
        manifest=ICLManifest.create(
            dataset_kind="exogenous",
            source_revision=source.manifest.source_revision,
            source_hash=source.manifest.source_hash,
            config_hash=source.manifest.config_hash,
            source_min_week=source.manifest.source_min_week,
            source_max_week=source.manifest.source_max_week,
            series_count=source.manifest.series_count,
            episodes=train_only.episodes,
            exogenous_schema=source.manifest.exogenous_schema,
        ),
    )
    schema = train_only.manifest.exogenous_schema
    assert schema is not None
    config = replace(
        _autotimes_config(schema.fingerprint),
        llm_revision="qwen-revision-r1",
    )
    model = AutoTimesModel(config)
    module = ICLEpisodeDataModule(train_only, batch_size=2, seed=42)
    trainer_config = ICLTrainerConfig(
        epochs=2,
        lr=1e-3,
        device="cpu",
        training_mode="production_refit",
    )

    with pytest.raises(ValueError, match="requires val_loader=None"):
        train_autotimes_icl(
            model,
            module.loader(ICLSplit.TRAIN, shuffle=False),
            module.loader(ICLSplit.TRAIN, shuffle=False),
            trainer_config=trainer_config,
        )

    result = train_autotimes_icl(
        model,
        module.loader(ICLSplit.TRAIN, shuffle=False),
        trainer_config=trainer_config,
    )
    final_state = {
        name: value.detach().clone()
        for name, value in result.model.state_dict().items()
    }
    assert result.training_mode == "production_refit"
    assert result.validation_enabled is False
    assert result.state_selection == "final_epoch"
    assert result.epochs_completed == 2

    cutoff = max(item.query_target.end_week for item in train_only.episodes)
    with pytest.raises(ValueError, match="complete eligible series set"):
        save_icl_production_checkpoint(
            result,
            tmp_path / "partial.pt",
            model_key="autotimes_base",
            bundle=train_only,
            trainer_config=trainer_config,
            random_seed=42,
            data_cutoff=cutoff,
            eligible_series_count=train_only.manifest.series_count + 1,
            backbone_contract={
                "model_id": "Qwen/Qwen2-0.5B",
                "revision": "qwen-revision-r1",
                "manifest_sha256": "a" * 64,
                "contract_sha256": "b" * 64,
            },
        )
    checkpoint_path = save_icl_production_checkpoint(
        result,
        tmp_path / "weekly_AutoTimesBase_L8_H4.pt",
        model_key="autotimes_base",
        bundle=train_only,
        trainer_config=trainer_config,
        random_seed=42,
        data_cutoff=cutoff,
        eligible_series_count=train_only.manifest.series_count,
        backbone_contract={
            "model_id": "Qwen/Qwen2-0.5B",
            "revision": "qwen-revision-r1",
            "manifest_sha256": "a" * 64,
            "contract_sha256": "b" * 64,
        },
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["meta"]["training_mode"] == "production_refit"
    assert payload["meta"]["validation_enabled"] is False
    assert payload["meta"]["state_selection"] == "final_epoch"
    assert payload["meta"]["random_seed"] == 42
    assert payload["meta"]["train_data_cutoff"] == cutoff
    assert payload["meta"]["eligible_series_count"] == (
        train_only.manifest.series_count
    )
    assert payload["meta"]["episode_schema_hash"] == schema.fingerprint
    assert payload["meta"]["backbone_contract"]["revision"] == (
        "qwen-revision-r1"
    )
    assert payload["config"]["training_mode"] == "production_refit"
    assert payload["config"]["random_seed"] == 42
    assert payload["config"]["epochs"] == 2
    assert payload["config"]["lr"] == pytest.approx(1e-3)
    assert payload["config"]["weight_decay"] == pytest.approx(0.0)
    assert payload["cfg_state"] == payload["config"]
    assert result.model.cfg.training_mode == "qualification"
    assert result.model.cfg.random_seed is None
    assert result.model.cfg.epochs == 1
    assert payload["state_dict"].keys() == final_state.keys()
    for name, value in final_state.items():
        torch.testing.assert_close(
            payload["state_dict"][name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    predictor = load_predictor(checkpoint_path, device="cpu", strict=True)
    restored_batch = next(
        iter(module.loader(ICLSplit.TRAIN, shuffle=False))
    )
    inputs = AutoTimesICLAdapter().adapt(restored_batch)
    with torch.inference_mode():
        expected = result.model.forward_icl(
            inputs.packed_context,
            prompt_mask=inputs.prompt_mask,
            packed_exogenous=inputs.packed_exogenous,
            query_target_exogenous=inputs.query_target_exogenous,
        )
        restored = predictor.model.forward_icl(
            inputs.packed_context,
            prompt_mask=inputs.prompt_mask,
            packed_exogenous=inputs.packed_exogenous,
            query_target_exogenous=inputs.query_target_exogenous,
        )
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


def test_inference_episode_has_no_future_label_and_filters_inactive_series(
    tmp_path: Path,
):
    start = date.fromisocalendar(2022, 1, 1)
    offsets = list(range(52))
    rows = []
    for series_id in ("active-part", "ended-part"):
        for offset in offsets:
            rows.append(
                {
                    "oper_part_no": series_id,
                    "demand_dt": _week(start, offset),
                    "demand_qty": float(5 + offset % 4),
                    "promo_observed": float(offset % 3 == 0),
                    "calendar_cycle": float((offset % 12) / 11),
                }
            )
    history = pl.DataFrame(rows)
    origin = _week(start, 52)
    future = pl.DataFrame(
        [
            {
                "oper_part_no": series_id,
                "demand_dt": _week(start, offset),
                "calendar_cycle": float((offset % 12) / 11),
            }
            for series_id in ("active-part", "ended-part")
            for offset in range(52, 56)
        ]
    )
    builder = ExogenousICLInferenceBuilder(
        ICLInferenceBuilderConfig(
            lookback=8,
            horizon=4,
            demonstration_stride=4,
            seasonal_period=12,
            past_feature_cols=("promo_observed", "calendar_cycle"),
            future_feature_cols=("calendar_cycle",),
        )
    )
    bundle = builder.build(
        history,
        future,
        active_series_ids=("active-part",),
        forecast_origin=origin,
        source_revision="operational-demand-r1",
        exogenous_source_revision="operational-exogenous-r1",
    )

    assert bundle.manifest.series_count == 1
    assert bundle.manifest.split_counts["inference"] == 1
    episode = bundle.for_split(ICLSplit.INFERENCE)[0]
    assert episode.series_id == "active-part"
    assert episode.query_target_observed is False
    assert len(episode.demonstrations) == 2
    assert set(episode.query_target.target) == {(0.0,)}
    assert episode.query_target.start_week == origin
    assert all(
        prompt.end_week < episode.query_context.start_week
        for prompt in episode.demonstrations
    )

    artifact_dir = tmp_path / "inference-episodes"
    write_icl_episode_artifact(bundle, artifact_dir)
    restored, _ = read_icl_episode_artifact(artifact_dir)
    assert restored == bundle

    schema = bundle.manifest.exogenous_schema
    assert schema is not None
    model = AutoTimesModel(_autotimes_config(schema.fingerprint))
    checkpoint_path = tmp_path / "autotimes-inference.pt"
    _save(model, checkpoint_path)
    forecast = forecast_icl(
        ICLForecastRequest(
            checkpoint_path=checkpoint_path,
            episode_artifact_dir=artifact_dir,
            expected_model_key="autotimes_base",
            split=ICLSplit.INFERENCE,
            runtime=ICLForecastRuntimeConfig(batch_size=1, device="cpu"),
        )
    )
    assert forecast.predictions.height == 4
    assert forecast.predictions["series_id"].unique().to_list() == [
        "active-part"
    ]
