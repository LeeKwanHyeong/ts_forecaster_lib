from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from modeling_module import (
    ArchitectureConfig,
    ArtifactConfig,
    AutoTimesArchitectureConfig,
    DataRequest,
    ForecastRequest,
    ForecastRuntimeConfig,
    RuntimeConfig,
    SSLConfig,
    TrainRequest,
    TrainerConfig,
    load_predictor,
    forecast,
    train,
)
from modeling_module.models.AutoTimes import AutoTimesConfig, AutoTimesModel
from modeling_module.models.AutoTimes.timestamp_artifact import TimestampEmbeddingArtifact
from modeling_module.models.registry import (
    PRODUCTION_REFIT_ARTIFACT_KEYS,
    expand_training_targets,
    get_model_spec,
    infer_artifact_model_key_from_checkpoint,
)
from modeling_module.utils.checkpoint import save_model


def _config(horizon: int) -> AutoTimesConfig:
    return AutoTimesConfig(
        lookback=52,
        horizon=horizon,
        y_dim=1,
        token_len=13,
        backbone_type="mock",
        hidden_size=16,
        mock_layers=1,
        mock_heads=4,
        mlp_hidden_dim=16,
        mlp_hidden_layers=1,
        dropout=0.0,
        use_exogenous_mode=False,
        use_intermittent=False,
    )


@pytest.mark.parametrize("horizon", (26, 27))
def test_autotimes_mock_backbone_autoregressive_horizon_contract(horizon: int):
    torch.manual_seed(19)
    model = AutoTimesModel(_config(horizon)).eval()
    x = torch.linspace(1.0, 52.0, 52).reshape(1, 52, 1)

    with torch.inference_mode():
        output = model(x)

    assert output.shape == (1, horizon, 1)
    assert torch.isfinite(output).all()
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.tokenizer.parameters())
    assert model.backbone.training is False


class _BF16Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.dtype_anchor = torch.nn.Parameter(
            torch.zeros((), dtype=torch.bfloat16)
        )
        self.last_input_dtype = None

    def forward(self, *, inputs_embeds: torch.Tensor):
        self.last_input_dtype = inputs_embeds.dtype
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def test_autotimes_bridges_bfloat16_backbone_and_float32_tokenizers():
    backbone = _BF16Backbone()
    model = AutoTimesModel(_config(26), backbone=backbone).train()
    output = model(torch.randn(2, 52, 1))
    output.sum().backward()

    assert backbone.last_input_dtype == torch.bfloat16
    assert output.dtype == torch.float32
    assert output.shape == (2, 26, 1)
    assert any(
        parameter.grad is not None
        for parameter in model.tokenizer.parameters()
    )


def test_autotimes_timestamp_artifact_hash_and_window_contract(tmp_path: Path):
    artifact_path = tmp_path / "weekly_timestamp.pt"
    tensor = torch.randn(6, 16)
    torch.save({"timestamp_embeddings": tensor}, artifact_path)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    artifact = TimestampEmbeddingArtifact.load(artifact_path, digest)
    cfg = _config(27)
    cfg.timestamp_artifact_path = str(artifact_path)
    cfg.timestamp_artifact_sha256 = digest
    model = AutoTimesModel(cfg).eval()

    with torch.inference_mode():
        output = model(torch.randn(2, 52, 1))

    assert artifact.sha256 == digest
    assert output.shape == (2, 27, 1)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        TimestampEmbeddingArtifact.load(artifact_path, "0" * 64)


def test_autotimes_strict_checkpoint_restore(tmp_path: Path):
    torch.manual_seed(23)
    model = AutoTimesModel(_config(26)).eval()
    checkpoint_path = tmp_path / "autotimes_base.pt"
    x = torch.randn(2, 52, 1)
    save_model(
        model,
        model.cfg,
        str(checkpoint_path),
        extra_meta={"model_key": "autotimes_base", "family_key": "autotimes"},
    )

    predictor = load_predictor(str(checkpoint_path), device="cpu", strict=True)
    with torch.inference_mode():
        expected = model(x)
        restored = predictor.model(x)

    assert predictor.model_key == "autotimes_base"
    assert torch.equal(expected, restored)
    assert infer_artifact_model_key_from_checkpoint(
        {"model_class": "AutoTimesModel"}
    ) == "autotimes_base"


def test_autotimes_registry_contract():
    spec = get_model_spec("autotimes_base")

    assert expand_training_targets(["autotimes"]) == ["autotimes_base"]
    assert spec.family == "autotimes"
    assert spec.exogenous_policy == "none"
    assert spec.fusion_strategy == "frozen_llm_numeric_tokens"
    assert "autotimes_base" not in PRODUCTION_REFIT_ARTIFACT_KEYS


def test_autotimes_icl_exogenous_config_requires_enabled_sha256_contract():
    with pytest.raises(ValueError, match="icl_enabled=True"):
        AutoTimesConfig(
            lookback=52,
            horizon=26,
            token_len=13,
            backbone_type="mock",
            icl_past_exogenous_dim=2,
            icl_future_exogenous_dim=1,
            icl_exogenous_schema_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA256"):
        AutoTimesConfig(
            lookback=52,
            horizon=26,
            token_len=13,
            backbone_type="mock",
            icl_enabled=True,
            icl_past_exogenous_dim=2,
            icl_future_exogenous_dim=1,
            icl_exogenous_schema_hash="not-a-sha",
        )


def test_public_train_api_builds_autotimes_checkpoint(tmp_path: Path):
    torch.manual_seed(29)
    np.random.seed(29)
    x = torch.randn(4, 52, 1)
    y = torch.randn(4, 26, 1)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    val_loader = DataLoader(TensorDataset(x[:2], y[:2]), batch_size=2, shuffle=False)

    result = train(
        TrainRequest(
            train_loader=train_loader,
            val_loader=val_loader,
            models=["autotimes_base"],
            freq="weekly",
            lookback=52,
            horizon=26,
            use_exogenous_mode=False,
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                use_intermittent=False,
                val_use_weights=False,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(save_dir=str(tmp_path), auto_save_dir=False),
            architecture=ArchitectureConfig(
                autotimes=AutoTimesArchitectureConfig(
                    token_len=13,
                    backbone_type="mock",
                    hidden_size=16,
                    mock_layers=1,
                    mock_heads=4,
                    mlp_hidden_dim=16,
                    dropout=0.0,
                )
            ),
        )
    )

    assert result.requested_models == ("autotimes_base",)
    assert result.primary_ckpt_path is not None
    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    prediction = predictor.predict(x[:1])
    assert predictor.model_key == "autotimes_base"
    assert np.asarray(prediction["point"]).shape == (26,)


def test_public_forecast_api_runs_autotimes_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "autotimes_base.pt"
    model = AutoTimesModel(_config(26)).eval()
    save_model(
        model,
        model.cfg,
        str(checkpoint_path),
        extra_meta={"model_key": "autotimes_base", "family_key": "autotimes"},
    )
    start = date(2025, 1, 1)
    rows = [
        {
            "unique_id": "part-1",
            "date": int((start + timedelta(days=offset)).strftime("%Y%m%d")),
            "y": float(offset + 1),
        }
        for offset in range(52)
    ]
    origin = int((start + timedelta(days=52)).strftime("%Y%m%d"))

    result = forecast(
        ForecastRequest(
            checkpoint_path=checkpoint_path,
            expected_model_key="autotimes_base",
            data=DataRequest(
                df=pl.DataFrame(rows),
                lookback=52,
                horizon=26,
                freq="daily",
            ),
            series_ids=["part-1"],
            forecast_origin=origin,
            runtime=ForecastRuntimeConfig(
                batch_size=1,
                num_workers=0,
                device="cpu",
                pin_memory=False,
            ),
        )
    )

    assert result.model_key == "autotimes_base"
    assert result.predictions.height == 26
    assert result.predictions["horizon_step"].to_list() == list(range(26))
    assert result.predictions["point"].is_finite().all()


def test_autotimes_upstream_manifest_is_pinned_and_excludes_benchmark_runtime():
    root = Path(__file__).parents[1] / "src/modeling_module/models/AutoTimes"
    manifest = json.loads((root / "upstream_manifest.json").read_text(encoding="utf-8"))

    assert manifest["upstream"]["commit"] == "9ff9aac5083e24c233404c35d7b7a3c0643f2c70"
    assert manifest["license"]["spdx"] == "MIT"
    assert hashlib.sha256((root / "LICENSE.upstream").read_bytes()).hexdigest() == (
        "29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849"
    )
    assert "upstream benchmark CLI" in manifest["product_scope"]["excluded"]
    assert not (root / "run.py").exists()
    assert not (root / "data_provider").exists()
