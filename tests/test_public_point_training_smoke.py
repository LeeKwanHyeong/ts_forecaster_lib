from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from modeling_module import (
    ArtifactConfig,
    ArchitectureConfig,
    DataRequest,
    DataWindowConfig,
    ExogenousConfig,
    ExoTSTArchitectureConfig,
    LoaderConfig,
    PatchMixerArchitectureConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TimexerArchitectureConfig,
    TitanArchitectureConfig,
    TrainerConfig,
    TrainRequest,
    load_predictor,
    train,
)
from modeling_module.training.forecater import _infer_d_future_expected


def _tiny_monthly_frame(n_rows: int = 4) -> pl.DataFrame:
    exogenous_values = [0.0, 0.5, 1.0, 0.5]
    return pl.DataFrame(
        {
            "unique_id": ["series-1"] * n_rows,
            "date": [202401 + idx for idx in range(n_rows)],
            "y": [1.0 + 0.5 * idx for idx in range(n_rows)],
            "exo_known": [exogenous_values[idx % len(exogenous_values)] for idx in range(n_rows)],
        }
    )


def _data_request(
    exogenous: ExogenousConfig | None,
    *,
    lookback: int = 2,
    n_rows: int = 4,
) -> DataRequest:
    return DataRequest(
        df=_tiny_monthly_frame(n_rows),
        backend="exo",
        window=DataWindowConfig(lookback=lookback, horizon=1, freq="monthly"),
        exogenous=exogenous,
        loader=LoaderConfig(
            batch_size=1,
            val_ratio=0.5,
            shuffle=False,
            seed=7,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        ),
    )


def _prediction_payload(
    model_key: str,
    *,
    lookback: int = 2,
) -> torch.Tensor | dict[str, torch.Tensor]:
    first = 1.0 + 0.5 * torch.arange(lookback, dtype=torch.float32)
    x = torch.stack((first, first + 0.5), dim=0).unsqueeze(-1)
    if model_key not in {"exotst_base", "timexer_base"}:
        return x

    payload = {
        "x": x,
        "past_exo_cont": torch.stack(
            (
                0.5 * torch.arange(lookback, dtype=torch.float32),
                0.5 * torch.arange(1, lookback + 1, dtype=torch.float32),
            ),
            dim=0,
        ).unsqueeze(-1),
    }
    if model_key == "exotst_base":
        payload["future_exo_batch"] = torch.tensor([[[1.0]], [[0.5]]])
    return payload


def _tiny_patchtst_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
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
            future_exo_fusion_dropout=0.0,
        )
    )


def _tiny_patchmixer_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        patchmixer=PatchMixerArchitectureConfig(
            patch_len=2,
            stride=1,
            d_model=4,
            e_layers=1,
            f_out=4,
            head_hidden=4,
            dropout=0.0,
            head_dropout=0.0,
            use_revin=False,
            final_nonneg=False,
            expander_n_harmonics=1,
        )
    )


def _tiny_titan_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        titan=TitanArchitectureConfig(
            d_model=4,
            n_layers=1,
            n_heads=1,
            d_ff=8,
            dropout=0.0,
            contextual_mem_size=0,
            persistent_mem_size=0,
            use_revin=False,
            final_clamp_nonneg=False,
        )
    )


def _tiny_exotst_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        exotst=ExoTSTArchitectureConfig(
            patch_len=2,
            stride=1,
            d_model=4,
            n_heads=2,
            d_ff=8,
            dropout=0.0,
            attn_dropout=0.0,
            exo_enc_layers=1,
            fusion_layers=1,
            endo_dec_layers=1,
            exo_memory_mode="agg",
            exo_nan_policy="zero",
            use_revin=False,
            subtract_last=False,
        )
    )


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)


POINT_SMOKE_CASES = [
    pytest.param(
        "patchtst_base",
        _tiny_patchtst_architecture(),
        None,
        id="patchtst",
    ),
    pytest.param(
        "patchmixer_base",
        _tiny_patchmixer_architecture(),
        None,
        id="patchmixer",
    ),
    pytest.param(
        "titan_base",
        _tiny_titan_architecture(),
        None,
        id="titan",
    ),
    pytest.param(
        "exotst_base",
        _tiny_exotst_architecture(),
        ExogenousConfig(
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=True,
            past_exo_cont_cols=["exo_known"],
            future_exo_cont_cols=["exo_known"],
        ),
        id="exotst",
    ),
    pytest.param(
        "timexer_base",
        ArchitectureConfig(
            timexer=TimexerArchitectureConfig(
                patch_len=2,
                d_model=4,
                n_heads=1,
                d_ff=8,
                e_layers=1,
                dropout=0.0,
                factor=1,
                activation="gelu",
                use_norm=False,
            )
        ),
        ExogenousConfig(
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=False,
            past_exo_cont_cols=["exo_known"],
        ),
        id="timexer",
    ),
]


@pytest.mark.parametrize("model_key,architecture,exogenous", POINT_SMOKE_CASES)
def test_public_point_train_checkpoint_load_predict_smoke(
    tmp_path: Path,
    model_key: str,
    architecture: ArchitectureConfig,
    exogenous: ExogenousConfig | None,
):
    torch.manual_seed(7)
    np.random.seed(7)
    artifact_dir = tmp_path / model_key

    result = train(
        TrainRequest(
            data=_data_request(exogenous),
            models=[model_key],
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                use_intermittent=False,
                val_use_weights=False,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(save_dir=str(artifact_dir), auto_save_dir=False),
            architecture=architecture,
        )
    )

    assert result.requested_models == (model_key,)
    assert result.primary_result_name == model_key
    assert result.primary_ckpt_path is not None
    assert result.primary_ckpt_path == result.ckpt_paths[model_key]
    assert result.best_ckpt_path == result.primary_ckpt_path
    assert Path(result.primary_ckpt_path).is_file()
    assert result.manifest_path is not None
    assert Path(result.manifest_path).is_file()

    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    payload = _prediction_payload(model_key)
    first = predictor.predict(payload)
    second = predictor.predict(payload)

    assert predictor.model_key == model_key
    assert predictor.default_horizon == 1
    assert "point" in first
    points = np.asarray(first["point"])
    assert points.shape == (2,)
    assert np.isfinite(points).all()
    np.testing.assert_array_equal(points, np.asarray(second["point"]))


REMAINING_ARTIFACT_SMOKE_CASES = [
    pytest.param(
        "patchtst_quantile",
        _tiny_patchtst_architecture(),
        2,
        4,
        "quantile",
        id="patchtst-quantile",
    ),
    pytest.param(
        "patchmixer_quantile",
        _tiny_patchmixer_architecture(),
        8,
        10,
        "quantile",
        id="patchmixer-quantile",
    ),
    pytest.param(
        "titan_lmm",
        _tiny_titan_architecture(),
        2,
        4,
        "point",
        id="titan-lmm",
    ),
    pytest.param(
        "titan_seq2seq",
        _tiny_titan_architecture(),
        2,
        4,
        "point",
        id="titan-seq2seq",
    ),
]


@pytest.mark.parametrize(
    "model_key,architecture,lookback,n_rows,output_mode",
    REMAINING_ARTIFACT_SMOKE_CASES,
)
def test_public_remaining_artifact_train_checkpoint_load_predict_smoke(
    tmp_path: Path,
    model_key: str,
    architecture: ArchitectureConfig,
    lookback: int,
    n_rows: int,
    output_mode: str,
):
    torch.manual_seed(7)
    np.random.seed(7)
    artifact_dir = tmp_path / model_key

    result = train(
        TrainRequest(
            data=_data_request(None, lookback=lookback, n_rows=n_rows),
            models=[model_key],
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                use_intermittent=False,
                val_use_weights=False,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(save_dir=str(artifact_dir), auto_save_dir=False),
            architecture=architecture,
        )
    )

    assert result.requested_models == (model_key,)
    assert result.primary_result_name == model_key
    assert result.primary_ckpt_path == result.ckpt_paths[model_key]
    assert result.primary_ckpt_path is not None
    assert Path(result.primary_ckpt_path).is_file()

    checkpoint = torch.load(result.primary_ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["output_spec"]["mode"] == output_mode

    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    restored_state = predictor.model.state_dict()
    assert restored_state.keys() == checkpoint["state_dict"].keys()
    for key, saved_value in checkpoint["state_dict"].items():
        torch.testing.assert_close(restored_state[key].cpu(), saved_value.cpu())

    payload = _prediction_payload(model_key, lookback=lookback)
    first = predictor.predict(payload)
    second = predictor.predict(payload)

    assert predictor.model_key == model_key
    if output_mode == "quantile":
        assert bool(getattr(predictor.model, "is_quantile", False)) is True
        assert checkpoint["cfg_state"]["quantiles"] == [0.1, 0.5, 0.9]
        assert set(first) == {"q10", "q50", "q90", "point"}
        for name in ("q10", "q50", "q90", "point"):
            values = np.asarray(first[name])
            assert values.shape == (2,)
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values, np.asarray(second[name]))
        np.testing.assert_array_equal(np.asarray(first["point"]), np.asarray(first["q50"]))
        assert np.all(np.asarray(first["q10"]) <= np.asarray(first["q50"]))
        assert np.all(np.asarray(first["q50"]) <= np.asarray(first["q90"]))
    else:
        assert set(first) == {"point"}
        points = np.asarray(first["point"])
        assert points.shape == (2,)
        assert np.isfinite(points).all()
        np.testing.assert_array_equal(points, np.asarray(second["point"]))


def test_public_patchtst_full_ssl_pretrain_finetune_checkpoint_smoke(tmp_path: Path):
    torch.manual_seed(7)
    np.random.seed(7)
    model_key = "patchtst_base"
    artifact_dir = tmp_path / "patchtst-full-ssl"

    result = train(
        TrainRequest(
            data=_data_request(None),
            models=[model_key],
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
            artifacts=ArtifactConfig(save_dir=str(artifact_dir), auto_save_dir=False),
            architecture=_tiny_patchtst_architecture(),
        )
    )

    assert result.primary_ckpt_path is not None
    assert Path(result.primary_ckpt_path).is_file()
    assert result.pretrain_ckpt_paths.keys() == {model_key}
    pretrain_path = Path(result.pretrain_ckpt_paths[model_key])
    assert pretrain_path.is_file()
    assert result.results[model_key]["pretrain_ckpt_path"] == str(pretrain_path)

    pretrain_checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    assert {"state_dict", "best_val"} <= set(pretrain_checkpoint)
    assert pretrain_checkpoint["state_dict"]
    assert "backbone.patch_embed.weight" in pretrain_checkpoint["state_dict"]
    assert np.isfinite(float(pretrain_checkpoint["best_val"]))

    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    payload = _prediction_payload(model_key)
    first = predictor.predict(payload)
    second = predictor.predict(payload)
    points = np.asarray(first["point"])

    assert predictor.model_key == model_key
    assert points.shape == (2,)
    assert np.isfinite(points).all()
    np.testing.assert_array_equal(points, np.asarray(second["point"]))


FUTURE_EXOGENOUS_SENSITIVITY_CASES = [
    pytest.param("patchtst_base", _tiny_patchtst_architecture(), id="patchtst"),
    pytest.param("patchmixer_base", _tiny_patchmixer_architecture(), id="patchmixer"),
    pytest.param("titan_base", _tiny_titan_architecture(), id="titan"),
    pytest.param("exotst_base", _tiny_exotst_architecture(), id="exotst"),
]


@pytest.mark.parametrize("model_key,architecture", FUTURE_EXOGENOUS_SENSITIVITY_CASES)
def test_public_future_exogenous_contract_and_sensitivity(
    tmp_path: Path,
    model_key: str,
    architecture: ArchitectureConfig,
):
    torch.manual_seed(7)
    np.random.seed(7)
    exogenous = ExogenousConfig(
        use_exogenous_mode=True,
        use_past_exogenous=True,
        use_future_exogenous=True,
        past_exo_cont_cols=["exo_known"],
        future_exo_cont_cols=["exo_known"],
    )

    result = train(
        TrainRequest(
            data=_data_request(exogenous),
            models=[model_key],
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                use_intermittent=False,
                val_use_weights=False,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(
                save_dir=str(tmp_path / f"{model_key}-future-exogenous"),
                auto_save_dir=False,
            ),
            architecture=architecture,
        )
    )

    assert result.primary_ckpt_path is not None
    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    prediction_payload = _prediction_payload(model_key)
    x = prediction_payload["x"] if isinstance(prediction_payload, dict) else prediction_payload
    assert torch.is_tensor(x)
    base_payload = {
        "x": x,
        "past_exo_cont": torch.tensor([[[0.0], [0.5]], [[0.5], [1.0]]]),
    }

    with pytest.raises(RuntimeError, match="expects future exogenous inputs.*not provided"):
        predictor.predict(base_payload)

    wrong_dim_payload = dict(base_payload)
    wrong_dim_payload["future_exo_batch"] = torch.zeros(2, 1, 2)
    with pytest.raises(RuntimeError, match="last dimension mismatch: got 2, expected 1"):
        predictor.predict(wrong_dim_payload)

    low_payload = dict(base_payload)
    low_payload["future_exo_batch"] = torch.zeros(2, 1, 1)
    high_payload = dict(base_payload)
    high_payload["future_exo_batch"] = torch.ones(2, 1, 1)

    low = np.asarray(predictor.predict(low_payload)["point"])
    low_repeat = np.asarray(predictor.predict(low_payload)["point"])
    high = np.asarray(predictor.predict(high_payload)["point"])

    assert _infer_d_future_expected(predictor.model) == 1
    assert low.shape == high.shape == (2,)
    assert np.isfinite(low).all()
    assert np.isfinite(high).all()
    np.testing.assert_array_equal(low, low_repeat)
    assert float(np.max(np.abs(low - high))) > 1e-6
