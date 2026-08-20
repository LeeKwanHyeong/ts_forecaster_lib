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
    NHITSArchitectureConfig,
    PatchMixerArchitectureConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SELLMArchitectureConfig,
    SSLConfig,
    TimeMixerArchitectureConfig,
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
    exogenous_model_keys = {
        "exotst_base",
        "timexer_base",
        "patchtst_exogenous",
        "patchtst_quantile_exogenous",
        "patchmixer_exo",
    }
    if model_key not in exogenous_model_keys:
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
    if model_key != "timexer_base":
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
            mixer_kernel_size=3,
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


def _tiny_nhits_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        nhits=NHITSArchitectureConfig(
            stack_types=("identity",),
            n_blocks=(1,),
            n_layers=(2,),
            n_theta_hidden=((8, 8),),
            n_pool_kernel_size=(1,),
            n_freq_downsample=(1,),
            pooling_mode="max",
            interpolation_mode="linear",
            activation="Softplus",
            initialization="glorot_uniform",
            batch_normalization=False,
            dropout_prob_theta=0.0,
            shared_weights=False,
        )
    )


def _tiny_timemixer_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        timemixer=TimeMixerArchitectureConfig(
            d_model=4,
            d_ff=8,
            e_layers=1,
            moving_avg=3,
            down_sampling_layers=1,
            down_sampling_window=2,
            dropout=0.0,
            use_norm=True,
        )
    )


def _tiny_sellm_architecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        sellm=SELLMArchitectureConfig(
            token_len=2,
            d_model=8,
            n_heads=2,
            dropout=0.0,
            mlp_hidden_dim=8,
            semantic_vocab_size=16,
            semantic_top_k=4,
            tscc_latent_dim=4,
            tscc_hidden_dim=8,
            use_pretrained_llm=False,
            use_time_adapter=False,
            fallback_layers=1,
            d_ff=16,
            head_hidden_dim=8,
            use_norm=False,
            final_nonneg=False,
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
        "patchmixer",
        _tiny_patchmixer_architecture(),
        None,
        id="patchmixer",
    ),
    pytest.param(
        "patchtst_exogenous",
        _tiny_patchtst_architecture(),
        ExogenousConfig(
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=True,
            past_exo_cont_cols=["exo_known"],
            future_exo_cont_cols=["exo_known"],
        ),
        id="patchtst-exogenous",
    ),
    pytest.param(
        "patchmixer_exo",
        _tiny_patchmixer_architecture(),
        ExogenousConfig(
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=True,
            past_exo_cont_cols=["exo_known"],
            future_exo_cont_cols=["exo_known"],
        ),
        id="patchmixer-exogenous",
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
        "nhits_base",
        _tiny_nhits_architecture(),
        None,
        id="nhits",
    ),
    pytest.param(
        "timemixer",
        _tiny_timemixer_architecture(),
        None,
        id="timemixer",
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
    pytest.param(
        "sellm_base",
        _tiny_sellm_architecture(),
        None,
        id="sellm",
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

    if model_key in {"nhits_base", "timemixer", "timexer_base"}:
        checkpoint = torch.load(
            result.primary_ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_identity = {
            "nhits_base": ("NHITSConfig", "NHITSModel", "nhits"),
            "timemixer": ("TimeMixerConfig", "TimeMixerModel", "timemixer"),
            "timexer_base": ("TimeXerConfig", "TimeXerModel", "timexer"),
        }
        cfg_cls, model_class, family_key = expected_identity[model_key]
        assert checkpoint["cfg_cls"] == cfg_cls
        assert checkpoint["model_class"] == model_class
        assert checkpoint["output_spec"] == {
            "mode": "point",
            "distribution": None,
            "out_mult": 1,
            "param_names": None,
        }
        assert checkpoint["meta"]["model_key"] == model_key
        assert checkpoint["meta"]["family_key"] == family_key
        if model_key == "timemixer":
            assert checkpoint["cfg_state"]["down_sampling_layers"] == 1
            assert checkpoint["meta"]["architecture_variant"] == "endogenous"

    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    if exogenous is not None:
        assert predictor.exogenous_schema is not None
        assert predictor.exogenous_schema.past_cont_names == ("exo_known",)
        expected_future = () if model_key == "timexer_base" else ("exo_known",)
        assert predictor.exogenous_schema.future_cont_names == expected_future
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

    if model_key in {"nhits_base", "timemixer", "timexer_base"}:
        restored_state = predictor.model.state_dict()
        assert restored_state.keys() == checkpoint["state_dict"].keys()
        for key, saved_value in checkpoint["state_dict"].items():
            torch.testing.assert_close(restored_state[key].cpu(), saved_value.cpu())


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


QUANTILE_FUTURE_EXOGENOUS_SMOKE_CASES = [
    pytest.param(
        "patchtst_quantile",
        _tiny_patchtst_architecture(),
        2,
        4,
        id="patchtst-quantile-future-exogenous",
    ),
    pytest.param(
        "patchtst_quantile_exogenous",
        _tiny_patchtst_architecture(),
        2,
        4,
        id="patchtst-quantile-explicit-exogenous",
    ),
]


@pytest.mark.parametrize(
    "model_key,architecture,lookback,n_rows",
    QUANTILE_FUTURE_EXOGENOUS_SMOKE_CASES,
)
def test_public_quantile_future_exogenous_train_checkpoint_load_predict_smoke(
    tmp_path: Path,
    model_key: str,
    architecture: ArchitectureConfig,
    lookback: int,
    n_rows: int,
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
            data=_data_request(exogenous, lookback=lookback, n_rows=n_rows),
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
                save_dir=str(tmp_path / model_key),
                auto_save_dir=False,
            ),
            architecture=architecture,
        )
    )

    assert result.requested_models == (model_key,)
    assert result.primary_ckpt_path == result.ckpt_paths[model_key]
    assert result.primary_ckpt_path is not None
    assert Path(result.primary_ckpt_path).is_file()
    checkpoint = torch.load(result.primary_ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["output_spec"]["mode"] == "quantile"
    assert checkpoint["cfg_state"]["quantiles"] == [0.1, 0.5, 0.9]
    assert checkpoint["cfg_state"]["past_exo_cont_dim"] == 1
    assert checkpoint["cfg_state"]["future_exo_dim"] == 1

    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)
    assert predictor.model_key == model_key
    restored_state = predictor.model.state_dict()
    assert restored_state.keys() == checkpoint["state_dict"].keys()
    for key, saved_value in checkpoint["state_dict"].items():
        torch.testing.assert_close(restored_state[key].cpu(), saved_value.cpu())

    prediction_payload = _prediction_payload(model_key, lookback=lookback)
    x = prediction_payload["x"] if isinstance(prediction_payload, dict) else prediction_payload
    assert torch.is_tensor(x)
    payload = {
        "x": x,
        "past_exo_cont": torch.stack(
            (
                0.5 * torch.arange(lookback, dtype=torch.float32),
                0.5 * torch.arange(1, lookback + 1, dtype=torch.float32),
            ),
            dim=0,
        ).unsqueeze(-1),
        "future_exo_batch": torch.tensor([[[1.0]], [[0.5]]]),
    }
    first = predictor.predict(payload)
    second = predictor.predict(payload)

    assert bool(getattr(predictor.model, "is_quantile", False)) is True
    assert set(first) == {"q10", "q50", "q90", "point"}
    for name in ("q10", "q50", "q90", "point"):
        values = np.asarray(first[name])
        assert values.shape == (2,)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, np.asarray(second[name]))
    np.testing.assert_array_equal(np.asarray(first["point"]), np.asarray(first["q50"]))
    assert np.all(np.asarray(first["q10"]) <= np.asarray(first["q50"]))
    assert np.all(np.asarray(first["q50"]) <= np.asarray(first["q90"]))


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
    pytest.param(
        "patchtst_base",
        _tiny_patchtst_architecture(),
        2,
        4,
        "point",
        id="patchtst-point-legacy-routing",
    ),
    pytest.param(
        "patchtst_exogenous",
        _tiny_patchtst_architecture(),
        2,
        4,
        "point",
        id="patchtst-exogenous-point",
    ),
    pytest.param(
        "patchmixer_exo",
        _tiny_patchmixer_architecture(),
        2,
        4,
        "point",
        id="patchmixer-point",
    ),
    pytest.param(
        "exotst_base",
        _tiny_exotst_architecture(),
        2,
        4,
        "point",
        id="exotst-point",
    ),
    pytest.param(
        "patchtst_quantile",
        _tiny_patchtst_architecture(),
        2,
        4,
        "q50",
        id="patchtst-quantile-legacy-routing",
    ),
    pytest.param(
        "patchtst_quantile_exogenous",
        _tiny_patchtst_architecture(),
        2,
        4,
        "q50",
        id="patchtst-quantile-exogenous",
    ),
]


@pytest.mark.parametrize(
    "model_key,architecture,lookback,n_rows,prediction_key",
    FUTURE_EXOGENOUS_SENSITIVITY_CASES,
)
def test_public_future_exogenous_contract_and_sensitivity(
    tmp_path: Path,
    model_key: str,
    architecture: ArchitectureConfig,
    lookback: int,
    n_rows: int,
    prediction_key: str,
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
            data=_data_request(exogenous, lookback=lookback, n_rows=n_rows),
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
    prediction_payload = _prediction_payload(model_key, lookback=lookback)
    x = prediction_payload["x"] if isinstance(prediction_payload, dict) else prediction_payload
    assert torch.is_tensor(x)
    base_payload = {
        "x": x,
        "past_exo_cont": torch.stack(
            (
                0.5 * torch.arange(lookback, dtype=torch.float32),
                0.5 * torch.arange(1, lookback + 1, dtype=torch.float32),
            ),
            dim=0,
        ).unsqueeze(-1),
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

    low_result = predictor.predict(low_payload)
    low_repeat_result = predictor.predict(low_payload)
    high_result = predictor.predict(high_payload)

    if prediction_key == "q50":
        assert set(low_result) == {"q10", "q50", "q90", "point"}
        np.testing.assert_array_equal(
            np.asarray(low_result["point"]),
            np.asarray(low_result["q50"]),
        )
    else:
        assert set(low_result) == {"point"}

    low = np.asarray(low_result[prediction_key])
    low_repeat = np.asarray(low_repeat_result[prediction_key])
    high = np.asarray(high_result[prediction_key])

    assert _infer_d_future_expected(predictor.model) == 1
    assert low.shape == high.shape == (2,)
    assert np.isfinite(low).all()
    assert np.isfinite(high).all()
    np.testing.assert_array_equal(low, low_repeat)
    assert float(np.max(np.abs(low - high))) > 1e-6
