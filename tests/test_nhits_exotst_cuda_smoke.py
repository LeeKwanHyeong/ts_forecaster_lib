from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from modeling_module import (
    ArchitectureConfig,
    ArtifactConfig,
    DataRequest,
    DataWindowConfig,
    DistributionLoss,
    ExogenousConfig,
    ExoTSTArchitectureConfig,
    LoaderConfig,
    NHITSArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TrainerConfig,
    TrainRequest,
    load_predictor,
    train,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA smoke requires an NVIDIA runtime.",
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["series-1"] * 4,
            "date": [202401, 202402, 202403, 202404],
            "y": [1.0, 1.5, 2.0, 2.5],
            "exo_known": [0.0, 0.5, 1.0, 0.5],
        }
    )


def _data_request(exogenous: ExogenousConfig | None) -> DataRequest:
    return DataRequest(
        df=_frame(),
        backend="exo",
        window=DataWindowConfig(lookback=2, horizon=1, freq="monthly"),
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


def _nhits_architecture() -> ArchitectureConfig:
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


def _exotst_architecture() -> ArchitectureConfig:
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


def _exogenous_config() -> ExogenousConfig:
    return ExogenousConfig(
        use_exogenous_mode=True,
        use_past_exogenous=True,
        use_future_exogenous=True,
        past_exo_cont_cols=["exo_known"],
        future_exo_cont_cols=["exo_known"],
    )


def _prediction_payload(model_key: str):
    x = torch.tensor([[[1.0], [1.5]], [[1.5], [2.0]]])
    if model_key == "nhits_base":
        return x
    return {
        "x": x,
        "past_exo_cont": torch.tensor([[[0.0], [0.5]], [[0.5], [1.0]]]),
        "future_exo_batch": torch.tensor([[[1.0]], [[0.5]]]),
    }


@pytest.mark.parametrize(
    ("model_key", "distribution"),
    [
        pytest.param("nhits_base", None, id="nhits-point"),
        pytest.param("exotst_base", None, id="exotst-point"),
        pytest.param("exotst_base", "Normal", id="exotst-normal"),
        pytest.param("exotst_base", "StudentT", id="exotst-studentt"),
    ],
)
def test_public_cuda_train_checkpoint_load_predict_smoke(
    tmp_path: Path,
    model_key: str,
    distribution: str | None,
):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    np.random.seed(7)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    is_exotst = model_key == "exotst_base"
    loss = (
        DistributionLoss(distribution=distribution, num_samples=16)
        if distribution is not None
        else None
    )
    result = train(
        TrainRequest(
            data=_data_request(_exogenous_config() if is_exotst else None),
            models=[model_key],
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                loss=loss,
                use_intermittent=False,
                val_use_weights=False,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cuda"),
            artifacts=ArtifactConfig(
                save_dir=str(tmp_path / model_key / str(distribution or "point").lower()),
                auto_save_dir=False,
            ),
            architecture=_exotst_architecture() if is_exotst else _nhits_architecture(),
        )
    )

    assert result.primary_ckpt_path is not None
    assert Path(result.primary_ckpt_path).is_file()
    checkpoint = torch.load(
        result.primary_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    expected_mode = "distribution" if distribution is not None else "point"
    assert checkpoint["output_spec"]["mode"] == expected_mode
    assert checkpoint["output_spec"]["distribution"] == distribution

    predictor = load_predictor(result.primary_ckpt_path, device="cuda", strict=True)
    assert next(predictor.model.parameters()).is_cuda
    first = predictor.predict(_prediction_payload(model_key))
    second = predictor.predict(_prediction_payload(model_key))
    torch.cuda.synchronize()

    assert set(first) == {"point"}
    points = np.asarray(first["point"])
    assert points.shape == (2,)
    assert np.isfinite(points).all()
    np.testing.assert_array_equal(points, np.asarray(second["point"]))
    assert torch.cuda.max_memory_allocated() > 0

