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
    DistributionLoss,
    ExogenousConfig,
    ExoTSTArchitectureConfig,
    LoaderConfig,
    PatchTSTArchitectureConfig,
    RuntimeConfig,
    SSLConfig,
    TitanArchitectureConfig,
    TrainerConfig,
    TrainRequest,
    load_predictor,
    train,
)


EXPECTED_PARAMS = {
    "Normal": ["-loc", "-scale"],
    "StudentT": ["-df", "-loc", "-scale"],
}
EXPECTED_HEADS = {
    "patchtst_base": "DistHeadWithExo",
    "titan_base": "Linear",
    "exotst_base": "HorizonDistMLPHead",
}


def _tiny_monthly_frame() -> pl.DataFrame:
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
        df=_tiny_monthly_frame(),
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


def _prediction_payload(model_key: str) -> torch.Tensor | dict[str, torch.Tensor]:
    x = torch.tensor([[[1.0], [1.5]], [[1.5], [2.0]]])
    if model_key != "exotst_base":
        return x
    return {
        "x": x,
        "past_exo_cont": torch.tensor([[[0.0], [0.5]], [[0.5], [1.0]]]),
        "future_exo_batch": torch.tensor([[[1.0]], [[0.5]]]),
    }


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


DISTRIBUTION_CHECKPOINT_CASES = [
    pytest.param(
        "patchtst_base",
        ArchitectureConfig(
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
            )
        ),
        None,
        id="patchtst",
    ),
    pytest.param(
        "titan_base",
        ArchitectureConfig(
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
        ),
        None,
        id="titan",
    ),
    pytest.param(
        "exotst_base",
        ArchitectureConfig(
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
        ),
        ExogenousConfig(
            use_exogenous_mode=True,
            use_past_exogenous=True,
            use_future_exogenous=True,
            past_exo_cont_cols=["exo_known"],
            future_exo_cont_cols=["exo_known"],
        ),
        id="exotst",
    ),
]


def _model_out_multiplier(model: torch.nn.Module) -> int:
    value = getattr(model, "out_mult", getattr(model, "out_mul", None))
    assert value is not None
    return int(value)


@pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
@pytest.mark.parametrize("model_key,architecture,exogenous", DISTRIBUTION_CHECKPOINT_CASES)
def test_public_distribution_checkpoint_restores_exact_contract(
    tmp_path: Path,
    model_key: str,
    architecture: ArchitectureConfig,
    exogenous: ExogenousConfig | None,
    distribution: str,
):
    torch.manual_seed(7)
    np.random.seed(7)
    expected_params = EXPECTED_PARAMS[distribution]
    expected_out_multiplier = len(expected_params)
    loss = DistributionLoss(
        distribution=distribution,
        num_samples=32,
        return_params=False,
        horizon_weight=np.asarray([1.0]),
        validate_args=False,
    )

    result = train(
        TrainRequest(
            data=_data_request(exogenous),
            models=[model_key],
            trainer=TrainerConfig(
                epochs=1,
                lr=1e-3,
                loss=loss,
                use_intermittent=False,
                val_use_weights=False,
            ),
            ssl=SSLConfig(mode="sl_only"),
            runtime=RuntimeConfig(device="cpu"),
            artifacts=ArtifactConfig(
                save_dir=str(tmp_path / model_key / distribution.lower()),
                auto_save_dir=False,
            ),
            architecture=architecture,
        )
    )

    assert result.primary_ckpt_path is not None
    checkpoint = torch.load(result.primary_ckpt_path, map_location="cpu", weights_only=False)
    predictor = load_predictor(result.primary_ckpt_path, device="cpu", strict=True)

    restored_loss = predictor.config["loss"]
    assert isinstance(restored_loss, DistributionLoss)
    assert restored_loss.distribution == distribution
    assert restored_loss.param_names == expected_params
    assert restored_loss.outputsize_multiplier == expected_out_multiplier
    assert restored_loss.num_samples == loss.num_samples
    assert restored_loss.return_params == loss.return_params
    assert restored_loss.output_names == loss.output_names
    assert restored_loss.distribution_kwargs == loss.distribution_kwargs
    torch.testing.assert_close(restored_loss.quantiles, loss.quantiles)
    torch.testing.assert_close(restored_loss.horizon_weight, loss.horizon_weight)

    output_spec = checkpoint["output_spec"]
    assert output_spec["mode"] == "distribution"
    assert output_spec["distribution"] == distribution
    assert output_spec["out_mult"] == expected_out_multiplier
    assert output_spec["param_names"] == expected_params

    assert predictor.model_key == model_key
    assert type(predictor.model.head).__name__ == EXPECTED_HEADS[model_key]
    assert _model_out_multiplier(predictor.model) == expected_out_multiplier
    assert list(predictor.model.param_names) == expected_params

    restored_state = predictor.model.state_dict()
    saved_state = checkpoint["state_dict"]
    assert restored_state.keys() == saved_state.keys()
    for key, saved_value in saved_state.items():
        torch.testing.assert_close(restored_state[key].cpu(), saved_value.cpu())

    first = predictor.predict(_prediction_payload(model_key))
    second = predictor.predict(_prediction_payload(model_key))
    assert set(first) == {"point"}
    points = np.asarray(first["point"])
    assert points.shape == (2,)
    assert np.isfinite(points).all()
    np.testing.assert_array_equal(points, np.asarray(second["point"]))
