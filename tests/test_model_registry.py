import pytest

from modeling_module.api.train import _make_result
from modeling_module.models.registry import (
    PATCHMIXER_CAPABILITY_DEFAULTS,
    PATCHTST_CAPABILITY_DEFAULTS,
    expand_training_targets,
    get_patchmixer_default_model_key,
    get_patchtst_default_model_key,
    get_model_spec,
    get_training_deprecation_messages,
    infer_artifact_model_key_from_checkpoint,
)


def test_expand_training_targets_supports_family_and_artifact_keys():
    assert expand_training_targets(None) == ["patchtst_base", "patchtst_quantile"]
    assert expand_training_targets([]) == ["patchtst_base", "patchtst_quantile"]
    assert expand_training_targets(["patchtst", "Titan_LMM", "patchmixer"]) == [
        "patchtst_base",
        "patchtst_quantile",
        "titan_lmm",
        "patchmixer",
    ]
    assert expand_training_targets(["titan"]) == [
        "titan_base",
        "titan_lmm",
        "titan_seq2seq",
    ]
    assert expand_training_targets(["timexer"]) == ["timexer_base"]
    assert expand_training_targets(["sellm"]) == ["sellm_base"]
    assert expand_training_targets(["nhits"]) == ["nhits_base"]


def test_expand_training_targets_preserves_single_artifact_requests():
    assert expand_training_targets(["titan_lmm"]) == ["titan_lmm"]
    assert expand_training_targets(["titan_seq2seq"]) == ["titan_seq2seq"]
    assert expand_training_targets(["titan_base"]) == ["titan_base"]
    assert expand_training_targets(["patchmixer"]) == ["patchmixer"]
    assert expand_training_targets(["patchmixer_exo"]) == ["patchmixer_exo"]
    assert expand_training_targets(["patchtst_quantile"]) == ["patchtst_quantile"]
    assert expand_training_targets(["timexer_base"]) == ["timexer_base"]
    assert expand_training_targets(["sellm_base"]) == ["sellm_base"]
    assert expand_training_targets(["nhits_base"]) == ["nhits_base"]
    assert expand_training_targets(["patchtst_exogenous"]) == ["patchtst_exogenous"]
    assert expand_training_targets(["patchtst_quantile_exogenous"]) == [
        "patchtst_quantile_exogenous"
    ]
    assert expand_training_targets(["patchmixer_exogenous"]) == ["patchmixer_exo"]
    with pytest.raises(ValueError, match="not trainable"):
        expand_training_targets(["patchmixer_quantile"])
    with pytest.raises(ValueError, match="not trainable"):
        expand_training_targets(["patchmixer_base"])


def test_patchmixer_capability_defaults_expose_only_point_responsibilities():
    assert PATCHMIXER_CAPABILITY_DEFAULTS == {
        "endogenous_point": "patchmixer",
        "exogenous_point": "patchmixer_exo",
    }
    assert get_patchmixer_default_model_key() == "patchmixer"
    assert get_patchmixer_default_model_key("point") == "patchmixer"
    assert get_patchmixer_default_model_key("exogenous-point") == "patchmixer_exo"
    assert expand_training_targets(["patchmixer"]) == ["patchmixer"]

    for unsupported in ("dist", "quantile", "exogenous-quantile", "classification"):
        with pytest.raises(ValueError, match="Unknown PatchMixer capability"):
            get_patchmixer_default_model_key(unsupported)


def test_patchtst_capability_defaults_encode_artifact_responsibilities():
    assert PATCHTST_CAPABILITY_DEFAULTS == {
        "endogenous_point": "patchtst_base",
        "exogenous_point": "patchtst_exogenous",
        "endogenous_distribution": "patchtst_base",
        "exogenous_distribution": "patchtst_exogenous",
        "endogenous_quantile": "patchtst_quantile",
        "exogenous_quantile": "patchtst_quantile_exogenous",
    }
    assert get_patchtst_default_model_key() == "patchtst_base"
    assert get_patchtst_default_model_key("point") == "patchtst_base"
    assert get_patchtst_default_model_key("exogenous-point") == "patchtst_exogenous"
    assert get_patchtst_default_model_key("dist") == "patchtst_base"
    assert (
        get_patchtst_default_model_key("exogenous-distribution")
        == "patchtst_exogenous"
    )
    assert get_patchtst_default_model_key("quantile") == "patchtst_quantile"
    assert (
        get_patchtst_default_model_key("exogenous-quantile")
        == "patchtst_quantile_exogenous"
    )
    assert expand_training_targets(["patchtst"]) == [
        "patchtst_base",
        "patchtst_quantile",
    ]

    with pytest.raises(ValueError, match="Unknown PatchTST capability"):
        get_patchtst_default_model_key("classification")


def test_titan_registry_entries_are_deprecated_but_remain_trainable_for_compatibility():
    titan_keys = expand_training_targets(["titan"])

    assert titan_keys == ["titan_base", "titan_lmm", "titan_seq2seq"]
    assert all(get_model_spec(key).trainable for key in titan_keys)
    assert all(get_model_spec(key).deprecated for key in titan_keys)
    messages = get_training_deprecation_messages(titan_keys)
    assert len(messages) == 1
    assert "checkpoints remain loadable" in messages[0]


def test_infer_artifact_model_key_from_checkpoint_prefers_meta():
    assert infer_artifact_model_key_from_checkpoint(
        {"meta": {"model_key": "patchmixer_quantile"}, "model_class": "PatchMixerModel"}
    ) == "patchmixer_quantile"
    assert infer_artifact_model_key_from_checkpoint({"model_class": "TitanLMMDist"}) == "titan_lmm"
    assert infer_artifact_model_key_from_checkpoint({"model_class": "SELLMModel"}) == "sellm_base"
    assert infer_artifact_model_key_from_checkpoint({"model_class": "NHITSModel"}) == "nhits_base"
    assert infer_artifact_model_key_from_checkpoint(
        {"model_class": "PatchTSTExogenousModel"}
    ) == "patchtst_exogenous"
    assert infer_artifact_model_key_from_checkpoint(
        {"model_class": "PatchMixerQuantileExogenousModel"}
    ) == "patchmixer_quantile_exogenous"


@pytest.mark.parametrize(
    ("model_key", "fusion_strategy"),
    (
        ("patchtst_exogenous", "patch_concat+future_cross_attention"),
        ("patchtst_quantile_exogenous", "patch_concat+future_cross_attention"),
        ("patchmixer_exo", "gated_residual+future_shift"),
    ),
)
def test_explicit_exogenous_registry_contract(model_key, fusion_strategy):
    spec = get_model_spec(model_key)

    assert spec.exogenous_policy == "required"
    assert spec.exogenous_inputs == ("past_cont", "past_cat", "future_cont")
    assert spec.fusion_strategy == fusion_strategy
    assert spec.load_only is False


@pytest.mark.parametrize(
    "model_key",
    ("patchmixer_base", "patchmixer_quantile", "patchmixer_quantile_exogenous"),
)
def test_retired_patchmixer_registry_entries_are_load_only(model_key):
    spec = get_model_spec(model_key)

    assert spec.family == "patchmixer"
    assert spec.trainable is False
    assert spec.load_only is True
    assert spec.deprecated is True


def test_nhits_registry_contract_is_endogenous_point_only():
    spec = get_model_spec("nhits_base")

    assert spec.family == "nhits"
    assert spec.exogenous_policy == "none"
    assert spec.exogenous_inputs == ()
    assert spec.fusion_strategy is None
    assert spec.trainable is True


def test_train_result_uses_canonical_model_keys(tmp_path):
    result = _make_result(
        request_payload={"models_to_run": ["patchtst_base"]},
        results={
            "PatchTST": {
                "model_key": "patchtst_base",
                "family_key": "patchtst",
                "ckpt_path": str(tmp_path / "patchtst.pt"),
            }
        },
        requested_models=["patchtst_base"],
        save_dir=None,
        datamodule=None,
    )

    assert list(result.results.keys()) == ["patchtst_base"]
    assert result.ckpt_paths["patchtst_base"].endswith("patchtst.pt")
