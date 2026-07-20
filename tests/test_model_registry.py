from modeling_module.api.train import _make_result
from modeling_module.models.registry import (
    expand_training_targets,
    get_model_spec,
    get_training_deprecation_messages,
    infer_artifact_model_key_from_checkpoint,
)


def test_expand_training_targets_supports_family_and_artifact_keys():
    assert expand_training_targets(None) == ["patchtst_base", "patchtst_quantile"]
    assert expand_training_targets([]) == ["patchtst_base", "patchtst_quantile"]
    assert expand_training_targets(["patchtst", "Titan_LMM", "patchmixer_quantile"]) == [
        "patchtst_base",
        "patchtst_quantile",
        "titan_lmm",
        "patchmixer_quantile",
    ]
    assert expand_training_targets(["titan"]) == [
        "titan_base",
        "titan_lmm",
        "titan_seq2seq",
    ]
    assert expand_training_targets(["timexer"]) == ["timexer_base"]


def test_expand_training_targets_preserves_single_artifact_requests():
    assert expand_training_targets(["titan_lmm"]) == ["titan_lmm"]
    assert expand_training_targets(["titan_seq2seq"]) == ["titan_seq2seq"]
    assert expand_training_targets(["titan_base"]) == ["titan_base"]
    assert expand_training_targets(["patchmixer_quantile"]) == ["patchmixer_quantile"]
    assert expand_training_targets(["patchtst_quantile"]) == ["patchtst_quantile"]
    assert expand_training_targets(["timexer_base"]) == ["timexer_base"]


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
