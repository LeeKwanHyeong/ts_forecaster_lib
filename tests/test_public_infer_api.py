import torch

from modeling_module import load_predictor, predict
from modeling_module.models.PatchTST.common.configs import AttentionConfig, PatchTSTConfig
from modeling_module.models.TimeXer.configs import TimeXerConfig
from modeling_module.models.model_builder import build_patchTST, build_timexer
from modeling_module.utils.checkpoint import save_model


def _make_tiny_patchtst_cfg() -> PatchTSTConfig:
    return PatchTSTConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        stride=2,
        d_model=16,
        d_ff=32,
        n_layers=1,
        future_exo_dim=0,
        past_exo_cont_dim=0,
        past_exo_cat_dim=0,
        use_exogenous_mode=False,
        use_revin=False,
        attn=AttentionConfig(
            n_heads=4,
            d_model=16,
            attn_dropout=0.0,
            proj_dropout=0.0,
        ),
    )


def test_load_predictor_and_predict_smoke(tmp_path):
    cfg = _make_tiny_patchtst_cfg()
    model = build_patchTST(cfg)
    ckpt_path = tmp_path / "tiny_patchtst.pt"

    save_model(
        model,
        cfg,
        str(ckpt_path),
        extra_meta={"model_key": "patchtst_base", "family_key": "patchtst"},
    )

    predictor = load_predictor(str(ckpt_path), device="cpu")
    x = torch.randn(2, cfg.lookback, 1)

    direct = predictor(x, horizon=cfg.horizon)
    via_helper = predict(str(ckpt_path), x, device="cpu", horizon=cfg.horizon)

    assert predictor.model_key == "patchtst_base"
    assert isinstance(direct, dict)
    assert isinstance(via_helper, dict)
    assert "point" in direct
    assert "point" in via_helper
    assert len(direct["point"]) == 2 * cfg.horizon
    assert len(via_helper["point"]) == 2 * cfg.horizon


def test_load_predictor_and_predict_timexer_smoke(tmp_path):
    cfg = TimeXerConfig(
        lookback=8,
        horizon=2,
        patch_len=4,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        past_exo_cont_dim=2,
        use_exogenous_mode=True,
        use_norm=False,
    )
    model = build_timexer(cfg)
    ckpt_path = tmp_path / "tiny_timexer.pt"

    save_model(
        model,
        cfg,
        str(ckpt_path),
        extra_meta={"model_key": "timexer_base", "family_key": "timexer"},
    )

    predictor = load_predictor(str(ckpt_path), device="cpu")
    x = torch.randn(2, cfg.lookback, 1)
    past_exo = torch.randn(2, cfg.lookback, cfg.past_exo_cont_dim)

    payload = {"x": x, "past_exo_cont": past_exo}
    direct = predictor(payload, horizon=cfg.horizon)
    via_helper = predict(str(ckpt_path), payload, device="cpu", horizon=cfg.horizon)

    assert predictor.model_key == "timexer_base"
    assert isinstance(direct, dict)
    assert isinstance(via_helper, dict)
    assert "point" in direct
    assert "point" in via_helper
    assert len(direct["point"]) == 2 * cfg.horizon
    assert len(via_helper["point"]) == 2 * cfg.horizon
