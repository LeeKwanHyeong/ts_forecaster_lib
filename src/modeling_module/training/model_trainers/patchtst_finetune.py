# modeling_module/training/model_trainers/patchtst_finetune.py
from __future__ import annotations

import os
from typing import Optional, Callable, Dict, Any, Tuple

import torch

from modeling_module.training.config import TrainingConfig, StageConfig
from modeling_module.training.model_trainers.patchtst_train import train_patchtst


def _sinusoidal_pos_emb(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return sinusoidal positional embedding of shape [1, n, d]."""
    if d <= 0:
        raise ValueError(f"d_model must be > 0. got d={d}")

    pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n,1]
    i = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)    # [1,d]
    angle_rates = 1.0 / torch.pow(10000.0, (2 * (i // 2)) / d)
    angles = pos * angle_rates

    pe = torch.zeros((n, d), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)  # [1,n,d]


# ---------------------------------------------------------------------
# State-dict utilities
# ---------------------------------------------------------------------
def _drop_revin_stats(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop RevIN running stats buffers that are not safe/necessary to load."""
    drop_keys = [
        "revin_layer.mean",
        "revin_layer.std",
        # 구현에 따라 아래처럼 prefix가 다를 수 있어 함께 제거
        "revin.mean",
        "revin.std",
        "revin_layer.running_mean",
        "revin_layer.running_std",
    ]
    for k in drop_keys:
        state.pop(k, None)
    return state


def _strip_common_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrappers like 'module.' (DDP) or 'model.' (wrapper modules)."""
    prefixes = ("module.", "model.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        kk = k
        for px in prefixes:
            if kk.startswith(px):
                kk = kk[len(px):]
        out[kk] = v
    return out


def _load_pretrain_blob(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Return (state_dict, meta). meta may include cfg/best_val etc."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        # common container: {"state_dict":..., "best_val":..., "cfg":...}
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
            return ckpt["state_dict"], meta
        # alternative container
        if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
            meta = {k: v for k, v in ckpt.items() if k != "encoder"}
            return ckpt["encoder"], meta
        # raw state_dict
        return ckpt, meta
    raise ValueError(f"Unsupported ckpt format: {type(ckpt)}")


def _infer_supervised_encoder_spec(model) -> Tuple[int, int]:
    """Infer (d_model, n_layers) from the supervised PatchTST model instance."""
    d_model = None
    n_layers = None

    # d_model
    if hasattr(model, "cfg") and hasattr(model.cfg, "d_model"):
        d_model = int(model.cfg.d_model)
    elif hasattr(model, "backbone") and hasattr(model.backbone, "d_model"):
        d_model = int(model.backbone.d_model)
    elif hasattr(model, "backbone") and hasattr(model.backbone, "norm_out"):
        try:
            # LayerNorm.normalized_shape -> (d_model,)
            d_model = int(model.backbone.norm_out.normalized_shape[0])
        except Exception:
            d_model = None

    # n_layers
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_layers"):
        n_layers = int(model.cfg.n_layers)
    else:
        try:
            enc = model.backbone.encoder
            if hasattr(enc, "layers"):
                n_layers = len(enc.layers)
        except Exception:
            n_layers = None

    if d_model is None or n_layers is None:
        raise RuntimeError(
            f"Failed to infer d_model/n_layers from model. d_model={d_model}, n_layers={n_layers}"
        )
    return d_model, n_layers


def _looks_like_torch_transformer_encoder(state: Dict[str, torch.Tensor]) -> bool:
    # torch.nn.TransformerEncoderLayer naming
    for k in state.keys():
        if ".self_attn.in_proj_weight" in k or ".self_attn.in_proj_bias" in k:
            return True
    return False


def _convert_torch_transformer_to_tst(
    state: Dict[str, torch.Tensor],
    *,
    d_model: int,
    n_layers: int,
    dst_prefix: str = "backbone.encoder.layers.",
    src_prefix: str = "backbone.encoder.layers.",
) -> Dict[str, torch.Tensor]:
    """
    Convert a torch.nn.TransformerEncoderLayer-style state_dict into this project's
    TSTEncoderLayer-style naming.

    Source keys (per layer i):
      - {src_prefix}{i}.self_attn.in_proj_weight / in_proj_bias
      - {src_prefix}{i}.self_attn.out_proj.weight / out_proj.bias
      - {src_prefix}{i}.linear1.* , linear2.*
      - {src_prefix}{i}.norm1.* , norm2.*   (LayerNorm)

    Destination keys (per layer i) expected by supervised backbone:
      - {dst_prefix}{i}.mha.W_Q.* / W_K.* / W_V.*
      - {dst_prefix}{i}.mha.to_out.0.*
      - {dst_prefix}{i}.ff.0.*  and {dst_prefix}{i}.ff.3.*
      - (norm_attn / norm_ffn) are frequently BatchNorm-based in this repo; we DO NOT map
        LayerNorm weights into BatchNorm buffers. Those remain randomly initialized.

    Notes:
      - We only emit keys that are present in `state` and can be safely split by d_model.
      - This is a best-effort conversion to maximize weight reuse.
    """
    out = dict(state)  # start from original, then we will add mapped keys and drop source keys

    def _pop(key: str):
        if key in out:
            return out.pop(key)
        return None

    for i in range(n_layers):
        sp = f"{src_prefix}{i}."

        in_w = _pop(sp + "self_attn.in_proj_weight")
        in_b = _pop(sp + "self_attn.in_proj_bias")
        out_w = _pop(sp + "self_attn.out_proj.weight")
        out_b = _pop(sp + "self_attn.out_proj.bias")
        l1_w = _pop(sp + "linear1.weight")
        l1_b = _pop(sp + "linear1.bias")
        l2_w = _pop(sp + "linear2.weight")
        l2_b = _pop(sp + "linear2.bias")

        # LayerNorm params in torch TransformerEncoderLayer
        # We map them into this repo's (BatchNorm-based) norm modules' affine params only.
        n1_w = _pop(sp + "norm1.weight")
        n1_b = _pop(sp + "norm1.bias")
        n2_w = _pop(sp + "norm2.weight")
        n2_b = _pop(sp + "norm2.bias")

        dp = f"{dst_prefix}{i}."

        # ---- QKV ----
        if in_w is not None:
            if in_w.shape[0] != 3 * d_model or in_w.shape[1] != d_model:
                # shape mismatch: skip mapping
                pass
            else:
                out[dp + "mha.W_Q.weight"] = in_w[0:d_model, :].contiguous()
                out[dp + "mha.W_K.weight"] = in_w[d_model:2*d_model, :].contiguous()
                out[dp + "mha.W_V.weight"] = in_w[2*d_model:3*d_model, :].contiguous()
        if in_b is not None:
            if in_b.shape[0] == 3 * d_model:
                out[dp + "mha.W_Q.bias"] = in_b[0:d_model].contiguous()
                out[dp + "mha.W_K.bias"] = in_b[d_model:2*d_model].contiguous()
                out[dp + "mha.W_V.bias"] = in_b[2*d_model:3*d_model].contiguous()

        # ---- out proj ----
        if out_w is not None:
            out[dp + "mha.to_out.0.weight"] = out_w.contiguous()
        if out_b is not None:
            out[dp + "mha.to_out.0.bias"] = out_b.contiguous()

        # ---- FFN ----
        if l1_w is not None:
            out[dp + "ff.0.weight"] = l1_w.contiguous()
        if l1_b is not None:
            out[dp + "ff.0.bias"] = l1_b.contiguous()
        if l2_w is not None:
            out[dp + "ff.3.weight"] = l2_w.contiguous()
        if l2_b is not None:
            out[dp + "ff.3.bias"] = l2_b.contiguous()

        # ---- Norm (best-effort) ----
        # Target expects BatchNorm buffers too (running_mean/var/num_batches_tracked).
        # We cannot derive those from LayerNorm, but mapping affine params is still useful.
        if n1_w is not None and n1_w.numel() == d_model:
            out[dp + "norm_attn.1.weight"] = n1_w.contiguous()
        if n1_b is not None and n1_b.numel() == d_model:
            out[dp + "norm_attn.1.bias"] = n1_b.contiguous()
        if n2_w is not None and n2_w.numel() == d_model:
            out[dp + "norm_ffn.1.weight"] = n2_w.contiguous()
        if n2_b is not None and n2_b.numel() == d_model:
            out[dp + "norm_ffn.1.bias"] = n2_b.contiguous()

    # SSL-only tokens / heads are not needed in supervised finetune
    out.pop("backbone.mask_token", None)

    return out


def _map_patch_embed_to_input_proj(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map SSL patch embedding params to supervised input projection params.

    SSL key: backbone.patch_embed.{weight,bias}
    Sup key: backbone.input_proj.{weight,bias}

    This mapping is valid when both are Linear(in=n_vars*patch_len, out=d_model).
    """
    out = dict(state)
    w = out.get("backbone.patch_embed.weight")
    b = out.get("backbone.patch_embed.bias")
    if w is not None:
        out["backbone.input_proj.weight"] = w
    if b is not None:
        out["backbone.input_proj.bias"] = b
    return out


def _maybe_add_sinusoidal_pos_enc(
    loaded: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """If model expects backbone.pos_enc but ckpt doesn't provide it, initialize with sinusoidal.

    This makes finetune start from the same positional prior as SSL backbone, which used
    sinusoidal embeddings (not learnable). If the supervised model already initializes pos_enc
    well, this is not strictly required, but it improves reproducibility and reduces missing keys.
    """
    if "backbone.pos_enc" in loaded:
        return loaded
    if "backbone.pos_enc" not in model_state:
        return loaded

    pe0 = model_state["backbone.pos_enc"]
    if pe0.dim() == 3 and pe0.shape[0] == 1:
        n, d = int(pe0.shape[1]), int(pe0.shape[2])
        pe = _sinusoidal_pos_emb(n, d, device=pe0.device, dtype=pe0.dtype)
        loaded = dict(loaded)
        loaded["backbone.pos_enc"] = pe
        return loaded
    if pe0.dim() == 2:
        n, d = int(pe0.shape[0]), int(pe0.shape[1])
        pe = _sinusoidal_pos_emb(n, d, device=pe0.device, dtype=pe0.dtype)[0]
        loaded = dict(loaded)
        loaded["backbone.pos_enc"] = pe
        return loaded
    # Unknown pos_enc shape -> skip
    return loaded


def _select_loadable_keys(
    state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    *,
    allow_prefixes: Tuple[str, ...] = ("backbone.", "revin_layer."),
) -> Dict[str, torch.Tensor]:
    """Keep only keys that exist in the target model, optionally constrained by prefixes."""
    model_keys = set(model_state.keys())
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not k.startswith(allow_prefixes):
            continue
        if k in model_keys:
            out[k] = v
    return out


def _report_match_stats(loaded: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]) -> None:
    mkeys = set(model_state.keys())
    lkeys = set(loaded.keys())
    matched = len(lkeys)
    missing = sorted(list(mkeys - lkeys))
    print(f"[Finetune] matched: {matched}")
    print(f"[Finetune] missing: {len(missing)} / total: {len(mkeys)} ratio={len(missing)/max(1,len(mkeys)):.3f}")
    if missing:
        print("[Finetune] missing sample (first 30):", missing[:30])


# ---------------------------------------------------------------------
# Optional freezing helpers
# ---------------------------------------------------------------------
def _freeze_encoder_blocks(model) -> None:
    freeze_prefixes = (
        "backbone.patch_embed.",
        "backbone.input_proj.",
        "backbone.encoder.",
        "backbone.norm_out.",
    )
    for n, p in model.named_parameters():
        if any(n.startswith(px) for px in freeze_prefixes):
            p.requires_grad = False


def _unfreeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------
def train_patchtst_finetune(
    model,
    train_loader,
    val_loader,
    *,
    train_cfg: Optional[TrainingConfig] = None,
    stages: list[StageConfig] | None = None,
    # pretrain loading
    pretrain_ckpt_path: Optional[str] = None,
    load_strict: bool = False,
    # optional freeze
    freeze_encoder_before_ft: bool = False,
    unfreeze_after_stage0: bool = True,
    # exo passthrough (기존 patchtst_train과 동일)
    future_exo_cb: Optional[Callable] = None,
    exo_is_normalized: bool = True,
) -> Dict[str, Any]:
    """
    Supervised fine-tune runner:
      1) (optional) load self-supervised pretrained weights into supervised model
      2) (optional) freeze encoder for stage0
      3) call existing train_patchtst()

    This runner supports a common real-world mismatch:
      - pretrain backbone used torch.nn.TransformerEncoderLayer (keys: self_attn.in_proj_*, linear1/2, norm1/2)
      - finetune backbone expects this repo's TSTEncoderLayer (keys: mha.W_Q/W_K/W_V, ff.0/ff.3, norm_attn/norm_ffn)

    In that case, we perform a best-effort key conversion and only load keys that actually exist in the target model.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # 1) load pretrain ckpt (if provided)
    if pretrain_ckpt_path is not None:
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(pretrain_ckpt_path)

        raw_state, meta = _load_pretrain_blob(pretrain_ckpt_path)
        raw_state = _strip_common_prefixes(raw_state)
        raw_state = _drop_revin_stats(raw_state)

        # Conversion (torch TransformerEncoder -> repo TSTEncoderLayer)
        model_state = model.state_dict()
        d_model, n_layers = _infer_supervised_encoder_spec(model)

        state = raw_state
        if _looks_like_torch_transformer_encoder(raw_state):
            state = _convert_torch_transformer_to_tst(raw_state, d_model=d_model, n_layers=n_layers)

        # SSL backbone uses `patch_embed` while supervised uses `input_proj`
        state = _map_patch_embed_to_input_proj(state)

        # Only keep loadable keys
        filtered = _select_loadable_keys(state, model_state)
        filtered = _maybe_add_sinusoidal_pos_enc(filtered, model_state)

        # Stats before actual load
        print("[Finetune] loaded pretrain ckpt:", pretrain_ckpt_path)
        if meta:
            print("[Finetune] ckpt meta keys:", list(meta.keys()))
        _report_match_stats(filtered, model_state)

        missing, unexpected = model.load_state_dict(filtered, strict=load_strict)

        # NOTE: because we pre-filter to model keys, unexpected should be 0 unless strict/load_state_dict differs.
        print(f"[Finetune] load_strict={load_strict}")
        print(f"[Finetune] load_state_dict -> missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            print("  - missing (first 30):", missing[:30])
        if unexpected:
            print("  - unexpected (first 30):", unexpected[:30])

    # 2) optional freeze encoder
    if freeze_encoder_before_ft:
        _freeze_encoder_blocks(model)
        print("[Finetune] encoder blocks frozen (stage0).")

    # 3) supervised training via existing pipeline
    out = train_patchtst(
        model,
        train_loader,
        val_loader,
        stages=stages,
        train_cfg=train_cfg,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=exo_is_normalized,
    )

    # 4) optionally unfreeze after stage0
    if freeze_encoder_before_ft and unfreeze_after_stage0:
        _unfreeze_all(model)
        print("[Finetune] encoder blocks unfrozen (post stage0).")

    return out
