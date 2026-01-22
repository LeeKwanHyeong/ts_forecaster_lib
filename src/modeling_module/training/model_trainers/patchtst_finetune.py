# modeling_module/training/model_trainers/patchtst_finetune.py
from __future__ import annotations

import os
from typing import Optional, Callable, Dict, Any, Tuple

import torch

from modeling_module.training.config import TrainingConfig, StageConfig
from modeling_module.training.model_trainers.patchtst_train import train_patchtst


def _sinusoidal_pos_emb(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    [1, n, d] 형태의 사인파 위치 임베딩(Sinusoidal Positional Embedding) 생성.

    기능:
    - n(길이), d(차원)에 따른 고정 위치 인코딩 행렬 계산.
    - 짝수 인덱스: sin, 홀수 인덱스: cos 적용.
    """
    if d <= 0:
        raise ValueError(f"d_model must be > 0. got d={d}")

    pos = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n,1]
    i = torch.arange(d, device=device, dtype=dtype).unsqueeze(0)  # [1,d]
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
    """
    로드 시 불필요하거나 안전하지 않은 RevIN 통계 버퍼 제거.

    기능:
    - mean, std, running_mean, running_std 등 데이터 의존적인 통계치 제외.
    - 사전학습 데이터와 파인튜닝 데이터의 분포 차이로 인한 초기화 문제 방지.
    """
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
    """
    'module.'(DDP), 'model.' 등 일반적인 래퍼 접두사 제거.

    기능:
    - 모델 키 매칭 성공률 향상을 위한 키 이름 정규화.
    """
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
    """
    체크포인트 파일 로드 및 state_dict와 메타데이터 분리.

    반환:
        (state_dict, meta_dict)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta: Dict[str, Any] = {}
    if isinstance(ckpt, dict):
        # 일반 컨테이너: {"state_dict":..., "best_val":..., "cfg":...}
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
            return ckpt["state_dict"], meta
        # 대안 컨테이너 (예: 인코더만 저장된 경우)
        if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
            meta = {k: v for k, v in ckpt.items() if k != "encoder"}
            return ckpt["encoder"], meta
        # raw state_dict인 경우
        return ckpt, meta
    raise ValueError(f"Unsupported ckpt format: {type(ckpt)}")


def _infer_supervised_encoder_spec(model) -> Tuple[int, int]:
    """
    PatchTST 지도학습 모델 인스턴스로부터 인코더 사양(d_model, n_layers) 추론.

    기능:
    - Config 객체, Backbone 속성, 또는 실제 레이어 구조를 검사하여 파라미터 확인.
    - 가중치 변환 로직에 필요한 차원 정보 제공.
    """
    d_model = None
    n_layers = None

    # d_model 추론
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

    # n_layers 추론
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
    """
    state_dict가 torch.nn.TransformerEncoderLayer 스타일의 네이밍을 따르는지 검사.
    """
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
    torch.nn.TransformerEncoderLayer 스타일의 state_dict를
    본 프로젝트의 TSTEncoderLayer 스타일로 변환.

    매핑 로직:
      - Source: in_proj_weight(QKV 통합), out_proj, linear1/2, norm1/2
      - Dest: mha.W_Q/K/V, mha.to_out, ff.0/3, norm_attn/norm_ffn

    참고:
      - QKV 통합 가중치를 분할(Slice)하여 각각의 가중치로 매핑.
      - LayerNorm 파라미터는 affine weight/bias만 매핑 (BatchNorm 버퍼는 변환 불가).
      - 차원 불일치 시 매핑 건너뜀 (Best-effort 방식).
    """
    out = dict(state)  # 원본 복사 후 매핑된 키 추가 및 원본 키 제거 방식

    def _pop(key: str):
        if key in out:
            return out.pop(key)
        return None

    for i in range(n_layers):
        sp = f"{src_prefix}{i}."

        # Source 키 추출
        in_w = _pop(sp + "self_attn.in_proj_weight")
        in_b = _pop(sp + "self_attn.in_proj_bias")
        out_w = _pop(sp + "self_attn.out_proj.weight")
        out_b = _pop(sp + "self_attn.out_proj.bias")
        l1_w = _pop(sp + "linear1.weight")
        l1_b = _pop(sp + "linear1.bias")
        l2_w = _pop(sp + "linear2.weight")
        l2_b = _pop(sp + "linear2.bias")

        # LayerNorm 파라미터
        n1_w = _pop(sp + "norm1.weight")
        n1_b = _pop(sp + "norm1.bias")
        n2_w = _pop(sp + "norm2.weight")
        n2_b = _pop(sp + "norm2.bias")

        dp = f"{dst_prefix}{i}."

        # ---- QKV 가중치 분할 및 매핑 ----
        if in_w is not None:
            if in_w.shape[0] != 3 * d_model or in_w.shape[1] != d_model:
                pass  # 모양 불일치 시 스킵
            else:
                out[dp + "mha.W_Q.weight"] = in_w[0:d_model, :].contiguous()
                out[dp + "mha.W_K.weight"] = in_w[d_model:2 * d_model, :].contiguous()
                out[dp + "mha.W_V.weight"] = in_w[2 * d_model:3 * d_model, :].contiguous()
        if in_b is not None:
            if in_b.shape[0] == 3 * d_model:
                out[dp + "mha.W_Q.bias"] = in_b[0:d_model].contiguous()
                out[dp + "mha.W_K.bias"] = in_b[d_model:2 * d_model].contiguous()
                out[dp + "mha.W_V.bias"] = in_b[2 * d_model:3 * d_model].contiguous()

        # ---- Output Projection 매핑 ----
        if out_w is not None:
            out[dp + "mha.to_out.0.weight"] = out_w.contiguous()
        if out_b is not None:
            out[dp + "mha.to_out.0.bias"] = out_b.contiguous()

        # ---- FFN 매핑 ----
        if l1_w is not None:
            out[dp + "ff.0.weight"] = l1_w.contiguous()
        if l1_b is not None:
            out[dp + "ff.0.bias"] = l1_b.contiguous()
        if l2_w is not None:
            out[dp + "ff.3.weight"] = l2_w.contiguous()
        if l2_b is not None:
            out[dp + "ff.3.bias"] = l2_b.contiguous()

        # ---- Norm 매핑 (Best-effort) ----
        # Target 모델이 BatchNorm을 사용하는 경우에도 Affine 파라미터는 매핑 시도
        if n1_w is not None and n1_w.numel() == d_model:
            out[dp + "norm_attn.1.weight"] = n1_w.contiguous()
        if n1_b is not None and n1_b.numel() == d_model:
            out[dp + "norm_attn.1.bias"] = n1_b.contiguous()
        if n2_w is not None and n2_w.numel() == d_model:
            out[dp + "norm_ffn.1.weight"] = n2_w.contiguous()
        if n2_b is not None and n2_b.numel() == d_model:
            out[dp + "norm_ffn.1.bias"] = n2_b.contiguous()

    # SSL 전용 토큰/헤드 제거
    out.pop("backbone.mask_token", None)

    return out


def _map_patch_embed_to_input_proj(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    SSL 모델의 patch_embed 파라미터를 지도학습 모델의 input_proj로 매핑.

    Key Mapping:
      backbone.patch_embed -> backbone.input_proj
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
    """
    모델이 위치 인코딩을 기대하지만 체크포인트에 없는 경우, 사인파 인코딩으로 초기화.

    기능:
    - 사전학습 백본이 비학습(sinusoidal) 인코딩을 사용했을 경우, 파인튜닝 시점에도 동일한 사전 분포 제공.
    """
    if "backbone.pos_enc" in loaded:
        return loaded
    if "backbone.pos_enc" not in model_state:
        return loaded

    pe0 = model_state["backbone.pos_enc"]
    # 3D: [1, n, d]
    if pe0.dim() == 3 and pe0.shape[0] == 1:
        n, d = int(pe0.shape[1]), int(pe0.shape[2])
        pe = _sinusoidal_pos_emb(n, d, device=pe0.device, dtype=pe0.dtype)
        loaded = dict(loaded)
        loaded["backbone.pos_enc"] = pe
        return loaded
    # 2D: [n, d]
    if pe0.dim() == 2:
        n, d = int(pe0.shape[0]), int(pe0.shape[1])
        pe = _sinusoidal_pos_emb(n, d, device=pe0.device, dtype=pe0.dtype)[0]
        loaded = dict(loaded)
        loaded["backbone.pos_enc"] = pe
        return loaded

    return loaded


def _select_loadable_keys(
        state: Dict[str, torch.Tensor],
        model_state: Dict[str, torch.Tensor],
        *,
        allow_prefixes: Tuple[str, ...] = ("backbone.", "revin_layer."),
) -> Dict[str, torch.Tensor]:
    """
    타겟 모델에 존재하고 허용된 접두사를 가진 키만 선별.
    """
    model_keys = set(model_state.keys())
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not k.startswith(allow_prefixes):
            continue
        if k in model_keys:
            out[k] = v
    return out


def _report_match_stats(loaded: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]) -> None:
    """로드된 가중치와 모델 파라미터 간 매칭 통계 출력."""
    mkeys = set(model_state.keys())
    lkeys = set(loaded.keys())
    matched = len(lkeys)
    missing = sorted(list(mkeys - lkeys))
    print(f"[Finetune] matched: {matched}")
    print(f"[Finetune] missing: {len(missing)} / total: {len(mkeys)} ratio={len(missing) / max(1, len(mkeys)):.3f}")
    if missing:
        print("[Finetune] missing sample (first 30):", missing[:30])


# ---------------------------------------------------------------------
# Optional freezing helpers
# ---------------------------------------------------------------------
def _freeze_encoder_blocks(model) -> None:
    """백본 인코더 및 관련 레이어 동결 (Requires Grad = False)."""
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
    """모델의 모든 파라미터 동결 해제."""
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
    PatchTST 지도학습 파인튜닝 실행 (Runner).

    절차:
      1) (선택) 사전학습(Self-supervised) 가중치 로드 및 변환.
      2) (선택) 초기 학습 단계(Stage 0) 동안 인코더 동결.
      3) 기존 지도학습 파이프라인(train_patchtst) 실행.
      4) (선택) 인코더 동결 해제.

    특징:
      - torch.nn.TransformerEncoderLayer와 본 프로젝트 TSTEncoderLayer 간의 가중치 변환 지원.
      - Best-effort 로딩: 타겟 모델 구조에 맞는 가중치만 선별하여 로드.
    """
    assert train_cfg is not None, "train_cfg는 필수입니다."

    # 1) 사전학습 체크포인트 로드 (존재 시)
    if pretrain_ckpt_path is not None:
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(pretrain_ckpt_path)

        raw_state, meta = _load_pretrain_blob(pretrain_ckpt_path)
        raw_state = _strip_common_prefixes(raw_state)
        raw_state = _drop_revin_stats(raw_state)

        # 가중치 변환 (TransformerEncoder -> TSTEncoder)
        model_state = model.state_dict()
        d_model, n_layers = _infer_supervised_encoder_spec(model)

        state = raw_state
        if _looks_like_torch_transformer_encoder(raw_state):
            state = _convert_torch_transformer_to_tst(raw_state, d_model=d_model, n_layers=n_layers)

        # SSL patch_embed -> Supervised input_proj 매핑
        state = _map_patch_embed_to_input_proj(state)

        # 로드 가능한 키 선별 및 위치 인코딩 보정
        filtered = _select_loadable_keys(state, model_state)
        filtered = _maybe_add_sinusoidal_pos_enc(filtered, model_state)

        # 매칭 통계 출력 및 로드 실행
        print("[Finetune] loaded pretrain ckpt:", pretrain_ckpt_path)
        if meta:
            print("[Finetune] ckpt meta keys:", list(meta.keys()))
        _report_match_stats(filtered, model_state)

        missing, unexpected = model.load_state_dict(filtered, strict=load_strict)

        # 로드 결과 요약
        print(f"[Finetune] load_strict={load_strict}")
        print(f"[Finetune] load_state_dict -> missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            print("  - missing (first 30):", missing[:30])
        if unexpected:
            print("  - unexpected (first 30):", unexpected[:30])

    # 2) 파인튜닝 전 인코더 동결 (옵션)
    if freeze_encoder_before_ft:
        _freeze_encoder_blocks(model)
        print("[Finetune] encoder blocks frozen (stage0).")

    # 3) 지도학습 파이프라인 실행
    out = train_patchtst(
        model,
        train_loader,
        val_loader,
        stages=stages,
        train_cfg=train_cfg,
        future_exo_cb=future_exo_cb,
        exo_is_normalized=exo_is_normalized,
    )

    # 4) 학습 후 동결 해제 (옵션)
    if freeze_encoder_before_ft and unfreeze_after_stage0:
        _unfreeze_all(model)
        print("[Finetune] encoder blocks unfrozen (post stage0).")

    return out