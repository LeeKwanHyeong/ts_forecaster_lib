from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "TitanBaseModel",
    "TitanLMMModel",
    "TitanSeq2SeqModel",
]

from modeling_module.models.common_layers.RevIN import RevIN

# --- 패키지/로컬 모두 대응 가능한 유연한 임포트 ---
try:
    from modeling_module.models.Titan.backbone import MemoryEncoder
    from modeling_module.models.Titan.common.decoder import TitanDecoder
except Exception:
    from backbone import MemoryEncoder          # type: ignore
    from decoder import TitanDecoder            # type: ignore

# TitanConfig 는 configs.py 에 정의되어 있다고 가정
try:
    from modeling_module.models.Titan.common.configs import TitanConfig
except Exception:
    try:
        from configs import TitanConfig         # type: ignore
    except Exception:
        TitanConfig = None                      # type: ignore


def _merge_cfg_kwargs(cfg_obj, **kwargs):
    """cfg(dataclass)와 kwargs 병합. kwargs 우선."""
    cfg_dict = asdict(cfg_obj) if cfg_obj is not None else {}
    cfg_dict.update(kwargs)
    return cfg_dict

class _PastExoEmbed(nn.Module):
    """
    past_exo_cont: [B, L, Dc]
    past_exo_cat : [B, L, K] (int/long 권장)
    -> out        : [B, L, Dc + sum(embed_dims)]
    """
    def __init__(
        self,
        *,
        cont_dim: int,
        cat_vocab_sizes: Sequence[int],
        cat_embed_dims: Sequence[int],
    ):
        super().__init__()
        self.cont_dim = int(cont_dim)
        self.cat_vocab_sizes = list(cat_vocab_sizes)
        self.cat_embed_dims = list(cat_embed_dims)

        assert len(self.cat_vocab_sizes) == len(self.cat_embed_dims), \
            "past_exo_cat_vocab_sizes and past_exo_cat_embed_dims must have same length"

        self.cat_embs = nn.ModuleList([
            nn.Embedding(int(vs), int(ed))
            for vs, ed in zip(self.cat_vocab_sizes, self.cat_embed_dims)
        ])

        self.out_dim = self.cont_dim + sum(self.cat_embed_dims)

    def forward(
        self,
        past_exo_cont: Optional[torch.Tensor],
        past_exo_cat: Optional[torch.Tensor],
        *,
        B: int,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feats = []

        # cont
        if self.cont_dim > 0:
            if past_exo_cont is None:
                feats.append(torch.zeros(B, L, self.cont_dim, device=device, dtype=dtype))
            else:
                feats.append(past_exo_cont.to(device=device, dtype=dtype))

        # cat
        K = len(self.cat_embs)
        if K > 0:
            if past_exo_cat is None:
                # unknown category -> 0으로 처리
                past_exo_cat = torch.zeros(B, L, K, device=device, dtype=torch.long)
            else:
                past_exo_cat = past_exo_cat.to(device=device)
                if past_exo_cat.dtype != torch.long:
                    past_exo_cat = past_exo_cat.long()

            for i, emb in enumerate(self.cat_embs):
                # [B,L] -> [B,L,ed]
                feats.append(emb(past_exo_cat[:, :, i]))

        if not feats:
            return torch.zeros(B, L, 0, device=device, dtype=dtype)

        return torch.cat(feats, dim=-1)


class _TitanBase(nn.Module):
    """
    공통 베이스: config 보관 + Encoder/Decoder 조립
    - 외생변수/비음수 클램프 등 공통 옵션은 여기서 통일
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__()
        params = _merge_cfg_kwargs(config, **kwargs)

        # Core
        self.input_dim: int = int(params["input_dim"])
        self.lookback: int = int(params["lookback"])
        self.horizon: int = int(params["horizon"])
        self.d_model: int = int(params.get("d_model", 256))
        self.n_layers: int = int(params.get("n_layers", 3))
        self.n_heads: int = int(params.get("n_heads", 4))
        self.d_ff: int = int(params.get("d_ff", 512))
        self.dropout: float = float(params.get("dropout", 0.1))

        # RevIN 옵션 (configs.py에 이미 존재)
        self.use_revin = bool(params.get("use_revin", True))
        self.revin_subtract_last = bool(params.get("revin_subtract_last", False))
        self.revin_affine = bool(params.get("revin_affine", True))
        self.revin_use_std = bool(params.get("revin_use_std", True))

        # Memory/LMM
        self.contextual_mem_size: int = int(params.get("contextual_mem_size", 256))
        self.persistent_mem_size: int = int(params.get("persistent_mem_size", 64))

        # Exogenous
        self.use_exogenous_mode: bool = bool(params.get("use_exogenous_mode", False))
        self.exo_dim: int = int(params.get("exo_dim", 0))
        self.use_calendar_exo: bool = bool(params.get("use_calendar_exo", False))

        # Output constraint
        self.final_clamp_nonneg: bool = bool(params.get("final_clamp_nonneg", True))

        # Seq2Seq decoder 전용 파라미터(필요 시 사용)
        self.dec_layers: int = int(params.get("dec_layers", 2))
        self.dec_heads: int = int(params.get("dec_heads", 4))
        self.dec_d_ff: int = int(params.get("dec_d_ff", 512))
        self.dec_dropout: float = float(params.get("dec_dropout", 0.1))

        # 원본 config 보관(트레이너가 참조 가능)
        self.config = config

        self.past_exo_cont_dim = int(params.get("past_exo_cont_dim", 0))
        self.past_exo_cat_dim = int(params.get("past_exo_cat_dim", 0))

        vocab_sizes = params.get("past_exo_cat_vocab_sizes", ())
        embed_dims = params.get("past_exo_cat_embed_dims", ())

        # 안전: cat_dim>0인데 tuple이 비어있으면 기본값 채움
        if self.past_exo_cat_dim > 0 and (not vocab_sizes or not embed_dims):
            vocab_sizes = tuple([512] * self.past_exo_cat_dim)
            embed_dims = tuple([16] * self.past_exo_cat_dim)

        self.past_exo_cat_vocab_sizes = tuple(vocab_sizes)
        self.past_exo_cat_embed_dims = tuple(embed_dims)

        # past embedding helper
        self.past_exo_embed = _PastExoEmbed(
            cont_dim=self.past_exo_cont_dim,
            cat_vocab_sizes=self.past_exo_cat_vocab_sizes,
            cat_embed_dims=self.past_exo_cat_embed_dims,
        )

        # Titan encoder는 "타깃 1채널 + past_exo_features"를 입력으로 받도록 고정
        self.encoder_input_dim = 1 + self.past_exo_embed.out_dim

        self.encoder = MemoryEncoder(
            self.encoder_input_dim,  # <= 기존 self.input_dim 대신 확장 dim 사용
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_ff,
            self.contextual_mem_size,
            self.persistent_mem_size,
            self.dropout,
            use_context_update=False,
        )

        self.target_channel = int(params.get("target_channel", 0))
        self.revin = RevIN(
            num_features=1,  # 단일 타깃 채널 기준
            affine=self.revin_affine,
            subtract_last=self.revin_subtract_last,
            use_std=self.revin_use_std
        ) if self.use_revin else None


        # Encoder
        self.encoder = MemoryEncoder(
            self.input_dim,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.d_ff,
            self.contextual_mem_size,
            self.persistent_mem_size,
            self.dropout,
            use_context_update=False  # 일단 안전하게 False로 고정하거나 config 옵션 연동
        )

    def _make_encoder_input(
            self,
            x: torch.Tensor,
            past_exo_cont: Optional[torch.Tensor],
            past_exo_cat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        x: [B,L,C]
        -> encoder_in: [B,L, 1 + past_exo_dim]
        """
        # 1) RevIN은 타깃 채널(1개)만 norm
        x = self._maybe_revin_norm(x)  # 기존 로직 유지
        tc = int(self.target_channel)
        x_t = x[:, :, tc:tc + 1]  # [B,L,1]

        B, L, _ = x_t.shape
        exo = self.past_exo_embed(
            past_exo_cont,
            past_exo_cat,
            B=B, L=L,
            device=x_t.device,
            dtype=x_t.dtype,
        )  # [B,L,past_dim]

        return torch.cat([x_t, exo], dim=-1)

    @classmethod
    def from_config(cls, config: "TitanConfig"):
        return cls(config=config)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        return y.clamp_min(0) if self.final_clamp_nonneg else y

    # 입력 x: [B, L, C] -> RevIN(norm) -> [B, L, C]
    def _maybe_revin_norm(self, x: torch.Tensor) -> torch.Tensor:
        if (self.revin is None) or (x.size(-1) == 0):
            print('revin is None So cannot normalize')
            return x
        # 타깃 채널만 정규화/복원(멀티채널 안전)
        tc = self.target_channel
        x_t = x[:, :, tc:tc+1]
        x_t = self.revin(x_t, mode='norm')   # [B,L,1]
        x = x.clone()
        x[:, :, tc:tc+1] = x_t
        return x

    # 출력 y: [B, H] -> RevIN(denorm) -> [B, H]
    def _maybe_revin_denorm(self, y: torch.Tensor) -> torch.Tensor:
        if (self.revin is None) or (y.dim() != 2):
            print('revin is None so cannot denormalize')
            return y
        # RevIN은 [B,L,C] 형태를 기대하므로 H축을 L로 보고, C=1로 맞춰 복원
        y_in = y.unsqueeze(-1)       # [B, H, 1]
        y_out = self.revin(y_in, mode='denorm')  # [B, H, 1]
        return y_out.squeeze(-1)

    def _clamp(self, y: torch.Tensor) -> torch.Tensor:
        return y.clamp_min(0) if self.final_clamp_nonneg else y


class TitanBaseModel(_TitanBase):
    """
    Encoder-only: TitanDecoder를 수평(H) 차원 투영기로 사용하고 Linear로 1채널 예측.
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=1,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous_mode else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_in = self._make_encoder_input(x, past_exo_cont, past_exo_cat)  # [B,L,1+past_dim]
        memory = self.encoder(encoder_in)                                      # [B,L,D]
        dec = self.decoder(memory, future_exo)                                 # [B,H,D]
        y = self.proj(dec).squeeze(-1)                                         # [B,H]
        y = self._maybe_revin_denorm(y)
        return self._clamp(y)


class TitanLMMModel(_TitanBase):
    """
    LMM 특화 디코딩이 필요하면 TitanDecoder 내부에서 분기하도록 구성(여기선 공용 Decoder 사용).
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=1,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous_mode else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, x, *, future_exo=None, past_exo_cont=None, past_exo_cat=None):
        encoder_in = self._make_encoder_input(x, past_exo_cont, past_exo_cat)
        memory = self.encoder(encoder_in)
        dec = self.decoder(memory, future_exo)
        y = self.proj(dec).squeeze(-1)
        y = self._maybe_revin_denorm(y)
        return self._clamp(y)

class TitanSeq2SeqModel(_TitanBase):
    """
    Seq2Seq: 다층 디코더를 이용하여 미래 H 단계의 컨텍스트를 생성 후 1채널 예측.
    """
    def __init__(self, *, config: Optional["TitanConfig"]=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.decoder = TitanDecoder(
            d_model=self.d_model,
            n_layers=self.dec_layers,
            n_heads=self.dec_heads,
            d_ff=self.dec_d_ff,
            dropout=self.dec_dropout,
            horizon=self.horizon,
            exo_dim=(self.exo_dim if self.use_exogenous_mode else 0),
        )
        self.proj = nn.Linear(self.d_model, 1)

    def forward(self, x, *, future_exo=None, past_exo_cont=None, past_exo_cat=None):
        encoder_in = self._make_encoder_input(x, past_exo_cont, past_exo_cat)
        memory = self.encoder(encoder_in)
        dec = self.decoder(memory, future_exo)
        y = self.proj(dec).squeeze(-1)
        y = self._maybe_revin_denorm(y)
        return self._clamp(y)
