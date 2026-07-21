from __future__ import annotations

from typing import Any, Optional

import torch

from .PatchTST import PatchTSTModel, PatchTSTQuantileModel


def patchtst_exogenous_widths(cfg: Any) -> tuple[int, int, int]:
    return (
        int(getattr(cfg, "past_exo_cont_dim", 0) or 0),
        int(getattr(cfg, "past_exo_cat_dim", 0) or 0),
        int(getattr(cfg, "future_exo_dim", getattr(cfg, "d_future", 0)) or 0),
    )


def patchtst_uses_exogenous_inputs(cfg: Any) -> bool:
    return any(width > 0 for width in patchtst_exogenous_widths(cfg))


def _require_endogenous_config(cfg: Any, *, model_name: str) -> None:
    widths = patchtst_exogenous_widths(cfg)
    if any(widths):
        raise ValueError(
            f"{model_name} requires zero exogenous widths, got "
            f"past_cont={widths[0]}, past_cat={widths[1]}, future_cont={widths[2]}."
        )


def _require_exogenous_config(cfg: Any, *, model_name: str) -> None:
    if not patchtst_uses_exogenous_inputs(cfg):
        raise ValueError(f"{model_name} requires at least one configured exogenous input.")


def _validate_required_inputs(
    cfg: Any,
    *,
    past_exo_cont: Optional[torch.Tensor],
    past_exo_cat: Optional[torch.Tensor],
    future_exo: Optional[torch.Tensor],
    model_name: str,
) -> None:
    past_cont_width, past_cat_width, future_width = patchtst_exogenous_widths(cfg)
    missing: list[str] = []
    if past_cont_width > 0 and past_exo_cont is None:
        missing.append("past_exo_cont")
    if past_cat_width > 0 and past_exo_cat is None:
        missing.append("past_exo_cat")
    if future_width > 0 and future_exo is None:
        missing.append("future_exo")
    if missing:
        raise RuntimeError(f"{model_name} is missing required inputs: {', '.join(missing)}.")


class PatchTSTEndogenousModel(PatchTSTModel):
    """PatchTST point/distribution variant with a strict target-only input contract."""

    architecture_variant = "endogenous"
    exogenous_fusion_strategy = "none"

    def __init__(self, cfg):
        _require_endogenous_config(cfg, model_name=type(self).__name__)
        super().__init__(cfg)

    def forward(
        self,
        x: torch.Tensor,
        *,
        part_ids=None,
        mode: Optional[str] = None,
    ):
        return super().forward(x)


class PatchTSTExogenousModel(PatchTSTModel):
    """PatchTST variant using past patch features and future token cross-attention."""

    architecture_variant = "exogenous"
    exogenous_fusion_strategy = "patch_concat+future_cross_attention"

    def __init__(self, cfg):
        _require_exogenous_config(cfg, model_name=type(self).__name__)
        super().__init__(cfg)

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        _validate_required_inputs(
            self.cfg,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            future_exo=future_exo,
            model_name=type(self).__name__,
        )
        return super().forward(
            x,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            **kwargs,
        )


class PatchTSTQuantileEndogenousModel(PatchTSTQuantileModel):
    """Quantile PatchTST variant with a strict target-only input contract."""

    architecture_variant = "endogenous"
    exogenous_fusion_strategy = "none"

    def __init__(self, cfg, attn_core=None):
        _require_endogenous_config(cfg, model_name=type(self).__name__)
        super().__init__(cfg, attn_core=attn_core)

    def forward(
        self,
        x: torch.Tensor,
        *,
        part_ids=None,
        mode: Optional[str] = None,
    ):
        return super().forward(x, part_ids=part_ids, mode=mode)


class PatchTSTQuantileExogenousModel(PatchTSTQuantileModel):
    """Quantile PatchTST variant with explicit exogenous inputs."""

    architecture_variant = "exogenous"
    exogenous_fusion_strategy = "patch_concat+future_cross_attention"

    def __init__(self, cfg, attn_core=None):
        _require_exogenous_config(cfg, model_name=type(self).__name__)
        super().__init__(cfg, attn_core=attn_core)

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids=None,
        mode: Optional[str] = None,
        **kwargs,
    ):
        _validate_required_inputs(
            self.cfg,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            future_exo=future_exo,
            model_name=type(self).__name__,
        )
        return super().forward(
            x,
            future_exo=future_exo,
            past_exo_cont=past_exo_cont,
            past_exo_cat=past_exo_cat,
            part_ids=part_ids,
            mode=mode,
            **kwargs,
        )
