from __future__ import annotations

from typing import Any, Optional

import torch

from .PatchMixer import _PatchMixerProjectCore


_RETIRED_EXOGENOUS_POINT_STATE = (
    "out_scale",
    "out_bias",
    "dw_gain",
    "dw_head.weight",
    "dw_head.bias",
)


def _drop_retired_exogenous_point_state(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    for name in _RETIRED_EXOGENOUS_POINT_STATE:
        state_dict.pop(prefix + name, None)


def patchmixer_exogenous_widths(cfg: Any) -> tuple[int, int, int]:
    return (
        int(getattr(cfg, "past_exo_cont_dim", 0) or 0),
        int(getattr(cfg, "past_exo_cat_dim", 0) or 0),
        int(getattr(cfg, "future_exo_dim", 0) or 0),
    )


def _require_exogenous_config(cfg: Any, *, model_name: str) -> None:
    past_cont_width, past_cat_width, future_width = patchmixer_exogenous_widths(cfg)
    if not any((past_cont_width, past_cat_width, future_width)):
        raise ValueError(f"{model_name} requires at least one configured exogenous input.")
    if (past_cont_width > 0 or past_cat_width > 0) and str(
        getattr(cfg, "past_exo_mode", "none")
    ).lower() != "z_gate":
        raise ValueError(
            f"{model_name} requires past_exo_mode='z_gate' for past exogenous fusion."
        )


def _validate_required_inputs(
    cfg: Any,
    *,
    past_exo_cont: Optional[torch.Tensor],
    past_exo_cat: Optional[torch.Tensor],
    future_exo: Optional[torch.Tensor],
    model_name: str,
) -> None:
    past_cont_width, past_cat_width, future_width = patchmixer_exogenous_widths(cfg)
    missing: list[str] = []
    if past_cont_width > 0 and past_exo_cont is None:
        missing.append("past_exo_cont")
    if past_cat_width > 0 and past_exo_cat is None:
        missing.append("past_exo_cat")
    if future_width > 0 and future_exo is None:
        missing.append("future_exo")
    if missing:
        raise RuntimeError(f"{model_name} is missing required inputs: {', '.join(missing)}.")


class PatchMixerExogenousModel(_PatchMixerProjectCore):
    """Point forecaster with gated past fusion and a future residual shift."""

    architecture_variant = "exogenous"
    exogenous_fusion_strategy = "gated_residual+future_shift"

    def __init__(self, cfg):
        _require_exogenous_config(cfg, model_name=type(self).__name__)
        if int(getattr(cfg, "out_mul", 1)) != 1:
            raise ValueError("PatchMixerExogenousModel supports point output only.")
        super().__init__(cfg)
        # Preserve the established initialization sequence, then remove five
        # distribution-only tensors that never participate in point forward.
        for name in ("out_scale", "out_bias", "dw_gain", "dw_head"):
            delattr(self, name)
        self.learn_output_scale = False
        self.learn_dw_gain = False
        self.register_load_state_dict_pre_hook(_drop_retired_exogenous_point_state)

    def forward(
        self,
        x: torch.Tensor,
        future_exo: Optional[torch.Tensor] = None,
        *,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        part_ids: Optional[torch.Tensor] = None,
        exo_is_normalized: Optional[bool] = None,
        **kwargs,
    ):
        self._validate_future_exo_contract(future_exo, batch_size=x.size(0))
        _validate_required_inputs(
            self.configs,
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
            exo_is_normalized=exo_is_normalized,
            **kwargs,
        )
