"""Model-specific tensor layouts derived from the shared ICL batch contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AutoTimesICLInput:
    """AutoTimes numeric prompt followed by the query context."""

    packed_context: torch.Tensor
    query_target: torch.Tensor
    prompt_mask: torch.Tensor
    series_ids: tuple[str, ...]
    packed_exogenous: torch.Tensor | None = None
    query_target_exogenous: torch.Tensor | None = None


@dataclass(frozen=True)
class SELLMICLInput:
    """SELLM prompt segments kept separate from the query context."""

    demonstration_contexts: torch.Tensor
    demonstration_targets: torch.Tensor
    query_context: torch.Tensor
    query_target: torch.Tensor
    prompt_mask: torch.Tensor
    series_ids: tuple[str, ...]
    demonstration_context_exogenous: torch.Tensor | None = None
    demonstration_target_exogenous: torch.Tensor | None = None
    query_context_exogenous: torch.Tensor | None = None
    query_target_exogenous: torch.Tensor | None = None


class AutoTimesICLAdapter:
    """Pack demonstrations using the numeric-series prompt layout from AutoTimes."""

    def adapt(self, batch) -> AutoTimesICLInput:
        _validate_batch(batch)
        batch_size, prompt_count, _, channels = batch.demonstration_contexts.shape
        prompt_pairs = torch.cat(
            [batch.demonstration_contexts, batch.demonstration_targets],
            dim=2,
        )
        packed_prompts = prompt_pairs.reshape(batch_size, -1, channels)
        packed_context = torch.cat([packed_prompts, batch.query_context], dim=1)
        packed_exogenous = None
        if batch.query_context_exogenous is not None:
            past_width = int(batch.query_context_exogenous.shape[-1])
            future_width = int(batch.query_target_exogenous.shape[-1])

            def past_role(value: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    [
                        value,
                        value.new_zeros(*value.shape[:-1], future_width),
                    ],
                    dim=-1,
                )

            def future_role(value: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    [
                        value.new_zeros(*value.shape[:-1], past_width),
                        value,
                    ],
                    dim=-1,
                )

            prompt_exogenous = torch.cat(
                [
                    past_role(batch.demonstration_context_exogenous),
                    future_role(batch.demonstration_target_exogenous),
                ],
                dim=2,
            )
            packed_exogenous = torch.cat(
                [
                    prompt_exogenous.reshape(batch_size, -1, prompt_exogenous.shape[-1]),
                    past_role(batch.query_context_exogenous),
                ],
                dim=1,
            )
        return AutoTimesICLInput(
            packed_context=packed_context,
            query_target=batch.query_target,
            prompt_mask=batch.prompt_mask,
            series_ids=batch.series_ids,
            packed_exogenous=packed_exogenous,
            query_target_exogenous=batch.query_target_exogenous,
        )


class SELLMICLAdapter:
    """Preserve prompt boundaries for SELLM semantic prompt encoding."""

    def adapt(self, batch) -> SELLMICLInput:
        _validate_batch(batch)
        return SELLMICLInput(
            demonstration_contexts=batch.demonstration_contexts,
            demonstration_targets=batch.demonstration_targets,
            query_context=batch.query_context,
            query_target=batch.query_target,
            prompt_mask=batch.prompt_mask,
            series_ids=batch.series_ids,
            demonstration_context_exogenous=batch.demonstration_context_exogenous,
            demonstration_target_exogenous=batch.demonstration_target_exogenous,
            query_context_exogenous=batch.query_context_exogenous,
            query_target_exogenous=batch.query_target_exogenous,
        )


def _validate_batch(batch) -> None:
    if batch.query_context.ndim != 3 or batch.query_target.ndim != 3:
        raise ValueError("ICL query tensors must be [B,L,C] and [B,H,C].")
    if batch.demonstration_contexts.ndim != 4:
        raise ValueError("ICL demonstration contexts must be [B,K,L,C].")
    if batch.demonstration_targets.ndim != 4:
        raise ValueError("ICL demonstration targets must be [B,K,H,C].")
    if batch.prompt_mask.ndim != 2:
        raise ValueError("ICL prompt mask must be [B,K].")
    if batch.demonstration_contexts.shape[:2] != batch.prompt_mask.shape:
        raise ValueError("ICL prompt mask dimensions do not match demonstrations.")
    if not bool(batch.prompt_mask.all()):
        raise ValueError(
            "AutoTimes and SELLM ICL v1 require every prompt slot to be populated."
        )
    exogenous_fields = (
        batch.query_context_exogenous,
        batch.query_target_exogenous,
        batch.demonstration_context_exogenous,
        batch.demonstration_target_exogenous,
    )
    populated = tuple(value is not None for value in exogenous_fields)
    if any(populated) and not all(populated):
        raise ValueError("ICL exogenous tensors must be provided as one complete set.")
