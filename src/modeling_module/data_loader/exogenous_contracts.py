from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Optional

import torch


EXOGENOUS_SCHEMA_VERSION = 1


def _normalize_feature_names(values: Optional[Iterable[str]], *, field: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in (values or ()))
    if any(not name for name in names):
        raise ValueError(f"{field} cannot contain empty feature names.")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate feature names: {names}.")
    return names


@dataclass(frozen=True, slots=True)
class ExogenousFeatureSchema:
    """Ordered exogenous feature identity used by data and checkpoints."""

    past_cont_names: tuple[str, ...] = ()
    past_cat_names: tuple[str, ...] = ()
    future_cont_names: tuple[str, ...] = ()
    past_cat_cardinalities: tuple[int, ...] = ()
    version: int = EXOGENOUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != EXOGENOUS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported exogenous schema version={self.version}; "
                f"expected {EXOGENOUS_SCHEMA_VERSION}."
            )

        past_cont = _normalize_feature_names(self.past_cont_names, field="past_cont_names")
        past_cat = _normalize_feature_names(self.past_cat_names, field="past_cat_names")
        future_cont = _normalize_feature_names(self.future_cont_names, field="future_cont_names")
        cardinalities = tuple(int(value) for value in self.past_cat_cardinalities)

        if cardinalities and len(cardinalities) != len(past_cat):
            raise ValueError(
                "past_cat_cardinalities must be empty or match past_cat_names: "
                f"{len(cardinalities)} != {len(past_cat)}."
            )
        if any(value <= 0 for value in cardinalities):
            raise ValueError("past_cat_cardinalities must contain positive integers.")

        continuous_names = set(past_cont).union(future_cont)
        categorical_overlap = set(past_cat).intersection(continuous_names)
        if categorical_overlap:
            overlap = ", ".join(sorted(categorical_overlap))
            raise ValueError(
                "A feature cannot be categorical and continuous in the same schema: "
                f"{overlap}."
            )

        object.__setattr__(self, "past_cont_names", past_cont)
        object.__setattr__(self, "past_cat_names", past_cat)
        object.__setattr__(self, "future_cont_names", future_cont)
        object.__setattr__(self, "past_cat_cardinalities", cardinalities)

    @classmethod
    def from_columns(
        cls,
        *,
        past_cont: Optional[Iterable[str]] = None,
        past_cat: Optional[Iterable[str]] = None,
        future_cont: Optional[Iterable[str]] = None,
        past_cat_cardinalities: Optional[Iterable[int]] = None,
    ) -> "ExogenousFeatureSchema":
        return cls(
            past_cont_names=tuple(past_cont or ()),
            past_cat_names=tuple(past_cat or ()),
            future_cont_names=tuple(future_cont or ()),
            past_cat_cardinalities=tuple(past_cat_cardinalities or ()),
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def is_empty(self) -> bool:
        return not (self.past_cont_names or self.past_cat_names or self.future_cont_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "past_cont_names": list(self.past_cont_names),
            "past_cat_names": list(self.past_cat_names),
            "future_cont_names": list(self.future_cont_names),
            "past_cat_cardinalities": list(self.past_cat_cardinalities),
        }


def _none_if_empty_feature_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[-1]) == 0:
        return None
    return value


@dataclass(frozen=True, slots=True, eq=False)
class ExogenousBatch:
    """Canonical in-memory exogenous tensors passed through training and inference."""

    past_cont: Optional[torch.Tensor] = None
    past_cat: Optional[torch.Tensor] = None
    future_cont: Optional[torch.Tensor] = None

    @classmethod
    def from_legacy(
        cls,
        *,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        future_exo: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> "ExogenousBatch":
        future = _none_if_empty_feature_tensor(future_exo)
        if torch.is_tensor(future) and future.ndim == 2 and batch_size is not None:
            future = future.unsqueeze(0).expand(int(batch_size), -1, -1)
        return cls(
            past_cont=_none_if_empty_feature_tensor(past_exo_cont),
            past_cat=_none_if_empty_feature_tensor(past_exo_cat),
            future_cont=future,
        )

    @property
    def is_empty(self) -> bool:
        return self.past_cont is None and self.past_cat is None and self.future_cont is None

    def provided_inputs(self) -> frozenset[str]:
        provided: set[str] = set()
        if self.past_cont is not None:
            provided.add("past_cont")
        if self.past_cat is not None:
            provided.add("past_cat")
        if self.future_cont is not None:
            provided.add("future_cont")
        return frozenset(provided)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ExogenousBatch":
        def move(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            return value.to(device, non_blocking=non_blocking)

        return ExogenousBatch(
            past_cont=move(self.past_cont),
            past_cat=move(self.past_cat),
            future_cont=move(self.future_cont),
        )

    def validate(
        self,
        *,
        batch_size: Optional[int] = None,
        lookback: Optional[int] = None,
        horizon: Optional[int] = None,
        schema: Optional[ExogenousFeatureSchema] = None,
    ) -> "ExogenousBatch":
        tensors = {
            "past_cont": self.past_cont,
            "past_cat": self.past_cat,
            "future_cont": self.future_cont,
        }
        devices: set[torch.device] = set()

        for name, value in tensors.items():
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
            if value.ndim != 3:
                raise ValueError(f"{name} must have rank 3 [B,T,E], got {tuple(value.shape)}.")
            if int(value.shape[-1]) <= 0:
                raise ValueError(f"{name} must contain at least one feature.")
            if batch_size is not None and int(value.shape[0]) != int(batch_size):
                raise ValueError(
                    f"{name} batch mismatch: {int(value.shape[0])} != {int(batch_size)}."
                )

            if name == "past_cat":
                if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise TypeError(f"past_cat must use an integer dtype, got {value.dtype}.")
            elif not torch.is_floating_point(value):
                raise TypeError(f"{name} must use a floating dtype, got {value.dtype}.")

            expected_time = horizon if name == "future_cont" else lookback
            if expected_time is not None and int(value.shape[1]) != int(expected_time):
                raise ValueError(
                    f"{name} time-axis mismatch: {int(value.shape[1])} != {int(expected_time)}."
                )
            devices.add(value.device)

        if len(devices) > 1:
            raise ValueError(f"Exogenous tensors must share one device, got {sorted(map(str, devices))}.")

        if schema is not None:
            expected_widths = {
                "past_cont": len(schema.past_cont_names),
                "past_cat": len(schema.past_cat_names),
                "future_cont": len(schema.future_cont_names),
            }
            for name, expected_width in expected_widths.items():
                value = tensors[name]
                if expected_width == 0 and value is not None:
                    raise ValueError(f"{name} is not declared by the exogenous schema.")
                if expected_width > 0 and value is None:
                    raise ValueError(f"{name} is required by the exogenous schema.")
                if value is not None and int(value.shape[-1]) != expected_width:
                    raise ValueError(
                        f"{name} feature width does not match schema: "
                        f"{int(value.shape[-1])} != {expected_width}."
                    )

        return self

    def as_legacy_kwargs(self) -> dict[str, Optional[torch.Tensor]]:
        return {
            "past_exo_cont": self.past_cont,
            "past_exo_cat": self.past_cat,
            "future_exo": self.future_cont,
        }
