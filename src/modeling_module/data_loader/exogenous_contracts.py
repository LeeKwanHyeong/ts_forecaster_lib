from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Optional

import torch


LEGACY_EXOGENOUS_SCHEMA_VERSION = 1
EXOGENOUS_SCHEMA_VERSION = 2


def _normalize_feature_names(values: Optional[Iterable[str]], *, field: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in (values or ()))
    if any(not name for name in names):
        raise ValueError(f"{field} cannot contain empty feature names.")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate feature names: {names}.")
    return names


@dataclass(frozen=True, slots=True)
class ExogenousFeatureSchema:
    """Ordered exogenous feature identity used by data and checkpoints.

    A categorical feature may appear in both past and future windows. In that
    case both tensors must use the same category-ID vocabulary.
    """

    past_cont_names: tuple[str, ...] = ()
    past_cat_names: tuple[str, ...] = ()
    future_cont_names: tuple[str, ...] = ()
    past_cat_cardinalities: tuple[int, ...] = ()
    version: int = EXOGENOUS_SCHEMA_VERSION
    future_cat_names: tuple[str, ...] = ()
    future_cat_cardinalities: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        version = int(self.version)
        if version not in (LEGACY_EXOGENOUS_SCHEMA_VERSION, EXOGENOUS_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported exogenous schema version={self.version}; "
                f"expected one of "
                f"{{{LEGACY_EXOGENOUS_SCHEMA_VERSION}, {EXOGENOUS_SCHEMA_VERSION}}}."
            )

        past_cont = _normalize_feature_names(self.past_cont_names, field="past_cont_names")
        past_cat = _normalize_feature_names(self.past_cat_names, field="past_cat_names")
        future_cont = _normalize_feature_names(self.future_cont_names, field="future_cont_names")
        future_cat = _normalize_feature_names(self.future_cat_names, field="future_cat_names")
        past_cardinalities = tuple(int(value) for value in self.past_cat_cardinalities)
        future_cardinalities = tuple(int(value) for value in self.future_cat_cardinalities)

        if version == LEGACY_EXOGENOUS_SCHEMA_VERSION and (
            future_cat or future_cardinalities
        ):
            raise ValueError(
                "future categorical features require exogenous schema version "
                f"{EXOGENOUS_SCHEMA_VERSION}."
            )

        if past_cardinalities and len(past_cardinalities) != len(past_cat):
            raise ValueError(
                "past_cat_cardinalities must be empty or match past_cat_names: "
                f"{len(past_cardinalities)} != {len(past_cat)}."
            )
        if future_cardinalities and len(future_cardinalities) != len(future_cat):
            raise ValueError(
                "future_cat_cardinalities must be empty or match future_cat_names: "
                f"{len(future_cardinalities)} != {len(future_cat)}."
            )
        if any(value <= 0 for value in past_cardinalities):
            raise ValueError("past_cat_cardinalities must contain positive integers.")
        if any(value <= 0 for value in future_cardinalities):
            raise ValueError("future_cat_cardinalities must contain positive integers.")

        continuous_names = set(past_cont).union(future_cont)
        categorical_names = set(past_cat).union(future_cat)
        categorical_overlap = categorical_names.intersection(continuous_names)
        if categorical_overlap:
            overlap = ", ".join(sorted(categorical_overlap))
            raise ValueError(
                "A feature cannot be categorical and continuous in the same schema: "
                f"{overlap}."
            )

        if past_cardinalities and future_cardinalities:
            past_by_name = dict(zip(past_cat, past_cardinalities))
            future_by_name = dict(zip(future_cat, future_cardinalities))
            for name in sorted(set(past_by_name).intersection(future_by_name)):
                if past_by_name[name] != future_by_name[name]:
                    raise ValueError(
                        "A categorical feature shared by past and future windows must "
                        f"use one cardinality: {name} has {past_by_name[name]} and "
                        f"{future_by_name[name]}."
                    )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "past_cont_names", past_cont)
        object.__setattr__(self, "past_cat_names", past_cat)
        object.__setattr__(self, "future_cont_names", future_cont)
        object.__setattr__(self, "future_cat_names", future_cat)
        object.__setattr__(self, "past_cat_cardinalities", past_cardinalities)
        object.__setattr__(self, "future_cat_cardinalities", future_cardinalities)

    @classmethod
    def from_columns(
        cls,
        *,
        past_cont: Optional[Iterable[str]] = None,
        past_cat: Optional[Iterable[str]] = None,
        future_cont: Optional[Iterable[str]] = None,
        past_cat_cardinalities: Optional[Iterable[int]] = None,
        future_cat: Optional[Iterable[str]] = None,
        future_cat_cardinalities: Optional[Iterable[int]] = None,
    ) -> "ExogenousFeatureSchema":
        return cls(
            past_cont_names=tuple(past_cont or ()),
            past_cat_names=tuple(past_cat or ()),
            future_cont_names=tuple(future_cont or ()),
            past_cat_cardinalities=tuple(past_cat_cardinalities or ()),
            future_cat_names=tuple(future_cat or ()),
            future_cat_cardinalities=tuple(future_cat_cardinalities or ()),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExogenousFeatureSchema":
        if not isinstance(payload, Mapping):
            raise TypeError("Exogenous schema payload must be a mapping.")
        allowed = {
            "version",
            "past_cont_names",
            "past_cat_names",
            "future_cont_names",
            "past_cat_cardinalities",
            "future_cat_names",
            "future_cat_cardinalities",
        }
        unexpected = set(payload).difference(allowed)
        if unexpected:
            raise ValueError(
                "Exogenous schema payload contains unsupported fields: "
                + ", ".join(sorted(unexpected))
            )
        return cls(
            version=int(payload.get("version", EXOGENOUS_SCHEMA_VERSION)),
            past_cont_names=tuple(payload.get("past_cont_names", ())),
            past_cat_names=tuple(payload.get("past_cat_names", ())),
            future_cont_names=tuple(payload.get("future_cont_names", ())),
            past_cat_cardinalities=tuple(
                payload.get("past_cat_cardinalities", ())
            ),
            future_cat_names=tuple(payload.get("future_cat_names", ())),
            future_cat_cardinalities=tuple(
                payload.get("future_cat_cardinalities", ())
            ),
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
        return not (
            self.past_cont_names
            or self.past_cat_names
            or self.future_cont_names
            or self.future_cat_names
        )

    @property
    def categorical_feature_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*self.past_cat_names, *self.future_cat_names))
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": int(self.version),
            "past_cont_names": list(self.past_cont_names),
            "past_cat_names": list(self.past_cat_names),
            "future_cont_names": list(self.future_cont_names),
            "past_cat_cardinalities": list(self.past_cat_cardinalities),
        }
        if self.version >= EXOGENOUS_SCHEMA_VERSION:
            payload["future_cat_names"] = list(self.future_cat_names)
            payload["future_cat_cardinalities"] = list(self.future_cat_cardinalities)
        return payload


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
    future_cat: Optional[torch.Tensor] = None

    @classmethod
    def from_legacy(
        cls,
        *,
        past_exo_cont: Optional[torch.Tensor] = None,
        past_exo_cat: Optional[torch.Tensor] = None,
        future_exo: Optional[torch.Tensor] = None,
        future_exo_cat: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> "ExogenousBatch":
        def normalize_future(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            future = _none_if_empty_feature_tensor(value)
            if torch.is_tensor(future) and future.ndim == 2 and batch_size is not None:
                future = future.unsqueeze(0).expand(int(batch_size), -1, -1)
            return future

        return cls(
            past_cont=_none_if_empty_feature_tensor(past_exo_cont),
            past_cat=_none_if_empty_feature_tensor(past_exo_cat),
            future_cont=normalize_future(future_exo),
            future_cat=normalize_future(future_exo_cat),
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.past_cont is None
            and self.past_cat is None
            and self.future_cont is None
            and self.future_cat is None
        )

    def provided_inputs(self) -> frozenset[str]:
        provided: set[str] = set()
        if self.past_cont is not None:
            provided.add("past_cont")
        if self.past_cat is not None:
            provided.add("past_cat")
        if self.future_cont is not None:
            provided.add("future_cont")
        if self.future_cat is not None:
            provided.add("future_cat")
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
            future_cat=move(self.future_cat),
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
            "future_cat": self.future_cat,
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

            if name in {"past_cat", "future_cat"}:
                if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                    raise TypeError(f"{name} must use an integer dtype, got {value.dtype}.")
            elif not torch.is_floating_point(value):
                raise TypeError(f"{name} must use a floating dtype, got {value.dtype}.")

            expected_time = horizon if name in {"future_cont", "future_cat"} else lookback
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
                "future_cat": len(schema.future_cat_names),
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

            categorical_cardinalities = {
                "past_cat": schema.past_cat_cardinalities,
                "future_cat": schema.future_cat_cardinalities,
            }
            for name, cardinalities in categorical_cardinalities.items():
                value = tensors[name]
                if value is None or not cardinalities or value.numel() == 0:
                    continue
                for feature_index, cardinality in enumerate(cardinalities):
                    feature_values = value[..., feature_index]
                    if bool(torch.any(feature_values < 0).item()):
                        raise ValueError(f"{name} category IDs must be non-negative.")
                    if bool(torch.any(feature_values >= cardinality).item()):
                        raise ValueError(
                            f"{name} category IDs exceed schema cardinality for "
                            f"feature index {feature_index}: expected values < {cardinality}."
                        )

        return self

    def as_legacy_kwargs(self) -> dict[str, Optional[torch.Tensor]]:
        return {
            "past_exo_cont": self.past_cont,
            "past_exo_cat": self.past_cat,
            "future_exo": self.future_cont,
            "future_exo_cat": self.future_cat,
        }
