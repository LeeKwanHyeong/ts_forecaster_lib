from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, ClassVar, Iterable, Literal, Mapping, Optional

import numpy as np

from modeling_module.data_loader.exogenous_contracts import ExogenousFeatureSchema


CATEGORICAL_UNK_ID = 0
CATEGORICAL_VOCABULARY_ARTIFACT_VERSION = 1
CATEGORICAL_VOCABULARY_ARTIFACT_TYPE = "modeling_module.categorical_vocabulary"

UnknownCategoryPolicy = Literal["use_unk", "error"]


def _normalize_feature_name(value: Any) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("Categorical vocabulary feature_name cannot be empty.")
    return name


def _category_record(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, Integral):
        return {"kind": "int", "value": str(int(value))}
    if isinstance(value, Real):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError("Categorical values cannot contain infinite floats.")
        if number == 0.0:
            number = 0.0
        return {"kind": "float", "value": number.hex()}
    raise TypeError(
        "Categorical values must be str, bool, int, finite float, date, datetime, "
        f"or None; got {type(value).__name__}."
    )


def _record_to_token(record: Mapping[str, Any]) -> str:
    if set(record) != {"kind", "value"}:
        raise ValueError("Categorical value records must contain exactly 'kind' and 'value'.")

    kind = record["kind"]
    value = record["value"]
    if not isinstance(kind, str):
        raise TypeError("Categorical value record kind must be a string.")

    try:
        if kind == "str":
            if not isinstance(value, str):
                raise TypeError("String categorical records require a string value.")
            decoded: Any = value
        elif kind == "bool":
            if not isinstance(value, bool):
                raise TypeError("Boolean categorical records require a boolean value.")
            decoded = value
        elif kind == "int":
            if not isinstance(value, str):
                raise TypeError("Integer categorical records require a decimal string value.")
            decoded = int(value)
        elif kind == "float":
            if not isinstance(value, str):
                raise TypeError("Float categorical records require a hexadecimal string value.")
            decoded = float.fromhex(value)
        elif kind == "date":
            if not isinstance(value, str):
                raise TypeError("Date categorical records require an ISO string value.")
            decoded = date.fromisoformat(value)
        elif kind == "datetime":
            if not isinstance(value, str):
                raise TypeError("Datetime categorical records require an ISO string value.")
            decoded = datetime.fromisoformat(value)
        else:
            raise ValueError(f"Unsupported categorical value kind: {kind!r}.")
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Invalid {kind!r} categorical value record.") from exc

    canonical_record = _category_record(decoded)
    if canonical_record is None or dict(record) != canonical_record:
        raise ValueError("Categorical value record is not in canonical form.")
    return json.dumps(
        canonical_record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _value_to_token(value: Any) -> Optional[str]:
    record = _category_record(value)
    if record is None:
        return None
    return _record_to_token(record)


def _token_to_value(token: str) -> Any:
    try:
        record = json.loads(token)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Categorical vocabulary contains an invalid value token.") from exc
    if not isinstance(record, Mapping):
        raise ValueError("Categorical vocabulary value token must decode to an object.")
    canonical_token = _record_to_token(record)
    if canonical_token != token:
        raise ValueError("Categorical vocabulary value token is not canonical.")

    kind = record["kind"]
    value = record["value"]
    if kind == "str":
        return value
    if kind == "bool":
        return value
    if kind == "int":
        return int(value)
    if kind == "float":
        return float.fromhex(value)
    if kind == "date":
        return date.fromisoformat(value)
    if kind == "datetime":
        return datetime.fromisoformat(value)
    raise AssertionError(f"Unhandled categorical value kind: {kind!r}.")


@dataclass(frozen=True, slots=True)
class CategoricalVocabulary:
    """Immutable category-ID mapping for one named feature."""

    feature_name: str
    value_tokens: tuple[str, ...] = ()
    _token_to_id: Mapping[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        feature_name = _normalize_feature_name(self.feature_name)
        tokens = tuple(str(token) for token in self.value_tokens)
        if len(set(tokens)) != len(tokens):
            raise ValueError(
                f"Categorical vocabulary {feature_name!r} contains duplicate values."
            )
        for token in tokens:
            _token_to_value(token)

        token_to_id = {
            token: index
            for index, token in enumerate(tokens, start=CATEGORICAL_UNK_ID + 1)
        }
        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "value_tokens", tokens)
        object.__setattr__(self, "_token_to_id", MappingProxyType(token_to_id))

    @classmethod
    def fit(
        cls,
        feature_name: str,
        values: Iterable[Any],
    ) -> "CategoricalVocabulary":
        tokens: set[str] = set()
        for value in values:
            token = _value_to_token(value)
            if token is not None:
                tokens.add(token)
        return cls(
            feature_name=_normalize_feature_name(feature_name),
            value_tokens=tuple(sorted(tokens)),
        )

    @property
    def unk_id(self) -> int:
        return CATEGORICAL_UNK_ID

    @property
    def known_size(self) -> int:
        return len(self.value_tokens)

    @property
    def cardinality(self) -> int:
        return self.known_size + 1

    @property
    def known_values(self) -> tuple[Any, ...]:
        return tuple(_token_to_value(token) for token in self.value_tokens)

    def id_of(
        self,
        value: Any,
        *,
        unknown_policy: UnknownCategoryPolicy = "use_unk",
    ) -> int:
        if unknown_policy not in {"use_unk", "error"}:
            raise ValueError("unknown_policy must be 'use_unk' or 'error'.")
        token = _value_to_token(value)
        category_id = self._token_to_id.get(token) if token is not None else None
        if category_id is not None:
            return category_id
        if unknown_policy == "use_unk":
            return CATEGORICAL_UNK_ID
        raise ValueError(
            f"Unknown categorical value for feature {self.feature_name!r}: {value!r}."
        )

    def encode(
        self,
        values: Iterable[Any],
        *,
        unknown_policy: UnknownCategoryPolicy = "use_unk",
    ) -> tuple[int, ...]:
        return tuple(
            self.id_of(value, unknown_policy=unknown_policy)
            for value in values
        )

    def map_series(
        self,
        values: Any,
        *,
        unknown_policy: UnknownCategoryPolicy = "use_unk",
    ) -> np.ndarray:
        """Encode a Polars-like series or iterable as a NumPy int64 array."""
        raw_values = values.to_list() if hasattr(values, "to_list") else values
        return np.asarray(
            self.encode(raw_values, unknown_policy=unknown_policy),
            dtype=np.int64,
        )

    def value_of(self, category_id: int) -> Any:
        if isinstance(category_id, bool) or not isinstance(category_id, Integral):
            raise TypeError("category_id must be an integer.")
        normalized_id = int(category_id)
        if normalized_id == CATEGORICAL_UNK_ID:
            return None
        token_index = normalized_id - 1
        if token_index < 0 or token_index >= len(self.value_tokens):
            raise ValueError(
                f"category_id {normalized_id} is outside [0, {self.cardinality - 1}] "
                f"for feature {self.feature_name!r}."
            )
        return _token_to_value(self.value_tokens[token_index])

    def decode(self, category_ids: Iterable[int]) -> tuple[Any, ...]:
        return tuple(self.value_of(category_id) for category_id in category_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "cardinality": self.cardinality,
            "values": [json.loads(token) for token in self.value_tokens],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CategoricalVocabulary":
        if not isinstance(payload, Mapping):
            raise TypeError("Categorical vocabulary payload must be a mapping.")
        expected_keys = {"feature_name", "cardinality", "values"}
        if set(payload) != expected_keys:
            raise ValueError(
                "Categorical vocabulary payload must contain exactly "
                "'feature_name', 'cardinality', and 'values'."
            )
        values = payload["values"]
        if not isinstance(values, list):
            raise TypeError("Categorical vocabulary values must be a list.")
        tokens = tuple(
            _record_to_token(record)
            if isinstance(record, Mapping)
            else _raise_invalid_record_type(record)
            for record in values
        )
        vocabulary = cls(
            feature_name=_normalize_feature_name(payload["feature_name"]),
            value_tokens=tokens,
        )
        cardinality = payload["cardinality"]
        if isinstance(cardinality, bool) or not isinstance(cardinality, Integral):
            raise TypeError("Categorical vocabulary cardinality must be an integer.")
        if int(cardinality) != vocabulary.cardinality:
            raise ValueError(
                f"Categorical vocabulary cardinality mismatch for "
                f"{vocabulary.feature_name!r}: {cardinality} != "
                f"{vocabulary.cardinality}."
            )
        return vocabulary


def _raise_invalid_record_type(value: Any) -> str:
    raise TypeError(
        "Categorical vocabulary values must contain mapping records, "
        f"got {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class CategoricalVocabularyArtifact:
    """Serializable bundle of ordered categorical feature vocabularies.

    Fit this artifact from the training partition only. Validation, test, and
    inference-only values must resolve through the reserved UNK ID.
    """

    ARTIFACT_TYPE: ClassVar[str] = CATEGORICAL_VOCABULARY_ARTIFACT_TYPE

    vocabularies: tuple[CategoricalVocabulary, ...] = ()
    version: int = CATEGORICAL_VOCABULARY_ARTIFACT_VERSION
    _by_name: Mapping[str, CategoricalVocabulary] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        version = int(self.version)
        if version != CATEGORICAL_VOCABULARY_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported categorical vocabulary artifact version={self.version}; "
                f"expected {CATEGORICAL_VOCABULARY_ARTIFACT_VERSION}."
            )
        vocabularies = tuple(self.vocabularies)
        if any(not isinstance(vocabulary, CategoricalVocabulary) for vocabulary in vocabularies):
            raise TypeError(
                "vocabularies must contain only CategoricalVocabulary instances."
            )
        names = tuple(vocabulary.feature_name for vocabulary in vocabularies)
        if len(set(names)) != len(names):
            raise ValueError(
                f"Categorical vocabulary artifact contains duplicate feature names: {names}."
            )

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "vocabularies", vocabularies)
        object.__setattr__(
            self,
            "_by_name",
            MappingProxyType(
                {
                    vocabulary.feature_name: vocabulary
                    for vocabulary in vocabularies
                }
            ),
        )

    @classmethod
    def fit(
        cls,
        columns: Mapping[str, Iterable[Any]],
        *,
        feature_names: Optional[Iterable[str]] = None,
    ) -> "CategoricalVocabularyArtifact":
        if not isinstance(columns, Mapping):
            raise TypeError("columns must be a mapping from feature name to values.")
        names = tuple(
            _normalize_feature_name(name)
            for name in (feature_names if feature_names is not None else columns.keys())
        )
        if len(set(names)) != len(names):
            raise ValueError(f"feature_names contains duplicates: {names}.")
        missing = tuple(name for name in names if name not in columns)
        if missing:
            raise ValueError(
                "Categorical vocabulary columns are missing required features: "
                + ", ".join(missing)
            )
        return cls(
            vocabularies=tuple(
                CategoricalVocabulary.fit(name, columns[name])
                for name in names
            )
        )

    @classmethod
    def fit_for_schema(
        cls,
        schema: ExogenousFeatureSchema,
        columns: Mapping[str, Iterable[Any]],
    ) -> "CategoricalVocabularyArtifact":
        if not isinstance(schema, ExogenousFeatureSchema):
            raise TypeError("schema must be an ExogenousFeatureSchema.")
        return cls.fit(columns, feature_names=schema.categorical_feature_names)

    @property
    def unk_id(self) -> int:
        return CATEGORICAL_UNK_ID

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(vocabulary.feature_name for vocabulary in self.vocabularies)

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(vocabulary.cardinality for vocabulary in self.vocabularies)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def vocabulary_for(self, feature_name: str) -> CategoricalVocabulary:
        normalized_name = _normalize_feature_name(feature_name)
        try:
            return self._by_name[normalized_name]
        except KeyError as exc:
            raise KeyError(
                f"Categorical vocabulary artifact does not contain "
                f"feature {normalized_name!r}."
            ) from exc

    def encode(
        self,
        feature_name: str,
        values: Iterable[Any],
        *,
        unknown_policy: UnknownCategoryPolicy = "use_unk",
    ) -> tuple[int, ...]:
        return self.vocabulary_for(feature_name).encode(
            values,
            unknown_policy=unknown_policy,
        )

    def bind_schema(
        self,
        schema: ExogenousFeatureSchema,
    ) -> ExogenousFeatureSchema:
        if not isinstance(schema, ExogenousFeatureSchema):
            raise TypeError("schema must be an ExogenousFeatureSchema.")
        expected_names = schema.categorical_feature_names
        if self.feature_names != expected_names:
            raise ValueError(
                "Categorical vocabulary feature order does not match schema: "
                f"{self.feature_names} != {expected_names}."
            )

        past_cardinalities = tuple(
            self.vocabulary_for(name).cardinality
            for name in schema.past_cat_names
        )
        future_cardinalities = tuple(
            self.vocabulary_for(name).cardinality
            for name in schema.future_cat_names
        )
        if (
            schema.past_cat_cardinalities
            and schema.past_cat_cardinalities != past_cardinalities
        ):
            raise ValueError(
                "Categorical vocabulary cardinalities do not match "
                "schema.past_cat_cardinalities."
            )
        if (
            schema.future_cat_cardinalities
            and schema.future_cat_cardinalities != future_cardinalities
        ):
            raise ValueError(
                "Categorical vocabulary cardinalities do not match "
                "schema.future_cat_cardinalities."
            )

        return ExogenousFeatureSchema(
            past_cont_names=schema.past_cont_names,
            past_cat_names=schema.past_cat_names,
            future_cont_names=schema.future_cont_names,
            past_cat_cardinalities=past_cardinalities,
            version=schema.version,
            future_cat_names=schema.future_cat_names,
            future_cat_cardinalities=future_cardinalities,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.ARTIFACT_TYPE,
            "version": self.version,
            "unk_id": CATEGORICAL_UNK_ID,
            "vocabularies": [
                vocabulary.to_dict()
                for vocabulary in self.vocabularies
            ],
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CategoricalVocabularyArtifact":
        if not isinstance(payload, Mapping):
            raise TypeError("Categorical vocabulary artifact payload must be a mapping.")
        expected_keys = {"artifact_type", "version", "unk_id", "vocabularies"}
        if set(payload) != expected_keys:
            raise ValueError(
                "Categorical vocabulary artifact payload has an invalid field set."
            )
        if payload["artifact_type"] != cls.ARTIFACT_TYPE:
            raise ValueError(
                f"Unsupported categorical vocabulary artifact type: "
                f"{payload['artifact_type']!r}."
            )
        version = payload["version"]
        if isinstance(version, bool) or not isinstance(version, Integral):
            raise TypeError("Categorical vocabulary artifact version must be an integer.")
        unk_id = payload["unk_id"]
        if (
            isinstance(unk_id, bool)
            or not isinstance(unk_id, Integral)
            or int(unk_id) != CATEGORICAL_UNK_ID
        ):
            raise ValueError(
                f"Categorical vocabulary artifact unk_id must be "
                f"{CATEGORICAL_UNK_ID}."
            )
        vocabularies = payload["vocabularies"]
        if not isinstance(vocabularies, list):
            raise TypeError("Categorical vocabulary artifact vocabularies must be a list.")
        return cls(
            vocabularies=tuple(
                CategoricalVocabulary.from_dict(vocabulary)
                for vocabulary in vocabularies
            ),
            version=int(version),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CategoricalVocabularyArtifact":
        if not isinstance(payload, str):
            raise TypeError("Categorical vocabulary artifact JSON payload must be a string.")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid categorical vocabulary artifact JSON.") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("Categorical vocabulary artifact JSON must contain an object.")
        return cls.from_dict(decoded)


__all__ = [
    "CATEGORICAL_UNK_ID",
    "CATEGORICAL_VOCABULARY_ARTIFACT_TYPE",
    "CATEGORICAL_VOCABULARY_ARTIFACT_VERSION",
    "CategoricalVocabulary",
    "CategoricalVocabularyArtifact",
    "UnknownCategoryPolicy",
]
