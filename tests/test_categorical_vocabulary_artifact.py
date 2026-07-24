from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone

import numpy as np
import pytest

from modeling_module import (
    CATEGORICAL_UNK_ID,
    CategoricalVocabulary,
    CategoricalVocabularyArtifact,
    ExogenousFeatureSchema,
)
from modeling_module.api import (
    CategoricalVocabularyArtifact as ApiCategoricalVocabularyArtifact,
)


def test_categorical_vocabulary_reserves_unk_and_assigns_deterministic_ids():
    assert ApiCategoricalVocabularyArtifact is CategoricalVocabularyArtifact

    first = CategoricalVocabulary.fit(
        "segment",
        ["west", "east", None, "west", float("nan")],
    )
    reordered = CategoricalVocabulary.fit(
        "segment",
        ["east", "west", "east"],
    )

    assert CATEGORICAL_UNK_ID == 0
    assert first.unk_id == 0
    assert first.known_values == ("east", "west")
    assert first.cardinality == 3
    assert first.to_dict() == reordered.to_dict()
    assert first.encode(["east", "west", "new", None]) == (1, 2, 0, 0)
    assert first.decode([1, 2, 0]) == ("east", "west", None)

    with pytest.raises(ValueError, match="Unknown categorical value"):
        first.id_of("new", unknown_policy="error")

    with pytest.raises(ValueError, match="outside"):
        first.value_of(3)


def test_categorical_vocabulary_preserves_supported_scalar_types_without_collisions():
    values = [
        1,
        "1",
        True,
        1.0,
        np.int64(2),
        np.float32(2.5),
        np.bool_(False),
        date(2024, 1, 1),
        datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc),
    ]
    vocabulary = CategoricalVocabulary.fit("typed", values)
    category_ids = vocabulary.encode(values)

    assert len(set(category_ids)) == len(values)
    assert vocabulary.decode(category_ids) == tuple(values)
    assert vocabulary.cardinality == len(values) + 1

    with pytest.raises(ValueError, match="infinite"):
        CategoricalVocabulary.fit("typed", [float("inf")])

    with pytest.raises(TypeError, match="must be str"):
        CategoricalVocabulary.fit("typed", [object()])


def test_vocabulary_artifact_binds_shared_past_future_schema_cardinalities():
    schema = ExogenousFeatureSchema.from_columns(
        past_cat=["segment", "store"],
        future_cat=["segment", "holiday"],
    )
    artifact = CategoricalVocabularyArtifact.fit_for_schema(
        schema,
        {
            "segment": ["B", "A", "B"],
            "store": [10, 20, 10],
            "holiday": [False, True],
        },
    )
    bound = artifact.bind_schema(schema)

    assert schema.categorical_feature_names == ("segment", "store", "holiday")
    assert artifact.feature_names == ("segment", "store", "holiday")
    assert artifact.cardinalities == (3, 3, 3)
    assert bound.past_cat_cardinalities == (3, 3)
    assert bound.future_cat_cardinalities == (3, 3)
    assert artifact.encode("segment", ["A", "unseen"]) == (1, 0)

    with pytest.raises(ValueError, match="missing required features"):
        CategoricalVocabularyArtifact.fit_for_schema(
            schema,
            {
                "segment": ["A"],
                "store": [10],
            },
        )


def test_vocabulary_artifact_rejects_schema_order_and_cardinality_mismatch():
    schema = ExogenousFeatureSchema.from_columns(
        past_cat=["segment", "store"],
        future_cat=["segment"],
    )
    reordered = CategoricalVocabularyArtifact.fit(
        {
            "segment": ["A", "B"],
            "store": [10, 20],
        },
        feature_names=["store", "segment"],
    )

    with pytest.raises(ValueError, match="feature order does not match"):
        reordered.bind_schema(schema)

    incompatible_schema = ExogenousFeatureSchema.from_columns(
        past_cat=["segment", "store"],
        future_cat=["segment"],
        past_cat_cardinalities=[4, 3],
        future_cat_cardinalities=[4],
    )
    artifact = CategoricalVocabularyArtifact.fit_for_schema(
        schema,
        {
            "segment": ["A", "B"],
            "store": [10, 20],
        },
    )

    with pytest.raises(ValueError, match="schema.past_cat_cardinalities"):
        artifact.bind_schema(incompatible_schema)


def test_vocabulary_artifact_json_roundtrip_and_fingerprint_are_stable():
    artifact = CategoricalVocabularyArtifact.fit(
        {"segment": ["B", "A", None]},
        feature_names=["segment"],
    )
    same = CategoricalVocabularyArtifact.fit(
        {"segment": ["A", "B", "A"]},
        feature_names=["segment"],
    )
    restored = CategoricalVocabularyArtifact.from_json(artifact.to_json())

    assert restored == artifact
    assert artifact.fingerprint == same.fingerprint
    assert artifact.fingerprint == restored.fingerprint
    assert artifact.fingerprint == "39fc7d86ddc1fb6d1a77ed3f5c1e681cedaff10f5d772ead6ed796478a08db6e"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["vocabularies"][0].update({"cardinality": 99}),
            "cardinality mismatch",
        ),
        (
            lambda payload: payload.update({"unk_id": 1}),
            "unk_id must be 0",
        ),
        (
            lambda payload: payload.update({"artifact_type": "other"}),
            "Unsupported categorical vocabulary artifact type",
        ),
        (
            lambda payload: payload.update({"version": 99}),
            "Unsupported categorical vocabulary artifact version",
        ),
    ],
)
def test_vocabulary_artifact_rejects_tampered_payload(mutate, match):
    artifact = CategoricalVocabularyArtifact.fit(
        {"segment": ["A", "B"]},
        feature_names=["segment"],
    )
    payload = deepcopy(artifact.to_dict())
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        CategoricalVocabularyArtifact.from_dict(payload)
