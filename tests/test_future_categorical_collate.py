from __future__ import annotations

import numpy as np
import pytest
import torch

from modeling_module.data_loader.future_scenario_store import (
    TrainCollateWithFutureExo,
)


def _sample(
    uid: str,
    *,
    start_idx: int = 10,
    future_cont: torch.Tensor | None = None,
    future_cat: torch.Tensor | None = None,
):
    sample = (
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([4.0, 5.0]),
        uid,
        start_idx if future_cont is None else future_cont,
        torch.tensor([[0.1], [0.2], [0.3]]),
        torch.tensor([[1], [2], [1]], dtype=torch.long),
    )
    if future_cat is not None:
        return (*sample, future_cat)
    return sample


def test_collate_preserves_legacy_six_tuple_contract():
    collate = TrainCollateWithFutureExo(horizon=2)

    batch = collate(
        [
            _sample("A", start_idx=10),
            _sample("B", start_idx=20),
        ]
    )

    assert len(batch) == 6
    assert batch[0].shape == (2, 3, 1)
    assert batch[1].shape == (2, 2)
    assert batch[2] == ["A", "B"]
    assert batch[3].shape == (2, 2, 0)
    assert batch[4].shape == (2, 3, 1)
    assert batch[5].dtype == torch.long
    assert batch[5].shape == (2, 3, 1)


def test_collate_stacks_future_category_with_direct_future_continuous_payload():
    collate = TrainCollateWithFutureExo(horizon=2)
    future_cont_a = torch.tensor([[0.1], [0.2]])
    future_cont_b = torch.tensor([[0.3], [0.4]])
    future_cat_a = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    future_cat_b = torch.tensor([[3, 2], [1, 0]], dtype=torch.long)

    batch = collate(
        [
            _sample(
                "A",
                future_cont=future_cont_a,
                future_cat=future_cat_a,
            ),
            _sample(
                "B",
                future_cont=future_cont_b,
                future_cat=future_cat_b,
            ),
        ]
    )

    assert len(batch) == 7
    assert batch[3].dtype == torch.float32
    assert batch[3].shape == (2, 2, 1)
    assert batch[6].dtype == torch.long
    assert batch[6].shape == (2, 2, 2)
    torch.testing.assert_close(batch[6][0], future_cat_a)
    torch.testing.assert_close(batch[6][1], future_cat_b)


def test_collate_preserves_future_category_with_callback_future_continuous():
    callback_calls: list[tuple[list[int], int, str]] = []

    def future_callback(start_idxs, horizon, *, device):
        normalized = [int(value) for value in start_idxs]
        callback_calls.append((normalized, int(horizon), str(device)))
        return np.asarray(
            [
                [[float(start_idx)], [float(start_idx + 1)]]
                for start_idx in normalized
            ],
            dtype=np.float32,
        )

    collate = TrainCollateWithFutureExo(
        horizon=2,
        future_exo_cb=future_callback,
    )
    future_cat_a = torch.tensor([[1], [2]], dtype=torch.long)
    future_cat_b = torch.tensor([[2], [0]], dtype=torch.long)

    batch = collate(
        [
            _sample("A", start_idx=10, future_cat=future_cat_a),
            _sample("B", start_idx=20, future_cat=future_cat_b),
        ]
    )

    assert callback_calls == [([10, 20], 2, "cpu")]
    assert len(batch) == 7
    assert batch[3].shape == (2, 2, 1)
    assert batch[3][:, 0, 0].tolist() == [10.0, 20.0]
    assert batch[6].shape == (2, 2, 1)
    assert batch[6][:, 1, 0].tolist() == [2, 0]


def test_collate_rejects_mixed_six_and_seven_tuple_samples():
    collate = TrainCollateWithFutureExo(horizon=2)

    with pytest.raises(ValueError, match="same tuple length"):
        collate(
            [
                _sample("A"),
                _sample(
                    "B",
                    future_cat=torch.tensor([[1], [2]], dtype=torch.long),
                ),
            ]
        )


def test_collate_rejects_future_category_with_wrong_horizon():
    collate = TrainCollateWithFutureExo(horizon=2)
    invalid = torch.tensor([[1], [2], [3]], dtype=torch.long)

    with pytest.raises(ValueError, match="future_cat payload must be shaped"):
        collate(
            [
                _sample("A", future_cat=invalid),
                _sample("B", future_cat=invalid),
            ]
        )
