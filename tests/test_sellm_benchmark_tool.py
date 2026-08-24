from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "benchmark_sellm_token_boundary_5090.py"
)
_SPEC = importlib.util.spec_from_file_location("sellm_benchmark_tool", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_train = _MODULE._train


class _TinyForecastModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * value[:, :1, :]

    def reg_loss(self) -> None:
        return None


def _batch(value: float) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    return (
        torch.full((2, 1, 1), value),
        torch.full((2, 1), value),
        ["a", "b"],
    )


def test_train_stops_at_exact_optimizer_update_budget(tmp_path):
    reports, _best_state, _best_epoch = _train(
        model=_TinyForecastModel(),
        train_loader=[_batch(1.0), _batch(2.0)],
        val_loader=[_batch(1.0)],
        device=torch.device("cpu"),
        epochs=4,
        learning_rate=1e-3,
        max_optimizer_updates=3,
        progress_path=tmp_path / "progress.json",
        run_contract={"token_len": 13, "seed": 42},
    )

    assert len(reports) == 2
    assert reports[0]["epoch_optimizer_updates"] == 2
    assert reports[0]["total_optimizer_updates"] == 2
    assert reports[1]["epoch_optimizer_updates"] == 1
    assert reports[1]["total_optimizer_updates"] == 3
