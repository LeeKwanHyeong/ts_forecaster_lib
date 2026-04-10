import importlib

import pytest

from modeling_module import train
from modeling_module.utils import device as device_utils


def _make_daily_df():
    import polars as pl

    rows = []
    for uid in ("A", "B"):
        for idx in range(1, 40):
            rows.append(
                {
                    "unique_id": uid,
                    "date": 20240100 + idx,
                    "y": float(idx),
                }
            )
    return pl.DataFrame(rows)


def test_select_default_device_falls_back_to_cpu_when_accelerators_fail(monkeypatch):
    def fake_probe(name):
        if str(name).startswith("cuda"):
            return False, "RuntimeError: no kernel image is available for execution on the device"
        if str(name).startswith("mps"):
            return False, "MPS is not available in this PyTorch environment."
        return True, None

    monkeypatch.setattr(device_utils, "probe_device", fake_probe)

    device, diagnostic = device_utils.select_default_device()

    assert device == "cpu"
    assert diagnostic is not None
    assert "cuda:" in diagnostic


def test_resolve_device_rejects_unusable_explicit_cuda(monkeypatch):
    monkeypatch.setattr(
        device_utils,
        "probe_device",
        lambda name: (False, "RuntimeError: no kernel image is available for execution on the device"),
    )

    with pytest.raises(RuntimeError, match="Requested device `cuda` is not usable"):
        device_utils.resolve_device("cuda")


def test_train_surfaces_explicit_device_probe_error(tmp_path, monkeypatch):
    train_module = importlib.import_module("modeling_module.api.train")

    monkeypatch.setattr(
        train_module,
        "resolve_device",
        lambda _: (_ for _ in ()).throw(
            RuntimeError(
                "Requested device `cuda` is not usable in this environment. "
                "RuntimeError: no kernel image is available for execution on the device"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Requested device `cuda` is not usable"):
        train(
            {
                "data": {
                    "df": _make_daily_df(),
                    "lookback": 14,
                    "horizon": 2,
                    "freq": "daily",
                    "batch_size": 2,
                },
                "models": ["patchtst_base"],
                "trainer": {"epochs": 1, "lr": 1e-3},
                "device": "cuda",
                "save_dir": str(tmp_path),
                "auto_save_dir": False,
            }
        )
