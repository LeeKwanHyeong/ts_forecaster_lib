from __future__ import annotations

import sys
from types import ModuleType

import pytest

from modeling_module.models.SELLM.SELLM import SELLMModel
from modeling_module.models.SELLM.configs import (
    DEFAULT_SELLM_LLM_MODEL_NAME,
    DEFAULT_SELLM_LLM_REVISION,
    SELLMConfig,
)


class _FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []
    result = object()

    @classmethod
    def from_pretrained(cls, target: str, **kwargs):
        cls.calls.append((target, kwargs))
        return cls.result


def _install_fake_transformers(monkeypatch) -> None:
    module = ModuleType("transformers")
    setattr(module, "AutoModel", _FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", module)
    _FakeAutoModel.calls = []


def test_sellm_default_backbone_is_sealed_qwen2_0_5b():
    cfg = SELLMConfig()

    assert cfg.llm_model_name == DEFAULT_SELLM_LLM_MODEL_NAME
    assert cfg.llm_model_name == "Qwen/Qwen2-0.5B"
    assert cfg.llm_revision == DEFAULT_SELLM_LLM_REVISION
    assert cfg.llm_revision == "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"


def test_load_llm_uses_hugging_face_model_and_revision(monkeypatch):
    _install_fake_transformers(monkeypatch)
    cfg = SELLMConfig(
        llm_source="huggingface",
        llm_model_name="Qwen/Qwen2-0.5B",
        llm_revision="fixed-revision",
    )

    loaded = SELLMModel._load_llm(cfg)

    assert loaded is _FakeAutoModel.result
    assert _FakeAutoModel.calls == [
        ("Qwen/Qwen2-0.5B", {"revision": "fixed-revision"})
    ]


def test_load_llm_uses_local_directory_without_hub_access(monkeypatch, tmp_path):
    _install_fake_transformers(monkeypatch)
    model_dir = tmp_path / "Qwen2-0.5B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    cfg = SELLMConfig(llm_source="local", llm_local_path=str(model_dir))

    loaded = SELLMModel._load_llm(cfg)

    assert loaded is _FakeAutoModel.result
    assert _FakeAutoModel.calls == [
        (str(model_dir), {"local_files_only": True})
    ]


def test_load_llm_requires_local_path():
    cfg = SELLMConfig(llm_source="local", llm_local_path=None)

    with pytest.raises(ValueError, match="llm_local_path is required"):
        SELLMModel._load_llm(cfg)


def test_load_llm_rejects_unknown_source():
    cfg = SELLMConfig()
    object.__setattr__(cfg, "llm_source", "ollama")

    with pytest.raises(ValueError, match="Unsupported llm_source"):
        SELLMModel._load_llm(cfg)
