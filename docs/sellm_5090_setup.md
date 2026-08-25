# SELLM 5090 Setup

This repo keeps SELLM dependencies optional so the core forecasting package stays light.

## Create Environment

```bash
conda env create -f environment.5090-sellm.yml
conda activate ts_forecaster_5090_sellm
```

## Verify CUDA

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## SELLM LLM Sources

`use_pretrained_llm=True` enables a pretrained LLM backbone. Select where the
backbone is loaded from the source selected by `llm_source`. Use `False` only for lightweight CPU
smoke tests that should use the built-in Transformer fallback.

### On-Premise Local Model

```python
from modeling_module import ArchitectureConfig, SELLMArchitectureConfig, TrainRequest

request = TrainRequest(
    models=["sellm_base"],
    architecture=ArchitectureConfig(
        sellm=SELLMArchitectureConfig(
            architecture_variant="paper_v1",
            use_pretrained_llm=True,
            llm_source="local",
            llm_local_path="/models/Qwen2-0.5B",
            token_len=27,
            semantic_vocab_size=256,
            semantic_top_k=32,
            time_adapter_rank=8,
            time_adapter_layers=2,
        )
    ),
)
```

Local mode validates the model directory and loads it with
`local_files_only=True`, so SELLM does not fall back to a Hub request.

### Hugging Face Hub Model

```python
request = TrainRequest(
    models=["sellm_base"],
    architecture=ArchitectureConfig(
        sellm=SELLMArchitectureConfig(
            architecture_variant="paper_v1",
            use_pretrained_llm=True,
            llm_source="huggingface",
            llm_model_name="Qwen/Qwen2-0.5B",
            llm_revision="91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
        )
    ),
)
```

`SELLMConfig` defaults to the sealed Qwen2-0.5B revision shown above. Override
the revision only for an explicitly isolated research comparison; maintained
qualification and deployment artifacts must keep the pinned revision.

## Notes

- `SELLM` is a direct forecasting model family, not a POSTTIME-style forecast reviser.
- The maintained default LLM backbone is `Qwen/Qwen2-0.5B` at revision
  `91d2aff3f957f99e4c74c962f2f408dcc88a18d8`. Qwen2-1.5B is research-only.
- Select `architecture_variant="paper_v1"` for the paper-based endogenous model. The
  `legacy_v1` default exists only so checkpoints created before architecture versioning
  retain their original state-dict contract.
- `paper_v1` rejects future exogenous inputs. A separately versioned `sellm_exo` contract
  has not been introduced yet.
- `llm_source` is used only when `use_pretrained_llm=True`.
- The default LLM backbone is frozen; trainable parameters are the vocabulary projection,
  numeric encoder and decoder, TSCC, and Time-Adapter.
- The environment file prefers CUDA 13.x PyTorch wheels for RTX 5090-class servers.
- If the server driver is pinned to an older CUDA-compatible stack, update the PyTorch wheel index in `environment.5090-sellm.yml` before creating the env.
