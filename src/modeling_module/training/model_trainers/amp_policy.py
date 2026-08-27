from modeling_module.training.config import TrainingConfig
import torch


def amp_type_set(config: TrainingConfig):
    amp_device = getattr(config, 'amp_device', 'cuda')
    amp_enabled = bool(
        getattr(config, 'use_amp', False)
        and amp_device == 'cuda'
        and torch.cuda.is_available()
    )
    amp_dtype_str = getattr(config, 'amp_dtype', 'bf16')

    if isinstance(amp_dtype_str, torch.dtype):
        amp_dtype = amp_dtype_str
    else:
        s = str(amp_dtype_str).lower()
        if s in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        elif s in ("fp16", "float16", "half"):
            amp_dtype = torch.float16
        elif s in ("fp32", "float32"):
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.bfloat16

    return amp_device, amp_enabled, amp_dtype
