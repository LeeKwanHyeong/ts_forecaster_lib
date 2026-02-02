import json
from dataclasses import asdict, is_dataclass
import os

def _json_safe(obj):
    """
    JSON 직렬화가 불가능한 객체(Torch Device, Tensor, Path 등)를 문자열이나 기본 타입으로 변환.
    설정(Config) 로깅 시 오류 방지 목적.
    """
    # torch.device 처리
    try:
        import torch
        if isinstance(obj, torch.device):
            return str(obj)
    except Exception:
        pass

    # numpy / torch scalar 처리
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass

    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    # pathlib.Path 등 처리
    try:
        import os
        if isinstance(obj, os.PathLike):
            return str(obj)
    except Exception:
        pass

    # set/tuple -> list 변환
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # 기본 문자열 변환
    return str(obj)


def dump_cfg(cfg, name: str, cfg_name: str = None, save_dir: str = None,) -> None:
    """학습 설정(Config) 내용 출력."""
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print(f"[train_{name}] Effective TrainingConfig:")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok = True)
        path = os.path.join(save_dir, cfg_name)

        payload = asdict(cfg) if is_dataclass(cfg) else {}
        if hasattr(cfg, "__dict__"):
            payload.update(cfg.__dict__)
        with open(path, "w", encoding = 'utf-8') as f:
            json.dump(payload, f, ensure_ascii = False, indent = 2, default = _json_safe)