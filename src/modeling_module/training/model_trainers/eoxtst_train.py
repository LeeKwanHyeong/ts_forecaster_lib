import json
from dataclasses import asdict, is_dataclass


def _dump_cfg(cfg):
    """학습 설정(Config) 내용 출력"""
    data = asdict(cfg) if is_dataclass(cfg) else cfg.__dict__
    print("[train_exotst] Effective TrainingConfig:")
    print(json.dumps(data, indent = 2, ensure_ascii = False, default = str))


def _infer_exo_dim_from_cb(future_exo_cb, horizon: int, device: str = 'cpu') -> int:
    '''
    콜백 함수 실행을 통한 미래 외생 변수 차원(E) 추론.
    '''