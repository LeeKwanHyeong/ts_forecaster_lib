from __future__ import annotations
from typing import Any, Callable, Optional

def load_predictor(ckpt_path: str) -> Callable[[Any], Any]:
    """
    ckpt -> callable predictor 로 표준화.
    """
    # TODO: 실제 체크포인트 로더로 연결
    # model = load_model(ckpt_path)
    # model.eval()
    # return lambda batch: model(batch)
    raise NotImplementedError("Implement checkpoint loading and inference callable.")

def predict(ckpt_path: str, batch: Any) -> Any:
    predictor = load_predictor(ckpt_path)
    return predictor(batch)