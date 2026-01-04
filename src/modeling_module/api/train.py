from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

# TODO: 실제 Trainer/Config 경로에 맞게 수정
# from modeling_module.training.trainer import Trainer
# from modeling_module.training.config import TrainingConfig

@dataclass
class TrainRequest:
    # 최소 입력만 먼저 정의하고, 점진적으로 확장
    config_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

def train(req: TrainRequest) -> Any:
    """
    Public training entrypoint.
    - 외부 프로젝트에서는 modeling_module.train(...)만 호출하게 만드는 것이 목표
    """
    # TODO: 아래는 뼈대. 현재 코드의 진짜 엔트리포인트에 맞게 연결
    # cfg = TrainingConfig.from_yaml(req.config_path) if req.config_path else TrainingConfig(**req.config)
    # trainer = Trainer(cfg)
    # return trainer.fit()
    raise NotImplementedError("Wire this to your actual Trainer/TrainingConfig.")