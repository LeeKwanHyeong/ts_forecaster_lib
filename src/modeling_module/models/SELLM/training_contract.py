from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal


@dataclass(frozen=True)
class SELLMTrainerContract:
    """Shared optimization contract for SELLM qualification and refit runs."""

    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_scheduler: Literal["constant"] = "constant"
    t_max: int = 6
    use_amp: bool = False
    amp_dtype: Literal["fp32"] = "fp32"
    loss: Literal["mae"] = "mae"
    max_grad_norm: float = 30.0

    def trainer_kwargs(self) -> dict[str, object]:
        """Return fields accepted by the public ``TrainerConfig``."""

        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "t_max": self.t_max,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "max_grad_norm": self.max_grad_norm,
        }

    def as_metadata(self) -> dict[str, object]:
        return asdict(self)


SELLM_TRAINER_CONTRACT: Final = SELLMTrainerContract()


__all__ = ["SELLMTrainerContract", "SELLM_TRAINER_CONTRACT"]
