from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ObjectiveConfig:
    objective: Literal["grpo", "gdpo", "cw_grpo"] = "grpo"
    clip_low: float = 0.2
    clip_high: float = 0.2
    eps: float = 1e-8
    reward_std_threshold: float = 1e-6

    def validate(self) -> None:
        if self.clip_low < 0 or self.clip_high < 0:
            raise ValueError("clip_low and clip_high must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.reward_std_threshold < 0:
            raise ValueError("reward_std_threshold must be non-negative")
