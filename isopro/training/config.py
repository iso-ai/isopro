"""TrainingConfig — configurable parameters for GRPOTrainer and DataCollector.

Default values are tuned for a 7B-parameter instruction-tuned model on the
ISOPro task environments. Override only what you need — the defaults should
work well out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TrainingConfig:
    """Full configuration surface for GRPOTrainer.

    Most users need only n_episodes and save_path. The remaining fields
    are for practitioners who want to control the training dynamics.

    Attributes:
        n_episodes: Total training episodes.
        group_size: Number of responses sampled per prompt (G in GRPO).
            Higher values give more stable advantage estimates but cost more.
        learning_rate: AdamW learning rate.
        kl_coef: KL penalty coefficient against the reference (frozen) model.
            Higher values keep the policy closer to the reference.
        max_grad_norm: Gradient clipping threshold.
        save_every: Save a checkpoint every N episodes.
        log_every: Log metrics every N episodes.
        warmup_episodes: Linear LR warmup over this many episodes.
        reward_shaping: Optional callable applied to raw rewards before
            computing advantages. E.g. ``lambda r: np.clip(r, -1, 1)``.
        curriculum: Optional CurriculumScheduler. If provided, task difficulty
            is managed automatically. If None, the environment's default
            difficulty is used throughout.
        max_new_tokens: Token budget for each generated response during training.
        temperature: Sampling temperature during training rollouts.
        seed: Master random seed for reproducibility.
    """

    n_episodes: int = 1000
    group_size: int = 8
    learning_rate: float = 1e-5
    kl_coef: float = 0.01
    max_grad_norm: float = 1.0
    save_every: int = 100
    log_every: int = 10
    warmup_episodes: int = 50
    reward_shaping: Optional[Callable[[float], float]] = None
    curriculum: Optional[object] = None   # CurriculumScheduler | None
    max_new_tokens: int = 512
    temperature: float = 0.9
    seed: Optional[int] = None

    def shape_reward(self, reward: float) -> float:
        """Apply reward_shaping if provided, otherwise return reward unchanged.

        Args:
            reward: Raw scalar reward from the environment scorer.

        Returns:
            Shaped reward float.
        """
        if self.reward_shaping is not None:
            return float(self.reward_shaping(reward))
        return reward
