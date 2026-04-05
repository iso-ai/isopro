"""
Reinforcement Learning module for the isopro package.
"""

from .rl_environment import BaseRLEnvironment, GymRLEnvironment, LLMRLEnvironment
from .rl_utils import calculate_discounted_rewards, update_q_table

# stable_baselines3 and its transitive deps (tensorboard/protobuf) can fail
# in some environments. Guard the import so the rest of the RL module remains
# usable even when SB3 is unavailable or broken.
try:
    from .rl_agent import RLAgent
    from .llm_cartpole_wrapper import LLMCartPoleWrapper
except Exception:  # noqa: BLE001
    RLAgent = None  # type: ignore[assignment,misc]
    LLMCartPoleWrapper = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseRLEnvironment",
    "LLMRLEnvironment",
    "GymRLEnvironment",
    "LLMCartPoleWrapper",
    "RLAgent",
    "calculate_discounted_rewards",
    "update_q_table",
]