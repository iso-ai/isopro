"""True base environment for all ISOPro simulation environments.

Replaces SimulationEnvironment as the canonical base. Adds:
  - ModelBackend integration (any model or agent, via the backends package)
  - Standardized EpisodeResult / StepResult dataclasses
  - run_episode() orchestrator so environments don't each re-implement the loop
  - compute_metrics() contract that all environments must fulfill

All domain-specific environments (ConversationEnvironment, LLMRLEnvironment,
CarRLEnvironment, etc.) inherit from BaseEnvironment, not SimulationEnvironment.
SimulationEnvironment is retained for backwards compatibility with legacy code.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Output of a single environment step.

    Attributes:
        step: 0-indexed step number within the episode.
        observation: The observation returned by env.step().
        reward: Scalar reward for this step.
        terminated: True if the episode ended naturally (goal or failure).
        truncated: True if the episode ended due to step limit.
        text_output: The model's raw text response this step, if any.
        info: Arbitrary info dict from the environment.
        latency_ms: Wall-clock time for this step in milliseconds.
    """

    step: int
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    text_output: Optional[str] = None
    info: dict = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class EpisodeResult:
    """Complete output of a single run_episode() call.

    Attributes:
        episode_id: Short unique identifier for this run.
        environment_name: Class name of the environment.
        model_id: model_id from ModelInfo, or "none" if no backend.
        provider: Backend provider string.
        total_steps: Number of steps executed.
        total_reward: Sum of all step rewards.
        mean_reward: Average reward per step.
        terminated: Whether the episode ended naturally.
        duration_s: Wall-clock seconds for the full episode.
        step_results: Ordered list of per-step results.
        metrics: Environment-specific computed metrics dict.
    """

    episode_id: str
    environment_name: str
    model_id: str
    provider: str
    total_steps: int
    total_reward: float
    mean_reward: float
    terminated: bool
    duration_s: float
    step_results: list[StepResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Return a compact summary dict for logging or streaming.

        Returns:
            Dict with key episode stats, excluding per-step data.
        """
        return {
            "episode_id": self.episode_id,
            "environment": self.environment_name,
            "model": self.model_id,
            "provider": self.provider,
            "steps": self.total_steps,
            "total_reward": round(self.total_reward, 4),
            "mean_reward": round(self.mean_reward, 4),
            "terminated": self.terminated,
            "duration_s": round(self.duration_s, 2),
            **self.metrics,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEnvironment(ABC):
    """Abstract base for all ISOPro environments.

    Subclasses must implement:
      - reset() → (observation, info)
      - step(action) → (observation, reward, terminated, truncated, info)
      - _get_action(observation) → action
      - compute_metrics() → dict

    Args:
        backend: Any ModelBackend instance. Can be None for environments
            that do not need a language model (e.g. pure physics sims).
    """

    def __init__(self, backend=None) -> None:
        # Import here to avoid circular imports at module level.
        from ..backends.base import ModelBackend

        if backend is not None and not isinstance(backend, ModelBackend):
            raise TypeError(
                f"backend must be a ModelBackend instance, got {type(backend).__name__}. "
                "Use ModelBackend.from_huggingface(), .from_api(), etc."
            )

        self.backend = backend
        self._step_results: list[StepResult] = []
        self._episode_id: str = ""
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Core interface — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> tuple:
        """Reset the environment to its initial state.

        Returns:
            Tuple of (initial_observation, info_dict).
        """

    @abstractmethod
    def step(self, action) -> tuple:
        """Execute one step of the environment.

        Args:
            action: The action to take, in the environment's action format.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """

    @abstractmethod
    def _get_action(self, observation, config=None):
        """Determine the next action given the current observation.

        For LLM-backed environments, this typically calls self.backend.generate().
        For non-LLM environments, this implements the policy directly.

        Args:
            observation: Current environment observation.
            config: Optional GenerationConfig for the backend call.

        Returns:
            An action compatible with this environment's action space.
        """

    @abstractmethod
    def compute_metrics(self) -> dict:
        """Compute environment-specific summary metrics for the current episode.

        Called at the end of run_episode(). Should aggregate self._step_results
        into meaningful statistics (e.g. BLEU, success rate, avg speed).

        Returns:
            Dict of metric_name → scalar value.
        """

    # ------------------------------------------------------------------
    # Orchestrator — environments inherit this, don't override
    # ------------------------------------------------------------------

    def run_episode(
        self,
        max_steps: int = 100,
        config=None,
    ) -> EpisodeResult:
        """Run a complete episode from reset to termination.

        Calls reset(), then loops step() until terminated, truncated,
        or max_steps is reached. Collects StepResult for each step.

        Args:
            max_steps: Maximum steps before truncating the episode.
            config: GenerationConfig forwarded to _get_action() and
                the backend on each step.

        Returns:
            EpisodeResult with full step history and computed metrics.
        """
        self._episode_id = uuid.uuid4().hex[:8]
        self._step_results = []
        episode_start = time.perf_counter()

        self.logger.info(
            "Starting episode %s on %s", self._episode_id, self.__class__.__name__
        )

        observation, _ = self.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        for step_num in range(max_steps):
            step_start = time.perf_counter()

            action = self._get_action(observation, config)
            observation, reward, terminated, truncated, info = self.step(action)

            step_latency = (time.perf_counter() - step_start) * 1000.0
            total_reward += reward

            step_result = StepResult(
                step=step_num,
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                text_output=info.pop("text_output", None),
                info=info,
                latency_ms=step_latency,
            )
            self._step_results.append(step_result)

            if terminated or truncated:
                break

        duration_s = time.perf_counter() - episode_start
        n_steps = len(self._step_results)
        metrics = self.compute_metrics()

        model_info = self.backend.get_model_info() if self.backend else None
        model_id = model_info.model_id if model_info else "none"
        provider = model_info.provider if model_info else "none"

        result = EpisodeResult(
            episode_id=self._episode_id,
            environment_name=self.__class__.__name__,
            model_id=model_id,
            provider=provider,
            total_steps=n_steps,
            total_reward=total_reward,
            mean_reward=total_reward / n_steps if n_steps > 0 else 0.0,
            terminated=terminated,
            duration_s=duration_s,
            step_results=self._step_results,
            metrics=metrics,
        )

        self.logger.info(
            "Episode %s complete: %d steps, reward=%.3f, duration=%.1fs",
            self._episode_id,
            n_steps,
            total_reward,
            duration_s,
        )
        return result
