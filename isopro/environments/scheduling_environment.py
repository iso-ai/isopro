"""Scheduling environment for IsoZero training and evaluation.

Wraps the scheduling problem generator and verifier into a BaseEnvironment
that integrates with ISOPro's training infrastructure (GRPOTrainer,
CurriculumScheduler, DataCollector).

Usage:
    from isopro.environments.scheduling_environment import SchedulingEnvironment
    from isopro.environments.tasks.scheduling_tasks import SchedulingTier

    env = SchedulingEnvironment(tier=SchedulingTier.SEQUENCING, seed=42)
    task = env.generate_task()
    score = env.score(task, model_response)
"""

from __future__ import annotations

import logging
import random

from .base_env import BaseEnvironment
from .tasks.base_task import DifficultyLevel, Task
from .tasks.scheduling_tasks import (
    PAIRWISE_SUBTYPES,
    SchedulingTier,
    generate_scheduling_task,
)
from .tasks.scheduling_verifier import score_scheduling_task

logger = logging.getLogger(__name__)


class SchedulingEnvironment(BaseEnvironment):
    """Environment for scheduling problems at controlled difficulty tiers.

    Implements the oracle interface (generate_task / score) required by
    GRPOTrainer and the IsoZero training loop.

    Attributes:
        tier: Current difficulty tier.
        seed: Base seed for reproducible generation.
        _task_counter: Monotonic counter to derive per-task seeds.
    """

    def __init__(
        self,
        tier: SchedulingTier = SchedulingTier.SEQUENCING,
        seed: int = 42,
        backend=None,
    ) -> None:
        """Initialize the scheduling environment.

        Args:
            tier: Starting difficulty tier.
            seed: Base seed for reproducible problem generation.
            backend: Optional ModelBackend (not required for generate/score).
        """
        super().__init__(backend=backend)
        self.tier = tier
        self.seed = seed
        self._task_counter = 0
        self._rng = random.Random(seed)
        self._eval_records: list[dict] = []

    @property
    def difficulty(self) -> DifficultyLevel:
        """Current difficulty level (mapped from tier)."""
        return self.tier.difficulty_level

    def set_tier(self, tier: SchedulingTier) -> None:
        """Change the active tier.

        Args:
            tier: New difficulty tier.
        """
        self.tier = tier

    def generate_task(self, difficulty: DifficultyLevel | None = None) -> Task:
        """Generate a scheduling task at the current or specified tier.

        If difficulty is provided, maps it to the closest tier:
            EASY -> SEQUENCING, MEDIUM -> DEADLINE_PRESSURE, HARD -> FULL_COMPOSITION.

        Args:
            difficulty: Optional DifficultyLevel override.

        Returns:
            A Task with NL prompt and ground-truth solution.

        Raises:
            RuntimeError: If problem generation fails after retries.
        """
        tier = self.tier
        if difficulty is not None:
            tier = {
                DifficultyLevel.EASY: SchedulingTier.SEQUENCING,
                DifficultyLevel.MEDIUM: SchedulingTier.DEADLINE_PRESSURE,
                DifficultyLevel.HARD: SchedulingTier.FULL_COMPOSITION,
            }[difficulty]

        self._task_counter += 1
        task_seed = self.seed + self._task_counter * 97

        # For pairwise, pick a random subtype
        subtype = None
        if tier == SchedulingTier.PAIRWISE:
            subtype = self._rng.choice(PAIRWISE_SUBTYPES)

        task = generate_scheduling_task(tier, task_seed, subtype=subtype)
        if task is None:
            raise RuntimeError(
                f"Failed to generate feasible scheduling problem "
                f"for tier={tier.value}, seed={task_seed}"
            )
        return task

    def score(self, task: Task, response: str) -> float:
        """Score a model response. Binary: 1.0 (pass) or 0.0 (fail).

        Uses the deterministic verifier — no LLM calls, no partial credit.

        Args:
            task: The scheduling Task.
            response: The model's raw text response.

        Returns:
            1.0 if all constraints satisfied, 0.0 otherwise.
        """
        reward, detail = score_scheduling_task(task, response)
        self._eval_records.append(detail)
        return reward

    # ------------------------------------------------------------------
    # BaseEnvironment abstract methods (minimal — this env is task-based)
    # ------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset environment state."""
        self._eval_records = []
        return {}, {}

    def step(self, action) -> tuple:
        """Single step (not used in task-based mode)."""
        return {}, 0.0, True, False, {}

    def _get_action(self, observation, config=None):
        """Get action from backend (not used in task-based mode)."""
        if self.backend is None:
            return ""
        result = self.backend.generate(str(observation), config)
        return result.text

    def compute_metrics(self) -> dict:
        """Compute aggregate metrics from accumulated eval records."""
        if not self._eval_records:
            return {}

        n = len(self._eval_records)
        n_passed = sum(1 for r in self._eval_records if r.get("passed"))
        accuracy = n_passed / n

        # Per-check breakdown
        check_names = ["parsing", "validity", "precedence", "resource_capacity", "deadlines"]
        check_rates = {}
        for check in check_names:
            vals = [
                r.get("checks", {}).get(check, False)
                for r in self._eval_records
            ]
            check_rates[f"{check}_rate"] = sum(vals) / len(vals) if vals else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "n_evaluated": n,
            "n_passed": n_passed,
            **{k: round(v, 4) for k, v in check_rates.items()},
        }

    def _collect_eval_data(self, task, response: str, reward: float) -> dict:
        """Collect per-episode data for EvaluationResult."""
        _, detail = score_scheduling_task(task, response)
        return {
            "passed": float(detail["passed"]),
            "parse_success": float(detail["checks"].get("parsing", False)),
        }
