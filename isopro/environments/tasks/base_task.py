"""Base task primitives shared across all ISOPro environments.

Defines the Task dataclass and DifficultyLevel enum that every task
generator produces and every environment consumes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DifficultyLevel(Enum):
    """Ordered difficulty levels used by CurriculumScheduler.

    Attributes:
        EASY: Entry-level tasks; expected accuracy > 80% for a capable model.
        MEDIUM: Intermediate tasks; expected accuracy 50-80%.
        HARD: Challenging tasks; expected accuracy < 50% for most models.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    def next(self) -> "DifficultyLevel":
        """Return the next harder difficulty level, or self if already HARD.

        Returns:
            The next DifficultyLevel in the ordered sequence.
        """
        order = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def prev(self) -> "DifficultyLevel":
        """Return the next easier difficulty level, or self if already EASY.

        Returns:
            The previous DifficultyLevel in the ordered sequence.
        """
        order = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        idx = order.index(self)
        return order[max(idx - 1, 0)]


@dataclass
class Task:
    """A single evaluation task produced by a task generator.

    Every environment's generate_task() returns a Task. The score()
    method on the environment takes a Task + model response and returns
    a deterministic float reward.

    Attributes:
        task_id: Unique identifier for reproducibility and logging.
        prompt: The full text prompt to send to the model.
        ground_truth: The correct answer in whatever form the environment
            expects (str, list, dict, int, etc.).
        difficulty: DifficultyLevel for curriculum scheduling.
        category: Environment-specific subcategory (e.g. "arithmetic",
            "single_needle", "lexical_constraint"). Used for per-category
            metric breakdowns in EvaluationResult.
        metadata: Arbitrary environment-specific extras (e.g. which tools
            are available, document length, constraint list). Not used by
            the base infrastructure — consumed by each environment's scorer.
    """

    task_id: str
    prompt: str
    ground_truth: Any
    difficulty: DifficultyLevel
    category: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        prompt: str,
        ground_truth: Any,
        difficulty: DifficultyLevel,
        category: str,
        metadata: dict | None = None,
    ) -> "Task":
        """Convenience constructor that auto-generates a unique task_id.

        Args:
            prompt: The full prompt text for the model.
            ground_truth: The correct answer or structured reference.
            difficulty: DifficultyLevel for this task.
            category: Subcategory string for metric breakdowns.
            metadata: Optional extra context for the environment scorer.

        Returns:
            A new Task with a freshly generated UUID task_id.
        """
        return cls(
            task_id=uuid.uuid4().hex[:12],
            prompt=prompt,
            ground_truth=ground_truth,
            difficulty=difficulty,
            category=category,
            metadata=metadata or {},
        )
