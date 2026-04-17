"""Implicit-Curriculum Replay Buffer for ISOPro.

Implements Sections 5.3 and 5.4 of the GCE paper:

  5.3 — Rejection Sampling as Continuous Self-Filter:
    "Only verified correct responses enter the replay buffer as training
    data. This creates a continuous evaluation regime: at every training
    iteration, the model's current capability is evaluated against ground
    truth, and only capability-consistent rollouts produce training signal.
    Capability dynamics are visible in the composition of the replay buffer
    at each iteration — a granularity that checkpoint-based evaluation
    cannot provide."

  5.4 — Implicit-Curriculum Replay Buffer:
    "Correct rollouts accumulate across all training iterations, creating
    a continuously updating training distribution anchored to the model's
    actual capability. Early iterations contribute easy wins that anchor
    weight updates; harder problems become accessible as capability
    develops. The curriculum falls out of the model's capability trajectory
    rather than being imposed by researcher curation. By iteration N, the
    model trains on everything it solved correctly in iterations 1 through
    N combined — compounding correct signal without compounding error."

Usage:
    buffer = ImplicitCurriculumBuffer()

    # After each iteration's rollouts are verified:
    buffer.add(correct_rollouts, iteration=1)

    # For training:
    traces = buffer.training_traces()

    # For the paper's figures:
    buffer.composition()           # tier -> count
    buffer.capability_trajectory() # per-iteration snapshots
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Rollout trace — what enters the buffer
# ---------------------------------------------------------------------------


@dataclass
class ReasoningTrace:
    """A verified-correct reasoning trace stored in the replay buffer.

    This is not just a (prompt, response) pair — it is a record of the
    model's own correct reasoning at a specific point in its capability
    trajectory. The iteration field anchors the trace in training time.

    Attributes:
        prompt: The full task prompt sent to the model.
        response: The model's generated response that passed verification.
        category: Task category / tier (e.g. "tier0_warmup", "tier3_deadline_pressure").
        iteration: The training iteration when this trace was generated.
        metadata: Arbitrary extra info (task_id, tier, seed, etc.).
    """

    prompt: str
    response: str
    category: str
    iteration: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capability snapshot — per-iteration record
# ---------------------------------------------------------------------------


@dataclass
class CapabilitySnapshot:
    """Snapshot of buffer state at a single training iteration.

    Captures what the model could solve at iteration N, enabling
    retrospective analysis of capability dynamics (Section 5.3).

    Attributes:
        iteration: Training iteration number.
        n_new_traces: Correct traces added this iteration.
        n_total_traces: Cumulative buffer size after this iteration.
        composition: Category -> count in the full buffer.
        hit_rate: Fraction of rollouts that passed verification.
        n_rollouts: Total rollouts attempted this iteration.
    """

    iteration: int
    n_new_traces: int = 0
    n_total_traces: int = 0
    composition: dict[str, int] = field(default_factory=dict)
    hit_rate: float = 0.0
    n_rollouts: int = 0

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "iteration": self.iteration,
            "n_new_traces": self.n_new_traces,
            "n_total_traces": self.n_total_traces,
            "composition": self.composition,
            "hit_rate": round(self.hit_rate, 4),
            "n_rollouts": self.n_rollouts,
        }


# ---------------------------------------------------------------------------
# Implicit Curriculum Buffer
# ---------------------------------------------------------------------------


class ImplicitCurriculumBuffer:
    """Capability-grounded replay buffer implementing GCE Sections 5.3-5.4.

    Core properties:
      - Accumulates correct traces across all iterations (no eviction).
      - Training distribution is grounded in the model's actual capability.
      - Early iterations contribute easy wins; harder traces enter as
        capability develops — the curriculum is implicit.
      - Full capability trajectory is recorded for observability.

    The buffer does NOT:
      - Impose a researcher-designed curriculum.
      - Evict old traces (compounding correct signal, not error).
      - Use a learned reward model (the verifier is the filter).
      - Require a reference model or KL penalty for stability.
    """

    def __init__(self) -> None:
        self._traces: list[ReasoningTrace] = []
        self._trajectory: list[CapabilitySnapshot] = []

    def add(
        self,
        correct_rollouts: list[dict],
        iteration: int,
        n_total_rollouts: int = 0,
    ) -> CapabilitySnapshot:
        """Add verified-correct rollouts from one training iteration.

        Each rollout dict must have keys: prompt, response, category.
        Optional keys: metadata (dict).

        Args:
            correct_rollouts: List of dicts with correct rollout data.
            iteration: Current training iteration number.
            n_total_rollouts: Total rollouts attempted (for hit rate).

        Returns:
            CapabilitySnapshot for this iteration.
        """
        new_traces = []
        for r in correct_rollouts:
            trace = ReasoningTrace(
                prompt=r["prompt"],
                response=r["response"],
                category=r.get("category", "unknown"),
                iteration=iteration,
                metadata=r.get("metadata", {}),
            )
            new_traces.append(trace)

        self._traces.extend(new_traces)

        # Compute hit rate
        hit_rate = (
            len(new_traces) / n_total_rollouts
            if n_total_rollouts > 0
            else 0.0
        )

        snapshot = CapabilitySnapshot(
            iteration=iteration,
            n_new_traces=len(new_traces),
            n_total_traces=len(self._traces),
            composition=self.composition(),
            hit_rate=hit_rate,
            n_rollouts=n_total_rollouts,
        )
        self._trajectory.append(snapshot)

        return snapshot

    def training_traces(
        self,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> list[ReasoningTrace]:
        """Return all accumulated traces for training.

        By iteration N, this returns everything the model solved correctly
        in iterations 1 through N — compounding correct signal.

        Args:
            shuffle: Randomize order for training.
            seed: Optional seed for reproducible shuffling.

        Returns:
            List of all ReasoningTrace objects in the buffer.
        """
        traces = list(self._traces)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(traces)
        return traces

    def training_pairs(
        self,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> list[tuple[str, str]]:
        """Return (prompt, response) pairs for training.

        Convenience method compatible with SFT training functions that
        expect (prompt, response) tuples.

        Args:
            shuffle: Randomize order.
            seed: Optional seed.

        Returns:
            List of (prompt, response) tuples.
        """
        traces = self.training_traces(shuffle=shuffle, seed=seed)
        return [(t.prompt, t.response) for t in traces]

    # ------------------------------------------------------------------
    # Observability — Section 5.3
    # ------------------------------------------------------------------

    def composition(self) -> dict[str, int]:
        """Current buffer composition by category.

        "Capability dynamics are visible in the composition of the
        replay buffer at each iteration."

        Returns:
            Dict of category -> trace count.
        """
        comp: dict[str, int] = defaultdict(int)
        for trace in self._traces:
            comp[trace.category] += 1
        return dict(comp)

    def capability_trajectory(self) -> list[CapabilitySnapshot]:
        """Full per-iteration capability trajectory.

        Returns the composition snapshot at every iteration, enabling
        the paper's Figure 2 (replay buffer composition over time) and
        Figure 3 (transition heatmap).

        Returns:
            Ordered list of CapabilitySnapshot objects.
        """
        return list(self._trajectory)

    def transition_points(self) -> dict[str, int]:
        """Identify when each category first entered the buffer.

        "The exact moment capabilities emerge" (Section 5.3).

        Returns:
            Dict of category -> iteration when first trace appeared.
        """
        first_seen: dict[str, int] = {}
        for trace in self._traces:
            if trace.category not in first_seen:
                first_seen[trace.category] = trace.iteration
        return first_seen

    def composition_over_time(self) -> list[dict]:
        """Buffer composition at each iteration (for Figure 2).

        Returns:
            List of dicts, one per iteration, with category counts.
        """
        return [s.to_dict() for s in self._trajectory]

    def hit_rate_over_time(self) -> list[tuple[int, float]]:
        """Hit rate (correct/total rollouts) at each iteration.

        Shows the model's improving capability over training time.

        Returns:
            List of (iteration, hit_rate) tuples.
        """
        return [(s.iteration, s.hit_rate) for s in self._trajectory]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total traces in the buffer."""
        return len(self._traces)

    @property
    def categories(self) -> set[str]:
        """Set of all categories represented in the buffer."""
        return {t.category for t in self._traces}

    @property
    def iteration_count(self) -> int:
        """Number of iterations that have contributed to the buffer."""
        return len(self._trajectory)

    def __len__(self) -> int:
        return len(self._traces)

    def __repr__(self) -> str:
        comp = self.composition()
        return (
            f"ImplicitCurriculumBuffer(size={self.size}, "
            f"categories={len(comp)}, "
            f"iterations={self.iteration_count})"
        )
