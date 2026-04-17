"""Multi-turn scheduling interaction for scope validity.

Implements a revision loop where the model submits a schedule, receives
structured feedback on which constraints failed, and can revise. The
trajectory — attempt, feedback, revision, verification — is the unit
of evaluation, not a single output.

This addresses scope validity (Section 3.3): evaluation at the trajectory
level rather than the single-turn output level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scheduling_verifier import VerificationResult, verify_schedule


@dataclass
class Turn:
    """One turn in a multi-turn scheduling interaction.

    Attributes:
        attempt: The attempt number (1-indexed).
        response: The model's response text.
        result: Verification result for this attempt.
        feedback: Structured feedback sent back to the model.
    """

    attempt: int
    response: str
    result: VerificationResult
    feedback: str


@dataclass
class TrajectoryResult:
    """Complete result of a multi-turn scheduling interaction.

    Attributes:
        turns: List of all turns in the trajectory.
        solved: Whether the problem was eventually solved.
        attempts_to_solve: Number of attempts before success (0 if unsolved).
        problem_tier: Tier of the problem.
    """

    turns: list[Turn] = field(default_factory=list)
    solved: bool = False
    attempts_to_solve: int = 0
    problem_tier: str = ""

    @property
    def n_turns(self) -> int:
        """Total number of turns."""
        return len(self.turns)

    @property
    def showed_recovery(self) -> bool:
        """True if the model failed at least once then succeeded."""
        return self.solved and self.attempts_to_solve > 1

    def score(self) -> float:
        """Trajectory-level score.

        Returns:
            1.0 if solved on first attempt.
            0.5 if solved after revision (shows error recovery).
            0.0 if never solved.
        """
        if not self.solved:
            return 0.0
        if self.attempts_to_solve == 1:
            return 1.0
        return 0.5


def format_feedback(result: VerificationResult, problem_dict: dict) -> str:
    """Convert a verification result into structured feedback for the model.

    The feedback tells the model which specific constraints failed without
    revealing the correct answer.

    Args:
        result: Verification result from the previous attempt.
        problem_dict: The problem specification.

    Returns:
        Human-readable feedback string.
    """
    if result.passed:
        return "Your schedule is correct. All constraints are satisfied."

    lines = ["Your schedule has the following issues:"]

    checks = result.checks
    if not checks.get("parsing", False):
        lines.append(
            "- Could not parse your answer. Please provide start times "
            "in the format: A: 0, B: 3, C: 5"
        )
        return "\n".join(lines)

    if not checks.get("validity", False):
        missing = []
        for f in result.failures:
            if "Missing jobs" in f:
                lines.append(f"- {f}. Every job needs a start time.")
                break
        for f in result.failures:
            if "negative start time" in f:
                lines.append(f"- {f}. Start times must be >= 0.")

    if not checks.get("precedence", False):
        for f in result.failures:
            if "Precedence" in f:
                lines.append(f"- {f}")

    if not checks.get("resource_capacity", False):
        for f in result.failures:
            if "Resource" in f or "overloaded" in f:
                lines.append(f"- {f}")

    if not checks.get("deadlines", False):
        for f in result.failures:
            if "Deadline" in f:
                lines.append(f"- {f}")

    lines.append("")
    lines.append(
        "Please revise your schedule to fix these issues. "
        "Provide updated start times in the format: A: 0, B: 3, C: 5"
    )

    return "\n".join(lines)


def run_multiturn_scheduling(
    backend,
    task,
    gen_config,
    max_attempts: int = 3,
) -> TrajectoryResult:
    """Run a multi-turn scheduling interaction.

    The model gets up to max_attempts to produce a valid schedule.
    After each failed attempt, it receives structured feedback on
    which constraints were violated.

    Args:
        backend: Any backend with a .generate() method.
        task: A scheduling Task with metadata containing 'problem' dict.
        gen_config: GenerationConfig for the backend.
        max_attempts: Maximum revision attempts.

    Returns:
        TrajectoryResult with the full interaction trajectory.
    """
    problem_dict = task.metadata["problem"]
    trajectory = TrajectoryResult(
        problem_tier=task.metadata.get("tier", "unknown"),
    )

    # First attempt uses the original prompt
    prompt = task.prompt

    for attempt in range(1, max_attempts + 1):
        response = backend.generate(prompt, gen_config)
        result = verify_schedule(response.text, problem_dict)

        feedback = format_feedback(result, problem_dict)
        turn = Turn(
            attempt=attempt,
            response=response.text,
            result=result,
            feedback=feedback,
        )
        trajectory.turns.append(turn)

        if result.passed:
            trajectory.solved = True
            trajectory.attempts_to_solve = attempt
            break

        # Build revision prompt with feedback
        prompt = (
            f"{task.prompt}\n\n"
            f"Previous attempt (attempt {attempt}):\n"
            f"{response.text}\n\n"
            f"Feedback:\n{feedback}"
        )

    return trajectory


def evaluate_multiturn_per_tier(
    backend,
    eval_sets: dict[str, list],
    gen_config,
    max_attempts: int = 3,
) -> dict[str, dict[str, float]]:
    """Run multi-turn evaluation across all tiers.

    Args:
        backend: Any backend with .generate().
        eval_sets: Dict of tier_name -> list of Tasks.
        gen_config: GenerationConfig.
        max_attempts: Max revision attempts per problem.

    Returns:
        Dict of tier_name -> metrics dict with keys:
            first_attempt_acc, trajectory_acc, recovery_rate, mean_attempts.
    """
    results: dict[str, dict[str, float]] = {}

    for tier_name, tasks in sorted(eval_sets.items()):
        if not tasks:
            results[tier_name] = {
                "first_attempt_acc": 0.0,
                "trajectory_acc": 0.0,
                "recovery_rate": 0.0,
                "mean_attempts": 0.0,
            }
            continue

        first_correct = 0
        eventually_correct = 0
        recovered = 0
        total_attempts = 0

        for task in tasks:
            traj = run_multiturn_scheduling(
                backend, task, gen_config, max_attempts=max_attempts
            )

            if traj.solved and traj.attempts_to_solve == 1:
                first_correct += 1
            if traj.solved:
                eventually_correct += 1
            if traj.showed_recovery:
                recovered += 1
            total_attempts += traj.n_turns

        n = len(tasks)
        results[tier_name] = {
            "first_attempt_acc": round(first_correct / n, 4),
            "trajectory_acc": round(eventually_correct / n, 4),
            "recovery_rate": round(recovered / n, 4),
            "mean_attempts": round(total_attempts / n, 2),
        }

    return results
