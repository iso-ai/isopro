"""Scheduling problem generator for the IsoZero scheduling experiment.

Generates resource-constrained project scheduling problems (RCPSP) at five
difficulty tiers, solves each with OR-Tools CP-SAT to confirm feasibility
and compute ground-truth optimal schedules, then renders problems as
natural language prompts.

Tier 1 — Sequencing:       6 jobs, no resources, loose deadlines, sparse DAG.
Tier 2 — Resource alloc:   6 jobs, 2 resource types, no deps, loose deadlines.
Tier 3 — Deadline pressure: 6 jobs, no deps, unlimited resources, tight deadlines.
Tier 4 — Pairwise:         8 jobs, two of three constraints active (3 subtypes).
Tier 5 — Full composition: 10 jobs, all constraints, held-out evaluation only.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ortools.sat.python import cp_model

from .base_task import DifficultyLevel, Task


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class SchedulingTier(Enum):
    """Six-tier difficulty progression for scheduling problems."""

    WARMUP = "tier0_warmup"
    SEQUENCING = "tier1_sequencing"
    RESOURCE_ALLOC = "tier2_resource_alloc"
    DEADLINE_PRESSURE = "tier3_deadline_pressure"
    PAIRWISE = "tier4_pairwise"
    FULL_COMPOSITION = "tier5_full_composition"

    @property
    def difficulty_level(self) -> DifficultyLevel:
        """Map tier to base DifficultyLevel for compatibility."""
        mapping = {
            SchedulingTier.WARMUP: DifficultyLevel.EASY,
            SchedulingTier.SEQUENCING: DifficultyLevel.EASY,
            SchedulingTier.RESOURCE_ALLOC: DifficultyLevel.EASY,
            SchedulingTier.DEADLINE_PRESSURE: DifficultyLevel.MEDIUM,
            SchedulingTier.PAIRWISE: DifficultyLevel.MEDIUM,
            SchedulingTier.FULL_COMPOSITION: DifficultyLevel.HARD,
        }
        return mapping[self]


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------

# Pairwise subtypes for Tier 4
PAIRWISE_SUBTYPES = [
    "sequencing_resources",
    "resources_deadlines",
    "sequencing_deadlines",
]


@dataclass
class TierConfig:
    """Parameters controlling problem generation for a tier.

    Attributes:
        n_jobs: Number of jobs in the problem.
        n_resource_types: Number of distinct resource types (0 = unlimited).
        resource_capacity_range: (min, max) capacity per resource type.
        job_duration_range: (min, max) duration for each job.
        job_resource_range: (min, max) resource requirement per job per type.
        n_dependency_edges: Number of precedence edges in the DAG.
        deadline_slack: Multiplier on makespan to set deadlines.
            1.0 = tight (deadline == optimal makespan), >1 = loose.
        use_dependencies: Whether to generate dependency edges.
        use_resources: Whether to enforce resource capacity limits.
        use_deadlines: Whether to enforce per-job deadlines.
    """

    n_jobs: int = 6
    n_resource_types: int = 0
    resource_capacity_range: tuple[int, int] = (2, 3)
    job_duration_range: tuple[int, int] = (1, 4)
    job_resource_range: tuple[int, int] = (1, 2)
    n_dependency_edges: int = 0
    deadline_slack: float = 2.0
    use_dependencies: bool = False
    use_resources: bool = False
    use_deadlines: bool = True


TIER_CONFIGS: dict[SchedulingTier, TierConfig] = {
    SchedulingTier.WARMUP: TierConfig(
        n_jobs=4,
        use_dependencies=True,
        n_dependency_edges=2,
        use_resources=False,
        use_deadlines=False,
        deadline_slack=3.0,
        job_duration_range=(1, 2),
    ),
    SchedulingTier.SEQUENCING: TierConfig(
        n_jobs=5,
        use_dependencies=True,
        n_dependency_edges=2,
        use_resources=False,
        use_deadlines=True,
        deadline_slack=3.0,
        job_duration_range=(1, 3),
    ),
    SchedulingTier.RESOURCE_ALLOC: TierConfig(
        n_jobs=6,
        use_dependencies=False,
        use_resources=True,
        n_resource_types=2,
        resource_capacity_range=(2, 3),
        use_deadlines=True,
        deadline_slack=2.0,
        job_duration_range=(1, 4),
        job_resource_range=(1, 2),
    ),
    SchedulingTier.DEADLINE_PRESSURE: TierConfig(
        n_jobs=6,
        use_dependencies=False,
        use_resources=False,
        use_deadlines=True,
        deadline_slack=1.15,
        job_duration_range=(2, 5),
    ),
    SchedulingTier.PAIRWISE: TierConfig(
        n_jobs=8,
        use_dependencies=True,
        n_dependency_edges=4,
        use_resources=True,
        n_resource_types=2,
        resource_capacity_range=(2, 4),
        use_deadlines=True,
        deadline_slack=1.5,
        job_duration_range=(1, 4),
        job_resource_range=(1, 2),
    ),
    SchedulingTier.FULL_COMPOSITION: TierConfig(
        n_jobs=10,
        use_dependencies=True,
        n_dependency_edges=6,
        use_resources=True,
        n_resource_types=2,
        resource_capacity_range=(2, 4),
        use_deadlines=True,
        deadline_slack=1.3,
        job_duration_range=(1, 5),
        job_resource_range=(1, 2),
    ),
}


# ---------------------------------------------------------------------------
# Problem data structures
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """A single job in the scheduling problem.

    Attributes:
        job_id: Human-readable identifier (e.g. "A", "B").
        duration: Number of time steps to complete.
        resource_requirements: Dict of resource_type -> units needed per step.
        deadline: Must complete by this time step (start + duration <= deadline).
    """

    job_id: str
    duration: int
    resource_requirements: dict[str, int] = field(default_factory=dict)
    deadline: int | None = None


@dataclass
class SchedulingProblem:
    """A complete scheduling problem instance.

    Attributes:
        jobs: List of jobs to schedule.
        dependencies: List of (predecessor_id, successor_id) edges.
        resource_capacities: Dict of resource_type -> max units per time step.
        horizon: Maximum time step (used as solver upper bound).
        tier: Which difficulty tier generated this problem.
        subtype: Pairwise subtype if tier == PAIRWISE, else None.
        seed: Random seed used for reproducibility.
    """

    jobs: list[Job]
    dependencies: list[tuple[str, str]]
    resource_capacities: dict[str, int]
    horizon: int
    tier: SchedulingTier
    subtype: str | None = None
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "duration": j.duration,
                    "resource_requirements": j.resource_requirements,
                    "deadline": j.deadline,
                }
                for j in self.jobs
            ],
            "dependencies": self.dependencies,
            "resource_capacities": self.resource_capacities,
            "horizon": self.horizon,
            "tier": self.tier.value,
            "subtype": self.subtype,
            "seed": self.seed,
        }


@dataclass
class SchedulingSolution:
    """Ground-truth solution from the OR-Tools solver.

    Attributes:
        start_times: Dict of job_id -> start time step.
        makespan: The optimal makespan (max completion time).
        is_optimal: Whether the solver proved optimality.
    """

    start_times: dict[str, int]
    makespan: int
    is_optimal: bool


# ---------------------------------------------------------------------------
# Job ID labels
# ---------------------------------------------------------------------------

_JOB_LABELS = [chr(ord("A") + i) for i in range(26)]


def _get_job_labels(n: int) -> list[str]:
    """Return n job labels: A, B, C, ..."""
    if n > 26:
        raise ValueError(f"Max 26 jobs supported, got {n}")
    return _JOB_LABELS[:n]


# ---------------------------------------------------------------------------
# DAG generation
# ---------------------------------------------------------------------------


def _generate_dag_edges(
    n_jobs: int,
    n_edges: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Generate a random DAG with exactly n_edges edges over n_jobs nodes.

    Edges only go from lower-index to higher-index nodes to guarantee
    acyclicity.

    Args:
        n_jobs: Number of nodes.
        n_edges: Desired number of edges.
        rng: Seeded RNG.

    Returns:
        List of (from_idx, to_idx) pairs with from_idx < to_idx.
    """
    all_possible = [
        (i, j) for i in range(n_jobs) for j in range(i + 1, n_jobs)
    ]
    n_edges = min(n_edges, len(all_possible))
    return rng.sample(all_possible, n_edges)


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------


def _generate_raw_problem(
    tier: SchedulingTier,
    rng: random.Random,
    subtype: str | None = None,
) -> SchedulingProblem:
    """Generate a scheduling problem from tier config (before solver check).

    For Tier 4 pairwise, the subtype determines which two of the three
    constraint dimensions are active.

    Args:
        tier: Difficulty tier.
        rng: Seeded RNG.
        subtype: Required for PAIRWISE tier.

    Returns:
        A SchedulingProblem instance (not yet verified feasible).
    """
    cfg = TIER_CONFIGS[tier]

    # For pairwise, override constraint flags based on subtype
    use_deps = cfg.use_dependencies
    use_res = cfg.use_resources
    use_deadlines = cfg.use_deadlines

    if tier == SchedulingTier.PAIRWISE:
        if subtype == "sequencing_resources":
            use_deps, use_res, use_deadlines = True, True, False
        elif subtype == "resources_deadlines":
            use_deps, use_res, use_deadlines = False, True, True
        elif subtype == "sequencing_deadlines":
            use_deps, use_res, use_deadlines = True, False, True
        else:
            raise ValueError(f"Unknown pairwise subtype: {subtype}")

    labels = _get_job_labels(cfg.n_jobs)

    # Generate jobs
    jobs: list[Job] = []
    for label in labels:
        duration = rng.randint(*cfg.job_duration_range)
        res_req: dict[str, int] = {}
        if use_res:
            for r in range(cfg.n_resource_types):
                res_req[f"R{r+1}"] = rng.randint(*cfg.job_resource_range)
        jobs.append(Job(job_id=label, duration=duration, resource_requirements=res_req))

    # Generate dependencies
    dep_edges: list[tuple[str, str]] = []
    if use_deps:
        idx_edges = _generate_dag_edges(cfg.n_jobs, cfg.n_dependency_edges, rng)
        dep_edges = [(labels[i], labels[j]) for i, j in idx_edges]

    # Resource capacities
    res_caps: dict[str, int] = {}
    if use_res:
        for r in range(cfg.n_resource_types):
            res_caps[f"R{r+1}"] = rng.randint(*cfg.resource_capacity_range)

    # Compute a rough horizon (sum of all durations is always feasible
    # for serial execution)
    total_duration = sum(j.duration for j in jobs)
    horizon = int(total_duration * cfg.deadline_slack) + 1

    # Set per-job deadlines
    if use_deadlines:
        for job in jobs:
            # Tight deadline: somewhere between job duration and horizon
            slack = cfg.deadline_slack
            min_deadline = job.duration
            max_deadline = max(min_deadline, int(horizon * 0.9))
            job.deadline = rng.randint(min_deadline, max_deadline)
    else:
        # No deadline constraint — set to horizon for all
        for job in jobs:
            job.deadline = horizon

    return SchedulingProblem(
        jobs=jobs,
        dependencies=dep_edges,
        resource_capacities=res_caps,
        horizon=horizon,
        tier=tier,
        subtype=subtype,
        seed=0,
    )


# ---------------------------------------------------------------------------
# OR-Tools CP-SAT solver
# ---------------------------------------------------------------------------


def solve_scheduling_problem(
    problem: SchedulingProblem,
    time_limit_s: float = 10.0,
) -> SchedulingSolution | None:
    """Solve a scheduling problem using OR-Tools CP-SAT.

    Minimizes makespan subject to precedence, resource, and deadline
    constraints. Returns None if no feasible solution exists.

    Args:
        problem: The scheduling problem to solve.
        time_limit_s: Solver time limit in seconds.

    Returns:
        SchedulingSolution if feasible, None otherwise.
    """
    model = cp_model.CpModel()
    horizon = problem.horizon

    # Decision variables: start time for each job
    starts: dict[str, cp_model.IntVar] = {}
    ends: dict[str, cp_model.IntVar] = {}
    intervals: dict[str, cp_model.IntervalVar] = {}

    for job in problem.jobs:
        start = model.new_int_var(0, horizon, f"start_{job.job_id}")
        end = model.new_int_var(0, horizon, f"end_{job.job_id}")
        interval = model.new_interval_var(
            start, job.duration, end, f"interval_{job.job_id}"
        )
        starts[job.job_id] = start
        ends[job.job_id] = end
        intervals[job.job_id] = interval

    # Precedence constraints
    job_map = {j.job_id: j for j in problem.jobs}
    for pred_id, succ_id in problem.dependencies:
        pred = job_map[pred_id]
        model.add(starts[succ_id] >= starts[pred_id] + pred.duration)

    # Resource capacity constraints (cumulative)
    for res_type, capacity in problem.resource_capacities.items():
        task_intervals = []
        demands = []
        for job in problem.jobs:
            req = job.resource_requirements.get(res_type, 0)
            if req > 0:
                task_intervals.append(intervals[job.job_id])
                demands.append(req)
        if task_intervals:
            model.add_cumulative(task_intervals, demands, capacity)

    # Deadline constraints
    for job in problem.jobs:
        if job.deadline is not None:
            model.add(ends[job.job_id] <= job.deadline)

    # Objective: minimize makespan
    makespan = model.new_int_var(0, horizon, "makespan")
    for job in problem.jobs:
        model.add(makespan >= ends[job.job_id])
    model.minimize(makespan)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution_starts = {
            job.job_id: solver.value(starts[job.job_id])
            for job in problem.jobs
        }
        return SchedulingSolution(
            start_times=solution_starts,
            makespan=solver.value(makespan),
            is_optimal=(status == cp_model.OPTIMAL),
        )
    return None


# ---------------------------------------------------------------------------
# Natural language rendering
# ---------------------------------------------------------------------------


def _resource_name(res_type: str, capacity: int) -> str:
    """Human-readable resource name for prompts."""
    # R1, R2 -> "Machine", "Worker" for readability
    names = {"R1": "machine", "R2": "worker"}
    return names.get(res_type, res_type.lower())


def render_problem_prompt(
    problem: SchedulingProblem,
    chain_of_thought: bool = True,
) -> str:
    """Convert a scheduling problem to a natural language prompt.

    Args:
        problem: The scheduling problem to render.
        chain_of_thought: If True, include structured reasoning instructions.

    Returns:
        A human-readable prompt string for the LLM.
    """
    n = len(problem.jobs)
    lines: list[str] = []

    # Opening
    lines.append("You are a project scheduler.")

    # Resource context
    if problem.resource_capacities:
        res_parts = []
        for rtype, cap in sorted(problem.resource_capacities.items()):
            name = _resource_name(rtype, cap)
            res_parts.append(f"{cap} {name}s")
        res_str = " and ".join(res_parts)
        lines.append(
            f"You have {n} jobs to schedule using {res_str}."
        )
    else:
        lines.append(f"You have {n} jobs to schedule.")

    lines.append("")

    # Job descriptions
    for job in problem.jobs:
        dur_str = f"{job.duration} time step{'s' if job.duration != 1 else ''}"
        parts = [f"Job {job.job_id} takes {dur_str}"]

        # Resource requirements
        if job.resource_requirements:
            res_parts = []
            for rtype, req in sorted(job.resource_requirements.items()):
                name = _resource_name(rtype, 0)
                res_parts.append(f"{req} {name}{'s' if req != 1 else ''}")
            parts.append("requires " + " and ".join(res_parts))

        # Dependencies
        deps = [pred for pred, succ in problem.dependencies if succ == job.job_id]
        if deps:
            dep_str = ", ".join(f"Job {d}" for d in deps)
            verb = "is" if len(deps) == 1 else "are"
            parts.append(f"cannot start until {dep_str} {verb} complete")

        # Deadline
        if job.deadline is not None and job.deadline < problem.horizon:
            parts.append(f"must complete by time step {job.deadline}")

        lines.append(", ".join(parts) + ".")

    # Resource capacity reminder
    if problem.resource_capacities:
        lines.append("")
        for rtype, cap in sorted(problem.resource_capacities.items()):
            name = _resource_name(rtype, cap)
            lines.append(
                f"At most {cap} jobs can use a {name} at the same time step."
            )

    # Reasoning instructions
    if chain_of_thought:
        lines.append("")
        lines.append("Think step by step:")

        if problem.dependencies:
            lines.append(
                "1. Identify the dependency chain — which jobs must "
                "finish before others can start."
            )
        if problem.resource_capacities:
            lines.append(
                f"{'2' if problem.dependencies else '1'}. Check resource "
                "limits — how many jobs can run at the same time step "
                "without exceeding capacity."
            )
        has_tight = any(
            j.deadline is not None and j.deadline < problem.horizon
            for j in problem.jobs
        )
        if has_tight:
            step = sum([bool(problem.dependencies), bool(problem.resource_capacities)]) + 1
            lines.append(
                f"{step}. Verify deadlines — ensure each job's start time "
                "plus duration does not exceed its deadline."
            )

    # Output instruction
    lines.append("")
    lines.append(
        "Provide your final answer as start times for each job in the "
        "format Job: start_time. For example: A: 0, B: 3, C: 5"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public generation API
# ---------------------------------------------------------------------------


def generate_scheduling_problem(
    tier: SchedulingTier,
    seed: int,
    subtype: str | None = None,
    max_retries: int = 50,
) -> tuple[SchedulingProblem, SchedulingSolution] | None:
    """Generate a feasible scheduling problem with ground-truth solution.

    Repeatedly generates candidate problems until one is feasible (solver
    finds a solution). Returns None if max_retries is exhausted.

    Args:
        tier: Difficulty tier.
        seed: Random seed for reproducibility.
        subtype: Required for PAIRWISE tier.
        max_retries: Maximum generation attempts before giving up.

    Returns:
        Tuple of (problem, solution) if feasible, None if all retries fail.
    """
    for attempt in range(max_retries):
        rng = random.Random(seed + attempt)
        problem = _generate_raw_problem(tier, rng, subtype=subtype)
        problem.seed = seed + attempt
        solution = solve_scheduling_problem(problem)
        if solution is not None:
            return problem, solution
    return None


def generate_scheduling_task(
    tier: SchedulingTier,
    seed: int,
    subtype: str | None = None,
) -> Task | None:
    """Generate a scheduling Task with NL prompt and ground-truth solution.

    Convenience wrapper that combines problem generation, solving, and
    prompt rendering into a single Task object compatible with the
    ISOPro training infrastructure.

    Args:
        tier: Difficulty tier.
        seed: Random seed.
        subtype: Required for PAIRWISE tier.

    Returns:
        A Task instance, or None if generation fails.
    """
    result = generate_scheduling_problem(tier, seed, subtype=subtype)
    if result is None:
        return None

    problem, solution = result
    prompt = render_problem_prompt(problem)

    category = tier.value
    if subtype:
        category = f"{tier.value}_{subtype}"

    return Task.make(
        prompt=prompt,
        ground_truth=solution.start_times,
        difficulty=tier.difficulty_level,
        category=category,
        metadata={
            "problem": problem.to_dict(),
            "solution": {
                "start_times": solution.start_times,
                "makespan": solution.makespan,
                "is_optimal": solution.is_optimal,
            },
            "tier": tier.value,
            "subtype": subtype,
            "seed": problem.seed,
        },
    )


# ---------------------------------------------------------------------------
# Batch generation helpers
# ---------------------------------------------------------------------------


def build_tier_task_bank(
    tier: SchedulingTier,
    n_problems: int,
    base_seed: int = 0,
    subtypes: list[str] | None = None,
) -> list[Task]:
    """Generate a batch of tasks for a given tier.

    For PAIRWISE tier, generates n_problems per subtype.

    Args:
        tier: Difficulty tier.
        n_problems: Number of problems to generate (per subtype for PAIRWISE).
        base_seed: Starting seed for reproducibility.
        subtypes: Subtypes to generate (only for PAIRWISE).

    Returns:
        List of Task objects.
    """
    tasks: list[Task] = []

    if tier == SchedulingTier.PAIRWISE:
        for st in (subtypes or PAIRWISE_SUBTYPES):
            for i in range(n_problems):
                seed = base_seed + hash((st, i)) % (2**31)
                task = generate_scheduling_task(tier, seed, subtype=st)
                if task is not None:
                    tasks.append(task)
    else:
        for i in range(n_problems):
            seed = base_seed + i * 100
            task = generate_scheduling_task(tier, seed)
            if task is not None:
                tasks.append(task)

    return tasks


def build_full_task_bank(
    n_per_tier: int = 20,
    n_per_pairwise_subtype: int = 10,
    base_seed: int = 42,
) -> dict[str, list[Task]]:
    """Generate the complete task bank across all tiers.

    Args:
        n_per_tier: Problems per tier (Tiers 1-3, 5).
        n_per_pairwise_subtype: Problems per Tier 4 subtype.
        base_seed: Master seed.

    Returns:
        Dict mapping tier name to list of Tasks.
    """
    bank: dict[str, list[Task]] = {}

    for tier in SchedulingTier:
        tier_seed = base_seed + hash(tier.value) % (2**31)
        if tier == SchedulingTier.PAIRWISE:
            bank[tier.value] = build_tier_task_bank(
                tier, n_per_pairwise_subtype, tier_seed
            )
        else:
            bank[tier.value] = build_tier_task_bank(
                tier, n_per_tier, tier_seed
            )

    return bank


def build_eval_set(
    n_per_tier: int = 5,
    n_per_pairwise_subtype: int = 3,
    base_seed: int = 9999,
) -> dict[str, list[Task]]:
    """Generate held-out evaluation tasks (separate seed space).

    Args:
        n_per_tier: Eval problems per tier.
        n_per_pairwise_subtype: Eval problems per Tier 4 subtype.
        base_seed: Eval seed (distinct from training).

    Returns:
        Dict mapping tier name to list of Tasks.
    """
    return build_full_task_bank(
        n_per_tier=n_per_tier,
        n_per_pairwise_subtype=n_per_pairwise_subtype,
        base_seed=base_seed,
    )
