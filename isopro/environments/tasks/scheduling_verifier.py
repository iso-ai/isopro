"""Deterministic verifier for scheduling problem responses.

Two-stage verification:
  Stage 1 — Parsing: extract job-to-start-time mappings from free-text
            model output using multiple fallback strategies.
  Stage 2 — Constraint checking: binary pass/fail on precedence, resource
            capacity, deadline satisfaction, and validity.

No partial credit. Any single failure rejects the entire response.
This is the core GCE architectural claim — the verifier replaces the
reward model entirely.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of verifying a model's scheduling response.

    Attributes:
        passed: True if all constraints are satisfied.
        parsed_schedule: Extracted start times, or None if parsing failed.
        parse_strategy: Which parsing strategy succeeded.
        failures: List of human-readable failure reasons.
        checks: Dict of constraint_name -> bool for detailed logging.
    """

    passed: bool
    parsed_schedule: dict[str, int] | None = None
    parse_strategy: str | None = None
    failures: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1: Parsing strategies
# ---------------------------------------------------------------------------


def _parse_colon_format(text: str) -> dict[str, int] | None:
    """Parse 'Job A: 0, Job B: 3' or 'A: 0, B: 3' format.

    Also handles newline-separated variants:
        Job A: 0
        Job B: 3

    Args:
        text: Model's raw response text.

    Returns:
        Dict of job_id -> start_time, or None if parsing fails.
    """
    # Match patterns like "A: 0" or "Job A: 0" with optional "Job" prefix
    pattern = r"(?:Job\s+)?([A-Z])\s*:\s*(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    return {m[0].upper(): int(m[1]) for m in matches}


def _parse_starts_at_format(text: str) -> dict[str, int] | None:
    """Parse 'A starts at 0, B starts at 3' format.

    Also handles variations like:
        Job A starts at time step 0
        A begins at 0

    Args:
        text: Model's raw response text.

    Returns:
        Dict of job_id -> start_time, or None if parsing fails.
    """
    pattern = (
        r"(?:Job\s+)?([A-Z])\s+"
        r"(?:starts?|begins?)\s+"
        r"(?:at\s+)?(?:time\s+(?:step\s+)?)?(\d+)"
    )
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    return {m[0].upper(): int(m[1]) for m in matches}


def _parse_equals_format(text: str) -> dict[str, int] | None:
    """Parse 'A = 0, B = 3' or 'A=0' format.

    Args:
        text: Model's raw response text.

    Returns:
        Dict of job_id -> start_time, or None if parsing fails.
    """
    pattern = r"(?:Job\s+)?([A-Z])\s*=\s*(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    return {m[0].upper(): int(m[1]) for m in matches}


def _parse_numbered_list(text: str) -> dict[str, int] | None:
    """Parse numbered list format.

    Handles:
        1. A: 0
        2. B: 3
    or:
        1. Job A - start time 0
        2. Job B - start time 3

    Args:
        text: Model's raw response text.

    Returns:
        Dict of job_id -> start_time, or None if parsing fails.
    """
    pattern = (
        r"\d+\.\s*(?:Job\s+)?([A-Z])\s*"
        r"(?::\s*|[-–—]\s*(?:start\s*(?:time\s*)?)?\s*)"
        r"(\d+)"
    )
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    return {m[0].upper(): int(m[1]) for m in matches}


def _parse_json_format(text: str) -> dict[str, int] | None:
    """Parse JSON-like format: {"A": 0, "B": 3}.

    Args:
        text: Model's raw response text.

    Returns:
        Dict of job_id -> start_time, or None if parsing fails.
    """
    # Find JSON-like blocks
    pattern = r'\{[^{}]*\}'
    json_matches = re.findall(pattern, text)
    for block in json_matches:
        kv_pattern = r'"?([A-Z])"?\s*:\s*(\d+)'
        matches = re.findall(kv_pattern, block, re.IGNORECASE)
        if matches:
            return {m[0].upper(): int(m[1]) for m in matches}
    return None


# Ordered list of parsing strategies — try each in sequence
_PARSE_STRATEGIES: list[tuple[str, Any]] = [
    ("colon", _parse_colon_format),
    ("starts_at", _parse_starts_at_format),
    ("equals", _parse_equals_format),
    ("numbered_list", _parse_numbered_list),
    ("json", _parse_json_format),
]


def parse_schedule(text: str) -> tuple[dict[str, int] | None, str | None]:
    """Extract job start times from model response using fallback strategies.

    Tries each parsing strategy in order. Returns the first successful parse.

    Args:
        text: The model's raw text response.

    Returns:
        Tuple of (parsed_schedule, strategy_name). Both None if all fail.
    """
    for name, strategy in _PARSE_STRATEGIES:
        result = strategy(text)
        if result and len(result) > 0:
            return result, name
    return None, None


# ---------------------------------------------------------------------------
# Stage 2: Constraint checking
# ---------------------------------------------------------------------------


def check_validity(
    schedule: dict[str, int],
    expected_jobs: list[str],
) -> tuple[bool, list[str]]:
    """Check that all jobs are assigned non-negative start times.

    Args:
        schedule: Parsed job_id -> start_time mapping.
        expected_jobs: List of job IDs that must be present.

    Returns:
        Tuple of (passed, list of failure reasons).
    """
    failures: list[str] = []

    missing = set(expected_jobs) - set(schedule.keys())
    if missing:
        failures.append(f"Missing jobs: {sorted(missing)}")

    for job_id, start in schedule.items():
        if start < 0:
            failures.append(f"Job {job_id} has negative start time: {start}")

    return len(failures) == 0, failures


def check_precedence(
    schedule: dict[str, int],
    dependencies: list[tuple[str, str]],
    job_durations: dict[str, int],
) -> tuple[bool, list[str]]:
    """Check that all precedence constraints are satisfied.

    For every edge (A, B): B's start >= A's start + A's duration.

    Args:
        schedule: Parsed job_id -> start_time mapping.
        dependencies: List of (predecessor, successor) edges.
        job_durations: Dict of job_id -> duration.

    Returns:
        Tuple of (passed, list of failure reasons).
    """
    failures: list[str] = []

    for pred, succ in dependencies:
        if pred not in schedule or succ not in schedule:
            continue  # Missing jobs caught by validity check
        pred_end = schedule[pred] + job_durations[pred]
        if schedule[succ] < pred_end:
            failures.append(
                f"Precedence violation: Job {succ} starts at {schedule[succ]} "
                f"but Job {pred} doesn't finish until {pred_end}"
            )

    return len(failures) == 0, failures


def check_resource_capacity(
    schedule: dict[str, int],
    jobs: list[dict],
    resource_capacities: dict[str, int],
) -> tuple[bool, list[str]]:
    """Check that resource usage never exceeds capacity at any time step.

    Args:
        schedule: Parsed job_id -> start_time mapping.
        jobs: List of job dicts with job_id, duration, resource_requirements.
        resource_capacities: Dict of resource_type -> max capacity.

    Returns:
        Tuple of (passed, list of failure reasons).
    """
    if not resource_capacities:
        return True, []

    failures: list[str] = []

    # Build per-timestep resource usage
    max_time = 0
    for job in jobs:
        jid = job["job_id"]
        if jid in schedule:
            end = schedule[jid] + job["duration"]
            max_time = max(max_time, end)

    for res_type, capacity in resource_capacities.items():
        for t in range(max_time):
            usage = 0
            active_jobs = []
            for job in jobs:
                jid = job["job_id"]
                if jid not in schedule:
                    continue
                start = schedule[jid]
                end = start + job["duration"]
                if start <= t < end:
                    req = job.get("resource_requirements", {}).get(res_type, 0)
                    usage += req
                    if req > 0:
                        active_jobs.append(jid)
            if usage > capacity:
                failures.append(
                    f"Resource {res_type} overloaded at time step {t}: "
                    f"usage={usage} > capacity={capacity} "
                    f"(active jobs: {active_jobs})"
                )
                break  # One violation per resource is enough

    return len(failures) == 0, failures


def check_deadlines(
    schedule: dict[str, int],
    jobs: list[dict],
    horizon: int,
) -> tuple[bool, list[str]]:
    """Check that all jobs complete by their deadlines.

    Args:
        schedule: Parsed job_id -> start_time mapping.
        jobs: List of job dicts with job_id, duration, deadline.
        horizon: Problem horizon (used when deadline == horizon means no constraint).

    Returns:
        Tuple of (passed, list of failure reasons).
    """
    failures: list[str] = []

    for job in jobs:
        jid = job["job_id"]
        deadline = job.get("deadline")
        if deadline is None or jid not in schedule:
            continue
        end = schedule[jid] + job["duration"]
        if end > deadline:
            failures.append(
                f"Deadline violation: Job {jid} finishes at {end} "
                f"but deadline is {deadline}"
            )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------


def verify_schedule(
    response: str,
    problem_dict: dict,
) -> VerificationResult:
    """Verify a model's scheduling response against the problem constraints.

    Two-stage process:
      1. Parse the response to extract start times.
      2. Check all constraints (validity, precedence, resources, deadlines).

    Any single failure means the whole response is rejected. No partial credit.

    Args:
        response: The model's raw text response.
        problem_dict: The problem specification dict (from SchedulingProblem.to_dict()).

    Returns:
        VerificationResult with pass/fail and detailed diagnostics.
    """
    jobs = problem_dict["jobs"]
    dependencies = problem_dict["dependencies"]
    resource_capacities = problem_dict["resource_capacities"]
    horizon = problem_dict["horizon"]
    expected_jobs = [j["job_id"] for j in jobs]
    job_durations = {j["job_id"]: j["duration"] for j in jobs}

    # Stage 1: Parse
    schedule, strategy = parse_schedule(response)

    if schedule is None:
        return VerificationResult(
            passed=False,
            failures=["Failed to parse any job start times from response"],
            checks={"parsing": False},
        )

    # Stage 2: Constraint checks
    all_failures: list[str] = []
    checks: dict[str, bool] = {"parsing": True}

    # Validity
    valid, fails = check_validity(schedule, expected_jobs)
    checks["validity"] = valid
    all_failures.extend(fails)

    # Precedence
    prec_ok, fails = check_precedence(schedule, dependencies, job_durations)
    checks["precedence"] = prec_ok
    all_failures.extend(fails)

    # Resource capacity
    res_ok, fails = check_resource_capacity(schedule, jobs, resource_capacities)
    checks["resource_capacity"] = res_ok
    all_failures.extend(fails)

    # Deadlines
    dl_ok, fails = check_deadlines(schedule, jobs, horizon)
    checks["deadlines"] = dl_ok
    all_failures.extend(fails)

    passed = len(all_failures) == 0

    return VerificationResult(
        passed=passed,
        parsed_schedule=schedule,
        parse_strategy=strategy,
        failures=all_failures,
        checks=checks,
    )


def score_scheduling_task(
    task: Any,
    response: str,
) -> tuple[float, dict]:
    """Score a scheduling task response. Binary: 1.0 or 0.0.

    Compatible with the ISOPro environment score() interface.

    Args:
        task: A Task object with metadata containing 'problem' dict.
        response: The model's raw text response.

    Returns:
        Tuple of (reward, detail_dict).
    """
    problem_dict = task.metadata["problem"]
    result = verify_schedule(response, problem_dict)

    detail = {
        "passed": result.passed,
        "parse_strategy": result.parse_strategy,
        "failures": result.failures,
        "checks": result.checks,
        "parsed_schedule": result.parsed_schedule,
    }

    return 1.0 if result.passed else 0.0, detail
