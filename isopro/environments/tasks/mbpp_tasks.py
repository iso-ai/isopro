"""Task generator for MBPP-based code-creation experiments.

Loads the sanitized MBPP dataset (Google Research) and renders each problem
as an ISOPro Task whose `score()` runs the reference unit tests against the
model's generated code in a sandboxed subprocess.

Difficulty tiering (by canonical-solution body length):

    T0 — warmup       1–2 lines    (~163 problems; trivial returns)
    T1 — short        3–4 lines    (~92  problems; one branch or one loop)
    T2 — medium       5–7 lines    (~97  problems; nested logic)
    T3 — long         8–12 lines   (~51  problems; multi-step transforms)
    T4 — held-out     13+ lines    (~24  problems; algorithmic composition)

The held-out tier is *never* shown to the model during training, mirroring
the scheduling experiment's T5 compositional-generalization split.

Reference:
    Austin et al. (2021), "Program Synthesis with Large Language Models."
    Sanitized MBPP from https://github.com/google-research/google-research/.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .base_task import DifficultyLevel, Task


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


_MBPP_SANITIZED_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/mbpp/sanitized-mbpp.json"
)


class MBPPTier(Enum):
    """Five-tier difficulty progression for MBPP code-creation problems.

    Tiers are assigned by the line count of the canonical reference solution,
    which is a stable proxy for problem complexity within MBPP.
    """

    WARMUP = "tier0_warmup"
    SHORT = "tier1_short"
    MEDIUM = "tier2_medium"
    LONG = "tier3_long"
    HELD_OUT = "tier4_held_out"


@dataclass
class _TierConfig:
    """Body-line-count cutoffs for one tier.

    Attributes:
        max_lines: Inclusive upper bound on canonical-solution body lines.
        difficulty: ISOPro DifficultyLevel for curriculum scheduling.
    """

    max_lines: int
    difficulty: DifficultyLevel


_TIER_CONFIGS: dict[MBPPTier, _TierConfig] = {
    MBPPTier.WARMUP: _TierConfig(max_lines=2, difficulty=DifficultyLevel.EASY),
    MBPPTier.SHORT: _TierConfig(max_lines=4, difficulty=DifficultyLevel.EASY),
    MBPPTier.MEDIUM: _TierConfig(max_lines=7, difficulty=DifficultyLevel.MEDIUM),
    MBPPTier.LONG: _TierConfig(max_lines=12, difficulty=DifficultyLevel.MEDIUM),
    MBPPTier.HELD_OUT: _TierConfig(
        max_lines=10_000, difficulty=DifficultyLevel.HARD
    ),
}


# ---------------------------------------------------------------------------
# Problem dataclass
# ---------------------------------------------------------------------------


@dataclass
class MBPPProblem:
    """One MBPP problem in a structure-friendly form.

    Attributes:
        task_id: Original MBPP task ID (stable across runs).
        prompt: Natural-language description of what to implement.
        canonical_code: Reference solution from MBPP. Used for tiering only —
            never shown to the model.
        test_imports: Imports the test cases require.
        test_list: Assertion strings the generated code must pass.
        body_lines: Number of non-blank, non-comment, non-`def` lines in
            the canonical solution.
        tier: Which MBPPTier this problem belongs to.
    """

    task_id: int
    prompt: str
    canonical_code: str
    test_imports: list[str]
    test_list: list[str]
    body_lines: int
    tier: MBPPTier


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _count_body_lines(code: str) -> int:
    """Count non-blank, non-comment, non-`def` lines in a function body.

    Args:
        code: Full source of one canonical solution.

    Returns:
        Integer body-line count used for tier assignment.
    """
    n = 0
    for raw in code.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("def "):
            continue
        n += 1
    return n


def _assign_tier(body_lines: int) -> MBPPTier:
    """Pick the tier whose max_lines threshold first contains body_lines.

    Args:
        body_lines: Line count from `_count_body_lines`.

    Returns:
        The MBPPTier this problem belongs to.
    """
    for tier in (
        MBPPTier.WARMUP,
        MBPPTier.SHORT,
        MBPPTier.MEDIUM,
        MBPPTier.LONG,
    ):
        if body_lines <= _TIER_CONFIGS[tier].max_lines:
            return tier
    return MBPPTier.HELD_OUT


def _resolve_dataset_path(local_path: str | None) -> Path:
    """Locate the sanitized MBPP JSON, downloading it if needed.

    Caches in ``$ISOPRO_CACHE_DIR/mbpp/sanitized-mbpp.json`` (default
    ``~/.cache/isopro``) so subsequent calls are offline.

    Args:
        local_path: Optional explicit path to the JSON file.

    Returns:
        Path to the on-disk JSON file.
    """
    if local_path:
        return Path(local_path).expanduser().resolve()

    cache_root = Path(
        os.environ.get("ISOPRO_CACHE_DIR")
        or (Path.home() / ".cache" / "isopro")
    )
    cache_path = cache_root / "mbpp" / "sanitized-mbpp.json"

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MBPP_SANITIZED_URL, cache_path)

    return cache_path


def load_mbpp_problems(local_path: str | None = None) -> list[MBPPProblem]:
    """Load the sanitized MBPP dataset and bin every problem into a tier.

    Args:
        local_path: Optional path to a local sanitized-mbpp.json. If None,
            the file is downloaded once to the ISOPro cache directory.

    Returns:
        Ordered list of MBPPProblem, sorted by task_id.
    """
    path = _resolve_dataset_path(local_path)
    with path.open() as f:
        raw = json.load(f)

    problems: list[MBPPProblem] = []
    for entry in raw:
        body = _count_body_lines(entry["code"])
        problems.append(
            MBPPProblem(
                task_id=int(entry["task_id"]),
                prompt=entry["prompt"],
                canonical_code=entry["code"],
                test_imports=list(entry.get("test_imports", [])),
                test_list=list(entry["test_list"]),
                body_lines=body,
                tier=_assign_tier(body),
            )
        )
    problems.sort(key=lambda p: p.task_id)
    return problems


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_mbpp_prompt(problem: MBPPProblem, n_example_tests: int = 1) -> str:
    """Render an MBPP problem as the user-facing prompt sent to the model.

    Follows the standard MBPP evaluation convention: problem description +
    a small number of assertion examples to disambiguate signature.

    Args:
        problem: The MBPPProblem to render.
        n_example_tests: How many assertions from `test_list` to include
            in the prompt (0 disables, full passes the entire test list).

    Returns:
        The prompt string.
    """
    examples = "\n".join(
        f"  {t}" for t in problem.test_list[:n_example_tests]
    ) if n_example_tests else ""

    blocks = [problem.prompt.strip()]
    if examples:
        blocks.append("Your code must pass at least these tests:\n" + examples)
    blocks.append(
        "Reply with a single ```python``` code block containing the function "
        "definition. No explanation, no other code."
    )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------


def generate_mbpp_task(problem: MBPPProblem, n_example_tests: int = 1) -> Task:
    """Wrap an MBPPProblem in an ISOPro Task.

    The Task's ``ground_truth`` is the full assertion list (the verifier runs
    every test, even ones not surfaced in the prompt). Metadata carries
    everything the verifier needs.

    Args:
        problem: The MBPPProblem to wrap.
        n_example_tests: How many tests to surface in the prompt.

    Returns:
        A Task ready to be passed to the verifier or any ISOPro environment.
    """
    return Task.make(
        prompt=render_mbpp_prompt(problem, n_example_tests=n_example_tests),
        ground_truth=problem.test_list,
        difficulty=_TIER_CONFIGS[problem.tier].difficulty,
        category=problem.tier.value,
        metadata={
            "task_id": problem.task_id,
            "test_imports": problem.test_imports,
            "test_list": problem.test_list,
            "tier": problem.tier.value,
            "body_lines": problem.body_lines,
        },
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def build_mbpp_splits(
    problems: list[MBPPProblem] | None = None,
    eval_per_tier: int = 10,
    seed: int = 42,
) -> dict[str, dict[MBPPTier, list[MBPPProblem]]]:
    """Split MBPP problems into a training pool and a per-tier eval set.

    The held-out tier (T4) does *not* appear in the training pool — it is
    reserved entirely for compositional-generalization evaluation, mirroring
    the scheduling T5 split in the GCE paper.

    Args:
        problems: Optional pre-loaded list. If None, loads MBPP from cache.
        eval_per_tier: How many problems per tier to reserve for evaluation.
        seed: Reproducibility seed for the eval-set sampler.

    Returns:
        Dict with keys "train" and "eval", each mapping MBPPTier to a list
        of MBPPProblem.
    """
    import random

    if problems is None:
        problems = load_mbpp_problems()

    rng = random.Random(seed)
    by_tier: dict[MBPPTier, list[MBPPProblem]] = {t: [] for t in MBPPTier}
    for p in problems:
        by_tier[p.tier].append(p)

    train: dict[MBPPTier, list[MBPPProblem]] = {}
    evalset: dict[MBPPTier, list[MBPPProblem]] = {}

    for tier, items in by_tier.items():
        rng.shuffle(items)
        if tier == MBPPTier.HELD_OUT:
            train[tier] = []
            evalset[tier] = items[: min(eval_per_tier, len(items))]
        else:
            evalset[tier] = items[: min(eval_per_tier, len(items))]
            train[tier] = items[min(eval_per_tier, len(items)) :]

    return {"train": train, "eval": evalset}
