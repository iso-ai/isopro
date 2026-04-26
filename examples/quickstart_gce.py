"""ISOPro quickstart — see Grounded Continuous Evaluation in 60 seconds.

This script demonstrates the core architectural claim of the GCE paper without
requiring a GPU, a model download, or a reward model:

    The verifier IS the reward signal.

We:
  1. Generate a real resource-constrained scheduling problem (RCPSP) and solve
     it with OR-Tools CP-SAT to get ground truth.
  2. Run three "model" responses through the deterministic verifier:
     a perfect schedule, a constraint-violating schedule, and a hallucinated
     plausible-but-wrong schedule.
  3. Show that the verifier returns a clean binary signal in every case —
     no preference judgments, no learned reward, no reward hacking surface.

Wall-clock: ~5 seconds. Memory: <100 MB. Hardware: any laptop.

To then plug a real LLM into this loop, see:
    examples/run_scheduling_experiment.py   (the paper's main experiment)
    examples/custom_verifier.py             (extend ISOPro to your domain)

Run:
    python examples/quickstart_gce.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — import ISOPro submodules directly to avoid the heavyweight
# top-level __init__ (transformers/torch import for a 5-second demo would be
# overkill).
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

for _pkg, _path in [
    ("isopro", "isopro"),
    ("isopro.environments", "isopro/environments"),
    ("isopro.environments.tasks", "isopro/environments/tasks"),
]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [str(_ROOT / _path)]
    sys.modules[_pkg] = _m

from isopro.environments.tasks.scheduling_tasks import (
    SchedulingTier,
    generate_scheduling_task,
)
from isopro.environments.tasks.scheduling_verifier import score_scheduling_task


def _banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def main() -> None:
    _banner("ISOPro quickstart — the verifier is the reward signal")

    # ------------------------------------------------------------------
    # 1. Generate a ground-truth-solved scheduling problem
    # ------------------------------------------------------------------
    task = generate_scheduling_task(SchedulingTier.SEQUENCING, seed=42)
    problem = task.metadata["problem"]
    oracle = task.metadata["solution"]["start_times"]

    print("\nProblem (Tier 1 — sequencing with precedence dependencies):\n")
    print(task.prompt[:600] + ("\n  ... [truncated]" if len(task.prompt) > 600 else ""))
    print("\nOracle solution from OR-Tools CP-SAT:")
    print("  " + json.dumps(oracle, indent=2).replace("\n", "\n  "))

    # ------------------------------------------------------------------
    # 2. Score three different "model" responses
    # ------------------------------------------------------------------
    correct_response = "\n".join(f"{job}: {t}" for job, t in oracle.items())

    # Mutilate the oracle: bump every dependent job earlier to violate precedence
    jobs = list(oracle.keys())
    bad = {job: 0 for job in jobs}
    violating_response = "\n".join(f"{job}: {t}" for job, t in bad.items())

    plausible_hallucination = (
        "Looking at the precedence graph, I'll schedule everything sequentially "
        "starting from time zero.\n"
        + "\n".join(f"{job}: {i}" for i, job in enumerate(jobs))
    )

    candidates = {
        "Oracle (perfect schedule)": correct_response,
        "Constraint-violating schedule": violating_response,
        "Plausible-sounding hallucination": plausible_hallucination,
    }

    _banner("Three responses → the deterministic verifier")
    print(f"\n{'Response':40s}  {'Score':>6s}  Failure modes")
    print("-" * 72)
    for label, response in candidates.items():
        score, detail = score_scheduling_task(task, response)
        failures = ", ".join(detail["failures"][:2]) if detail["failures"] else "—"
        print(f"{label:40s}  {score:>6.1f}  {failures[:30]}")

    # ------------------------------------------------------------------
    # 3. The architectural claim
    # ------------------------------------------------------------------
    _banner("Why this matters")
    print(
        """
Standard RLHF replaces this verifier with a *learned* reward model trained on
human preference pairs. That reward model:

  - is evaluated under conditions that differ from RL training inputs
    (distributional invalidity);
  - is evaluated once before RL begins and not monitored thereafter
    (temporal invalidity);
  - cannot assess multi-step trajectory coherence
    (scope invalidity);
  - rewards plausible-looking outputs over correct reasoning
    (process invalidity).

These four validity failures are why reward hacking happens. They compound at
the architectural level — the policy finds the reward model's blind spots and
produces fluent, confident, wrong outputs that score well.

ISOPro replaces the learned reward with a deterministic verifier (above).
Reward hacking is not mitigated; it is eliminated by construction. The same
pattern that DeepSeek-R1's GRPO uses at 671B parameters, ISOPro implements on
a 3B model on a laptop.

Next steps:
  - Train a real LLM with this loop:        examples/run_scheduling_experiment.py
  - Plug in your own domain's verifier:     examples/custom_verifier.py
  - Watch the implicit curriculum form:     examples/watch_curriculum_emerge.py
"""
    )


if __name__ == "__main__":
    main()
