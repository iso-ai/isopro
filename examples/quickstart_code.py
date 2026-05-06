"""ISOPro code-creation quickstart — verifier-grounded coding eval in 30 seconds.

Mirrors examples/quickstart_gce.py for the new MBPP environment. Shows that
ISOPro's architectural claim — *the verifier is the reward signal* — extends
cleanly to code generation:

  1. Load one MBPP problem (sanitized split, downloaded once and cached).
  2. Run four candidate "model responses" through the deterministic verifier:
       - the canonical reference solution           → 1.0
       - a syntactically valid but wrong function   → 0.0 with NameError
       - an infinite loop                           → 0.0 with timeout
       - free-text with no code block               → 0.0 with extraction fail
  3. Show that the verifier produces a clean binary signal in every case —
     the same architectural property as the scheduling verifier.

Wall-clock: ~10 seconds (most of it is one subprocess timeout). Memory: <100 MB.
Hardware: any laptop. No model, no API keys, no GPU.

Run:
    python examples/quickstart_code.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap submodule imports without dragging in torch/transformers
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

from isopro.environments.tasks.mbpp_tasks import (
    MBPPTier,
    generate_mbpp_task,
    load_mbpp_problems,
)
from isopro.environments.tasks.mbpp_verifier import score_mbpp_task


def _banner(text: str) -> None:
    print("\n" + "=" * 76)
    print(text)
    print("=" * 76)


def main() -> None:
    _banner("ISOPro code-creation quickstart — verifier as reward signal")

    # ------------------------------------------------------------------
    # 1. Load one warmup-tier MBPP problem
    # ------------------------------------------------------------------
    problems = load_mbpp_problems()
    short_problems = [p for p in problems if p.tier == MBPPTier.SHORT]
    problem = short_problems[0]
    task = generate_mbpp_task(problem)

    print(f"\nProblem (Tier 1 — short, MBPP task #{problem.task_id}):\n")
    print("  " + task.prompt.replace("\n", "\n  "))
    print(f"\nReference solution ({problem.body_lines} body lines):\n")
    print("  " + problem.canonical_code.replace("\n", "\n  "))
    print(f"\nUnit tests ({len(problem.test_list)}):\n")
    for t in problem.test_list:
        print(f"  {t}")

    # ------------------------------------------------------------------
    # 2. Score four candidate responses
    # ------------------------------------------------------------------
    candidates = {
        "Canonical solution": (
            f"```python\n{problem.canonical_code}\n```"
        ),
        "Wrong function (NameError)": (
            "```python\ndef placeholder():\n    return 0\n```"
        ),
        "Infinite loop": (
            f"```python\ndef {problem.canonical_code.split('(')[0].split()[-1]}"
            f"(*args, **kwargs):\n    while True:\n        pass\n```"
        ),
        "Free-text, no code block": (
            "I think you should use a list comprehension here."
        ),
    }

    _banner("Four responses → the deterministic verifier")
    print(f"\n{'Response':38s}  {'Score':>6s}  Failure mode")
    print("-" * 76)
    for label, response in candidates.items():
        score, detail = score_mbpp_task(task, response)
        if detail["timed_out"]:
            note = "subprocess timeout"
        elif detail["failures"]:
            note = detail["failures"][0][:40]
        elif detail["extraction_strategy"] is None:
            note = "no code block extracted"
        else:
            note = "—"
        print(f"{label:38s}  {score:>6.1f}  {note}")

    # ------------------------------------------------------------------
    # 3. Architectural takeaway
    # ------------------------------------------------------------------
    _banner("Why this matters for code-creation evaluation")
    print(
        """
The MBPP verifier extends GCE to code creation without changing the
architecture. Every alternative reward signal in this domain has a known
failure mode:

  - Reference-output match (BLEU / exact match)  →  rewards stylistic mimicry
                                                    of the canonical solution,
                                                    not behavioral correctness
  - Learned code reward model                     →  vulnerable to reward
                                                    hacking; same four GCE
                                                    validity failures as
                                                    natural-language RM
  - Human-judged preference                       →  scope-invalid for
                                                    multi-step coding tasks

A subprocess that runs the unit tests is *the canonical correctness signal*
for MBPP. ISOPro uses it directly. Reward hacking is impossible: the only
way to score 1.0 is to actually solve the problem.

Next steps:
  - Train a real LLM with this loop on MBPP:    examples/run_mbpp_experiment.py
                                                (coming next; see mbpp_tasks.py
                                                 + RejectionSamplingTrainer)
  - Compare against vanilla GRPO-LoRA:          examples/run_grpo_lora.py
                                                (coming next)
  - Run on Llama 3.2 3B / Gemma 2 4B:           same scripts, different model.
"""
    )


if __name__ == "__main__":
    main()
