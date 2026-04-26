"""Plug your own deterministic verifier into ISOPro — under 100 lines.

ISOPro's killer feature is that the *only thing standing between your domain
and a verifier-grounded fine-tuning loop* is a Python function:

    def score(task, response) -> tuple[float, dict]: ...

If you can write that function, you get:
  - rejection sampling without a learned reward model;
  - implicit curriculum without researcher curation;
  - reward hacking eliminated by construction.

This script shows the full pattern on a tiny made-up domain — *modular
arithmetic word problems* — then runs three candidate responses through the
verifier you wrote. No model, no GPU, no API keys.

When you're ready to swap in a real LLM, the same `score()` function plugs
straight into RejectionSamplingTrainer (see examples/run_scheduling_experiment.py).

Run:
    python examples/custom_verifier.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

# Bootstrap so we can import the lightweight base task without dragging in
# torch/transformers for a 2-second demo.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import types as _types
for _pkg, _path in [
    ("isopro", "isopro"),
    ("isopro.environments", "isopro/environments"),
    ("isopro.environments.tasks", "isopro/environments/tasks"),
]:
    _m = _types.ModuleType(_pkg)
    _m.__path__ = [str(_ROOT / _path)]
    sys.modules[_pkg] = _m

from isopro.environments.tasks.base_task import DifficultyLevel, Task


# ---------------------------------------------------------------------------
# 1. Generate a task with a deterministic ground-truth answer
# ---------------------------------------------------------------------------


def generate_modular_task(seed: int) -> Task:
    """Generate one (a + b) mod m problem with the answer baked into the Task.

    Args:
        seed: Reproducibility seed.

    Returns:
        Task whose ground_truth is the integer answer.
    """
    rng = random.Random(seed)
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    m = rng.choice([7, 11, 13])
    answer = (a + b) % m

    prompt = (
        f"Compute ({a} + {b}) mod {m}. "
        f"Show your reasoning, then output the final answer on the last line "
        f"in the form 'Answer: <integer>'."
    )
    return Task.make(
        prompt=prompt,
        ground_truth=answer,
        difficulty=DifficultyLevel.EASY,
        category=f"mod{m}",
        metadata={"a": a, "b": b, "m": m},
    )


# ---------------------------------------------------------------------------
# 2. Write the verifier — this is the WHOLE integration surface
# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"answer\s*[:=]\s*(-?\d+)", re.IGNORECASE)


def score_modular_task(task: Task, response: str) -> tuple[float, dict]:
    """Deterministic verifier for modular-arithmetic responses.

    Two-stage, just like the scheduling verifier:
      1. Parse: pull an integer out of "Answer: N".
      2. Check:  compare to the ground-truth integer.

    Args:
        task: A Task whose ground_truth is the correct integer.
        response: Raw text from the model.

    Returns:
        (1.0, detail) on a correct answer; (0.0, detail) otherwise.
    """
    match = _ANSWER_RE.search(response)
    if not match:
        return 0.0, {"passed": False, "reason": "no 'Answer: <n>' line found"}

    parsed = int(match.group(1))
    correct = parsed == task.ground_truth
    return (
        1.0 if correct else 0.0,
        {
            "passed": correct,
            "parsed_answer": parsed,
            "ground_truth": task.ground_truth,
        },
    )


# ---------------------------------------------------------------------------
# 3. Drive a few candidate responses through the verifier
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("Custom verifier on a fresh domain — modular arithmetic")
    print("=" * 72)

    task = generate_modular_task(seed=7)
    a, b, m = task.metadata["a"], task.metadata["b"], task.metadata["m"]

    print(f"\nTask: ({a} + {b}) mod {m}")
    print(f"Ground truth: {task.ground_truth}\n")

    candidates = {
        "Correct + reasoning": (
            f"({a} + {b}) = {a + b}. "
            f"{a + b} mod {m} = {task.ground_truth}.\n"
            f"Answer: {task.ground_truth}"
        ),
        "Off-by-one": (
            f"Answer: {task.ground_truth + 1}"
        ),
        "Plausible-sounding hallucination": (
            f"By Fermat's little theorem the answer is clearly small, around "
            f"{m // 2}.\nAnswer: {m // 2}"
        ),
        "No answer line": (
            "I think the answer is somewhere between 0 and m. Hard to be sure."
        ),
    }

    print(f"{'Response':40s}  {'Score':>6s}  Detail")
    print("-" * 72)
    for label, response in candidates.items():
        score, detail = score_modular_task(task, response)
        if "parsed_answer" in detail:
            note = f"parsed={detail['parsed_answer']}, truth={detail['ground_truth']}"
        else:
            note = detail["reason"]
        print(f"{label:40s}  {score:>6.1f}  {note}")

    print(
        """
Wiring this into ISOPro training (sketch):

    from isopro.training.rejection_sampling_trainer import (
        RejectionSamplingTrainer, RSTrainingConfig,
    )

    class ModularEnv:
        def generate_task(self): return generate_modular_task(seed=...)
        def score(self, task, response): return score_modular_task(task, response)

    trainer = RejectionSamplingTrainer(
        env=ModularEnv(),
        backend=my_huggingface_lora_backend,
        config=RSTrainingConfig(n_iterations=10, k_samples=8),
    )
    log = trainer.train()

That's the whole integration. Same loop, same architecture, your domain.
The four GCE validity failures cannot occur because there is no learned reward
model to invalidate.
"""
    )


if __name__ == "__main__":
    main()
