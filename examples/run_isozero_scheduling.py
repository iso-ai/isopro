#!/usr/bin/env python3
"""IsoZero scheduling evaluation — simulation loop, no weight updates.

Uses the isozero library's ReasonSimulation wrapper to give LLaMA 3.1
structured multi-step reasoning on scheduling problems. The model
reasons step-by-step through the simulation, then the deterministic
verifier checks the final answer.

No LoRA. No training. No HuggingFace. Pure Ollama inference through
the isozero simulation framework.

This establishes what the simulation-based evaluation framework alone
can achieve, before ISOPro adds LoRA training on correct traces.

Usage:
    python examples/run_isozero_scheduling.py
    python examples/run_isozero_scheduling.py --eval-problems 5 --max-steps 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Package bootstrap
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

for _pkg, _path in [
    ("isopro", "isopro"),
    ("isopro.environments", "isopro/environments"),
    ("isopro.environments.tasks", "isopro/environments/tasks"),
    ("isopro.backends", "isopro/backends"),
]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [str(_ROOT / _path)]
    _m.__package__ = _pkg
    sys.modules[_pkg] = _m

from isopro.backends.base import GenerationConfig
from isopro.backends.ollama_backend import OllamaBackend
from isopro.environments.tasks.scheduling_tasks import (
    SchedulingTier,
    build_eval_set,
)
from isopro.environments.tasks.scheduling_verifier import (
    score_scheduling_task,
    verify_schedule,
)
from isopro.environments.tasks.base_task import Task

from isozero.reason_sim import ReasonSimulation, ReasonSimulationWrapper

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Ollama agent compatible with isozero's agent interface
# ---------------------------------------------------------------------------


class OllamaSchedulingAgent:
    """Agent that uses Ollama for multi-step scheduling reasoning.

    Compatible with isozero's ReasonSimulationWrapper — implements the
    run() method that takes agent_input and returns structured output.

    Args:
        backend: OllamaBackend instance.
        gen_config: GenerationConfig for Ollama calls.
    """

    def __init__(
        self,
        backend: OllamaBackend,
        gen_config: GenerationConfig,
    ) -> None:
        self._backend = backend
        self._gen_config = gen_config

    def run(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one reasoning step via Ollama.

        Args:
            agent_input: Dict with 'text' key containing task, step,
                reasoning history, and max_steps.

        Returns:
            Dict with 'text_data' containing either a reasoning step
            or the final solution.
        """
        text_data = agent_input["text"]
        task = text_data["task"]
        step = text_data["step"]
        reasoning = text_data["reasoning"]
        max_steps = text_data["max_steps"]

        prompt = self._build_prompt(task, step, reasoning, max_steps)
        result = self._backend.generate(prompt, self._gen_config)
        return self._parse_response(result.text, step, max_steps)

    def _build_prompt(
        self,
        task: str,
        step: int,
        reasoning: List[str],
        max_steps: int,
    ) -> str:
        """Build a step-appropriate prompt for the scheduling task.

        Args:
            task: The scheduling problem description.
            step: Current step number (0-indexed).
            reasoning: List of previous reasoning steps.
            max_steps: Total reasoning steps before final answer.

        Returns:
            Prompt string.
        """
        if step == 0:
            return (
                f"{task}\n\n"
                f"Step 1 of {max_steps}: Analyze the constraints. "
                "Identify all dependencies, resource limits, and deadlines. "
                "List which jobs depend on which, and what the binding "
                "constraints are."
            )
        elif step < max_steps - 1:
            prev = "\n".join(
                f"Step {i+1}: {r}" for i, r in enumerate(reasoning)
            )
            return (
                f"{task}\n\n"
                f"Previous analysis:\n{prev}\n\n"
                f"Step {step+1} of {max_steps}: Based on your analysis, "
                "assign tentative start times for each job. Check each "
                "constraint — does any job violate a dependency, exceed "
                "a resource limit, or miss a deadline? If so, adjust."
            )
        else:
            prev = "\n".join(
                f"Step {i+1}: {r}" for i, r in enumerate(reasoning)
            )
            return (
                f"{task}\n\n"
                f"Previous analysis:\n{prev}\n\n"
                f"Final step: Provide your final answer as start times "
                "for each job in the format: A: 0, B: 3, C: 5\n"
                "List ALL jobs with their start times."
            )

    def _parse_response(
        self,
        response: str,
        step: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        """Parse Ollama response into isozero simulation input format.

        Args:
            response: Raw text from Ollama.
            step: Current step.
            max_steps: Total steps.

        Returns:
            Dict compatible with ReasonSimulation.step().
        """
        if step < max_steps - 1:
            return {"text_data": {"reasoning": response}}
        else:
            return {"text_data": {"solution": response}}


# ---------------------------------------------------------------------------
# Run one problem through the isozero simulation
# ---------------------------------------------------------------------------


def run_isozero_on_task(
    agent: OllamaSchedulingAgent,
    task: Task,
    max_steps: int = 3,
) -> dict:
    """Run a scheduling task through the isozero simulation loop.

    Args:
        agent: OllamaSchedulingAgent.
        task: Scheduling Task.
        max_steps: Number of reasoning steps before final answer.

    Returns:
        Dict with solution text, verification result, and reasoning trace.
    """
    sim = ReasonSimulation(
        reason=task.prompt,
        max_steps=max_steps,
    )
    wrapper = ReasonSimulationWrapper(agent, sim)

    # Run through all steps
    for step in range(max_steps):
        state = wrapper.step()

    # Extract the final solution
    solution_text = state["text_data"].get("solution", "")
    reasoning_trace = state["text_data"].get("reasoning", [])

    # Verify
    problem_dict = task.metadata["problem"]
    vr = verify_schedule(solution_text, problem_dict)

    wrapper.close()

    return {
        "solution": solution_text,
        "passed": vr.passed,
        "parsed_schedule": vr.parsed_schedule,
        "parse_strategy": vr.parse_strategy,
        "failures": vr.failures,
        "checks": vr.checks,
        "reasoning_trace": reasoning_trace,
        "n_reasoning_steps": len(reasoning_trace),
    }


# ---------------------------------------------------------------------------
# Evaluate across all tiers
# ---------------------------------------------------------------------------


def evaluate_isozero(
    agent: OllamaSchedulingAgent,
    eval_sets: dict[str, list[Task]],
    max_steps: int = 3,
) -> dict[str, dict]:
    """Run isozero simulation evaluation across all tiers.

    Args:
        agent: OllamaSchedulingAgent.
        eval_sets: Dict of tier_name -> list of Tasks.
        max_steps: Reasoning steps per problem.

    Returns:
        Dict of tier_name -> metrics dict.
    """
    results: dict[str, dict] = {}

    for tier_name, tasks in sorted(eval_sets.items()):
        if not tasks:
            results[tier_name] = {"accuracy": 0.0, "n": 0}
            continue

        correct = 0
        parse_fails = 0
        details = []

        for i, task in enumerate(tasks, 1):
            with console.status(
                f"[dim]IsoZero {tier_name} [{i}/{len(tasks)}][/dim]",
                spinner="dots",
            ):
                result = run_isozero_on_task(agent, task, max_steps)

            if result["passed"]:
                correct += 1
            if not result["checks"].get("parsing", False):
                parse_fails += 1

            details.append(result)

        n = len(tasks)
        results[tier_name] = {
            "accuracy": round(correct / n, 4),
            "n": n,
            "correct": correct,
            "parse_fails": parse_fails,
            "details": details,
        }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_isozero_results(results: dict[str, dict]) -> None:
    """Display isozero evaluation results."""
    console.print()
    console.print(Rule("[bold cyan]IsoZero Simulation Results[/bold cyan]"))

    for tier_name in sorted(results.keys()):
        r = results[tier_name]
        acc = r["accuracy"]
        bar_w = 30
        filled = int(acc * bar_w)
        color = (
            "bright_green" if acc >= 0.75
            else "yellow" if acc >= 0.5
            else "red"
        )
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (bar_w - filled))
        bar.append(f"  {acc*100:5.1f}%", style="bold")

        short = tier_name.replace("tier", "T").replace("_", " ")
        console.print(f"  {short:35s}", bar)

    n = len(results)
    if n:
        overall = sum(r["accuracy"] for r in results.values()) / n
        console.print(
            f"\n  [bold]Mean accuracy[/bold]  "
            f"[bold cyan]{overall*100:.1f}%[/bold cyan]"
        )
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="IsoZero scheduling evaluation (simulation, no training)"
    )
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--eval-problems", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="data/scheduling_experiment"
    )
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold]IsoZero Scheduling Evaluation[/bold]\n"
        f"Model: {args.ollama_model}  |  Steps: {args.max_steps}  |  "
        f"Seed: {args.seed}\n"
        "No training — simulation + verification only",
        style="cyan",
    ))

    # Build eval sets
    console.print(Rule("[bold]Building Evaluation Sets[/bold]"))
    n_pw = max(1, args.eval_problems // 2 + 1)
    eval_sets = build_eval_set(
        n_per_tier=args.eval_problems,
        n_per_pairwise_subtype=n_pw,
        base_seed=9999,
    )
    total_eval = sum(len(t) for t in eval_sets.values())
    for tier_name, tasks in sorted(eval_sets.items()):
        console.print(f"  {tier_name}: {len(tasks)} problems")
    console.print(f"  [bold]Total: {total_eval} problems[/bold]")

    # Connect
    console.print()
    ollama = OllamaBackend(args.ollama_model)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    agent = OllamaSchedulingAgent(ollama, gen_cfg)
    console.print(f"  Ollama connected: {args.ollama_model}")

    # Run
    console.print()
    console.print(Rule("[bold]Running IsoZero Simulation[/bold]"))
    t0 = time.perf_counter()
    results = evaluate_isozero(agent, eval_sets, max_steps=args.max_steps)
    wall_time = time.perf_counter() - t0

    print_isozero_results(results)
    console.print(f"  Wall-clock: {wall_time:.1f}s")

    # Save
    output_path = Path(args.output_dir) / "isozero_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable details for JSON
    json_results = {}
    for tier_name, r in results.items():
        json_results[tier_name] = {
            "accuracy": r["accuracy"],
            "n": r["n"],
            "correct": r["correct"],
            "parse_fails": r["parse_fails"],
        }

    payload = {
        "experiment": "isozero_simulation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.ollama_model,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "wall_clock_s": round(wall_time, 2),
        "tier_results": json_results,
        "mean_accuracy": round(
            sum(r["accuracy"] for r in results.values()) / len(results), 4
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    console.print(f"\n  [green]Results saved -> [cyan]{output_path}[/cyan]")

    console.print()
    console.print(Panel(
        f"[bold green]IsoZero Evaluation Complete[/bold green]\n"
        f"Time: {wall_time/60:.1f} minutes  |  "
        f"Mean: {payload['mean_accuracy']*100:.1f}%",
        style="green",
    ))


if __name__ == "__main__":
    main()
