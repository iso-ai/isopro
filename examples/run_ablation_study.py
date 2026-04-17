#!/usr/bin/env /opt/homebrew/bin/python3.11
"""Ablation study + multi-seed evaluation for the ISOPro scheduling experiment.

Runs the ISOPro training loop under four conditions to isolate component
contributions, each with multiple seeds for statistical significance:

  1. Full ISOPro — all components enabled (baseline for ablation)
  2. No chain-of-thought — removes structured reasoning instructions
  3. No buffer accumulation — trains only on current iteration's traces
  4. Random LoRA layers — targets random layers instead of activation-guided

Each condition runs with 3 seeds (42, 123, 456). Reports mean ± std
per tier for the paper's ablation table.

Usage:
    /opt/homebrew/bin/python3.11 examples/run_ablation_study.py
    /opt/homebrew/bin/python3.11 examples/run_ablation_study.py --seeds 42 123 456 789 1337
    /opt/homebrew/bin/python3.11 examples/run_ablation_study.py --ablations full no_cot
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

for _pkg, _path in [
    ("isopro", "isopro"),
    ("isopro.environments", "isopro/environments"),
    ("isopro.environments.tasks", "isopro/environments/tasks"),
    ("isopro.backends", "isopro/backends"),
    ("isopro.training", "isopro/training"),
]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [str(_ROOT / _path)]
    _m.__package__ = _pkg
    sys.modules[_pkg] = _m

from isopro.environments.tasks.scheduling_tasks import (
    PAIRWISE_SUBTYPES,
    SchedulingTier,
    build_eval_set,
    build_tier_task_bank,
    render_problem_prompt,
)
from isopro.environments.tasks.scheduling_verifier import score_scheduling_task
from isopro.environments.tasks.base_task import Task
from isopro.training.replay_buffer import ImplicitCurriculumBuffer

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx_lm import load, generate
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.sample_utils import make_sampler

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console(highlight=False)

TRAINING_TIERS = [
    SchedulingTier.WARMUP,
    SchedulingTier.SEQUENCING,
    SchedulingTier.RESOURCE_ALLOC,
    SchedulingTier.DEADLINE_PRESSURE,
    SchedulingTier.PAIRWISE,
]

ALL_TIER_NAMES = [t.value for t in SchedulingTier]


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Configuration for one ablation condition.

    Attributes:
        name: Human-readable name.
        use_cot: Whether to include chain-of-thought in prompts.
        accumulate_buffer: Whether to accumulate traces across iterations.
        use_activation_layers: Whether to use activation-guided LoRA targeting.
        description: What this ablation tests.
    """

    name: str
    use_cot: bool = True
    accumulate_buffer: bool = True
    use_activation_layers: bool = True
    description: str = ""


ABLATIONS = {
    "full": AblationConfig(
        name="Full ISOPro",
        use_cot=True,
        accumulate_buffer=True,
        use_activation_layers=True,
        description="All components enabled (baseline)",
    ),
    "no_cot": AblationConfig(
        name="No Chain-of-Thought",
        use_cot=False,
        accumulate_buffer=True,
        use_activation_layers=True,
        description="Removes structured reasoning from prompts",
    ),
    "no_accumulation": AblationConfig(
        name="No Buffer Accumulation",
        use_cot=True,
        accumulate_buffer=False,
        use_activation_layers=True,
        description="Trains only on current iteration's traces (no implicit curriculum)",
    ),
    "random_layers": AblationConfig(
        name="Random LoRA Layers",
        use_cot=True,
        accumulate_buffer=True,
        use_activation_layers=False,
        description="Targets random layers instead of activation-guided top-K",
    ),
}


# ---------------------------------------------------------------------------
# MLX helpers
# ---------------------------------------------------------------------------


def mlx_generate(model, tokenizer, prompt: str, max_tokens: int, temp: float) -> str:
    """Generate via MLX."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temp)
    return generate(
        model, tokenizer, prompt=formatted,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
    )


def strip_cot(task: Task) -> Task:
    """Return a copy of the task with chain-of-thought removed from prompt.

    Args:
        task: Original task with CoT.

    Returns:
        New task with CoT lines stripped.
    """
    lines = task.prompt.split("\n")
    filtered = []
    skip = False
    for line in lines:
        if line.strip() == "Think step by step:":
            skip = True
            continue
        if skip and line.strip() and line[0].isdigit() and "." in line[:3]:
            continue
        skip = False
        filtered.append(line)

    return Task(
        task_id=task.task_id,
        prompt="\n".join(filtered),
        ground_truth=task.ground_truth,
        difficulty=task.difficulty,
        category=task.category,
        metadata=task.metadata,
    )


# ---------------------------------------------------------------------------
# Training loop (parameterized by ablation config)
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single training run."""

    ablation: str
    seed: int
    baseline_accs: dict[str, float]
    final_accs: dict[str, float]
    loss_curve: list[float]
    hit_rate_curve: list[float]
    buffer_sizes: list[int]
    buffer_composition_final: dict[str, int]
    transition_points: dict[str, int]
    wall_clock_s: float


def run_single(
    ablation_config: AblationConfig,
    seed: int,
    model_name: str,
    iterations: int,
    k_samples: int,
    tasks_per_tier: int,
    eval_problems: int,
    max_tokens: int,
    lora_layers: int,
    lora_rank: int,
    lr: float,
) -> RunResult:
    """Run one training loop with a specific ablation config and seed.

    Args:
        ablation_config: Which components are enabled/disabled.
        seed: Random seed.
        model_name: HuggingFace model ID.
        iterations: Training iterations.
        k_samples: Rollouts per task.
        tasks_per_tier: Tasks generated per tier per iteration.
        eval_problems: Eval problems per tier.
        max_tokens: Max generation tokens.
        lora_layers: Number of LoRA layers.
        lora_rank: LoRA rank.
        lr: Learning rate.

    Returns:
        RunResult with all metrics.
    """
    ab = ablation_config
    console.print(f"\n  [bold]{ab.name}[/bold] | seed={seed}")

    # Load fresh model each run
    model, tokenizer = load(model_name)
    model.freeze()

    # LoRA layer selection
    num_model_layers = len(model.model.layers)
    if ab.use_activation_layers:
        # Activation-guided: top-K layers (last K)
        target_layers = list(range(num_model_layers - lora_layers, num_model_layers))
    else:
        # Random layers
        rng_layers = random.Random(seed + 999)
        target_layers = sorted(rng_layers.sample(range(num_model_layers), lora_layers))

    console.print(f"    LoRA layers: {target_layers}")

    lora_config = {
        "rank": lora_rank,
        "alpha": lora_rank * 2,
        "dropout": 0.05,
        "scale": 2.0,
    }
    linear_to_lora_layers(model, num_layers=lora_layers, config=lora_config)
    optimizer = opt.Adam(learning_rate=lr)

    # Build eval sets
    n_pw = max(1, eval_problems // 2 + 1)
    eval_sets = build_eval_set(
        n_per_tier=eval_problems,
        n_per_pairwise_subtype=n_pw,
        base_seed=9999,
    )

    # Baseline eval
    baseline_accs = _evaluate(model, tokenizer, eval_sets, max_tokens)
    console.print(f"    Baseline: {_mean_acc(baseline_accs):.1f}%")

    # Training loop
    rng = random.Random(seed)
    buffer = ImplicitCurriculumBuffer()
    loss_curve = []
    hit_rate_curve = []
    buffer_sizes = []

    run_start = time.perf_counter()

    for iteration in range(1, iterations + 1):
        # Generate tasks
        iter_tasks = []
        for tier in TRAINING_TIERS:
            tier_seed = seed + iteration * 1000 + hash(tier.value) % (2**31)
            n = tasks_per_tier
            if tier == SchedulingTier.PAIRWISE:
                tasks = build_tier_task_bank(tier, n, tier_seed, subtypes=PAIRWISE_SUBTYPES)
            else:
                tasks = build_tier_task_bank(tier, n, tier_seed)
            iter_tasks.extend(tasks)
        rng.shuffle(iter_tasks)

        # Optionally strip CoT
        if not ab.use_cot:
            iter_tasks = [strip_cot(t) for t in iter_tasks]

        # Generate rollouts
        rollouts = []
        for task in iter_tasks:
            for _ in range(k_samples):
                resp = mlx_generate(model, tokenizer, task.prompt, max_tokens, temp=0.8)
                reward, _ = score_scheduling_task(task, resp)
                rollouts.append({
                    "prompt": task.prompt,
                    "response": resp,
                    "score": reward,
                    "category": task.category,
                })

        correct = [r for r in rollouts if r["score"] == 1.0]
        hit_rate = len(correct) / len(rollouts) * 100 if rollouts else 0
        hit_rate_curve.append(hit_rate)

        # Add to buffer
        snapshot = buffer.add(
            correct_rollouts=correct,
            iteration=iteration,
            n_total_rollouts=len(rollouts),
        )
        buffer_sizes.append(buffer.size)

        # Train
        if ab.accumulate_buffer:
            train_traces = buffer.training_pairs(shuffle=True, seed=seed + iteration)
        else:
            # Only current iteration's correct traces
            train_traces = [(r["prompt"], r["response"]) for r in correct]

        train_loss = None
        if train_traces:
            train_loss = _train_epoch(model, tokenizer, optimizer, train_traces)
            loss_curve.append(train_loss)

        console.print(
            f"    Iter {iteration}: {len(correct)}/{len(rollouts)} correct "
            f"({hit_rate:.0f}%) | buffer={buffer.size} | "
            f"loss={train_loss:.4f}" if train_loss else
            f"    Iter {iteration}: {len(correct)}/{len(rollouts)} correct "
            f"({hit_rate:.0f}%) | buffer={buffer.size} | no training"
        )

    wall_time = time.perf_counter() - run_start

    # Final eval
    eval_sets_final = eval_sets
    if not ab.use_cot:
        eval_sets_final = {
            k: [strip_cot(t) for t in v] for k, v in eval_sets.items()
        }
    final_accs = _evaluate(model, tokenizer, eval_sets_final, max_tokens)
    console.print(f"    Final: {_mean_acc(final_accs):.1f}% | time={wall_time/60:.1f}min")

    return RunResult(
        ablation=ablation_config.name,
        seed=seed,
        baseline_accs=baseline_accs,
        final_accs=final_accs,
        loss_curve=loss_curve,
        hit_rate_curve=hit_rate_curve,
        buffer_sizes=buffer_sizes,
        buffer_composition_final=buffer.composition(),
        transition_points=buffer.transition_points(),
        wall_clock_s=round(wall_time, 2),
    )


def _evaluate(model, tokenizer, eval_sets, max_tokens) -> dict[str, float]:
    """Evaluate per-tier accuracy."""
    results = {}
    for tier_name, tasks in sorted(eval_sets.items()):
        if not tasks:
            results[tier_name] = 0.0
            continue
        correct = 0
        for task in tasks:
            resp = mlx_generate(model, tokenizer, task.prompt, max_tokens, temp=0.1)
            reward, _ = score_scheduling_task(task, resp)
            if reward == 1.0:
                correct += 1
        results[tier_name] = round(correct / len(tasks), 4)
    return results


def _mean_acc(accs: dict[str, float]) -> float:
    """Mean accuracy across tiers."""
    return sum(accs.values()) / len(accs) * 100 if accs else 0


def _train_epoch(model, tokenizer, optimizer, pairs, batch_size=4, max_length=512):
    """One SFT epoch on (prompt, response) pairs via MLX."""
    shuffled = list(pairs)
    random.shuffle(shuffled)
    total_loss = 0.0
    n_steps = 0

    def loss_fn(model, input_ids, prompt_len):
        logits = model(input_ids)
        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        seq_len = shift_labels.shape[1]
        mask = mx.arange(seq_len) >= (prompt_len - 1)
        mask = mask.astype(mx.float32)
        token_losses = nn.losses.cross_entropy(shift_logits, shift_labels)
        return (token_losses * mask).sum() / mx.maximum(mask.sum(), 1)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    for prompt, response in shuffled:
        full_text = prompt + "\n" + response
        prompt_tokens = tokenizer.encode(prompt + "\n")
        full_tokens = tokenizer.encode(full_text)
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[:max_length]

        input_ids = mx.array([full_tokens])
        prompt_len = min(len(prompt_tokens), len(full_tokens) - 1)

        loss, grads = loss_grad_fn(model, input_ids, prompt_len)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------


def aggregate_results(all_results: list[RunResult]) -> dict:
    """Aggregate results across seeds per ablation.

    Args:
        all_results: List of all RunResult objects.

    Returns:
        Dict of ablation_name -> aggregated metrics with mean/std.
    """
    by_ablation: dict[str, list[RunResult]] = defaultdict(list)
    for r in all_results:
        by_ablation[r.ablation].append(r)

    agg = {}
    for ab_name, runs in by_ablation.items():
        tier_accs: dict[str, list[float]] = defaultdict(list)
        means = []
        for run in runs:
            for tier, acc in run.final_accs.items():
                tier_accs[tier].append(acc * 100)
            means.append(_mean_acc(run.final_accs))

        tier_stats = {}
        for tier in ALL_TIER_NAMES:
            vals = tier_accs.get(tier, [0.0])
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0
            tier_stats[tier] = {"mean": round(mean, 1), "std": round(std, 1)}

        overall_mean = sum(means) / len(means)
        overall_std = math.sqrt(sum((m - overall_mean) ** 2 for m in means) / len(means)) if len(means) > 1 else 0

        agg[ab_name] = {
            "n_seeds": len(runs),
            "seeds": [r.seed for r in runs],
            "tier_stats": tier_stats,
            "overall_mean": round(overall_mean, 1),
            "overall_std": round(overall_std, 1),
            "mean_wall_clock_s": round(sum(r.wall_clock_s for r in runs) / len(runs), 1),
        }

    return agg


def print_ablation_table(agg: dict) -> None:
    """Print the ablation results table."""
    console.print()
    console.print(Rule("[bold cyan]Ablation Study Results[/bold cyan]"))

    t = Table(title="Accuracy by Tier (mean ± std across seeds)", box=box.ROUNDED, style="cyan")
    t.add_column("Condition", style="bold")
    for tn in ALL_TIER_NAMES:
        short = tn.replace("tier", "T").replace("_", " ")[:12]
        t.add_column(short, justify="right")
    t.add_column("Mean", justify="right", style="bold")
    t.add_column("Seeds", justify="right")

    for ab_name, stats in agg.items():
        row = [ab_name]
        for tn in ALL_TIER_NAMES:
            ts = stats["tier_stats"].get(tn, {"mean": 0, "std": 0})
            if ts["std"] > 0:
                row.append(f"{ts['mean']:.0f}±{ts['std']:.0f}")
            else:
                row.append(f"{ts['mean']:.0f}%")
        if stats["overall_std"] > 0:
            row.append(f"{stats['overall_mean']:.0f}±{stats['overall_std']:.0f}")
        else:
            row.append(f"{stats['overall_mean']:.0f}%")
        row.append(str(stats["n_seeds"]))
        t.add_row(*row)

    console.print(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study + multi-seed evaluation"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument(
        "--ablations", nargs="+",
        default=["full", "no_cot", "no_accumulation", "random_layers"],
        choices=list(ABLATIONS.keys()),
    )
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--tasks-per-tier", type=int, default=3)
    parser.add_argument("--eval-problems", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", default="data/ablation_study")
    args = parser.parse_args()

    n_runs = len(args.ablations) * len(args.seeds)
    console.print(Panel(
        f"[bold]ISOPro Ablation Study[/bold]\n"
        f"Ablations: {', '.join(args.ablations)}\n"
        f"Seeds: {args.seeds}\n"
        f"Total runs: {n_runs}  |  {args.iterations} iterations each",
        style="cyan",
    ))

    all_results: list[RunResult] = []
    run_num = 0

    for ab_key in args.ablations:
        ab_config = ABLATIONS[ab_key]
        console.print()
        console.print(Rule(f"[bold magenta]{ab_config.name}[/bold magenta]"))
        console.print(f"  {ab_config.description}")

        for seed in args.seeds:
            run_num += 1
            console.print(
                f"\n  [dim]Run {run_num}/{n_runs}[/dim]"
            )

            result = run_single(
                ablation_config=ab_config,
                seed=seed,
                model_name=args.model,
                iterations=args.iterations,
                k_samples=args.k_samples,
                tasks_per_tier=args.tasks_per_tier,
                eval_problems=args.eval_problems,
                max_tokens=args.max_tokens,
                lora_layers=args.lora_layers,
                lora_rank=args.lora_rank,
                lr=args.lr,
            )
            all_results.append(result)

            # Save incrementally
            _save_results(args.output_dir, all_results, args)

    # Final aggregation
    agg = aggregate_results(all_results)
    print_ablation_table(agg)

    # Save final
    _save_results(args.output_dir, all_results, args, agg)

    console.print(Panel(
        f"[bold green]Ablation Study Complete[/bold green]\n"
        f"{n_runs} runs across {len(args.ablations)} conditions × {len(args.seeds)} seeds",
        style="green",
    ))


def _save_results(output_dir, all_results, args, agg=None):
    """Save results incrementally."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "ablation_study",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "config": {
            "iterations": args.iterations,
            "k_samples": args.k_samples,
            "tasks_per_tier": args.tasks_per_tier,
            "lora_rank": args.lora_rank,
            "lora_layers": args.lora_layers,
            "seeds": args.seeds,
        },
        "runs": [
            {
                "ablation": r.ablation,
                "seed": r.seed,
                "baseline_accs": r.baseline_accs,
                "final_accs": r.final_accs,
                "loss_curve": r.loss_curve,
                "hit_rate_curve": r.hit_rate_curve,
                "buffer_sizes": r.buffer_sizes,
                "buffer_composition": r.buffer_composition_final,
                "transition_points": r.transition_points,
                "wall_clock_s": r.wall_clock_s,
            }
            for r in all_results
        ],
    }
    if agg:
        payload["aggregated"] = agg

    with open(out / "ablation_results.json", "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
