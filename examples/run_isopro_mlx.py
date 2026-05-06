#!/usr/bin/env python3
"""ISOPro scheduling experiment on MLX (Apple Silicon native).

Single-process, single-model architecture:
  - MLX for both inference AND LoRA training
  - No reference model, no KL penalty
  - Rejection sampling: verifier replaces reward model
  - Implicit curriculum via replay buffer composition

Requires Python 3.9+ and Apple Silicon.

Usage:
    /opt/homebrew/bin/python3.11 examples/run_isopro_mlx.py
    /opt/homebrew/bin/python3.11 examples/run_isopro_mlx.py --iterations 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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

from isopro.environments.tasks.scheduling_tasks import (
    PAIRWISE_SUBTYPES,
    SchedulingTier,
    build_eval_set,
    build_tier_task_bank,
)
from isopro.environments.tasks.scheduling_verifier import score_scheduling_task
from isopro.environments.tasks.base_task import Task

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
from rich.text import Text

console = Console(highlight=False)

TRAINING_TIERS = [
    SchedulingTier.WARMUP,
    SchedulingTier.SEQUENCING,
    SchedulingTier.RESOURCE_ALLOC,
    SchedulingTier.DEADLINE_PRESSURE,
    SchedulingTier.PAIRWISE,
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """One model attempt at a scheduling task."""

    prompt: str
    response: str
    score: float
    tier: str
    tokens: list[int] = field(default_factory=list)


@dataclass
class IterationLog:
    """Metrics for one training iteration."""

    iteration: int
    tier_accuracies: dict[str, float] = field(default_factory=dict)
    buffer_composition: dict[str, int] = field(default_factory=dict)
    n_rollouts: int = 0
    n_correct: int = 0
    train_loss: float | None = None
    wall_clock_s: float = 0.0


# ---------------------------------------------------------------------------
# MLX generation helper
# ---------------------------------------------------------------------------


def mlx_generate(model, tokenizer, prompt: str, max_tokens: int, temp: float) -> str:
    """Generate a response using MLX.

    Args:
        model: MLX model with LoRA.
        tokenizer: Tokenizer.
        prompt: Raw prompt text.
        max_tokens: Max tokens to generate.
        temp: Sampling temperature.

    Returns:
        Generated text string.
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=temp)
    return generate(
        model, tokenizer, prompt=formatted,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_on_buffer(
    model,
    tokenizer,
    optimizer: opt.Adam,
    replay_buffer: list[Rollout],
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """One epoch of prompt-masked SFT on correct traces via MLX.

    No reference model. No KL penalty. The rejection sampler already
    ensures stability by filtering to correct-only traces.

    Args:
        model: MLX model with LoRA.
        tokenizer: Tokenizer.
        optimizer: MLX Adam optimizer over LoRA params.
        replay_buffer: Accumulated correct rollouts.
        batch_size: Batch size.
        max_length: Max sequence length.

    Returns:
        Mean cross-entropy loss.
    """
    shuffled = list(replay_buffer)
    random.shuffle(shuffled)

    total_loss = 0.0
    n_steps = 0

    def loss_fn(model, input_ids, prompt_len):
        """Compute cross-entropy loss on response tokens only."""
        logits = model(input_ids)
        # Shift for next-token prediction
        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        # Create mask: 0 for prompt tokens, 1 for response tokens
        seq_len = shift_labels.shape[1]
        mask = mx.arange(seq_len) >= (prompt_len - 1)
        mask = mask.astype(mx.float32)
        # Per-token loss
        token_losses = nn.losses.cross_entropy(shift_logits, shift_labels)
        # Masked mean
        masked_loss = (token_losses * mask).sum() / mx.maximum(mask.sum(), 1)
        return masked_loss

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    for i in range(0, len(shuffled), batch_size):
        batch = shuffled[i : i + batch_size]
        batch_loss = 0.0

        for rollout in batch:
            full_text = rollout.prompt + "\n" + rollout.response
            prompt_tokens = tokenizer.encode(rollout.prompt + "\n")
            full_tokens = tokenizer.encode(full_text)

            if len(full_tokens) > max_length:
                full_tokens = full_tokens[:max_length]

            input_ids = mx.array([full_tokens])
            prompt_len = min(len(prompt_tokens), len(full_tokens) - 1)

            loss, grads = loss_grad_fn(model, input_ids, prompt_len)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            batch_loss += loss.item()

        total_loss += batch_loss / len(batch)
        n_steps += 1

    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_per_tier(
    model,
    tokenizer,
    eval_sets: dict[str, list[Task]],
    max_tokens: int,
) -> dict[str, float]:
    """Evaluate per-tier accuracy.

    Args:
        model: MLX model.
        tokenizer: Tokenizer.
        eval_sets: Dict of tier_name -> eval Tasks.
        max_tokens: Max generation tokens.

    Returns:
        Dict of tier_name -> accuracy.
    """
    results: dict[str, float] = {}
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


def print_tier_accuracies(accs: dict[str, float], label: str) -> None:
    """Display per-tier accuracy bar chart."""
    console.print()
    console.print(Rule(f"[bold cyan]{label}[/bold cyan]"))
    for tier_name in sorted(accs.keys()):
        acc = accs[tier_name]
        bar_w = 30
        filled = int(acc * bar_w)
        color = "bright_green" if acc >= 0.75 else "yellow" if acc >= 0.5 else "red"
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (bar_w - filled))
        bar.append(f"  {acc*100:5.1f}%", style="bold")
        short = tier_name.replace("tier", "T").replace("_", " ")
        console.print(f"  {short:35s}", bar)
    n = len(accs)
    if n:
        overall = sum(accs.values()) / n
        console.print(f"\n  [bold]Mean accuracy[/bold]  [bold cyan]{overall*100:.1f}%[/bold cyan]")
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ISOPro training loop on MLX."""
    parser = argparse.ArgumentParser(
        description="ISOPro scheduling experiment on MLX (Apple Silicon)"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--tasks-per-tier", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--eval-problems", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/scheduling_experiment_mlx")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold]ISOPro Scheduling Experiment — MLX[/bold]\n"
        f"Model: {args.model}  |  Iterations: {args.iterations}\n"
        f"LoRA: rank {args.lora_rank}, {args.lora_layers} layers  |  "
        f"K={args.k_samples}  |  Tasks/tier={args.tasks_per_tier}\n"
        f"No reference model. No KL penalty. Single model in memory.",
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
    console.print(f"  [bold]Total: {total_eval} eval problems[/bold]")

    # Load model + LoRA
    console.print()
    console.print(Rule("[bold]Loading Model + LoRA[/bold]"))
    model, tokenizer = load(args.model)
    model.freeze()
    lora_config = {
        "rank": args.lora_rank,
        "alpha": args.lora_rank * 2,
        "dropout": 0.05,
        "scale": 2.0,
    }
    linear_to_lora_layers(model, num_layers=args.lora_layers, config=lora_config)
    print_trainable_parameters(model)

    optimizer = opt.Adam(learning_rate=args.lr)

    # Baseline eval
    console.print()
    console.print(Rule("[bold]Baseline Evaluation (before training)[/bold]"))
    baseline_accs = evaluate_per_tier(model, tokenizer, eval_sets, args.max_tokens)
    print_tier_accuracies(baseline_accs, "Baseline")

    # Training loop
    rng = random.Random(args.seed)
    replay_buffer: list[Rollout] = []
    history: list[IterationLog] = []
    experiment_start = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        iter_start = time.perf_counter()
        console.print()
        console.print(Rule(
            f"[bold yellow]ISOPro Iteration {iteration}/{args.iterations}[/bold yellow]"
        ))

        # Generate tasks
        iter_tasks: list[Task] = []
        for tier in TRAINING_TIERS:
            tier_seed = args.seed + iteration * 1000 + hash(tier.value) % (2**31)
            n = args.tasks_per_tier
            if tier == SchedulingTier.PAIRWISE:
                tasks = build_tier_task_bank(tier, n, tier_seed, subtypes=PAIRWISE_SUBTYPES)
            else:
                tasks = build_tier_task_bank(tier, n, tier_seed)
            iter_tasks.extend(tasks)
        rng.shuffle(iter_tasks)

        total_rollouts = len(iter_tasks) * args.k_samples
        console.print(f"  Tasks: {len(iter_tasks)} x {args.k_samples} = {total_rollouts} rollouts")

        # Generate rollouts (same model that gets trained)
        rollouts: list[Rollout] = []
        done = 0
        for task in iter_tasks:
            for _ in range(args.k_samples):
                done += 1
                with console.status(f"[dim]Rollout {done}/{total_rollouts}[/dim]", spinner="dots"):
                    resp = mlx_generate(
                        model, tokenizer, task.prompt,
                        args.max_tokens, temp=0.8,
                    )
                reward, _ = score_scheduling_task(task, resp)
                rollouts.append(Rollout(
                    prompt=task.prompt,
                    response=resp,
                    score=reward,
                    tier=task.metadata.get("tier", "unknown"),
                ))

        # Rejection sampling
        correct = [r for r in rollouts if r.score == 1.0]
        replay_buffer.extend(correct)

        buf_comp: dict[str, int] = defaultdict(int)
        for r in replay_buffer:
            buf_comp[r.tier] += 1

        n_corr = len(correct)
        hit_rate = n_corr / len(rollouts) * 100 if rollouts else 0
        console.print(
            f"  Correct: [bold]{n_corr}/{len(rollouts)}[/bold] ({hit_rate:.0f}%)  |  "
            f"Buffer: [bold]{len(replay_buffer)}[/bold]"
        )
        console.print(f"  Buffer: {dict(buf_comp)}")

        # Train on replay buffer
        train_loss = None
        if replay_buffer:
            console.print(f"  Training on {len(replay_buffer)} correct traces...")
            train_loss = train_on_buffer(
                model, tokenizer, optimizer,
                replay_buffer, batch_size=args.batch_size,
            )
            console.print(f"  Train loss: [cyan]{train_loss:.4f}[/cyan]")
        else:
            console.print("  [yellow]No correct rollouts — skipping training.[/yellow]")

        # Per-tier rollout accuracy
        rollout_by_tier: dict[str, list[float]] = defaultdict(list)
        for r in rollouts:
            rollout_by_tier[r.tier].append(r.score)
        tier_accs = {t: round(sum(s)/len(s), 4) for t, s in rollout_by_tier.items()}

        iter_time = time.perf_counter() - iter_start
        console.print(f"  Iter time: {iter_time:.1f}s")

        history.append(IterationLog(
            iteration=iteration,
            tier_accuracies=tier_accs,
            buffer_composition=dict(buf_comp),
            n_rollouts=len(rollouts),
            n_correct=n_corr,
            train_loss=train_loss,
            wall_clock_s=round(iter_time, 2),
        ))

    # Final evaluation
    console.print()
    console.print(Rule("[bold]Final Evaluation (trained model)[/bold]"))
    final_accs = evaluate_per_tier(model, tokenizer, eval_sets, args.max_tokens)
    print_tier_accuracies(final_accs, "ISOPro Final (after training)")

    total_time = time.perf_counter() - experiment_start

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    lora_path = output_dir / "lora_adapters"
    lora_path.mkdir(exist_ok=True)
    mx.savez(str(lora_path / "adapters.npz"), **dict(model.trainable_parameters()))
    console.print(f"  [green]LoRA saved -> {lora_path}[/green]")

    # Save results JSON
    results = {
        "experiment": "isopro_mlx",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "config": {
            "iterations": args.iterations,
            "k_samples": args.k_samples,
            "tasks_per_tier": args.tasks_per_tier,
            "lora_rank": args.lora_rank,
            "lora_layers": args.lora_layers,
            "lr": args.lr,
            "seed": args.seed,
        },
        "baseline": baseline_accs,
        "final": final_accs,
        "history": [
            {
                "iteration": h.iteration,
                "tier_accuracies": h.tier_accuracies,
                "buffer_composition": h.buffer_composition,
                "n_rollouts": h.n_rollouts,
                "n_correct": h.n_correct,
                "train_loss": h.train_loss,
                "wall_clock_s": h.wall_clock_s,
            }
            for h in history
        ],
        "total_wall_clock_s": round(total_time, 2),
    }

    results_path = output_dir / "isopro_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"  [green]Results saved -> {results_path}[/green]")

    # Summary
    console.print()
    t = Table(title="Training Progress", box=box.ROUNDED, style="cyan")
    t.add_column("Iter", justify="right")
    t.add_column("Correct", justify="right")
    t.add_column("Buffer", justify="right")
    t.add_column("Loss", justify="right")
    t.add_column("Time (s)", justify="right")
    for h in history:
        t.add_row(
            str(h.iteration),
            f"{h.n_correct}/{h.n_rollouts}",
            str(sum(h.buffer_composition.values())),
            f"{h.train_loss:.4f}" if h.train_loss else "—",
            f"{h.wall_clock_s:.0f}",
        )
    console.print(t)

    console.print()
    console.print(Panel(
        f"[bold green]Experiment Complete[/bold green]\n"
        f"Total time: {total_time/60:.1f} minutes\n"
        f"Baseline mean: {sum(baseline_accs.values())/len(baseline_accs)*100:.1f}%\n"
        f"Final mean: {sum(final_accs.values())/len(final_accs)*100:.1f}%",
        style="green",
    ))


if __name__ == "__main__":
    main()
