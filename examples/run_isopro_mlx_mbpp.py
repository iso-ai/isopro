#!/usr/bin/env python3
"""ISOPro MBPP code-creation experiment on MLX (Apple Silicon native).

Mirrors examples/run_isopro_mlx.py but operates on the MBPP code-generation
domain instead of scheduling:

  - Task bank:       sanitized MBPP (Austin et al. 2021), binned by canonical
                     solution length into 5 tiers (T0-T4). T4 is held out.
  - Verifier:        deterministic subprocess sandbox running the MBPP
                     assertion list. Binary 1.0/0.0, no partial credit.
  - Architecture:    same as scheduling — MLX for inference + LoRA training,
                     no reference model, no KL penalty, rejection sampling.

Requires Python 3.9+ and Apple Silicon.

Usage:
    /opt/homebrew/bin/python3.11 examples/run_isopro_mlx_mbpp.py
    /opt/homebrew/bin/python3.11 examples/run_isopro_mlx_mbpp.py \
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \
        --output-dir data/mbpp_llama32_3b
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

from isopro.environments.tasks.mbpp_tasks import (
    MBPPTier,
    build_mbpp_splits,
    generate_mbpp_task,
)
from isopro.environments.tasks.mbpp_verifier import score_mbpp_task
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
    MBPPTier.WARMUP,
    MBPPTier.SHORT,
    MBPPTier.MEDIUM,
    MBPPTier.LONG,
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """One model attempt at an MBPP task.

    Attributes:
        prompt: The task prompt sent to the model.
        response: Generated text including the code block.
        score: Verifier reward (0.0 or 1.0).
        tier: Tier label for per-tier accounting.
        tokens: Token IDs (unused, kept for parity with scheduling).
    """

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
    """Generate a response using MLX with the chat template applied.

    Args:
        model: MLX model with LoRA adapters attached.
        tokenizer: Matching tokenizer.
        prompt: User-facing prompt text (no chat formatting).
        max_tokens: Token budget.
        temp: Sampling temperature.

    Returns:
        Generated text string (post-chat-template).
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
# Training step (verbatim from run_isopro_mlx.py — no domain dependency)
# ---------------------------------------------------------------------------


def train_on_buffer(
    model,
    tokenizer,
    optimizer: opt.Adam,
    replay_buffer: list[Rollout],
    batch_size: int = 4,
    max_length: int = 1024,
) -> float:
    """One epoch of prompt-masked SFT on correct traces via MLX.

    Args:
        model: MLX model with LoRA.
        tokenizer: Matching tokenizer.
        optimizer: Adam optimizer for the LoRA parameters.
        replay_buffer: Accumulated correct rollouts.
        batch_size: Mini-batch size for gradient accumulation.
        max_length: Max tokenized sequence length per example.

    Returns:
        Mean cross-entropy loss across the buffer.
    """
    model.train()
    rng = random.Random(time.time())
    rng.shuffle(replay_buffer)

    losses: list[float] = []

    def loss_fn(model, inputs, targets, mask):
        logits = model(inputs).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, targets)
        ce = ce * mask
        return ce.sum() / mask.sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for i in range(0, len(replay_buffer), batch_size):
        batch = replay_buffer[i : i + batch_size]
        inputs_list, targets_list, masks_list = [], [], []
        for r in batch:
            messages = [
                {"role": "user", "content": r.prompt},
                {"role": "assistant", "content": r.response},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            if len(ids) < 2:
                continue
            ids = ids[:max_length]

            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": r.prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            inputs_list.append(ids[:-1])
            targets_list.append(ids[1:])
            mask = [0] * (prompt_len - 1) + [1] * (len(ids) - 1 - (prompt_len - 1))
            mask = mask[: len(ids) - 1]
            masks_list.append(mask)

        if not inputs_list:
            continue

        max_len = max(len(x) for x in inputs_list)
        pad = tokenizer.pad_token_id or 0
        inputs_arr = mx.array(
            [x + [pad] * (max_len - len(x)) for x in inputs_list]
        )
        targets_arr = mx.array(
            [x + [pad] * (max_len - len(x)) for x in targets_list]
        )
        masks_arr = mx.array(
            [m + [0] * (max_len - len(m)) for m in masks_list],
            dtype=mx.float32,
        )

        loss, grads = loss_and_grad(model, inputs_arr, targets_arr, masks_arr)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_per_tier(
    model,
    tokenizer,
    eval_sets: dict[str, list[Task]],
    max_tokens: int,
) -> dict[str, float]:
    """Evaluate per-tier accuracy on the held-out eval set.

    Args:
        model: MLX model.
        tokenizer: Matching tokenizer.
        eval_sets: Dict of tier_name -> list of Tasks.
        max_tokens: Generation token budget.

    Returns:
        Dict of tier_name -> accuracy in [0.0, 1.0].
    """
    results: dict[str, float] = {}
    for tier_name, tasks in sorted(eval_sets.items()):
        if not tasks:
            results[tier_name] = 0.0
            continue
        correct = 0
        for task in tasks:
            resp = mlx_generate(model, tokenizer, task.prompt, max_tokens, temp=0.1)
            reward, _ = score_mbpp_task(task, resp)
            if reward == 1.0:
                correct += 1
        results[tier_name] = round(correct / len(tasks), 4)
    return results


def print_tier_accuracies(accs: dict[str, float], label: str) -> None:
    """Display per-tier accuracy as a small ASCII bar chart."""
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
        console.print(
            f"\n  [bold]Mean accuracy[/bold]  [bold cyan]{overall*100:.1f}%[/bold cyan]"
        )
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ISOPro training loop on MBPP via MLX."""
    parser = argparse.ArgumentParser(
        description="ISOPro MBPP code-creation experiment on MLX (Apple Silicon)"
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--tasks-per-tier", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--eval-problems", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/mbpp_experiment_mlx")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold]ISOPro MBPP Experiment — MLX[/bold]\n"
        f"Model: {args.model}  |  Iterations: {args.iterations}\n"
        f"LoRA: rank {args.lora_rank}, {args.lora_layers} layers  |  "
        f"K={args.k_samples}  |  Tasks/tier={args.tasks_per_tier}\n"
        f"No reference model. No KL penalty. Single model in memory.",
        style="cyan",
    ))

    # ------------------------------------------------------------------
    # Build train pool + eval sets from the sanitized MBPP dataset
    # ------------------------------------------------------------------
    console.print(Rule("[bold]Loading MBPP + Building Splits[/bold]"))
    splits = build_mbpp_splits(eval_per_tier=args.eval_problems, seed=args.seed)

    train_pool: dict[MBPPTier, list[Task]] = {
        tier: [generate_mbpp_task(p) for p in splits["train"][tier]]
        for tier in TRAINING_TIERS
    }
    eval_sets: dict[str, list[Task]] = {
        tier.value: [generate_mbpp_task(p) for p in splits["eval"][tier]]
        for tier in MBPPTier
    }

    for tier in TRAINING_TIERS:
        console.print(
            f"  {tier.value:18s}  train={len(train_pool[tier]):3d}  "
            f"eval={len(eval_sets[tier.value])}"
        )
    console.print(f"  {MBPPTier.HELD_OUT.value:18s}  train=  0  "
                  f"eval={len(eval_sets[MBPPTier.HELD_OUT.value])} (held-out)")
    total_eval = sum(len(t) for t in eval_sets.values())
    console.print(f"  [bold]Total: {total_eval} eval problems[/bold]")

    # ------------------------------------------------------------------
    # Load model + LoRA
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Baseline eval
    # ------------------------------------------------------------------
    console.print()
    console.print(Rule("[bold]Baseline Evaluation (before training)[/bold]"))
    baseline_accs = evaluate_per_tier(model, tokenizer, eval_sets, args.max_tokens)
    print_tier_accuracies(baseline_accs, "Baseline")

    # ------------------------------------------------------------------
    # Training loop — sample a fresh batch of training tasks each iter
    # ------------------------------------------------------------------
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

        # Sample tasks_per_tier tasks per training tier (with replacement so
        # small tiers don't bottleneck the iteration).
        iter_tasks: list[Task] = []
        for tier in TRAINING_TIERS:
            pool = train_pool[tier]
            n = min(args.tasks_per_tier, len(pool)) if pool else 0
            if n == 0:
                continue
            iter_tasks.extend(rng.sample(pool, n))
        rng.shuffle(iter_tasks)

        total_rollouts = len(iter_tasks) * args.k_samples
        console.print(
            f"  Tasks: {len(iter_tasks)} x {args.k_samples} = {total_rollouts} rollouts"
        )

        # Generate rollouts (same model that gets trained — no separate ref)
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
                reward, _ = score_mbpp_task(task, resp)
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

        # Train on accumulated correct traces
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

        # Per-tier rollout accuracy (this iteration only)
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

    # ------------------------------------------------------------------
    # Final evaluation + persist results (JSON first, LoRA second)
    # ------------------------------------------------------------------
    console.print()
    console.print(Rule("[bold]Final Evaluation (trained model)[/bold]"))
    final_accs = evaluate_per_tier(model, tokenizer, eval_sets, args.max_tokens)
    print_tier_accuracies(final_accs, "ISOPro Final (after training)")

    total_time = time.perf_counter() - experiment_start

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "isopro_mlx_mbpp",
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

    lora_path = output_dir / "lora_adapters"
    lora_path.mkdir(exist_ok=True)
    try:
        mx.savez(str(lora_path / "adapters.npz"), **dict(model.trainable_parameters()))
        console.print(f"  [green]LoRA saved -> {lora_path}[/green]")
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"  [yellow]LoRA save skipped ({exc.__class__.__name__}: {exc})[/yellow]"
        )

    # Summary table
    console.print()
    t = Table(title="Training Progress", box=box.ROUNDED, style="cyan")
    t.add_column("Iter", justify="right")
    t.add_column("Hit rate", justify="right")
    t.add_column("Buffer", justify="right")
    t.add_column("Loss", justify="right")
    for h in history:
        t.add_row(
            str(h.iteration),
            f"{h.n_correct}/{h.n_rollouts}",
            str(sum(h.buffer_composition.values())),
            f"{h.train_loss:.4f}" if h.train_loss is not None else "—",
        )
    console.print(t)


if __name__ == "__main__":
    main()
