#!/usr/bin/env python3
"""Run the full ISOPro scheduling experiment using Ollama (LLaMA 3.1 8B).

Self-contained runner that bypasses the top-level isopro __init__ to avoid
the transformers/torch version conflict. Runs five experiment modes:

  1. Prompting baselines (zero-shot + 3-shot) via Ollama
  2. ISOPro training loop (Ollama rollouts + HuggingFace LoRA training)
  3. SFT on shuffled oracle solutions (HuggingFace)
  4. SFT with explicit curriculum (HuggingFace)
  5. Multi-turn evaluation (scope validity — revision trajectories)

Usage:
    python examples/run_scheduling_experiment.py                   # all modes
    python examples/run_scheduling_experiment.py --mode prompting  # prompting only
    python examples/run_scheduling_experiment.py --mode isopro     # ISOPro loop
    python examples/run_scheduling_experiment.py --mode multiturn  # multi-turn eval
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import resource
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Package bootstrap — set up minimal package structure to import ISOPro
# submodules without triggering the broken top-level __init__.py
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
    PAIRWISE_SUBTYPES,
    SchedulingTier,
    build_eval_set,
    build_full_task_bank,
    build_tier_task_bank,
)
from isopro.environments.tasks.scheduling_verifier import (
    score_scheduling_task,
)
from isopro.environments.tasks.scheduling_multiturn import (
    evaluate_multiturn_per_tier,
    run_multiturn_scheduling,
)
from isopro.environments.tasks.base_task import Task

# Rich for display
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


class Rollout(NamedTuple):
    """One model attempt at a scheduling task."""

    prompt: str
    response: str
    score: float
    tier: str
    category: str


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
    peak_memory_mb: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_peak_memory_mb() -> float:
    """Get peak RSS memory in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def format_solution_for_sft(task: Task) -> str:
    """Format ground-truth solution as a clean model completion."""
    start_times = task.ground_truth
    parts = [f"{jid}: {t}" for jid, t in sorted(start_times.items())]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Evaluation (shared)
# ---------------------------------------------------------------------------


def evaluate_per_tier(
    backend,
    eval_sets: dict[str, list[Task]],
    cfg: GenerationConfig,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate accuracy on each tier's held-out set.

    Args:
        backend: Any backend with .generate() method.
        eval_sets: Dict of tier_name -> list of eval Tasks.
        cfg: Generation config.
        verbose: Show per-problem status.

    Returns:
        Dict of tier_name -> accuracy.
    """
    results: dict[str, float] = {}
    total_correct = 0
    total_problems = 0

    for tier_name, tasks in sorted(eval_sets.items()):
        if not tasks:
            results[tier_name] = 0.0
            continue

        correct = 0
        for i, task in enumerate(tasks, 1):
            if verbose:
                with console.status(
                    f"[dim]Eval {tier_name} [{i}/{len(tasks)}][/dim]",
                    spinner="dots",
                ):
                    result = backend.generate(task.prompt, cfg)
            else:
                result = backend.generate(task.prompt, cfg)

            reward, detail = score_scheduling_task(task, result.text)
            if reward == 1.0:
                correct += 1

        results[tier_name] = round(correct / len(tasks), 4)
        total_correct += correct
        total_problems += len(tasks)

    return results


def print_tier_accuracies(
    accs: dict[str, float],
    label: str = "Eval",
) -> None:
    """Display per-tier accuracy bar chart."""
    console.print()
    console.print(Rule(f"[bold cyan]{label}[/bold cyan]"))

    for tier_name in sorted(accs.keys()):
        acc = accs[tier_name]
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

    n = len(accs)
    if n:
        overall = sum(accs.values()) / n
        console.print(
            f"\n  [bold]Mean accuracy[/bold]  "
            f"[bold cyan]{overall*100:.1f}%[/bold cyan]"
        )
    console.print()


# ---------------------------------------------------------------------------
# Mode 1: Prompting baselines (Ollama only)
# ---------------------------------------------------------------------------


def run_prompting(
    ollama: OllamaBackend,
    eval_sets: dict[str, list[Task]],
    args,
) -> dict:
    """Run zero-shot and 3-shot prompting baselines.

    Args:
        ollama: Ollama backend.
        eval_sets: Held-out eval tasks.
        args: CLI args.

    Returns:
        Dict with zero_shot and few_shot results.
    """
    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)

    # --- Zero-shot ---
    console.print(Rule("[bold]Zero-Shot Evaluation[/bold]"))
    t0 = time.perf_counter()
    zero_shot_accs = evaluate_per_tier(ollama, eval_sets, eval_cfg)
    zero_time = time.perf_counter() - t0
    print_tier_accuracies(zero_shot_accs, "Zero-Shot Results")
    console.print(f"  Wall-clock: {zero_time:.1f}s")

    # --- 3-Shot ---
    console.print(Rule("[bold]3-Shot Evaluation[/bold]"))

    # Build 3 solved examples from Tier 1 (simplest, separate seed)
    example_tasks = build_tier_task_bank(
        SchedulingTier.SEQUENCING, n_problems=3, base_seed=77777
    )
    few_shot_prefix = "Here are some solved scheduling examples:\n\n"
    for i, ex in enumerate(example_tasks[:3], 1):
        sol_str = format_solution_for_sft(ex)
        few_shot_prefix += f"Example {i}:\n{ex.prompt}\nAnswer: {sol_str}\n\n"
    few_shot_prefix += "Now solve this new problem:\n\n"

    # Wrap eval tasks with few-shot prefix
    few_shot_eval: dict[str, list[Task]] = {}
    for tier_name, tasks in eval_sets.items():
        wrapped = []
        for task in tasks:
            wrapped_task = Task(
                task_id=task.task_id,
                prompt=few_shot_prefix + task.prompt,
                ground_truth=task.ground_truth,
                difficulty=task.difficulty,
                category=task.category,
                metadata=task.metadata,
            )
            wrapped.append(wrapped_task)
        few_shot_eval[tier_name] = wrapped

    t0 = time.perf_counter()
    few_shot_accs = evaluate_per_tier(ollama, few_shot_eval, eval_cfg)
    few_time = time.perf_counter() - t0
    print_tier_accuracies(few_shot_accs, "3-Shot Results")
    console.print(f"  Wall-clock: {few_time:.1f}s")

    return {
        "zero_shot": {
            "tier_accuracies": zero_shot_accs,
            "wall_clock_s": round(zero_time, 2),
        },
        "few_shot": {
            "tier_accuracies": few_shot_accs,
            "wall_clock_s": round(few_time, 2),
        },
    }


# ---------------------------------------------------------------------------
# HuggingFace helpers (lazy-loaded for training modes)
# ---------------------------------------------------------------------------


def _load_hf_backend(model_id: str):
    """Load HuggingFace backend with LoRA support.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        Loaded HuggingFaceBackend.
    """
    import torch
    from isopro.backends.huggingface import HuggingFaceBackend

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    console.print(f"  Device: {device.upper()}")

    device_map = {"": device} if device in ("mps", "cpu") else "auto"
    console.print(f"  Loading {model_id}...")

    backend = HuggingFaceBackend(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float16 if device in ("mps", "cuda") else None,
    )
    backend._ensure_loaded()
    return backend, device


def _apply_lora(backend, top_k_layers: int = 8, lora_r: int = 16, lora_alpha: int = 32):
    """Probe layers and apply LoRA adapter.

    Args:
        backend: HuggingFace backend.
        top_k_layers: Layers to target.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
    """
    import re
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    model = backend._model

    # Find transformer layers
    layer_modules: dict[int, torch.nn.Module] = {}
    for name, module in model.named_modules():
        m = re.match(r".*\.layers\.(\d+)$", name)
        if m:
            layer_modules[int(m.group(1))] = module

    if not layer_modules:
        top_layers = list(range(top_k_layers))
    else:
        # Probe activations
        activation_norms: dict[int, list[float]] = defaultdict(list)
        hooks = []

        def make_hook(idx):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if isinstance(o, torch.Tensor):
                    activation_norms[idx].append(o.detach().float().norm().item())
            return hook

        for idx, module in layer_modules.items():
            hooks.append(module.register_forward_hook(make_hook(idx)))

        cfg = GenerationConfig(max_new_tokens=50, temperature=0.1)
        model.eval()
        with torch.no_grad():
            # Use a simple probe prompt
            backend.generate("Schedule 3 jobs: A (2 steps), B (1 step), C (3 steps).", cfg)

        for h in hooks:
            h.remove()

        if activation_norms:
            ranked = sorted(
                activation_norms.items(),
                key=lambda kv: sum(kv[1]) / len(kv[1]),
                reverse=True,
            )
            top_layers = sorted([idx for idx, _ in ranked[:top_k_layers]])
        else:
            top_layers = list(range(top_k_layers))

    console.print(f"  Targeting layers: {top_layers}")

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=top_layers,
        lora_dropout=0.05,
        bias="none",
    )
    backend._model = get_peft_model(backend._model, cfg)

    trainable = sum(p.numel() for p in backend._model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backend._model.parameters())
    console.print(
        f"  LoRA: [cyan]{trainable:,}[/cyan] trainable / "
        f"{total:,} total ({100*trainable/total:.3f}%)"
    )


def _train_epoch(backend, training_pairs, optimizer, batch_size=4, max_length=1024):
    """One SFT epoch on (prompt, response) pairs.

    Args:
        backend: HuggingFace backend with LoRA.
        training_pairs: List of (prompt, response).
        optimizer: Optimizer.
        batch_size: Batch size.
        max_length: Max token length.

    Returns:
        Mean loss.
    """
    import torch

    model = backend._model
    tok = backend._tokenizer
    device = next(model.parameters()).device

    model.train()
    total_loss = 0.0
    n_steps = 0

    shuffled = list(training_pairs)
    random.shuffle(shuffled)

    for i in range(0, len(shuffled), batch_size):
        batch = shuffled[i : i + batch_size]
        optimizer.zero_grad()

        step_loss = torch.tensor(0.0, device=device)
        for prompt, response in batch:
            full_text = prompt + "\n" + response
            prompt_enc = tok(
                prompt + "\n",
                return_tensors="pt",
                add_special_tokens=False,
            )
            full_enc = tok(
                full_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )

            input_ids = full_enc["input_ids"].to(device)
            labels = input_ids.clone()
            n_prompt = min(
                prompt_enc["input_ids"].shape[1], input_ids.shape[1] - 1
            )
            labels[:, :n_prompt] = -100

            outputs = model(input_ids=input_ids, labels=labels)
            step_loss = step_loss + outputs.loss / len(batch)

        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()

        total_loss += step_loss.item()
        n_steps += 1

    model.eval()
    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Mode 2: ISOPro training loop
# ---------------------------------------------------------------------------


def run_isopro_loop(
    ollama: OllamaBackend,
    eval_sets: dict[str, list[Task]],
    args,
) -> list[IterationLog]:
    """ISOPro: rejection-sampling fine-tuning.

    Uses Ollama for rollout generation, HuggingFace for LoRA training.

    Args:
        ollama: Ollama backend for rollout generation.
        eval_sets: Per-tier eval tasks.
        args: CLI args.

    Returns:
        List of iteration logs.
    """
    import torch

    console.print(Rule("[bold]ISOPro — Loading HuggingFace Model for Training[/bold]"))
    backend, device = _load_hf_backend(args.hf_model)
    _apply_lora(backend, args.top_k_layers, args.lora_r, args.lora_alpha)
    optimizer = torch.optim.AdamW(
        [p for p in backend._model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    rollout_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.8)
    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)
    rng = random.Random(args.seed)

    replay_buffer: list[Rollout] = []
    history: list[IterationLog] = []

    for iteration in range(1, args.iterations + 1):
        iter_start = time.perf_counter()
        console.print()
        console.print(Rule(
            f"[bold yellow]ISOPro Iteration {iteration}/{args.iterations}[/bold yellow]"
        ))

        # Generate fresh tasks
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

        console.print(
            f"  Tasks: {len(iter_tasks)} x {args.k_samples} samples = "
            f"{len(iter_tasks) * args.k_samples} rollouts"
        )

        # Generate rollouts through HuggingFace (same model that gets
        # LoRA updates, so training actually affects generation)
        rollouts: list[Rollout] = []
        done = 0
        total = len(iter_tasks) * args.k_samples

        for task in iter_tasks:
            for _ in range(args.k_samples):
                done += 1
                with console.status(
                    f"[dim]Rollout {done}/{total}[/dim]", spinner="dots"
                ):
                    result = backend.generate(task.prompt, rollout_cfg)

                reward, _ = score_scheduling_task(task, result.text)
                rollouts.append(Rollout(
                    prompt=task.prompt,
                    response=result.text,
                    score=reward,
                    tier=task.metadata.get("tier", "unknown"),
                    category=task.category,
                ))

        # Filter correct into replay buffer
        correct = [r for r in rollouts if r.score == 1.0]
        replay_buffer.extend(correct)

        buf_comp: dict[str, int] = defaultdict(int)
        for r in replay_buffer:
            buf_comp[r.tier] += 1

        n_total = len(rollouts)
        n_corr = len(correct)
        hit_rate = n_corr / n_total * 100 if n_total else 0
        console.print(
            f"  Correct: [bold]{n_corr}/{n_total}[/bold] ({hit_rate:.0f}%)  |  "
            f"Buffer: [bold]{len(replay_buffer)}[/bold]"
        )
        console.print(f"  Buffer composition: {dict(buf_comp)}")

        # Train on replay buffer via HuggingFace
        train_loss = None
        if replay_buffer:
            training_pairs = [(r.prompt, r.response) for r in replay_buffer]
            console.print(f"  Training on {len(training_pairs)} correct traces...")
            train_loss = _train_epoch(
                backend, training_pairs, optimizer, batch_size=args.batch_size
            )
            console.print(f"  Train loss: [cyan]{train_loss:.4f}[/cyan]")
        else:
            console.print("  [yellow]No correct rollouts — skipping training.[/yellow]")

        # Track accuracy from rollout hit rates per tier (avoids slow HF eval)
        tier_accs: dict[str, float] = {}
        rollout_by_tier: dict[str, list[float]] = defaultdict(list)
        for r in rollouts:
            rollout_by_tier[r.tier].append(r.score)
        for tier_name, scores in rollout_by_tier.items():
            tier_accs[tier_name] = round(sum(scores) / len(scores), 4)

        # Also track buffer tiers as proxy for capability
        for tier_name in eval_sets:
            if tier_name not in tier_accs:
                tier_accs[tier_name] = 0.0

        console.print(f"  Rollout accuracy by tier: {tier_accs}")

        iter_time = time.perf_counter() - iter_start
        history.append(IterationLog(
            iteration=iteration,
            tier_accuracies=tier_accs,
            buffer_composition=dict(buf_comp),
            n_rollouts=n_total,
            n_correct=n_corr,
            train_loss=train_loss,
            wall_clock_s=round(iter_time, 2),
            peak_memory_mb=round(get_peak_memory_mb(), 1),
        ))

    # Save LoRA weights
    save_dir = Path(args.output_dir) / "checkpoints" / "isopro_loop_lora"
    save_dir.mkdir(parents=True, exist_ok=True)
    backend._model.save_pretrained(str(save_dir))
    console.print(f"  [green]LoRA saved -> {save_dir}[/green]")

    # Final evaluation through HuggingFace (has the trained LoRA weights)
    console.print()
    console.print(Rule("[bold]Final Evaluation (trained model)[/bold]"))
    final_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
    print_tier_accuracies(final_accs, "ISOPro Final")

    # Append final eval as last entry
    if history:
        history[-1].tier_accuracies = final_accs

    # Free HF model
    del backend, optimizer
    gc.collect()

    return history


# ---------------------------------------------------------------------------
# Mode 3: SFT on shuffled oracle solutions
# ---------------------------------------------------------------------------


def run_sft_shuffled(
    eval_sets: dict[str, list[Task]],
    args,
) -> list[IterationLog]:
    """SFT baseline: train on OR-Tools solutions, randomly shuffled.

    Args:
        eval_sets: Held-out eval tasks.
        args: CLI args.

    Returns:
        List of iteration logs.
    """
    import torch

    console.print(Rule("[bold]SFT-Shuffled — Loading Model[/bold]"))
    backend, device = _load_hf_backend(args.hf_model)
    _apply_lora(backend, args.top_k_layers, args.lora_r, args.lora_alpha)
    optimizer = torch.optim.AdamW(
        [p for p in backend._model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)

    # Build oracle training set
    all_training = []
    for tier in TRAINING_TIERS:
        tier_seed = args.seed + hash(tier.value) % (2**31)
        n = args.tasks_per_tier * args.iterations
        if tier == SchedulingTier.PAIRWISE:
            tasks = build_tier_task_bank(tier, n, tier_seed, subtypes=PAIRWISE_SUBTYPES)
        else:
            tasks = build_tier_task_bank(tier, n, tier_seed)
        all_training.extend(tasks)

    training_pairs = [(t.prompt, format_solution_for_sft(t)) for t in all_training]
    random.Random(args.seed).shuffle(training_pairs)
    console.print(f"  Training set: {len(training_pairs)} oracle pairs (shuffled)")

    chunk_size = max(1, len(training_pairs) // args.iterations)
    history: list[IterationLog] = []

    for iteration in range(1, args.iterations + 1):
        iter_start = time.perf_counter()
        console.print()
        console.print(Rule(
            f"[bold yellow]SFT-Shuffled Epoch {iteration}/{args.iterations}[/bold yellow]"
        ))

        end_idx = min(chunk_size * iteration, len(training_pairs))
        epoch_pairs = training_pairs[:end_idx]

        train_loss = _train_epoch(
            backend, epoch_pairs, optimizer, batch_size=args.batch_size
        )
        console.print(f"  Loss: [cyan]{train_loss:.4f}[/cyan] on {len(epoch_pairs)} pairs")

        tier_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
        print_tier_accuracies(tier_accs, f"SFT-Shuffled Epoch {iteration}")

        iter_time = time.perf_counter() - iter_start
        history.append(IterationLog(
            iteration=iteration,
            tier_accuracies=tier_accs,
            n_rollouts=len(epoch_pairs),
            n_correct=len(epoch_pairs),
            train_loss=train_loss,
            wall_clock_s=round(iter_time, 2),
            peak_memory_mb=round(get_peak_memory_mb(), 1),
        ))

    save_dir = Path(args.output_dir) / "checkpoints" / "sft_shuffled_lora"
    save_dir.mkdir(parents=True, exist_ok=True)
    backend._model.save_pretrained(str(save_dir))
    console.print(f"  [green]LoRA saved -> {save_dir}[/green]")

    del backend, optimizer
    gc.collect()

    return history


# ---------------------------------------------------------------------------
# Mode 4: SFT with explicit curriculum
# ---------------------------------------------------------------------------


def run_sft_curriculum(
    eval_sets: dict[str, list[Task]],
    args,
) -> list[IterationLog]:
    """SFT baseline: train tier-by-tier until convergence.

    Args:
        eval_sets: Held-out eval tasks.
        args: CLI args.

    Returns:
        List of iteration logs.
    """
    import torch

    console.print(Rule("[bold]SFT-Curriculum — Loading Model[/bold]"))
    backend, device = _load_hf_backend(args.hf_model)
    _apply_lora(backend, args.top_k_layers, args.lora_r, args.lora_alpha)
    optimizer = torch.optim.AdamW(
        [p for p in backend._model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)
    convergence_threshold = 0.8
    max_epochs_per_tier = max(2, args.iterations // len(TRAINING_TIERS))

    history: list[IterationLog] = []
    global_iter = 0

    for tier in TRAINING_TIERS:
        tier_name = tier.value
        tier_seed = args.seed + hash(tier_name) % (2**31)
        n = args.tasks_per_tier * args.iterations

        if tier == SchedulingTier.PAIRWISE:
            tier_tasks = build_tier_task_bank(tier, n, tier_seed, subtypes=PAIRWISE_SUBTYPES)
        else:
            tier_tasks = build_tier_task_bank(tier, n, tier_seed)

        tier_pairs = [(t.prompt, format_solution_for_sft(t)) for t in tier_tasks]
        console.print(
            f"\n  Training on [bold]{tier_name}[/bold]: {len(tier_pairs)} pairs"
        )

        for epoch in range(1, max_epochs_per_tier + 1):
            global_iter += 1
            iter_start = time.perf_counter()
            console.print(Rule(
                f"[bold yellow]Curriculum {tier_name} Epoch {epoch}[/bold yellow]"
            ))

            train_loss = _train_epoch(
                backend, tier_pairs, optimizer, batch_size=args.batch_size
            )
            console.print(f"  Loss: [cyan]{train_loss:.4f}[/cyan]")

            tier_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
            print_tier_accuracies(tier_accs, f"Curriculum Iter {global_iter}")

            iter_time = time.perf_counter() - iter_start
            history.append(IterationLog(
                iteration=global_iter,
                tier_accuracies=tier_accs,
                n_rollouts=len(tier_pairs),
                n_correct=len(tier_pairs),
                train_loss=train_loss,
                wall_clock_s=round(iter_time, 2),
                peak_memory_mb=round(get_peak_memory_mb(), 1),
            ))

            current_acc = tier_accs.get(tier_name, 0.0)
            if current_acc >= convergence_threshold:
                console.print(
                    f"  [green]Converged ({current_acc*100:.0f}% >= "
                    f"{convergence_threshold*100:.0f}%)[/green]"
                )
                break

    save_dir = Path(args.output_dir) / "checkpoints" / "sft_curriculum_lora"
    save_dir.mkdir(parents=True, exist_ok=True)
    backend._model.save_pretrained(str(save_dir))
    console.print(f"  [green]LoRA saved -> {save_dir}[/green]")

    del backend, optimizer
    gc.collect()

    return history


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def save_results(output_dir: str, all_results: dict) -> None:
    """Save all experiment results to JSON.

    Args:
        output_dir: Directory for output files.
        all_results: Complete results dict.
    """
    path = Path(output_dir) / "experiment_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    console.print(f"\n[green]Results saved -> [cyan]{path}[/cyan]")


def print_comparison_table(all_results: dict) -> None:
    """Print final comparison table across all modes.

    Args:
        all_results: Complete results dict.
    """
    console.print()
    console.print(Rule("[bold cyan]Final Comparison[/bold cyan]"))

    # Collect all tier names
    all_tiers: set[str] = set()
    for mode_data in all_results.values():
        if isinstance(mode_data, dict):
            if "tier_accuracies" in mode_data:
                all_tiers.update(mode_data["tier_accuracies"].keys())
            elif "history" in mode_data and mode_data["history"]:
                all_tiers.update(mode_data["history"][-1]["tier_accuracies"].keys())
            elif "zero_shot" in mode_data:
                all_tiers.update(mode_data["zero_shot"]["tier_accuracies"].keys())

    tier_list = sorted(all_tiers)

    t = Table(title="Accuracy by Tier (final iteration)", box=box.ROUNDED, style="cyan")
    t.add_column("Mode", style="bold")
    for tn in tier_list:
        short = tn.replace("tier", "T").replace("_", " ")[:15]
        t.add_column(short, justify="right")
    t.add_column("Mean", justify="right", style="bold")
    t.add_column("Time (s)", justify="right")

    def add_row(name, accs, wall_time=None):
        row = [name]
        vals = []
        for tn in tier_list:
            acc = accs.get(tn, 0.0)
            vals.append(acc)
            color = (
                "bright_green" if acc >= 0.75
                else "yellow" if acc >= 0.5
                else "red"
            )
            row.append(f"[{color}]{acc*100:.0f}%[/{color}]")
        mean = sum(vals) / len(vals) if vals else 0
        row.append(f"{mean*100:.0f}%")
        row.append(f"{wall_time:.0f}" if wall_time else "—")
        t.add_row(*row)

    # Prompting results
    if "prompting" in all_results:
        p = all_results["prompting"]
        if "zero_shot" in p:
            add_row(
                "Zero-shot",
                p["zero_shot"]["tier_accuracies"],
                p["zero_shot"].get("wall_clock_s"),
            )
        if "few_shot" in p:
            add_row(
                "3-shot",
                p["few_shot"]["tier_accuracies"],
                p["few_shot"].get("wall_clock_s"),
            )

    # Training results
    for mode in ["isopro_loop", "sft_shuffled", "sft_curriculum"]:
        if mode in all_results and "history" in all_results[mode]:
            hist = all_results[mode]["history"]
            if hist:
                last = hist[-1]
                total_time = sum(h["wall_clock_s"] for h in hist)
                add_row(
                    mode.replace("_", "-").upper(),
                    last["tier_accuracies"],
                    total_time,
                )

    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="ISOPro scheduling experiment (Ollama + HuggingFace)"
    )
    parser.add_argument(
        "--mode",
        choices=["prompting", "isopro_loop", "sft-shuffled", "sft-curriculum", "multiturn", "all"],
        default="all",
    )
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument(
        "--hf-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model for LoRA training modes",
    )
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--tasks-per-tier", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--top-k-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="data/scheduling_experiment",
    )
    parser.add_argument(
        "--eval-problems",
        type=int,
        default=5,
        help="Eval problems per tier (per pairwise subtype: this // 2 + 1)",
    )
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold]ISOPro Scheduling Experiment[/bold]\n"
        f"Mode: {args.mode}  |  Ollama: {args.ollama_model}\n"
        f"Iterations: {args.iterations}  |  K-samples: {args.k_samples}  |  "
        f"Tasks/tier: {args.tasks_per_tier}\n"
        f"Seed: {args.seed}",
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
    console.print(f"  [bold]Total: {total_eval} evaluation problems[/bold]")

    # Connect to Ollama
    console.print()
    ollama = OllamaBackend(args.ollama_model)
    console.print(f"  Ollama connected: {args.ollama_model}")

    modes = (
        ["prompting", "multiturn", "isopro_loop", "sft-shuffled", "sft-curriculum"]
        if args.mode == "all"
        else [args.mode]
    )

    all_results: dict = {
        "experiment": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ollama_model": args.ollama_model,
            "hf_model": args.hf_model,
            "seed": args.seed,
            "iterations": args.iterations,
            "k_samples": args.k_samples,
            "tasks_per_tier": args.tasks_per_tier,
            "eval_problems_per_tier": args.eval_problems,
        },
    }

    experiment_start = time.perf_counter()

    for mode in modes:
        console.print()
        console.print(Panel(f"[bold magenta]Mode: {mode}[/bold magenta]", style="magenta"))

        if mode == "prompting":
            results = run_prompting(ollama, eval_sets, args)
            all_results["prompting"] = results

        elif mode == "multiturn":
            console.print(Rule("[bold]Multi-Turn Revision Evaluation[/bold]"))
            mt_cfg = GenerationConfig(
                max_new_tokens=args.max_tokens, temperature=0.1
            )
            mt_results = evaluate_multiturn_per_tier(
                ollama, eval_sets, mt_cfg, max_attempts=3
            )
            all_results["multiturn"] = mt_results

            # Display
            console.print()
            t = Table(
                title="Multi-Turn Results (max 3 attempts)",
                box=box.ROUNDED, style="cyan",
            )
            t.add_column("Tier", style="bold")
            t.add_column("1st Attempt", justify="right")
            t.add_column("Trajectory", justify="right")
            t.add_column("Recovery", justify="right")
            t.add_column("Mean Attempts", justify="right")
            for tn in sorted(mt_results.keys()):
                m = mt_results[tn]
                t.add_row(
                    tn.replace("tier", "T").replace("_", " "),
                    f"{m['first_attempt_acc']*100:.0f}%",
                    f"{m['trajectory_acc']*100:.0f}%",
                    f"{m['recovery_rate']*100:.0f}%",
                    f"{m['mean_attempts']:.1f}",
                )
            console.print(t)

        elif mode == "isopro_loop":
            history = run_isopro_loop(ollama, eval_sets, args)
            all_results["isopro_loop"] = {
                "history": [
                    {
                        "iteration": h.iteration,
                        "tier_accuracies": h.tier_accuracies,
                        "buffer_composition": h.buffer_composition,
                        "n_rollouts": h.n_rollouts,
                        "n_correct": h.n_correct,
                        "train_loss": h.train_loss,
                        "wall_clock_s": h.wall_clock_s,
                        "peak_memory_mb": h.peak_memory_mb,
                    }
                    for h in history
                ],
            }

        elif mode == "sft-shuffled":
            history = run_sft_shuffled(eval_sets, args)
            all_results["sft_shuffled"] = {
                "history": [
                    {
                        "iteration": h.iteration,
                        "tier_accuracies": h.tier_accuracies,
                        "n_rollouts": h.n_rollouts,
                        "train_loss": h.train_loss,
                        "wall_clock_s": h.wall_clock_s,
                        "peak_memory_mb": h.peak_memory_mb,
                    }
                    for h in history
                ],
            }

        elif mode == "sft-curriculum":
            history = run_sft_curriculum(eval_sets, args)
            all_results["sft_curriculum"] = {
                "history": [
                    {
                        "iteration": h.iteration,
                        "tier_accuracies": h.tier_accuracies,
                        "n_rollouts": h.n_rollouts,
                        "train_loss": h.train_loss,
                        "wall_clock_s": h.wall_clock_s,
                        "peak_memory_mb": h.peak_memory_mb,
                    }
                    for h in history
                ],
            }

        # Save after each mode in case of interruption
        save_results(args.output_dir, all_results)

    total_time = time.perf_counter() - experiment_start
    all_results["total_wall_clock_s"] = round(total_time, 2)
    save_results(args.output_dir, all_results)

    # Final comparison
    print_comparison_table(all_results)

    console.print(Panel(
        f"[bold green]Experiment Complete[/bold green]\n"
        f"Total time: {total_time/60:.1f} minutes\n"
        f"Peak memory: {get_peak_memory_mb():.0f} MB",
        style="green",
    ))


if __name__ == "__main__":
    main()
