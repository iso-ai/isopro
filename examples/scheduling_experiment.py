"""IsoZero scheduling experiment: training loop + three baselines.

Implements the full experimental protocol for the scheduling domain:

  1. IsoZero — rejection-sampling fine-tuning with implicit curriculum.
     Rollouts at temperature 0.8 via Ollama → verify with deterministic
     scheduler → correct traces enter replay buffer → LoRA update on
     correct traces only. Per-tier evaluation at every iteration.

  2. Baseline: SFT on oracle solutions (shuffled). Standard supervised
     fine-tuning on OR-Tools ground-truth solutions for all training
     tiers, randomly shuffled. Same LoRA config.

  3. Baseline: SFT with explicit curriculum. Same as Baseline 1 but
     trains Tier 1 → Tier 2 → Tier 3 → Tier 4 sequentially until
     convergence per tier.

  4. Baseline: Zero-shot and few-shot prompting. No fine-tuning — raw
     model evaluation with 0 and 3 example shots.

Logging captures: per-tier accuracy at every iteration, replay buffer
composition, transition iteration per tier, wall-clock time, and peak
memory usage.

Usage:
    # IsoZero training loop
    python examples/scheduling_experiment.py --mode isozero

    # SFT shuffled baseline
    python examples/scheduling_experiment.py --mode sft-shuffled

    # SFT curriculum baseline
    python examples/scheduling_experiment.py --mode sft-curriculum

    # Zero/few-shot baselines (Ollama)
    python examples/scheduling_experiment.py --mode prompting

    # Run all modes sequentially
    python examples/scheduling_experiment.py --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from isopro.backends.base import GenerationConfig
from isopro.backends.huggingface import HuggingFaceBackend
from isopro.environments.tasks.base_task import DifficultyLevel, Task
from isopro.environments.tasks.scheduling_tasks import (
    PAIRWISE_SUBTYPES,
    SchedulingTier,
    build_eval_set,
    build_full_task_bank,
    build_tier_task_bank,
    render_problem_prompt,
)
from isopro.environments.tasks.scheduling_verifier import (
    score_scheduling_task,
    verify_schedule,
)

console = Console(highlight=False)

# Training tiers (Tier 5 is held-out, never in training)
TRAINING_TIERS = [
    SchedulingTier.SEQUENCING,
    SchedulingTier.RESOURCE_ALLOC,
    SchedulingTier.DEADLINE_PRESSURE,
    SchedulingTier.PAIRWISE,
]
ALL_TIERS = list(SchedulingTier)


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
    """Metrics for one training iteration.

    Attributes:
        iteration: 1-indexed iteration number.
        tier_accuracies: Dict of tier_name -> accuracy on held-out eval.
        buffer_composition: Dict of tier_name -> count in replay buffer.
        n_rollouts: Total rollouts this iteration.
        n_correct: Correct rollouts this iteration.
        train_loss: Mean loss for this iteration's training step.
        wall_clock_s: Wall-clock seconds for this iteration.
        peak_memory_mb: Peak RSS memory in MB.
    """

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
    """Get peak RSS memory usage in MB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns kilobytes
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def format_solution_for_sft(task: Task) -> str:
    """Format a ground-truth solution as a model completion for SFT.

    Generates a clean response in the expected output format so the
    SFT baseline trains on the same format the verifier expects.

    Args:
        task: Task with ground_truth containing start_times dict.

    Returns:
        Formatted response string.
    """
    start_times = task.ground_truth
    parts = [f"{jid}: {t}" for jid, t in sorted(start_times.items())]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Layer probe + LoRA (reused from petroleum_training pattern)
# ---------------------------------------------------------------------------


def probe_layer_activations(
    backend: HuggingFaceBackend,
    probe_tasks: list[Task],
    top_k: int = 8,
) -> list[int]:
    """Find which transformer layers activate most on scheduling prompts.

    Args:
        backend: Loaded HuggingFace backend.
        probe_tasks: Small set of tasks for probing.
        top_k: Number of top layers to return.

    Returns:
        Sorted list of layer indices.
    """
    import re as re_mod

    model = backend._model
    layer_modules: dict[int, torch.nn.Module] = {}
    for name, module in model.named_modules():
        m = re_mod.match(r".*\.layers\.(\d+)$", name)
        if m:
            layer_modules[int(m.group(1))] = module

    if not layer_modules:
        console.print("[yellow]No layer modules found — using first K.[/yellow]")
        return list(range(top_k))

    activation_norms: dict[int, list[float]] = defaultdict(list)
    hooks = []

    def make_hook(idx: int):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                activation_norms[idx].append(
                    out.detach().float().norm().item()
                )
        return hook

    for idx, module in layer_modules.items():
        hooks.append(module.register_forward_hook(make_hook(idx)))

    cfg = GenerationConfig(max_new_tokens=50, temperature=0.1)
    model.eval()
    with torch.no_grad():
        for task in probe_tasks[:4]:
            try:
                backend.generate(task.prompt, cfg)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    if not activation_norms:
        return list(range(top_k))

    ranked = sorted(
        activation_norms.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
        reverse=True,
    )
    top_layers = sorted([idx for idx, _ in ranked[:top_k]])

    t = Table(title="Layer Activation Norms (scheduling)", box=box.SIMPLE)
    t.add_column("Layer", justify="right")
    t.add_column("Mean norm", justify="right")
    for idx, norms in ranked[:top_k]:
        t.add_row(str(idx), f"{sum(norms)/len(norms):.1f}")
    console.print(t)

    return top_layers


def apply_lora(
    backend: HuggingFaceBackend,
    target_layers: list[int],
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> None:
    """Apply LoRA adapter to the model in-place.

    Args:
        backend: HuggingFace backend.
        target_layers: Layer indices.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=target_layers,
        lora_dropout=0.05,
        bias="none",
    )
    backend._model = get_peft_model(backend._model, cfg)
    trainable = sum(p.numel() for p in backend._model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backend._model.parameters())
    console.print(
        f"  LoRA adapter: [cyan]{trainable:,}[/cyan] trainable / "
        f"{total:,} total  ([dim]{100*trainable/total:.3f}%[/dim])"
    )


# ---------------------------------------------------------------------------
# Training step (shared by IsoZero and SFT baselines)
# ---------------------------------------------------------------------------


def train_epoch(
    backend: HuggingFaceBackend,
    training_pairs: list[tuple[str, str]],
    optimizer: torch.optim.Optimizer,
    batch_size: int = 4,
    max_length: int = 1024,
) -> float:
    """One SFT epoch on (prompt, response) pairs.

    Prompt tokens are masked so loss is computed only on response tokens.

    Args:
        backend: HuggingFace backend with LoRA.
        training_pairs: List of (prompt, response) tuples.
        optimizer: AdamW optimizer.
        batch_size: Gradient accumulation batch size.
        max_length: Max tokenized length.

    Returns:
        Mean cross-entropy loss.
    """
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
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_per_tier(
    backend: HuggingFaceBackend,
    eval_sets: dict[str, list[Task]],
    cfg: GenerationConfig,
    use_ollama_backend=None,
) -> dict[str, float]:
    """Evaluate accuracy on each tier's held-out set.

    Args:
        backend: HuggingFace backend (or None if using Ollama).
        eval_sets: Dict of tier_name -> list of eval Tasks.
        cfg: Generation config.
        use_ollama_backend: If provided, use this Ollama backend instead.

    Returns:
        Dict of tier_name -> accuracy (float in [0, 1]).
    """
    gen_backend = use_ollama_backend or backend
    results: dict[str, float] = {}

    for tier_name, tasks in eval_sets.items():
        if not tasks:
            results[tier_name] = 0.0
            continue

        correct = 0
        for task in tasks:
            with console.status(
                f"[dim]Eval {tier_name}[/dim]", spinner="dots"
            ):
                result = gen_backend.generate(task.prompt, cfg)
            reward, _ = score_scheduling_task(task, result.text)
            if reward == 1.0:
                correct += 1

        results[tier_name] = round(correct / len(tasks), 4)

    return results


def print_tier_accuracies(
    accs: dict[str, float],
    iteration: int,
    label: str = "Eval",
) -> None:
    """Display per-tier accuracy bar chart."""
    console.print()
    console.print(Rule(f"[bold cyan]{label} — Iteration {iteration}[/bold cyan]"))

    for tier_name in sorted(accs.keys()):
        acc = accs[tier_name]
        bar_w = 24
        filled = int(acc * bar_w)
        color = (
            "bright_green" if acc >= 0.75
            else "yellow" if acc >= 0.5
            else "red"
        )
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * (bar_w - filled))
        bar.append(f"  {acc*100:.0f}%", style="bold")
        # Shorten tier name for display
        short = tier_name.replace("tier", "T").replace("_", " ")
        console.print(f"  {short:30s}", bar)

    overall = sum(accs.values()) / len(accs) if accs else 0
    console.print(f"\n  [bold]Mean[/bold]  [bold cyan]{overall*100:.0f}%[/bold cyan]")
    console.print()


# ---------------------------------------------------------------------------
# Task bank builders for training
# ---------------------------------------------------------------------------


def build_training_tasks(
    n_per_tier: int = 20,
    n_per_pairwise_subtype: int = 10,
    seed: int = 42,
) -> list[Task]:
    """Build the training task bank (Tiers 1-4 only).

    Args:
        n_per_tier: Problems per tier (Tiers 1-3).
        n_per_pairwise_subtype: Problems per Tier 4 subtype.
        seed: Base seed.

    Returns:
        Flat list of all training Tasks.
    """
    bank = build_full_task_bank(
        n_per_tier=n_per_tier,
        n_per_pairwise_subtype=n_per_pairwise_subtype,
        base_seed=seed,
    )
    # Exclude Tier 5 from training
    tasks: list[Task] = []
    for tier_name, tier_tasks in bank.items():
        if "tier5" not in tier_name:
            tasks.extend(tier_tasks)
    return tasks


def build_training_tasks_by_tier(
    n_per_tier: int = 20,
    n_per_pairwise_subtype: int = 10,
    seed: int = 42,
) -> dict[str, list[Task]]:
    """Build training tasks grouped by tier (for curriculum baseline).

    Args:
        n_per_tier: Problems per tier.
        n_per_pairwise_subtype: Problems per Tier 4 subtype.
        seed: Base seed.

    Returns:
        Dict of tier_name -> list of Tasks (Tiers 1-4 only).
    """
    bank = build_full_task_bank(
        n_per_tier=n_per_tier,
        n_per_pairwise_subtype=n_per_pairwise_subtype,
        base_seed=seed,
    )
    return {k: v for k, v in bank.items() if "tier5" not in k}


# ---------------------------------------------------------------------------
# Mode 1: IsoZero training loop
# ---------------------------------------------------------------------------


def run_isozero(args, backend, eval_sets, optimizer) -> list[IterationLog]:
    """IsoZero: rejection-sampling fine-tuning with implicit curriculum.

    At each iteration:
      1. Sample tasks from training tiers (proportional to current buffer).
      2. Generate K rollouts per task at exploration temperature.
      3. Verify each rollout with the deterministic scheduler.
      4. Correct rollouts enter the replay buffer.
      5. Train on all accumulated correct traces (LoRA update).
      6. Evaluate on all tiers (including held-out Tier 5).

    Args:
        args: CLI arguments.
        backend: HuggingFace backend with LoRA.
        eval_sets: Per-tier held-out eval tasks.
        optimizer: AdamW optimizer.

    Returns:
        List of IterationLog entries.
    """
    console.print(Rule("[bold]IsoZero Training Loop[/bold]"))

    rollout_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens, temperature=0.8
    )
    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)
    rng = random.Random(args.seed)

    replay_buffer: list[Rollout] = []
    history: list[IterationLog] = []

    for iteration in range(1, args.iterations + 1):
        iter_start = time.perf_counter()
        console.print()
        console.print(Rule(
            f"[bold yellow]IsoZero Iteration {iteration}/{args.iterations}[/bold yellow]"
        ))

        # Step 1: Generate fresh tasks across training tiers
        iter_tasks: list[Task] = []
        for tier in TRAINING_TIERS:
            tier_seed = args.seed + iteration * 1000 + hash(tier.value) % (2**31)
            n = args.tasks_per_tier
            if tier == SchedulingTier.PAIRWISE:
                tasks = build_tier_task_bank(
                    tier, n, tier_seed, subtypes=PAIRWISE_SUBTYPES
                )
            else:
                tasks = build_tier_task_bank(tier, n, tier_seed)
            iter_tasks.extend(tasks)

        rng.shuffle(iter_tasks)
        console.print(
            f"  Task bank: {len(iter_tasks)} tasks × "
            f"{args.k_samples} samples = "
            f"{len(iter_tasks) * args.k_samples} rollouts"
        )

        # Step 2: Generate rollouts
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

        # Step 3: Filter correct rollouts into replay buffer
        correct = [r for r in rollouts if r.score == 1.0]
        replay_buffer.extend(correct)

        # Buffer composition
        buf_comp: dict[str, int] = defaultdict(int)
        for r in replay_buffer:
            buf_comp[r.tier] += 1

        console.print(
            f"  Correct: [bold]{len(correct)}/{len(rollouts)}[/bold]  |  "
            f"Buffer total: [bold]{len(replay_buffer)}[/bold]"
        )
        console.print(f"  Buffer composition: {dict(buf_comp)}")

        # Step 4: Train on replay buffer
        train_loss = None
        if replay_buffer:
            training_pairs = [(r.prompt, r.response) for r in replay_buffer]
            train_loss = train_epoch(
                backend, training_pairs, optimizer, batch_size=args.batch_size
            )
            console.print(f"  Train loss: [cyan]{train_loss:.4f}[/cyan]")
        else:
            console.print(
                "  [yellow]No correct rollouts — skipping training.[/yellow]"
            )

        # Step 5: Per-tier evaluation
        tier_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
        print_tier_accuracies(tier_accs, iteration, "IsoZero Eval")

        iter_time = time.perf_counter() - iter_start
        peak_mem = get_peak_memory_mb()

        log = IterationLog(
            iteration=iteration,
            tier_accuracies=tier_accs,
            buffer_composition=dict(buf_comp),
            n_rollouts=len(rollouts),
            n_correct=len(correct),
            train_loss=train_loss,
            wall_clock_s=round(iter_time, 2),
            peak_memory_mb=round(peak_mem, 1),
        )
        history.append(log)

    return history


# ---------------------------------------------------------------------------
# Mode 2: SFT on oracle solutions (shuffled)
# ---------------------------------------------------------------------------


def run_sft_shuffled(args, backend, eval_sets, optimizer) -> list[IterationLog]:
    """SFT baseline: train on OR-Tools solutions, randomly shuffled.

    All training tasks across Tiers 1-4 are formatted as (prompt, oracle)
    pairs and shuffled. Training runs for the same number of epochs as
    IsoZero iterations, with evaluation after each epoch.

    Args:
        args: CLI arguments.
        backend: HuggingFace backend with LoRA.
        eval_sets: Per-tier held-out eval tasks.
        optimizer: AdamW optimizer.

    Returns:
        List of IterationLog entries.
    """
    console.print(Rule("[bold]SFT Shuffled Baseline[/bold]"))

    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)

    # Build training set from oracle solutions
    all_tasks = build_training_tasks(
        n_per_tier=args.tasks_per_tier * args.iterations,
        n_per_pairwise_subtype=args.tasks_per_tier * args.iterations // 3,
        seed=args.seed,
    )
    training_pairs = [
        (t.prompt, format_solution_for_sft(t)) for t in all_tasks
    ]
    random.Random(args.seed).shuffle(training_pairs)

    console.print(f"  Training set: {len(training_pairs)} oracle pairs (shuffled)")

    # Split into chunks to train epoch-by-epoch
    chunk_size = max(1, len(training_pairs) // args.iterations)
    history: list[IterationLog] = []

    for iteration in range(1, args.iterations + 1):
        iter_start = time.perf_counter()
        console.print()
        console.print(Rule(
            f"[bold yellow]SFT-Shuffled Epoch {iteration}/{args.iterations}[/bold yellow]"
        ))

        # Train on a chunk (cumulative — include all data seen so far)
        end_idx = min(chunk_size * iteration, len(training_pairs))
        epoch_pairs = training_pairs[:end_idx]

        train_loss = train_epoch(
            backend, epoch_pairs, optimizer, batch_size=args.batch_size
        )
        console.print(f"  Train loss: [cyan]{train_loss:.4f}[/cyan] on {len(epoch_pairs)} pairs")

        # Evaluate
        tier_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
        print_tier_accuracies(tier_accs, iteration, "SFT-Shuffled Eval")

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

    return history


# ---------------------------------------------------------------------------
# Mode 3: SFT with explicit curriculum
# ---------------------------------------------------------------------------


def run_sft_curriculum(args, backend, eval_sets, optimizer) -> list[IterationLog]:
    """SFT baseline: train on oracle solutions with explicit tier ordering.

    Trains Tier 1 until convergence, then Tier 2, then Tier 3, then Tier 4.
    "Convergence" is defined as eval accuracy on the current tier reaching
    a threshold or a max-epochs limit.

    Args:
        args: CLI arguments.
        backend: HuggingFace backend with LoRA.
        eval_sets: Per-tier held-out eval tasks.
        optimizer: AdamW optimizer.

    Returns:
        List of IterationLog entries.
    """
    console.print(Rule("[bold]SFT Curriculum Baseline[/bold]"))

    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)

    tasks_by_tier = build_training_tasks_by_tier(
        n_per_tier=args.tasks_per_tier * args.iterations,
        n_per_pairwise_subtype=args.tasks_per_tier * args.iterations // 3,
        seed=args.seed,
    )

    convergence_threshold = 0.8
    max_epochs_per_tier = max(2, args.iterations // len(TRAINING_TIERS))

    history: list[IterationLog] = []
    global_iter = 0

    for tier in TRAINING_TIERS:
        tier_name = tier.value
        tier_tasks = tasks_by_tier.get(tier_name, [])
        if not tier_tasks:
            continue

        tier_pairs = [(t.prompt, format_solution_for_sft(t)) for t in tier_tasks]
        console.print(f"\n  Training on [bold]{tier_name}[/bold]: {len(tier_pairs)} pairs")

        for epoch in range(1, max_epochs_per_tier + 1):
            global_iter += 1
            iter_start = time.perf_counter()

            console.print(Rule(
                f"[bold yellow]Curriculum {tier_name} Epoch {epoch}/{max_epochs_per_tier}[/bold yellow]"
            ))

            train_loss = train_epoch(
                backend, tier_pairs, optimizer, batch_size=args.batch_size
            )
            console.print(f"  Train loss: [cyan]{train_loss:.4f}[/cyan]")

            # Full eval
            tier_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
            print_tier_accuracies(tier_accs, global_iter, "Curriculum Eval")

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

            # Check convergence on current tier
            current_acc = tier_accs.get(tier_name, 0.0)
            if current_acc >= convergence_threshold:
                console.print(
                    f"  [green]Converged on {tier_name} "
                    f"({current_acc*100:.0f}% >= {convergence_threshold*100:.0f}%)[/green]"
                )
                break

    return history


# ---------------------------------------------------------------------------
# Mode 4: Zero-shot and few-shot prompting
# ---------------------------------------------------------------------------


def run_prompting(args, eval_sets) -> dict[str, dict[str, float]]:
    """Zero-shot and few-shot baselines using Ollama (no fine-tuning).

    Args:
        args: CLI arguments.
        eval_sets: Per-tier held-out eval tasks.

    Returns:
        Dict with 'zero_shot' and 'few_shot' keys, each mapping to
        per-tier accuracy dicts.
    """
    from isopro.backends.ollama_backend import OllamaBackend

    console.print(Rule("[bold]Prompting Baselines (Ollama)[/bold]"))

    ollama_model = args.ollama_model or "llama3.1:8b"
    console.print(f"  Model: {ollama_model}")

    try:
        ollama = OllamaBackend(ollama_model)
    except Exception as e:
        console.print(f"[red]Failed to connect to Ollama: {e}[/red]")
        console.print("[dim]Ensure Ollama is running: ollama serve[/dim]")
        return {}

    eval_cfg = GenerationConfig(max_new_tokens=args.max_tokens, temperature=0.1)

    # Zero-shot
    console.print("\n  [bold]Zero-shot evaluation[/bold]")
    zero_shot_accs = evaluate_per_tier(
        None, eval_sets, eval_cfg, use_ollama_backend=ollama
    )
    print_tier_accuracies(zero_shot_accs, 0, "Zero-shot")

    # Few-shot (3 examples)
    console.print("\n  [bold]3-shot evaluation[/bold]")

    # Build 3 example shots from Tier 1 (simplest)
    example_tasks = build_tier_task_bank(
        SchedulingTier.SEQUENCING, n_problems=3, base_seed=77777
    )
    few_shot_prefix = "Here are some solved examples:\n\n"
    for i, ex in enumerate(example_tasks[:3], 1):
        solution_str = format_solution_for_sft(ex)
        few_shot_prefix += f"Example {i}:\n{ex.prompt}\nAnswer: {solution_str}\n\n"
    few_shot_prefix += "Now solve this problem:\n\n"

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

    few_shot_accs = evaluate_per_tier(
        None, few_shot_eval, eval_cfg, use_ollama_backend=ollama
    )
    print_tier_accuracies(few_shot_accs, 0, "3-shot")

    return {"zero_shot": zero_shot_accs, "few_shot": few_shot_accs}


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------


def save_results(
    output_path: str,
    mode: str,
    args,
    history: list[IterationLog] | None = None,
    prompting_results: dict | None = None,
) -> None:
    """Save experiment results to JSON.

    Args:
        output_path: File path for JSON output.
        mode: Experiment mode name.
        args: CLI arguments.
        history: List of IterationLog (for training modes).
        prompting_results: Dict of prompting results.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "experiment": {
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.hf_model,
            "seed": args.seed,
            "iterations": args.iterations,
            "k_samples": args.k_samples,
            "tasks_per_tier": args.tasks_per_tier,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
    }

    if history:
        payload["history"] = [
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
        ]
        if history:
            payload["final_tier_accuracies"] = history[-1].tier_accuracies
            payload["total_wall_clock_s"] = round(
                sum(h.wall_clock_s for h in history), 2
            )
            payload["total_training_tokens"] = sum(
                h.n_rollouts for h in history
            )

    if prompting_results:
        payload["prompting"] = prompting_results

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    console.print(f"[green]Results saved -> [cyan]{path}[/cyan]")


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


def print_final_summary(
    mode: str,
    history: list[IterationLog] | None = None,
) -> None:
    """Print a final summary table."""
    if not history:
        return

    console.print()
    console.print(Rule(f"[bold cyan]{mode} — Final Summary[/bold cyan]"))

    t = Table(box=box.ROUNDED, style="cyan")
    t.add_column("Iter", justify="right")
    t.add_column("Correct", justify="right")
    t.add_column("Loss", justify="right")
    t.add_column("Time (s)", justify="right")
    t.add_column("Mem (MB)", justify="right")

    # Add tier columns
    tier_names = sorted(history[0].tier_accuracies.keys()) if history else []
    for tn in tier_names:
        short = tn.replace("tier", "T").replace("_", " ")[:12]
        t.add_column(short, justify="right")

    for h in history:
        row = [
            str(h.iteration),
            f"{h.n_correct}/{h.n_rollouts}" if h.n_rollouts else "—",
            f"{h.train_loss:.4f}" if h.train_loss is not None else "—",
            f"{h.wall_clock_s:.1f}",
            f"{h.peak_memory_mb:.0f}",
        ]
        for tn in tier_names:
            acc = h.tier_accuracies.get(tn, 0)
            color = (
                "bright_green" if acc >= 0.75
                else "yellow" if acc >= 0.5
                else "red"
            )
            row.append(f"[{color}]{acc*100:.0f}%[/{color}]")
        t.add_row(*row)

    console.print(t)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_backend(args) -> HuggingFaceBackend:
    """Load HuggingFace model and return backend.

    Args:
        args: CLI arguments with hf_model.

    Returns:
        Loaded HuggingFaceBackend.
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    console.print(f"  Device: {device.upper()}")

    device_map = {"": device} if device in ("mps", "cpu") else "auto"
    console.print(f"  Loading {args.hf_model}…")

    backend = HuggingFaceBackend(
        args.hf_model,
        device_map=device_map,
        torch_dtype=torch.float16 if device in ("mps", "cuda") else None,
    )
    backend._ensure_loaded()
    return backend


def setup_lora_and_optimizer(
    backend: HuggingFaceBackend,
    eval_tasks: list[Task],
    args,
) -> torch.optim.Optimizer:
    """Probe layers, apply LoRA, create optimizer.

    Args:
        backend: Loaded HuggingFace backend.
        eval_tasks: Tasks for layer probing.
        args: CLI arguments.

    Returns:
        AdamW optimizer over LoRA parameters.
    """
    top_layers = probe_layer_activations(
        backend, eval_tasks, top_k=args.top_k_layers
    )
    console.print(f"  Targeting layers: {top_layers}")

    apply_lora(backend, top_layers, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    optimizer = torch.optim.AdamW(
        [p for p in backend._model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the scheduling experiment."""
    parser = argparse.ArgumentParser(
        description="IsoZero scheduling experiment with baselines"
    )
    parser.add_argument(
        "--mode",
        choices=["isozero", "sft-shuffled", "sft-curriculum", "prompting", "all"],
        default="isozero",
        help="Which experiment mode to run",
    )
    parser.add_argument(
        "--hf-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        metavar="MODEL_ID",
        help="HuggingFace model ID for fine-tuning modes",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Ollama model name for prompting baseline",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--tasks-per-tier", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--top-k-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="data/scheduling_experiment",
        help="Directory for result JSON files",
    )
    parser.add_argument("--save-path", default="checkpoints/scheduling_lora")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold]IsoZero Scheduling Experiment[/bold]\n"
        f"Mode: {args.mode}  |  Model: {args.hf_model}  |  Seed: {args.seed}",
        style="cyan",
    ))

    # Build eval sets (shared across all modes)
    console.print()
    console.print(Rule("[bold]Building Evaluation Sets[/bold]"))
    eval_sets = build_eval_set(
        n_per_tier=5, n_per_pairwise_subtype=3, base_seed=9999
    )
    for tier_name, tasks in eval_sets.items():
        console.print(f"  {tier_name}: {len(tasks)} eval tasks")

    # Flatten for probe purposes
    all_eval_tasks = [t for tasks in eval_sets.values() for t in tasks]

    modes_to_run = (
        ["isozero", "sft-shuffled", "sft-curriculum", "prompting"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes_to_run:
        console.print()
        console.print(Rule(f"[bold magenta]Mode: {mode}[/bold magenta]"))

        if mode == "prompting":
            prompting_results = run_prompting(args, eval_sets)
            save_results(
                f"{args.output_dir}/{mode}_results.json",
                mode, args, prompting_results=prompting_results,
            )
            continue

        # Fine-tuning modes need model + LoRA
        backend = load_backend(args)
        optimizer = setup_lora_and_optimizer(backend, all_eval_tasks, args)

        # Baseline eval before training
        console.print()
        console.print(Rule("[bold]Baseline Evaluation[/bold]"))
        eval_cfg = GenerationConfig(
            max_new_tokens=args.max_tokens, temperature=0.1
        )
        baseline_accs = evaluate_per_tier(backend, eval_sets, eval_cfg)
        print_tier_accuracies(baseline_accs, 0, f"{mode} Baseline")

        # Run the mode
        if mode == "isozero":
            history = run_isozero(args, backend, eval_sets, optimizer)
        elif mode == "sft-shuffled":
            history = run_sft_shuffled(args, backend, eval_sets, optimizer)
        elif mode == "sft-curriculum":
            history = run_sft_curriculum(args, backend, eval_sets, optimizer)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print_final_summary(mode, history)

        # Save LoRA weights
        save_dir = Path(f"{args.save_path}/{mode}")
        save_dir.mkdir(parents=True, exist_ok=True)
        backend._model.save_pretrained(str(save_dir))
        console.print(
            f"[green]LoRA adapter saved -> [cyan]{save_dir}[/cyan]"
        )

        # Save results JSON
        save_results(
            f"{args.output_dir}/{mode}_results.json",
            mode, args, history=history,
        )

        # Free model memory between modes (if running "all")
        if args.mode == "all":
            del backend
            del optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    console.print()
    console.print(Panel("[bold green]Experiment Complete[/bold green]", style="green"))


if __name__ == "__main__":
    main()
