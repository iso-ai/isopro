"""RejectionSamplingTrainer — verifier-grounded fine-tuning without KL or reference models.

Implements the training architecture described in the GCE paper (Section 5):

  1. Generate K rollouts per task at exploration temperature.
  2. Score each rollout against a deterministic ground-truth verifier.
  3. Correct rollouts (score == 1.0) enter an implicit-curriculum replay buffer.
  4. LoRA adapter weights update on accumulated correct traces via
     prompt-masked cross-entropy loss.
  5. Evaluate continuously at every iteration.

Key architectural properties:
  - NO frozen reference model. Rejection sampling replaces the KL penalty
    as the stability mechanism — only outputs the model generates correctly
    at its current capability enter the training set.
  - NO learned reward model. The deterministic verifier is perfectly
    calibrated by construction. Reward hacking is eliminated, not mitigated.
  - Single model in memory. LoRA adapters are the only trainable parameters.
    The frozen base model can be quantized. CPU-updatable.
  - Implicit curriculum. The replay buffer composition reflects actual model
    capability — easy wins accumulate first, harder problems enter as
    capability develops. No researcher-designed curriculum needed.

This is the same conceptual loop as DeepSeek-R1's GRPO at accessible scale,
minus the KL divergence term that requires dual-model VRAM.

Usage:
    from isopro.training.rejection_sampling_trainer import (
        RejectionSamplingTrainer,
        RSTrainingConfig,
    )
    from isopro.environments.scheduling_environment import SchedulingEnvironment

    env = SchedulingEnvironment(tier=SchedulingTier.WARMUP, seed=42)
    backend = HuggingFaceBackend("meta-llama/Llama-3.1-8B-Instruct")
    trainer = RejectionSamplingTrainer(env, backend)
    log = trainer.train(save_path="checkpoints/rs_lora")
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RSTrainingConfig:
    """Configuration for RejectionSamplingTrainer.

    Attributes:
        n_iterations: Number of rollout-train-eval cycles.
        k_samples: Rollout attempts per task per iteration.
        tasks_per_iteration: Number of fresh tasks generated each iteration.
        learning_rate: AdamW learning rate for LoRA parameters.
        max_grad_norm: Gradient clipping threshold.
        batch_size: Training batch size (gradient accumulation).
        max_length: Max tokenized sequence length for training.
        explore_temperature: Sampling temperature for rollout generation.
        eval_temperature: Temperature for evaluation (near-greedy).
        max_new_tokens: Token budget for each generated response.
        correct_threshold: Minimum score to enter the replay buffer.
        save_every: Save checkpoint every N iterations.
        log_every: Log metrics every N iterations.
        seed: Master random seed.
    """

    n_iterations: int = 10
    k_samples: int = 4
    tasks_per_iteration: int = 20
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    batch_size: int = 4
    max_length: int = 1024
    explore_temperature: float = 0.8
    eval_temperature: float = 0.1
    max_new_tokens: int = 512
    correct_threshold: float = 1.0
    save_every: int = 5
    log_every: int = 1
    seed: int = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """One model attempt at a task.

    Attributes:
        prompt: The task prompt.
        response: The model's generated response.
        score: Reward from the verifier (0.0 or 1.0 for binary).
        category: Task category for per-tier tracking.
        metadata: Arbitrary extra info from the task.
    """

    prompt: str
    response: str
    score: float
    category: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RSTrainingStep:
    """Metrics for a single rejection-sampling iteration.

    Attributes:
        iteration: 1-indexed iteration number.
        n_rollouts: Total rollouts generated this iteration.
        n_correct: Rollouts that passed verification.
        buffer_size: Total replay buffer size after this iteration.
        buffer_composition: Category -> count in the replay buffer.
        train_loss: Mean cross-entropy loss on correct traces.
        eval_scores: Category -> accuracy on held-out eval set.
        wall_clock_s: Wall-clock seconds for this iteration.
        peak_memory_mb: Peak RSS memory in MB (if available).
    """

    iteration: int
    n_rollouts: int = 0
    n_correct: int = 0
    buffer_size: int = 0
    buffer_composition: dict[str, int] = field(default_factory=dict)
    train_loss: float | None = None
    eval_scores: dict[str, float] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class RSTrainingLog:
    """Full log from a RejectionSamplingTrainer.train() call.

    Attributes:
        steps: Ordered list of per-iteration metrics.
        total_duration_s: Wall-clock seconds for the full run.
        total_correct_traces: Total correct rollouts accumulated.
        config: The training configuration used.
    """

    steps: list[RSTrainingStep] = field(default_factory=list)
    total_duration_s: float = 0.0
    total_correct_traces: int = 0
    config: dict = field(default_factory=dict)

    def mean_eval_over_last(self, n: int = 5) -> dict[str, float]:
        """Average per-category eval accuracy over the last n iterations.

        Args:
            n: Number of recent iterations to average.

        Returns:
            Dict of category -> mean accuracy.
        """
        recent = self.steps[-n:]
        if not recent:
            return {}
        agg: dict[str, list[float]] = defaultdict(list)
        for step in recent:
            for cat, acc in step.eval_scores.items():
                agg[cat].append(acc)
        return {cat: sum(v) / len(v) for cat, v in agg.items()}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RejectionSamplingTrainer:
    """Verifier-grounded fine-tuning trainer for ISOPro environments.

    Implements the four mechanisms from the GCE paper:
      1. Gradient descent on correct reasoning traces (prompt-masked loss).
      2. Rejection sampling as continuous self-filter (verifier replaces RM).
      3. Implicit-curriculum replay buffer (capability-grounded distribution).
      4. Activation-guided LoRA targeting (optional, via apply_lora).

    No frozen reference model. No KL penalty. Single model in memory.

    Args:
        env: Any task-based BaseEnvironment with generate_task() and score().
        backend: A HuggingFaceBackend with LoRA applied (.model and .tokenizer).
        config: RSTrainingConfig. Uses sensible defaults if not provided.
        rollout_backend: Optional separate backend for rollout generation
            (e.g. OllamaBackend). If None, uses the HuggingFace backend.
        eval_tasks: Optional dict of category -> list[Task] for continuous
            evaluation. If None, evaluation is skipped.
    """

    def __init__(
        self,
        env,
        backend,
        config: RSTrainingConfig | None = None,
        rollout_backend=None,
        eval_tasks: dict[str, list] | None = None,
    ) -> None:
        from .replay_buffer import ImplicitCurriculumBuffer

        self._env = env
        self._backend = backend
        self._rollout_backend = rollout_backend or backend
        self.config = config or RSTrainingConfig()
        self._eval_tasks = eval_tasks
        self._buffer = ImplicitCurriculumBuffer()
        self._log = RSTrainingLog()

    def train(
        self,
        save_path: str | None = None,
    ) -> RSTrainingLog:
        """Run the full rejection-sampling training loop.

        At each iteration:
          1. Generate tasks from the environment.
          2. Produce K rollouts per task at exploration temperature.
          3. Verify each rollout with the environment's score() method.
          4. Correct rollouts (score >= threshold) enter the replay buffer.
          5. Train LoRA weights on all accumulated correct traces.
          6. Evaluate on held-out tasks (if provided).

        Args:
            save_path: Directory to save LoRA checkpoints. No saves if None.

        Returns:
            RSTrainingLog with per-iteration metrics.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "RejectionSamplingTrainer requires torch: pip install torch"
            ) from exc

        self._validate_backend()

        from .replay_buffer import ImplicitCurriculumBuffer

        cfg = self.config
        rng = random.Random(cfg.seed)
        self._buffer = ImplicitCurriculumBuffer()
        self._log = RSTrainingLog(
            config={
                "n_iterations": cfg.n_iterations,
                "k_samples": cfg.k_samples,
                "tasks_per_iteration": cfg.tasks_per_iteration,
                "learning_rate": cfg.learning_rate,
                "explore_temperature": cfg.explore_temperature,
                "correct_threshold": cfg.correct_threshold,
                "batch_size": cfg.batch_size,
            }
        )

        model = self._backend._model
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )

        from ..backends.base import GenerationConfig
        rollout_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.explore_temperature,
        )
        eval_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.eval_temperature,
        )

        run_start = time.perf_counter()

        logger.info(
            "Starting rejection-sampling training: %d iterations, "
            "k=%d samples, %d tasks/iter",
            cfg.n_iterations,
            cfg.k_samples,
            cfg.tasks_per_iteration,
        )

        for iteration in range(1, cfg.n_iterations + 1):
            iter_start = time.perf_counter()

            # ----- Step 1: Generate fresh tasks -----
            tasks = [
                self._env.generate_task()
                for _ in range(cfg.tasks_per_iteration)
            ]

            # ----- Step 2: Produce rollouts -----
            rollouts: list[Rollout] = []
            for task in tasks:
                for _ in range(cfg.k_samples):
                    result = self._rollout_backend.generate(
                        task.prompt, rollout_cfg
                    )
                    score = self._env.score(task, result.text)
                    rollouts.append(Rollout(
                        prompt=task.prompt,
                        response=result.text,
                        score=score,
                        category=task.category,
                        metadata={
                            "task_id": task.task_id,
                            "tier": task.metadata.get("tier", ""),
                        },
                    ))

            # ----- Step 3: Rejection sampling into implicit curriculum buffer -----
            correct = [
                r for r in rollouts if r.score >= cfg.correct_threshold
            ]
            snapshot = self._buffer.add(
                correct_rollouts=[
                    {
                        "prompt": r.prompt,
                        "response": r.response,
                        "category": r.category,
                        "metadata": r.metadata,
                    }
                    for r in correct
                ],
                iteration=iteration,
                n_total_rollouts=len(rollouts),
            )

            # ----- Step 4: Train on full replay buffer -----
            train_loss = None
            if self._buffer.size > 0:
                train_loss = self._train_on_correct_traces(
                    model,
                    self._backend._tokenizer,
                    optimizer,
                    torch,
                    cfg,
                )

            # ----- Step 5: Evaluate -----
            eval_scores: dict[str, float] = {}
            if self._eval_tasks:
                eval_scores = self._evaluate(eval_cfg)

            # ----- Logging -----
            iter_time = time.perf_counter() - iter_start
            peak_mem = self._get_peak_memory_mb()

            step = RSTrainingStep(
                iteration=iteration,
                n_rollouts=len(rollouts),
                n_correct=len(correct),
                buffer_size=self._buffer.size,
                buffer_composition=snapshot.composition,
                train_loss=train_loss,
                eval_scores=eval_scores,
                wall_clock_s=round(iter_time, 2),
                peak_memory_mb=round(peak_mem, 1),
            )
            self._log.steps.append(step)

            if iteration % cfg.log_every == 0:
                logger.info(
                    "Iter %d/%d | correct=%d/%d (%.0f%%) | buffer=%d | "
                    "loss=%s | time=%.1fs",
                    iteration,
                    cfg.n_iterations,
                    len(correct),
                    len(rollouts),
                    snapshot.hit_rate * 100,
                    self._buffer.size,
                    f"{train_loss:.4f}" if train_loss is not None else "N/A",
                    iter_time,
                )

            # ----- Checkpointing -----
            if save_path and iteration % cfg.save_every == 0:
                self._save_checkpoint(save_path, iteration)

        self._log.total_duration_s = round(
            time.perf_counter() - run_start, 2
        )
        self._log.total_correct_traces = self._buffer.size

        if save_path:
            self._save_checkpoint(save_path, cfg.n_iterations, final=True)

        logger.info(
            "Training complete | duration=%.1fs | total_correct=%d | "
            "buffer_size=%d",
            self._log.total_duration_s,
            self._log.total_correct_traces,
            self._buffer.size,
        )

        return self._log

    # ------------------------------------------------------------------
    # Core training step — prompt-masked cross-entropy on correct traces
    # ------------------------------------------------------------------

    def _train_on_correct_traces(
        self,
        model,
        tokenizer,
        optimizer,
        torch,
        cfg: RSTrainingConfig,
    ) -> float:
        """One epoch of SFT on the model's own correct rollouts.

        Prompt tokens are masked with -100 so loss is computed only on
        response tokens — the gradient signal is the token-by-token
        reasoning sequence that produced correctness.

        No KL penalty. No reference model. The rejection sampler already
        ensures training stability by filtering to correct-only traces.

        Args:
            model: The LoRA-wrapped model.
            tokenizer: The tokenizer.
            optimizer: AdamW over LoRA parameters only.
            torch: The torch module.
            cfg: Training configuration.

        Returns:
            Mean cross-entropy loss over the epoch.
        """
        device = next(model.parameters()).device
        model.train()

        total_loss = 0.0
        n_steps = 0

        # Get all accumulated correct traces from the implicit curriculum buffer
        traces = self._buffer.training_traces(shuffle=True)

        for i in range(0, len(traces), cfg.batch_size):
            batch = traces[i : i + cfg.batch_size]
            optimizer.zero_grad()

            step_loss = torch.tensor(0.0, device=device)
            for trace in batch:
                full_text = trace.prompt + "\n" + trace.response
                prompt_enc = tokenizer(
                    trace.prompt + "\n",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                full_enc = tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=cfg.max_length,
                    truncation=True,
                )

                input_ids = full_enc["input_ids"].to(device)
                labels = input_ids.clone()

                # Mask prompt tokens — only penalise response tokens
                n_prompt = min(
                    prompt_enc["input_ids"].shape[1],
                    input_ids.shape[1] - 1,
                )
                labels[:, :n_prompt] = -100

                outputs = model(input_ids=input_ids, labels=labels)
                step_loss = step_loss + outputs.loss / len(batch)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg.max_grad_norm,
            )
            optimizer.step()

            total_loss += step_loss.item()
            n_steps += 1

        model.eval()
        return total_loss / max(n_steps, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, eval_cfg) -> dict[str, float]:
        """Evaluate per-category accuracy on held-out tasks.

        Args:
            eval_cfg: GenerationConfig for evaluation (low temperature).

        Returns:
            Dict of category -> accuracy.
        """
        results: dict[str, float] = {}

        for category, tasks in self._eval_tasks.items():
            if not tasks:
                results[category] = 0.0
                continue

            correct = 0
            for task in tasks:
                result = self._backend.generate(task.prompt, eval_cfg)
                score = self._env.score(task, result.text)
                if score >= self.config.correct_threshold:
                    correct += 1

            results[category] = round(correct / len(tasks), 4)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_backend(self) -> None:
        """Ensure the backend exposes model and tokenizer for training."""
        if not hasattr(self._backend, "_model") or not hasattr(
            self._backend, "_tokenizer"
        ):
            raise AttributeError(
                "RejectionSamplingTrainer requires a HuggingFaceBackend "
                "with ._model and ._tokenizer. API backends (OpenAI, "
                "Anthropic, Ollama) can be used as rollout_backend for "
                "generation, but not as the training backend."
            )

    def _save_checkpoint(
        self,
        save_path: str,
        iteration: int,
        final: bool = False,
    ) -> None:
        """Save LoRA adapter weights.

        Args:
            save_path: Directory for checkpoint files.
            iteration: Current iteration number.
            final: If True, save as 'final' instead of iteration number.
        """
        import os

        suffix = "final" if final else f"iter_{iteration}"
        path = os.path.join(save_path, suffix)
        os.makedirs(path, exist_ok=True)
        self._backend._model.save_pretrained(path)
        logger.info("Checkpoint saved -> %s", path)

    @staticmethod
    def _get_peak_memory_mb() -> float:
        """Get peak RSS memory in MB."""
        import resource
        import sys

        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024

    @property
    def buffer(self) -> "ImplicitCurriculumBuffer":
        """Access the implicit curriculum replay buffer."""
        return self._buffer

    @property
    def buffer_composition(self) -> dict[str, int]:
        """Current replay buffer composition by category."""
        return self._buffer.composition()

    @property
    def capability_trajectory(self) -> list:
        """Per-iteration capability snapshots for the paper's figures."""
        return self._buffer.capability_trajectory()

    @property
    def transition_points(self) -> dict[str, int]:
        """When each category first entered the buffer."""
        return self._buffer.transition_points()
