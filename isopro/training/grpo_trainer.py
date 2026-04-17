"""GRPOTrainer — batteries-included GRPO training loop for ISOPro environments.

GRPO (Group Relative Policy Optimization) samples G responses for the same
prompt, scores all of them, normalises rewards within the group to produce
advantages, and updates the model. No value network is needed — the group
mean is the baseline.

This trainer abstracts away all RL machinery. The user provides:
  - an ISOPro environment (the oracle)
  - a HuggingFaceBackend (the model to train)
  - optionally a TrainingConfig

And gets back a trained model + a training log.

Reference: DeepSeek-R1, Shao et al. 2024 "DeepSeekMath".

Usage:
    env = ReasoningEnvironment(difficulty=DifficultyLevel.EASY)
    backend = HuggingFaceBackend.from_huggingface("Qwen/Qwen2.5-7B-Instruct")
    trainer = GRPOTrainer(env, backend)
    log = trainer.train(n_episodes=1000, save_path="./checkpoints")
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Metrics for a single GRPO update step.

    Attributes:
        episode: Global episode index.
        difficulty: DifficultyLevel.value string for the task.
        category: Task category string.
        group_rewards: Raw rewards for all G responses in the group.
        mean_reward: Mean reward in the group.
        advantages: Normalised advantages after group-mean subtraction.
        policy_loss: Scalar policy loss for this step.
        kl_penalty: KL divergence penalty term.
        total_loss: policy_loss + kl_coef * kl_penalty.
        grad_norm: Gradient norm after clipping.
        latency_ms: Wall-clock time for this update step.
    """

    episode: int
    difficulty: str
    category: str
    group_rewards: list[float]
    mean_reward: float
    advantages: list[float]
    policy_loss: float
    kl_penalty: float
    total_loss: float
    grad_norm: float
    latency_ms: float


@dataclass
class TrainingLog:
    """Full log from a GRPOTrainer.train() call.

    Attributes:
        steps: Ordered list of TrainingStep records.
        final_level: DifficultyLevel.value at the end of training.
        total_duration_s: Wall-clock seconds for the full training run.
        n_promotions: Number of curriculum promotions that occurred.
        n_demotions: Number of curriculum demotions that occurred.
    """

    steps: list[TrainingStep] = field(default_factory=list)
    final_level: str = "easy"
    total_duration_s: float = 0.0
    n_promotions: int = 0
    n_demotions: int = 0

    def mean_reward_over_last(self, n: int = 100) -> float:
        """Compute mean reward over the last n steps.

        Args:
            n: Number of recent steps to average.

        Returns:
            Float mean reward, or 0.0 if no steps recorded.
        """
        recent = self.steps[-n:]
        if not recent:
            return 0.0
        return sum(s.mean_reward for s in recent) / len(recent)


class GRPOTrainer:
    """Batteries-included GRPO trainer for ISOPro task environments.

    Requires a HuggingFaceBackend — weight access is necessary for gradient
    updates. API-only backends (OpenAI, Anthropic) are not supported for
    training, only for evaluate() and DataCollector.

    Args:
        env: Any task-based BaseEnvironment with generate_task() and score().
        backend: A HuggingFaceBackend instance (must have .model and .tokenizer).
        config: TrainingConfig with hyperparameters. Uses sensible defaults
            if not provided.
    """

    def __init__(self, env, backend, config=None) -> None:
        from .config import TrainingConfig

        self._env = env
        self._backend = backend
        self.config = config or TrainingConfig()
        self._log = TrainingLog()

    def train(
        self,
        n_episodes: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> TrainingLog:
        """Run the full GRPO training loop.

        Args:
            n_episodes: Total training episodes. Overrides config.n_episodes
                if provided.
            save_path: Directory to save checkpoints. No checkpoints saved
                if None.

        Returns:
            TrainingLog with per-step metrics for the full run.

        Raises:
            ImportError: If torch is not installed.
            AttributeError: If backend does not expose .model and .tokenizer
                (i.e. not a HuggingFaceBackend).
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("GRPOTrainer requires torch: pip install torch") from exc

        self._validate_backend()

        total = n_episodes or self.config.n_episodes
        cfg = self.config
        curriculum = cfg.curriculum
        self._log = TrainingLog()

        model = self._backend.model
        tokenizer = self._backend.tokenizer
        ref_model = self._clone_reference_model(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
        )
        scheduler = self._make_lr_scheduler(optimizer, total, cfg.warmup_episodes)

        start = time.perf_counter()
        logger.info("Starting GRPO training: %d episodes, group_size=%d", total, cfg.group_size)

        for episode in range(total):
            step_start = time.perf_counter()

            # --- Task generation ---
            if curriculum is not None:
                task = curriculum.next_task()
            else:
                task = self._env.generate_task()

            # --- Sample G responses ---
            from ..backends.base import GenerationConfig
            gen_cfg = GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                system_prompt=None,
            )

            responses = []
            raw_rewards = []
            for _ in range(cfg.group_size):
                result = self._backend.generate(task.prompt, gen_cfg)
                r = self._env.score(task, result.text)
                r = cfg.shape_reward(r)
                responses.append(result.text)
                raw_rewards.append(r)

            # --- Compute group-normalised advantages ---
            mean_r = sum(raw_rewards) / len(raw_rewards)
            std_r = math.sqrt(
                sum((r - mean_r) ** 2 for r in raw_rewards) / len(raw_rewards)
            ) + 1e-8
            advantages = [(r - mean_r) / std_r for r in raw_rewards]

            # --- Policy gradient update ---
            policy_loss, kl_penalty, grad_norm = self._grpo_update(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                prompt=task.prompt,
                responses=responses,
                advantages=advantages,
                kl_coef=cfg.kl_coef,
                max_grad_norm=cfg.max_grad_norm,
                torch=torch,
            )

            if scheduler is not None:
                scheduler.step()

            total_loss = policy_loss + cfg.kl_coef * kl_penalty
            latency = (time.perf_counter() - step_start) * 1000.0

            step = TrainingStep(
                episode=episode,
                difficulty=task.difficulty.value,
                category=task.category,
                group_rewards=raw_rewards,
                mean_reward=mean_r,
                advantages=advantages,
                policy_loss=policy_loss,
                kl_penalty=kl_penalty,
                total_loss=total_loss,
                grad_norm=grad_norm,
                latency_ms=round(latency, 1),
            )
            self._log.steps.append(step)

            # --- Curriculum update ---
            if curriculum is not None:
                transition = curriculum.update(mean_r)
                if transition is not None:
                    if transition.trigger == "promotion":
                        self._log.n_promotions += 1
                    else:
                        self._log.n_demotions += 1

            # --- Logging ---
            if episode % cfg.log_every == 0:
                logger.info(
                    "Episode %d/%d | mean_reward=%.3f | loss=%.4f | kl=%.4f | grad=%.3f",
                    episode,
                    total,
                    mean_r,
                    total_loss,
                    kl_penalty,
                    grad_norm,
                )

            # --- Checkpointing ---
            if save_path and (episode + 1) % cfg.save_every == 0:
                self._save_checkpoint(model, tokenizer, save_path, episode)

        self._log.total_duration_s = round(time.perf_counter() - start, 2)
        self._log.final_level = (
            curriculum.current_level.value if curriculum else task.difficulty.value
        )

        if save_path:
            self._save_checkpoint(model, tokenizer, save_path, total - 1, final=True)

        logger.info(
            "Training complete | duration=%.1fs | final_reward=%.3f | promotions=%d | demotions=%d",
            self._log.total_duration_s,
            self._log.mean_reward_over_last(100),
            self._log.n_promotions,
            self._log.n_demotions,
        )
        return self._log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_backend(self) -> None:
        """Ensure the backend exposes .model and .tokenizer attributes."""
        if not hasattr(self._backend, "model") or not hasattr(self._backend, "tokenizer"):
            raise AttributeError(
                "GRPOTrainer requires a HuggingFaceBackend with .model and .tokenizer. "
                "API backends (OpenAI, Anthropic, etc.) can only be used with "
                "DataCollector or evaluate()."
            )

    def _clone_reference_model(self, model):
        """Create a frozen copy of the model to use as the KL reference.

        Args:
            model: The live (trainable) model.

        Returns:
            A frozen copy of model with requires_grad=False on all parameters.
        """
        import copy
        ref = copy.deepcopy(model)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        return ref

    def _make_lr_scheduler(self, optimizer, total_episodes: int, warmup_episodes: int):
        """Build a linear warmup + cosine decay LR scheduler.

        Args:
            optimizer: The AdamW optimizer.
            total_episodes: Total training episodes.
            warmup_episodes: Number of linear warmup steps.

        Returns:
            A torch.optim.lr_scheduler instance, or None if torch is absent.
        """
        try:
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step: int) -> float:
                if step < warmup_episodes:
                    return step / max(warmup_episodes, 1)
                progress = (step - warmup_episodes) / max(total_episodes - warmup_episodes, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return LambdaLR(optimizer, lr_lambda)
        except Exception:
            return None

    def _grpo_update(
        self,
        model,
        ref_model,
        tokenizer,
        optimizer,
        prompt: str,
        responses: list[str],
        advantages: list[float],
        kl_coef: float,
        max_grad_norm: float,
        torch,
    ) -> tuple[float, float, float]:
        """Execute one GRPO gradient update.

        For each (response, advantage) pair:
          1. Compute log-prob of response tokens under the live policy.
          2. Compute log-prob under the frozen reference policy (for KL).
          3. Loss = -advantage * log_prob + kl_coef * KL(live || ref).

        All G pairs are averaged into a single backward pass.

        Args:
            model: Live trainable model.
            ref_model: Frozen reference model.
            tokenizer: Tokenizer shared by both models.
            optimizer: AdamW optimizer.
            prompt: The task prompt string.
            responses: List of G generated response strings.
            advantages: List of G normalised advantages.
            kl_coef: KL penalty coefficient.
            max_grad_norm: Gradient clipping threshold.
            torch: The torch module (passed in to avoid repeated imports).

        Returns:
            Tuple of (policy_loss, kl_penalty, grad_norm) as floats.
        """
        model.train()
        optimizer.zero_grad()

        total_policy_loss = torch.tensor(0.0)
        total_kl = torch.tensor(0.0)

        for response, adv in zip(responses, advantages):
            full_text = prompt + response
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            input_ids = inputs["input_ids"]
            prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

            with torch.no_grad():
                ref_out = ref_model(input_ids=input_ids, labels=input_ids)
                ref_logprob = -ref_out.loss  # mean NLL → approx log-prob

            live_out = model(input_ids=input_ids, labels=input_ids)
            live_logprob = -live_out.loss

            kl = live_logprob - ref_logprob  # positive when policy diverges from ref
            policy_loss = -adv * live_logprob

            total_policy_loss = total_policy_loss + policy_loss
            total_kl = total_kl + kl

        n = len(responses)
        loss = (total_policy_loss + kl_coef * total_kl) / n
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        return (
            float(total_policy_loss.item() / n),
            float(total_kl.item() / n),
            float(grad_norm),
        )

    def _save_checkpoint(
        self, model, tokenizer, save_path: str, episode: int, final: bool = False
    ) -> None:
        """Save model + tokenizer checkpoint to disk.

        Args:
            model: The live model.
            tokenizer: The tokenizer.
            save_path: Root directory for checkpoints.
            episode: Current episode number (used in subdirectory name).
            final: If True, saves to ``final/`` instead of ``ep_{episode}/``.
        """
        tag = "final" if final else f"ep_{episode}"
        out_dir = os.path.join(save_path, tag)
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("Saved checkpoint to %s", out_dir)
