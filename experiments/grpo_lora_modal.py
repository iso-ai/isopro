"""GRPO-LoRA on Modal — CLI replacement for the Colab notebook.

Same architecture as ``experiments/grpo_lora_colab.ipynb`` (Unsloth + TRL
``GRPOTrainer`` + per-iteration eval callback writing the same JSON schema as
``examples/run_scheduling_experiment.py``), but as a single script you run
from your terminal:

    modal run experiments/grpo_lora_modal.py \
        --model qwen-2.5-3b --domain scheduling

    modal run experiments/grpo_lora_modal.py \
        --model llama-3.2-3b --domain mbpp --iterations 6

Results are written to a Modal Volume at ``/results/<config>/`` inside the
GPU container and downloaded to the local filesystem at the end of the run.

GPU defaults to A10G — fast enough that one (model, domain) pair finishes
in ~20–30 min and the whole 6-cell table (3 models × 2 domains) fits inside
Modal's $30/mo free credit.

For Gemma 2 9B or any 7B+ model, pass ``--gpu A100-40GB``.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# App, image, persistent volume
# ---------------------------------------------------------------------------


app = modal.App("isopro-grpo-lora")


# Pin the CUDA wheel of vLLM and let Unsloth pull a matching torch. Avoid
# letting bitsandbytes auto-detect — pin a known-good wheel for CUDA 12.1.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        # Bump to torch 2.7 so every transitive dep (peft 0.14+, torchao 0.16+,
        # Unsloth's LoraConfig(target_parameters=...) call) resolves cleanly.
        # This avoids the torchao/peft version-skew rabbithole on torch 2.5.
        "torch>=2.7,<2.8",
        "transformers>=4.46.0,<5.0",
        "trl>=0.13,<1.0",
        "accelerate>=1.0,<2.0",
        "peft>=0.14",
        "datasets>=3.0,<4.0",
        "bitsandbytes>=0.45",
        "sentencepiece",
        "protobuf",
        "vllm>=0.7",
        "huggingface_hub>=0.27",
    )
    .pip_install("unsloth", "unsloth_zoo")
    .pip_install("isopro==0.3.2", "ortools")
    .pip_install("matplotlib")
)


results_volume = modal.Volume.from_name(
    "isopro-grpo-lora-results", create_if_missing=True
)


# ---------------------------------------------------------------------------
# Model name shortcuts
# ---------------------------------------------------------------------------


MODEL_ALIASES: dict[str, str] = {
    "qwen-2.5-3b":   "unsloth/Qwen2.5-3B-Instruct",
    "llama-3.2-3b":  "unsloth/Llama-3.2-3B-Instruct",
    "gemma-2-2b":    "unsloth/gemma-2-2b-it",
    "gemma-2-9b":    "unsloth/gemma-2-9b-it",
    "phi-3.5-mini":  "unsloth/Phi-3.5-mini-instruct",
}


def _resolve_model_id(name: str) -> str:
    """Translate a CLI alias to a full HuggingFace repo id.

    Args:
        name: Alias (e.g. ``llama-3.2-3b``) or full HF repo id.

    Returns:
        The HuggingFace repo id Unsloth should load.
    """
    return MODEL_ALIASES.get(name.lower(), name)


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,                       # 2 hours per run
    # bnb 4-bit conversion is CPU-bound and peaks well above the model's FP32
    # size during quantization. 32GB host RAM was still being OOM-killed on
    # the second container retry, suggesting a sustained peak. Bumping to 64GB
    # plus 16 CPUs to make the conversion both feasible and fast.
    memory=64 * 1024,                          # 64 GiB host RAM
    cpu=16.0,                                  # 16 CPU cores
    volumes={"/results": results_volume},
)
def train_grpo(
    model_id: str,
    domain: str,
    n_iterations: int,
    steps_per_iteration: int,
    group_size: int,
    tasks_per_iter: int,
    max_prompt_length: int,
    max_completion_len: int,
    learning_rate: float,
    lora_rank: int,
    eval_per_tier: int,
    seed: int,
    n_problems_scheduling: int,
) -> dict:
    """Run one GRPO-LoRA training run on the chosen (model, domain).

    Args:
        model_id: Full HuggingFace repo id (Unsloth-prefixed).
        domain: ``"scheduling"`` or ``"mbpp"``.
        n_iterations: Number of training iterations (one iteration =
            ``steps_per_iteration`` gradient steps).
        steps_per_iteration: Gradient steps between eval probes.
        group_size: GRPO group size G.
        tasks_per_iter: Number of prompts per iteration. ``G * tasks_per_iter``
            is the rollout count per iteration.
        max_prompt_length: Truncation length for prompts.
        max_completion_len: Truncation length for generations.
        learning_rate: Adam learning rate.
        lora_rank: LoRA rank.
        eval_per_tier: Held-out eval problems per tier.
        seed: Reproducibility seed.
        n_problems_scheduling: Tasks per tier when building the scheduling
            task bank. MBPP uses its full split.

    Returns:
        The same payload that gets written to ``experiment_results.json``.
    """
    import json
    import random
    import time
    from collections import Counter
    from pathlib import Path

    import torch

    from datasets import Dataset
    from transformers import TrainerCallback
    from unsloth import FastLanguageModel, PatchFastRL

    PatchFastRL("GRPO", FastLanguageModel)

    # ---------------- Build the task bank for the chosen domain ----------------

    if domain == "scheduling":
        from isopro.environments.tasks.scheduling_tasks import (
            SchedulingTier,
            build_eval_set,
            build_tier_task_bank,
        )
        from isopro.environments.tasks.scheduling_verifier import score_scheduling_task as score_fn

        train_pool = {
            tier: build_tier_task_bank(tier, n_problems=n_problems_scheduling, base_seed=seed)
            for tier in SchedulingTier
            if tier != SchedulingTier.FULL_COMPOSITION
        }
        eval_set = build_eval_set(n_per_tier=eval_per_tier, base_seed=seed + 1)
        tier_values = [t.value for t in SchedulingTier]

    elif domain == "mbpp":
        from isopro.environments.tasks.mbpp_tasks import (
            MBPPTier,
            build_mbpp_splits,
            generate_mbpp_task,
        )
        from isopro.environments.tasks.mbpp_verifier import score_mbpp_task as score_fn

        splits = build_mbpp_splits(eval_per_tier=eval_per_tier, seed=seed)
        train_pool = {
            tier: [generate_mbpp_task(p) for p in splits["train"][tier]]
            for tier in MBPPTier
            if tier != MBPPTier.HELD_OUT
        }
        eval_set = {
            tier.value: [generate_mbpp_task(p) for p in splits["eval"][tier]]
            for tier in MBPPTier
        }
        tier_values = [t.value for t in MBPPTier]

    else:
        raise ValueError(f"Unknown domain: {domain!r}")

    # Flatten with tier labels.
    flat: list[tuple[str, object]] = []
    for tier, tasks in train_pool.items():
        label = tier.value if hasattr(tier, "value") else str(tier)
        flat.extend((label, t) for t in tasks)
    rng = random.Random(seed)
    rng.shuffle(flat)

    task_lookup = {t.task_id: (t, label) for label, t in flat}
    train_dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": t.prompt}],
                "task_id": t.task_id,
                "tier": label,
            }
            for label, t in flat
        ]
    )
    print(f"[setup] {len(train_dataset)} training rows across {len(train_pool)} tiers")

    # -------------- Load model with Unsloth + LoRA --------------

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_prompt_length + max_completion_len,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.55,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # -------------- Eval helpers + per-iteration callback --------------

    from vllm import SamplingParams

    eval_sampling = SamplingParams(
        temperature=0.1, top_p=0.95, max_tokens=max_completion_len,
    )

    def _vllm_generate(prompts: list[str]) -> list[str]:
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        # Snapshot the in-memory LoRA to disk so vLLM can pick it up. This
        # is the canonical Unsloth pattern for evaluating the trained adapter
        # mid-loop — without it, fast_generate would run on the base model.
        model.save_lora("grpo_saved_lora")
        outs = model.fast_generate(
            chat_prompts,
            sampling_params=eval_sampling,
            lora_request=model.load_lora("grpo_saved_lora"),
        )
        return [o.outputs[0].text for o in outs]

    def evaluate_per_tier() -> dict[str, float]:
        scores: dict[str, list[float]] = {tier: [] for tier in tier_values}
        flat_prompts: list[str] = []
        flat_keys: list[tuple[str, object]] = []
        for tier_label, tasks in eval_set.items():
            label = tier_label.value if hasattr(tier_label, "value") else tier_label
            for t in tasks:
                flat_prompts.append(t.prompt)
                flat_keys.append((label, t))
        completions = _vllm_generate(flat_prompts)
        for (label, task), text in zip(flat_keys, completions):
            score, _ = score_fn(task, text)
            scores[label].append(score)
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}

    history: list[dict] = []
    rollout_log: list[dict] = []
    t0 = time.time()

    output_root = Path("/results") / f"{model_id.split('/')[-1]}_{domain}"
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = output_root / "experiment_results.json"

    def _save_payload() -> dict:
        peak_mb = (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else None
        )
        if history and peak_mb is not None:
            history[-1]["peak_memory_mb"] = round(peak_mb, 1)
        payload = {
            "experiment": {
                "method": "grpo_lora",
                "model_id": model_id,
                "domain": domain,
                "seed": seed,
                "n_iterations": n_iterations,
                "steps_per_iteration": steps_per_iteration,
                "group_size": group_size,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "hardware": "modal_a10g",
            },
            "isopro_loop": {"history": history},
            "total_wall_clock_s": round(time.time() - t0, 2),
        }
        with results_file.open("w") as f:
            json.dump(payload, f, indent=2)
        return payload

    class IterationCallback(TrainerCallback):
        """Run held-out eval every `steps_per_iteration` training steps."""

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step == 0 or step % steps_per_iteration != 0:
                return
            iteration = step // steps_per_iteration
            eval_scores = evaluate_per_tier()
            buffer_comp = Counter(r["tier"] for r in rollout_log if r["score"] >= 1.0)
            n_rollouts = len(rollout_log)
            n_correct = sum(1 for r in rollout_log if r["score"] >= 1.0)
            last_loss = state.log_history[-1].get("loss") if state.log_history else None
            history.append(
                {
                    "iteration": iteration,
                    "tier_accuracies": eval_scores,
                    "buffer_composition": dict(buffer_comp),
                    "n_rollouts": n_rollouts,
                    "n_correct": n_correct,
                    "train_loss": last_loss,
                    "wall_clock_s": round(time.time() - t0, 2),
                    "peak_memory_mb": None,
                }
            )
            _save_payload()
            results_volume.commit()
            print(
                f"[iter {iteration}/{n_iterations}] "
                f"eval={eval_scores} buffer={dict(buffer_comp)} loss={last_loss}"
            )

    # -------------- Reward function (logs every rollout for buffer comp) --------------

    class RolloutLoggingReward:
        __name__ = "correctness_reward"

        def __call__(self, completions, task_id, tier, **_):
            rewards = []
            for completion, tid, ti in zip(completions, task_id, tier):
                text = (
                    completion[0]["content"]
                    if isinstance(completion, list)
                    else str(completion)
                )
                task, _label = task_lookup[tid]
                score, _ = score_fn(task, text)
                rollout_log.append(
                    {"task_id": tid, "tier": ti, "score": float(score)}
                )
                rewards.append(float(score))
            return rewards

    # -------------- Train --------------

    from trl import GRPOConfig, GRPOTrainer

    args = GRPOConfig(
        use_vllm=True,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=group_size,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_len,
        max_steps=n_iterations * steps_per_iteration,
        save_steps=n_iterations * steps_per_iteration,
        output_dir=str(output_root / "checkpoints"),
        seed=seed,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[RolloutLoggingReward()],
        args=args,
        train_dataset=train_dataset,
        callbacks=[IterationCallback()],
    )

    print(f"[train] starting GRPO loop: {n_iterations} iterations x {steps_per_iteration} steps")
    trainer.train()

    payload = _save_payload()
    results_volume.commit()
    print(f"[done] wrote {results_file}")
    return payload


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model: str = "qwen-2.5-3b",
    domain: str = "scheduling",
    iterations: int = 6,
    steps_per_iteration: int = 14,
    group_size: int = 4,
    tasks_per_iter: int = 21,
    max_prompt_length: int = 768,
    max_completion_len: int = 384,
    learning_rate: float = 5e-6,
    lora_rank: int = 16,
    eval_per_tier: int = 5,
    seed: int = 42,
    n_problems_scheduling: int = 20,
    output_dir: str = "data",
):
    """Run one GRPO-LoRA experiment on Modal and download the results.

    Args:
        model: Alias from MODEL_ALIASES or a full HuggingFace repo id.
        domain: ``scheduling`` or ``mbpp``.
        iterations: Number of training iterations.
        steps_per_iteration: Gradient steps between eval probes.
        group_size: GRPO group size G.
        tasks_per_iter: Prompts per iteration.
        max_prompt_length: Prompt truncation length.
        max_completion_len: Completion truncation length.
        learning_rate: Optimizer LR.
        lora_rank: LoRA rank.
        eval_per_tier: Held-out problems per tier.
        seed: Reproducibility seed.
        n_problems_scheduling: Task-bank size per scheduling tier.
        output_dir: Local directory to download results into.
    """
    model_id = _resolve_model_id(model)
    print(f"resolved model: {model} -> {model_id}")

    payload = train_grpo.remote(
        model_id=model_id,
        domain=domain,
        n_iterations=iterations,
        steps_per_iteration=steps_per_iteration,
        group_size=group_size,
        tasks_per_iter=tasks_per_iter,
        max_prompt_length=max_prompt_length,
        max_completion_len=max_completion_len,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        eval_per_tier=eval_per_tier,
        seed=seed,
        n_problems_scheduling=n_problems_scheduling,
    )

    short = model_id.split("/")[-1]
    local_dir = Path(output_dir) / f"grpo_lora_{short}_{domain}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "experiment_results.json"
    with local_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== Run complete ===")
    print(f"Local results:  {local_path}")
    final = payload["isopro_loop"]["history"][-1] if payload["isopro_loop"]["history"] else {}
    print(f"Final iter:     {json.dumps(final, indent=2)}")
    print(f"Wall clock:     {payload['total_wall_clock_s']:.1f} s")
