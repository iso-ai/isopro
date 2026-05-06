"""ISOPro on Modal — verifier-grounded rejection-sampling training.

Same image and harness as ``experiments/grpo_lora_modal.py``, but the training
loop is the GCE paper's rejection sampler (no KL penalty, no reference model,
single model in memory) instead of TRL's GRPO. Lets us run the apples-to-apples
ISOPro vs. GRPO comparison on the same Modal hardware budget.

Per iteration:
    1. Sample tasks from the per-tier training pool.
    2. Generate K rollouts per task at exploration temperature (T=0.8) via vLLM.
    3. Score each rollout with the deterministic verifier.
    4. Append correct rollouts to the implicit-curriculum replay buffer.
    5. Run prompt-masked SFT on the *full accumulated* buffer for one epoch.
    6. Probe held-out per-tier accuracy via vLLM, snapshotting the live LoRA.

Run:
    modal run experiments/isopro_modal.py --model qwen-2.5-3b --domain scheduling
    modal run experiments/isopro_modal.py --model llama-3.2-3b --domain mbpp
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# App, image, persistent volume — reuse the GRPO Modal image so cache hits
# ---------------------------------------------------------------------------


app = modal.App("isopro-isopro-loop")


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(
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
# Model aliases (same as GRPO runner)
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
# Remote ISOPro training function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    memory=64 * 1024,
    cpu=16.0,
    volumes={"/results": results_volume},
)
def train_isopro(
    model_id: str,
    domain: str,
    n_iterations: int,
    k_samples: int,
    tasks_per_tier: int,
    sft_epochs: int,
    sft_lr: float,
    sft_batch_size: int,
    max_prompt_length: int,
    max_completion_len: int,
    lora_rank: int,
    eval_per_tier: int,
    seed: int,
    n_problems_scheduling: int,
) -> dict:
    """Run one ISOPro rejection-sampling training run.

    Args:
        model_id: Full HuggingFace repo id.
        domain: ``"scheduling"`` or ``"mbpp"``.
        n_iterations: Number of rollout-train-eval cycles.
        k_samples: Rollouts per task per iteration.
        tasks_per_tier: Tasks sampled per tier per iteration.
        sft_epochs: Epochs of SFT on the replay buffer per iteration.
        sft_lr: Learning rate for the prompt-masked SFT step.
        sft_batch_size: Micro-batch size (with gradient accumulation).
        max_prompt_length: Prompt truncation length.
        max_completion_len: Completion truncation length.
        lora_rank: LoRA rank.
        eval_per_tier: Held-out problems per tier.
        seed: Reproducibility seed.
        n_problems_scheduling: Tasks per tier when building the scheduling
            task bank. MBPP uses its full split.

    Returns:
        The same payload that gets written to ``experiment_results.json``.
    """
    import json
    import random
    import time
    from collections import Counter, defaultdict
    from pathlib import Path

    import torch

    from unsloth import FastLanguageModel

    # ----- Build task bank for the chosen domain -----

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

    # Flatten the train pool with tier labels for tracking.
    train_tasks: list[tuple[str, object]] = []
    for tier, tasks in train_pool.items():
        label = tier.value if hasattr(tier, "value") else str(tier)
        train_tasks.extend((label, t) for t in tasks)
    rng = random.Random(seed)
    print(f"[setup] {len(train_tasks)} train tasks across {len(train_pool)} tiers; "
          f"{sum(len(v) for v in eval_set.values())} eval tasks")

    # ----- Load model + LoRA -----

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

    # ----- Generation helpers -----

    from vllm import SamplingParams

    rollout_sampling = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_completion_len,
    )
    eval_sampling = SamplingParams(
        temperature=0.1, top_p=0.95, max_tokens=max_completion_len,
    )

    def _vllm_generate(prompts: list[str], sp: SamplingParams) -> list[str]:
        """Generate completions through vLLM, snapshotting the live LoRA."""
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        model.save_lora("isopro_saved_lora")
        outs = model.fast_generate(
            chat_prompts,
            sampling_params=sp,
            lora_request=model.load_lora("isopro_saved_lora"),
        )
        return [o.outputs[0].text for o in outs]

    def _evaluate_per_tier() -> dict[str, float]:
        scores: dict[str, list[float]] = {tier: [] for tier in tier_values}
        flat_prompts: list[str] = []
        flat_keys: list[tuple[str, object]] = []
        for tier_label, tasks in eval_set.items():
            label = tier_label.value if hasattr(tier_label, "value") else tier_label
            for t in tasks:
                flat_prompts.append(t.prompt)
                flat_keys.append((label, t))
        completions = _vllm_generate(flat_prompts, eval_sampling)
        for (label, task), text in zip(flat_keys, completions):
            score, _ = score_fn(task, text)
            scores[label].append(score)
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}

    # ----- SFT step (prompt-masked CE on accumulated correct traces) -----

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=sft_lr,
    )

    def _sft_on_buffer(buffer: list[dict]) -> float:
        """One epoch of prompt-masked SFT on the implicit-curriculum buffer.

        Args:
            buffer: List of {"prompt": str, "response": str, "tier": str} dicts.

        Returns:
            Mean cross-entropy loss across the epoch.
        """
        if not buffer:
            return float("nan")
        FastLanguageModel.for_training(model)
        rng_local = random.Random()
        shuffled = list(buffer)
        rng_local.shuffle(shuffled)
        losses: list[float] = []
        for _ in range(sft_epochs):
            for i in range(0, len(shuffled), sft_batch_size):
                batch = shuffled[i : i + sft_batch_size]
                input_ids_list, labels_list = [], []
                for ex in batch:
                    full_msgs = [
                        {"role": "user", "content": ex["prompt"]},
                        {"role": "assistant", "content": ex["response"]},
                    ]
                    full_text = tokenizer.apply_chat_template(
                        full_msgs, tokenize=False, add_generation_prompt=False,
                    )
                    full_ids = tokenizer(
                        full_text, return_tensors="pt",
                        truncation=True,
                        max_length=max_prompt_length + max_completion_len,
                    ).input_ids[0]
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": ex["prompt"]}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    prompt_ids = tokenizer(
                        prompt_text, return_tensors="pt",
                    ).input_ids[0]
                    labels = full_ids.clone()
                    labels[: len(prompt_ids)] = -100  # mask prompt
                    input_ids_list.append(full_ids)
                    labels_list.append(labels)

                # Pad to longest in batch
                max_len = max(t.size(0) for t in input_ids_list)
                pad_id = tokenizer.pad_token_id or 0
                input_batch = torch.stack([
                    torch.cat([t, torch.full((max_len - t.size(0),), pad_id, dtype=t.dtype)])
                    for t in input_ids_list
                ]).to(model.device)
                label_batch = torch.stack([
                    torch.cat([t, torch.full((max_len - t.size(0),), -100, dtype=t.dtype)])
                    for t in labels_list
                ]).to(model.device)
                attention = (input_batch != pad_id).long()

                optimizer.zero_grad()
                out = model(input_ids=input_batch, attention_mask=attention, labels=label_batch)
                out.loss.backward()
                optimizer.step()
                losses.append(out.loss.item())
        # Switch back to inference mode for vLLM
        FastLanguageModel.for_inference(model)
        return sum(losses) / len(losses) if losses else float("nan")

    # ----- Main rejection-sampling loop -----

    history: list[dict] = []
    replay_buffer: list[dict] = []
    t0 = time.time()

    output_root = Path("/results") / f"isopro_{model_id.split('/')[-1]}_{domain}"
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = output_root / "experiment_results.json"

    def _save_payload() -> dict:
        peak_mb = (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available() else None
        )
        if history and peak_mb is not None:
            history[-1]["peak_memory_mb"] = round(peak_mb, 1)
        payload = {
            "experiment": {
                "method": "isopro_rejection_sampling",
                "model_id": model_id,
                "domain": domain,
                "seed": seed,
                "n_iterations": n_iterations,
                "k_samples": k_samples,
                "tasks_per_tier": tasks_per_tier,
                "sft_lr": sft_lr,
                "lora_rank": lora_rank,
                "hardware": "modal_a100_80gb",
            },
            "isopro_loop": {"history": history},
            "total_wall_clock_s": round(time.time() - t0, 2),
        }
        with results_file.open("w") as f:
            json.dump(payload, f, indent=2)
        return payload

    # Baseline eval
    print("[baseline] eval ...")
    baseline_scores = _evaluate_per_tier()
    print(f"[baseline] {baseline_scores}")

    for iteration in range(1, n_iterations + 1):
        iter_start = time.time()
        print(f"\n[iter {iteration}/{n_iterations}]")

        # Sample tasks for this iteration (one batch per tier).
        iter_tasks: list[tuple[str, object]] = []
        for tier, tasks in train_pool.items():
            if not tasks:
                continue
            label = tier.value if hasattr(tier, "value") else str(tier)
            n = min(tasks_per_tier, len(tasks))
            iter_tasks.extend((label, t) for t in rng.sample(tasks, n))

        # Generate K rollouts per task at exploration temp.
        flat_prompts: list[str] = []
        flat_meta: list[tuple[str, object]] = []
        for label, t in iter_tasks:
            for _ in range(k_samples):
                flat_prompts.append(t.prompt)
                flat_meta.append((label, t))
        completions = _vllm_generate(flat_prompts, rollout_sampling)

        # Score and accumulate correct ones.
        n_correct = 0
        for (label, task), text in zip(flat_meta, completions):
            score, _ = score_fn(task, text)
            if score >= 1.0:
                n_correct += 1
                replay_buffer.append({
                    "prompt": task.prompt,
                    "response": text,
                    "tier": label,
                })
        n_rollouts = len(completions)
        hit_rate = (n_correct / n_rollouts * 100) if n_rollouts else 0.0
        buffer_comp = Counter(r["tier"] for r in replay_buffer)
        print(f"  rollouts={n_correct}/{n_rollouts} ({hit_rate:.1f}%) "
              f"buffer={dict(buffer_comp)}")

        # SFT on full buffer (prompt-masked).
        train_loss = _sft_on_buffer(replay_buffer)
        print(f"  sft_loss={train_loss:.4f}")

        # Eval probe.
        eval_scores = _evaluate_per_tier()
        wall_clock = time.time() - t0
        history.append({
            "iteration":          iteration,
            "tier_accuracies":    eval_scores,
            "buffer_composition": dict(buffer_comp),
            "n_rollouts":         n_rollouts,
            "n_correct":          n_correct,
            "train_loss":         train_loss,
            "wall_clock_s":       round(wall_clock, 2),
            "peak_memory_mb":     None,
        })
        _save_payload()
        results_volume.commit()
        print(f"  eval={eval_scores} elapsed={iter_start - t0:.0f}s")

    # Final save with baseline metadata.
    payload = _save_payload()
    payload["baseline"] = baseline_scores
    payload["final"] = history[-1]["tier_accuracies"] if history else {}
    with results_file.open("w") as f:
        json.dump(payload, f, indent=2)
    results_volume.commit()
    print(f"[done] wrote {results_file}")
    return payload


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model: str = "qwen-2.5-3b",
    domain: str = "mbpp",
    iterations: int = 6,
    k_samples: int = 4,
    tasks_per_tier: int = 4,
    sft_epochs: int = 1,
    sft_lr: float = 2e-4,
    sft_batch_size: int = 2,
    max_prompt_length: int = 768,
    max_completion_len: int = 384,
    lora_rank: int = 16,
    eval_per_tier: int = 5,
    seed: int = 42,
    n_problems_scheduling: int = 20,
    output_dir: str = "data",
):
    """Run one ISOPro experiment on Modal and download results."""
    model_id = _resolve_model_id(model)
    print(f"resolved model: {model} -> {model_id}")

    payload = train_isopro.remote(
        model_id=model_id,
        domain=domain,
        n_iterations=iterations,
        k_samples=k_samples,
        tasks_per_tier=tasks_per_tier,
        sft_epochs=sft_epochs,
        sft_lr=sft_lr,
        sft_batch_size=sft_batch_size,
        max_prompt_length=max_prompt_length,
        max_completion_len=max_completion_len,
        lora_rank=lora_rank,
        eval_per_tier=eval_per_tier,
        seed=seed,
        n_problems_scheduling=n_problems_scheduling,
    )

    short = model_id.split("/")[-1]
    local_dir = Path(output_dir) / f"isopro_{short}_{domain}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "experiment_results.json"
    with local_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n=== Run complete ===")
    print(f"Local results:  {local_path}")
    final = payload["isopro_loop"]["history"][-1] if payload["isopro_loop"]["history"] else {}
    print(f"Final iter:     {json.dumps(final, indent=2)}")
    print(f"Wall clock:     {payload['total_wall_clock_s']:.1f} s")
