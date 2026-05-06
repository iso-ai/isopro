# Full Results: ISOPro vs GRPO-LoRA across 3 models × 2 domains

Eval set: 5 problems per tier. Single seed (42).

## Final mean accuracy

| Model | Scheduling — ISOPro | Scheduling — GRPO | MBPP — ISOPro | MBPP — GRPO |
|---|---:|---:|---:|---:|
| qwen-2.5-3b | 38.9% | 13.3% | 52.0% | 44.0% |
| llama-3.2-3b | 36.1% | 41.9% | 48.0% | 44.0% |
| gemma-2-2b | 22.2% | 16.7% | 24.0% | 32.0% |

## Net learning Δ (final − baseline)

Positive = the method moved the model up from its zero-shot baseline. Negative = regression. GRPO baselines marked † use the iter-1 eval (after 14 training steps) as a proxy because the GRPO runner does not probe the model pre-step-0.

| Model | Domain | Method | Baseline | Final | Δ |
|---|---|---|---:|---:|---:|
| qwen-2.5-3b | scheduling | ISOPro | 13.3% | 38.9% | **+25.6pp** |
| qwen-2.5-3b | scheduling | GRPO | 23.3%† | 13.3% | **-10.0pp** |
| qwen-2.5-3b | mbpp | ISOPro | 52.0% | 52.0% | **-0.0pp** |
| qwen-2.5-3b | mbpp | GRPO | 52.0%† | 44.0% | **-8.0pp** |
| llama-3.2-3b | scheduling | ISOPro | 41.7% | 36.1% | **-5.6pp** |
| llama-3.2-3b | scheduling | GRPO | 33.3%† | 41.9% | **+8.5pp** |
| llama-3.2-3b | mbpp | ISOPro | 32.0% | 48.0% | **+16.0pp** |
| llama-3.2-3b | mbpp | GRPO | 40.0%† | 44.0% | **+4.0pp** |
| gemma-2-2b | scheduling | ISOPro | 0.0% | 22.2% | **+22.2pp** |
| gemma-2-2b | scheduling | GRPO | 20.0%† | 16.7% | **-3.3pp** |
| gemma-2-2b | mbpp | ISOPro | 28.0% | 24.0% | **-4.0pp** |
| gemma-2-2b | mbpp | GRPO | 32.0%† | 32.0% | **+0.0pp** |

## Per-tier breakdown

### isopro / qwen-2.5-3b / scheduling (source: paper; mean: 38.9%, Δ: +25.6pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 80.0% | 100.0% | +20.0pp |
| tier1_sequencing | 0.0% | 66.7% | +66.7pp |
| tier2_resource_alloc | 0.0% | 0.0% | +0.0pp |
| tier3_deadline_pressure | 0.0% | 66.7% | +66.7pp |
| tier4_pairwise | 0.0% | 0.0% | +0.0pp |
| tier5_full_composition | 0.0% | 0.0% | +0.0pp |

### isopro / llama-3.2-3b / scheduling (source: m1; mean: 36.1%, Δ: -5.6pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 100.0% | 100.0% | +0.0pp |
| tier1_sequencing | 66.7% | 0.0% | -66.7pp |
| tier2_resource_alloc | 0.0% | 0.0% | +0.0pp |
| tier3_deadline_pressure | 66.7% | 100.0% | +33.3pp |
| tier4_pairwise | 16.7% | 16.7% | +0.0pp |
| tier5_full_composition | 0.0% | 0.0% | +0.0pp |

### isopro / gemma-2-2b / scheduling (source: m1; mean: 22.2%, Δ: +22.2pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 0.0% | 66.7% | +66.7pp |
| tier1_sequencing | 0.0% | 0.0% | +0.0pp |
| tier2_resource_alloc | 0.0% | 0.0% | +0.0pp |
| tier3_deadline_pressure | 0.0% | 66.7% | +66.7pp |
| tier4_pairwise | 0.0% | 0.0% | +0.0pp |
| tier5_full_composition | 0.0% | 0.0% | +0.0pp |

### isopro / qwen-2.5-3b / mbpp (source: modal; mean: 52.0%, Δ: -0.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 60.0% | 60.0% | +0.0pp |
| tier1_short | 100.0% | 80.0% | -20.0pp |
| tier2_medium | 60.0% | 40.0% | -20.0pp |
| tier3_long | 40.0% | 40.0% | +0.0pp |
| tier4_held_out | 0.0% | 40.0% | +40.0pp |

### isopro / llama-3.2-3b / mbpp (source: modal; mean: 48.0%, Δ: +16.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 40.0% | 60.0% | +20.0pp |
| tier1_short | 60.0% | 60.0% | +0.0pp |
| tier2_medium | 0.0% | 40.0% | +40.0pp |
| tier3_long | 40.0% | 40.0% | +0.0pp |
| tier4_held_out | 20.0% | 40.0% | +20.0pp |

### isopro / gemma-2-2b / mbpp (source: modal; mean: 24.0%, Δ: -4.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 40.0% | 40.0% | +0.0pp |
| tier1_short | 40.0% | 60.0% | +20.0pp |
| tier2_medium | 40.0% | 0.0% | -40.0pp |
| tier3_long | 20.0% | 20.0% | +0.0pp |
| tier4_held_out | 0.0% | 0.0% | +0.0pp |

### grpo_lora / qwen-2.5-3b / scheduling (source: modal; mean: 13.3%, Δ: -10.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 80.0% | 80.0% | +0.0pp |
| tier1_sequencing | 20.0% | 0.0% | -20.0pp |
| tier2_resource_alloc | 0.0% | 0.0% | +0.0pp |
| tier3_deadline_pressure | 40.0% | 0.0% | -40.0pp |
| tier4_pairwise | 0.0% | 0.0% | +0.0pp |
| tier5_full_composition | 0.0% | 0.0% | +0.0pp |

### grpo_lora / llama-3.2-3b / scheduling (source: modal; mean: 41.9%, Δ: +8.5pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 100.0% | 100.0% | +0.0pp |
| tier1_sequencing | 60.0% | 80.0% | +20.0pp |
| tier2_resource_alloc | 0.0% | 0.0% | +0.0pp |
| tier3_deadline_pressure | 40.0% | 60.0% | +20.0pp |
| tier4_pairwise | 0.0% | 11.1% | +11.1pp |
| tier5_full_composition | 0.0% | 0.0% | +0.0pp |

### grpo_lora / gemma-2-2b / scheduling (source: modal; mean: 16.7%, Δ: -3.3pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 100.0% | 100.0% | +0.0pp |
| tier1_sequencing | 20.0% | 0.0% | -20.0pp |
| tier2_resource_alloc | 0.0% | 0.0% | +0.0pp |
| tier3_deadline_pressure | 0.0% | 0.0% | +0.0pp |
| tier4_pairwise | 0.0% | 0.0% | +0.0pp |
| tier5_full_composition | 0.0% | 0.0% | +0.0pp |

### grpo_lora / qwen-2.5-3b / mbpp (source: modal; mean: 44.0%, Δ: -8.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 60.0% | 60.0% | +0.0pp |
| tier1_short | 80.0% | 80.0% | +0.0pp |
| tier2_medium | 60.0% | 40.0% | -20.0pp |
| tier3_long | 40.0% | 40.0% | +0.0pp |
| tier4_held_out | 20.0% | 0.0% | -20.0pp |

### grpo_lora / llama-3.2-3b / mbpp (source: modal; mean: 44.0%, Δ: +4.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 40.0% | 40.0% | +0.0pp |
| tier1_short | 80.0% | 80.0% | +0.0pp |
| tier2_medium | 20.0% | 40.0% | +20.0pp |
| tier3_long | 40.0% | 40.0% | +0.0pp |
| tier4_held_out | 20.0% | 20.0% | +0.0pp |

### grpo_lora / gemma-2-2b / mbpp (source: modal; mean: 32.0%, Δ: +0.0pp)

| Tier | Baseline | Final | Δ |
|---|---:|---:|---:|
| tier0_warmup | 40.0% | 40.0% | +0.0pp |
| tier1_short | 40.0% | 40.0% | +0.0pp |
| tier2_medium | 60.0% | 60.0% | +0.0pp |
| tier3_long | 20.0% | 20.0% | +0.0pp |
| tier4_held_out | 0.0% | 0.0% | +0.0pp |
