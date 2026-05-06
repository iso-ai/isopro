# Full Results: ISOPro vs GRPO-LoRA across 3 models × 2 domains
Cells with — are not yet available.

## Mean accuracy

| Model | Scheduling — ISOPro | Scheduling — GRPO | MBPP — ISOPro | MBPP — GRPO |
|---|---:|---:|---:|---:|
| qwen-2.5-3b | 38.9% | 13.3% | 52.0% | 44.0% |
| llama-3.2-3b | 36.1% | 41.9% | 48.0% | 44.0% |
| gemma-2-2b | 22.2% | 16.7% | 24.0% | 32.0% |

## Per-tier breakdown

### isopro / qwen-2.5-3b / scheduling (source: paper; mean: 38.9%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 100.0% |
| tier1_sequencing | 66.7% |
| tier2_resource_alloc | 0.0% |
| tier3_deadline_pressure | 66.7% |
| tier4_pairwise | 0.0% |
| tier5_full_composition | 0.0% |

### isopro / llama-3.2-3b / scheduling (source: m1; mean: 36.1%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 100.0% |
| tier1_sequencing | 0.0% |
| tier2_resource_alloc | 0.0% |
| tier3_deadline_pressure | 100.0% |
| tier4_pairwise | 16.7% |
| tier5_full_composition | 0.0% |

### isopro / gemma-2-2b / scheduling (source: m1; mean: 22.2%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 66.7% |
| tier1_sequencing | 0.0% |
| tier2_resource_alloc | 0.0% |
| tier3_deadline_pressure | 66.7% |
| tier4_pairwise | 0.0% |
| tier5_full_composition | 0.0% |

### isopro / qwen-2.5-3b / mbpp (source: modal; mean: 52.0%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 60.0% |
| tier1_short | 80.0% |
| tier2_medium | 40.0% |
| tier3_long | 40.0% |
| tier4_held_out | 40.0% |

### isopro / llama-3.2-3b / mbpp (source: modal; mean: 48.0%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 60.0% |
| tier1_short | 60.0% |
| tier2_medium | 40.0% |
| tier3_long | 40.0% |
| tier4_held_out | 40.0% |

### isopro / gemma-2-2b / mbpp (source: modal; mean: 24.0%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 40.0% |
| tier1_short | 60.0% |
| tier2_medium | 0.0% |
| tier3_long | 20.0% |
| tier4_held_out | 0.0% |

### grpo_lora / qwen-2.5-3b / scheduling (source: modal; mean: 13.3%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 80.0% |
| tier1_sequencing | 0.0% |
| tier2_resource_alloc | 0.0% |
| tier3_deadline_pressure | 0.0% |
| tier4_pairwise | 0.0% |
| tier5_full_composition | 0.0% |

### grpo_lora / llama-3.2-3b / scheduling (source: modal; mean: 41.9%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 100.0% |
| tier1_sequencing | 80.0% |
| tier2_resource_alloc | 0.0% |
| tier3_deadline_pressure | 60.0% |
| tier4_pairwise | 11.1% |
| tier5_full_composition | 0.0% |

### grpo_lora / gemma-2-2b / scheduling (source: modal; mean: 16.7%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 100.0% |
| tier1_sequencing | 0.0% |
| tier2_resource_alloc | 0.0% |
| tier3_deadline_pressure | 0.0% |
| tier4_pairwise | 0.0% |
| tier5_full_composition | 0.0% |

### grpo_lora / qwen-2.5-3b / mbpp (source: modal; mean: 44.0%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 60.0% |
| tier1_short | 80.0% |
| tier2_medium | 40.0% |
| tier3_long | 40.0% |
| tier4_held_out | 0.0% |

### grpo_lora / llama-3.2-3b / mbpp (source: modal; mean: 44.0%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 40.0% |
| tier1_short | 80.0% |
| tier2_medium | 40.0% |
| tier3_long | 40.0% |
| tier4_held_out | 20.0% |

### grpo_lora / gemma-2-2b / mbpp (source: modal; mean: 32.0%)

| Tier | Accuracy |
|---|---:|
| tier0_warmup | 40.0% |
| tier1_short | 40.0% |
| tier2_medium | 60.0% |
| tier3_long | 20.0% |
| tier4_held_out | 0.0% |
