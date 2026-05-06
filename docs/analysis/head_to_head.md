# ISOPro vs GRPO-LoRA — Head-to-head margins

Eval set: 5 problems per tier. Single seed (42). GRPO baseline marked † uses iter-1 eval as proxy.

## Final accuracy comparison

| Model | Domain | ISOPro | GRPO | Winner | Margin |
|---|---|---:|---:|---|---:|
| qwen-2.5-3b | scheduling | 38.9% | 13.3% | **ISOPro** | +25.6pp |
| qwen-2.5-3b | mbpp | 52.0% | 44.0% | **ISOPro** | +8.0pp |
| llama-3.2-3b | scheduling | 36.1% | 41.9% | **GRPO** | +5.7pp |
| llama-3.2-3b | mbpp | 48.0% | 44.0% | **ISOPro** | +4.0pp |
| gemma-2-2b | scheduling | 22.2% | 16.7% | **ISOPro** | +5.6pp |
| gemma-2-2b | mbpp | 24.0% | 32.0% | **GRPO** | +8.0pp |

**Score:** ISOPro 4 (10.8pp avg winning margin), GRPO 2 (6.9pp avg winning margin).

## Net learning Δ (final − baseline)

This separates 'method moved the model' from 'method ended at a high number because the baseline was high.' Both methods produce mixed Δ at consumer-budget hyperparameters.

| Model | Domain | ISOPro Δ | GRPO Δ |
|---|---|---:|---:|
| qwen-2.5-3b | scheduling | +25.6pp | -10.0pp† |
| qwen-2.5-3b | mbpp | -0.0pp | -8.0pp† |
| llama-3.2-3b | scheduling | -5.6pp | +8.5pp† |
| llama-3.2-3b | mbpp | +16.0pp | +4.0pp† |
| gemma-2-2b | scheduling | +22.2pp | -3.3pp† |
| gemma-2-2b | mbpp | -4.0pp | +0.0pp† |

**Mean Δ across cells:** ISOPro +9.0pp, GRPO -1.5pp.
