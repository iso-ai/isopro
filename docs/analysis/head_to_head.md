# ISOPro vs GRPO-LoRA — Head-to-head margins

| Model | Domain | ISOPro | GRPO | Winner | Margin |
|---|---|---:|---:|---|---:|
| qwen-2.5-3b | scheduling | 38.9% | 13.3% | **ISOPro** | +25.6pp |
| qwen-2.5-3b | mbpp | 52.0% | 44.0% | **ISOPro** | +8.0pp |
| llama-3.2-3b | scheduling | 36.1% | 41.9% | **GRPO** | +5.7pp |
| llama-3.2-3b | mbpp | 48.0% | 44.0% | **ISOPro** | +4.0pp |
| gemma-2-2b | scheduling | 22.2% | 16.7% | **ISOPro** | +5.6pp |
| gemma-2-2b | mbpp | 24.0% | 32.0% | **GRPO** | +8.0pp |

**Score:** ISOPro 4 (43.1pp avg margin), GRPO 2 (13.7pp avg margin).
