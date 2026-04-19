# ISOPro: A Reference Implementation of Grounded Continuous Evaluation

ISOPro is a simulation-based fine-tuning and evaluation framework for language models. It is the reference implementation of the **Grounded Continuous Evaluation (GCE)** framework described in:

> **Beyond Static Snapshots: A Grounded Evaluation Framework for Language Models at the Agentic Frontier.**
> *Under review, NeurIPS 2026.*

GCE argues that current LLM evaluation practice suffers from four structural validity failures — **distributional**, **temporal**, **scope**, and **process** invalidity — that compound in RLHF and make reward hacking a predictable consequence of evaluation design rather than an unpredictable training pathology. ISOPro demonstrates that these failures can be addressed architecturally, on a consumer laptop, by replacing the learned reward model with a deterministic verifier and updating LoRA adapter weights on CPU.

---

## Headline Results

On resource-constrained project scheduling (RCPSP) with Qwen 2.5 3B Instruct, across six compositional difficulty tiers (T0–T5):

| Method                 | T0   | T1     | T2  | T3     | T4  | T5  | **Mean**    | Trains? |
|------------------------|------|--------|-----|--------|-----|-----|-------------|---------|
| Zero-shot              | 80%  | 0%     | 0%  | 0%     | 0%  | 0%  | 13.3%       | No      |
| 3-shot                 | 20%  | 0%     | 0%  | 20%    | 0%  | 0%  | 6.7%        | No      |
| Multi-turn (×3)        | 100% | 0%     | 0%  | 0%     | 0%  | 0%  | 16.7%       | No      |
| IsoZero (simulation)   | 100% | 60%    | 20% | 20%    | 0%  | 0%  | 33.3%       | No      |
| **ISOPro + LoRA**      | 100% | 66.7%  | 0%  | 66.7%  | 0%  | 0%  | **39.8% ± 3.5** | **Yes** |

*Eval: 3–5 problems per tier, 3 seeds. ISOPro: 6 iterations, 504 rollouts, 119 correct traces. Hardware: Apple M1, 32GB unified memory, ~90 min, peak memory <8GB, 0.216% trainable parameters, no GPU required.*

A 3.0× improvement over zero-shot is achieved without oracle solutions, without a reward model, and without a KL penalty.

<p align="center">
  <img src="docs/figures/fig4_method_comparison.png" width="85%" alt="Per-tier accuracy across evaluation conditions" />
</p>

---

## What ISOPro Implements

ISOPro consists of three layers — a simulation environment layer with deterministic verifiers, an LLM agent layer, and a communication wrapper managing state, evaluation, and feedback loops — and four mechanisms that collectively instantiate GCE:

### 1. Gradient descent on correct reasoning traces
When the model produces a verified-correct answer, ISOPro runs a forward pass with prompt tokens masked (labels set to `-100`) and computes loss only on the generated tokens. The gradient signal *is* the reasoning trajectory that produced correctness. This is process-level supervision: the model is trained on the reasoning, not on correctness as a label.

### 2. Rejection sampling as continuous self-filter
The model generates at high temperature (T = 0.8); a deterministic verifier accepts only correct responses into the replay buffer. Every iteration evaluates capability against ground truth, making training dynamics observable at the granularity that checkpoint evaluation cannot provide.

### 3. Implicit-curriculum replay buffer
Correct rollouts accumulate across iterations. Easy wins dominate early; harder problems enter as capability develops. The curriculum emerges from the model's own trajectory rather than from researcher curation. Ablation shows that removing accumulation drops mean accuracy by 12 pp and inflates seed-to-seed variance by roughly 4×.

### 4. Activation-guided LoRA targeting
Top-K layers are identified by activation probing and receive the LoRA updates (6.6M params = 0.216% of 3.1B). LoRA weights update on **CPU** — the base model stays frozen in quantized form. This is what eliminates the dual-model VRAM constraint imposed by RLHF's KL penalty.

<p align="center">
  <img src="docs/figures/fig2_training_dynamics.png" width="46%" alt="Training dynamics: rollout hit rate and loss" />
  <img src="docs/figures/fig1_buffer_composition.png" width="46%" alt="Implicit curriculum: replay buffer composition" />
</p>
<p align="center"><em>Left: rollout hit rate doubles between iterations 2 and 3, coincident with the sharpest loss decrease — an inflection visible only through continuous evaluation. Right: the implicit curriculum that forms in the replay buffer without researcher curation.</em></p>

<p align="center">
  <img src="docs/figures/fig3_transition_heatmap.png" width="70%" alt="Capability emergence heatmap" />
</p>
<p align="center"><em>Capability emergence: each cell marks the iteration in which a tier first produces correct traces. T2 and T5 remain unreached — the framework honestly reports capability boundaries.</em></p>

---

## ISOPro vs. RLHF vs. GRPO

|                          | **ISOPro (ours)**       | RLHF (standard)           | DeepSeek-R1 GRPO          |
|--------------------------|-------------------------|---------------------------|---------------------------|
| Reward signal            | Deterministic verifier  | Learned reward model      | Deterministic verifier    |
| Stability mechanism      | Rejection sampling      | KL penalty (dual model)   | Group-relative advantages |
| Models in memory         | 1                       | 2+                        | 1                         |
| Trainable parameters     | 0.216% (6.6M)           | 100% (full)               | 100% (full)               |
| Min. memory (reference)  | ~6 GB (3B)              | ~28 GB (7B × 2)           | ~280 GB (70B)             |
| Hardware                 | Consumer laptop         | Data center GPU           | GPU cluster               |
| Reward hacking           | Impossible (by construction) | Predictable          | Impossible (by construction) |

ISOPro and DeepSeek-R1 GRPO converged on the same architectural insight at orders-of-magnitude-different scales: **for verifiable-reward domains, the verifier is the reward signal, and the learned reward model is an unnecessary intermediary.**

---

## Installation

```bash
pip install isopro
```

For the paper's training pipeline (LoRA adapters, MLX backend, OR-Tools verifier):

```bash
pip install "isopro[train]"
pip install mlx mlx-lm ortools   # Apple Silicon; OR-Tools is the ground-truth solver
```

For adversarial / conversation / workflow-simulation features:

```bash
pip install opencv-python stable-baselines3 gymnasium tqdm
```

Optional: if you use the Claude-backed agents,

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

---

## Reproducing the Paper

All experiments run on an Apple M1 with 32GB unified memory. Full pipeline completes in ~90 minutes.

### Main scheduling experiment (Table 2)

```bash
python examples/run_scheduling_experiment.py                  # all five modes
python examples/run_scheduling_experiment.py --mode prompting # zero-shot + 3-shot baselines
python examples/run_scheduling_experiment.py --mode isopro    # ISOPro training loop
python examples/run_scheduling_experiment.py --mode multiturn # multi-turn revision (scope validity)
```

Alternate MLX-native runner (used in the paper's main results):

```bash
python examples/run_isopro_mlx.py
```

IsoZero simulation baseline (no training):

```bash
python examples/run_isozero_scheduling.py
```

### Ablation study (Table 3 / Figure 5)

```bash
python examples/run_ablation_study.py
```

Reproduces: full ISOPro, no chain-of-thought (−8.3 pp), no buffer accumulation (−12.0 pp, 4× variance), and the random-layer control (+0.9 pp, ns). Seeds: 42, 123, 456.

<p align="center">
  <img src="docs/figures/fig_ablation.png" width="80%" alt="Ablation results" />
</p>

### Scheduling domain

Tasks are generated programmatically at six difficulty tiers:

- **T0** — 4-job warmup, dependencies only
- **T1** — sequencing
- **T2** — resource allocation
- **T3** — deadline satisfaction
- **T4** — pairwise composition (two constraints)
- **T5** — full composition (all three; **held out from training**)

Ground truth is produced by an OR-Tools CP-SAT solver. The verifier checks precedence, resource capacity, and deadline satisfaction. Source: `isopro/environments/tasks/scheduling_tasks.py`, `scheduling_verifier.py`, `scheduling_multiturn.py`.

---

## Additional Simulation Modules

ISOPro ships with simulation environments beyond the scheduling domain used in the paper. These are orthogonal to GCE and were developed for earlier work; they remain supported.

<details>
<summary><strong>Adversarial Simulation</strong></summary>

```python
from isopro.adversarial_simulation import AdversarialSimulator, AdversarialEnvironment
from isopro.agents.ai_agent import AI_Agent

adv_env = AdversarialEnvironment(
    agent_wrapper=my_agent,
    num_adversarial_agents=2,
    attack_types=["textbugger", "deepwordbug"],
    attack_targets=["input", "output"],
)
simulator = AdversarialSimulator(adv_env)
results = simulator.run_simulation(
    ["What is the capital of France?", "How does photosynthesis work?"],
    num_steps=1,
)
```
</details>

<details>
<summary><strong>Conversation Simulation</strong></summary>

```python
from isopro.conversation_simulation.conversation_simulator import ConversationSimulator

simulator = ConversationSimulator(
    ai_prompt="You are a customer service agent. Respond politely and professionally.",
)
history = simulator.run_simulation("upset", num_turns=3)
```
</details>

<details>
<summary><strong>Workflow Simulation</strong></summary>

```python
from isopro.workflow_simulation import WorkflowAutomation

WorkflowAutomation(
    video="path/to/workflow.mp4",
    config="config.json",
    output="output_dir",
    logs="logs_dir",
).run()
```
</details>

<details>
<summary><strong>AI Orchestration</strong></summary>

```python
from isopro.orchestration_simulation import OrchestrationEnv
from isopro.orchestration_simulation.components import LLaMAAgent, AnalysisAgent, WritingAgent
from isopro.orchestration_simulation.evaluator import Evaluator

env = OrchestrationEnv()
env.add_component(LLaMAAgent("Research", "analyze AI impact on labor markets"))
env.add_component(AnalysisAgent("Analysis"))
env.add_component(WritingAgent("Writing"))

results = {m: env.run_simulation(mode=m, input_data={"task": task}) for m in ("parallel", "sequence", "node")}
best_mode = Evaluator().evaluate(results)
```
</details>

---

## Core Simulation API

```bash
pip install "isopro[api]"
python -m isopro.api_server
```

Standard response format:

```json
{
  "run_id": "unique-identifier",
  "output": "simulation-specific-output",
  "metadata": { "timestamp": "...", "simulation_type": "..." }
}
```

Endpoints: `GET /healthcheck`, `POST /simulate`, `POST /simulate/reason`, `POST /simulate/qa`, `POST /simulate/adversarial`, `POST /simulate/orchestration`. See `render.yaml` for a Render deployment template.

---

## Repository Layout

```
isopro/
├── training/           # rejection-sampling trainer, replay buffer, GRPO trainer, config
├── environments/       # simulation environments
│   └── tasks/          # scheduling tasks, verifier, multi-turn harness
├── curriculum/         # scheduler for tier progression
├── metrics/            # evaluation
├── backends/           # MLX, Ollama, HF backends
├── rl/                 # RL wrappers (CartPole, car, LLM envs)
├── adversarial_simulation/
├── conversation_simulation/
├── workflow_simulation/
├── orchestration_simulation/
└── api_server.py       # RESTful simulation API

examples/
├── run_scheduling_experiment.py   # Table 2 main experiment
├── run_ablation_study.py          # Table 3 / Figure 5 ablations
├── run_isopro_mlx.py              # MLX-native training loop
└── run_isozero_scheduling.py      # IsoZero baseline
```

---

## Scope and Limitations

- **Verifiable-reward domains only.** Reward hacking elimination is an architectural guarantee of using a deterministic verifier; it does not extend to domains where no verifier exists (safety, style). GCE extends to such domains through rubric-based trajectory assessment, but this is left to future work.
- **Single-domain validation.** Results in the paper are on RCPSP scheduling. Broader validation across tasks, model families, and scales is needed.
- **Small per-tier sample sizes** (3–5 problems). Per-tier accuracies are directional; the main findings are confirmed through multi-seed averaging (n = 3) and ablation.
- **T2 and T5 at 0%** across all configurations suggest resource reasoning requires more iterations, scaffolding, or stronger base models — this is reported honestly rather than papered over.

---

## Citation

If you use ISOPro or the GCE framework, please cite:

```bibtex
@inproceedings{henry2026gce,
  title     = {Beyond Static Snapshots: A Grounded Evaluation Framework for Language Models at the Agentic Frontier},
  author    = {Henry, Jazmia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review}
}

@software{isopro,
  author    = {Henry, Jazmia},
  title     = {{ISOPro}: A Reference Implementation of Grounded Continuous Evaluation},
  year      = {2026},
  url       = {https://github.com/iso-ai/isopro}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Support

Questions, issues, or reproduction problems: please [open an issue](https://github.com/iso-ai/isopro/issues).
