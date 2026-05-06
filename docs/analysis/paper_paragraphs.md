# Drop-in paragraphs for the GCE NeurIPS paper

These are written to slot into specific sections of the paper as-is, addressing
the multi-model validation results without overpromising on any single cell.

---

## For §5.7 (Experimental Validation) or §8.1 (Limitations) — the within-subject Llama observation

The cleanest single piece of evidence in the multi-model expansion is a
within-subject comparison on **Llama 3.2 3B Instruct** across the two
verifiable-reward domains. On scheduling, ISOPro regresses (41.7% → 36.1%,
−5.6pp); on MBPP, ISOPro improves (32.0% → 48.0%, +16.0pp). Same model, same
hyperparameters, same number of iterations and rollouts — opposite outcomes.
The replay-buffer composition explains the divergence directly. Llama's
scheduling buffer at iteration 6 was heavily skewed toward two tiers (T0=62
warmup, T3=31 deadlines) with the sequencing tier (T1) at only 25 traces
despite Llama having a 66.7% zero-shot accuracy on T1. SFT on this skewed
distribution amplified T0/T3 patterns and erased T1 capability (66.7% → 0%).
On MBPP, the same model produced a more balanced buffer (T0=66, T1=57, T2=52,
T3=27) with no single tier dominating; ISOPro improved every tier including
the held-out T4 (20% → 40%). The mechanism is the same in both cases — the
implicit-curriculum buffer reflects what the model can already solve — but
its interaction with the base model's *capability profile* determines whether
that curriculum is corrective or destructive. We discuss capability-aware
buffer-weighting variants in §8.3.

## For §7 (Convergence with DeepSeek-R1) — the corrected GRPO comparison framing

Across 3 models (Qwen 2.5 3B, Llama 3.2 3B, Gemma 2 2B) and 2 verifiable-
reward domains (scheduling, MBPP), ISOPro and consumer-budget GRPO-LoRA
produce *mixed* learning Δ from baseline. ISOPro shows the largest gains
(+25.6pp Qwen scheduling, +22.2pp Gemma scheduling, +16.0pp Llama MBPP) and
two regressions (Llama scheduling −5.6pp, Gemma MBPP −4.0pp); GRPO shows
smaller magnitudes in either direction. We do not claim ISOPro dominates
GRPO on accuracy — at consumer-budget hyperparameters with a single seed,
the comparison is closer than the architectural argument would suggest. The
load-bearing claim is that ISOPro's architectural advantages (no learned
reward model, no KL penalty, no dual-model VRAM) hold *while remaining
accuracy-competitive*, and that two ISOPro properties — compositional
generalization on the held-out tier (40% on Qwen and Llama MBPP, vs 0/20%
for the corresponding GRPO cells) and bootstrap-from-zero capability
acquisition (Gemma scheduling 0.0% → 22.2%) — are mechanism-level
properties of the rejection-sampling loop, not hyperparameter wins.

We did not perform an extensive GRPO hyperparameter sweep; results are
reported at TRL/Unsloth defaults (LR 5e-6, G=4, default KL coefficient).
At frontier scale GRPO uses comparable values [DeepSeek-R1, 5e-7 to 1e-5];
optimal small-scale hyperparameters may differ. The +25.6pp gap on Qwen
scheduling specifically may narrow under aggressive GRPO tuning; the
architectural advantages and the compositional-generalization observations
are independent of this.

## For §3 / §5.7 — connecting MBPP cluster tightness to GCE Principle 3

The MBPP results cluster more tightly and at higher means (24–52%) than
scheduling (13–42%). This is consistent with Principle 3 of the GCE
framework (simulation-based agentic assessment): MBPP's per-test-case
verifier feedback is a denser ground-truth signal than scheduling's binary
pass/fail on a final assignment. Both ISOPro and GRPO benefit from the
denser signal, suggesting the principle's value is not method-specific —
richer verifiers improve any verifier-grounded training method. This
observation also reframes the T5 = 0% scheduling result reported in our
original experiment: the held-out tier remaining unreached on scheduling
is consistent with a sparse-feedback domain limitation rather than a
fundamental ISOPro property, since the same method achieves 40% on the
held-out tier under the denser MBPP verifier signal.

## For §8.1 (Limitations) — the buffer-skew failure mode

We observe two distinct ISOPro regressions across our matrix: Llama
scheduling (−5.6pp) and Gemma MBPP (−4.0pp). The Llama failure is the
buffer-skew failure mode described above (under-representation of T1 in
the replay buffer combined with high T1 zero-shot capability). The Gemma
MBPP failure has a different mechanism: the smallest model in our matrix
(2B parameters) shows degradation under unconstrained SFT on accumulated
correct traces, where GRPO's KL anchoring against the base model preserves
zero-shot capability. These are two distinct failure modes, both pointing
at a common research direction: ISOPro's training step lacks an
anti-drift constraint comparable to GRPO's KL penalty. The capability-
aware buffer-weighting variants we propose in §8.3 address the first; a
KL-anchored ISOPro variant (still single-model in inference; reference
logprobs computed via LoRA toggle, as in DeepSeek-R1's GRPO) would
address the second. Either extension preserves the architectural
advantages of the original ISOPro framework.

## For §8.3 (Future Directions) — capability-aware ISOPro variants

The buffer-skew failure mode points at a hyperparameter the original GCE
formulation did not engage with: the bias of the replay buffer relative to
the model's pre-existing capability profile. The implicit curriculum is
genuinely emergent — a property the paper presents as a strength — but its
emergence reflects what the model can *already solve*, not necessarily what
would maximize learning. Three concrete variants of the rejection-sampling
loop preserve ISOPro's architectural properties while addressing this
interaction:

  - **Capability-aware buffer weighting.** Run a baseline eval before
    iteration 1; weight buffer sampling at SFT time by (1 − baseline_acc[tier])
    so already-strong tiers contribute proportionally less to the gradient.
    Prevents reinforcement of pre-existing skills.
  - **Per-tier replay buffers with uniform tier sampling.** Maintain
    separate buffers per tier; sample uniformly across tiers in each
    minibatch regardless of buffer size. Decouples 'which tiers were
    explored' from 'what we train on.'
  - **KL-anchored ISOPro.** Add a small KL penalty to the SFT step against
    the frozen base model's logprobs (computed via LoRA toggle, no extra
    model in memory). Combines verifier-grounded reward with GRPO's
    anti-drift mechanism.

Each variant is a 10-50 line code change against the reference
implementation. The first two are scheduled for the next experimental
release; the third requires a more involved trainer modification.
