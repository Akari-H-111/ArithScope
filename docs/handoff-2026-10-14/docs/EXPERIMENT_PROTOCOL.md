# Proposed preregistration: attention experiments

**This file is a PRE-EXPERIMENT PROTOCOL, not a results report.** Fix actual resource limits and baseline implementations before freezing datasets.

## H1/H2 head-to-head protocol

Model families:

1. Standard Transformer with chosen baseline position mechanism.
2. Matched additive Fourier bias using identical number of feature terms/parameters.
3. RoPE / ALiBi control variants; identify if attention kernels differ across implementations.
4. ArithScope: normalized Ramanujan mixture (order bank frozen initially), gamma trainable.
5. Ablations: (a) random alpha fixed; (b) learned alpha; (c) shuffled position differences; (d) no gamma; (e) order sets with gcd alias; (f) equal-parameter sinusoidal bank.

All train/eval examples should use common canonical input semantics and comparable resource budgets; report deviations. Random control must be realistic, not intentionally crippled.

## Four datasets (small, generated from code)

**T1 Modular delayed copy:** a query at position i requests earlier token at modular offset family. Many distractors share content but different residues. Vary period train/test split, retrieval distance and token identities independently. Distinguish copying ability from simply recognizing a periodic token label.

**T2 Interacting periods:** target depends on conjunction of two or three periodic constraints, including `H={6,10}` / `{6,10,15}` and `L=120` collision fixtures. Hold out selected period combinations or longer sequence positions, not only random examples.

**T3 Aperiodic negative control:** same token/compute distribution but target keys chosen using nonperiodic rule. This checks whether arithmetic prior harms general-purpose retrieval.

**T4 Natural-text sanity task:** only after robust T1/T2; use a small, legally obtained fixed corpus and an explicitly identified tokenizer, reproducible split, perplexity/loss and context length generalization. Do not claim LLM language superiority from synthetic tests.

Also isolate the AHR waveform alias fixture near nominal f0=8000/29 as a **separate signal-geometry benchmark**, not literal token-position evidence. If imported, read its exact parity/aperiodic sampling conditions.

## Preliminary training/evaluation matrix

- Sequence lengths: e.g. train 128/256; test 128/256/512/1024 and further if hardware allows. Long-context evaluation must use same decoding/masking semantics as training.
- At least 3 independent random seeds for pilot and 5 for confirmatory claims if feasible. Paired runs share seeds, token distributions and optimization budgets.
- Keep device fixed within a comparison (CPU / Apple MPS / CUDA), report actual backend, `torch` and driver versions, FP dtype, compiled/eager path, batch size, gradient accumulation, train steps and wall-time.
- Optimize on validation only; freeze configs/data generation before fresh test. Do not re-label consumed validation as new confirmation.

## Metrics

Required: accuracy per task/period group; sequence loss or perplexity as appropriate; retrieval accuracy vs distance; out-of-range degradation; worst-case alias group; train and inference wall-time (warmup separated); tokens/sec; peak measured memory; parameter count; activation size and actual attention storage footprint; gradient finiteness and convergence; seed-wise deltas vs Fourier/standard controls.

Statistical reports: show sample counts, paired per-seed effect sizes and confidence intervals or exact counts; distinguish statistical and practical significance; include negative/noise cases and failure subgroup plots.

## Acceptance / stop rules

**G0**: 13 independent reference tests pass as actually executed.
**G1**: additive arithmetic bias passes kernel/oracle checks, attention mask and gradient tests; gamma=0 comparison passes.
**G2**: each listed control trains and evaluates successfully under matched resource and data conditions.
**G3**: for a performance conclusion, the candidate improves a pre-selected relevant task relative to a strong matched Fourier control across seeds without erasing nonperiodic performance or hiding runtime cost. If not, report a constrained negative result and diagnose the failure.
**G4**: publishable claim requires versioned code/data generator, frozen config, independent replay and explicit limitations. No numeric threshold for a 'win' is invented after seeing data.

Synthetic tasks alone never qualify as 'state-of-the-art LLM' benchmarks. A model that wins by changing sampling geometry rather than its arithmetic kernel must be reported that way.
