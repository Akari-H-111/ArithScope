# ArithScope
## Identifiability-Aware Arithmetic Attention for Transformers

**Research transfer package, 2026-10-14 (Asia/Taipei).**

> **LLM architecture research at the computation layer:** investigate exact arithmetic phase identifiability, cyclotomic/Ramanujan relative-position kernels, evidence-aware routing, and reproducible Transformer benchmarks. This is not a prompt-engineering project or a claim of state-of-the-art LLM performance.

**Suggested repository slug:** `arithscope`. Do not create, push, publish, or change an existing repository without a further explicit user instruction. The name received a limited discovery check, not exhaustive trademark clearance.

### Where to start

1. Read **[CODEX_START_HERE.md](CODEX_START_HERE.md)** and **[AGENTS.md](AGENTS.md)**.
2. Reproduce the included finite mathematical reference checks, no optional dependencies:
   ```bash
   python3 -m unittest discover -s tests -v
   ```
3. Inspect **[docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md)** before importing ideas/code from earlier research.
4. Execute the staged implementation and benchmarks; distinguish math correctness, implementation correctness and measured ML utility.

### Package status

- **Available today:** a checked-in Python standard-library **mathematical reference**, **13 finite tests**, source pointers (pinned repository commit SHAs), candidate theorem statements and proof sketches, stage gates, preregistered evaluation proposals, claim-boundary guidance.
- **Not available today:** a working PyTorch attention layer, an end-to-end Transformer training run, proof that the method beats RoPE/ALiBi/FoPE, a speedup claim, production kernel, a peer-reviewed paper, a public ArithScope GitHub repository.
- **Prior AHR results:** relevant evidence *in a different signal-reconstruction setting*. They are **not** ArithScope experiments.

### Primary hypotheses

- **H1 (position structure):** finite integer-phase feature banks with a proven identifiability modulus can improve robustness on tasks requiring multiple interacting periods, at matched compute and parameter count.
- **H2 (attention bias):** a learnable mixture of normalized Ramanujan-sum kernels, added to content logits, can deliver useful periodic inductive bias relative to comparable Fourier and positional baselines.
- **H3 (routing; optional):** cyclotomic descent priors and independently evaluated candidate-root evidence can improve constrained routing, *if* suitable model-level semantics and well-calibrated likelihoods exist.
- **H4 (memory; optional):** frame-normalized and rank-observed memory channels can improve retention/efficiency *under explicitly measured resource constraints*.

H1/H2 are the **first research track**. H3/H4 are contingent extensions, not prerequisites.

### Technical target

The first model variant must keep standard attention intact except for a testable **additive arithmetic relative-position bias**:

```text
logits[b,h,i,j] = (Q @ K^T)[b,h,i,j]/sqrt(d_head)
                  + gamma[h] * sum_m alpha[h,m] * c_m(i-j)/phi(m)
attention = masked_softmax(logits)
```

Use stable softmax, handle padding and causal masks, expose an exact `gamma=0` equivalence check, and never claim subquadratic complexity for a dense `[N,N]` bias. Candidate alternatives are explicitly relegated to subsequent ablations.

### Handoff contents

| File | Purpose |
|---|---|
| `CODEX_START_HERE.md` | Copy-ready agent instruction and first milestone |
| `AGENTS.md` | Work rules and proof/benchmark boundaries |
| `docs/RESEARCH_CHARTER.md` | Questions, distinction between hypotheses and results |
| `docs/MATH_SPEC.md` | Precise definitions, short derivations, applicability limits |
| `docs/IMPLEMENTATION_PLAN.md` | Stage gates and API contracts |
| `docs/EXPERIMENT_PROTOCOL.md` | Controls, baselines, leakage avoidance, acceptance criteria |
| `docs/SOURCE_PROVENANCE.md` | Source repository/commit/path matrix |
| `docs/NOVELTY_AND_RELATED_WORK.md` | Adjacent published work and novelty guardrails |
| `docs/CLAIM_LEDGER.md` | What may be said today, what needs verification |
| `docs/RISK_REGISTER.md` | Failure modes and stop/repair rules |
| `docs/PUBLIC_POSITIONING.md` | LinkedIn/research README positioning, stage-safe language |
| `reference/arithmetic_reference.py` | Dependency-free math oracle, **not** a model |
| `tests/test_arithmetic_reference.py` | 13 reference tests |
| `templates/EXPERIMENT_RESULT_SCHEMA.json` | Evidence shape for future runs |
| `templates/RESEARCH_DECISIONS.md` | Decision/negative-result log template |
| `SOURCES_MANIFEST.json` | Machine-readable source SHAs |

**Default mode: research and experiment only.** No GitHub writes, papers posted, DOI claims, third-party code reuse, or trained-model performance claims are authorized by this package.
