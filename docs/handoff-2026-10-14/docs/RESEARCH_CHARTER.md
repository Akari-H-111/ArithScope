# Research charter: computation-layer Transformer work

## Purpose

Public professional goal: make the author's ability to conduct **hands-on LLM/Transformer architecture research** immediately legible. Technical objective: test whether mathematically controlled arithmetic phase representations improve the **attention computation layer**, not prompting, RAG wrappers, or a branded math demo.

## Primary problem

For positions `t,s`, what does a finite arithmetic position encoding actually distinguish? When do multiple integer periods alias? Can an attention head exploit a normalized Ramanujan/cyclotomic relative-position kernel to improve out-of-range, multi-period or distractor-rich retrieval without exceeding matched baseline compute?

The novelty to test is **composition and measured model utility**, not rediscovery of Ramanujan sums, Möbius inversion, Fourier positional features, or the general concept of attention identifiability.

### Three separately assessed research layers

1. **Mathematical layer:** prove exact kernel PSD/boundedness; determine position-feature identifiability modulus; expose nullspaces/collisions and conditions for stable phase reconstruction. These are ordinary algebraic results or new application-specific lemmas; cite both honestly.
2. **Model layer:** train a real Transformer block with arithmetic relative-position bias and, subsequently, evidence-aware routing. Specify content path, positional path, mask semantics, gradients and time/memory complexity.
3. **Empirical layer:** test against competent controls with matched parameters and training schedules, include negative controls and new eval splits. Report measured overhead and failure groups.

## Phase assignments

- **Track A, required:** arithmetic kernel + integer-phase identifiability in attention logit computation.
- **Track B, required comparison:** standard, same-budget Fourier, RoPE, ALiBi and verified newer Fourier PE when feasible.
- **Track C, conditional:** AHR noise-aware Bézout certificates; root-mixture evidence calibrated over genuinely held-out observations.
- **Track D, conditional:** SAMT Parseval features / metric-stable memory / marked-response closure.
- **Track E, presentation:** reproducible technical README, proof table and results, optionally an interactive browser lab after real model evidence.

## Explicit non-goals of v0.1

- Prove a better asymptotic attention complexity; replacing an additive dense logit bias is still quadratic.
- Prove LLM knowledge/reasoning superiority from synthetic phase tests.
- Claim zero hallucinations, RH, full mathematical correctness of generated language, direct operational interpretation of homology, or general softmax confidence calibration.
- Train foundation-scale LLMs before a meaningful matched small-model effect has been established.

## Success definition

First valid outcome is **reproducible model-level evidence**, including negative results. Research success requires either robust improvements under named regimes or a high-quality negative/no-go account delimiting when arithmetic priors do not outperform strong Fourier features.
