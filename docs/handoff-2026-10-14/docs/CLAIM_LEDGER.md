# Claim ledger (2026-10-14)

| Claim | Status | What would upgrade it? |
|---|---|---|
| The exact Ramanujan trace kernel is a normalized PSD positional Gram kernel | **EXACT** | Proof in `MATH_SPEC.md`; finite oracle in `tests/` |
| Integer-phase position features have identifiability modulus `L/gcd(L,gcd(H))` | **EXACT** for stated features | Mathematical proof and exhaustive finite checks |
| Finite convex mixture of these kernels preserves PSD and unit diagonal | **EXACT** | Algebra, nonnegative normalized weights |
| Cyclotomic descendant weights sum to 1 | **EXACT** | Counting proof + finite tests |
| Baseline-normalized arithmetic feature bank has pointwise squared norm 1 | **EXACT** | Algebra, finite test |
| AHR reports repaired alias cases and calibrated phase reconstruction gains | **SOURCE-REPORTED** in signal-setting only | Re-run its pinned source conditions if needed; not ArithScope results |
| AHR conditional e-value source and derivative of logsumexp | **SOURCE-REPORTED + EXACT ALGEBRA** | Need model-specific null/held-out proof before a routing guarantee |
| ArithScope attention implementation trains and generalizes | **NOT YET TESTED** | Implement, run, log paired benchmarks |
| ArithScope beats RoPE / Fourier / ALiBi / FoPE | **NOT ESTABLISHED** | Fair baselines, repeat across seeds/lengths/cost |
| ArithScope reduces attention from quadratic to linear | **NO CLAIM; CURRENT DENSE DESIGN IS QUADRATIC** | New algorithm, measured implementation, complexity proof |
| ArithScope improves real-world LLM inference | **NOT ESTABLISHED** | Natural-text tasks, actual model integration & latency |
| e-value false-alarm bounds apply to arbitrary softmax heads | **FALSE WITHOUT EXTRA ASSUMPTIONS** | Proper likelihoods, held-out calibration and theorem |
| Information-exclusion homology equals neural semantic capacity | **NOT ESTABLISHED** | Explicit semantics/bridge, counterexamples, capacity metric |
| Number-field self-similar CY guarantees Transformer routing | **NOT ESTABLISHED** | Model-specific representation, mathematically justified transport |

## Promotion policy

Only upgrade a claim when the evidence actually changes. Report **source code exists**, **finite identity passes**, **written proof**, **Lean proof**, **empirical observation**, and **peer-reviewed finding** as different labels. Do not fill in a missing theorem with analogy. Do not interpret synthetic improvements as LLM SOTA.

## Caution on prior assistant chat

Earlier discussion included numerical sample values and rankings. This handoff does not adopt numerical findings absent a reproducible current-turn experiment file. In particular, run the included tests and **use only their actual outcome**. Previous AHR experiment counts remain source-reported with their original conditions, not verified anew as current ArithScope results.
