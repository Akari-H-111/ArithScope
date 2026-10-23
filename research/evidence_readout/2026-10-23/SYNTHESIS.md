# Evidence readout: synthesis after rounds 1–3 (2026-10-23)

Status: development study. Nothing here changes `docs/CLAIM_LEDGER.md`. The
pilot is one seed on development data (data seed 7, seed 0); its numbers are
copied from `pilot/pilot_evidence.md` and carry no intervals or verdicts.

## What was done

| Track | Who | Output |
|---|---|---|
| T9 check script | Claude Sonnet 5.5 via Antigravity CLI | `theory/check_t9.py` (the Antigravity quota ran out after this file; the rest moved to Claude Code CLI Sonnet) |
| T9 statements, proofs, tests | Claude Sonnet via Claude Code CLI | `theory/T9_DRAFT.md`, `theory/test_check_t9.py`, `theory/NOTES.md` |
| Adversarial audit of T9 and theory-vs-pilot consistency | Claude Sonnet via Claude Code CLI (fresh session) | `audit/REVIEW_T9.md`, `audit/CONSISTENCY.md` |
| Opt-in evidence head, tests, pilot suite and report | Claude Sonnet via Claude Code CLI | worktree `../ArithScope-evidence`, branch `research/evidence-readout` (uncommitted) |
| Orchestration, review, pilot run | Claude Opus 5.5 | this file, `PROGRAM.md` |

The evidence head adds, after the final LayerNorm,
`logits_i += W_out · Σ_{j≤i} κ(i−j) onehot(x_j)`, with `W_out` zero-initialized
(bitwise equal to the base model at initialization, tested). Kernels: `arith`
(Σ β_m K_m, unconstrained β), `cosine_fund` (same 16 orders, fundamental
cosines only), `cosine_full` (every reduced harmonic; 41 coefficients),
`cosine_learned` (learned frequencies), `uniform` (κ = 1, a plain count).

## Theory (labels as in T9_DRAFT.md; audit found no wrong theorem)

- **T9a (PROVED).** With κ = D_P = Σ_{m|P} φ(m)/P K_m = 1[P | n] and W = (log ρ) I, the head's softmax is exactly the Bayes posterior of B1 at every position. It escapes the P3 floor because the count is added after the final normalization.
- **T9b (PROVED, exact checks).** D_P lies in span{K_m : m ∈ M} iff every divisor of P is in M. It lies in the span of fundamental cosines of any orders iff φ(P) ≤ 2 (P ∈ {1, 2, 3, 4, 6}). The CRT bank B16 (with order 1) contains D_12 only if orders 6 and 12 are added: **exact counting needs a divisor-closed bank, while softmax comb selection needs only the lcm (T4).** In an unnormalized sum, off-comb leakage δ gives a negative linear margin for some motif once δ > 1/(P−1).
- **T9c (PROVED).** An exactly periodic kernel gives log-odds that grow linearly along each residue class, and an exponential NLL rate wherever the class margin is positive; only the orders dividing P survive the fold. Worst-motif dominance is sufficient. The audit showed that the **random-motif failure probability** is what separates the pilot's arms (being added as T9c.6).
- **T9d (ASYMPTOTIC).** For kernels satisfying T5's non-resonance hypothesis, the linear part of the evidence is the same for every query residue. With two distinct motif symbols, some residue class has no exponential NLL rate, for every W and every bounded rest-of-network term. No finite-length rate; near a resonance the theorem is silent at the pilot's lengths.
- **T9e (PROVED by enumeration of exact probabilities; logs in floating point).** If each observation is copied k times, the count head stops being Bayes. At V = 3, ε = 1/2, k = 2, one distinct vote: expected NLL 1.041979 (head) vs 0.867563 (Bayes). The count head is only as good as the independence assumption behind it.

## Development pilot (one seed, dev data; not confirmatory)

NLL at N = 8,192 (whole sequence), from `pilot/pilot_evidence.md`:

| arm | P = 12 | P = 13 |
|---|---|---|
| Bayes (exact) | 0.0118 | 0.0147 |
| ArithScope alone | 0.032 | 0.037 |
| ArithScope + evidence `arith` | 0.012 | 0.015 |
| ArithScope + evidence `cosine_full` (41 coefficients) | 0.012 | 0.015 |
| ArithScope + evidence `cosine_fund` (16 coefficients) | 0.011 | 5.937 |
| ArithScope + evidence `cosine_learned` | 66.546 | 3.330 |
| ArithScope + evidence `uniform` | 12.280 | 8.732 |
| RoPE alone | 2.109 | 2.280 |
| RoPE + evidence `arith` | 0.012 | 0.015 |
| RoPE + ArithScope | 0.341 | 0.073 |
| RoPE + ArithScope + evidence `arith` | 0.012 | 0.015 |

Accuracy at N = 8,192: every `arith`-kernel arm 0.996–0.997 (Bayes oracle 0.997 / 0.996); RoPE alone 0.423 / 0.334; `cosine_fund` at P = 13 0.602.

What the pilot shows, as observations on one seed:
1. **SGD finds the T9a construction.** The learned β are proportional to D_P's coefficients (cosine 1.000 at both periods), and the effective log-odds scale κ_P(0)·(diag − offdiag of W_out) is 2.84 against log 17 = 2.833 (audit/CONSISTENCY.md). This is the evidence-head analogue of S2.
2. **It follows the Bayes NLL from the training length to 128× beyond it**, at every position bucket (ratio 1.00–1.18 where the Bayes NLL exceeds 10⁻⁶).
3. **The exact basis matters where the theory says.** With the same 16 coefficients, fundamental cosines tie at P = 12 and collapse at P = 13. That matches T9b and the random-motif failure rates the audit computed (about 10⁻⁴ of motifs at P = 12, more than 99.9% at P = 13). The full harmonic bank also works, with 41 coefficients. In the softmax setting (C2) the difference at P = 12 was zero as well, so the prediction pattern carries over, now with a far larger effect at P = 13.
4. **Inexact kernels are worse than no head.** Learned frequencies and the plain count make confident wrong predictions at long lengths (thousands of the 8,192 far positions above 1 nat). An unnormalized readout amplifies kernel error.
5. **Drop-in.** The same head on RoPE alone restores oracle accuracy at N = 8,192, and on RoPE + ArithScope it removes the NLL rise that E6 diagnosed.

What it does not show: anything beyond one seed, two test sequences at N = 8,192 (four at 4,096, 64 at 1,024; a two-motif mean is accurate to about ±0.05–0.07 in accuracy, per the audit), and one synthetic law that is exactly the head's Bayes model (independent noise, known vocabulary, fixed motif); behaviour under dependence (T9e predicts harm); natural data; speed (the head is computed densely here; nothing is claimed about cost).

## Proposed next step: a preregistered E10

> **Update.** The E10 preregistration is [`docs/PREREGISTRATION.md`](../../../docs/PREREGISTRATION.md) §13, built from
> [`E10_SPEC.md`](E10_SPEC.md), with the seed count n = 8 from the spread pilot and the boundary claim B1. The sketch
> below is the earlier outline; where they differ, §13 holds.

Before anything is preregistered, the owner gets a short plain-language summary and decides (repository practice). The draft design:

- **Test sets:** more sequences at N = 8,192 than the pilot's two, since failures are per motif (the motif varies per sequence).
- **Arms** (fresh seeds, new data seed; code committed and pushed before the first run): ArithScope alone; + evidence `arith`; + evidence `cosine_fund`; RoPE alone; RoPE + evidence `arith`. P = 12 and 13. Seed count from a power rule on the pilot spread (the effects are large; the binding comparison is V2 at P = 12, where a tie is predicted).
- **V1 (primary):** + evidence `arith` has lower excess NLL over Bayes than ArithScope alone on the extrapolated positions at N = 8,192, at both periods, with accuracy not lower (the E6–E9 accuracy guard).
- **V2 (the matched control; C2's structure):** at P = 13, `cosine_fund` has higher far NLL and lower far accuracy than `arith`. At P = 12, no difference larger than a preregistered margin is expected.
- **V3 (drop-in):** RoPE + evidence `arith` keeps oracle accuracy at N = 8,192 where RoPE alone does not.
- **Secondary:** learned β cosine to D_P ≥ 0.99; effective scale within 10% of log 17.
- **Boundary (strongly recommended, preregistered as a prediction of harm):** a copied-observation task variant (T9e). The count head should do worse than ArithScope alone there. Reporting where the method fails belongs in the same experiment.

## Integration questions for the owner

1. Commit the branch `research/evidence-readout` (library change, tests, pilot suite, 20 pilot run files) and open a PR, or keep it local until E10 is preregistered?
2. Commit the `research/` directory (the two Codex studies of 2026-10-15 and this one) as a non-ledger exploratory record?
3. Whether to preregister E10 as drafted above, with or without the dependence boundary arm.
