# Claim ledger

The public-truth boundary of the project: what may be said, at what evidence level,
and what would change it. Evidence labels as in [`MATH.md`](MATH.md):
PROVED / CLASSICAL / ASYMPTOTIC / NUMERICAL / EMPIRICAL / OPEN. "Finite check"
means exhaustive exact verification on a stated finite range, not a proof.

## Mathematics

| Claim | Status | Evidence |
|---|---|---|
| `K_m(n) = mu(q)/phi(q)` with `q = m/gcd(m,n)`; `K_m` is PSD with unit diagonal and bounded by 1 | CLASSICAL + finite check | MATH L1, L1'; `tests/test_arith.py` |
| Comb gap: off the comb `K_m <= 1/2`, attained iff `6 | m`; `<= 0` for prime powers | PROVED + finite check (m < 400) | MATH L2 |
| The optimal PSD `P`-periodic comb has margin `P/(P−1)`; it is unique and equals a convex Ramanujan mixture | PROVED + exact identity check (P < 80) + LP (P ≤ 36) | MATH T3 |
| Matched fundamental-cosine banks are strictly weaker for every prime period, with a margin ratio of about `P²/(2π²)` | PROVED (primes); NUMERICAL (composites) | MATH C3.1 |
| Selectable combs of a convex ArithScope head are exactly `lcm(S)Z`, `S ⊆ M`; CRT composition has margin `>= 1/k` | PROVED + finite check (P < 300) | MATH T4 |
| No bias in `span{K_m}` separates offsets with equal signatures (alias partition; gcd classes for divisor-closed banks) | PROVED (classical core) + finite check | MATH P1 |
| With fixed content and standard RoPE frequencies, the attention mass on any residue class mod P tends to `1/P` as context grows | ASYMPTOTIC (proof); NUMERICAL illustration; **no rate** | MATH T5 |
| For learned frequencies, T5 needs its joint non-resonance hypothesis; individual irrationality of each frequency is **not** enough | counterexample (analytic + numerical); corrected 2026-10-15 after an external audit | MATH §4; `tests/test_theory_notes.py`; [`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md) F3 |
| A model ending in LayerNorm and a linear readout has an NLL floor `log(1 + sum_j exp(-G_cj)) > 0`, so it cannot follow the Bayes NLL of periodic denoising to 0, with or without RoPE | PROVED (elementary) + randomized check; per-checkpoint readout minimum NUMERICAL | MATH P3; `tests/test_theory_notes.py`; `experiments/posthoc_readout.py` |
| If attention logits depend on the offset only through finitely many arithmetic classes (ArithScope without RoPE), attention outputs converge as votes accumulate: they report vote shares, not counts, and the NLL tends to a constant | ASYMPTOTIC (proof sketch, not given in full) | MATH §4, remark after P3 |
| The ArithScope bias has an exact rank-`sum phi(m)` factorization, so it runs inside fused SDPA without an `[N, N]` tensor | PROVED + numerical check (1e−12, float64) | MATH T7; `tests/` |
| The dense ArithScope bias is bitwise invariant under any common position shift | Implementation property, tested (shifts to `2^61`) | MATH P2; `tests/` |

## Implementation

| Claim | Status | Evidence |
|---|---|---|
| A trainable PyTorch module exists; `gamma = 0` reproduces the base model's outputs **and gradients** bitwise | tested | `tests/test_attention.py::GammaZeroEquivalenceTests` |
| Explicit, SDPA, and folded backends compute the same function (with padding, per-sequence positions, query blocking) | tested to 1e−10 (float64) | `tests/test_attention.py` |
| Causal and padding masks are respected; fully masked rows output 0, never NaN | tested | `tests/test_attention.py` |
| Causality is by token index in every backend: no token sees a later one for repeated, unsorted, packed, or sparse positions, and all backends agree on such layouts | tested; fixed 2026-10-15 (before, the explicit and FlexAttention backends compared positions) | `tests/test_attention.py`; [`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md) F1 |
| Bias evaluation allocates memory bounded by the number of query/key pairs, not by the numeric span of the positions; positions must satisfy `\|position\| < 2^62` | tested; fixed 2026-10-15 | `tests/test_attention.py`; [`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md) F2 |
| The FlexAttention backend (`backend="flex"`) computes the same forward function | tested on CPU, forward only: 1e−12 in float64 (eager), 1e−5 in float32 (compiled kernels, opt-in test) | `tests/test_attention.py::BackendAndMaskTests::test_*flex_backend_matches_explicit` |

## Experiments (synthetic tasks, CPU, tiny models)

Verdicts are computed by `experiments/analyze.py` from the preregistered decision rules; numbers and intervals are in [`RESULTS.md`](RESULTS.md).

| Claim | Status |
|---|---|
| C1: RoPE + ArithScope extrapolates periodic structure where RoPE fails (N = 1,024, both periods) | **supported** (EMPIRICAL; 5/5 seeds per period) |
| C2: at the prime period 13, ArithScope beats the matched cosine bank; no difference at 12 | **supported** (EMPIRICAL); the P = 12 difference is zero, as predicted |
| C3: alias fixture follows the identifiability theory (lcm decides, not bank size) | **supported** (EMPIRICAL; all six banks meet the preregistered checks) |
| C4: no harm on the aperiodic control (lower CI bound above −1 pp) | **supported** (EMPIRICAL) |
| S1: learned frequencies lose accuracy between N = 1,024 and N = 8,192 | **observed** at both periods (secondary) |
| S2: learned mixtures concentrate on divisors of P; at P = 13 on order 13 | **observed**; at P = 13 they coincide with the T3 optimum (secondary; replicates a pilot observation) |
| RoPE + ArithScope keeps its accuracy under absolute shifts up to 10^9 at the training length | **observed** (post hoc, not preregistered) |
| Mixed unlabeled periods (E5) | exploratory, 3 seeds: learned frequencies ahead at short lengths, RoPE + ArithScope ahead at the longest |
| At long lengths, the RoPE-containing variants that keep oracle accuracy (RoPE + ArithScope and RoPE + full Fourier bank at both periods, the matched cosine bank at P = 12) have NLL that rises with length, above ArithScope alone, whose NLL does not rise | **observed** (post hoc; NLL was recorded but not preregistered as a decision metric); diagnosis in [`RESULTS.md`](RESULTS.md) |
| In those models the change of NLL with query position beyond 1,024 tokens runs through distant keys in the second attention layer: limiting that layer to the last 1,024 keys removes it with accuracy unchanged, in all 20 models with RoPE that were diagnosed; limiting the first (comb) layer does not | **observed** (post-hoc intervention; two-layer models, one task, four fresh sequences) | [`RESULTS.md`](RESULTS.md); `results/posthoc/nll_diagnostics.json` |
| ArithScope alone keeps a flat NLL above the Bayes NLL at long lengths, and the floor is not a readout limit | **observed** (post hoc); explained by MATH P3 (proved) and the proof sketch after it | [`RESULTS.md`](RESULTS.md); `results/posthoc/readout_floor.json` |
| The confirmatory verdicts hold after Bonferroni and Holm adjustment for the interval-based claims | **observed** (post-hoc sensitivity analysis) |
| L1 (E6): limiting RoPE + ArithScope's second layer to the last 64 keys (its training length) lowers the NLL on positions 4,096–8,191 at N = 8,192, with accuracy unchanged, at P = 12 and P = 13 | **supported** (EMPIRICAL; second preregistration, `PREREGISTRATION.md` Section 9; 5/5 seeds per period). The window never binds in training, so the weights are those of the unwindowed model |
| L2 (E6): training RoPE + ArithScope at length 256 with the same tokens per step lowers the same NLL, with accuracy unchanged, at both periods | **supported** (EMPIRICAL; 5/5 seeds per period). Confounded with batch size and attention compute |
| E6 secondary predictions: the windowed model's NLL stops rising (S3); longer training leaves a small rise (S5); the windowed model stays above ArithScope alone (S4) | S3 and S5 **as predicted**; S4 **not as predicted**: the windowed model ends below ArithScope alone (interval excludes 0 at P = 12, not at P = 13) |
| W1 (E7): limiting ArithScope alone's second layer to the last 64 keys (its training length) lowers its NLL on positions 4,096–8,191 at N = 8,192, with accuracy unchanged, at P = 12 and P = 13 | **falsified** by the preregistered rule (third preregistration, `PREREGISTRATION.md` Section 10): the interval excludes 0 at P = 13 but not at P = 12. Observed, not claimed: lower in 5/5 seeds at both periods, closing most of the gap to the limited RoPE + ArithScope model. A test-time limit on unchanged weights |
| E7 secondary predictions: the limited ArithScope model stays above the limited RoPE + ArithScope model (S6); E6's limited RoPE + ArithScope model ends below ArithScope alone again (S7) | both **not as predicted**: each interval includes 0 at P = 12; S7's direction holds in 5/5 seeds at both periods |
| W2 (E8, W1 retested with 21 fresh seeds planned by a power rule): limiting ArithScope alone's second layer to the last 64 keys lowers its NLL on positions 4,096–8,191 at N = 8,192, with accuracy unchanged, at P = 12 and P = 13 | **supported** (EMPIRICAL; fourth preregistration, `PREREGISTRATION.md` Section 11; 21/21 seeds per period, new initial weights, training batches, and test data). W1 (E7) stays falsified as recorded. A test-time limit on unchanged weights; two-layer models, one synthetic task |
| E8 secondary predictions: the seed-to-seed spread of far NLL is at least 3× larger without the limit (S8); the limited model's NLL stops rising (S9) | both **as predicted** |

## Not claimed

| Claim | Status |
|---|---|
| Speedups; sub-quadratic or linear attention | **not claimed.** The folded path removes bias memory, not quadratic compute |
| FlexAttention on GPU, and the flex backward pass | **not tested.** FlexAttention has no CPU backward, and no GPU was available |
| Benefits on natural language or in pretrained LLMs | **not tested** |
| State of the art on any benchmark | **not claimed**; synthetic tasks only |
| PSD-ness or calibration of attention weights after softmax | **not claimed** |
| That ArithScope's NLL floor comes *entirely* from its broad second layer, or that vote-share averaging is the mechanism | **not claimed.** W2 (E8) locates part of the floor of these two-layer models in the second layer's distant keys; the P3 remark is a proof sketch, and the first layer was not tested |
| Calibration in general, or any remedy beyond E6's two-layer synthetic setting (e.g. a hard window in models whose upper layers need distant context) | **not claimed.** E6 measures NLL on one synthetic task; accuracy was at the ceiling in every arm |
| Novelty of Ramanujan sums, Fourier position features, or relative biases as such | **not claimed.** These are classical or prior work. The contribution is the specific attention channel, its theory, and the measured comparison. A literature screen (2026-10-14) found no prior Ramanujan-sum attention bias; this is not an exhaustive novelty search |
