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
| The ArithScope bias has an exact rank-`sum phi(m)` factorization, so it runs inside fused SDPA without an `[N, N]` tensor | PROVED + numerical check (1e−12, float64) | MATH T7; `tests/` |
| The dense ArithScope bias is bitwise invariant under any common position shift | Implementation property, tested (shifts to `2^61`) | MATH P2; `tests/` |

## Implementation

| Claim | Status | Evidence |
|---|---|---|
| A trainable PyTorch module exists; `gamma = 0` reproduces the base model's outputs **and gradients** bitwise | tested | `tests/test_attention.py::GammaZeroEquivalenceTests` |
| Explicit, SDPA, and folded backends compute the same function (with padding, per-sequence positions, query blocking) | tested to 1e−10 (float64) | `tests/test_attention.py` |
| Causal and padding masks are respected; fully masked rows output 0, never NaN | tested | `tests/test_attention.py` |

## Experiments (synthetic tasks, CPU, tiny models)

| Claim | Status |
|---|---|
| C1: periodic extrapolation vs RoPE | pending (see `RESULTS.md` after the confirmatory runs) |
| C2: prime-period advantage over the matched cosine bank | pending |
| C3: alias fixture follows the identifiability theory | pending |
| C4: no harm on aperiodic tasks | pending |

## Not claimed

| Claim | Status |
|---|---|
| Speedups; sub-quadratic or linear attention | **not claimed.** The folded path removes bias memory, not quadratic compute |
| Benefits on natural language or in pretrained LLMs | **not tested** |
| State of the art on any benchmark | **not claimed**; synthetic tasks only |
| PSD-ness or calibration of attention weights after softmax | **not claimed** |
| Novelty of Ramanujan sums, Fourier position features, or relative biases as such | **not claimed.** These are classical or prior work. The contribution is the specific attention channel, its theory, and the measured comparison. A literature screen (2026-10-14) found no prior Ramanujan-sum attention bias; this is not an exhaustive novelty search |
