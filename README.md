# ArithScope

### Exact arithmetic position channels for Transformer attention

ArithScope gives every attention head a learnable, zero-initialized term built from **Ramanujan sums**:

```
logit(i, j) = <q_i, k_j> / sqrt(d)  +  gamma_h * sum_m alpha_{h,m} * c_m(i - j) / phi(m)
```

The term is computed with exact integer arithmetic. Its periodic structure therefore never drifts: not at 128× the training length, and not at absolute position 10^15. This repository contains:

- **Theory.** Proofs that this channel contains the *optimal* positive-definite periodic attention bias, and that RoPE-style frequency banks are *asymptotically blind* to periodic structure.
- **Implementation.** A drop-in PyTorch module with three numerically equivalent backends, including one that runs inside fused attention kernels.
- **Evidence.** Preregistered, multi-seed experiments against matched baselines.

> **Scope.** This is computation-layer research on attention, run with small models on synthetic tasks on CPU. It makes no claims about natural language, pretrained LLMs, or speed. See [Scope](#scope-and-limitations).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="results/figures/e1_periodic_P13_dark.png">
    <img alt="Accuracy versus evaluation length for periodic denoising with period 13" src="results/figures/e1_periodic_P13_light.png" width="920">
  </picture>
</p>

<!-- AUTO:headline:BEGIN -->
(results pending)
<!-- AUTO:headline:END -->

## The problem

Many inputs carry exact periodic structure: rows of a serialized table or image, bars in music, codons in DNA, fixed-width records. An attention head that should "look every *P* tokens back" needs a positional score that peaks on the comb `{t − P, t − 2P, …}` at **every** distance.

- **Additive learned biases** are clipped beyond the training range.
- **ALiBi** decays monotonically.
- **RoPE** builds scores from incommensurate frequencies `theta_k = b^(−2k/d)`. We prove that the closure of RoPE's phase orbit is then a *connected* torus. By Weyl equidistribution, the attention mass any fixed RoPE head puts on a residue class mod *P* tends to the uniform share `1/P` as the context grows (Theorem T5). Periodic preference is not representable at long range.

## The mechanism

`c_m(n) = sum_{gcd(a,m)=1} exp(2 pi i a n / m)` is an integer. The normalized kernel `K_m(n) = c_m(n)/phi(m)` has the closed form `mu(q)/phi(q)` with `q = m / gcd(m, n)`. Three properties matter for attention:

1. **Sharp, exact combs.** `K_m(n) = 1` iff `m | n`; otherwise `K_m(n) <= 1/2` (Lemma L2). It is computed by table lookup on `n mod m` in int64, with no floating-point phase.
2. **Optimality.** Among *all* positive-definite `P`-periodic biases, the unique best comb (margin `P/(P−1)`) is the convex Ramanujan mixture `sum_{d|P, d>1} phi(d)/(P−1) · K_d` (Theorem T3). The matched Fourier control, with the same periods and parameter count, is up to `~P²/(2π²)` times weaker. Bank-wise, CRT composition selects every period whose prime-power factors are in the bank (Theorem T4).
3. **Fused-kernel friendly.** `K_m(i − j)` factors exactly through `phi(m)` real features. The whole bias folds into extra q/k dimensions, so it runs inside `scaled_dot_product_attention` with `is_causal=True` and no `[N, N]` bias tensor (Theorem T7).

`gamma` starts at 0, so adding ArithScope to a model changes nothing until training moves it. The tests check this bitwise, for outputs **and** gradients.

```python
from arithscope.attention import CausalSelfAttention, PositionConfig

attn = CausalSelfAttention(d_model=64, num_heads=4, pos=PositionConfig.from_name("rope+arith"))
y = attn(x, positions)                      # explicit logits (reference)
y = attn(x, positions, backend="folded")    # same function, fused SDPA, no N x N bias
```

## Theory at a glance

Full statements, proofs, and evidence labels are in [`docs/MATH.md`](docs/MATH.md). Every exact claim is also checked exhaustively on finite ranges with rational arithmetic (`tests/test_arith.py`).

| | Statement | Status |
|---|---|---|
| L2 | Off the comb, `K_m <= 1/2` (attained iff `6 | m`); `<= 0` for prime powers | proved + checked, m < 400 |
| T3 | The optimal PSD periodic comb is unique: margin `P/(P−1)`, a convex Ramanujan mixture | proved + checked, P < 80 |
| C3.1 | The matched cosine bank has margin `2 sin²(π/P)` at prime periods: 9× weaker at P = 13, 44× at P = 29 | proved (primes), LP (composites) |
| T4 | A head's selectable combs are exactly `lcm(S)·Z`; CRT composition has margin `>= 1/k` | proved + checked, P < 300 |
| P1 | No bias in the span separates offsets with equal signature (alias / identifiability classes) | proved + checked |
| T5 | RoPE positional mass on any residue class tends to `1/P` (incommensurate frequencies, Lindemann + Weyl) | asymptotic proof, no rate |
| T7 | Exact rank-`sum phi(m)` factorization, so the bias runs inside fused attention | proved + checked to 1e−12 |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="results/figures/theory_comb_mass_dark.png">
    <img alt="Attention mass on the 13-comb versus context length for several positional mechanisms" src="results/figures/theory_comb_mass_light.png" width="640">
  </picture>
</p>

## Results

Confirmatory experiments are running under the frozen protocol in [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md). This section is generated from the run files when they complete. Development-pilot results are in [`docs/EXECUTION_LEDGER.md`](docs/EXECUTION_LEDGER.md).

## Scope and limitations

- **Synthetic tasks, tiny models, CPU.** The models have 2 layers, `d_model = 64`, and are trained for 2000 steps. These experiments test a mechanism, not language modeling.
- **ArithScope cannot express offset-specific patterns.** Its heads only see `gcd(i − j, m)`, so "previous token" is out of reach; this is why the practical variant is `RoPE + ArithScope`.
- **Not sub-quadratic.** The folded backend removes the `[N, N]` bias memory, not the quadratic attention compute.
- **T5 is asymptotic.** A well-optimized RoPE head with large logits can hold a comb for thousands of tokens before it decays (see the figure above).
- **Natural data is untested.** Periodic structure in real data is noisy and partial; testing it is the first item in [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Reproduce

```bash
# Python >= 3.10, torch >= 2.4 (CPU is enough), numpy, scipy, matplotlib, pytest
make oracle        # exact arithmetic checks, standard library only
make test          # 54 tests: oracle match, masks, gradients, gamma = 0 equivalence, backends, Bayes oracles
make theory        # theory figures
make experiments   # preregistered suites (~3 CPU-hours with 4 workers; resumable)
make analyze       # regenerates every table and figure from results/runs/*.json
```

Each run writes one immutable JSON file: config, environment, test-set hashes, training curve, per-length metrics next to the exact Bayes oracle, and learned positional parameters. All runs, including failures, are committed under `results/runs/`.

## Repository map

| Path | Contents |
|---|---|
| `arithscope/arith.py` | exact integer arithmetic (no dependencies): Ramanujan sums, kernels, optimal combs, alias classes |
| `arithscope/bias.py` | ArithScope bias (dense and folded) and the baselines: cosine banks, learned frequencies, clipped relative bias, ALiBi |
| `arithscope/attention.py`, `model.py`, `rope.py` | attention with three backends; minimal decoder |
| `arithscope/tasks.py`, `train.py` | tasks with exact Bayes oracles; training and evaluation harness |
| `arithscope/theory.py` | linear programs and comb-mass computations behind the theory figures |
| `docs/` | [`MATH.md`](docs/MATH.md), [`PREREGISTRATION.md`](docs/PREREGISTRATION.md), [`RESULTS.md`](docs/RESULTS.md), [`CLAIM_LEDGER.md`](docs/CLAIM_LEDGER.md), [`SPEC.md`](docs/SPEC.md), [`HANDOFF.md`](docs/HANDOFF.md), [`EXECUTION_LEDGER.md`](docs/EXECUTION_LEDGER.md) |
| `reference/` | the original standard-library oracle from the planning package, kept byte-identical |
| `experiments/` | suites, runner, analysis, plots, overhead benchmark |

## Related work

- **Position encodings and length generalization:** RoPE (Su et al., 2021), ALiBi (Press et al., 2022), T5 relative bias (Raffel et al., 2020), NoPE (Kazemnejad et al., 2023), FIRE (Li et al., 2024), FoPE (Hua et al., 2025).
- **RoPE numerics at long context:** Wang et al. (2024, BF16 breaks RoPE); Liu (2026, RoPE as phase modulation).
- **Ramanujan sums in signal processing:** Vaidyanathan (2014).
- **Prime-indexed Fourier features for integer *values*:** Hwang, Bae & Lee (2026).

A literature screen (2026-10-14) found no prior use of Ramanujan sums as an attention bias. That screen is not exhaustive, and Ramanujan sums, Fourier position features, and relative biases are each classical; the contribution is their combination into this attention channel, its theory, and the measured comparison. Full references are in [`docs/MATH.md`](docs/MATH.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
