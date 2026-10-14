# Handoff: how to continue ArithScope

This is the entry point for anyone, human or agent, resuming the work. It
records what exists, what was decided and why, the exact commands, and the
open questions, in priority order.

## 1. State at handoff

| Area | State | Where |
|---|---|---|
| Exact arithmetic core (stdlib only) | done, tested exhaustively on finite ranges | `arithscope/arith.py`, `tests/test_arith.py` |
| Theory (comb gap, optimal PSD comb, CRT composition, alias partition, RoPE residue-blindness, folding) | written with proofs and evidence labels | `docs/MATH.md` |
| PyTorch modules (ArithScope bias, three backends, five baselines, model) | done; 47 tests | `arithscope/`, `tests/` |
| Preregistered confirmatory experiments | frozen protocol; runs and results committed | `docs/PREREGISTRATION.md`, `results/runs/`, `docs/RESULTS.md` |
| Overhead benchmark | see `results/overhead/` | `experiments/bench_overhead.py` |
| Claim boundary | current | `docs/CLAIM_LEDGER.md` |
| Original planning package | archived unchanged | `docs/handoff-2026-10-14/` |

## 2. Commands

```bash
# environment: Python >= 3.10; torch >= 2.4 (CPU is enough), numpy, scipy, matplotlib, pytest
make oracle        # stdlib-only exact checks (no torch)
make test          # full suite
make theory        # theory figures + results/summary/theory.json
make pilots        # dev-data pilots + preregistered hyperparameter rule
make experiments   # confirmatory suites (resumable; ~3 CPU-hours with 4 workers)
make analyze       # regenerate all tables/figures and the AUTO blocks in docs/RESULTS.md
make bench         # backend overhead (idle machine)
```

## 3. Decisions already made (do not re-litigate without new evidence)

1. **ArithScope is an additive channel, not a replacement for RoPE.** `K_m` depends only on `gcd(n, m)`, so it cannot express "previous token" (offset 1 shares its gcd class with every unit). That is why the practical variant is `rope + arith`.
2. **`gamma` is zero-initialized.** Every `rope + X` model equals `rope` bitwise at initialization (tested). This keeps comparisons clean and allows drop-in use.
3. **Positional parameters use a separate learning-rate multiplier.** Adam moves a scalar by about `lr` per step, so `|gamma| <= ~lr * steps` in short runs. The multiplier was selected by the preregistered validation rule, never by extrapolation results.
4. **Exact arithmetic, never floating phases.** Tables are indexed by `n mod m` in int64; folded features use `(a (t mod m)) mod m`.
5. **Hypothesis naming.** Confirmatory claims are C1–C4, so they don't collide with the planning package's H1–H4.

## 4. Open questions, in priority order

1. **Natural data.** Does the channel help where periodicity is real but imperfect, e.g. fixed-width tables, code indentation, serialized grids, music, or genomic codon structure? Use a legally clear corpus, matched baselines, and report perplexity *and* length extrapolation. This is the most important next step and the one most likely to fail.
2. **Pretrained-model drop-in.** Add zero-initialized ArithScope channels to a small pretrained RoPE model and fine-tune only `gamma` and `alpha`, a few dozen parameters per layer. Measure whether anything changes on periodic or format-heavy data.
3. **Query-conditioned routing.** Make `alpha_h(x_i) = softmax(W x_i)`. This is still exactly foldable, since only the query-side features change. Test it on the mixed-period suite `e5_mixed`.
4. **Bank design.** Compare the prime-power (CRT) bank `B_16` (10 orders, horizon 720,720, rank 55) with `{1..16}` (16 orders, rank 80) under a feature budget.
5. **Finite-N rates for Theorem T5** as a function of RoPE base and head dimension (see the numerical caveat in `docs/MATH.md`).
6. **Kernels.** A FlexAttention `score_mod`, i.e. a table lookup by `(q_idx - kv_idx) mod m`, or a Triton kernel. Measure on GPU before claiming anything.
7. **Planning-package extensions** (evidence-aware routing, memory). These remain hypotheses with no code here; see `docs/handoff-2026-10-14/`.

## 5. Traps

- `alpha` gets exactly zero gradient while `gamma = 0`. This is expected.
- An ArithScope head cannot distinguish offsets `n` and `u n` for units `u mod lcm(M)`. That is its inductive bias, not a bug (MATH.md, P1).
- Theorem T5 is asymptotic. Well-optimized RoPE heads with large logits keep a comb for thousands of tokens (MATH.md, numerical note). Do not overstate it.
- The folded backend is not sub-quadratic. It removes the `[N, N]` bias memory, not the quadratic compute.
- The pytorch.org wheel index was blocked in the original container; conda-forge `pytorch=*=cpu_mkl*` worked.
