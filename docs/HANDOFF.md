# Handoff: how to continue ArithScope

This is the entry point for anyone, human or agent, resuming the work. It
records what exists, what was decided and why, the exact commands, and the
open questions, in priority order.

## 1. State at handoff

| Area | State | Where |
|---|---|---|
| Exact arithmetic core (stdlib only) | done, tested exhaustively on finite ranges | `arithscope/arith.py`, `tests/test_arith.py` |
| Theory (comb gap, optimal PSD comb, CRT composition, alias partition, RoPE residue-blindness, folding) | written with proofs and evidence labels | `docs/MATH.md` |
| PyTorch modules (ArithScope bias, four backends, baselines, model) | done; 65 tests (one opt-in); causality by token index | `arithscope/`, `tests/` |
| Preregistered confirmatory experiments | complete: 182 runs, 0 failures; C1–C4 supported, S1 and S2 observed | `docs/PREREGISTRATION.md`, `results/runs/`, `docs/RESULTS.md` |
| Post-hoc analyses (labeled) | shift test at the training length; cosine-bank mixtures; attention profile; learned-frequency drift; gate magnitudes; NLL diagnosis; multiplicity sensitivity | `experiments/posthoc_shift.py`, `experiments/posthoc_nll.py`, `experiments/analyze.py`, `experiments/attention_figure.py` |
| First external audit (2026-10-15) | F1–F3 fixed or corrected, with tests; F4 diagnosed post hoc; F5 answered in wording and a multiplicity analysis | `docs/AUDIT_RESPONSE.md`, `docs/audits/` |
| Replays with the current code | every E1 checkpoint re-scored and 11 confirmatory runs retrained, bitwise identical to the stored records | `experiments/replay.py`, `results/replay/` |
| FlexAttention backend | forward verified on CPU (eager and compiled); no CPU backward, GPU untested | `arithscope/flex.py`, `tests/test_attention.py` |
| Overhead benchmark | CPU, one layer, inference and training steps, five backends | `experiments/bench_overhead.py`, `results/overhead/overhead.json` |
| Interactive explainer | standalone page built from committed summaries | `experiments/lab/`, output `docs/lab/index.html` |
| Environment | conda-forge CPU PyTorch; exact lock committed | `environment.yml`, `results/environment/conda-explicit.lock` |
| Claim boundary | current | `docs/CLAIM_LEDGER.md` |
| Original planning package | archived unchanged | `docs/handoff-2026-10-14/` |

## 2. Commands

```bash
# environment: Python >= 3.10; torch >= 2.4 (CPU is enough), numpy, scipy, matplotlib, pytest
make oracle        # stdlib-only exact checks (no torch)
make test          # full suite
make lint          # ruff static checks
make theory        # theory figures + results/summary/theory.json
make pilots        # dev-data pilots + preregistered hyperparameter rule
make e6            # E6 long-length calibration suite + verdicts (40 runs; preregistered in Section 9)
make experiments   # confirmatory suites (resumable; about 3 hours wall-clock with 4 workers)
make analyze       # regenerate all tables/figures and the AUTO blocks in docs/RESULTS.md and README.md
make lab           # rebuild docs/lab/index.html from the committed summaries
make bench         # backend overhead (idle machine; compiled FlexAttention may need LD_PRELOAD, see below)
python experiments/posthoc_shift.py      # post-hoc shift test (needs checkpoints/, which are not committed)
python experiments/posthoc_nll.py        # post-hoc NLL diagnosis at N = 8,192 (needs checkpoints/; about 70 min, 4 workers)
python experiments/posthoc_readout.py    # lowest NLL each model's readout can produce (needs checkpoints/; minutes)
python experiments/attention_figure.py   # attention profile figure (needs checkpoints/)
python experiments/replay.py eval        # re-score E1 checkpoints, compare bitwise with results/runs/
python experiments/replay.py train       # retrain 11 confirmatory runs, compare bitwise with results/runs/
ARITHSCOPE_TEST_COMPILED_FLEX=1 python -m pytest tests/test_attention.py -k flex   # opt-in compiled-kernel test
```

With the conda-forge PyTorch build, compiled kernels need `LD_PRELOAD=$CONDA_PREFIX/lib/libsleef.so`: without it, inductor's CPU vector-ISA probe fails on an undefined SLEEF symbol (details in `docs/EXECUTION_LEDGER.md`).

`checkpoints/` is gitignored, about 0.4 MB per model; rerun `make experiments` to regenerate the checkpoints. To publish the explainer, make the repository public, then enable GitHub Pages (Settings → Pages → deploy from branch `main`, folder `/docs`). It is then served at `/lab/`.

## 3. Decisions already made (do not re-litigate without new evidence)

1. **ArithScope is an additive channel, not a replacement for RoPE.** `K_m` depends only on `gcd(n, m)`, so it cannot express "previous token" (offset 1 shares its gcd class with every unit). That is why the practical variant is `rope + arith`.
2. **`gamma` is zero-initialized.** Every `rope + X` model equals `rope` bitwise at initialization (tested). This keeps comparisons clean and allows drop-in use.
3. **Positional parameters use a separate learning-rate multiplier.** Adam moves a scalar by about `lr` per step, so `|gamma| <= ~lr * steps` in short runs. The multiplier was selected by the preregistered validation rule, never by extrapolation results.
4. **Exact arithmetic, never floating phases.** Tables are indexed by `n mod m` in int64; folded features use `(a (t mod m)) mod m`.
5. **Hypothesis naming.** Confirmatory claims are C1–C4, so they don't collide with the planning package's H1–H4.
6. **Causality is by token index** (`j <= i`) in every backend; positions enter only the biases and RoPE, so they may repeat, restart, or be unsorted (audit F1, `docs/SPEC.md` §4).
7. **Bias memory is bounded by the number of query/key pairs**, never by the numeric span of the positions, and positions must satisfy `|position| < 2^62` (audit F2).
8. **Report NLL next to accuracy.** Oracle-level accuracy did not imply oracle-level NLL here (audit F4).

## 4. Open questions, in priority order

1. **Natural data.** Does the channel help where periodicity is real but imperfect, e.g. fixed-width tables, code indentation, serialized grids, music, or genomic codon structure? Use a legally clear corpus, matched baselines, and report perplexity *and* length extrapolation. This is the most important next step and the one most likely to fail.
2. **Pretrained-model drop-in.** Add zero-initialized ArithScope channels to a small pretrained RoPE model and fine-tune only `gamma` and `alpha`, a few dozen parameters per layer. Measure whether anything changes on periodic or format-heavy data.
3. **Calibration at long lengths.** In models with RoPE, distant keys in the second layer raise the NLL beyond about 1,024 tokens while accuracy holds (post hoc, `docs/RESULTS.md`). The recent-keys limit used there is a diagnostic, not a fix. Untested remedies: train on longer sequences, limit or bias the broad layer's window during training, or give the readout a vote-count signal so that confidence can grow with evidence (`docs/MATH.md`, P3 and the proof sketch after it). **Preregistered as E6** (`docs/PREREGISTRATION.md`, Section 9; `make e6`): a second-layer window equal to the training length, and longer training sequences at the same tokens per step. The vote-count readout is still untested.
4. **Query-conditioned routing.** Make `alpha_h(x_i) = softmax(W x_i)`. This is still exactly foldable, since only the query-side features change. Test it on the mixed-period suite `e5_mixed`, where fixed mixtures fall behind learned frequencies at short lengths (exploratory, 3 seeds).
5. **Bank design.** Compare the prime-power (CRT) bank `B_16` (10 orders, horizon 720,720, rank 55) with `{1..16}` (16 orders, rank 80) under a feature budget.
6. **Finite-N rates for Theorem T5** as a function of RoPE base and head dimension (see the numerical caveat in `docs/MATH.md`).
7. **Kernels.** `backend="flex"` (`arithscope/flex.py`) already expresses every bias as a FlexAttention `score_mod` over a per-head offset table. Its forward pass is verified on CPU, eager and compiled (`flex.compile_kernels()`; with conda-forge PyTorch, preload `$CONDA_PREFIX/lib/libsleef.so` so inductor's vector-ISA probe works). Next: run it on a GPU, check gradients through the captured table against the explicit backend, and measure. For ArithScope alone, a table of `sum(M)` residue entries indexed by `(q_idx - kv_idx) mod m` would make the state independent of `N`. Claim nothing about speed before measuring.
8. **Planning-package extensions** (evidence-aware routing, memory). These remain hypotheses with no code here; see `docs/handoff-2026-10-14/`.

## 5. Traps

- `alpha` gets exactly zero gradient while `gamma = 0`. This is expected.
- An ArithScope head cannot distinguish offsets `n` and `u n` for units `u mod lcm(M)`. That is its inductive bias, not a bug (MATH.md, P1).
- Theorem T5 is asymptotic. Well-optimized RoPE heads with large logits keep a comb for thousands of tokens (MATH.md, numerical note). Do not overstate it.
- The folded backend is not sub-quadratic. It removes the `[N, N]` bias memory, not the quadratic compute.
- The pytorch.org wheel index was blocked in the original container; conda-forge `pytorch=*=cpu_mkl*` worked.
- Run long diagnoses alone. `posthoc_nll.py` needs about 1.6 GB per worker; run next to the two replays, it was killed by the out-of-memory killer once.
- `results/runs/` is immutable, and each run file records the commit that produced it. After changing the library, rerun `experiments/replay.py`; if the replays are no longer bitwise identical, record why in `docs/EXECUTION_LEDGER.md` before claiming that stored results reproduce with the current code.
