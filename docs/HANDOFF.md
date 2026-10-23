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
| E6, calibration at long lengths (second preregistration) | complete: 40 runs, 0 failures, on an Apple M2; L1 and L2 supported, S4 not as predicted | `docs/PREREGISTRATION.md` §9, `results/runs/e6_calibration/`, `results/summary/e6.json` |
| E7, the NLL floor of ArithScope alone (third preregistration) | complete: 30 runs, 0 failures, on the Apple M2; W1 falsified by its rule (P = 12 interval reaches 0), S6 and S7 not as predicted; the direction held in every seed | `docs/PREREGISTRATION.md` §10, `results/runs/e7_floor/`, `results/summary/e7.json` |
| E8, a powered retest of E7's W1 (fourth preregistration) | complete: 84 runs, 0 failures, on the Apple M2; 21 fresh seeds (6–26) from `experiments/e8_power.py`; W2 supported, S8 and S9 as predicted | `docs/PREREGISTRATION.md` §11, `results/runs/e8_power_retest/`, `results/summary/e8.json`, `results/summary/e8_power.json` |
| E9, a paired window dose-response (fifth preregistration) | complete: 30 training runs and 120 window evaluations, 0 failures, on the Apple M2; 15 fresh seeds (27–41) from `experiments/e9_power.py`; R1 supported, S10 and S11 as predicted | `docs/PREREGISTRATION.md` §12, `results/runs/e9_paired_window/`, `results/runs/e9_paired_window_eval/`, `results/summary/e9.json`, `results/summary/e9_power.json` |
| E10, an evidence-count readout (sixth preregistration) | complete: 80 training runs and 160 copied-law evaluations, 0 failures, on the Apple M2; 8 fresh seeds (42–49) from `experiments/e10_power.py`; V1–V4 and the boundary B1 supported, S12–S15 as predicted | `docs/PREREGISTRATION.md` §13, `results/runs/e10_evidence/`, `results/runs/e10_evidence_copy_eval/`, `results/summary/e10.json`, `results/summary/e10_power.json` |
| Evidence head (opt-in) and its theory | `arithscope/evidence.py`: zero-initialized unnormalized count readout after the final LayerNorm, kernels `arith`, `cosine_fund`, `cosine_full`, `cosine_learned`, `uniform`; T9 in `docs/MATH.md` | `tests/test_evidence.py`, `tests/test_t9.py`, `research/evidence_readout/2026-10-23/` |
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
make e7            # E7 ArithScope-only NLL floor suite + verdicts (30 runs; preregistered in Section 10)
python experiments/e8_power.py   # E8 seed count from the E6/E7 spread -> results/summary/e8_power.json
make e8            # E8 powered retest of W1 + verdicts (84 runs; preregistered in Section 11)
make e9            # E9 paired window dose-response: 30 runs, 120 window evaluations, verdicts (Section 12)
python experiments/e10_power.py  # E10 seed count from the development spread pilot -> results/summary/e10_power.json
make e10           # E10 evidence head: 80 runs, 160 copied-law evaluations, verdicts (Section 13)
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
9. **The evidence head is opt-in and zero-initialized.** `W_out = 0` makes a model with the head equal the model without it bitwise at initialization (tested), and with `evidence = None` nothing changes, including run ids (E10, Section 13). It reads raw tokens and adds counts after the final LayerNorm, so it escapes P3's floor by construction; that is its purpose, and also why it needs independent evidence (B1).

## 4. Open questions, in priority order

1. **Natural data.** Does the channel help where periodicity is real but imperfect, e.g. fixed-width tables, code indentation, serialized grids, music, or genomic codon structure? Use a legally clear corpus, matched baselines, and report perplexity *and* length extrapolation. This is the most important next step and the one most likely to fail.
2. **Pretrained-model drop-in.** Add zero-initialized ArithScope channels to a small pretrained RoPE model and fine-tune only `gamma` and `alpha`, a few dozen parameters per layer. Measure whether anything changes on periodic or format-heavy data.
3. **Calibration at long lengths.** In models with RoPE, distant keys in the second layer raise the NLL beyond about 1,024 tokens while accuracy holds (post hoc, `docs/RESULTS.md`). The recent-keys limit used there is a diagnostic, not a fix. Untested remedies: train on longer sequences, limit or bias the broad layer's window during training, or give the readout a vote-count signal so that confidence can grow with evidence (`docs/MATH.md`, P3 and the proof sketch after it). **Tested in E6** (`docs/PREREGISTRATION.md`, Section 9; `make e6`; results in `docs/RESULTS.md`): a second-layer window equal to the training length (L1) and training at length 256 with the same tokens per step (L2) both lower the far NLL at both periods, with accuracy unchanged. Contrary to S4, the windowed model ends *below* ArithScope alone. **Tested in E7** (Section 10; `make e7`): the same window on ArithScope alone lowered its far NLL in every seed at both periods, to near the windowed RoPE + ArithScope model, but the preregistered rule failed at P = 12 (interval reaching 0), because the unwindowed floor varies widely between seeds. W1 is falsified as written. **Retested in E8** (Section 11; `make e8`): with 21 fresh seeds, planned by `experiments/e8_power.py` on E6's and E7's spread, the same window lowered ArithScope alone's far NLL in every seed, with intervals below 0 at both periods (W2 supported). Open: the first layer's share of the floor (a `(64, 64)` window would mostly remove evidence; Section 11.2), how much of the remaining gap to windowed RoPE + ArithScope is real (E8 did not include that arm), separating length from batch size in L2, and the vote-count readout (tested in E10, below). **Dose-response in E9** (Section 12; `make e9`): one training per seed, evaluated with last-layer windows of 64, 256, 1,024, and 8,192 keys; far NLL rises with the log of the window in every seed, mostly between 64 and 256 keys (R1 supported). Open beyond that: what sets each seed's unlimited floor, and windows shorter than the training length. **The vote-count readout, tested in E10** (Section 13; `make e10`): an opt-in evidence head adds kernel-weighted token counts to the logits after the final LayerNorm. With an ArithScope kernel it removes the floor and follows the exact Bayes NLL from the training length to 128× beyond it; SGD finds the T9a construction (kernel proportional to `D_P`, scale near log 17) in every seed; the matched fundamental-cosine kernel fails at P = 13 and ties at P = 12, as T9 predicts; on RoPE alone it restores extrapolation. It does harm when observations are copied (B1): the count treats copies as independent evidence. Open: the head on hidden states rather than raw tokens, unknown or mixed periods, dependence other than copying, and natural data (question 1), where dependence is the expected failure mode.
4. **Query-conditioned routing.** Make `alpha_h(x_i) = softmax(W x_i)`. This is still exactly foldable, since only the query-side features change. Test it on the mixed-period suite `e5_mixed`, where fixed mixtures fall behind learned frequencies at short lengths (exploratory, 3 seeds).
5. **Bank design.** Compare the prime-power (CRT) bank `B_16` (10 orders, horizon 720,720, rank 55) with `{1..16}` (16 orders, rank 80) under a feature budget. T9b adds a constraint: an exact counter for period `P` needs every divisor of `P` in the bank, so `B_16` can select the 12-comb in softmax (T4) but cannot count it in the evidence head.
6. **Finite-N rates for Theorem T5** as a function of RoPE base and head dimension (see the numerical caveat in `docs/MATH.md`).
7. **Kernels.** `backend="flex"` (`arithscope/flex.py`) already expresses every bias as a FlexAttention `score_mod` over a per-head offset table. Its forward pass is verified on CPU, eager and compiled (`flex.compile_kernels()`; with conda-forge PyTorch, preload `$CONDA_PREFIX/lib/libsleef.so` so inductor's vector-ISA probe works). Next: run it on a GPU, check gradients through the captured table against the explicit backend, and measure. For ArithScope alone, a table of `sum(M)` residue entries indexed by `(q_idx - kv_idx) mod m` would make the state independent of `N`. Claim nothing about speed before measuring.
8. **Planning-package extensions** (evidence-aware routing, memory). These remain hypotheses with no code here; see `docs/handoff-2026-10-14/`.

## 5. Traps

- The evidence head is unnormalized. An inexact kernel (learned frequencies, a plain count) is worse than no head at long lengths, because errors grow with the count (E10 development pilot; T9b.4, T9d). Its kernel coefficients start at `1/F`, not 0: with `W_out = 0` they would never move.
- The evidence head assumes independent observations. Copied or bursty evidence makes it put high probability on wrong symbols at short lengths (B1, T9e).

- `alpha` gets exactly zero gradient while `gamma = 0`. This is expected.
- An ArithScope head cannot distinguish offsets `n` and `u n` for units `u mod lcm(M)`. That is its inductive bias, not a bug (MATH.md, P1).
- Theorem T5 is asymptotic. Well-optimized RoPE heads with large logits keep a comb for thousands of tokens (MATH.md, numerical note). Do not overstate it.
- The folded backend is not sub-quadratic. It removes the `[N, N]` bias memory, not the quadratic compute.
- The pytorch.org wheel index was blocked in the original container; conda-forge `pytorch=*=cpu_mkl*` worked.
- Run long diagnoses alone. `posthoc_nll.py` needs about 1.6 GB per worker; run next to the two replays, it was killed by the out-of-memory killer once.
- `results/runs/` is immutable, and each run file records the commit that produced it. After changing the library, rerun `experiments/replay.py`; if the replays are no longer bitwise identical, record why in `docs/EXECUTION_LEDGER.md` before claiming that stored results reproduce with the current code.
