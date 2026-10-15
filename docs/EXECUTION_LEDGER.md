# Execution ledger

What was actually run, where, and with what outcome. Statuses: **PASS**, **FAIL**,
**NOT RUN**, **PARTIAL**. Nothing here is copied from earlier research; every
entry was executed in this repository.

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-14 |
| Machine | cloud container, 4 vCPU (x86_64), 15 GB RAM, no GPU |
| OS | Ubuntu 24.04.5 LTS, Linux 6.18 |
| Python (stdlib checks) | 3.13.16 (system) |
| Python (torch work) | 3.13 (conda-forge env) |
| PyTorch | 2.14.1, CPU build `cpu_mkl` from conda-forge (the pytorch.org wheel index was not reachable from this container) |
| NumPy / SciPy / Matplotlib | 2.5.4 / 1.18.1 / 3.11.2 |
| Threads | 1 torch thread per training run, 4 runs in parallel; 4 threads for the overhead benchmark |

## Stage 0: preflight (gate G0)

| Step | Command | Result |
|---|---|---|
| Handoff package integrity | `sha256sum -c` against `PACKAGE_SHA256.json` | **PASS**: 22/22 files match |
| Reference oracle (13 tests) | `python3 -m unittest discover -s tests -v` (stdlib only) | **PASS**: 13/13, exit 0 |
| Oracle copied unchanged | SHA-256 of `reference/*.py`, `tests/test_arithmetic_reference.py` | identical to the package manifest |

## Stage 1: implementation (gate G1)

| Step | Command | Result |
|---|---|---|
| Exact lemma checks + oracle cross-checks | `make oracle` | **PASS**: 25 run, 29 torch tests skipped without torch |
| Full suite: bias, backends, masks, gradients, γ = 0, shift invariance, query blocking, Bayes oracles, training smoke test | `make test` | **PASS**: 54/54 |

One defect found and fixed during G1. The explicit attention path upcast logits to float32 for the softmax, which *downcast* float64 inputs and broke the float64 backend-agreement test (difference 3e−8). Softmax now uses float32 only for half-precision inputs.

## Stage 2: development pilots (dev data, `data_seed = 7`, seed 0)

### `pilot`: periodic denoising, P = 12 / 13, two optimizer settings per variant (36 runs)

Command: `python experiments/summarize.py results/runs/pilot`. Column `val` is validation accuracy at the training length; length columns are accuracy over all positions; the oracle row uses the first run's data.

```text
experiment                         variant                seed    val      64     256    1024   |gamma|max
pilot_p12_lr0.001_m1               alibi                     0  0.537   0.532   0.546   0.556   -
pilot_p12_lr0.001_m1               arith                     0  0.664   0.666   0.871   0.970   1.19
pilot_p12_lr0.001_m1               nope                      0  0.534   0.531   0.542   0.551   -
pilot_p12_lr0.001_m1               rope                      0  0.680   0.676   0.649   0.543   -
pilot_p12_lr0.001_m1               rope+arith                0  0.664   0.659   0.634   0.513   0.83
pilot_p12_lr0.001_m1               rope+fourier_full         0  0.667   0.660   0.627   0.572   -
pilot_p12_lr0.001_m1               rope+fourier_fund         0  0.665   0.659   0.618   0.554   -
pilot_p12_lr0.001_m1               rope+fourier_learned      0  0.680   0.678   0.900   0.980   -
pilot_p12_lr0.001_m1               rope+relbias              0  0.674   0.678   0.687   0.570   -
pilot_p12_lr0.003_m10              alibi                     0  0.535   0.532   0.546   0.556   -
pilot_p12_lr0.003_m10              arith                     0  0.674   0.677   0.906   0.979   7.24
pilot_p12_lr0.003_m10              nope                      0  0.534   0.532   0.543   0.555   -
pilot_p12_lr0.003_m10              rope                      0  0.679   0.677   0.604   0.495   -
pilot_p12_lr0.003_m10              rope+arith                0  0.676   0.678   0.906   0.980   8.36
pilot_p12_lr0.003_m10              rope+fourier_full         0  0.679   0.677   0.907   0.979   -
pilot_p12_lr0.003_m10              rope+fourier_fund         0  0.676   0.678   0.906   0.979   -
pilot_p12_lr0.003_m10              rope+fourier_learned      0  0.676   0.678   0.901   0.977   -
pilot_p12_lr0.003_m10              rope+relbias              0  0.679   0.678   0.835   0.720   -
pilot_p13_lr0.001_m1               alibi                     0  0.539   0.537   0.529   0.550   -
pilot_p13_lr0.001_m1               arith                     0  0.544   0.537   0.552   0.602   1.09
pilot_p13_lr0.001_m1               nope                      0  0.538   0.534   0.524   0.543   -
pilot_p13_lr0.001_m1               rope                      0  0.657   0.655   0.581   0.619   -
pilot_p13_lr0.001_m1               rope+arith                0  0.656   0.655   0.589   0.622   0.83
pilot_p13_lr0.001_m1               rope+fourier_full         0  0.657   0.656   0.591   0.622   -
pilot_p13_lr0.001_m1               rope+fourier_fund         0  0.657   0.655   0.579   0.634   -
pilot_p13_lr0.001_m1               rope+fourier_learned      0  0.666   0.672   0.791   0.713   -
pilot_p13_lr0.001_m1               rope+relbias              0  0.671   0.669   0.667   0.605   -
pilot_p13_lr0.003_m10              alibi                     0  0.538   0.536   0.527   0.548   -
pilot_p13_lr0.003_m10              arith                     0  0.665   0.669   0.889   0.975   6.93
pilot_p13_lr0.003_m10              nope                      0  0.538   0.534   0.526   0.545   -
pilot_p13_lr0.003_m10              rope                      0  0.661   0.661   0.537   0.528   -
pilot_p13_lr0.003_m10              rope+arith                0  0.670   0.671   0.888   0.975   8.20
pilot_p13_lr0.003_m10              rope+fourier_full         0  0.669   0.668   0.890   0.975   -
pilot_p13_lr0.003_m10              rope+fourier_fund         0  0.669   0.672   0.621   0.519   -
pilot_p13_lr0.003_m10              rope+fourier_learned      0  0.669   0.671   0.886   0.974   -
pilot_p13_lr0.003_m10              rope+relbias              0  0.670   0.670   0.790   0.654   -
Bayes oracle                                                            0.679   0.904   0.981
```

### `pilot_controls`: tagged denoising and MQAR (two settings), dense alias fixture P = 24 (22 runs)

Command: `python experiments/summarize.py results/runs/pilot_controls`. Column `val` is validation accuracy at the training length; length columns are accuracy over all positions; the oracle row uses the first run's data.

```text
experiment                         variant                seed    val      64     256    1024   |gamma|max
pilotc_alias_2-3-4-6-12            arith                     0  0.529   0.547   0.644   0.724   6.16
pilotc_alias_3-8                   arith                     0  0.549   0.570   0.813   0.951   8.39
pilotc_mqar_lr0.001_m1             alibi                     0  0.099   0.084   0.042     -     -
pilotc_mqar_lr0.001_m1             arith                     0  0.077   0.075   0.042     -     0.96
pilotc_mqar_lr0.001_m1             nope                      0  0.081   0.073   0.042     -     -
pilotc_mqar_lr0.001_m1             rope                      0  0.074   0.075   0.043     -     -
pilotc_mqar_lr0.001_m1             rope+arith                0  0.074   0.074   0.037     -     0.96
pilotc_mqar_lr0.003_m10            alibi                     0  0.272   0.267   0.083     -     -
pilotc_mqar_lr0.003_m10            arith                     0  0.056   0.075   0.036     -     5.20
pilotc_mqar_lr0.003_m10            nope                      0  0.079   0.068   0.039     -     -
pilotc_mqar_lr0.003_m10            rope                      0  0.085   0.094   0.049     -     -
pilotc_mqar_lr0.003_m10            rope+arith                0  0.058   0.073   0.036     -     6.11
pilotc_tagged_lr0.001_m1           alibi                     0  0.700   0.698   0.897     -     -
pilotc_tagged_lr0.001_m1           arith                     0  0.700   0.701   0.898     -     0.53
pilotc_tagged_lr0.001_m1           nope                      0  0.700   0.700   0.899     -     -
pilotc_tagged_lr0.001_m1           rope                      0  0.703   0.701   0.895     -     -
pilotc_tagged_lr0.001_m1           rope+arith                0  0.701   0.700   0.894     -     0.23
pilotc_tagged_lr0.003_m10          alibi                     0  0.701   0.698   0.896     -     -
pilotc_tagged_lr0.003_m10          arith                     0  0.702   0.701   0.899     -     11.29
pilotc_tagged_lr0.003_m10          nope                      0  0.700   0.702   0.900     -     -
pilotc_tagged_lr0.003_m10          rope                      0  0.699   0.702   0.896     -     -
pilotc_tagged_lr0.003_m10          rope+arith                0  0.696   0.701   0.896     -     2.36
Bayes oracle                                                            0.569   0.812   0.952
```

### `pilot_alias_query`: alias fixture with query tokens, P = 24, training length 64 (2 runs)

Command: `python experiments/summarize.py results/runs/pilot_alias_query`. Column `val` is validation accuracy at the training length; length columns are accuracy over all positions; the oracle row uses the first run's data.

```text
experiment                         variant                seed    val      64     256    1024   |gamma|max
pilotq_alias_2-3-4-6-12            arith                     0  0.234   0.230   0.393   0.472   10.11
pilotq_alias_3-8                   arith                     0  0.288   0.301   0.496   0.500   10.44
Bayes oracle                                                            0.300   0.653   0.908
```

### `pilot_alias_design`: alias fixture with query tokens, P = 12 (and P = 24 at training length 128) (3 runs)

Command: `python experiments/summarize.py results/runs/pilot_alias_design`. Column `val` is validation accuracy at the training length; length columns are accuracy over all positions; the oracle row uses the first run's data.

```text
experiment                         variant                seed    val      64     128     256     512    1024    2048   |gamma|max
pilotd_p12_2-3                     arith                     0  0.293   0.299     -     0.460     -     0.440     -     7.85
pilotd_p12_3-4                     arith                     0  0.456   0.461     -     0.812     -     0.934     -     10.09
pilotd_p24_n128_3-8                arith                     0  0.477     -     0.446     -     0.799     -     0.954   10.00
Bayes oracle                                                            0.460     -     0.811     -     0.950     -
```

### `pilot_mqar_easy`: MQAR with 64 keys / 16 values, 4000 steps (2 runs)

Command: `python experiments/summarize.py results/runs/pilot_mqar_easy`. Column `val` is validation accuracy at the training length; length columns are accuracy over all positions; the oracle row uses the first run's data.

```text
experiment                         variant                seed    val      64     128     256   |gamma|max
pilotm_mqar_k64_v16_4000           rope                      0  0.189   0.193   0.133   0.106   -
pilotm_mqar_k64_v16_4000           rope+arith                0  0.180   0.190   0.143   0.097   4.04
Bayes oracle                                                            1.000   1.000   1.000
```

**Status: PASS.** 65 development runs, 0 failures. Hyperparameters were selected by the preregistered rule (`experiments/select_hparams.py`). Design decisions are recorded in [`PREREGISTRATION.md`](PREREGISTRATION.md), Sections 3 and 6.

## Stage 3: confirmatory runs

Command: `python experiments/run_suite.py e1_periodic e2_alias e3_tagged e5_mixed --workers 4`, started 2026-10-14 15:29 UTC, right after the preregistration commit `cdea384` (15:28:53 UTC). The last run finished at 18:21 UTC.

| Suite | Runs | Failures | Wall time per run (4 in parallel) | Role |
|---|---|---|---|---|
| `e1_periodic` | 90 | 0 | 212–328 s (mean 268 s) | C1, C2, S1, S2 |
| `e2_alias` | 30 | 0 | 160–337 s (mean 188 s) | C3 |
| `e3_tagged` | 35 | 0 | 148–199 s (mean 182 s) | C4 |
| `e5_mixed` | 27 | 0 | 145–197 s (mean 179 s) | exploratory |

**Status: PASS.** 182 runs, all `COMPLETED`, every training loss finite. Wall times include evaluation; E1 evaluates up to N = 8,192 and adds the shift tests. Some runs shared the CPU with analysis jobs (the post-hoc shift evaluation from 17:10 to 17:17 UTC, and short test and benchmark checks), so wall times are indicative only.

**Provenance.** Every run file records the git HEAD at completion. The recorded commits are (hashes after the authorship rewrite of Stage 6; the run files hold the original ones) `88edbef`, `7cec74c`, `2e4519f`, `b42bfa4`, `33f75c7`, `db1f18d`, and `fce8c33`, mostly with `git_dirty = true` because documentation and analysis scripts were being edited. For each of them, `git diff --quiet cdea384 <commit> -- arithscope experiments/suites.py experiments/run_suite.py` succeeds: the library, the suite definitions, and the runner were identical to the preregistration commit for every run. The first change to `arithscope/` (`d430f01`, 18:22:39 UTC) came after the last run.

## Stage 4: post-hoc analyses (labeled in every table and figure)

| Step | Command | Result |
|---|---|---|
| Shift test at the training length | `python experiments/posthoc_shift.py --workers 2` | **PASS**: 90 checkpoints, 17:10–17:17 UTC, `results/posthoc/shift_n64.json` |
| Attention profile, P = 13, seed 1, N = 1,024 | `python experiments/attention_figure.py` | **PASS**: `results/summary/attention_profile.json`, `results/figures/attention_profile_*.png` |
| Drift of learned-frequency heads, gate magnitudes | part of `python experiments/analyze.py` | **PASS** |

Reasons for each addition are in [`PREREGISTRATION.md`](PREREGISTRATION.md), Section 8.

## Stage 5: FlexAttention backend and overhead benchmark (after all runs)

| Step | Command | Result |
|---|---|---|
| Static checks | `make lint` | **PASS** |
| Full test suite | `make test` | **PASS**: 56 passed, 1 skipped (the opt-in compiled-kernel test) |
| Standard-library run | `python3 -m unittest discover -s tests` | **PASS**: 25 run, 32 skipped without torch |
| Compiled FlexAttention test | `ARITHSCOPE_TEST_COMPILED_FLEX=1 LD_PRELOAD=$CONDA_PREFIX/lib/libsleef.so python -m pytest tests/test_attention.py -k flex` | **PASS**: 2 passed |
| Overhead benchmark, first run (`d430f01`, idle machine) | `LD_PRELOAD=$CONDA_PREFIX/lib/libsleef.so python experiments/bench_overhead.py` | **FAIL** for the folded backend: it used more memory than the explicit backend (kept as `results/overhead/overhead_before_folded_padding.json`) |
| Fix | pad folded q, k, v to one head dimension (`3a7a021`) | **PASS**: new exactness test; a single-kernel probe at N = 8,192 went from 4.9 GB to 23 MB |
| Overhead benchmark, rerun (`3a7a021`, idle machine) | same command | **PASS**: `results/overhead/overhead.json` (100 rows; compiled FlexAttention has forward rows only; two dense training-step rows at N = 8,192 skipped, about 6 GB) |

**Toolchain note.** With the conda-forge PyTorch build, inductor's CPU vector-ISA probe fails because a compiled test library cannot resolve `Sleef_expf8_u10`. Compiled kernels then refuse to build ("trans_b requires AVX512 or AVX2"). Preloading the environment's `libsleef.so` resolves the symbol, and compiled FlexAttention then builds and runs on this CPU. Eager-mode FlexAttention does not need the workaround.

**Why the first benchmark failed.** The folded backend appends 55 or 80 feature dimensions to queries and keys but not to values. In this PyTorch build, CPU `scaled_dot_product_attention` used its fused kernel only when queries, keys, and values had the same head dimension (observed, not looked up in the PyTorch source); with wider queries and keys it silently fell back to a path that materializes the attention matrix. Zero-padding fixes it without changing the function.

## Stage 6: authorship rewrite (after all results)

At the repository owner's request, every commit after the initial one was re-attributed to the owner (A. Hayami) with `git filter-branch --env-filter`. Contents, commit dates, and messages are unchanged; for every commit, the rewritten tree is identical to the original tree (checked with `git rev-parse <commit>^{tree}`). Commit hashes changed. All hashes in the documentation now refer to the rewritten history, while run files and `results/overhead/*.json` keep the hashes recorded at run time. [`results/environment/commit_map.tsv`](../results/environment/commit_map.tsv) maps each original hash to its rewritten one, so every provenance check above can be repeated on the new hashes.

## Stage 7: response to the first external audit (2026-10-15)

The audit is archived in [`audits/2026-10-15-first-cold-audit.md`](audits/2026-10-15-first-cold-audit.md); findings and decisions are in [`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md). On top of `0ce3dce`, the library changed in two places: masks are causal by token index in every backend (F1), and bias evaluation is bounded by the number of query/key pairs, with a position range check (F2).

| Step | Command | Result |
|---|---|---|
| New tests against the audited library | `tests/test_attention.py` of this stage, run in a worktree of `0ce3dce` | **FAIL, as intended**, for the four regression tests: backends disagree on repeated positions; a token's output changes with a later token; positions at 2<sup>62</sup> overflow instead of raising `ValueError`; sparse positions try to allocate 32 PB. A fifth new test fails there only because it imports the new `allowed_mask` helper |
| Mutation check | the blocked-evaluation test with the query offset forced to 0 | **FAIL, as intended** |
| Static checks | `make lint` | **PASS** |
| Full test suite | `make test` | **PASS**: 64 passed, 1 skipped (the opt-in compiled-kernel test) |
| Standard-library run | `make oracle` (system Python without torch) | **PASS**: 65 run, 38 skipped; the oracle files are unchanged (SHA-256 `21932d7b…`, `50d46c70…`) |
| Compiled FlexAttention test | `ARITHSCOPE_TEST_COMPILED_FLEX=1 LD_PRELOAD=$CONDA_PREFIX/lib/libsleef.so python -m pytest tests/test_attention.py -k flex` | **PASS**: 2 passed (eager and compiled kernels) |
| Replay, evaluation | `python experiments/replay.py eval --workers 4`, from 00:54 UTC | **PASS**: 180 of 180 evaluations (90 checkpoints at N = 1,024 and 8,192) bitwise identical to the run files, `results/replay/eval.json` |
| Replay, training | `python experiments/replay.py train --workers 4`, from 00:54 UTC | **PASS**: 11 of 11 runs retrained from scratch bitwise identical (loss and validation curves, test and shift metrics, positional parameters), `results/replay/train.json` |
| NLL diagnosis, first attempt | `python experiments/posthoc_nll.py --workers 4`, run next to both replays | **FAIL**: the out-of-memory killer stopped a worker at 3.9 GB resident; no output. The attention probe ran all four 8,192-token sequences as one batch |
| Fix and extension | probe one sequence at a time; add accuracy under the restriction, outlier statistics, per-sequence means, the corrupted-token split, and the vote-share correlation (smoke-tested at reduced length on one checkpoint first) | about 1.6 GB per worker |
| Restarts | three later runs of the same command were stopped by hand, none of them after any result was read: two to add the statistics above, and one at 23 minutes because the session's 90-minute limit on background jobs could expire before the end, and the script wrote nothing until the end | the script now writes one file per checkpoint as it finishes (keyed by the script's hash, under `checkpoints/`) and resumes from them |
| NLL diagnosis | same command, alone and detached from the session, 02:09–03:10 UTC | **PASS**: 30 checkpoints, `results/posthoc/nll_diagnostics.json` |
| Readout limit | `python experiments/posthoc_readout.py` (weights only; 4 random starts × 300 projected-gradient steps per target, after a check that 8–40 starts and 400–2,000 steps give the same values to four digits) | **PASS**: 30 checkpoints, `results/posthoc/readout_floor.json` |
| Range check tightened (after the replays, during the diagnosis) | review found that `abs()` of `-2^63` overflows in int64, so that position passed the check, and that FlexAttention did not run the check; it now compares the minimum and maximum with the bounds, in both paths | **PASS**: the extended test fails on the previous commit at `-2^63` and passes now. In-range positions run the same code as before, so the replays and the diagnosis, which ran the previous code, are unaffected |
| Analysis and explainer | `make analyze`, `make lab` | **PASS** |

**Benchmark not rerun.** For the benchmark's inputs (contiguous positions, no padding), the fixes change only how the boolean mask is built: token indices are compared instead of positions, with the same shapes and operations. Every bias takes the same code path as before, plus a range check that is linear in the number of positions. `results/overhead/overhead.json` (`3a7a021`) remains the reference.

## Stage 8: E6 preparation on a local machine (2026-10-15)

Environment for E6, which runs entirely on this machine ([`PREREGISTRATION.md`](PREREGISTRATION.md), Section 9):

| Item | Value |
|---|---|
| Machine | Apple M2 (4 performance + 4 efficiency cores), 16 GB RAM, no CUDA |
| OS | macOS 27.0.1, arm64 |
| Python / PyTorch | 3.13.16 / 2.14.1, conda-forge osx-arm64 build (the `cpu_mkl` build of `environment.yml` exists only for Linux), created with micromamba 2.9.0 |
| NumPy / SciPy / Matplotlib / ruff | 2.5.4 / 1.18.1 / 3.11.2 / 0.17.0 (pip) |
| Exact lock | [`results/environment/conda-explicit-osx-arm64.lock`](../results/environment/conda-explicit-osx-arm64.lock) (conda packages; ruff is installed with pip) |

| Step | Command | Result |
|---|---|---|
| Checks on the cloned `main` (`216f2d4`) | `make oracle`, `make test`, `make lint` | **PASS**: 65 run (1 skipped); 64 passed, 1 skipped; ruff clean |
| Cross-platform retrain | retrain of `e1_p12__rope+arith__s1` with the stored spec (scratch script, output not kept) | **PARTIAL, as expected across platforms**: 172 s in total (109 s training); accuracy identical to the stored record at every length; training losses within 5 × 10⁻⁸ and test NLL within 1.1 × 10⁻⁵ relative error, not bitwise. E1's bitwise replays hold on x86-64 Linux only |
| Library: optional per-layer key window | `attn_windows` in `ModelConfig` and `RunSpec`; left out of the run id at its default | **PASS**: new tests (`KeyWindowTests`) show that a window equals attention over the last keys, that blocking stays exact, that initialization is unchanged, and that the ids of all 189 current-schema run files are unchanged; a mutation check (window ignored; new field hashed) makes them fail |
| Full checks after the change | `make test`, `make lint`, `make oracle` (system Python without torch) | **PASS**: 68 passed, 1 skipped; ruff clean; 69 run, 42 skipped |
| Feasibility pilot | `python experiments/run_suite.py pilot_e6 --workers 4` (development data, seed 0, evaluated at length 64 only) | **PASS**: 4 of 4 runs; `window` and `baseline` bitwise identical; `results/runs/pilot_e6/` |

The 58 earliest pilot files (`pilot`, `pilot_controls`) were written before `RunSpec` gained some fields, so their file names could not be recomputed from their specs even before this change; the run-id test skips them.


## Stage 9: E6 runs (2026-10-15)

Command: `make e6 PY=<env python> WORKERS=4` (`python experiments/run_suite.py e6_calibration --workers 4`, then `python experiments/e6_verdicts.py`), started 2026-10-15 05:36:46 UTC from a clean working tree at `039a56d`. That commit was pushed to `origin/prereg/long-length-calibration` at 05:36:37 UTC, before the runner started. The verdicts were written at 06:51:43 UTC.

| Suite | Runs | Failures | Wall time per run (4 in parallel) | Training time |
|---|---|---|---|---|
| `e6_calibration` | 40 | 0 | 310–712 s (mean 433 s) | 194–312 s for the length-64 arms; 359–585 s for `long_train` |

**Status: PASS.** 40 runs, all `COMPLETED`, every training loss finite. Machine and environment as in Stage 8; nothing else ran on it.

**Provenance.** Every run file records `git_commit = 039a56d`. The first run to finish records `git_dirty = false`. The other 39 record `true` only because the earlier run files were already in `results/runs/e6_calibration/` and still untracked; `git status` after the suite showed no other change.

**Verdicts** (`experiments/e6_verdicts.py` → [`results/summary/e6.json`](../results/summary/e6.json); rules in [`PREREGISTRATION.md`](PREREGISTRATION.md), Section 9.4):

| Item | Outcome |
|---|---|
| Sanity check: `window` and `baseline` train bitwise alike | **PASS** for all 10 period-seed pairs |
| L1, window remedy | **SUPPORTED** |
| L2, longer training | **SUPPORTED** |
| S3, `window` growth within ±0.02 nats | **as predicted** |
| S4, `window` far NLL above `reference` | **not as predicted** |
| S5, `long_train` growth positive | **as predicted** |

## Stage 10: E7 preparation (2026-10-15)

E7 ([`PREREGISTRATION.md`](PREREGISTRATION.md), Section 10) runs on the Stage 8 machine and environment. The library is unchanged since E6; E7 adds the suite `e7_floor` (`experiments/suites.py`), the frozen verdict script `experiments/e7_verdicts.py`, and `make e7`. No pilot (Section 10.3).

| Step | Command | Result |
|---|---|---|
| Suite definition | `python experiments/run_suite.py e7_floor --dry-run` | 30 runs, 10 per arm; no file name shared with `e6_calibration` |
| Verdict script plumbing | scratch script: E6's run files relabelled as E7 arms in a temporary directory (`baseline` posing as `arith_window`), then `e7_verdicts.main()` | runs end to end and writes every field; output discarded (not E7 evidence) |


## Stage 11: E7 runs (2026-10-15)

Command: `make e7 PY=<env python> WORKERS=4` (`python experiments/run_suite.py e7_floor --workers 4`, then `python experiments/e7_verdicts.py`), started 2026-10-15 07:11:29 UTC from a clean working tree at `c2d793f`. That commit was pushed to `origin/prereg/e7-far-nll-floor` at 07:11:20 UTC, before the runner started. The verdicts were written at 07:55:12 UTC.

| Suite | Runs | Failures | Wall time per run (4 in parallel) | Training time |
|---|---|---|---|---|
| `e7_floor` | 30 | 0 | 208–387 s (mean 335 s) | 134–238 s |

**Status: PASS.** 30 runs, all `COMPLETED`, every training loss finite. Machine and environment as in Stage 8. The Google Drive client used about one CPU core throughout; this changes wall times only (each run is single-threaded and deterministic).

**Provenance.** Every run file records `git_commit = c2d793f`. The first run to finish records `git_dirty = false`. The other 29 record `true` only because the earlier run files were already in `results/runs/e7_floor/` and still untracked; `git status` after the suite showed no other change than those files and `results/summary/e7.json`.

**Verdicts** (`experiments/e7_verdicts.py` → [`results/summary/e7.json`](../results/summary/e7.json); rules in [`PREREGISTRATION.md`](PREREGISTRATION.md), Section 10.4):

| Item | Outcome |
|---|---|
| Sanity check: `arith_window` and `reference` train bitwise alike | **PASS** for all 10 period-seed pairs |
| W1, second-layer window on ArithScope alone lowers its far NLL | **FALSIFIED** (the far-NLL interval reaches 0 at P = 12) |
| S6, `arith_window` far NLL above `window` | **not as predicted** |
| S7, `window` far NLL below `reference` | **not as predicted** |
