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

Pending: the confirmatory suites (`make experiments`) start after the preregistration commit.
