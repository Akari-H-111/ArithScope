# Result: orthogonal arithmetic addresses, separate evidence state

Status: conditional theory, exact finite checks, numerical identities, and
untrained mechanism diagnostics. Main claims are unchanged. This report is
generated from the saved JSON files by `render_report.py`.

## The useful bridge

The parent's optimal periodic comb kernel is a simplex Gram matrix of rank
P-1. Restoring the constant character gives

    D_P = [1+(P-1)f*_P]/P = sum_(m|P) phi(m)/P K_m = 1_(P|offset).

Its P residue features are orthonormal, eliminating interference between those
addresses during a delta write. This is a classical kernel identity with a
proved application to recurrent-memory geometry, not a new kernel claim.
After rescaling, softmax cannot distinguish the two kernels, but recurrent
writes can: their ranks and cross-address inner products differ.

The rank obstruction concerns a plain linear readout with the T3-optimal
period features as its only keys. It does not establish an inability of the
parent's full Transformer, full feature bank, residual paths, or nonlinear
readout. An affine value bias can add the missing degree of freedom too.

## Checks of the geometry

The divisor identity and its affine form matched exactly on
**527 residues**, P =
2 through 32.
The following exact-memory example writes scalar value 1 once at each address,
with unit gain, then queries every address. A simplex linear readout always
has zero sum and cannot represent the target constant vector.

| P | Geometry | Effective rank | Mean squared recall error |
|---|---|---|---|
| 3 | simplex | 2 | 1.593750 |
| 3 | orthogonal | 3 | 0.000000 |
| 12 | simplex | 11 | 1.300123 |
| 12 | orthogonal | 12 | 0.000000 |
| 13 | simplex | 12 | 1.295163 |
| 13 | orthogonal | 13 | 0.000000 |

This is an associative-regression counterexample, not a learned denoising score.
The allocated state dimension was equalized by padding the simplex features
with a zero coordinate; the effective ranks still differ.

Using the parent's actual integer-residue primitive features, the maximum
Gram error was **3.75e-16**. Whole-period shifts of 10^12 P preserved
features bitwise. The delta regression step matched autograd with error
**0**; the recurrent and independent triangular QKV
implementations differed by at most **3.33e-16**.
The projection constraint error was **3.33e-16**.
These are float64 identity checks, not performance measurements.

The causal value coefficients in a two-key example are
**[0.75, -0.5]**. A PSD feature geometry does not make
these signed coefficients probabilities. Keys and step gains must also meet
the write-contraction condition; an unnormalized scalar key can amplify error.

## Counts help under the ideal law; DeltaNet is not necessary for them

With orthogonal phase addresses and gain 1/n_r, a delta state stores empirical
symbol shares. Multiplying its readout by n_r reconstructs counts. A simpler
additive state stores counts directly and produces the same posterior. Both
matched exact count logits within **3.41e-13**
on **16 fresh streams**, each length 1,024.

The table below is an **untrained analytic mechanism demonstration**, with
known period, known noise, and declared fixed logit scale log(17). It compares
five mechanism references plus the exact oracle on the same observations and
allocated state budget. It is not a comparison with trained DeltaNet or main
ArithScope models. A larger tuned fixed scale could change these finite-length
scores; no scale was tuned. The count references are engineered to realize the
oracle under the assumed law, so their low loss is not discovery of model
generalization or a calibration guarantee.

Mean scores at N = 1,024 on positions 768-1,023, eight independent streams per
period; ties split uniformly. Per-sequence metrics and all shorter lengths are
saved. These are descriptions, with no significance or population claim.

| P | Analytic mechanism | NLL | Accuracy |
|---|---|---|---|
| 12 | simplex_constant | 1.67006 | 85.693% |
| 12 | orthogonal_constant | 1.58707 | 89.209% |
| 12 | orthogonal_mean | 1.52397 | 100.000% |
| 12 | orthogonal_mean_count | 1.6147e-30 | 100.000% |
| 12 | orthogonal_additive_count | 1.6147e-30 | 100.000% |
| 12 | bayes_oracle | 1.6147e-30 | 100.000% |
| 13 | simplex_constant | 1.63366 | 86.914% |
| 13 | orthogonal_constant | 1.55584 | 89.648% |
| 13 | orthogonal_mean | 1.54119 | 100.000% |
| 13 | orthogonal_mean_count | 1.17994e-20 | 100.000% |
| 13 | orthogonal_additive_count | 1.17994e-20 | 100.000% |
| 13 | bayes_oracle | 1.17994e-20 | 100.000% |

The important comparison is equality of the two count constructions with the
oracle, not a superiority claim for DeltaNet. Allocations are VP fast weights
and P count registers per mechanism; additive counting does not mathematically
need the separate count registers. The fixed-scale mean state can recover the
correct symbol while failing to express the strength of its growing evidence.

These are logical state budgets. Shared oracle counts, raw records, and feature
tables are verification storage; the code is not a production memory benchmark.
Mean gains use the causal count obtained from that shared prefix count array.

## Where selective correction can matter

For independent stationary observations, constant gain
1/4 has a limiting variance-equivalent sample size of
**7**, even as total
visits grow. This is not posterior confidence. After a deterministic change
point, 16 zeros followed by 4 ones, the mean is
**0.200000** while the fixed-gain
state is **0.683594**, closer to
the new value 1. Thus a selective adaptive summary and a cumulative evidence
summary serve different tasks. No optimal gate or change detector was learned.

Concatenating content and phase features leaves cross-phase content inner
product **0.500000**
in the constructed example. Tensor-product phase blocking reduces it to
**0.000000**,
at a content-feature dimension multiplied by P. This is a conditional design
option, with explicit capacity cost, not a trained QKV proposal validated here.

## Continuation decision

The strongest candidate for later review is an explicitly phase-addressed
memory with separate adaptive and cumulative-evidence states. First justify
period inference, dependence assumptions, and the readout. Include additive
count memory as a mandatory simple control. An uncertainty-dependent write
gain is already related to Kalman Delta Networks; do not claim that general
idea as new. The current categorical posterior and its arithmetic addressing
are the specific bounded application.

Proofs, assumptions, and counterexamples: [MATHEMATICS.md](MATHEMATICS.md).
Primary paper versions and scope: [SOURCES.md](SOURCES.md).
Review and integration conditions: [HANDOFF.md](HANDOFF.md).
No learned-model, natural-data, GPU, runtime, general calibration, or main-line
improvement result has been established.
