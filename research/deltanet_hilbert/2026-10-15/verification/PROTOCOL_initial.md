# Delta-Hilbert geometry v1

Suite: `delta_hilbert_geometry_v1`. Status: prospective DEVELOPMENT protocol,
not a main-suite confirmatory preregistration. Written before the calculations
below. The mathematical examples were chosen analytically, not by searching
their observed scores.

## Scope and fixed design

1. Verify the divisor-weighted identity
   `sum_(m|P) phi(m)/P K_m(n) = 1_(P|n)` for every residue at P = 2 through 32.
   Use the unchanged exact arithmetic core and a direct reduced-residue
   count for each totient weight. Compare with the affine transformation of
   the main T3 optimum. These are finite identity checks, not novelty claims.
2. At P = 3, 12, and 13, compute exact Gram ranks, readout sums, and a constant
   value-writing example. Compare the simplex kernel with the orthogonal
   indicator kernel using the same sequence, values, and unit write gain.
3. At P = 3, exactly verify that gain `1/n_r` on orthogonal addresses produces
   observed-symbol shares. Verify a fixed-gain order-dependence example and a
   deterministic change point (16 zero values followed by 4 ones), contrasting
   the running mean and fixed gain 1/4.
4. In float64, verify online squared-regression SGD against the delta update,
   its contraction identity, and recurrent QKV outputs against a separate
   lower-triangular solve. Tolerances are 1e-10. Include a signed-influence
   example, a gain beyond the contraction range, and a simple concatenation
   versus tensor-product counterexample.
5. Build position features through the unchanged `RamanujanBias.key_features`,
   with exact integer residue lookup. For each P = 12, 13, apply symmetric
   square-root mixture weights on both Q and K. Check the simplex and augmented
   Gram matrices against their rational values. A whole-period shift of
   10^12 P must preserve features bitwise.
6. Generate 8 independent sequences per P = 12, 13, vocabulary 16, replacement
   noise 1/2, length 1,024. Use Python seed strings
   `2026101502:<period>:<sequence>`. Every mechanism sees the same tokens;
   clean motif labels are used only after prediction for scoring. There is no
   training, optimizer, validation selection, or threshold search.
7. Compare four analytic mechanisms: simplex keys with fixed gain 1/4,
   orthogonal keys with fixed gain 1/4, orthogonal keys with gain `1/n_r`, and
   that same mean state with count-scaled readout. Logits are `log(17)*readout`,
   with an additional `n_r` multiplier only in the last mechanism. All states
   allocate `16*P` scalar fast weights and P count registers; zero-pad the
   simplex features to dimension P. Report their effective ranks separately.
8. At lengths 128, 256, 512, and 1,024, score each sequence's last quarter.
   Report NLL and accuracy together. Split ties within 1e-9 uniformly for
   every mechanism and oracle. Save raw sequences and per-sequence metrics.
   Verify the count-scaled logits against exact count logits within 1e-9;
   this is an engineered known-period oracle, not learned-model performance.

## Interpretation

The analytic mechanisms are controlled references, not trained DeltaNet
baselines and not parameter-selected competitive results. The purpose is to
separate geometry, gain scheduling, and readout scale. Counts are allocated to
every arm, including fixed-gain controls that do not use them for prediction.
Known period and noise are assumptions available equally to all mechanisms.
The count method has no claim outside the independent stationary law. The
change-point example is a counterexample, not a second benchmark suite.

The dense triangular form is a small verification oracle, not a fast kernel.
No timing, GPU, language-model, general calibration, or main-line speed claim
is evaluated. The parent library and previous research directory remain
read-only; their hashes are checked after execution. No E9 observation or
checkpoint is consumed. Local deviations belong here so the owner's main-line
isolation instruction is respected.

## Deviations log

None at protocol creation.

