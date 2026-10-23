# MC information audit v1

Status: prospective development protocol, not a confirmatory preregistration.
Written before this branch's computations. The identities and counterexamples
were chosen analytically before running them. Suite name: `mc_information_v1`.

## Question

Does a label-free gradient supply information beyond a specified forward state
that could improve output control? Under what assumptions can accumulated
evidence supply the missing information?

## Fixed checks

1. For vocabulary 3 and replacement noise 1/2, enumerate every observation
   history of lengths 0 through 6. Compare the count posterior with a separate
   product-likelihood posterior using `fractions.Fraction`. Check total
   probability and expected Brier risk exactly. Report expected log loss
   numerically, with exact probability weights.
2. Compare count vectors `(2, 1, 0, ..., 0)` and `(20, 10, 0, ..., 0)` at
   vocabulary 16 and noise 1/2. Compute the loss from their identical shares
   and the exact lower bound on Brier regret for any common prediction.
3. Under a different, explicitly correlated law, repeat one noisy observation
   1, 2, 4, 8, and 32 times. Compare the correct one-observation posterior with
   the posterior that incorrectly assumes independent copies.
4. In float64, check the uniform-target KL gradient identity analytically,
   through autograd, and by central differences with step 1e-6. Require maximum
   analytic/autograd error below 1e-10 and finite-difference error below 1e-7.
5. Check a replication-invariant mean encoder's outputs and all parameter
   gradients under tenfold replication. Separately show that a collision only
   at one parameter value need not imply a gradient collision.
6. Use E6 ArithScope-only checkpoints, seed 1, periods 12 and 13, selected
   before these probes. E8 checkpoints are absent locally. Draw four new
   independent sequences per period, length 256, using Python `random.Random`
   with seed strings `2026101501:<period>:<sequence>`. Query indices are 63,
   127, and 255. Use only each query's causal prefix. Use float64, explicit
   attention, the saved weights, and no optimizer or parameter update.
7. Compute `KL(uniform || prediction)` and its head and second-layer arithmetic
   gate gradients without accepting a clean label as an argument. Compare
   head gradients with the closed form. Compare unchanged full and 64-key
   second-layer predictions after monitoring. Clean labels and the known
   generating period are scoring/oracle information, not learned-controller
   inputs. Save the raw observations and every per-query row.

## Interpretation and cost

These are identity checks and development diagnostics, not a trained controller
or a test of long-context improvement. There are only two selected checkpoints;
query rows share sequences, so they are not independent statistical replicates.
No threshold, hyperparameter, or policy is fitted to these observations. Report
sample means as descriptions, without significance tests or population claims.
The head-gradient identity can be evaluated directly from forward tensors;
no empirical speedup is claimed. Full backward passes and extra forwards count
as additional compute in any later comparison.

## Isolation

All outputs belong here. Parent `results/runs/`, library, claims and protocol
are read-only. No E9 file or checkpoint is used. Parent hashes are recorded
before and checked after; concurrent changes by the main experiment must be
reported rather than reverted. Global tests are required by the root contract;
they run one-threaded at low priority. This is a new, isolated development
suite, not a changed main-suite protocol. Its deviations are logged locally
to honor the owner's explicit instruction to leave the main line untouched.

## Deviations log

None at protocol creation.

