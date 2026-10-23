# Handoff: Delta-Hilbert geometry study

## Outcome

The information/geometry study is complete. Learned-model usefulness remains
OPEN. Start with `REPORT.md` and D2-D7 of `MATHEMATICS.md`. Parent main files
and the metacognition branch are read-only dependencies. Final isolation
evidence is in `verification/isolation.json`.

| Finding | Status | Boundary |
|---|---|---|
| Delta writes are rank-one online regression corrections and unit-key beta=1 is an affine projection | CLASSICAL; PROVED application; NUMERICAL check | Not a global QKV parameter update |
| T3 simplex keys cannot store arbitrary P independent linear values | PROVED rank obstruction; EXACT FINITE examples | Sole key geometry and plain linear state readout |
| Restoring the constant character produces orthogonal arithmetic phase addresses | CLASSICAL identity; PROVED application; EXACT FINITE and NUMERICAL checks | Known period; symmetric nonnegative feature weights |
| Softmax-equivalent scores can have different delta-memory behavior | PROVED conditional example | Rescaling is included; a fixed gamma budget still differs |
| Tensor-product phase blocks eliminate cross-phase content interference | PROVED product identity; NUMERICAL example | Feature/state dimension grows by P |
| Mean delta memory plus count equals additive count memory and the ideal posterior | PROVED under the specified law; checked | Independent stationary noise and known period |
| Fixed gain forgets while cumulative counting resists changes | CLASSICAL variance identity; EXACT counterexample | Not an optimal learned schedule |
| A trained hybrid improves ArithScope on imperfect periodic data | OPEN | No model training or main integration |

## The most useful next decision

Decide whether the target is stationary denoising or adaptive association
tracking. For the former, additive count memory is a mandatory simpler control:
there is no demonstrated need for a delta update. For the latter, specify the
change/dependence model and ask whether controlled forgetting helps while a
separate evidence state prevents false confidence.

A bounded follow-up could compare a phase-blocked additive state, a phase-blocked
delta state, and a hybrid retaining both adaptive and cumulative summaries.
The current work does not show that the hybrid is necessary or useful. Before
training, specify how the period is inferred without test labels, how a key
norm and gain are constrained, what state budget is matched, and which output
loss or tracking metric will decide success. Select gains and readout scales
only from independent development/validation data.

## Integration obligations

1. Keep main default semantics, zero-gate equivalence, and token-index causality.
   A recurrent memory is a new architecture, not a silent backend substitution.
2. Do not reuse asymmetric folded `query_features` as a symmetric RKHS map.
   Use square-root nonnegative mixture weights on both sides; positional gamma
   and write gain beta are different quantities.
3. Distinguish capacity from period horizon. A large lcm does not provide an
   independent linear memory slot for every residue at low feature rank.
4. Budget any constant coordinate, affine value bias, phase count register,
   covariance state, and content tensor product explicitly. The full parent
   feature bank and content projections are not covered by the simplex-only
   rank obstruction.
5. If gating decays the state, the exact stationary running-mean/count proof
   no longer applies unchanged. Uncertainty and forgetting require a model.
6. Feed any justified evidence scale after final normalization if the target is
   asymptotically vanishing NLL. Existing P3 otherwise still applies.
7. Test independent sequences and model seeds, uncertain/mixed periods, and
   dependence or change points. The supplied period is an oracle assumption
   here, not a learned discovery result.
8. Compare a trained method with fair data, seeds, memory and compute budgets,
   a well-chosen fixed baseline, and the additive control. Do not treat the
   untrained fixed-scale mechanism table as a trained-model benchmark.
9. Review current Kalman Delta Networks and earlier fast-weight literature
   before making novelty claims about uncertainty-dependent write gains.

## Reproducibility and local protocol history

Initial parent commit: `5ce34b874584be86b39b97353f6fb76e347248cd`, branch
`prereg/e9-paired-window`. The initial audit pins 535 tracked parent files and
35 files in the earlier metacognition directory. No E9 data or checkpoint is
used. All neural arithmetic is float64; simulations are one-threaded and low
priority. No dependency installation or training occurred.

After initial branch unit tests and before any frozen result or fresh
simulation, the derivation identified additive counts as the necessary simple
control. It was added prospectively; the initial protocol is retained in
`verification/PROTOCOL_initial.md`, and the change is logged in `PROTOCOL.md`.
No seed, length, period, gain, readout scale, or metric changed. The final
protocol hash is embedded in both result files. No failed experiment is hidden.

Frozen source outputs refuse overwrite. Replays check their hashes and exact
JSON equality in the recorded environment. Root tests do not discover these
branch tests; run the explicit command in `README.md`. Validation logs record
the root oracle/full suite, branch tests, lint, replay, and isolation outcomes.

## Delivery and coordination boundary

No parent integration, commit, push, new repository, or message to another chat
was performed. All shared content here is English. The directory is untracked
in the shared checkout and may contribute to concurrent `git_dirty` metadata.
It should be selected explicitly if another chat later prepares a commit.
Use `README.md` for replay and `PROTOCOL.md` for the exact development design.

