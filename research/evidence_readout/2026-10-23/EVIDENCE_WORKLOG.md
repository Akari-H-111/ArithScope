# Evidence head: worklog (branch `research/evidence-readout`)

Development machinery only. No claim in `docs/CLAIM_LEDGER.md` is touched; nothing under `results/` changed.
Not committed or pushed.

## Tests before and after

| Command | Before | After |
|---|---|---|
| `make oracle PY=/usr/bin/python3` | 86 tests, OK (55 skipped: no torch) | 107 tests, OK (76 skipped: no torch) |
| `make test PY=$PY` | 85 passed, 1 skipped (opt-in compiled FlexAttention) | 106 passed, 1 skipped (same opt-in) |
| `make lint PY=$PY` | all checks passed | all checks passed |

The 21 new tests are all in `tests/test_evidence.py`. I could not set `OMP_NUM_THREADS=1`: the sandbox asked
for approval on an env-var prefix, so every command ran with the default thread count.

## Changes, file by file

- `arithscope/evidence.py` (new). `EvidenceConfig(kernel, orders, num_frequencies, min_period, max_period)`;
  three kernels as `OffsetBias` subclasses with one "head" (`arith`: unconstrained `beta`, init `1/|orders|`, exact
  `K_m` residue tables via `arith.kernel_table` / `bias._residue_table` / `bias._lookup`; `cosine_learned`: `omega`
  as in `LearnedFrequencyBias`, `amplitude` init `1/F`, float64 phases; `uniform`: no parameters).
  `EvidenceHead` holds `kernel` and `w_out` (a raw zero-initialized `nn.Parameter`, no `nn.Linear`, so no RNG draw).
  `counts()` builds `kappa(pos_i - pos_j)` per query block of 2048 rows (`EVIDENCE_QUERY_BLOCK`), masks `j <= i` by
  token index, zeroes padded keys in the one-hot, and does a matmul with the one-hot counts.
- `arithscope/model.py`. `ModelConfig.evidence: EvidenceConfig | None = None`; `to_dict()` drops the key when None.
  `TinyDecoder` builds the head after `self.apply(self._init_weights)` (excluded from normal init, no RNG use), adds
  `W_out @ E` to the logits after `head(ln_f(x))`, and `positional_parameters()` appends the kernel parameters.
  With `evidence=None` the module tree, state_dict keys, parameter count, outputs and RNG use are unchanged.
- `arithscope/train.py`. `RunSpec.evidence: dict | None = None`, registered in `_LATER_FIELDS` (run ids unchanged).
  `build_model` makes `EvidenceConfig` from the dict (`orders` default to `spec.orders`). New `evidence_snapshot()`;
  `run()` adds a top-level `"evidence"` record key (kernel, `beta` or `omega`/`amplitude`, `w_out` summary:
  shape, diag, diag mean, off-diag mean / abs mean / max abs, Frobenius) only when the head exists.
  `positional_snapshot()` is untouched. Kernel parameters are in the positional optimizer group via
  `positional_parameters()`; `w_out` falls into the normal weight-decay group.
- `experiments/suites.py`. Suite `pilot_evidence` (16 runs: P in 12, 13; arith x {none, arith, cosine_learned,
  uniform}, rope x {none, arith}, rope+arith x {none, arith}); `data_seed = DEV_DATA_SEED`, seed 0,
  `eval_lengths (64, 256, 1024, 4096, 8192)`, hyperparameters from `_hp`. Experiment name `pilot_evidence_p{P}`
  is shared by the arms of one period (same init, same training batches). Registered in `SUITES`.
- `experiments/pilot_evidence_report.py` (new). Reads `results/runs/pilot_evidence/*.json`, regenerates each test
  set (checks the stored digest, exits non-zero on mismatch), computes exact Bayes NLL per position (B1), writes
  `results/summary/pilot_evidence.json` and `.md`. Tables: accuracy and NLL per length with the Bayes row; far NLL
  (second half of the longest length, from `by_position`) vs Bayes far NLL on the same positions; W_out diag/off-diag;
  learned `beta` vs `D_P` (cosine, L1, off-divisor max, `kappa(0)`, `kappa(P)`) and, for `cosine_learned`, the
  frequency nearest `2 pi k / P` with its drift from initialization. `--runs` and `--summary` let it run on a
  scratch directory. Labeled "development pilot, not confirmatory".
- `tests/test_attention.py` (one-token edit). `test_earlier_run_ids_are_unchanged` computes the expected key set as
  `fields(RunSpec) - {"attn_windows"}`; the new field made it skip every stored run and fail its count check, so I
  added `"evidence"` to the subtracted set. Its intent is unchanged.
- `tests/test_evidence.py` (new): default semantics (no module / state_dict keys / `to_dict` key); all 360+ existing
  run ids of E1-E3, E5-E9 exist as filenames in `results/runs/`, and the `model` dict of one run per suite equals
  `build_model(spec).cfg.to_dict()`; zero-init equivalence (bitwise logits and all base-parameter gradients, for 3
  kernels x nope/rope/arith/rope+arith; RNG state after construction equal; initial values); brute-force
  reference for all kernels with batched, shifted, permuted positions and padding; causality by index; padded keys
  contribute nothing; block exactness; bitwise shift invariance at 10**12 (head and full model); the exact Bayes
  construction (`beta_m = phi(m)/P` on divisors, `W_out = log(17) I`, zero base logits, softmax equals
  `rho^c / sum rho^c` to 1e-9 in float64; P = 12 with the default bank and with (1, 2, 3, 4, 6, 12), P = 13);
  optimizer groups; the record layout; the report's Bayes NLL against a brute-force posterior; the pilot suite's
  16 arms and no filename or experiment-name collision with any other suite.

## Smoke run (not part of results)

One spec per arm, `arith`, P = 13, steps 200, eval lengths (64, 1024), development data seed 7, seed 0, written to
the scratchpad (`.../scratchpad/smoke/pilot_evidence/`), not to `results/`. About 7 s of training each.

| arm | acc 64 | acc 1024 | NLL 64 | NLL 1024 | far NLL (512-1023) |
|---|---|---|---|---|---|
| arith + none | 0.659 | 0.975 | 1.253 | 0.214 | 0.120 |
| arith + arith evidence | 0.659 | 0.975 | 1.181 | 0.090 | 0.000 |
| exact Bayes | 0.658 (oracle) | 0.975 | 1.1368 | 0.0805 | ~0 |

The learned `beta` (200 steps) put 3.16 on m = 13 and 0.25 on m = 1 (target 0.923 and 0.077; scale is absorbed by
`W_out`); cosine to `D_P` 0.892; 0.86 max |beta| off the divisors. One seed and 200 steps: an end-to-end check
only. It says nothing about the 2000-step suite.

## Unsure about / deliberately not done

- The shared `experiments/suites.py` name choice means the "none" arms have the same `experiment` as the evidence
  arms, so they are paired by initialization and batches. They are not byte-equal to E1 runs (different data seed).
- Gradient clipping uses the global norm, so at step 0 the clip scale differs between a model with and without the
  head (the head's `W_out` gradient is nonzero). The zero-init guarantee covers logits and per-parameter gradients, not
  the training trajectory; the pilot arms therefore diverge from "none" after the first step by design.
- `w_out` is in the weight-decay group (`weight_decay` 0.01). Not tuned.
- The kernel is evaluated on position offsets (as the biases are), so a position layout that is not
  increasing can give kernel values that do not match token distance; masking stays by token index.
  `uniform` ignores positions entirely.
- `cosine_learned` shift invariance is bitwise too (same int64 offsets), though the spec asked only for `arith`.
- No Makefile target for the pilot; run `python experiments/run_suite.py pilot_evidence --workers 4`, then
  `python experiments/pilot_evidence_report.py`. Nothing was added to docs (`PREREGISTRATION.md`, `CLAIM_LEDGER.md`,
  `HANDOFF.md`): not requested, and the pilot is outside the preregistered protocol until it is run and reviewed.
- Not run: the 16-run suite, `make replay`, `make analyze`. I did not check that `analyze.py` ignores the new
  suite directory (it globs by suite name in the code I saw, but I did not run it).
- The evidence head has no FlexAttention / sdpa / folded interaction: it sits outside the attention backends,
  so `backend` is irrelevant to it. Nothing is claimed or commented about speed.

# Round 2

Supersedes the round 1 numbers where they differ (suite size is now 20, not 16). Still no commit, push, or pilot run.

## Tests

| Command | Round 1 end | Round 2 end |
|---|---|---|
| `make oracle` | 107 tests, OK (76 skipped) | 108 tests, OK (77 skipped) |
| `make test` | 106 passed, 1 skipped | 107 passed, 1 skipped (same opt-in) |
| `make lint` | pass | pass |

## Changes

- `arithscope/evidence.py`: kernels `cosine_fund` (`CosineBankBias.fundamental(orders)`, 16 coefficients) and
  `cosine_full` (`CosineBankBias.full(orders)`, 41 coefficients) via `_CosineBankKernel`. Rows are built as in
  `CosineBankBias` (`(a * r) % m` integer residues, one float64 table). Parameter `coef`, unconstrained, init
  `1/len(components)`. They enter `KERNELS`, so the existing zero-init (bitwise) and 10**12 shift tests cover them.
- `arithscope/train.py`: `evidence_snapshot` records `orders`, `components`, `coef` for both.
- `experiments/suites.py`: `pilot_evidence` is now 20 runs per the orchestrator's list (arith x 6 kernels; rope and
  rope+arith x {none, arith}); everything else unchanged.
- `tests/test_evidence.py`: initial coefficient values; snapshot keys; `CombSpanTests` (float64 least squares of
  `D_P` on the kernels' own residue tables over n in [0, 20160)); pilot-suite test updated to 20 arms.
- `experiments/pilot_evidence_report.py`, new sections per arm and period: NLL - Bayes difference and ratio (only where
  Bayes > 1e-6) per position bucket at N = 1024 and 8192; accuracy vs oracle accuracy per bucket at 1024 (all lengths
  are in the JSON); far-region max NLL and count of positions above 1 nat, with the Bayes counterpart; learned kernel
  `kappa(0..2P-1)`, `kappa(0)`, max |kappa(n)|, n = 1..P-1, and the least-squares fit to `c D_P` on [0, 720) (c and
  `|kappa - c D_P| / |kappa|`); cosine_learned omega nearest `2 pi a / P` for every harmonic with epsilon (to target)
  and drift (from init); W_out diag mean, off-diag mean, difference (log 17 = 2.833 printed). Every table caption
  carries "development pilot, not confirmatory".

## Results of the new test (relative rms residual of D_P, default bank 1..16)

| kernel | coefficients | P = 12 | P = 13 |
|---|---|---|---|
| arith | 16 | 3e-15 (n < 720), 1e-13 (n < 20160) | 3e-15, 1e-13 |
| cosine_full | 41 | 2e-15, 7e-15 | 2e-15, 5e-14 |
| cosine_fund | 16 | 0.408, 0.408 | 0.876, 0.877 |

This matches the orchestrator's 41% / 88%.

## Smoke run 2 (scratchpad `smoke2/`, not results; arith base, P = 12, 200 steps, eval (64, 1024), dev data, seed 0)

| arm | acc 1024 | NLL 1024 | far NLL | far max NLL | far > 1 nat |
|---|---|---|---|---|---|
| none | 0.974 | 0.203 | 0.110 | 0.19 | 0 / 32768 |
| arith | 0.974 | 0.085 | 0.000 | 0.00 | 0 |
| cosine_fund | 0.972 | 0.096 | 0.001 | 4.08 | 14 |
| cosine_full | 0.974 | 0.087 | 0.000 | 0.00 | 0 |
| cosine_learned | 0.614 | 6.218 | 11.96 | 85.6 | 21542 |
| exact Bayes | 0.974 (oracle) | 0.0831 | 0 | 0 | 0 |

These are 200-step, one-seed numbers; they only show that the arms and the report run. Note the `cosine_learned`
arm is overconfident and wrong at long positions already at N = 1024, and `cosine_fund` has a few 14 overconfident
errors: the new report sections exist to expose exactly this.

## Notes

- Far-region per-position statistics need the checkpoint (`checkpoints/pilot_evidence/<run>.pt`, written by
  `run_suite.py`); the report re-evaluates the model on the regenerated test set, and records the difference between the
  recomputed mean NLL and the stored one (`recomputed_mean_nll_minus_stored`) as a sanity check. Without checkpoints
  those fields are null/"n/a". Use `--checkpoints` to point elsewhere.
- Kernel fits for `cosine_learned` are computed too (a non-periodic kernel, so the fit is not expected to be good).
- The kernel is rebuilt in float64 from float32-stored coefficients, so it differs from the trained float32 kernel at
  the 1e-7 level.

# Round 3 (E10 machinery)

Implements `research/evidence_readout/2026-10-23/E10_SPEC.md`. No commit, no push; spread pilot and E10 not run.

## Tests

| Command | Round 2 end | Round 3 end |
|---|---|---|
| `make oracle` | 108 tests, OK (77 skipped) | 133 tests, OK (102 skipped) |
| `make test` | 107 passed, 1 skipped | 132 passed, 1 skipped (same opt-in) |
| `make lint` | pass | pass |

(25 new tests in `tests/test_e10.py`.) Dry runs: `e10_evidence` 50 runs (2 periods x 5 arms x E10_N_SEEDS = 5),
`pilot_e10_spread` 40 runs (2 x 5 x 4).

## Files

- `arithscope/tasks.py`: `CopiedPeriodicDenoising` (kind `periodic_copied`; periods, vocab_size 16, noise 0.5,
  copies; same draws as `PeriodicDenoising`, then leader replacement `r + P (v - v mod k)`; `observed` = leaders;
  oracle counts leaders only); registered in `TASKS`. `periodic_bayes_nll(batch, task)`: exact B1 NLL per position,
  counting `batch.observed` when set, per period group, rho from the task. `PeriodicDenoising` untouched (a test pins
  its `task_dict`); `copies = 1` equals it bitwise (tokens, targets, oracle, digest).
- `experiments/suites.py`: `E10_DATA_SEED = 20261020`, `E10_FIRST_SEED = 42`, `E10_N_SEEDS = 5` (PLACEHOLDER, commented:
  to be set from `e10_power.json`), `E10_SEEDS`, `E10_SPREAD_SEEDS = 1000..1003`, `E10_ARMS`, `e10_arm(spec)`,
  suites `pilot_e10_spread` (dev data seed 7) and `e10_evidence`, both in `SUITES`. Eval lengths (64, 256, 1024, 8192),
  `eval_tokens = eval_tokens_long = 65536`, `_hp("periodic", variant)`. Experiment names `e10_p12/13`, `pilot_e10_p12/13`.
  A test checks no filename or experiment-name collision with any other suite.
- `experiments/e10_copy_eval.py`: as `e9_window_eval.py`. Every checkpoint of `--suite` (default `e10_evidence`, or
  `pilot_e10_spread`) on the copied law (k = 4, the run's periods/vocab/noise) at N = 256 and 1,024 with
  `train.evaluate`, plus the exact copied-law Bayes NLL on the same batch. One record per (checkpoint, length) in
  `results/runs/<suite>_copy_eval/` (`<stem>__N256.json`, `__N1024.json`): checkpoint SHA-256, spec, copied task dict,
  digest, `bayes_nll`, environment (code commit). Resumable, `--workers`, failures to `<stem>.error.txt`.
  `evaluate_checkpoint(ckpt, lengths, out_dir, copies=None)` is the importable core.
- `experiments/e10_power.py`: the spec's rule. V1 inputs are read from the summaries (no recompute needed):
  `e8.json periods[p].far_nll.reference.values` (21 seeds) and `e9.json periods[p].far_nll_by_window["8192"].values`
  (15 seeds), 36 per period. `power` and `sd` are `e9_power`'s, unchanged. Writes `results/summary/e10_power.json`
  (per claim and part: n_values, mean, sd, UCL spread, d, d with point SD, stop reason; claim n; power per part and
  joint at n; assumption check; stop reasons; sensitivity with point SD, target 0.80, effects halved). Options
  `--e8 --e9 --spread-root --out` (defaults: the real paths).
- `experiments/e10_verdicts.py`: V1-V4, B1, S12-S15, sanity, Holm sensitivity; refuses to run unless
  `len(E10_SEEDS) == e10_power.json n_seeds` (also when that is null). Writes `results/summary/e10.json`. Also holds
  the shared loaders/metrics (`load`, `collect`, `run_metrics`, `excess_nll`, `kernel_quantities`, `sanity`) that
  `e10_power.py` imports. Uses `e6_verdicts.ci95`, `holm`, `metrics`, `region` unchanged.
- `tests/test_e10.py`: see the list in the task; plus the copy-eval core on a tiny checkpoint, suite sizes and settings.
- `Makefile`: `e10` target (run_suite e10_evidence -> e10_copy_eval -> e10_verdicts), in `.PHONY` and with help text.

## Commands for the spread pilot (no Makefile target by instruction)

```
python experiments/run_suite.py pilot_e10_spread --workers 4
python experiments/e10_copy_eval.py --suite pilot_e10_spread --workers 4
python experiments/e10_power.py            # -> results/summary/e10_power.json
# then set E10_N_SEEDS in experiments/suites.py to its n_seeds, and run `make e10`
```
`run_suite.py` writes the checkpoints that the copy evaluation needs (`checkpoints/<suite>/`, not committed).

## Smoke (scratchpad `e10smoke/`, never `results/runs/`)

40 tiny runs (1 layer, d_model 16, 30 steps) written under the real spread-pilot file names, each with
`evaluate_checkpoint` at (256, 1024); then `e10_power.py --spread-root` and the verdict functions on them.
`e10_power.py` ran end to end (V1 n needed 8 from the real E8/E9 spread; the other claims stop because untrained
30-step models have the wrong-signed effects, e.g. "arith_ev far NLL exceeds 0.001" and "copy NLL ... wrong side";
the stop path and the >20-seed path are exercised in tests too). `e10_verdicts.sanity` on those records: all True, one
code commit, checkpoint SHA-256 of the copy records equal the on-disk files. The smoke found one bug, fixed: kernel
quantities (cos to D_P, lambda_eff) exist only for the `arith` kernel (`cosine_fund` has no `beta`).

## Deviations from E10_SPEC.md

None known. Choices where the spec is silent (flagging, not improvising a rule):
- `CopiedPeriodicDenoising.copies` has default 4 (E10's k); the spec shows `copies=k`.
- Holm sensitivity: I entered every interval condition, 14 tests (V1 far NLL and the accuracy guard x 2 periods, V2 x 2,
  V3 two conditions at P = 13, V4 two conditions x 2, B1 x 2). "Interval above/below" = 95% two-sided bound.
- V1's `d = |mean D| / spread` is taken literally (no stop on sign); the spec's V1 stop is the arith_ev far-NLL
  assumption check only.
- S12 "lies within +-1 pp": the whole 95% interval inside [-0.01, 0.01]. S13/S14: every seed of `arith_ev` and `rope_ev`
  at both periods; they need the head's `beta` (arith kernel), so `arith_evfund` has no such metric.
- "Checkpoint SHA matches a run of the suite": the SHA of the on-disk `checkpoints/e10_evidence/<run>.pt` for the record's
  `source_run` is compared (run records do not store a hash); the check fails if the checkpoint is missing.
- A failed run or copy evaluation drops its seed only from comparisons that use that arm; if a record is neither present nor
  failed the loader exits, as in E8/E9.
- `pilot_evidence_report.py` still uses its own `bayes_nll`; I did not switch it to the shared function.

## Unsure / notes

- `git status` also shows `docs/PREREGISTRATION.md`, `results/runs/pilot_evidence/` and
  `results/summary/pilot_evidence.*` changed or new; those are not mine (made outside these edits) and I left them.
- `E10_N_SEEDS = 5` is only a placeholder so the suite and tests have a size; `make e10` must wait for the real value.
