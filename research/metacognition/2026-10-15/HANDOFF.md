# Handoff: isolated metacognition information audit

## Outcome and status

The local study is complete as an information audit and a small frozen-model
diagnostic. The trained-controller hypothesis remains OPEN. Parent files were
treated as read-only; the final hash audit is `verification/isolation.json`.
This directory is deliberately not connected to main-suite generation,
analysis, test discovery, or claim publication.

Read `REPORT.md`, then `MATHEMATICS.md` M2-M6. Evidence levels are explicit:

| Statement | Level | Limit |
|---|---|---|
| Uniform-KL head gradient is `(p-u) h^T` | CLASSICAL algebra; conditional information result PROVED | Forward baseline must include both p and h |
| Global replication invariance implies equal parameter gradients | PROVED, conditional; NUMERICAL check | Not established for ArithScope's complete architecture |
| A forward collision at one parameter value can have unequal gradients | PROVED example; NUMERICAL check | The graph already contains the extra variable |
| Independent known-period counts yield the exact posterior | CLASSICAL Bayes calculation; EXACT FINITE checks | Known noise, known period, correct independent law |
| Correlated copies invalidate naive independent counting | PROVED counterexample; EXACT FINITE checks | Deliberately different law, not a natural-data result |
| Frozen ArithScope head gradients match forward reconstruction | NUMERICAL / DEVELOPMENT | Two seed-1 models, short prefixes, no population inference |
| A learned gradient controller improves main-line output quality | OPEN | No controller was trained or compared |

## What a continuation should decide

The most valuable next question is whether a specific **richer** Jacobian
predicts intervention benefit beyond a matched forward state. Do not begin by
adding an unrestricted self-improvement loop. First decide what the monitor
can observe, its action menu, and its causal proxy loss.

Required controls for a separate prospective study:

1. Best fixed-window baseline, with its rule chosen on development data.
2. Forward monitor containing p, h, their exact head-gradient interaction,
   evidence-count features, and any candidate-prediction disagreements it uses.
3. The same monitor plus the proposed richer gradient, with matched capacity
   and training examples. Count all extra forwards and VJPs in its budget.
4. A count-aware readout control, kept distinct from gradient monitoring.
5. Independent evaluation sequences and model seeds, plus a dependence or
   change-point stress case. Do not fit thresholds on this branch's results,
   E9's test set, or previously inspected main extrapolation outputs.

Use full-coverage NLL and an accuracy guard if the task is improving
probabilities. If abstention is the task, specify coverage and rejection cost.
For the known synthetic law, use posterior-expected action value to distinguish
probability improvement from incidental correctness of a realized symbol.
Any future implementation needs its own protocol and checks; this directory
does not authorize an upgrade of the parent's claims.

## Findings that constrain the design

- More conditionally independent evidence cannot increase Bayes-optimal
  average proper loss. Window benefits in fixed learned models do not prove
  distant observations are statistically harmful.
- A finite menu of fixed bounded readouts can reduce a practical plateau but
  still has a positive asymptotic NLL floor. Count scale must affect the readout
  after normalization to escape that bound under the ideal law.
- Approximate attention vote shares are not necessarily exact category shares.
  The exact readout construction is a reference, not a validated neural remedy.
- The count-oracle advantage depends on a correct dependence model. Effective
  sample size and calibration are not automatically obtained from softmax.
- A pseudo-label gradient can be small on a confident error. A proxy loss must
  have a justified relationship to the target task.
- A noisy-token self-supervised loss must hold out the token being scored.
  The existing denoiser sees its own token, so direct reconstruction risks
  self-copying and is not an adequate adaptation design by itself.

## Provenance, execution, and a repaired failure

Parent state at setup: commit `5ce34b874584be86b39b97353f6fb76e347248cd`,
branch `prereg/e9-paired-window`. The initial main tracked-file hashes and
global checks are in `verification/`. E9 was active; no E9 checkpoint, result,
process signal, or protocol was used or modified by this study.

Two completed E6 checkpoints were selected by period and seed before branch
probe results. E8 checkpoint files were unavailable locally. Their copies and
run JSON files are hashed in `inputs/manifest.json`. Spec equality is checked
after JSON normalization of tuple/list representations. Historical checkpoint
bytes were not pinned by the original run JSON; the present seal is new.

The first `probe.py` invocation failed with `KeyError: 'vocab_size'`: original
RunSpec task dictionaries may omit defaults. The repair uses the parent's
`make_task` to resolve the registered defaults. The failed log and source are
preserved as `verification/gradient_run.log` and
`verification/probe_before_schema_fix.py`; successful execution uses
`verification/gradient_run_schema_fix.log`. No result file was written by
the failed attempt. This was a schema repair, not a change of data, selections,
decision rules, or protocol. `PROTOCOL.md` remains byte-identical to its
pre-computation version, which is hashed in both result files.

No commit, push, main integration, trained gate, or performance benchmark was
performed. The new directory is untracked in the shared checkout and may
affect concurrent `git_dirty` metadata. Never claim that dirty status came
only from E9 outputs without checking this directory too.

## Commands

Use the replay commands in `README.md`. For root-contract checks, run from the
repository root, with `PYTHONDONTWRITEBYTECODE=1`, one-thread environment
variables, low priority, and `PYTEST_ADDOPTS='-p no:cacheprovider'`:

```sh
nice -n 15 make oracle PY=/usr/bin/python3
nice -n 15 make test PY=/Users/akari_hayami_64/mamba/envs/arithscope/bin/python
make lint PY=/Users/akari_hayami_64/mamba/envs/arithscope/bin/python
```

The root test suite does not run this directory's tests. Run those explicitly
as documented in `README.md`. See `verification/` for delivered outcomes,
including the expected skipped opt-in compiled FlexAttention test.

