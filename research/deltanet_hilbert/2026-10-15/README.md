# DeltaNet, Hilbert geometry, and QKV memory

This isolated development study asks whether ArithScope's exact arithmetic
features can improve the addressing and evidence handling of a delta-rule
memory. It adds no main-suite arm and changes no parent claim.

The useful finding is specific: the T3-optimal comb kernel gives simplex keys,
whereas adding the constant character gives orthogonal phase addresses.
Softmax-equivalent score kernels can therefore have different recurrent-write
geometry. Orthogonal addressing removes one interference source; it does not
by itself supply confidence. A count-aware delta mean and simpler additive
count memory realize the same oracle under the ideal generating law.

Read [REPORT.md](REPORT.md) for generated findings and limits,
[MATHEMATICS.md](MATHEMATICS.md) for conditional arguments, and
[HANDOFF.md](HANDOFF.md) before continuing. The four-part professional framing
and rewritten request are in [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md).
[SOURCES.md](SOURCES.md) pins primary literature, including existing
uncertainty-aware delta memory. [PROTOCOL.md](PROTOCOL.md) records the local
prospective design and the addition of a necessary simpler control.

## Contents

- `geometry.py`: exact kernel identity, Gram ranks, memory recall and running
  mean/count identities, and order/change-point counterexamples. Stdlib only.
- `probe.py`: Hilbert-operator/autograd checks, recurrent versus triangular QKV,
  actual residue-feature maps, and five untrained analytic mechanisms plus an
  exact posterior oracle. No model training or checkpoint access.
- `test_geometry.py`, `test_probe.py`: explicit branch tests, outside the root
  test-discovery path. `render_report.py` generates all report numbers.
- `results/exact_v1.json`, `results/numerical_v1.json`: frozen outputs, including
  every fresh observation and per-sequence metric.
- `inputs/manifest.json`: initial source hashes and paper versions.
- `verification/`: root and branch checks, the initial protocol, replay logs,
  and the isolation audit for main files and the previous research branch.
- `MANIFEST.sha256`: delivered file hashes, excluding itself.

## Replay

From this directory, use the existing local interpreter and one-thread limits.
No installation is needed. Keep low priority while main experiments run.

```sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/akari_hayami_64/mamba/envs/arithscope/bin/python

shasum -a 256 -c MANIFEST.sha256
nice -n 15 /usr/bin/python3 -B geometry.py --verify results/exact_v1.json
nice -n 15 "$PY" -B probe.py --verify results/numerical_v1.json
/usr/bin/python3 -B render_report.py --verify REPORT.md
nice -n 15 "$PY" -B -m unittest discover -s . -p 'test_*.py' -v
"$PY" -B -m ruff check .
```

Replay compares exact JSON equality on the recorded environment. Different
environments may differ numerically or in metadata; preserve the frozen
output and investigate. Relevant parent source drift aborts replay. For a new
version, use a new protocol/output filename; `--output` refuses overwrites.
Do not run the parent's analyzers or training suites to replay this directory.

## Limits

The period and noise law are supplied, Q and K share a registered phase map,
and V is an observed noisy-symbol one-hot vector. These are mechanism controls,
not learned QKV features or a trained DeltaNet benchmark. The fixed logit scale
is declared rather than optimized. Low count-reference loss is an oracle
identity, not a demonstrated general calibration or main-model improvement.

Reported state budgets are logical mechanism budgets. The diagnostic code
also materializes shared oracle counts, feature tables, and raw records; it is
not a production memory benchmark. It obtains each mean gain's n_r from the
shared prefix count array, which is equivalent to a causal per-phase counter.
The two mean readouts share a state in the diagnostic implementation.

The tensor-product option costs P times the content-feature dimension. Known
phase addressing does not solve mixed/unknown periods or dependent data.
Adding the arithmetic features to a new state does not remove the parent's
bounded-final-readout issue unless evidence scale reaches logits appropriately.
The original ArithScope module is not converted to linear attention, and no
runtime, GPU, or language-model result is established.

This is a subdirectory in the shared checkout, not a separate worktree. It can
affect concurrent `git_dirty` metadata and shares hardware. Parent and prior
research content are protected by hash comparison; timings under concurrent
load are not benchmarks. Another chat should review and select the scope before
staging or integrating it.

