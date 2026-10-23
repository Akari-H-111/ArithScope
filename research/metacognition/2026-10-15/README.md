# Metacognition: an isolated information audit

This directory studies one question: **does a label-free backward pass add
information that a forward evidence state lacks, and can that information
support better output control?** It is a development branch in a subdirectory,
not a Git branch, new main-suite arm, or amendment to the public claim ledger.

The strongest result is a boundary: the uniform-target last-layer gradient is
exactly recoverable from forward tensors. Under a stronger replication
invariance assumption, even all parameter gradients cannot recover missing
evidence count. A different computational graph can expose additional
sensitivities. Independent evidence counts give an exact constructive oracle
in the registered toy law, and fail under a deliberately correlated law.

Start with [REPORT.md](REPORT.md) for generated results and limits. Read
[MATHEMATICS.md](MATHEMATICS.md) for the arguments, [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md)
for the four-part professional framing and rewritten request, and
[HANDOFF.md](HANDOFF.md) before continuing. Primary sources are in
[SOURCES.md](SOURCES.md); the prospective development design is
[PROTOCOL.md](PROTOCOL.md).

## Files and provenance

- `study.py`: rational posterior enumeration, a same-share information
  collision, and a correlated-copy counterexample. Standard library only.
- `probe.py`: analytic/autograd/central-difference checks, two functional
  counterexamples, and two frozen E6 checkpoint probes. No optimizer or training.
- `test_study.py`, `test_probe.py`: separate branch tests; the parent's default
  `make test` does not discover them.
- `inputs/manifest.json`: copied weights, original run records, selection rule,
  parent commit, and library source hashes. Main E9 inputs are not used.
- `results/exact_v1.json`, `results/gradient_v1.json`: immutable outputs, including
  raw fresh observations and per-query measurements.
- `render_report.py`: creates the report from those outputs. It does not train
  or re-score the parent experiments.
- `verification/`: before/after checks, replay logs, the first failed parser
  attempt and its source, and the main-file isolation audit.
- `MANIFEST.sha256`: all delivered file hashes, excluding the manifest itself.

## Replay

Run from this directory. The exact check uses the system interpreter. Use the
registered local environment for the neural check; no dependency installation
is needed here. Keep the one-thread limits and low priority while E9 is active.

```sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/akari_hayami_64/mamba/envs/arithscope/bin/python

shasum -a 256 -c MANIFEST.sha256
nice -n 15 /usr/bin/python3 -B study.py --verify results/exact_v1.json
nice -n 15 "$PY" -B probe.py --verify results/gradient_v1.json
/usr/bin/python3 -B render_report.py --verify REPORT.md
nice -n 15 "$PY" -B -m unittest discover -s . -p 'test_*.py' -v
"$PY" -B -m ruff check .
```

The numerical replay checks exact JSON equality on the recorded environment.
A different environment may fail that comparison because of numerical or
environment-field differences; investigate and save a new diagnostic rather
than replacing the frozen file. Parent library drift intentionally aborts the
checkpoint probe. Weights were cast from their saved float32 values to float64;
this is not a replay of the historical main-suite metrics.

To generate new outputs, use a new filename with `--output`; overwriting is
refused. A changed design also needs a new protocol version and local deviation
entry. Do not call the parent's `make analyze`, `make experiments`, or `make e9`
to replay this branch.

## Integration limits

No learned gate, abstention policy, test-time adaptation, or count-aware neural
readout is implemented. The exact count readout assumes the true period,
independent noise, and exact shares; it does not justify scaling an existing
network's logits by sequence length. No general calibration, natural-data,
language-model, speedup, or main-line improvement claim follows.

This directory shares the checkout and hardware with the main experiment.
Its files appear as untracked Git content, and can contribute to `git_dirty`
in concurrent run metadata. Parent content is checked by hashes rather than
by claiming a separate worktree. Timing under concurrent load is not a
performance benchmark. Another chat should review this directory explicitly
before staging or integrating anything.

