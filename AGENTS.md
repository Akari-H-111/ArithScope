# Agent contract (ArithScope)

Read [`docs/HANDOFF.md`](docs/HANDOFF.md) first. It holds the current state, open
questions, and exact commands.

## Rules

- **Claims.** [`docs/CLAIM_LEDGER.md`](docs/CLAIM_LEDGER.md) is the public-truth boundary. Upgrade a claim only when new evidence exists in this repository. Keep these labels distinct:
  - proved
  - classical
  - asymptotic
  - numerical
  - empirical
  - open
- **Tests.** Before and after any change, run:
  - `make oracle` (stdlib only)
  - `make test` (needs torch)
- **Results are immutable.** Never edit or delete files in `results/runs/`. A changed protocol needs a new suite name and an entry in the deviations log of [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).
- **No hand-typed numbers.** Every confirmatory or post-hoc number in `README.md` and `docs/RESULTS.md` must come from `experiments/analyze.py`; edit the narrative, not the auto-generated blocks. Development-pilot numbers quoted in prose are copied from the pilot tables in `docs/EXECUTION_LEDGER.md`, and theory numbers come from `results/summary/theory.json` or exact computation.
- **Lint.** `make lint` (ruff) must pass. `tests/test_arithmetic_reference.py` and `reference/` are byte-identical copies of the planning package's oracle and are never reformatted.
- **Fair baselines.**
  - Baselines get the same budget, data, and seeds as the candidate.
  - Hyperparameters are chosen by the preregistered validation rule, never on test or extrapolation data.
- **Keep the default semantics.** `gamma = 0` must reproduce the base model bitwise; causality is by token index in every backend and positions never enter the mask ([`docs/SPEC.md`](docs/SPEC.md), §4); `tests/test_attention.py` guards both.
- **Notation.** Arithmetic indices are *orders* (`m`); the attention query is `q`/`Q`. Never mix them.
- **Phases.** Integer positions with integer frequencies give trivial phases. Phases must come from exact residues `(a * (t mod m)) mod m`.
- **Forbidden without new evidence:**
  - speedup claims
  - sub-quadratic or linear-attention claims
  - state-of-the-art claims
  - natural-language claims
  - calibration or e-value claims
- **External actions.** The repository owner chose the MIT License and public release (2026-10-14). Do not change the license, publish elsewhere, or create repositories without explicit authorization from the repository owner.
