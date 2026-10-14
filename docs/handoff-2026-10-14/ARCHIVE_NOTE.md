# Archived planning package (2026-10-14)

These files are the English planning documents that seeded this repository.
They are kept **unmodified** for provenance; their SHA-256 hashes match the original
package manifest. They describe the state *before* implementation. Their
statements such as "not available today: a working PyTorch attention layer" are
historical and have been superseded by:

| Planning document | Superseded by |
|---|---|
| `docs/MATH_SPEC.md` | [`../MATH.md`](../MATH.md) |
| `docs/EXPERIMENT_PROTOCOL.md` | [`../PREREGISTRATION.md`](../PREREGISTRATION.md) |
| `docs/CLAIM_LEDGER.md` | [`../CLAIM_LEDGER.md`](../CLAIM_LEDGER.md) |
| `docs/IMPLEMENTATION_PLAN.md`, `CODEX_START_HERE.md` | [`../EXECUTION_LEDGER.md`](../EXECUTION_LEDGER.md), [`../HANDOFF.md`](../HANDOFF.md) |
| `AGENTS.md` | [`../../AGENTS.md`](../../AGENTS.md) |

**Not archived here:**
- the source-repository manifest and provenance table, which point to the author's other repositories;
- the personal positioning notes;
- a non-English reading guide;
- the package hash manifest, which covers the full package.

The stdlib reference oracle (`reference/`) and its 13 tests (`tests/test_arithmetic_reference.py`) were copied byte-for-byte into the repository root and still run in CI.
