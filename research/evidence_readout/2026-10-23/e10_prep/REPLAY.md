# Library replay before E10 (2026-10-23)

The E10 library (worktree `../ArithScope-evidence`, uncommitted on `354ea6c`; source hashes in
`source_sha256_before_spread_pilot.txt`) was checked against stored E9 results made on the same Apple M2.

| Check | Script | Result |
|---|---|---|
| Re-evaluate stored E9 checkpoints, P = 12 and 13 seed 27, windows 64 and 8,192 at N = 8,192 | `replay_e9.py` | 4 of 4 records bitwise identical (digest, all, extrapolated, by_position; checkpoint SHA-256 equal) |
| Retrain `e9_p13__arith__s27` from scratch | `replay_e9_train.py` | bitwise identical: loss curve, validation curve, test metrics, positional snapshot; no `evidence` key written |
