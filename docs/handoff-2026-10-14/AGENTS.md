# ArithScope agent contract

- Treat `CODEX_START_HERE.md` as the scoped work order and `docs/CLAIM_LEDGER.md` as the public-truth boundary.
- Read and respect `SOURCES_MANIFEST.json`. External repositories are read-only **sources**, not worktrees to change or wholesale copy.
- Keep changes small; preserve standard PyTorch attention and preexisting baseline behavior.
- Always run the mathematical oracle (`python3 -m unittest discover -s tests -v`) and the newly relevant model tests after changes. Record commands, environment, outcomes, hashes, and skipped steps.
- Separate: exact theorem; ordinary literature theorem; finite checker; numerical diagnostic; statistical finding; hypothesis; negative result.
- Use deterministic seeds and held-out splits. Avoid training on test-set data or optimizing baselines less than the candidate.
- No unverified speedup, SOTA, memory-saving, formal proof, calibration, or e-value claim.
- Do not mix source model `q` as harmonic/channel/order, with query vector `Q` in attention. Use `order`/`harmonic_index` for arithmetic indices.
- Respect integer-position and torus-coordinate semantics. `exp(2π i q t)` with integer `q,t` is trivial, so use `t/L` or explicit dimensionless torus maps for phases.
- No auto-creation of public repos, no publishing or licensing decisions without explicit user authorization.
- If tests fail, fix/minimize before adding features; leave exact failures visible, and don't delete negative outcomes.
