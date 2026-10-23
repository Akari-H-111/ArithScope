# Evidence readout: research program (2026-10-23)

Status: development study. No claim in `docs/CLAIM_LEDGER.md` changes from
anything in this directory until a preregistered experiment supports it.

## Why this question

Both earlier studies in `research/` (DeltaNet/Hilbert geometry and the
metacognition audit, 2026-10-15) end at the same open item, which is also
`docs/HANDOFF.md` §4.3: a softmax average reports vote *shares*, not vote
*counts* (MATH T8a), so a model ending in LayerNorm + linear readout has a
positive NLL floor (P3, C8), while the Bayes NLL of periodic denoising decays
exponentially (B1). The DeltaNet study showed that restoring the constant
character turns the T3 simplex comb into the exact indicator
`D_P = sum_{m|P} phi(m)/P K_m = 1_{P | n}`, and that an *additive* (unnormalized)
memory on those addresses gives exact counts and the Bayes posterior. Those
were untrained, oracle-period constructions.

This program asks whether that construction becomes a trainable ArithScope
component with an advantage that follows from the theory:

> An unnormalized evidence channel whose positional kernel is an ArithScope
> bank can turn context length into evidence: its log-odds can grow linearly
> with the number of comb votes at any length, following the Bayes rate.
> A kernel that is not exactly periodic (RoPE-like or learned frequencies)
> cannot beyond its drift horizon `1/|δ|`, because its comb-specific evidence
> is then residue-blind (T5; T9d and its finite-window remark).

## Tracks

| Track | Worker | Output | Status |
|---|---|---|---|
| T: theory (T9 draft, exact checks) | Antigravity CLI, Claude Sonnet 5.5 (high), then Claude Code CLI Sonnet after the Antigravity quota ran out | `theory/` in this directory | revised after audit |
| E: opt-in evidence head, tests, dev pilot machinery | Claude Code CLI, Sonnet | worktree `../ArithScope-evidence`, branch `research/evidence-readout` | done (uncommitted) |
| X: adversarial audit of T9 and theory-vs-pilot consistency | Claude Code CLI, Sonnet (fresh session) | `audit/` | done |
| P: development pilot (dev data seed 7, seed 0; 20 runs) | orchestrator | `pilot/` | done |

Results and the proposed next step: [`SYNTHESIS.md`](SYNTHESIS.md).

Orchestrated by Claude Opus 5.5 in the Claude desktop app. The repository
rules in `AGENTS.md` apply to every track: no edits to `results/runs/` of
existing suites, default semantics unchanged, evidence labels kept distinct,
no speed, linear-attention, natural-language, calibration, or novelty claims.

## Prior work to cite, not to claim

Cache language models (Kuhn and De Mori, 1990; Grave et al., 2017), trigger
features in maximum-entropy language models (Rosenfeld, 1996), linear
attention and fast weights (Katharopoulos et al., 2020; Schlag et al., 2021),
DeltaNet (Yang et al., 2024). An unnormalized count of earlier tokens is
classical; the specific object here is the exact arithmetic kernel on it and
the rate separation that the theory predicts.
