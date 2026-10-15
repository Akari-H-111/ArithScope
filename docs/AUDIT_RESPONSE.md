# Response to the first cold audit (2026-10-15)

An independent adversarial review of `main` at `0ce3dce` found three correctness problems and one experimental anomaly. The audit is archived verbatim in [`audits/2026-10-15-first-cold-audit.md`](audits/2026-10-15-first-cold-audit.md). Every finding was re-verified here before anything was changed. Each code fix comes with regression tests that fail on the audited code, and the corrected theorem statement comes with a numeric check of the counterexample.

| | Finding | Verified here | Action | Effect on published results |
|---|---|---|---|---|
| F1 | Backends disagree on causality when positions repeat or are unsorted | **Yes.** The explicit and FlexAttention backends let a token see later tokens | Causality is now by token index in every backend | None: every experiment used strictly increasing positions, where both rules coincide |
| F2 | Sparse positions make bias evaluation allocate memory proportional to the position span | **Yes.** The old code tried to allocate 32 PB for positions near 10<sup>15</sup> | Evaluation is bounded by the number of query/key pairs; positions are range-checked | None: contiguous positions take the same code path as before |
| F3 | The T5 corollary for learned frequencies is wrong | **Yes.** The counterexample holds analytically and numerically | `MATH.md` corrected; numeric regression test added | None: no experiment or verdict used the corollary |
| F4 | At long lengths, RoPE + ArithScope has much higher NLL than ArithScope alone, at the same accuracy | **Yes, and broader**: it holds for every RoPE-containing variant that keeps oracle accuracy | Diagnosed post hoc; NLL tables added to `RESULTS.md` | New post-hoc result; preregistered verdicts unchanged |
| F5 | Scope and comparison constraints | Agreed | Wording in `README.md` and `RESULTS.md`; multiplicity sensitivity analysis | Verdicts unchanged under Bonferroni and Holm adjustment |

The library changes did not alter any stored result that was checked. `experiments/replay.py` re-scores all 90 E1 checkpoints at N = 1,024 and 8,192 and retrains 11 confirmatory runs from scratch with the fixed code. It compares them bit for bit with the committed run files: 180 of 180 evaluations and 11 of 11 runs are identical ([`RESULTS.md`](RESULTS.md), replay table). A stricter range check came after the replays; it affects only positions outside the supported range (F2).

## F1: causal masking

**Reproduction.** With positions `[0, 0, 1, 2]` or `[2, 1, 3, 4]`, the explicit and FlexAttention backends compared positions (`pos_j <= pos_i`). Token 0 then attended to token 1, and changing token 1 changed the output at token 0. The fused SDPA fast path and the folded backend used token order, so the backends disagreed by up to 0.55 in output units.

**Decision.** Causality is by token index (`j <= i`) in all four backends. Positions now enter only the relative-position biases and RoPE, so they may repeat, restart in packed sequences, be unsorted, or carry offsets without leaking future tokens. This follows the audit's option (B). Rejecting such layouts, option (A), would have ruled out packed sequences. Blocked explicit evaluation passes the global query offset of each block. The fused fast paths now also apply to per-sequence positions, because the mask no longer depends on them.

**Tests** (`tests/test_attention.py`):

- All backends agree for repeated, unsorted, packed, and sparse positions, alone and with left padding and per-sequence positions.
- No token's output depends on later tokens for any of these layouts.
- Blocked explicit evaluation equals unblocked evaluation for these layouts, so each block uses its global query index. A test that drops the offset fails.
- For increasing positions, the new mask equals the old one.

The first two fail on the audited code, run in a worktree of `0ce3dce`.

**KV-cache offsets.** The attention layer has no KV-cache interface: queries and keys are always the same tokens. What a cache would add is covered. Positions with an offset are tested per sequence, for example starting at 1,000 or 10<sup>6</sup>. The bias module's `Nq ≠ Nk` case has its own test. A cache interface would pass the same query offset that blocked evaluation already uses (`allowed_mask(..., q_offset)`).

**Specification.** [`SPEC.md`](SPEC.md) §2 and §4 state the rule and record the change.

## F2: memory proportional to the position span

**Reproduction.** For positions `[0, 10^15, 2·10^15 + 1]` the audited code builds the offset range `[lo, hi]` and fails with `DefaultCPUAllocator: can't allocate memory: you tried to allocate 32000000000000024 bytes`. Positions near `2^62` overflowed instead of raising a clear error.

**Fix.**

- *Dense biases* (`OffsetBias.forward`): when the offset range is no larger than the number of query/key pairs, which always holds for contiguous positions, the range is evaluated as before. Otherwise only the offsets that occur are evaluated (`torch.unique`). Memory is bounded by the number of pairs in both cases.
- *FlexAttention*: contiguous positions keep the offset table, which stays small. Sparse positions use a pointwise `score_fn` for every bias type: residue lookups for ArithScope and the cosine banks, closed forms for ALiBi, learned frequencies, and the clipped relative bias. Compiled on CPU, the pointwise path was about ten times slower than the table path at N = 4,096, so the table remains the default for contiguous positions.
- *Range check*: every position must satisfy `|position| < 2^62`, so that int64 offsets cannot overflow; other values raise `ValueError`, in the dense path and in FlexAttention. A first version took `abs()` of the positions and missed `-2^63`, whose absolute value overflows in int64; this was found in review after the replays, and the check now compares the minimum and maximum with the bounds.
- The folded backend never had this problem, as the audit noted: its features depend on residues only.

**Tests.** Every bias type gives the same values for positions `[0, 10^15, 2·10^15 + 1]` through the dense path and through direct evaluation of the occurring offsets. Positions at `2^62`, `-2^62`, and `-2^63` raise `ValueError` in the dense path, and `-2^63` in FlexAttention; positions at `±(2^62 - 1)` are accepted. Both tests fail on the audited code: the first with the 32 PB allocation error above, the second with an overflow error instead of a `ValueError`. The residue tables have `len(orders) × max(orders)` entries for the kernels and `rank × max(orders)` for the folded features: 16 × 16 and 80 × 16 for the default bank `{1, …, 16}`. Their size is fixed by the configuration, not by the data.

## F3: the T5 corollary for learned frequencies

**Verification.** For `θ = (√2, √2 + π)`, neither frequency is a rational multiple of `2π`, but `θ₂ − θ₁ = π` violates T5's joint hypothesis. The even-offset mass tends to `I₀(2)/(I₀(2)+1) = 0.69508…`; the sum over 200,000 offsets gives 0.695087, matching the audit.

**Correction** ([`MATH.md`](MATH.md), §4). T5 itself and its application to standard RoPE frequencies, where Lindemann's theorem gives the joint hypothesis, are unchanged. The sentence that extended T5 to any learned frequencies that are individually irrational is replaced by:

- the joint hypothesis;
- the counterexample;
- the facts that violations form a Lebesgue-null set and that every rational frequency vector satisfies the hypothesis;
- a remark on near-resonances, which govern behavior at finite length and generalize the existing drift remark.

`tests/test_theory_notes.py` checks the counterexample numerically.

## F4: accuracy and NLL diverge at long lengths

**Verification.** At P = 13 and N = 8,192, the committed run files give a mean NLL on unseen positions of about 0.07 for ArithScope only and 0.22 for RoPE + ArithScope, with the same accuracy. The paired difference that `analyze.py` now generates, +0.152 nats (95% CI [+0.081, +0.223], all five seeds higher), reproduces the audit's independent computation. The pattern is broader than the audit reports. Every RoPE-containing variant whose accuracy stays at the oracle shows NLL falling up to 512–1,024 tokens and rising after: RoPE + ArithScope and RoPE + full Fourier bank at both periods, and the matched cosine bank at P = 12. Learned frequencies show a similar rise but also lose accuracy (S1). ArithScope without RoPE does not rise: from N = 1,024 to 8,192 its NLL goes from 0.072 to 0.064 at P = 12 and from 0.074 to 0.067 at P = 13.

**Diagnosis** (post hoc, `experiments/posthoc_nll.py`; numbers in [`RESULTS.md`](RESULTS.md)). DIAGNOSIS_PENDING

**What changes in the reporting.** NLL tables with paired differences are now generated for E1 from the immutable run files. NLL was recorded for every run but was not a preregistered decision metric, and it is labeled that way. The README states the calibration gap next to the accuracy results.

## F5: scope and comparisons

- **Full Fourier bank.** It matches ArithScope in E1 because its span contains every Ramanujan kernel; `RESULTS.md` already said so. The README now says it too: the contribution is the parameterization and its exact algebraic properties, not exclusive expressivity.
- **Stronger baselines and other probes.** The two strongest Fourier baselines here are both in E1: the full Fourier bank (fixed frequencies, more mixture parameters) and learned frequencies. The aperiodic control (E3) tests that ArithScope does no harm; it does not test retrieval. The content-retrieval probe (MQAR) was dropped after weak pilots, so, as the audit says, there is no general content-retrieval result yet.
- **Multiplicity.** The preregistration used unadjusted 95% intervals for four claims. A post-hoc sensitivity analysis applies Bonferroni (98.75% intervals) and Holm adjustments to the four interval-based comparisons: C1 at both periods, C2, and C4. All of them still hold. C2 is the closest: its adjusted interval stays above zero, but not by much.
- **E5 and MQAR.** E5 is exploratory with three seeds, and learned frequencies beat RoPE + ArithScope at short lengths there; MQAR was dropped. Both were already reported this way, and the README keeps them under "What did not work".
- **Out of scope**, as before: natural language, pretrained models, GPU training with the FlexAttention backward pass, and speed.
