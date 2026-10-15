# Preregistration: confirmatory experiments (frozen before any confirmatory run)

This protocol was written after the development pilots (Section 6) and **before**
any confirmatory run. It is committed to git first. The commit history is the
timestamp: the results commit must come after this file. Any later change is
logged in Section 8 and never edited silently.

## 1. Confirmatory claims

The handoff named its hypotheses H1–H4 (H3 routing, H4 memory; both future work).
To avoid a clash, the confirmatory claims here are **C1–C4**. Each is derived from
a statement in [`MATH.md`](MATH.md) and has a falsification condition.

| | Claim | Derived from | Falsified if |
|---|---|---|---|
| **C1** | *Periodic extrapolation.* Trained at length 64 on periodic denoising, `rope+arith` keeps near-Bayes accuracy at 16× length, while `rope` degrades. | T5, T6 | The paired difference (`rope+arith − rope`) in extrapolated-region accuracy at N = 1024 has a 95% CI that includes 0, for either P = 12 or P = 13. |
| **C2** | *Comb sharpness.* At a **prime** period (P = 13), `rope+arith` beats the matched fundamental-cosine bank `rope+fourier_fund` (same periods, same parameter count). At P = 12 we predict **no** meaningful difference. | C3.1 (margins 1.083 vs 0.115 at P = 13; 1.091 vs 0.987 at P = 12) | The P = 13 paired difference at N = 1024 has a CI including 0. |
| **C3** | *Identifiability.* In the alias fixture (target period 12, ArithScope as the only positional signal), banks with `lcm = 6` stay at or below the alias-oracle accuracy at long lengths, whatever their size. Banks with `12 | lcm` approach Bayes accuracy, even with only two orders. | P1, T4 | At N = 1024, an `lcm = 6` bank's mean accuracy exceeds its alias oracle by more than 2 pp, or a `12 | lcm` bank's mean stays more than 5 pp below Bayes. |
| **C4** | *No harm on an aperiodic task.* On tagged denoising, adding ArithScope to RoPE does not reduce in-distribution accuracy. | γ = 0 equivalence (exact); the rest is empirical | The lower 95% CI bound of (`rope+arith − rope`) at N = 64 is below −1 pp. Inconclusive CIs are reported as inconclusive. |

**Secondary predictions.** These are stated in advance, but no decision depends on them:

- **S1 (drift).** The T5 corollary says learned real frequencies drift at offsets of about `1/|eps|`.
  - Prediction: `rope+fourier_learned` has lower mean accuracy at N = 8192 than at N = 1024 for at least one of the two periods.
  - Contrast: `rope+arith` is within 1 pp of the Bayes oracle at both lengths.
- **S2 (learned mixtures).** Layer-0 ArithScope heads put most of their mixture mass on the divisors of P. For prime P = 13 the mass concentrates on order 13 itself, the unique optimum of T3. *Caveat: this was already observed in the development pilot, so the confirmatory runs test replication across seeds, not a blind prediction.*

All other comparisons (other baselines, other lengths, position-shift robustness,
mixed periods) are **exploratory** and labeled as such in [`RESULTS.md`](RESULTS.md).

## 2. Fixed setup (all confirmatory runs)

- **Model:** pre-LN decoder-only Transformer, `d_model = 64`, 2 layers, 4 heads (`d_head = 16`), FFN 256 (GELU), no dropout, no absolute position embedding, untied output head, N(0, 0.02) init. About 102K backbone parameters. Positional parameters per variant are logged in every run file.
- **Variants:** `nope`, `alibi`, `rope`, `arith` (ArithScope only), `rope+arith`, `rope+fourier_fund`, `rope+fourier_full`, `rope+fourier_learned`, `rope+relbias`.
  - Bank: `M = {1, …, 16}` unless stated.
  - `relbias` is clipped at the training length.
  - `fourier_learned` has 16 free frequencies per head.
  - All additive channels are zero-initialized, so every `rope+X` model equals `rope` at initialization (tested bitwise).
- **Optimization:** AdamW (β = 0.9, 0.98), weight decay 0.01 on weight matrices only, 100 warmup steps, cosine decay to 10%, gradient clip 1.0, batch 32, 2000 steps, training length 64. Fresh training batches every step from a seed-specific generator. The (base LR, positional-LR multiplier) pair comes from the pilot rule in Section 6.
- **Seeds:** 1–5 for primary suites, 1–3 for secondary suites. Seed 0 is reserved for pilots. For a given seed, every variant starts from **bitwise-identical backbone weights** and sees **identical training batches**. Only the positional mechanism differs. This was verified before the runs.
- **Data:** test sets use `data_seed = 20261014` and are identical across variants and seeds. Their SHA-256 digests are logged per run. Pilots used `data_seed = 7`.
- **Evaluation:** greedy accuracy and NLL at each length (65,536 predictions per length), overall and on the **extrapolated region** (`t >= 64`). Every number is reported next to the exact Bayes oracle on the same examples, and the alias oracle where defined.
- **Hardware:** CPU only, float32, one thread per run, four runs in parallel. Wall-times are reported but not compared across suites.

## 3. Suites

| Suite | Task | Variants | Seeds | Eval lengths | Role |
|---|---|---|---|---|---|
| `e1_periodic` | periodic denoising, P = 12 and P = 13 (separate models) | all 9 | 1–5 | 64 … 8192 (16,384 predictions per length above 2048) | C1, C2 (primary); S1, S2. Also position-shift evals at N = 256 with shifts 10^3 … 10^9 (exploratory). |
| `e2_alias` | periodic **retrieval with query tokens** (25% of positions are uninformative `QUERY` tokens; only queries are scored), P = 12, `arith` only | aliased (lcm 6): `{2,3}`, `{1,2,3,6}`; resolved (lcm 12): `{3,4}`, `{4,6}`, `{12}`, `{1,2,3,4,6,12}` | 1–5 | 64 … 2048 | C3 (primary) |
| `e3_tagged` | tagged denoising, 12 tags | `nope`, `alibi`, `rope`, `arith`, `rope+arith`, `rope+fourier_fund`, `rope+relbias` | 1–5 | 64 … 2048 | C4 (primary) |
| ~~`e4_mqar`~~ | MQAR | dropped before confirmatory runs (floor effect in pilots; see Section 6) | – | – | – |
| `e5_mixed` | periodic denoising, P ∈ {7, 11, 13} per sequence (unlabeled) | all 9 | 1–3 | 64 … 2048 | exploratory |

## 4. Statistics

- **Unit of analysis:** the training seed. For each seed, compute the paired difference of a metric between two variants on the same test set.
- **Reported per comparison:** mean difference, 95% CI from the t distribution with n − 1 degrees of freedom, the per-seed values, and a sign count (e.g. 5/5).
- **Tests:** no multiplicity correction. The four confirmatory claims are stated in advance, and every other comparison is labeled exploratory.
- **Failures:** a run with a non-finite loss counts as a failure. It is reported as such and never replaced or rerun with another seed.

### Design history of E2 (all on development data, before any E2 confirmatory run)

1. **Dense denoising, P = 24** (`pilot_controls`). With the lcm-12 bank `{2,3,4,6,12}` the model reached 72.4% at N = 1024, well above the pure 12-comb vote (≈ 53%). The reason: the model always sees its own noisy token `x_t`, which breaks the tie between the two aliased candidates `mu[t mod 24]` and `mu[(t+12) mod 24]` (an analytic estimate gives ≈ 75%). The pure-vote alias oracle is therefore not the information ceiling in dense denoising.
2. **Query tokens, P = 24, training length 64** (`pilot_alias_query`). Scored positions show an uninformative `QUERY` token. The aliased bank then sat exactly at the alias oracle (49.0% vs 49.2% at N = 1024).
   - The resolving CRT bank `{3,8}` learned the correct comb: α ≈ {3: 0.4, 8: 0.6}, γ ≈ 7–10.
   - But it did **not** length-generalize (51.6% vs Bayes 95.0%). A query sees at most three comb positions during training, one of them its own `QUERY` token, so the content readout faces a distribution shift at long lengths.
3. **Query tokens, P = 12, training length 64** (`pilot_alias_design`). Six comb positions are visible in training.
   - The aliased bank `{2,3}` (lcm 6) sits at its alias oracle (50.3% vs 50.1% at N = 256).
   - The resolving bank `{3,4}` tracks Bayes (96.1% vs 98.0% at N = 1024).
   - P = 24 trained at length 128 also works (98.2% vs 98.6% at N = 2048) but costs twice as much.

   **This design is adopted.** It keeps E2 at E1's exact setup.

**Alias oracle.** It is the plurality vote over the `gcd(P, lcm(M))`-comb of **observed** positions. It is the exact information ceiling for any positional aggregator restricted to the bank's alias classes, except for coarse absolute-position information leaking through the causal mask. C3's 2 pp tolerance and its long evaluation length (N = 1024) account for that leak.

## 5. What is *not* tested

- Natural-language modeling and pretrained LLMs.
- GPU kernels and speed claims; FoPE, which has no verified implementation here.
- Data-dependent order routing.
- The handoff's H3 (evidence-aware routing) and H4 (memory).

## 6. Pilot rule and outcome (development data only)

- **Rule:** for each variant, choose between (lr = 1e-3, ×1) and (lr = 3e-3, ×10) by **final validation NLL at the training length**. Use development data (`data_seed = 7`, seed 0), pooled over the P = 12 and P = 13 pilots. Extrapolation-length results from pilots are *not* used for any decision.
- **Outcome:** filled in below before the confirmatory runs start. See [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md) for the raw pilot table.

**Selected hyperparameters** (`results/summary/hparam_selection.json`; validation NLL is the mean over the pilot tasks of each family):

| family | variants | choice | note |
|---|---|---|---|
| periodic | `nope`, `rope`, `arith`, `rope+arith`, `rope+fourier_fund`, `rope+fourier_full`, `rope+fourier_learned`, `rope+relbias` | lr 3e-3, positional ×10 | every listed variant had lower validation NLL with this pair |
| periodic | `alibi` | lr 1e-3, ×1 | 1.8166 vs 1.8182 (ALiBi has no learned positional parameters) |
| tagged | all piloted variants | lr 3e-3, ×10 | `rope+fourier_fund` and `rope+relbias` inherit their periodic choice (the same pair) |

**Design decisions taken from the pilots** (development data only):

- **E2** uses query tokens and P = 12. See the design history in Section 3.
- **MQAR (E4) is dropped.** No variant learned it within budget.
  - Original configuration (512 keys, 64 values, 2000 steps): 6–27% accuracy at N = 64 vs a 100% oracle.
  - Easier configuration (64 keys, 16 values, 4000 steps; `rope` and `rope+arith`): 19% vs 100%.
  - A floor effect cannot test non-inferiority, so C4 rests on E3 alone. The pilot files are committed.
- **Evaluation in E1 is extended to N = 8192** (16,384 predictions per length above 2048) to test S1. This affects cost only; no tuning was done on any extrapolation result.

## 7. Analysis artifacts

`experiments/analyze.py` regenerates every table and figure from the run JSON files in `results/runs/`. No number in `README.md` or `RESULTS.md` is typed by hand.

## 8. Deviations log

Entries made after the preregistration commit (`cdea384`, 2026-10-14 15:28:53 UTC; the confirmatory runner started at 15:29 UTC). **No entry changes a confirmatory claim, decision rule, suite, seed, hyperparameter, or test set.**

1. **Test tolerance** (`88edbef`, 15:32 UTC, before any confirmatory run had finished). One unit test compared biases computed over two different offset ranges bitwise. It failed by one ulp on the pip-installed CPU build used in CI. The test now uses a 1e−12 tolerance, and [`SPEC.md`](SPEC.md) documents the rounding semantics. No library code changed.
2. **Post-hoc shift test at the training length.** The preregistered shift evaluation (E1, N = 256) is confounded for RoPE-based models: they already fail at 4× their training length, so their shifted accuracy says nothing about shift robustness. `experiments/posthoc_shift.py` re-evaluates the saved E1 checkpoints at N = 64, where every RoPE-based model works, with the same shifts. It is labeled post hoc wherever it appears, and no verdict depends on it.
3. **Mechanism analyses added after seeing data** (exploratory):
   - where the matched cosine bank puts its layer-0 mixture weight (`analyze_cosine_mixtures`), a mechanism check for C2;
   - the attention profile of one query at 16× the training length (`experiments/attention_figure.py`);
   - the comb selectivity of trained learned-frequency heads as a function of offset (`analyze_drift`), a mechanism check for S1, added after the first E1 results showed the predicted accuracy drop.

   Two presentation items use no new data: the mechanism figure (`experiments/theory_figures.py`) and the interactive explainer (`docs/lab/`, built from the same summaries).
4. **Generated README summaries.** The TL;DR, the cost paragraph, and the P = 12 / P = 13 overview figure in `README.md` are generated by `experiments/analyze.py` from the same summaries. The rule of Section 7 extends to them.
5. **Scope of the Section 7 rule.** It covers confirmatory and post-hoc results. Development-pilot numbers quoted in prose (Section 3 of this file; "What did not work" in `README.md`) are copied from the pilot tables in [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md), which `experiments/summarize.py` generated.
6. **Verdict code.** The per-comparison tables, paired differences, and the C3 and C4 checks were in `experiments/analyze.py` at `cdea384`. The function that prints the verdicts for C1–C4, S1, and S2 (`findings`) was written afterwards, while the confirmatory runs were in progress. It encodes the falsification conditions of Section 1 as written, without changing a threshold, length, or metric.
7. **Code edits while the runs were in progress.** Nothing in `arithscope/`, `experiments/suites.py`, or `experiments/run_suite.py` changed while confirmatory runs were executing; `git diff cdea384 -- arithscope experiments/suites.py experiments/run_suite.py` was empty when the last run finished. Edits in that period touched documentation, tests (unused imports), analysis and figure scripts, and tooling. Run files record the git HEAD at completion with `git_dirty = true` for this reason. Changes made after all results existed are listed in [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md).
8. **External cold audit (2026-10-15)** ([`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md)). After all results existed, an independent review found two library defects that no experiment exercised: position-based causality for repeated or unsorted positions (F1) and span-sized memory for sparse positions (F2). Both were fixed. Every experiment used strictly increasing positions, so no run, table, or verdict changes. `experiments/replay.py` confirms it with the fixed code: re-scoring every E1 checkpoint (180 of 180 evaluations) and retraining 11 confirmatory runs (11 of 11) reproduced the stored records bit for bit.
9. **Post-hoc NLL diagnosis** (prompted by the audit's F4). `experiments/posthoc_nll.py` evaluates E1 checkpoints on a fresh diagnostic test set (data seed 20261015, never used before) and compares NLL with the exact Bayes NLL, measures attention off the comb, and restricts attention to recent keys as an intervention. After a smoke test on one checkpoint at reduced length, and before the full run, the script was extended with accuracy under the restriction, outlier statistics, per-sequence means (the audit asked for both), and two exploratory checks: the NLL of positions whose own token is corrupted, and the correlation of NLL with the vote share on the comb. A separate script, `experiments/posthoc_readout.py`, computes from the weights alone the lowest NLL that each model's readout can produce, to tell a readout limit from a representation limit. NLL tables from the committed run files are added to `RESULTS.md`. NLL was recorded for every run but was never a decision metric, so these are labeled exploratory.
10. **Multiplicity sensitivity** (post hoc). Bonferroni and Holm adjustments of the interval-based confirmatory claims, reported next to the preregistered unadjusted verdicts, which remain the verdicts.

## 9. Second preregistration: E6, calibration at long lengths (2026-10-15)

This section was written after the post-hoc NLL diagnosis (Section 8, item 9) and **before**
any E6 run. It is committed before the runner starts; the results commit must come after it.
Sections 1–8 are unchanged. Later changes to this section go to Section 9.8 and are never made
silently.

### 9.1 Motivation

In E1, every model with RoPE whose accuracy stays at the oracle has an NLL that falls up to about
1,024 tokens and then rises, while ArithScope alone stays flat. The post-hoc diagnosis
([`RESULTS.md`](RESULTS.md), "Calibration") located the rise in the second layer: limiting that
layer to the last 1,024 keys at test time removed the rise without hurting accuracy. That was a
test-time intervention on trained models, chosen after seeing the data. E6 tests two remedies
named in [`HANDOFF.md`](HANDOFF.md), open question 3, as preregistered training-time designs on
fresh test data.

### 9.2 Arms

All arms use E1's task (periodic denoising, P = 12 and P = 13, separate models), architecture, bank
`M = {1, …, 16}`, optimizer, 2,000 steps, and E1's preregistered periodic hyperparameters
(lr 3e-3, positional ×10). **Nothing is tuned for E6.**

| Arm | Variant | Change from E1 | Role |
|---|---|---|---|
| `reference` | `arith` | none | no RoPE; flat NLL in E1 |
| `baseline` | `rope+arith` | none | the model whose NLL rises |
| `window` | `rope+arith` | the second layer attends only to its last 64 keys (`j > i − 64`, by token index), in training and at test | remedy 1: the broad layer only ever sees offsets it saw in training |
| `long_train` | `rope+arith` | training length 256, batch 8 | remedy 2: longer training sequences at the same 2,048 tokens per step |

- The window equals the training length, so it never binds in training. `window` and `baseline` therefore train bitwise identically, and the remedy acts only beyond the training length. This is checked for every seed (sanity check, Section 9.4). It is the training-time counterpart of the diagnosis's limit, with the window set by the training length instead of by inspecting test data.
- `long_train` keeps the number of training tokens fixed (8 × 256 = 32 × 64), not the compute: attention costs more per token at length 256. Its lr is inherited, not re-selected; the batch size change is a known confound of this arm and is reported as such.
- Within a period and seed, all arms share the initial weights and the training-batch generator (`long_train` draws batches of a different shape from the same generator).
- The vote-count readout remedy (HANDOFF, open question 3) needs a new architecture with no design yet and is **not tested** in E6.

### 9.3 Fixed setup

- **Suite:** `e6_calibration` in [`experiments/suites.py`](../experiments/suites.py); 2 periods × 4 arms × seeds 1–5 = 40 runs.
- **Test data:** a fresh data seed, `20261016` (E1–E5 used `20261014`, the post-hoc diagnosis `20261015`). Evaluation lengths are E1's (64 … 8,192), with 65,536 predictions per length up to 2,048 and 32,768 above (4 sequences at N = 8,192; E1 had 2). No shift evaluations.
- **Hardware:** Apple M2 (4 performance + 4 efficiency cores, 16 GB), macOS, conda-forge PyTorch 2.14.1 (osx-arm64 build), float32, one thread per run, four runs in parallel. All 40 runs run on this machine, so every comparison is within one platform. E6 numbers are not compared bit for bit with E1, which ran on x86-64 Linux (Section 9.7).
- **Library change:** an optional per-layer key window (`attn_windows`, explicit backend only). At its default (no window) every computation is unchanged, and the ids of all earlier runs are unchanged (both tested).
- **Feasibility pilot** (`pilot_e6`, development data seed 7, seed 0, P = 12, one run per arm): evaluated only at length 64, so no extrapolation result from it exists. Its only use is to confirm that each arm trains (finite loss) and to time the runs. Outcome in Section 9.6.

### 9.4 Claims and decision rules

All metrics come from the N = 8,192 test records, by position (buckets of 1,024 positions), so they
cover the same positions in every arm whatever its training length:

- **far NLL** and **far accuracy**: mean over positions 4,096–8,191;
- **growth**: NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047.

Each comparison is a paired difference per seed within a period, with a 95% t interval over the 5
seeds, the per-seed values, and a sign count, as in Section 4.

| | Claim | Supported if (at **both** P = 12 and P = 13) | Falsified otherwise |
|---|---|---|---|
| **L1** | *Window remedy.* `window` has lower far NLL than `baseline`, at no cost in accuracy. | the interval for far NLL (`window − baseline`) lies below 0, **and** the lower bound for far accuracy (`window − baseline`) is above −1 pp | far-NLL interval reaches 0 at either period, or the accuracy guard fails |
| **L2** | *Longer training.* `long_train` has lower far NLL than `baseline`, at no cost in accuracy. | the same two conditions for `long_train − baseline` | as for L1 |

**Secondary predictions** (stated in advance; no decision depends on them):

- **S3 (flat).** `window`'s mean growth is within ±0.02 nats of 0 at both periods.
- **S4 (not the whole gap).** `window`'s far NLL stays above `reference`'s (lower bound of `window − reference` above 0 at both periods). The diagnosis found part of the gap within the first 1,024 positions, which a window cannot remove.
- **S5 (delay, not removal).** `long_train`'s mean growth stays positive at both periods.

**Sanity check (not a claim).** `window` and `baseline` have identical loss curves, validation curves, and positional parameters for every seed. If they do not, the window code is wrong: the runs are stopped, the cause is logged in Section 9.8, and no verdict is reported until it is fixed.

**Statistics.** No multiplicity correction for the verdicts, as in Section 4. Holm-adjusted one-sided p-values for the four far-NLL comparisons (two claims × two periods) are reported next to them as a sensitivity analysis. A run with a non-finite loss is a failure: it is reported and never replaced. `experiments/e6_verdicts.py` encodes these rules and is committed with this section.

### 9.5 Scope of any conclusion

E6 can show whether these remedies lower the NLL of these 2-layer models on this synthetic task at up to 128× the training length. It says nothing about language models, other tasks, or calibration in general. [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md) still forbids general calibration claims; an E6 result would add only a scoped, synthetic entry.

### 9.6 Feasibility pilot outcome

Filled in before any E6 run, from `results/runs/pilot_e6/` and [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md), Stage 8.

- **Every arm trains.** Four runs, all `COMPLETED` with finite losses. At length 64, every arm's test accuracy is within 0.1 pp of the Bayes oracle on the development test set.
- **Sanity check passes.** `window` and `baseline` have identical loss curves, validation curves, positional parameters, and test metrics, as the design requires.
- **Cost.** With four runs in parallel, training took 242–253 s for the length-64 arms and 353 s for `long_train`. A single E1 run took 172 s alone on this machine, 63 s of it evaluation (including shift tests). E6 has 40 runs in ten rounds of four, with twice E1's evaluation above 2,048 tokens: about one to one and a half hours, with about 1.7 GB per worker. **The suite fits on this CPU; no GPU is needed.**
- **No design change follows from the pilot.**

### 9.7 Platform

A retrain of one E1 run (`e1_p12`, `rope+arith`, seed 1) on the E6 machine gave the same accuracy at every evaluation length as the stored record, with training losses within 5 × 10⁻⁸ and test NLL within 1.1 × 10⁻⁵ relative error, but not bitwise identical. E1's bitwise replays (Section 8, item 8) hold on x86-64 Linux only. This is why every E6 arm, including the reference and baseline, is trained afresh on one machine instead of reusing E1 runs.

### 9.8 Deviations log (E6)

None yet.
