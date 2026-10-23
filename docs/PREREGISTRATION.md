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

No entry changes a claim, decision rule, arm, seed, hyperparameter, or test set.

1. **Reporting code after the results** (after `409c42b`, which holds the 40 run files and the verdicts). `experiments/analyze.py` gained `analyze_e6`, which formats the frozen verdicts and paired intervals from `results/summary/e6.json`, adds per-arm intervals and an NLL-by-position figure computed from the same run files with `e6_verdicts.py`'s metric functions, and fills the `e6_findings` block in [`RESULTS.md`](RESULTS.md). `experiments/e6_verdicts.py` is unchanged since `039a56d`.

## 10. Third preregistration: E7, the NLL floor of ArithScope alone (2026-10-15)

This section was written after the E6 results (`409c42b`, `664f681`) and **before** any E7 run. It
is committed and pushed before the runner starts; the results commit must come after it. Sections
1–9 are unchanged. Later changes to this section go to Section 10.7 and are never made silently.

### 10.1 Motivation

E6's secondary prediction S4 failed: RoPE + ArithScope with its second layer limited to the last 64
keys ended *below* ArithScope alone in far NLL (interval excluding 0 at P = 12, not at P = 13; lower
in 5/5 seeds at both periods). So ArithScope alone's flat NLL floor is not the best these models
reach. **Hypothesis:** the floor partly comes from ArithScope's own broad second layer, which also
attends far away and averages vote shares instead of counting votes ([`MATH.md`](MATH.md), P3 and
the remark after it).

Prior evidence, stated so that it is not mistaken for a test: in the post-hoc diagnosis
(`results/posthoc/nll_diagnostics.json`; E1 checkpoints, data seed 20261015), limiting ArithScope
alone's second layer to its last 1,024 keys lowered its NLL on positions 4,096–8,191 in all 10
models, by 0.0001 to 0.008 nats (floors of 0.026–0.128). That was a test-time intervention chosen after seeing data,
with a window 16× wider than E7's. It motivates the direction of the prediction; it is not E7
evidence.

### 10.2 Arms

All arms use E1's task (periodic denoising, P = 12 and P = 13, separate models), architecture, bank
`M = {1, …, 16}`, optimizer, 2,000 steps at length 64 with batch 32, and E1's preregistered periodic
hyperparameters (lr 3e-3, positional ×10). The window is E6's: the second layer attends only to its
last 64 keys (`j > i − 64`, by token index), in training and at test. **Nothing is tuned for E7.**

| Arm | Variant | Second-layer window | Role |
|---|---|---|---|
| `reference` | `arith` | none | ArithScope alone, E6's `reference` |
| `arith_window` | `arith` | last 64 keys | **new:** ArithScope alone without its broad second layer |
| `window` | `rope+arith` | last 64 keys | E6's `window` arm, the model that went below `reference` |

- The window equals the training length, so it never binds in training: `arith_window` and `reference` train bitwise identically, and the window acts only beyond the training length. This is checked for every seed (sanity check, Section 10.4). E7 therefore asks what the trained ArithScope-only model does when its second layer is kept within the offsets it was trained on.
- E6's `baseline` (unwindowed RoPE + ArithScope) and `long_train` arms are not repeated: neither bears on the hypothesis, and L1 and L2 are already decided.
- Runs use a new experiment name (`e7_p12`, `e7_p13`), hence new training batches. Initial weights depend on the seed only, so each seed starts from the same weights as in E1 and E6; E7 is a fresh draw of training data and test data, not of initializations. Within a period and seed, all arms share the initial weights and the training-batch generator.
- The first layer stays broad in every arm. If ArithScope alone's floor also comes from its first (comb) layer, which averages shares too, a second-layer window cannot remove that part. E7 does not window the first layer.

### 10.3 Fixed setup

- **Suite:** `e7_floor` in [`experiments/suites.py`](../experiments/suites.py); 2 periods × 3 arms × seeds 1–5 = 30 runs.
- **Test data:** a fresh data seed, `20261017` (E1–E5 used `20261014`, the post-hoc diagnosis `20261015`, E6 `20261016`). Evaluation lengths are E1's (64 … 8,192), with 65,536 predictions per length up to 2,048 and 32,768 above (4 sequences at N = 8,192), as in E6. No shift evaluations.
- **Hardware:** the E6 machine (Apple M2, macOS, conda-forge PyTorch 2.14.1 osx-arm64, float32), one thread per run, four runs in parallel ([`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md), Stage 8). All 30 runs run on it. E7 numbers are compared with each other, not bit for bit with E6 (different training batches and test data).
- **Library:** unchanged since E6. No pilot: every arm's training is a configuration E6 already ran (the `arith_window` arm trains exactly as `reference`), and the window code is covered by `KeyWindowTests` and E6's sanity check.

### 10.4 Claim and decision rules

Metrics are E6's (Section 9.4), from the N = 8,192 test records by position, computed by the same
functions: **far NLL** and **far accuracy** are the means over positions 4,096–8,191; **growth** is
NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. Each comparison is a paired
difference per seed within a period, with a 95% t interval over the 5 seeds, the per-seed values,
and a sign count.

| | Claim | Supported if (at **both** P = 12 and P = 13) | Falsified otherwise |
|---|---|---|---|
| **W1** | *ArithScope's floor is partly its broad second layer.* `arith_window` has lower far NLL than `reference`, at no cost in accuracy. | the interval for far NLL (`arith_window − reference`) lies below 0, **and** the lower bound for far accuracy (`arith_window − reference`) is above −1 pp | far-NLL interval reaches 0 at either period, or the accuracy guard fails |

A falsified W1 is reported as such: it would mean that, in these models, removing distant keys from
ArithScope's second layer does not lower its floor, and the hypothesis of Section 10.1 fails.

**Secondary predictions** (stated in advance; no decision depends on them):

- **S6 (part, not all).** `arith_window`'s far NLL stays above `window`'s (lower bound of `arith_window − window` above 0 at both periods). Reason: the first layer stays broad and also averages shares, and the post-hoc 1,024-key limit removed only a small part of the floor.
- **S7 (E6's surprise replicates).** `window`'s far NLL is below `reference`'s on fresh training batches and test data (upper bound of `window − reference` below 0 at both periods). In E6 this held at P = 12 and not at P = 13, so this is uncertain.

**Descriptive (no prediction).** The share of the `reference` − `window` far-NLL gap that the window
closes on ArithScope alone, from the period means; per-arm growth.

**Sanity check (not a claim).** `arith_window` and `reference` have identical loss curves, validation curves, and positional parameters for every seed. If they do not, the runs are stopped, the cause is logged in Section 10.7, and no verdict is reported until it is fixed.

**Statistics.** No multiplicity correction for the verdict, as in Sections 4 and 9. Holm-adjusted one-sided p-values for W1's two far-NLL comparisons are reported next to it as a sensitivity analysis. A run with a non-finite loss is a failure: it is reported and never replaced. `experiments/e7_verdicts.py` encodes these rules, reusing `experiments/e6_verdicts.py`'s metric and interval functions unchanged, and is committed with this section.

### 10.5 Scope of any conclusion

E7 can show whether limiting the second layer of these 2-layer ArithScope-only models to their
training length lowers their NLL at up to 128× that length, on this synthetic task. A supported W1
locates part of the floor in the broad second layer of these models; it does not prove the mechanism
of the P3 remark, which stays a proof sketch, and it says nothing about language models or
calibration in general.

### 10.6 Commands

```bash
make e7 PY=~/mamba/envs/arithscope/bin/python WORKERS=4   # 30 runs, then experiments/e7_verdicts.py
```

### 10.7 Deviations log (E7)

No entry changes a claim, decision rule, arm, seed, hyperparameter, or test set.

1. **Reporting code after the results** (after `cf30e8f`, which holds the 30 run files and the verdicts). `experiments/analyze.py` gained `analyze_e7`, which formats the frozen verdicts and paired intervals from `results/summary/e7.json`, adds per-arm intervals and an NLL-by-position figure computed from the same run files with `e6_verdicts.py`'s metric functions, and fills the `e7_findings` block in [`RESULTS.md`](RESULTS.md). `experiments/e7_verdicts.py` is unchanged since `c2d793f`.

## 11. Fourth preregistration: E8, a powered retest of E7's W1 (2026-10-15)

This section was written after the E7 results (`cf30e8f`, `e8737db`) and **before** any E8 run. It
is committed and pushed before the runner starts; the results commit must come after it. Sections
1–10 are unchanged, and E7's verdict stands as recorded: W1 is falsified as written. E8 is a new
claim, **W2**, on new seeds and new data. It does not overturn W1, and it is never pooled with E7.
Later changes to this section go to Section 11.8 and are never made silently.

### 11.1 Motivation

In E7, limiting ArithScope alone's second layer to its last 64 keys lowered its far NLL in 5/5 seeds
at both periods (`results/summary/e7.json`). The preregistered rule still failed at P = 12 (95%
interval −0.1205 to +0.0049), because the unlimited model's floor varies widely between seeds (far NLL
0.027–0.154 at P = 12, against 0.010–0.019 with the window). Five seeds were too few for that spread.
E8 asks the same question with a seed count planned on the observed spread.

Two further reasons for new seeds: E1, E6, and E7 all used seeds 1–5, and initial weights depend on the
seed only, so their models share initializations and are not independent draws of them. E8 uses seeds
that no earlier suite used.

### 11.2 Arms

All arms use E7's setup (Section 10.2): E1's task (periodic denoising, P = 12 and P = 13, separate
models), architecture, bank `M = {1, …, 16}`, optimizer, 2,000 steps at length 64 with batch 32, and
E1's preregistered periodic hyperparameters (lr 3e-3, positional ×10). **Nothing is tuned for E8.**

| Arm | Variant | Key windows (layer 1, layer 2) | Role |
|---|---|---|---|
| `reference` | `arith` | none, none | ArithScope alone, E7's `reference` |
| `arith_window` | `arith` | none, last 64 keys | E7's `arith_window` |

- As in E7, the window equals the training length and never binds in training, so the two arms train bitwise identically and differ only beyond length 64 (sanity check, Section 11.5).
- E7's `window` arm (RoPE + ArithScope) is not repeated: W2 does not involve it, and every run goes to the claim.
- **Considered and not run: windowing both layers, `(64, 64)`.** It would not isolate the averaging of the first layer. With both layers limited to 64 keys, a prediction can draw on at most the last 127 tokens. In E7, the `arith_window` models' NLL was 0.447–0.455 (P = 12) and 0.511–0.514 (P = 13) on positions 56–63 of the length-64 test, and 0.110–0.116 and 0.144–0.148 on positions 112–127 of the length-128 test, against far NLL of 0.010–0.019. Such an arm would mostly measure lost evidence, not the first layer's share-averaging. A test of the first layer needs a different design, such as a first-layer window much wider than the training length. It is left open.

### 11.3 Seed count

`experiments/e8_power.py` computes the seed count from the committed E6 and E7 summaries, by a rule written into the script before it was first run:

- **Spread.** E7's `arith_window − reference` far-NLL differences (the W1 quantity), together with E6's `window − reference` differences. In E7 the two kinds of difference have nearly the same spread (standard deviations 0.0505 and 0.0545 at P = 12, 0.0300 and 0.0309 at P = 13), because the `reference` floor dominates both. Per period the two variances are pooled (8 degrees of freedom), and the pooled standard deviation is raised to its one-sided 80% upper confidence limit, so that a spread that happened to be small in five seeds does not shrink the plan.
- **Effect.** Per period, the smaller of the two mean differences in absolute value (0.0411 at P = 12 and 0.0472 at P = 13, both from E6).
- **Power.** Under a normal model of the paired differences, the probability that the upper bound of the 95% t interval lies below 0 is a noncentral t probability. The two periods are independent models, so the joint probability is the product. The seed count is the smallest n whose joint power is at least 0.90.

Outcome (`results/summary/e8_power.json`): standardized effects d = 0.815 (P = 12) and 0.836 (P = 13). This gives **n = 21 seeds** (joint power 0.901), so E8 uses **seeds 6–26**. The same plan gives E7's five seeds a joint power of 0.087. Sensitivity (no decision depends on it): E7's own point estimates would give n = 11, and halving the planned effects would need n = 78.

The accuracy guard is not in the plan: far accuracy was 100% in every arm and seed of E6 and E7.

### 11.4 Fixed setup

- **Suite:** `e8_power_retest` in [`experiments/suites.py`](../experiments/suites.py); 2 periods × 2 arms × seeds 6–26 = 84 runs. Its experiment names are new (`e8_p12`, `e8_p13`), so training batches are new as well. Within a period and seed, both arms share the initial weights and the training-batch generator.
- **Test data:** a fresh data seed, `20261018` (E1–E5 used `20261014`, the post-hoc diagnosis `20261015`, E6 `20261016`, E7 `20261017`). Evaluation lengths are E1's (64 … 8,192), with 65,536 predictions per length up to 2,048 and 32,768 above (4 sequences at N = 8,192), as in E6 and E7. No shift evaluations.
- **Hardware:** the E6 and E7 machine (Apple M2, macOS, conda-forge PyTorch 2.14.1 osx-arm64, float32), with one thread per run and four runs in parallel. All 84 runs run on it.
- **Library:** unchanged since E6. There is no pilot, because both arms are configurations that E7 ran.

### 11.5 Claim and decision rules

Metrics are E6's (Section 9.4), computed by the same functions from the N = 8,192 test records by position:

- **far NLL** and **far accuracy** are the means over positions 4,096–8,191;
- **growth** is NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047.

Each comparison is a paired difference per seed within a period. It is reported with a 95% t interval over the seeds, the per-seed values, and a sign count.

| | Claim | Supported if (at **both** P = 12 and P = 13) | Falsified otherwise |
|---|---|---|---|
| **W2** | *E7's W1, retested with 21 fresh seeds.* `arith_window` has lower far NLL than `reference`, at no cost in accuracy. | the interval for far NLL (`arith_window − reference`) lies below 0, **and** the lower bound for far accuracy (`arith_window − reference`) is above −1 pp | far-NLL interval reaches 0 at either period, or the accuracy guard fails |

A supported W2 would locate part of ArithScope alone's far-NLL floor, in these models, in its broad second layer. A falsified W2 is reported as such, and the claim stays unsupported.

**Secondary predictions** (stated in advance; no decision depends on them):

- **S8 (the spread is in the unlimited floor).** The seed-to-seed standard deviation of far NLL is at least 3 times larger for `reference` than for `arith_window`, at both periods. In E7, the ratio was about 16 at both periods.
- **S9 (flat).** `arith_window`'s mean growth is within ±0.02 nats of 0 at both periods, as for E6's window arm (S3).

**Sanity check (not a claim).** `arith_window` and `reference` have identical loss curves, validation curves, and positional parameters for every seed. If they do not, the runs are stopped, the cause is logged in Section 11.8, and no verdict is reported until it is fixed.

**Statistics.**

- No multiplicity correction for the verdict, as in Sections 4, 9, and 10. Holm-adjusted one-sided p-values for W2's two far-NLL comparisons are reported next to it as a sensitivity analysis.
- A run with a non-finite loss is a failure. It is reported and never replaced, and its seed is left out of that period's paired analysis (both arms).
- E8 is analyzed alone. It is not pooled with E7, whose seeds share initial weights with E1 and E6.
- `experiments/e8_verdicts.py` encodes these rules. It reuses `experiments/e6_verdicts.py`'s metric and interval functions unchanged, checks that the suite's seed count equals `e8_power.json`'s, and is committed with this section.

### 11.6 Scope of any conclusion

As for E7 (Section 10.5): E8 concerns these 2-layer ArithScope-only models on this synthetic task, at up to 128× the training length. A supported W2 does not prove the mechanism of the P3 remark in [`MATH.md`](MATH.md), and it says nothing about language models or calibration in general.

### 11.7 Commands

```bash
python experiments/e8_power.py                            # seed count -> results/summary/e8_power.json
make e8 PY=~/mamba/envs/arithscope/bin/python WORKERS=4   # 84 runs, then experiments/e8_verdicts.py
```

### 11.8 Deviations log (E8)

No entry changes a claim, decision rule, arm, seed, hyperparameter, or test set.

1. **Reporting code after the results** (after `4c74cfa`, which holds the 84 run files and the verdicts). `experiments/analyze.py` gained `analyze_e8`, which formats the frozen verdicts and paired intervals from `results/summary/e8.json`, adds per-arm intervals and an NLL-by-position figure computed from the same run files with `e6_verdicts.py`'s metric functions, and fills the `e8_findings` block in [`RESULTS.md`](RESULTS.md). `experiments/e8_verdicts.py` and `experiments/e8_power.py` are unchanged since `20b8ffa`.

## 12. Fifth preregistration: E9, a paired window dose-response (2026-10-15)

This section was written after the E8 results (`4c74cfa`) and **before** any E9 run. It is committed
and pushed before the runner starts; the results commit must come after it. Sections 1–11 are
unchanged. Later changes to this section go to Section 12.8 and are never made silently.

### 12.1 Motivation

E8 supported W2: limiting ArithScope alone's second layer to its last 64 keys lowered its far NLL in
21 of 21 seeds at both periods (`results/summary/e8.json`). E6–E8 compared only two points, a window
of 64 keys and no window. E9 asks whether the far NLL changes with the window size as a dose-response,
over windows of 64, 256, 1,024, and 8,192 keys.

E9 also removes a cost of E8. A window equal to the training length never binds in training, so E6–E8
trained the windowed and unwindowed arms bitwise identically, twice. E9 trains each seed once and
evaluates the same weights with each window.

The design adapts a draft from an external review (the repository owner's ChatGPT workspace,
2026-10-15). Changes from the draft: the repository's two-sided 95% intervals replace its one-sided
ones, and the seed count comes from the rule in Section 12.3.

### 12.2 Design

- **Training.** ArithScope alone (`arith`) is trained without a window, with E7's setup: E1's task (periodic denoising, P = 12 and P = 13, separate models), architecture, bank `M = {1, …, 16}`, optimizer, 2,000 steps at length 64 with batch 32, and lr 3e-3 with positional ×10. The training run is evaluated at length 64 only. **Nothing is tuned for E9.**
- **Window evaluations.** `experiments/e9_window_eval.py` loads each checkpoint and evaluates the unchanged weights at N = 8,192 with the last (second) layer limited to its last W keys, for W ∈ {64, 256, 1,024, 8,192}. The first layer is never limited. W = 8,192 at N = 8,192 is the ordinary unwindowed evaluation (tested bit for bit).
- **Test set.** The test set is the one `train.run` would draw at this length: 4 sequences, data seed `20261019`. It is fresh: E1–E5 used `20261014`, the post-hoc diagnosis `20261015`, E6 `20261016`, E7 `20261017`, and E8 `20261018`. Every window and seed of a period sees the same test set.
- **Records.** Each window evaluation is its own record in `results/runs/e9_paired_window_eval/`, holding the checkpoint's SHA-256.

### 12.3 Seed count

**When the rule was fixed.** `experiments/e9_power.py` was committed and pushed (`543e402`; GitHub records the branch creation at 2026-10-15 08:59:41 UTC) while E8 had 36 of its 84 runs done. `results/summary/e8.json` was written at 10:07:43 UTC.

**The rule.**
- **Input.** E8's per-seed differences D = far NLL(window 64) − far NLL(no window).
- **Spread.** The standard deviation of D, raised to its one-sided 80% upper confidence limit.
- **Power.** The smallest n, at least 5, whose joint power for "95% slope interval above 0 at both periods" is at least 0.90.
- **Model.** The plan assumes the per-seed slope is proportional to −D. That assumption does not model noise from the intermediate windows.

**Outcome** (`results/summary/e9_power.json`): d = 1.353 (P = 12) and 0.911 (P = 13), so **n = 15 seeds** per period (joint power 0.904). E9 uses **seeds 27–41**; E8 used 6–26. Sensitivity (no decision depends on it): n = 12 with the point estimate of the spread, and 53 with half the effect.

### 12.4 Fixed setup

- **Suite:** `e9_paired_window` in [`experiments/suites.py`](../experiments/suites.py): 2 periods × 15 seeds = 30 training runs, then 30 × 4 = 120 window evaluations. Experiment names are new (`e9_p12`, `e9_p13`).
- **Hardware:** the E6–E8 machine (Apple M2, macOS, conda-forge PyTorch 2.14.1 osx-arm64, float32), one thread per process, four in parallel. A window evaluation took 55–75 s on random weights while E8 ran; the whole suite is about 1–1.5 hours.
- **Library:** unchanged since E6.

### 12.5 Claim and decision rules

**Metrics.** Far NLL and far accuracy are E6's (Section 9.4): the means over positions 4,096–8,191 of the N = 8,192 test. They are computed by `experiments/e6_verdicts.py`, unchanged.

**Per-seed quantities**, all within a period:
- **slope:** the least-squares slope of far NLL on log2(W) over the four windows;
- **accuracy difference:** far accuracy at W = 64 minus far accuracy at W = 8,192;
- **steps:** far NLL differences between adjacent windows.

Each has a 95% t interval over seeds.

| | Claim | Supported if (at **both** P = 12 and P = 13) | Falsified otherwise |
|---|---|---|---|
| **R1** | *Window dose-response.* ArithScope alone's far NLL rises with the log of its last layer's window, at no cost in accuracy. | the 95% interval of the mean slope lies above 0, **and** the 95% lower bound of the accuracy difference is above −1 pp | the slope interval reaches 0 at either period, or the accuracy guard fails |

R1 claims a positive trend, not monotonicity. For fixed weights, adding keys to a softmax average can move the NLL either way, and monotonicity of the limit is OPEN ([`MATH.md`](MATH.md), remark after C8).

**Secondary predictions** (stated in advance; no decision depends on them):

- **S10 (near-monotone).** For each period and each of the three steps (64→256, 256→1,024, 1,024→8,192), the one-sided lower confidence bound at level 1 − 0.05/6 for the step is above −0.005 nats.
- **S11 (E8 replicates).** Far NLL at W = 64 minus W = 8,192 has a 95% interval below 0 at both periods. This is E8's W2 quantity on fresh seeds and data, not a new claim.

**Sanity checks.** If any of these fails, no verdict is reported until the cause is found and logged in Section 12.8:
- each seed's four records come from one checkpoint (same SHA-256);
- each period's records share one test set (same digest);
- all window records were made at one code commit;
- every training run completed.

**Statistics.**
- No multiplicity correction for the verdict; the two periods are an intersection, as in Sections 9–11. Holm-adjusted one-sided p-values for the two slope tests are reported as a sensitivity analysis.
- A failed training run or window evaluation is reported and never replaced, and its seed is left out of that period's analysis.
- E9 is analyzed alone and not pooled with E7 or E8.
- `experiments/e9_verdicts.py` encodes these rules and checks that the suite's seed count equals `e9_power.json`'s.

### 12.6 Scope of any conclusion

As for E7 and E8 (Sections 10.5, 11.6):
- **Models and task.** E9 concerns these 2-layer ArithScope-only models on this synthetic task, at 128× the training length.
- **Population.** The four test sequences are shared across seeds, so the result is about seeds on this test set, not about a population of sequences.
- **Mechanism.** A supported R1 does not prove a mechanism.
- **Outside scope.** It says nothing about language models or calibration in general.

### 12.7 Commands

```bash
python experiments/e9_power.py                            # seed count -> results/summary/e9_power.json
make e9 PY=~/mamba/envs/arithscope/bin/python WORKERS=4   # 30 runs, 120 window evaluations, then e9_verdicts.py
```

### 12.8 Deviations log (E9)

No entry changes a claim, decision rule, window, seed, hyperparameter, or test set.

1. **Reporting code after the results** (after `d6b365f`, which holds the 30 training runs, the 120 window records, and the verdicts). `experiments/analyze.py` gained `analyze_e9`, `plot_e9`, and `e9_findings`. They format the frozen verdicts and intervals from `results/summary/e9.json`, draw `results/figures/e9_dose_*.png` from the same window records with E6's metric functions, and fill the `e9_findings` block in [`RESULTS.md`](RESULTS.md). `experiments/e9_verdicts.py` is unchanged since `5ce34b8`.

## 13. Sixth preregistration: E10, an evidence-count readout (2026-10-23)

This section is written before any E10 run. It is committed and pushed before the runner starts; the results commit
must come after it. Sections 1–12 are unchanged. Later changes to this section go to Section 13.10 and are never made
silently. The authoritative design file, from which the code was written, is
`research/evidence_readout/2026-10-23/E10_SPEC.md`.

### 13.1 Motivation

A softmax average reports vote shares, not vote counts ([`MATH.md`](MATH.md), T8a), so a model ending in LayerNorm and a
linear readout has a positive NLL floor (P3, C8), while the Bayes NLL of periodic denoising decays exponentially in the
number of comb votes (B1). E6–E9 lowered that floor by limiting the broad layer; they did not remove it. `HANDOFF.md`
(open question 3) names the remaining remedy: give the readout a vote-count signal.

E10 tests that remedy. The *evidence head* adds, after the final LayerNorm,

```math
\ell_i \mathrel{+}= W_{\text{out}}\,E_i,\qquad E_i=\sum_{j\le i}\kappa(i-j)\,\mathrm{onehot}(x_j),
```

with causality by token index and `W_out` initialized to 0, so a model with the head equals the model without it bitwise
at initialization (tested). The kernel κ is an unconstrained combination of the bank's Ramanujan kernels
(`kernel = "arith"`) or, as the matched control, of fundamental cosines of the same 16 orders (`"cosine_fund"`).

The theory is a development draft, `research/evidence_readout/2026-10-23/theory/T9_DRAFT.md`, audited in
`research/evidence_readout/2026-10-23/audit/` (no theorem found wrong). Its predictions for this design:

- **T9a.** With κ = D_P = Σ_{m|P} φ(m)/P K_m = 1[P | n] and W_out = (log 17) I, the head's output is exactly the Bayes posterior at every position.
- **T9b.** D_P is in the span of the Ramanujan bank {1, …, 16} for P = 12 and 13, and not in the span of fundamental cosines of any orders unless φ(P) ≤ 2.
- **T9c.6.** For a learned fundamental-cosine kernel, a random motif fails to get a positive margin at some residue rarely at P = 12 and almost always at P = 13 (Monte Carlo on the pilot's folds).
- **T9e.** If each observation is copied, the count overstates the evidence and the head is no longer Bayes.

In the development pilot (Section 13.8) the `arith` head learned β proportional to D_P's coefficients and tracked the
exact Bayes NLL from the training length to 128× beyond it, at one seed. E10 tests this on fresh seeds and data, with a
matched control, a drop-in arm on RoPE, and a preregistered boundary where the head is predicted to do harm.

### 13.2 Arms

All arms use E8/E9's setup: periodic denoising (V = 16, ε = 1/2), P = 12 and P = 13 as separate models, bank
`M = {1, …, 16}`, 2 layers, 2,000 steps at length 64 with batch 32, E1's preregistered periodic hyperparameters per
variant. **Nothing is tuned for E10.**

| Arm | Variant | Evidence head | Role |
|---|---|---|---|
| `arith` | `arith` | none | ArithScope alone (E8's `reference`, E9's W = 8,192) |
| `arith_ev` | `arith` | `arith` kernel | the candidate |
| `arith_evfund` | `arith` | `cosine_fund` kernel (same 16 coefficients) | matched control for the basis |
| `rope` | `rope` | none | RoPE alone |
| `rope_ev` | `rope` | `arith` kernel | drop-in on a model that fails to extrapolate |

- Within a period and seed, `arith`, `arith_ev`, and `arith_evfund` share initial base weights and the training-batch generator; so do `rope` and `rope_ev`. Gradient clipping is global, so the arms separate after the first step.
- The head's kernel coefficients are positional parameters (positional learning-rate multiplier); `W_out` is in the ordinary groups.
- **Not run:** `rope+arith` with the head (it behaved like `arith_ev` in the pilot; no claim needs it), the `cosine_full` and `cosine_learned` kernels, and the plain count (development pilot only; Section 13.8).

### 13.3 The copied-observation law (boundary)

`CopiedPeriodicDenoising` (`periodic_copied`) draws exactly the random numbers of `PeriodicDenoising`, in the same order,
then cuts each residue class into blocks of k = 4 consecutive visits and copies the block's first observation to the whole
block. The per-position marginal law is unchanged; only independence is broken. With k = 1 it equals `PeriodicDenoising`
bitwise (tested). The exact Bayes posterior counts block leaders only.

`experiments/e10_copy_eval.py` evaluates every E10 checkpoint, unchanged, on this law at N = 256 and 1,024. Models are
trained on the ordinary law only: the boundary asks what the head does when its independence assumption is wrong at test time.

### 13.4 Seed count

`experiments/e10_power.py` encodes the rule of `E10_SPEC.md`, fixed before the spread pilot ran:

- **Spread** is the sample SD raised to its one-sided 80% upper confidence limit; **power** of one part with standardized effect d and n seeds is the noncentral t probability that the 95% t interval clears its threshold; parts of a claim (periods, conditions) multiply.
- **V1** is planned on confirmatory data from the same setup: E8's `reference` (seeds 6–26) and E9's W = 8,192 (seeds 27–41) far NLL of ArithScope alone, 36 values per period, with the head's far NLL taken as 0 (checked: at most 0.001 in every pilot seed, else stop).
- **V2, V3, V4, B1** are planned on the development spread pilot `pilot_e10_spread` (seeds 1000–1003, data seed 7), with the thresholds of Section 13.6 subtracted from the effects.
- n is the largest of the per-claim smallest n ≥ 5 with joint power ≥ 0.90; above 20 seeds per period the script stops and the design needs a decision.

**Outcome** (`results/summary/e10_power.json`; no stop condition fired, and the head's far NLL was at most 0.001 in every
pilot seed): V1 binds, with d = 1.84 (P = 12) and 1.39 (P = 13) from E8/E9's floor of ArithScope alone (mean far NLL
0.064 and 0.065, SD 0.031 and 0.042 over 36 seeds), and needs **n = 8 seeds** (joint power 0.914). V2, V3, V4, and B1
each need only the minimum of 5. E10 uses **seeds 42–49**. Sensitivity (no decision depends on it): n = 7 with the point
SD or with a 0.80 target, and 25 if every effect were halved.

### 13.5 Fixed setup

- **Suite:** `e10_evidence` in [`experiments/suites.py`](../experiments/suites.py): 2 periods × 5 arms × 8 seeds = 80 training runs, then 80 × 2 copied-law evaluations; experiment names `e10_p12`, `e10_p13`. About 1.5–2 hours with four workers (the spread pilot's runs took about 4 minutes each).
- **Test data:** data seed `20261020` (fresh: E1–E5 `20261014`, diagnosis `20261015`, E6–E9 `20261016`–`20261019`). Lengths 64, 256, 1,024, 8,192; 65,536 predictions per length (8 sequences at N = 8,192, 64 at 1,024, 256 at 256). The copied-law test sets use the same data seed with the copied task's description, so they are different sequences.
- **Hardware:** the E6–E9 machine (Apple M2, macOS, conda-forge PyTorch 2.14.1 osx-arm64, float32), one thread per run, four in parallel.
- **Library changes since E9** (all opt-in): the evidence head (`arithscope/evidence.py`; `ModelConfig.evidence`, `RunSpec.evidence`), the copied-observation task, and the exact Bayes NLL helper. With the head absent every computation, record field, and earlier run id is unchanged (tested), and E9 replays bit for bit (Section 13.8).

### 13.6 Claims and decision rules

**Metrics.** Far NLL and far accuracy are E6's (positions 4,096–8,191 of the N = 8,192 test, `experiments/e6_verdicts.py`
unchanged). *Excess NLL at 1,024* is the NLL over all positions of the N = 1,024 test minus the exact Bayes NLL on the
same positions. *Copy NLL at 256* is the NLL over all positions of the copied-law test at N = 256. *Independent NLL at
256* is the same on the ordinary N = 256 test. Every comparison is a paired difference per seed within a period, with a
95% t interval over seeds, the per-seed values, and a sign count.

| | Claim | Supported if | Falsified otherwise |
|---|---|---|---|
| **V1** | The head lowers ArithScope alone's far NLL, at no cost in accuracy. | at both periods: far NLL (`arith_ev − arith`) interval below 0, **and** far accuracy (`arith_ev − arith`) lower bound above −1 pp | either fails at either period |
| **V2** | The head follows the Bayes NLL at 16× the training length. | at both periods: upper bound of `arith_ev`'s excess NLL at 1,024 below 0.005 nats | upper bound ≥ 0.005 at either period |
| **V3** | At the prime period the exact basis matters (C2's structure, now for counting). | at P = 13: far NLL (`arith_evfund − arith_ev`) interval above 0, **and** far accuracy (`arith_ev − arith_evfund`) lower bound above +10 pp | either fails |
| **V4** | Drop-in: the head restores RoPE's extrapolation. | at both periods: far accuracy (`rope_ev − rope`) lower bound above +20 pp, **and** far NLL (`rope_ev − rope`) interval below 0 | either fails at either period |
| **B1** | Boundary, a preregistered prediction of harm: with copied observations the head does worse than no head. | at both periods: copy NLL at 256 (`arith_ev − arith`) interval above 0 | interval reaches 0 at either period |

A supported B1 is reported as a limitation of the method, with the same weight as V1–V4.

**Secondary predictions** (stated in advance; no decision depends on them):

- **S12 (no difference at P = 12).** The far-accuracy interval of `arith_ev − arith_evfund` lies within ±1 pp at P = 12.
- **S13 (SGD finds the construction).** The cosine between the learned β and D_P's coefficients is at least 0.99 in every seed of `arith_ev` and `rope_ev`, at both periods.
- **S14 (the Bayes scale).** The effective scale (Σ_{m|P} β_m) × (mean diagonal − mean off-diagonal of `W_out`) is within ±10% of log 17 in every seed of `arith_ev` and `rope_ev`.
- **S15 (crossover).** On the ordinary law the head lowers the NLL at 256: interval of `arith_ev − arith` below 0 at both periods, so B1's harm comes from the dependence, not from the head at short lengths.

**Sanity checks.** No verdict is reported until a failing one is resolved and logged in Section 13.10: every run
completed; all records come from one code commit; within a period and length all records share one test digest; each
copy record's checkpoint SHA-256 matches a run of the suite; the suite's seed count equals `e10_power.json`'s.

**Statistics.**
- No multiplicity correction for the verdicts; the periods are an intersection, as in Sections 9–12. Holm-adjusted one-sided p-values over the interval tests of V1–V4 and B1 are reported as a sensitivity analysis.
- A failed run is reported and never replaced; its seed is left out of every comparison of that period that uses the failed arm.
- E10 is analyzed alone and not pooled with the pilots or with E8/E9.
- `experiments/e10_verdicts.py` encodes these rules and is committed with this section.

### 13.7 Scope of any conclusion

- **Task.** One synthetic law that is exactly the head's Bayes model when observations are independent. V1–V4 say nothing about data where they are not; B1 tests one deliberate departure (copying), not dependence in general.
- **Models.** Two-layer models, 128× the training length; the head reads raw tokens with one fixed kernel per model.
- **Population.** The test sequences are shared across seeds, so results concern seeds on these test sets.
- **Not claimed:** speed or memory (the head is computed densely here), linear or sub-quadratic attention, calibration in general, natural language, or novelty of counting earlier tokens (cache language models are classical).

### 13.8 Design history (development data only)

- **`pilot_evidence`** (data seed 7, seed 0, 20 runs; [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md), Stage 16): chose the arms and the thresholds of V2 (0.005 nats), V3 (+10 pp), and V4 (+20 pp), each well inside the pilot's single-seed effect. It also ran `cosine_full`, `cosine_learned`, the plain count, and `rope+arith` with the head, which E10 does not repeat.
- **E10_SPEC.md** fixed the copied law (k = 4), the boundary metric (copy NLL at 256), and the seed-count rule before any copied-law data or spread-pilot run existed.
- **`pilot_e10_spread`** (data seed 7, seeds 1000–1003; 40 runs and 80 copied-law evaluations, 0 failures; [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md), Stage 16). It ran after `E10_SPEC.md` and `experiments/e10_power.py` were fixed; the source hashes recorded before it still match. Its only use is the seed count. Applying the frozen decision code to it as a dry run, every claim and secondary prediction comes out as predicted on development data; that is a check of the code and of feasibility, not evidence.
- **Library replay** (the same machine): stored E9 checkpoints for P = 12 and 13, seed 27, re-evaluated with windows 64 and 8,192, are bitwise identical to their committed records (4 of 4), and a retrain of `e9_p13__arith__s27` reproduces its loss curve, validation curve, test metrics, and positional snapshot bit for bit.

### 13.9 Commands

```bash
python experiments/run_suite.py pilot_e10_spread --workers 4      # development spread pilot (data seed 7)
python experiments/e10_copy_eval.py --suite pilot_e10_spread --workers 4
python experiments/e10_power.py                                   # seed count -> results/summary/e10_power.json
make e10 PY=~/mamba/envs/arithscope/bin/python WORKERS=4          # E10 runs, copied-law evaluations, verdicts
```

### 13.10 Deviations log (E10)

No entries.
