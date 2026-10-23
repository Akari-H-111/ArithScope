# E10 specification (authoritative for code and for PREREGISTRATION.md §13)

Written by the orchestrator, 2026-10-23, before any E10 or E10-spread run. Code must match this
file exactly; if something here is impossible or ambiguous, stop and report instead of improvising.

## Arms (per period P ∈ {12, 13} and seed s)

| Arm | `variant` | `evidence` | Role |
|---|---|---|---|
| `arith` | `arith` | none | ArithScope alone (E8 `reference`, E9 W = 8,192) |
| `arith_ev` | `arith` | `{"kernel": "arith"}` | + evidence head, ArithScope kernel |
| `arith_evfund` | `arith` | `{"kernel": "cosine_fund"}` | + evidence head, matched fundamental cosines (same 16 coefficients) |
| `rope` | `rope` | none | RoPE alone |
| `rope_ev` | `rope` | `{"kernel": "arith"}` | + evidence head, ArithScope kernel |

Experiment names `e10_p12`, `e10_p13` (spread pilot: `pilot_e10_p12`, `pilot_e10_p13`). Within a period and
seed, `arith`, `arith_ev`, `arith_evfund` share initial base weights and training batches; so do `rope`, `rope_ev`.
Everything else is E8/E9's setup: task periodic denoising (V = 16, ε = 1/2), bank M = {1..16}, 2 layers,
2,000 steps at length 64, batch 32, hyperparameters `_hp("periodic", variant)`. Nothing is tuned for E10.

Evaluation lengths (64, 256, 1024, 8192); `eval_tokens` 65,536 (64 sequences at 1,024; 256 at 256);
`eval_tokens_long` 65,536 (8 sequences at 8,192). Data seed `E10_DATA_SEED = 20261020` (fresh; E1–E9 used
20261014–20261019). Seeds `E10_FIRST_SEED = 42` onward (E1–E9 used 0–41); the count comes from the rule below.

Spread pilot `pilot_e10_spread`: the same arms and evaluation, development data seed 7, seeds 1000–1003.

## Copied-observation law (task kind `periodic_copied`)

`CopiedPeriodicDenoising(periods, vocab_size=16, noise=0.5, copies=k)`. It draws exactly the random
numbers `PeriodicDenoising` (query_rate 0) draws, in the same order, then replaces each token by the token of
its block leader: for position t with residue r = t mod P and visit v = t // P, the leader is
r + P·(v − v mod k). So each residue class is cut into blocks of k consecutive visits, one observation per
block is copied to the whole block, and the per-position marginal law is unchanged. `Batch.observed` marks
the leaders (the positions carrying new evidence), so the plurality oracle and the exact Bayes posterior count
leaders only. With k = 1 it must equal `PeriodicDenoising` bitwise (tokens, targets, oracle).
E10 uses **k = 4**.

Exact Bayes NLL (both laws): NLL_t = logsumexp_b(c_b log ρ) − c_true log ρ, with c_b the number of evidence
positions j ≤ t, j ≡ t (mod P), x_j = b (all positions for the independent law, leaders for the copied law),
ρ = (1 − ε + ε/V)/(ε/V) = 17.

## Metrics (per run; all from records, never from checkpoints except the copy evaluation)

- far NLL, far accuracy: E6's, positions 4,096–8,191 of the N = 8,192 test (`e6_verdicts.region`).
- excess NLL at 1,024: NLL over all positions of the N = 1,024 test minus the exact Bayes NLL on the same
  positions (Bayes recomputed from the regenerated test batch; the batch digest must match the record).
- copy NLL at 256: NLL over all positions of the copied-law (k = 4) test at N = 256 (256 sequences, data seed
  E10_DATA_SEED, generator parts as `train.run` builds them from the copied task's dict), from
  `experiments/e10_copy_eval.py`, which loads each checkpoint and records its SHA-256. The copy evaluation also
  records N = 1,024, and the exact copied-law Bayes NLL, for description only.
- independent NLL at 256: NLL over all positions of the ordinary N = 256 test (from the run record).
- learned-kernel quantities (from the run record's `evidence` key): cosine between β and the D_P
  coefficients (φ(m)/P for m | P, 0 otherwise, over the bank's orders), and the effective scale
  λ_eff = (Σ_{m | P} β_m) · (mean diagonal − mean off-diagonal of W_out).

All comparisons are paired per seed within a period, with 95% t intervals over seeds (`e6_verdicts.ci95`).

## Claims and decision rules

| | Claim | Supported if | Falsified otherwise |
|---|---|---|---|
| **V1** | The head lowers the far NLL of ArithScope alone, at no cost in accuracy. | at both periods: far NLL (`arith_ev − arith`) interval below 0, **and** far accuracy (`arith_ev − arith`) lower bound above −1 pp | either fails at either period |
| **V2** | The head follows the Bayes NLL at 16× the training length. | at both periods: upper bound of the interval of `arith_ev`'s excess NLL at 1,024 below **0.005** nats | upper bound ≥ 0.005 at either period |
| **V3** | At the prime period the exact basis matters (C2's structure). | at P = 13: far NLL (`arith_evfund − arith_ev`) interval above 0, **and** far accuracy (`arith_ev − arith_evfund`) lower bound above **+10 pp** | either fails |
| **V4** | Drop-in: the head restores RoPE's extrapolation. | at both periods: far accuracy (`rope_ev − rope`) lower bound above **+20 pp**, **and** far NLL (`rope_ev − rope`) interval below 0 | either fails at either period |
| **B1** | Boundary (preregistered prediction of harm): under copied observations the head is worse than no head. | at both periods: copy NLL at 256 (`arith_ev − arith`) interval above 0 | interval reaches 0 at either period |

Secondary predictions (no decision depends on them):
- **S12** (no failure at P = 12): at P = 12 the far-accuracy interval of `arith_ev − arith_evfund` lies within ±1 pp.
- **S13** (SGD finds the construction): cosine(β, D_P) ≥ 0.99 in every seed of `arith_ev` and `rope_ev`, both periods.
- **S14** (Bayes scale): λ_eff within ±10% of log 17 in every seed of `arith_ev` and `rope_ev`, both periods.
- **S15** (crossover): independent NLL at 256 (`arith_ev − arith`) interval below 0 at both periods.

Sanity checks (no verdict until resolved): every run completed; one code commit for all records; within a period
and length all records share one test digest; each copy record's checkpoint SHA-256 matches a run of the suite;
`len(E10_SEEDS)` equals `e10_power.json`'s `n_seeds`. A failed run is reported, never replaced, and its seed is
dropped from every comparison of that period that uses the failed arm.

Sensitivity only: Holm-adjusted one-sided p-values over the interval tests of V1–V4 and B1.

## Seed-count rule (`experiments/e10_power.py`, fixed before the spread pilot is run)

For each claim, a standardized effect per period part, then joint power = product over its parts (periods,
and the two conditions of V3/V4); power of one part with standardized effect d and n seeds is the noncentral t
probability P(T' > t_{0.975, n−1}), T' ~ t(n − 1, d√n) (as `e9_power.power`). Spread = sample SD raised to its
one-sided 80% upper confidence limit, s·sqrt(df/χ²_{0.20, df}).

- **V1** (far NLL part only; the accuracy guard is not planned, as in E8/E9): D = −(far NLL of ArithScope alone),
  from E8's `reference` seeds 6–26 and E9's W = 8,192 seeds 27–41 (36 values per period, confirmatory data from
  the same setup). d = |mean D| / spread. Assumption check: in the spread pilot, `arith_ev`'s far NLL is ≤ 0.001
  in every seed and period; otherwise stop.
- **V2**: excess NLL at 1,024 of `arith_ev` in the spread pilot; d = (0.005 − mean)/spread; stop if mean ≥ 0.005.
- **V3** (P = 13): parts far NLL (`arith_evfund − arith_ev`), d = mean/spread; and far accuracy (`arith_ev − arith_evfund`), d = (mean − 0.10)/spread; stop if a mean is on the wrong side.
- **V4** (both periods): far accuracy (`rope_ev − rope`), d = (mean − 0.20)/spread; far NLL (`rope_ev − rope`), d = −mean/spread.
- **B1** (both periods): copy NLL at 256 (`arith_ev − arith`), d = mean/spread; stop if mean ≤ 0.
- n = max over claims of the smallest n ≥ 5 with that claim's joint power ≥ 0.90. If any claim needs more than
  20 seeds per period, record it and set no seed count (the design then needs a decision).
- Record a sensitivity table (point SD instead of the UCL; target 0.80; effects halved). No decision uses it.
