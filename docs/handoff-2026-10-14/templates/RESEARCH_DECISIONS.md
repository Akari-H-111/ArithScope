# Decisions / negative findings log (template)

Record every material design decision without rewriting past experimental evidence.

| Date | Scope | Change/hypothesis | Math condition | Counterfactual baseline | Evidence command/log | Status | Next action |
|---|---|---|---|---|---|---|---|
| 2026-10-14 | initial handoff | prioritize math reference + attention logit bias | normalized Ramanujan kernel and identifiability theorem | RoPE, Fourier, ALiBi | `python3 -m unittest discover -s tests -v` | reference tests only | implement Torch stage |

Fields for negative reports: failure condition, exact task/sample size, whether development or frozen-test data, scope of the null, actual runtime overhead, mechanism explanation supported by test, and whether result is a counterexample or merely negative evidence.
