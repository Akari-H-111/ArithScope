> **Archived external audit.** Independent adversarial review produced with ChatGPT and supplied by the repository owner on 2026-10-15. Reproduced verbatim below; it reviews `main` at `0ce3dce`. The response, with verification and fixes for each finding, is [`../AUDIT_RESPONSE.md`](../AUDIT_RESPONSE.md).

# ArithScope: First Independent Adversarial Cold Audit

**Audit date:** 2026-10-15 (Asia/Taipei)  
**Frozen target:** `Akari-H-111/ArithScope`, `main` at `0ce3dce2e415821793e1af33852675031dd3d25a`  
**Scope:** read-only. No GitHub writes, no model retraining.  
**Auditor status:** Independent inspection of repository sources and selected run JSON; independent mathematical counterexamples and small calculations; GitHub Actions status verified. Full repository clone/CI/training replay was **not** completed because the audit container could not resolve github.com for `git clone`. Consequently this document does **not** certify the 182 runs independently reproduced.

## Overall verdict

- **Core arithmetic and finite-comb mathematics: provisionally PASS.** The PSD feature-map argument and T3 optimal periodic comb proof withstand this pass. Independently recomputed the exact identity for periods 12, 13, 24 with rational arithmetic. No proof assistant audit is claimed.
- **General attention API/backends: CORRECTION REQUIRED.** Position-index causality disagrees between backends on duplicate/out-of-order 1-D positions; sparse positions can trigger allocation proportional to offset span instead of token count.
- **Learned-frequency corollary in MATH/T5: CORRECTION REQUIRED.** The theorem's joint nonresonance hypothesis is correct; its informal replacement by individual frequency irrationality is not.
- **Empirical periodic-denoising claim: meaningful and recorded, but not independently retrained.** The preregistration commit precedes confirmatory run commits according to the repository history and records. CI green is independently observed, but CI runs tests, not the 182 CPU training jobs.
- **LLM-use claims: not established.** The existing repo explicitly limits itself to tiny CPU synthetic tasks. Do not generalize to pretrained LLM perplexity, natural language, speed, or SOTA.

## Finding F1: inconsistent causal masking (P1, confirmed by exact minimal reproducer)

**Files:**
- `arithscope/attention.py` lines 95-106 (`allowed_mask`, uses `key_position <= query_position`), 158-162 (`sdpa`, index-triangular fast path), 190 (`explicit`), 215-216 (`folded`, index-triangular fast path).
- `tests/test_attention.py` tests use increasing `torch.arange` position sequences; no test covers duplicate or out-of-order positions.

For `positions=[0,0]`, the explicit path allows both keys for both queries, including future token index 1 at query index 0. With q=k=1 and v=[1,10], the explicit attention outputs `[5.5,5.5]`, whereas index-causal fused SDPA outputs `[1.0,5.5]`. For `positions=[2,1]`, explicit outputs `[5.5,10]`, fused outputs `[1,5.5]`. Independently reproduced using `torch.nn.functional.scaled_dot_product_attention` on CPU PyTorch 2.10.0.

**Fix:** either (A) explicitly validate that all live positions are strictly increasing, and reject anything else, or (B) make index causality `key_index <= query_index` authoritative for all backends, including blocked explicit queries and FlexAttention, and specify optional additional position-order constraints. For practical packed/padded inference (B) is preferable. Extend regression tests for duplicates, unsorted positions, per-batch positions, left padding and cache-offset semantics. Beware query blocking requires a global query-index offset. Do not merely modify the fast path to use `position <=` and retain future-token leakage.

## Finding F2: span-dependent catastrophic allocation (P1/P2, static definite)

**Files:** `arithscope/bias.py` lines 53-57 (`diff`, `lo/hi`, `torch.arange(lo, hi+1)`), `arithscope/flex.py` lines 60-65 (same contiguous range approach).

For two positions `[0, 10**12]`, the pairwise differences are only `0` and `±10**12`, but the code attempts to construct 2,000,000,000,001 distinct offset entries. Merely storing the offset vector as int64 would require roughly 16 TB, and a four-head float32 offset table approximately 32 TB. This was assessed without calling the allocating code to avoid an intentional OOM. The same issue can affect `sdpa` and the explicit bias path, and FlexAttention's table. The low-rank **folded** ArithScope path avoids it.

**Fix:** avoid contiguous offset ranges for sparse positions. Use unique observed offsets for an explicit dense mode (with a cost-aware branch), or evaluate `offset % m` directly for Ramanujan feature/bias lookup. For FlexAttention, use compact per-order residue tables and integer modular lookup inside `score_mod`, subject to compilation benchmarks. Validate subtraction overflow near int64 limits and bound supported orders/feature budgets. Add safe memory-limit tests that do not allocate huge ranges.

## Finding F3: invalid informal generalization of Theorem T5 (P1 mathematical statement)

**File:** `docs/MATH.md`, around lines 133-155, particularly line 144.

T5's formal hypothesis disallows *every joint integer relation* `c·theta ∈ 2π(Q\Z)`. The application text says the same conclusion holds for learned real frequencies **unless one frequency individually** is a rational multiple of `2π`; that criterion is insufficient.

**Counterexample:** Let `theta_1=sqrt(2)`, `theta_2=sqrt(2)+π`. Neither `theta_j/(2π)` is rational, yet `theta_2-theta_1=π=2π*(1/2)`. Choose the standard separable RoPE-like logit

`g_n=cos(n theta_1)+cos(n theta_2)=[1+(-1)^n] cos(n sqrt(2))`.

With residue classes modulo `P=2`, even `n` have mean weight `I_0(2)` and odd `n` have mean weight `1` (equidistribution of `n sqrt(2)` modulo `2π`). Hence the even-position attention mass converges to

`I_0(2)/(I_0(2)+1) ≈ 0.6950834`, not `1/2`.

A direct NumPy calculation over 200,000 offsets yielded `0.6950870`. This does **not** refute T5 under its correct joint nonresonance hypothesis, nor its application to ideal standard algebraic RoPE frequencies; it refutes only the text's extension based on individual irrationality.

**Fix:** replace the informal learned-frequency sentence with the full joint nonresonance assumption or a suitable almost-everywhere measure-theoretic result for continuously distributed frequency vectors, with clearly stated qualifications for optimization dynamics and finite precision.

## Finding F4: accuracy vs NLL divergence (P2, confirmed in committed run JSON)

**Files:** `results/runs/e1_periodic/e1_p13__rope+arith__s[1..5]__*.json` and matching `e1_p13__arith__s[1..5]__*.json`; `docs/RESULTS.md` primarily plots accuracy.

At `P=13`, `N=8192`, the **extrapolated region** mean NLL across five matched seeds is:

- ArithScope only: `0.0672505`
- RoPE + ArithScope: `0.2195708`
- Paired increase: `+0.1523204`, all five seeds worse, 95% t interval over seeds approximately `[+0.0814,+0.2232]`.

By contrast, mean RoPE+ArithScope accuracy at this length remains `~99.94%`. The same five RoPE+ArithScope seeds have mean NLL `0.09786` at `N=1024`, rising to `0.21957` at `N=8192` while accuracy rises. These observations indicate that near-oracle classification accuracy does **not** imply NLL/calibration stability, but they are not a formal calibration diagnosis. The N=8192 sample comprises fewer evaluated predictions (16,384 tokens total vs 65,536 at N=1024), so test-sampling effects and rare extreme losses must also be analyzed.

**Fix:** make paired NLL/perplexity tables and sequence/position-stratified loss distributions first-class. Compare per-example NLL outliers, logits, calibration (ECE/Brier), head ablations, and change in RoPE/Arith-only behavior. Include uncertainty due to finite test sets, not just model seeds. Do not label RoPE+ArithScope as better than ArithScope only at N=8192 on NLL.

## Finding F5: scope and comparison constraints (research-risk, not a code defect)

**Files:** `results/summary/tables.md`, `docs/PREREGISTRATION.md`.

- At `P=13,N=1024`, the full Fourier bank matches ArithScope accuracy (both 99.5%); at `P=12` it also matches. The same-period same-parameter **fundamental cosine** baseline is weaker at prime periods by construction. The defensible contribution is tied primitive-character parameterization and its exact algebraic/length behavior, not exclusive expressivity compared with a full Fourier bank.
- `E5` mixed unlabeled periods is exploratory, three seeds; at shorter extrapolation lengths learned frequencies can outperform RoPE+ArithScope. Do not promote selective longest-length wins into universal superiority.
- The confirmatory family uses four declared claims with unadjusted 95% CIs and n=5 seeds. Large C1/C2 effects are informative, but familywise error is not controlled.
- The dropped MQAR task after weak pilots means the project has no supported general content-retrieval result yet. Natural-language, pretrained-LLM loss, GPU trainability/Flex backward, and large-scale cost remain open.

## Positive controls independently checked

- For `P in {12,13,24}`, `Σ_{d|P,d>1} φ(d)/(P-1) K_d(n) = [P*(P|n)-1]/(P-1)` for all `0 <= n < 2P` using independent exact-rational SymPy/Python derivation. Corresponding margins: `12/11`, `13/12`, `24/23`.
- The per-position feature Gram argument for PSD is mathematically valid; the independent check does not rely on the project's oracle code.
- GitHub tree at reviewed SHA contained 247 run JSONs, of which 182 are the declared E1/E2/E3/E5 confirmatory runs and 65 are pilots; `docs/PREREGISTRATION.md` was part of commit `cdea384eb0966a544365a2a6427bd6e1bc930ae7` (2026-10-14 15:28:53 UTC).
- Latest GitHub Actions `tests` run 38657195083 at the reviewed SHA completed successfully: `ruff`, stdlib-oracle, and CPU PyTorch test jobs. Actions testing does not rerun training or verify all stored run JSONs.

## Suggested next Codex pass (do not rewrite source results)

1. **F1**: add regression tests for repeated/unsorted positions and align causal semantics in every backend, or document/reject unsupported layouts.
2. **F2**: guard/reroute sparse offsets so allocation is bounded by sequence length/unique offsets, not numeric position span; add overflow tests.
3. **F3**: correct T5 learned-frequency statement and add this exact counterexample to math regression notes; do not weaken the valid RoPE theorem.
4. **F4**: regenerate paired NLL report and outlier diagnostics from immutable JSONs without retuning; preserve all original evaluation outcomes.
5. **F5**: expand GPU backward/throughput evidence, add stronger Fourier baselines and non-periodic retrieval probes before an LLM-scope announcement.

**Public research wording today:** "ArithScope investigates an arithmetic relative-position attention bias for exact long-range periodic retrieval, with a finite-group optimality theorem, a PyTorch implementation and preregistered synthetic CPU results. Natural-language gains and production-scale improvements remain open." 

**Repository was not modified by this audit.**
