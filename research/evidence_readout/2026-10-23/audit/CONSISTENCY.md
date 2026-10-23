# Theory vs pilot: does T9 predict what the evidence-head pilot did?

Referee audit, 2026-10-23. Code: `consistency.py` (main), `consistency_extra.py` (window-fold horizon, rare-failure bound), `ptread.py` (stdlib checkpoint reader).
Outputs: `consistency_out.txt`, `consistency_out.json`, `consistency_extra.txt`, `consistency_run.log`. Run with `/usr/bin/python3 -I -B`.
Everything below is **NUMERICAL** (a computation from fitted coefficients), applied to a **development pilot** (one training seed, development data seed 7). Nothing here upgrades any claim.

## 1. What was computed, and from what

- **Source of coefficients.** `pilot_evidence.json` holds the learned kernel only partially: $`\beta`$ for `arith` (full), the kernel row $`\kappa(0..2P-1)`$, six of the sixteen learned frequencies for `cosine_learned`, and only *summary statistics* of $`W_{\rm out}`$ (diag mean, off-diag mean, …). It does **not** hold the `cosine_fund`/`cosine_full` coefficients or the full $`W_{\rm out}`$. I read these from the checkpoints `ArithScope-evidence/checkpoints/pilot_evidence/*.pt` (read-only; zip+pickle, no torch).
  Validation against the JSON: the reconstructed kernel matches the JSON row $`\kappa(n)`$, $`n<2P`$, to $`\le5\cdot10^{-16}`$ in every arm; $`W_{\rm out}`$ diag-mean and off-diag-mean equal the JSON values exactly. The JSON-only route is therefore a subset of what I used (for `arith`: $`\kappa_P(0)=\sum_{m\mid12}\beta_m=0.7414+0.7554+1.4501+1.4524+1.4924+2.9765=8.868`$, matching).
- **Kernels** exactly as `arithscope/evidence.py` builds them: `arith` $`\sum\beta_mK_m`$ with `arith.kernel_table` residue tables; `cosine_fund` $`\cos(2\pi((n)\bmod m)/m)`$, orders $`1..16`$; `cosine_full` $`\cos(2\pi((an)\bmod m)/m)`$ over the 41 reduced harmonics ($`2a\le m`$); `cosine_learned` $`\sum_fA_f\cos(\omega_fn)`$ (float64 phases, the pilot's learned $`\omega,A`$); `uniform` $`\kappa\equiv1`$.
- **Fold.** For periodic arms $`\kappa_P(s)`$ is the mean of $`\kappa`$ over $`n\equiv s\ (P)`$ across a full period $`\Lambda=\mathrm{lcm}(L,P)`$ ($`=720\,720`$ for orders $`1..16`$), computed per component over $`\mathrm{lcm}(m_c,P)`$ and checked against the brute-force average over all $`720\,720`$ offsets for four arms (max deviation $`3.2\cdot10^{-12}`$). For `cosine_learned` there is no period: I use the *window fold* $`\hat\kappa_{P,N}(s)=\tfrac PN\sum_{n<N,\,n\equiv s}\kappa(n)`$ at $`N=8192`$ (the pilot's longest length); for the periodic arms the *dominance margin* of the window fold at $`N=256..8192`$ differs from that of the exact fold by $`\le0.3`$ out of $`\sim6\!-\!9`$ (worst: `cosine_full`, $`P=13`$, $`N=256`$: $`8.09`$ vs $`7.83`$), so the finite-length transient is small there (cf. REVIEW_T9 §1, item 1).
- **Margin and Monte Carlo.** For a motif $`\mu`$ and query residue $`t`$: $`e_t(a)=\tfrac1P\sum_s\kappa_P(s)\pi_{t-s}(a)`$ ($`V=16`$, $`p=17/32`$, $`\nu=1/32`$), logits $`\ell=We_t`$, margin $`s_t=\ell(\mu_t)-\max_{b\ne\mu_t}\ell(b)`$ (T9c). 4000 uniformly random motifs, seed $`20261023+P`$, both $`W=I`$ and the checkpoint's learned $`W_{\rm out}`$. "All positive" = every residue has $`s_t>10^{-9}`$.
  Because the pilot only has summary stats of $`W_{\rm out}`$, note that any $`W=(d-o)I+oJ`$ with $`d>o`$ gives the same signs as $`W=I`$; the learned $`W_{\rm out}`$ is close to this (diag mean $`0.18..0.36`$, off-diag $`-0.04..-0.10`$) but not equal, and the difference matters only for ties (uniform arm).
- **Finite-$`i`$ noise-free predictor.** $`D_i(a)=\mathbb E E_i(a)=\sum_{n\le i}\kappa(n)\pi_{i-n}(a)`$ exactly (class sums), $`\ell=WD_i`$, argmax with ties split, averaged over $`i\in[4096,4096+P)\cup[8192-P,8192)`$ (all residues) and 1500 random motifs. This ignores the rest-of-network logits $`b_i`$; "$`G_0`$" below is a crude stand-in (constant extra logit on the true symbol).

## 2. Results (arith backbone; `rope`/`rope+arith` backbones with the `arith` kernel behave identically)

$`\kappa_P(0)`$ and $`\sum_{s\ne0}|\kappa_P(s)|`$ are of the fold; "dom." = $`\kappa_P(0)-\sum_{s\ne0}|\kappa_P(s)|`$. $`\lambda_{\rm eff}=\kappa_P(0)\,(\bar W_{\rm diag}-\bar W_{\rm off})`$.
"Linear-rate MC" columns are the fraction of random motifs with every residue margin positive (all-pos), and the fraction of (motif, residue) pairs with positive / negative margin. Obs = pilot at $`N=8192`$ (far half, 2 test sequences).

| $`P`$ | arm | $`\kappa_P(0)`$ | $`\sum\lvert{\rm off}\rvert`$ | dom. | $`\lambda_{\rm eff}`$ | all-pos, $`W=I`$ | res +/− ($`W_{\rm out}`$) | noise-free pred. acc. | obs far acc | obs far NLL |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | arith | 8.868 | 0.295 | **+8.57** | 2.844 | 1.000 | 1.00 / 0.00 | 1.000 | 1.000 | 0.000 |
| 12 | cosine_full | 9.759 | 1.073 | **+8.69** | 2.793 | 1.000 | 1.00 / 0.00 | 1.000 | 1.000 | 0.000 |
| 12 | cosine_fund | 5.806 | 8.032 | −2.23 | 1.538 | 1.000 (0/4000; 10/$`10^5`$ fail) | 1.00 / 0.00 | 1.000 | 1.000 | 0.000 |
| 12 | cosine_learned | −0.195* | 4.495* | −4.69* | −0.06 | 0.000 | 0.02 / 0.98 | 0.052 | 0.050 | 104.07 |
| 12 | uniform | 1.000 | 11.000 | −10.00 | 0.111 | 0.000 | 0.21 / 0.79 | 0.212 | 0.292 | 19.99 |
| 13 | arith | 6.437 | 0.115 | **+6.32** | 2.838 | 1.000 | 1.00 / 0.00 | 1.000 | 1.000 | 0.000 |
| 13 | cosine_full | 10.301 | 2.476 | **+7.83** | 2.761 | 1.000 | 1.00 / 0.00 | 1.000 | 1.000 | 0.000 |
| 13 | cosine_fund | 0.964 | 5.385 | −4.42 | 0.209 | **0.001** | 0.36 / 0.64 | 0.362 | 0.482 | 9.64 |
| 13 | cosine_learned | 0.036* | 0.762* | −0.73* | 0.008 | 0.000 | 0.13 / 0.87 | 0.071 | 0.475 | 3.08 |
| 13 | uniform | 1.000 | 12.000 | −11.00 | 0.088 | 0.000 | 0.21 / 0.79 | 0.205 | 0.271 | 14.47 |

\* window fold at $`N=8192`$ (no exact fold exists). Per-arm detail, rope backbones, window-fold margins at $`N=256..8192`$, and fold vectors are in `consistency_out.txt`.
Reference points: $`\log17=2.833`$ (Bayes scale for $`\rho=17`$); chance accuracy $`1/16=0.0625`$.

Fold vectors (arith backbone), $`s=0..P-1`$:
`cosine_fund`, $`P=12`$: $`5.81,\,0.80,\,-1.13,\,0.03,\,0.63,\,-0.68,\,1.48,\,\dots`$ (a bumpy kernel, off-comb entries up to a quarter of $`\kappa_P(0)`$);
`cosine_fund`, $`P=13`$: $`0.96,\,0.91,\,0.76,\,0.55,\,0.32,\,0.13,\,0.03,\,0.03,\dots`$ (a smooth bump $`a_1+a_{13}\cos(2\pi s/13)`$: neighbours of residue 0 carry $`0.91`$ and $`0.76`$ of $`\kappa_P(0)`$).

## 3. Does T9 predict the pilot?

| Brief's expectation | T9's prediction (computed) | Pilot | Match |
|---|---|---|---|
| arith fine at both $`P`$ | dominant (+8.57, +6.32), all motifs positive; $`D_P`$ in span (T9b.1) | far NLL $`0.000`$, acc $`1.000`$ | **yes** |
| cosine_full fine at both $`P`$ | all harmonics $`a\le\lfloor P/2\rfloor`$ present ⇒ $`D_P`$ in span (T9b.3); dominant (+8.69, +7.83) | same | **yes** |
| cosine_fund fine at $`P=12`$ | *not dominant* (−2.23) — and no coefficient choice can be (best margin is exactly $`0`$ at $`(\tfrac16,\tfrac16,\tfrac13,0,\tfrac13,0)`$ [`recheck.py` G]); but fails on only $`\sim10^{-4}`$ of random motifs | acc $`1.000`$, NLL $`0.011`$ at 8192 | **yes, but not via T9c.5** |
| cosine_fund failing at $`P=13`$ | fold is a smooth bump; $`99.9\%`$ of random motifs have a non-positive margin somewhere; $`\approx64\%`$ of (motif, residue) pairs wrong sign | acc $`0.482`$, $`4172/8192`$ positions $`>1`$ nat | **yes** (direction and rough size) |
| uniform failing | $`\kappa_P\equiv1`$; $`s_t\propto h_c-h_b`$ (motif symbol counts); all-pos only for constant motifs (prob $`\approx0`$); acc $`\approx\mathbb E[h_{\max}]/P`$ | acc $`0.292`$ ($`P=12`$), $`0.271`$ ($`P=13`$); predicted $`0.21`$, $`0.205`$ | **yes**; difference within 2-motif sampling (probability that a 2-motif mean lies within $`0.1`$ of the observed value: $`0.72`$, $`0.79`$) |
| cosine_learned failing | T9d: asymptotically no linear rate at all ($`\bar\kappa=0`$); here, finite-window drift | acc $`0.050`$ ($`P=12`$), $`0.475`$ ($`P=13`$) | **direction yes; $`P=12`$ size yes; $`P=13`$ size no** (below) |

Additional quantitative agreement that is not a failure test but a sanity check on the theory: for the arith-kernel arms $`\lambda_{\rm eff}=\kappa_P(0)\cdot(\bar W_{\rm diag}-\bar W_{\rm off})`$ is $`2.844,2.844,2.844`$ ($`P=12`$; arith, rope, rope+arith) and $`2.838,2.838,2.838`$ ($`P=13`$) against $`\log17=2.833`$ — within $`0.4\%`$ of the Bayes scale of T9a — and the learned $`\beta_m/\beta_1`$ for $`P=12`$ are $`1.02,1.96,1.96,2.01,4.01`$ against $`\varphi(m)/\varphi(1)=1,2,2,2,4`$ for $`m=2,3,4,6,12`$. The off-divisor $`\beta`$ are $`\le0.032`$ (JSON). (One seed; development data.)

### 3.1 Where the match is weaker than the table suggests

1. **T9c.5 (strict dominance) is not the discriminating prediction.** Both `cosine_fund` arms fail it, and at $`P=12`$ every fundamental-cosine fold fails it ($`\max`$ margin $`=0`$). What separates $`P=12`$ from $`P=13`$ is the *random-motif* computation, which T9 does not contain: at $`V=16`$, adverse collision patterns are rare unless the off-comb entries are comparable to $`\kappa_P(0)`$ (they are at $`P=13`$: $`0.91`$ and $`0.76`$ vs $`0.96`$; not at $`P=12`$: $`\le1.48`$ vs $`5.81`$). T9c.5 alone would "predict" a failure at $`P=12`$ that the pilot (rightly) does not show.
2. **`cosine_learned` fails by pre-asymptotic drift, not by T9d's residue-blindness.** Its window fold at $`N=8192`$ is far from constant (see §2: off-comb entries up to $`0.78`$ vs $`\hat\kappa(0)=-0.20`$ at $`P=12`$). Over $`N=8192,65\,536,262\,144,1\,048\,576`$ the window fold shrinks as $`\hat\kappa(0)=-0.196,\,0.028,\,0.0057,\,0.0039`$ ($`P=12`$) and $`0.036,\,0.0075,\,0.0016,\,0.0001`$ ($`P=13`$) with dominance margins $`-4.69,-0.64,-0.11,-0.025`$ / $`-0.73,-0.077,-0.022,-0.006`$: i.e. T9d's constant fold $`\bar\kappa=0`$ is reached only at $`N\gtrsim10^{5}\!-\!10^{6}`$ (`consistency_extra.txt`). The learned frequencies within $`\delta\le4.6\cdot10^{-3}`$ of a harmonic $`2\pi a/P`$ (eight at $`P=12`$, three at $`P=13`$) have drift horizons $`1/\delta\in[217,\,1914]`$, all well below $`8192`$ — the drift regime that MATH.md T5 and T9d's remarks mention but that no theorem quantifies. T9 is *consistent* with the failure; it does not *predict* its length scale.
3. **$`P=13`$ `cosine_learned`: predicted far accuracy $`0.07`$ (evidence alone) to $`0.34`$ (with $`G_0=6`$), observed $`0.475`$.** The probability that the mean of two random motifs lands within $`0.1`$ of $`0.475`$ is $`0.00`$ for $`G_0\in\{0,4\}`$ under my predictor, so this is not sampling noise of the predictor. The most likely cause is the bounded rest-of-network term $`b_i`$ (the base network alone has $`99\%`$ accuracy at these lengths, "arith + none"), which T9 treats only as bounded and my crude $`G_0`$ does not model; the learned window fold for this arm is tiny ($`|\hat\kappa|\le0.12`$), so the evidence term is comparable to $`b_i`$ and neither dominates. Not resolvable from this JSON.
4. In all failing arms the predicted wrong-sign fraction is at least the observed error rate ($`0.79`$ vs $`0.71`$, $`0.79`$ vs $`0.73`$, $`0.64`$ vs $`0.52`$, $`0.98`$ vs $`0.95`$, $`0.87`$ vs $`0.53`$), the direction expected if $`b_i`$ pulls toward the truth.

## 4. What one seed (and two test sequences) can and cannot show

- **Sampling.** At $`N=8192`$ the pilot evaluated $`16384//8192=2`$ sequences ($`4`$ at $`N=4096`$, $`64`$ at $`1024`$). A "failure rate" is per residue within a sequence, not a fraction of sequences: a motif with a bad residue pattern fails at those residues only. With per-motif SD of accuracy $`0.04\!-\!0.10`$ (my Monte Carlo), a two-motif mean is accurate to about $`\pm0.05\!-\!0.07`$. This is enough to separate $`\approx1.0`$ from $`\approx0.3`$; it cannot separate $`0.36`$ from $`0.48`$ ($`P=13`$ `cosine_fund`) or $`0.21`$ from $`0.29`$ (uniform).
- **Motif-specific.** The pilot's motifs are fixed by data seed 7; the prediction is for uniformly random motifs. The `arith`/`cosine_full` arms would pass on any motif (dominance), so those two conclusions do not depend on the draw; the others do.
- **One training seed.** The learned coefficients are one draw. The prediction is conditional on them; it says nothing about whether another seed reaches a dominant fold (for `cosine_fund` at $`P=12`$ no seed can, in the strict sense; for `cosine_learned` the outcome depends on where frequencies drift).
- **What T9 would have to say to be falsifiable here.** A pre-registered prediction would need (i) the random-motif corollary for T9c.5, (ii) a finite-$`N`$ statement for T9d with the $`1/\delta`$ horizon, (iii) a model for $`b_i`$. Without them, the checks above are consistency checks after the fact, on a development pilot, run by the same people who wrote the theory (although the code here is independent of `check_t9.py`).
- **Labels.** Nothing here changes `CLAIM_LEDGER.md`. These are NUMERICAL computations on EMPIRICAL development-pilot outputs.

## 5. Reproduce

```
/usr/bin/python3 -I -B research/evidence_readout/2026-10-23/audit/consistency.py        # ~1 min
/usr/bin/python3 -I -B research/evidence_readout/2026-10-23/audit/consistency_extra.py  # ~3 min (N up to 2^20)
/usr/bin/python3 -I -B research/evidence_readout/2026-10-23/audit/recheck.py            # ~15 s
```
The scripts read `pilot/pilot_evidence.json` and, read-only, `../ArithScope-evidence/checkpoints/pilot_evidence/*.pt` and `arithscope/arith.py`.
