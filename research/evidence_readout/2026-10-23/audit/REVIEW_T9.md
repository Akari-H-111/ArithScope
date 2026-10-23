# Referee report: T9 draft (`theory/T9_DRAFT.md`, 2026-10-23)

Referee: adversarial audit, 2026-10-23. Scope: Part 1 of the brief. Part 2 (theory vs pilot) is in `CONSISTENCY.md`.
Everything I ran is in this directory (`recheck.py` → `recheck.log`; helper `ptread.py`). I used stdlib Python only and my own
implementations of $`K_m`$, $`\varphi`$, $`\mu`$, ranks over $`\mathbb Q`$, enumerations; I did not import `check_t9.py`.
Nothing outside `audit/` was edited; nothing was committed.

## 0. Bottom line

I found **no wrong theorem**. Every statement L9.1–L9.3, T9a, T9b.1–4, T9c, T9d, T9e survives re-derivation, and every
claim I could test numerically (identity, fold lemma, span criteria, dominance, leakage numbers, T9e table) reproduced.
What I did find, most serious first:

1. **L9.1's constant makes T9c vacuous at the lengths of the pilot.** The bound is $`2\Lambda G`$ with $`\Lambda=\mathrm{lcm}(L,P)=720\,720`$ for the default bank. On the pilot's bank
   ($`\{1..16\}`$, $`P=12`$, random coefficients) the draft's bound is $`1.2\cdot10^{7}`$ while the true maximum deviation over $`i\le8192`$ is $`2.3`$
   [`recheck.py` K]. The draft never says that its finite-$`i`$ content is empty below $`i\sim10^{6}`$; "What this does not say" for T9c
   only mentions non-uniformity in $`\kappa`$. A per-component bound ($`\sum_c 2|\beta_c|\,\mathrm{lcm}(m_c,P)`$, still valid) is $`1.5\cdot10^4`$ times smaller.
2. **The strict-dominance criterion (T9c.5) does not discriminate the pilot's arms, and the narrative around it invites the wrong reading.**
   It is a *worst-motif* criterion. For $`V=16`$ random motifs have few collisions, so non-dominant kernels still succeed on almost all motifs:
   the fitted `cosine_fund` fold at $`P=12`$ has dominance margin $`-2.23`$ yet fails on only $`10/10^{5}`$ random motifs. At $`P=13`$ the same arm fails on $`>99.9\%`$ of motifs.
   Both fail dominance; only a random-motif computation separates them (`CONSISTENCY.md`). The draft has no random-motif corollary.
3. **T9d is silent exactly where the pilot lives, and the abstract row oversells it.** "Non-resonant kernels are residue-blind at linear order"
   is an $`i\to\infty`$ statement. For the pilot's `cosine_learned` the window fold at $`N=8192`$ is *not* residue-blind (off-comb entries up to $`0.78`$ vs $`\hat\kappa(0)=-0.20`$);
   it decays like $`1/N`$ and reaches the T9d value only for $`N\gtrsim10^{5}\!-\!10^{6}`$ (`consistency_extra.txt`). The pilot's failure is near-resonance drift, which T9 mentions but does not quantify.
4. **Small factual slips** (all fixable in a line): "240 identity checks" is **288** in code and JSON (L9.1 check); Hoeffding exponent in T9c.3 is off by a factor 2 for small $`i`$;
   T9b.2 cites "T9c.3" for what is T9c.5; "non-comb kernels have $`|\mathrm{gap}/i|\le5\cdot10^{-4}`$" is false for the period-13 cosine row ($`2.8\cdot10^{-3}`$);
   L9.2 is used for cosines but proved only for $`K_m`$; the T9e table omits the $`m=3,5`$ rows, which break monotonicity of the excess in $`m`$.
5. **Label / forbidden-claims hygiene**: "overconfident" and "confidence exceeds accuracy" (T9e) are calibration vocabulary;
   "a periodic unnormalized sum can be kept as $`P`$ running counts" (T9f) is a complexity remark; the novelty sentence in T9f.
   Details in §4.

## 1. Verdict table

| Item | Verdict | Step at issue | Fix |
|---|---|---|---|
| L9.1 | correct, with a gap | constant $`2\Lambda G`$ vacuous at pilot lengths; "240 identity checks" (l. 47) is 288 | state the per-component bound; fix count |
| L9.2 | correct, with a gap | proved for $`K_m`$ only, used for cosines (l. 208, bank sweeps) | add the character version (proof is the same) |
| L9.3 | correct | none (union over $`i`$ is handled; see §2.1) | optional: use $`\|\kappa_{[0,i]}\|_2`$ instead of $`G\sqrt{i+1}`$ |
| T9a (1–4) | correct | none | open item (ii) is closable (§2.4) |
| T9b.1 | correct | none ($`P=1`$, orders $`>P`$ handled; §2.2) | — |
| T9b.2 | correct, with a gap | "By T9c.3" (l. 121) should be T9c.5; NOTES.md "best CRT choice" is not best | strengthen: no dominant kernel exists in the whole span (§2.3) |
| T9b.3 | correct | none (§2.5) | — |
| T9b.4 | correct | none | — |
| T9c.1, .2, .4, .6 | correct | none | — |
| T9c.3 | correct, constant slip | $`\exp(-s_t^2 i/(8R^2))`$ (l. 176) | $`\exp(-s_t^2 i^2/(8(i+1)R^2))\le\exp(-s_t^2 i/(16R^2))`$ |
| T9c.5 | correct | needs only $`V\ge2`$ (§2.6); but is a worst-case criterion | add a random-motif corollary; do not use it to predict pilot arms |
| T9d | correct (as ASYMPTOTIC) | pre-asymptotic regime unquantified; table row wording | add the $`1/|\delta|`$ horizon to the statement |
| T9e | correct | label/vocabulary; omitted rows | rename "overconfident"; show all rows |

## 2. Re-derivations

### 2.1 L9.3 (Hoeffding + Borel–Cantelli) and the a.s. events

For fixed $`i,a`$: $`E_i(a)-D_i(a)=\sum_{j\le i}\kappa(i-j)(\mathbf 1[X_j=a]-\pi_j(a))`$, independent (given $`\mu`$), the $`j`$-th term in an interval of length $`|\kappa(i-j)|\le G`$.
Hoeffding: $`\Pr(|\cdot|\ge x)\le2\exp(-2x^2/\sum_j\kappa(i-j)^2)\le2\exp(-2x^2/((i+1)G^2))`$. With $`x_i^2=\tfrac32G^2(i+1)\log(i+2)`$ the exponent is $`-3\log(i+2)`$, so the bound is $`2(i+2)^{-3}`$ ✓.
$`\sum_{i,a}2(i+2)^{-3}\le2V\zeta(3)<\infty`$, and first Borel–Cantelli needs only these *marginal* probabilities; no independence between different $`i`$ is used.
So "the union over $`i`$" is handled exactly by summability, as the parenthetical says ✓. The a.s. event $`\Omega_L=\{\exists I:\forall i\ge I,\forall a:\ldots\}`$
depends on $`\kappa`$ (through $`G`$) and $`\mu`$ only, not on $`W`$ or $`b`$ ✓; that is what T9c.1 ("one event … hence for all $`W`$") and T9d.4 use ✓.

T9c.2's a.s. statement is along $`i\equiv t\ (P)`$ for each of finitely many $`t`$, so a finite intersection of probability-one events ✓. T9d.4's $`\Omega_0`$ is the event of L9.3 plus the deterministic Weyl limits ✓ and, again, does not involve $`W`$ or $`b`$ ✓.

### 2.2 T9b.1 (span criterion)

$`c_m(n)=\sum_{(a,m)=1}e^{2\pi i(a/m)n}`$: a sum of characters of $`\mathbb Z`$ indexed by $`x=a/m\in\mathbb Q/\mathbb Z`$ of exact denominator $`m`$. The index sets $`X_m`$ are nonempty ($`\varphi(m)\ge1`$) and pairwise disjoint,
characters with distinct $`x`$ are linearly independent (Dedekind–Artin; orthogonality on $`\mathbb Z/L`$) ⇒ $`\{K_m\}`$ independent, *for any set of orders including orders $`>P`$* ✓.
"If": T9a.1. "Only if": a representation $`D_P=\sum_{m\in M}\beta_mK_m`$ and T9a.1's are two representations in the independent family $`\{K_m:m\in M\cup\mathrm{div}(P)\}`$, hence equal; the coefficient of $`K_m`$ for $`m\mid P`$, $`m\notin M`$ is $`\varphi(m)/P>0`$ on one side and $`0`$ on the other ✓.
$`P=1`$: $`D_1\equiv1=K_1`$, divisor set $`\{1\}`$ ✓. Orders $`m\in M`$, $`m\nmid P`$ get coefficient $`0`$ ✓.
Independent test: 216 random banks (96 with the criterion false, 120 true), including $`P=1`$ and orders $`>P`$, own exact rank over $`\mathbb Q`$ on $`\mathbb Z/L`$: **0 mismatches**, all families independent [`recheck.py` C].
The "largest divisor-closed subset of $`M`$" consequence and the list "prime powers $`\le16`$" for $`B_{16}\cup\{1\}`$ are right (divisors of a prime power are prime powers).

### 2.3 T9b.2 and the claim that a CRT bank "cannot" count composite $`P`$

Numbers: $`b=\tfrac12(K_3+K_4)`$ on residues $`0..11`$ = $`1,-\tfrac14,-\tfrac34,\tfrac12,\tfrac14,-\tfrac14,0,-\tfrac14,\tfrac14,\tfrac12,-\tfrac34,-\tfrac14`$ ✓; $`\sum_{s\ne0}|b(s)|=4`$ ✓;
adversarial motif $`(0,0,0,1,1,0,0,0,1,1,0,0)`$ puts the competitor on the four positive off-comb residues and the truth on the negative ones: bracket $`1-4=-3`$, slope $`\tfrac{p-\nu}{12}(-3)=-\tfrac18`$ ✓.
**Issue 1:** the draft says "By T9c.3" (l. 121); the slope formula is T9c.5 (T9c.3 is the expectation bound).
**Issue 2 (strengthening, not an error):** the draft's conclusion is about *exact* counting. I computed the best achievable dominance inside the whole fold span $`\{K_1,K_2,K_3,K_4\}`$ (the orders of $`B_{16}\cup\{1\}`$ dividing 12) by L1 vertex enumeration:
$`\min\sum_{s\ne0}|f(s)|`$ subject to $`f(0)=1`$ is $`2.0`$ (attained at $`\tfrac14K_1+\tfrac14K_2+\tfrac12K_4`$), so **no** coefficient choice makes the kernel dominant; margin $`\le-1`$ [`recheck.py` F].
The CRT choice $`S=\{3,4\}`$ is the best convex mixture of $`K_3,K_4`$ ($`a=\tfrac12`$, sum $`4`$) [H] but not the best element of the span; NOTES.md item 7 ("the best CRT choice") should say "best among two-order mixtures".
Conclusion of T9b.2 stands, and is stronger than stated.

### 2.4 T9a

(1) ✓ (grouping frequencies $`h=aP/m`$). (2) ✓. (3) $`\mathrm{NLL}\le\log(1+e^{B^+}\sum\rho^{D_b})\le e^{B^+/2}\sum\rho^{D_b/2}`$ and $`\mathbb E\rho^{D_b/2}=p\rho^{-1/2}+\nu\rho^{1/2}+(V-2)\nu=2\sqrt{p\nu}+(V-2)\nu=\chi`$ ✓ (the pathwise bound is why data-dependent $`b`$ is harmless ✓).
(4) concavity step ✓; strong law along the $`n_i`$ votes ✓. Rates: $`(p-\nu)\log\rho=1.4166`$ ($`V=16`$), $`\log2`$ ($`V=3`$) ✓; $`-\log\chi=0.36356,\,0.18232`$ ✓.
Note (not an error): open item (ii) ("tightness of $`\chi^n`$") is easy to settle at the level of the *exponent*. The Chernoff tilt of $`\mathbb E\rho^{sD}=p\rho^{-s}+\nu\rho^{s}+(V-2)\nu`$ is minimized exactly at $`s=\tfrac12`$ (derivative $`-p\rho^{-s}+\nu\rho^{s}=0\iff\rho^{2s}=\rho`$),
so a standard Cramér/Bahadur–Rao lower bound on $`\Pr(D_b\ge0)`$ with $`\mathrm{NLL}\ge\log2\cdot\mathbf 1[D_b\ge0]`$ gives $`\mathbb E\,\mathrm{NLL}\ge c\,\chi^{m}/\sqrt m`$ (sketch; I did not write out the local limit step). The draft's own table shows $`\mathbb E\,\mathrm{NLL}/\chi^m`$ between $`2.8`$ and $`3.2`$ for $`V=16`$, consistent with the exponent being tight.

### 2.5 T9b.3 ($`D_P\in\mathcal C\iff\varphi(P)\le2`$)

$`D_P`$ has spectrum $`\{h/P\}`$, all coefficients $`1/P`$; $`\mathcal C`$ has spectrum in $`\{0\}\cup\{\pm1/d\}`$ with symmetric coefficients, any symmetric pattern attainable. $`h/P=(h/g)/(P/g)`$ reduced, so $`h/P\equiv\pm1/d`$ iff $`d=P/g`$ and $`h\in\{g,P-g\}`$ ✓.
Units force $`\varphi(P)\le2`$; conversely if $`\varphi(P)\le2`$ every divisor $`d=P/g`$ has $`\varphi(d)\mid\varphi(P)\le2`$, so $`h/g`$ (a unit mod $`P/g`$) is $`\pm1`$ ✓ (the draft verifies this by listing the five spectra, equivalent).
Independent test: the spectral criterion equals $`\{\varphi(P)\le2\}=\{1,2,3,4,6\}`$ for all $`P\le300`$; the harmonic criterion ($`1\le a\le P/2`$: fundamental iff $`a\mid P`$) has 0 mismatches for $`P\le120`$ [D]. ✓
Consistency with the pilot: `cosine_full` contains every reduced harmonic of orders $`\le16`$, hence all harmonics $`a=0..\lfloor P/2\rfloor`$ of both $`P=12`$ and $`P=13`$, so $`D_P`$ is in its span, as the pilot's `cosine_full` success requires; `cosine_fund` has only $`a=1`$ per order.

### 2.6 T9c.5 and the role of $`V`$

With $`W=I`$: $`s_t(c,b)=\tfrac{p-\nu}{P}\big(\kappa_P(0)+\sum_{s\ne0}\kappa_P(s)\varepsilon_s\big)`$ ✓. Sufficiency is $`|\sum\kappa\varepsilon|\le\sum|\kappa|`$.
Necessity: choose $`\mu_{t-s}=b`$ if $`\kappa_P(s)>0`$ and $`\mu_{t-s}=c`$ if $`\kappa_P(s)<0`$ (arbitrary if $`0`$); the residues $`t-s`$, $`s\ne0`$, are distinct and distinct from $`t`$, so the motif is well defined.
**It needs only two symbols, so $`V\ge2`$ suffices**; no larger $`V`$ is needed for the worst-case motif to exist ✓ (the draft's $`V=3`$ brute force is not a restriction). I re-ran the brute force at $`V=2`$ ($`P=3..6`$) and $`V=3`$ ($`P=4,5`$), including equality cases and zero entries: 240 cases, 79 dominant, **0 mismatches** [E].
The criterion is however a statement about the *worst* motif. For large $`V`$ the fraction of motifs with a collision pattern adverse enough to flip a margin is small (see `CONSISTENCY.md`); T9c.5 should not be used as a predictor of typical-motif behaviour, and T9 should say so in "What this does not say".

### 2.7 T9c.2/.3, T9d

T9c.2 ✓ (both directions; the $`s_t=0`$ case ✓). T9c.3 ✓ up to the constant (§1). T9c.4 ✓, T9c.6 ✓ ($`\sum_{s\in\mathbb Z/P}K_m(s)=0`$ for $`1<m\mid P`$).
Example $`\beta_1K_1+\beta_PK_P`$: $`\kappa_P(0)=\beta_1+\beta_P`$, $`\kappa_P(s)=\beta_1-\beta_P/(P-1)`$, dominance $`\iff\beta_1>0`$, $`\beta_P>\tfrac{P-2}2\beta_1`$ ✓ (I redid both cases of the absolute value).
T9d.1: $`H_P=H`$ because $`PH=H`$ for a torus ✓; Weyl for the monothetic group generated by $`P\theta`$ ✓; the decomposition of $`D_i`$ by $`i-r`$ mod $`P`$ ✓. T9d.4 ✓. The hypothesis holds literally for float parameters (rational radians, $`\pi`$ irrational), as MATH.md T5 says.
Quantifier on $`r^*`$: it depends on $`W`$ (through the sign of $`\Delta`$) but not on $`b`$ or $`\omega`$ — consistent with the draft's "there is a class $`r^*`$" being inside the "for every $`W`$" ✓.

### 2.8 T9e

Power law ✓; $`\mathbb E\,\mathrm{KL}>0`$ because $`\varpi`$ is non-uniform with positive probability for $`m\ge1`$ ✓ (needs $`V\ge2`$); lower bound $`k\lambda\,\mathbb E(\max D)^+`$ ✓; upper bound via $`(1+\sum x)^k\ge1+\sum x^k`$ ✓; $`k\to\infty`$ limit ✓. Hand case ($`V=3,m=1,k=2`$): $`0.867563`$, $`1.041979`$, $`\mathbb E p_{\rm true}=\tfrac12,\tfrac{11}{18}`$, top probability $`\tfrac89`$ vs accuracy $`\tfrac23`$ ✓.
My own enumeration reproduces all 14 rows of the draft's table to 6 decimals and also the omitted rows [I]. **Omitted rows**: for $`V=3,k=2`$ the excess is $`0.1744,0.1325,\mathbf{0.1392},0.1181,0.1010,0.0863`$ for $`m=1..6`$, i.e. **not monotone in $`m`$**; the draft prints $`m=1,2,4,6`$ only. The draft makes no monotonicity claim, but the selection hides a feature a reader would want to know.

## 3. Numbers in `T9_DRAFT.md` vs `verification/check_t9_run2.log` and `results_t9.json`

All other quoted numbers match (rates, $`\chi`$, E-table rows, d-table rows and limit gaps, $`b`$-vector, $`\sum|b|=4`$, slopes $`-\tfrac1{12},-\tfrac1{130},0,\tfrac1{338},\tfrac1{26}`$, $`-\tfrac{19}2`$, 1093/4369/341 sequences, 84 count vectors, 156 and 240 and 48 cases, $`1.55\cdot10^{-15}`$ ≤ "1.6e-15", `E NLL` 1.960591/0.356482/0.197606 and bounds 10.43/1.693/0.465, $`1.2\cdot10^{-4}`$ vs $`1.1\cdot10^{-3}`$). Exceptions:

| Draft | Log / JSON | Comment |
|---|---|---|
| l. 47: "240 identity checks" | `results_t9.json` `c_decomposition.identity_checks` = **288** (code: 18 values of $`t`$ × 4 × $`V=4`$) | fix the count |
| l. 3: "Every number below comes from `check_t9.py`" | e.g. "0.03846" limit gap is $`(p-\nu)/13`$ rounded from $`0.04087-0.00240`$; the run log is not the source of the L9.1 count | soften to "every quoted check result" |
| l. 222: "non-comb kernels have $`|\mathrm{gap}/i|\le5\cdot10^{-4}`$ at $`i=65533`$" | the period-13 cosine row is $`0.00276`$ (the three constant-fold kernels: $`2.3,4.4,4.7\cdot10^{-4}`$) | say "the three constant-fold kernels" |
| T9d-num header "limit gap" for $`D_{13}`$ $`0.03846`$ | log columns give $`0.04087-0.00240=0.03847`$ | rounding only |

The run log (`check_t9_run2.log`) does not print the E/B tables for `a_rate_bound`; those numbers come from `results_t9.json`, which is fine, but the draft's wording says "output in `verification/check_t9_run2.log`".

## 4. Labels and the forbidden-claims list (AGENTS.md)

- **Labels.** The status table and headings use the repo labels correctly (PROVED/CLASSICAL/ASYMPTOTIC/NUMERICAL/OPEN). Two things blur them:
  (i) the Check index introduces "EXACT" and "EXACT-ENUM" as result labels; under the repo's six labels, an enumeration with floating-point logarithms is NUMERICAL, and the table values in T9e (all $`\log`$ values) are NUMERICAL even though the inequalities they illustrate are PROVED;
  (ii) T9e's status "PROVED (exact enumeration, logs in floating point)" conflates the theorem (PROVED) with its table (NUMERICAL). Split them as T9d/T9d-num already are.
- **Calibration.** T9e's table row calls the head "overconfident" (l. 12) and the hand case says "its stated confidence exceeds its accuracy" (l. 235). The mathematics is a statement about a toy law, and the draft scopes it ("meant literally for the toy law"), but the repository rule forbids calibration claims without new evidence and these are the words that will be quoted. Suggest "does not equal the posterior (excess NLL $`=\mathbb E\,\mathrm{KL}`$)" and drop "overconfident"/"confidence exceeds accuracy" from the status table.
- **Linear attention / sub-quadratic.** T9f: "that a periodic unnormalized sum can be kept as $`P`$ running counts is noted". This is a structural complexity remark (true only for exactly $`P`$-periodic kernels) placed next to a linear-attention citation. The disclaimer follows it, but the sentence is the claim. Delete it or mark it OPEN with a pointer to an experiment.
- **Novelty.** T9f: "The only specific objects here are the exact arithmetic kernel $`D_P`$ on the counts and the rate separation between T9c and T9d." PROGRAM.md says no novelty claims. Rephrase as pure attribution ("the cache-model construction is classical; the kernel is this repository's") or delete the second half.
- No speedup, state-of-the-art, natural-language or e-value claims found. `docs/CLAIM_LEDGER.md` is untouched ✓. Notation rules (orders $`m`$ vs query $`q`$; renaming B1's $`q`$ and $`\kappa`$) are respected ✓; the only collision is the letter $`K`$ for the vote count in T9b.4 ("$`K=100`$") vs $`K_m`$ and T5's dimension $`K`$.
- **PROGRAM.md headline.** "A kernel that is not exactly periodic … cannot [turn context length into evidence], because its comb-specific evidence is asymptotically residue-blind" is true only beyond the drift horizon $`1/|\delta|`$ of its near-resonant components; below it a learned-frequency kernel can look like a comb (MATH.md T5 drift remark, `CONSISTENCY.md`). The sentence should carry "asymptotically" on "cannot".

## 5. Suggested edits, in priority order

1. L9.1: add the per-component bound $`|D_i-(i+1)e_t|\le2\sum_c|\beta_c|\,\mathrm{lcm}(m_c,P)`$ (periodic kernels $`\sum_c\beta_c\chi_c`$ with $`\chi_c`$ of exact period $`m_c`$; same proof applied per component) and say that the $`\Lambda G`$ form is vacuous below $`i\sim\Lambda`$; fix "240" → 288.
2. T9c: add Corollary (random motifs): for i.i.d. uniform motifs the failure probability is controlled by collision events; state that strict dominance is the worst-motif condition and is not necessary for typical motifs. Fix the Hoeffding exponent.
3. T9d: add to the statement or remarks the pre-asymptotic horizon (window fold $`\hat\kappa_{P,N}=O(1/(N\delta))`$ per component, $`\delta`$ = distance to the nearest $`2\pi a/P`$), and soften the table row.
4. L9.2: state the character version ($`\cos(2\pi an/m)`$, $`\gcd(a,m)=1`$ folds to itself if $`m\mid P`$ and to $`0`$ otherwise).
5. T9b.2: correct the cross-reference; add that dominance is impossible in the whole span for $`B_{16}\cup\{1\}`$ at $`P=12`$ (minimum off-comb mass $`2>1`$).
6. Labels and vocabulary (§4); print all T9e rows; fix the l. 222 sentence; soften l. 3.
7. Optional: close open item (ii) for the exponent.

## 6. What I checked and found correct (no action)

Comb identity for $`P\le40`$ (own code); fold lemma for $`P\le30,m\le24`$ (own code); span criterion on random banks; $`\varphi(P)\le2`$ equivalence to $`P\le300`$; dominance equivalence incl. $`V=2`$; T9b.4 formula and all five slopes, and $`-19/2`$ ($`101`$ comb votes, $`1200`$ off-comb positions, $`0.5(101-0.1\cdot1200)`$); T9e table; hand case; $`\chi`$, rates.
