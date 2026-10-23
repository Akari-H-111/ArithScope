# T9 (draft): exact-period evidence accumulation

Status: **development draft, 2026-10-23.** Nothing here enters `docs/MATH.md` or `docs/CLAIM_LEDGER.md` until the orchestrator and the repository owner accept it. Labels follow `docs/MATH.md` (PROVED, CLASSICAL, ASYMPTOTIC, NUMERICAL, EMPIRICAL, OPEN). Every check result quoted below comes from `check_t9.py` (summary in `verification/check_t9_run3.log`, all numbers including the tables in `results_t9.json`); the check is named in brackets, e.g. [`b_bank_design`]. Check labels follow the repository: EXACT is a finite computation in `Fraction`/integer arithmetic, NUMERICAL is anything with floating point or simulation (this includes enumerations whose probabilities are exact but whose logarithms are floating point). A theorem keeps its own label (PROVED, ...) whatever the label of the check that illustrates it. No statement was checked by a proof assistant. Provenance and open points are in `NOTES.md`; the referee report and the point-by-point response are in `NOTES.md`, "Revision after audit (round 3)".

| Item | Statement | Label |
|---|---|---|
| T9a | $`D_P`$ count head + $`W=(\log\rho)I`$ gives the Bayes posterior exactly; bounded $`b`$ keeps exponential NLL decay at the Bayes a.s. rate; the exponent $`\log\chi`$ of the expected NLL is attained | PROVED |
| T9b | $`D_P\in\mathrm{span}\{K_m\}`$ iff all divisors of $`P`$ are in the bank; cosine banks; CRT banks; off-comb leakage can give a negative rate | PROVED (+ exact checks) |
| T9c | exactly periodic kernels: a.s. linear log-odds per residue class, exact exponential rate of the NLL; worst-motif criterion (T9c.5); random-motif failure probability (T9c.6) | PROVED |
| T9c.6-num | Monte Carlo failure probabilities, $`V=16`$, $`P=12,13`$ | NUMERICAL |
| T9d | non-resonant kernels are residue-blind at linear order *as $`i\to\infty`$*; no exponential rate on at least one class; closed-form window fold near a resonance | ASYMPTOTIC (closed form PROVED) |
| T9d-num | finite-$`i`$ illustration, one seed | NUMERICAL |
| T9e | exact copies make the count head differ from the posterior: expected-NLL excess $`=\mathbb E\,\mathrm{KL}>0`$ | PROVED |
| T9e-num | the table of expected NLLs (exact probabilities, floating-point logarithms) | NUMERICAL |
| T9f | what is not claimed | — |

## 0. Setting and notation

Notation follows `docs/MATH.md` §0: arithmetic indices are **orders** $`m`$; the attention query is never called $`q`$. Two letters of B1 are renamed here to avoid collisions.

- **Channel.** Symbols $`a\in\mathcal A`$, $`|\mathcal A|=V\ge2`$, noise $`0<\varepsilon<1`$. Put
  ```math
  p=1-\varepsilon+\tfrac{\varepsilon}{V},\qquad \nu=\tfrac{\varepsilon}{V},\qquad \rho=\tfrac p\nu>1,\qquad \lambda=\log\rho,\qquad p+(V-1)\nu=1 .
  ```
  $`\nu`$ is B1's per-wrong-symbol probability (called $`q`$ there; renamed because $`q`$ is the attention query). For $`V=16`$, $`\varepsilon=\tfrac12`$: $`p=\tfrac{17}{32}`$, $`\nu=\tfrac1{32}`$, $`\rho=17`$, $`p-\nu=\tfrac12`$ [`a_bayes_equality`].
  $`\chi=1-(\sqrt p-\sqrt\nu)^2=2\sqrt{p\nu}+(V-2)\nu<1`$ is B1's $`\kappa`$ (renamed because $`\kappa`$ is the positional kernel here). For $`V=16`$: $`\chi=(14+2\sqrt{17})/32=0.695194`$; for $`V=3`$, $`\varepsilon=\frac12`$: $`p=\frac23`$, $`\nu=\frac16`$, $`\rho=4`$, $`\chi=\frac56`$ [`a_rate_bound`].
- **Data (H-data).** A motif $`\mu=(\mu_r)_{r\in\mathbb Z/P}\in\mathcal A^P`$. The tokens $`X_0,X_1,\dots`$ are independent, $`\Pr(X_j=a)=\pi_{j\bmod P}(a)`$ with $`\pi_r(a)=p`$ if $`a=\mu_r`$ and $`\nu`$ otherwise. The target of a query at index $`i`$ is the clean symbol $`c_i=\mu_{i\bmod P}`$; the query has $`n_i=\lfloor i/P\rfloor+1`$ **comb votes** (positions $`j\le i`$ with $`P\mid i-j`$, its own token included). Where a Bayes posterior is mentioned, the motif entries are also i.i.d. uniform (**H-prior**), as in B1.
- **Head (H-head).** For a kernel $`\kappa:\mathbb Z\to\mathbb R`$ and a matrix $`W\in\mathbb R^{V\times V}`$,
  ```math
  E_i(a)=\sum_{0\le j\le i}\kappa(i-j)\,\mathbf 1[X_j=a],\qquad \ell_i=b_i+WE_i,\qquad \mathrm{NLL}_i=\log\Big(1+\sum_{b\ne c_i}e^{-(\ell_i(c_i)-\ell_i(b))}\Big).
  ```
  $`b_i\in\mathbb R^V`$ is "the rest of the network"; it may depend on $`X_0,\dots,X_i`$. **(H-b):** $`B:=\sup_{i,\omega,a,a'}|b_i(a)-b_i(a')|<\infty`$. By P3 this holds when the rest of the network ends in LayerNorm and a linear readout without bias ($`B\le\max G_{aa'}`$). The architectural hypothesis is that $`WE_i`$ is added to the logits *after* the final LayerNorm, so P3 does not bound it. Whether such a network is trainable is not addressed.
- **Comb and kernels.** $`D_P(n)=\sum_{m\mid P}\frac{\varphi(m)}{P}K_m(n)`$. $`K_m`$ is $`m`$-periodic; a bank $`M`$ gives kernels $`\sum_{m\in M}\beta_mK_m`$ with real $`\beta_m`$ (the *span*). For $`L\ge1`$ and an $`L`$-periodic $`\kappa`$ put $`\Lambda=\mathrm{lcm}(L,P)`$, $`G=\sup|\kappa|`$, and
  ```math
  \kappa_P(s)=\frac P\Lambda\sum_{\substack{n\in\mathbb Z/\Lambda\\ n\equiv s\,(P)}}\kappa(n)\quad(s\in\mathbb Z/P),\qquad
  e_t(a)=\frac1\Lambda\sum_{n\in\mathbb Z/\Lambda}\kappa(n)\pi_{t-n}(a)=\frac1P\sum_{s\in\mathbb Z/P}\kappa_P(s)\pi_{t-s}(a).
  ```
  $`\kappa_P`$ is the **fold** of $`\kappa`$ to modulus $`P`$ (the mean of $`\kappa`$ over each residue class mod $`P`$); the two expressions for $`e_t`$ agree by grouping $`n`$ by its class mod $`P`$.

## 1. Three lemmas on the evidence sums

**L9.1 (expected evidence; PROVED, elementary).** Let $`\kappa=\sum_{c=1}^{C}\beta_c\chi_c`$ with $`\chi_c`$ of exact period $`m_c`$ and $`G_c=\sup|\chi_c|`$ (for $`K_m`$ and $`\cos(2\pi an/m)`$, $`G_c\le1`$). Then $`\kappa`$ is $`L`$-periodic, $`L=\mathrm{lcm}_c\,m_c`$. Write $`D_i(a)=\mathbb E\,E_i(a)=\sum_{n=0}^i\kappa(n)\pi_{i-n}(a)`$. For all $`i`$ and $`a`$,
```math
\big|D_i(a)-(i+1)\,e_{i\bmod P}(a)\big|\;\le\;B_{\rm comp}:=2\sum_{c}|\beta_c|\,G_c\,\mathrm{lcm}(m_c,P).
```
For $`C=1`$, $`\chi_1=\kappa`$ this reads $`\le2\Lambda G`$ with $`\Lambda=\mathrm{lcm}(L,P)`$, $`G=\sup|\kappa|`$: the earlier form of the lemma, which is valid but, for a bank, is governed by the lcm of *all* orders. Exactly, for $`i=t+\Lambda k`$ with $`0\le t<\Lambda`$: $`D_i(a)=(i-t)\,e_t(a)+\sum_{n=0}^{t}\kappa(n)\pi_{t-n}(a)`$.

*Proof.* Put $`\Lambda_c=\mathrm{lcm}(m_c,P)`$ and $`D^{(c)}_i(a)=\sum_{n=0}^i\chi_c(n)\pi_{i-n}(a)`$, so $`D_i=\sum_c\beta_cD^{(c)}_i`$. For fixed $`i`$, $`n\mapsto\chi_c(n)\pi_{i-n}(a)`$ is $`\Lambda_c`$-periodic and its sum over one period is $`\Lambda_ce^{(c)}_{i\bmod P}(a)`$, where $`e^{(c)}_t(a)=\frac1{\Lambda_c}\sum_{n\in\mathbb Z/\Lambda_c}\chi_c(n)\pi_{t-n}(a)`$. Since $`\Lambda_c\mid\Lambda`$, the average over $`\mathbb Z/\Lambda`$ of a $`\Lambda_c`$-periodic function equals its average over $`\mathbb Z/\Lambda_c`$; hence $`e_t=\sum_c\beta_ce^{(c)}_t`$. Write $`i+1=q\Lambda_c+s`$, $`0\le s<\Lambda_c`$: the sum over $`0\le n\le i`$ is $`q\Lambda_ce^{(c)}`$ plus $`s`$ further terms, each of modulus $`\le G_c`$ (as $`\pi\le1`$), while $`(i+1)e^{(c)}-q\Lambda_ce^{(c)}=se^{(c)}`$ with $`|e^{(c)}|\le G_c`$. So $`|D^{(c)}_i-(i+1)e^{(c)}|\le2sG_c<2\Lambda_cG_c`$; multiply by $`|\beta_c|`$ and sum. The exact form follows by splitting the sum at $`n=t`$: the terms $`n=t+1,\dots,t+\Lambda k`$ are $`k`$ full periods. ∎

*Size of the constant (what the lemma says at finite $`i`$).* $`B_{\rm comp}`$ does not depend on $`i`$, so the lemma is informative only once it is small against the leading term. For the gap $`E_i(a)-E_i(b)`$ of a comb-dominant fold the leading term is $`(i+1)g`$, $`g=(p-\nu)\kappa_P(0)/P`$, and the error is at most $`2B_{\rm comp}`$; call the lemma *informative* when $`2B_{\rm comp}\le\frac12(i+1)g`$. For $`V=16`$, $`\varepsilon=\frac12`$ [`l91_component_bound`]:
- the comb $`D_{12}`$ itself: $`B_{\rm comp}=24`$ ($`26`$ for $`D_{13}`$), informative from $`i\ge2303`$ ($`2703`$);
- the default bank $`\{1,\dots,16\}`$ with every order at weight $`1`$ ($`\kappa_P(0)=6`$ at $`P=12`$, $`2`$ at $`P=13`$): $`B_{\rm comp}=1632`$ ($`3224`$); informative from $`i\ge26\,111`$ ($`167\,647`$). The $`\Lambda`$-form is $`2\Lambda G=2.3\cdot10^{7}`$ here and becomes informative only at $`i\gtrsim3.7\cdot10^{8}`$ ($`P=12`$).

So for the default bank **the lemma has no finite-$`i`$ content below $`i\sim10^{3}`$–$`10^{5}`$**, and none at the lengths of the development pilot ($`i\le8192`$) when many orders carry weight. This is a statement about the *constant*: the true deviations are far smaller (at most $`6.2`$ over the random $`K_m`$ cases and $`7.6`$ over the random cosine cases of the check, against bounds $`10^2`$–$`10^3`$; $`2.3`$ over $`i\le8192`$ for the audit's example with $`P=12`$, random coefficients on $`\{1,\dots,16\}`$, where the $`\Lambda`$-form gives $`1.2\cdot10^{7}`$ and $`B_{\rm comp}=751`$, see `audit/recheck.log` [K]). T9c.1, T9c.2 and T9d are limit statements and do not depend on the constant; the finite-$`i`$ constants $`C_0`$ in T9c.3 inherit it.
*Check:* the exact form, 288 identity checks (18 values of $`t`$, 4 values of $`k`$, 4 symbols), $`P=12`$, kernel a mixture of $`K_1,K_4,K_5,K_8,K_{12}`$ ($`L=120`$) [`c_decomposition`]; the inequality $`B_{\rm comp}`$ for 12 random rational mixtures of four $`K_m`$ (exact, $`P=12,13`$, $`i\le460`$) and 8 random cosine mixtures (floating point) [`l91_component_bound`].

**L9.2 (fold of a Ramanujan kernel; PROVED, CLASSICAL core).** For every order $`m`$ and period $`P`$,
```math
(K_m)_P(s)=K_m(s)\ \text{ if } m\mid P,\qquad (K_m)_P(s)=0\ \text{ if } m\nmid P .
```
Hence for $`\kappa=\sum_{m\in M}\beta_mK_m`$: $`\kappa_P=\sum_{m\in M,\,m\mid P}\beta_mK_m|_{\mathbb Z/P}`$. In particular $`(D_P)_P=\delta_0`$ (the indicator of residue $`0`$), and **only the part of $`\kappa`$ that is $`P`$-periodic contributes to the linear rate**.

*Proof.* If $`m\mid P`$, $`K_m(n)`$ depends only on $`n\bmod m`$, which is determined by $`n\bmod P`$, so $`K_m`$ is constant on classes mod $`P`$. If $`m\nmid P`$, put $`g=\gcd(m,P)<m`$. As $`n`$ runs over a class $`s+P\mathbb Z`$, $`n\bmod m`$ runs uniformly over the coset $`s+g\mathbb Z/m\mathbb Z`$, so the fold is the mean of $`c_m(n)/\varphi(m)`$ over that coset. With $`n=s+gu`$, $`u\in\mathbb Z/(m/g)`$:
```math
\sum_u c_m(s+gu)=\sum_{a\in(\mathbb Z/m)^\times}e^{2\pi ias/m}\sum_{u\in\mathbb Z/(m/g)}e^{2\pi iau/(m/g)}=\frac mg\sum_{a:\ (m/g)\mid a}e^{2\pi ias/m}.
```
A unit $`a`$ mod $`m`$ divisible by $`m/g`$ forces $`m/g=1`$, i.e. $`g=m`$, which was excluded; so the sum is empty and the fold is $`0`$. ∎

**L9.2′ (character and cosine version; PROVED, CLASSICAL core).** Let $`\gcd(a,m)=1`$ and $`\chi(n)=e^{2\pi ian/m}`$ or $`\chi(n)=\cos(2\pi an/m)`$ (for $`m=1`$: $`\chi\equiv1`$). Then
```math
\chi_P(s)=\chi(s)\ \text{ if } m\mid P,\qquad \chi_P(s)=0\ \text{ if } m\nmid P .
```
So for every kernel $`\kappa=\sum_c\beta_c\chi_c`$ built from such components (in particular all `cosine_fund`/`cosine_full` banks with reduced harmonics), $`\kappa_P=\sum_{c:\,m_c\mid P}\beta_c\chi_c|_{\mathbb Z/P}`$; L9.2 is the case $`K_m=\varphi(m)^{-1}\sum_{(a,m)=1}e^{2\pi ian/m}`$ by linearity.

*Proof.* The class $`s+P\mathbb Z`$ mod $`\Lambda=\mathrm{lcm}(m,P)`$ is $`\{s+Pu:u\in\mathbb Z/(\Lambda/P)\}`$ and the fold is the mean of $`\chi`$ over it. For the exponential, $`e^{2\pi ia(s+Pu)/m}=e^{2\pi ias/m}\,\omega^u`$ with $`\omega=e^{2\pi iaP/m}`$. If $`m\mid P`$, $`\omega=1`$ and the mean is $`e^{2\pi ias/m}`$. If $`m\nmid P`$, then $`aP/m\notin\mathbb Z`$ (as $`\gcd(a,m)=1`$), so $`\omega\ne1`$, $`\omega^{\Lambda/P}=e^{2\pi ia\Lambda/m}=1`$ and the geometric sum over $`u\in\mathbb Z/(\Lambda/P)`$ vanishes. The cosine is $`\frac12(\chi_{a}+\chi_{-a})`$ and $`-a`$ is again a unit. ∎
*Check:* floating point, all $`P\le20`$, $`m\le24`$, all units $`a`$: maximal deviation $`4.9\cdot10^{-14}`$ over 37 800 values [`l92_character_fold`, NUMERICAL]. The scope of the original L9.2 proof is the Ramanujan kernels; the cosine banks of the pilot use L9.2′.
*Check:* exact for all $`P\le24`$, $`m\le16`$; $`D_{12}`$ folds to $`\delta_0`$ at modulus 12, is unchanged at modulus 24, and folds to the constant $`\frac1{12}`$ at modulus 5 [`c_fold_projection`]. (The statement is slightly stronger than "components of period coprime to $`P`$ average out": the right condition is $`m\nmid P`$, e.g. $`K_4`$ folds to $`0`$ at $`P=2`$.)

**L9.3 (noise term; PROVED, Hoeffding + Borel–Cantelli).** Let $`\kappa`$ be any bounded kernel, $`G=\sup|\kappa|`$. Almost surely there is $`I(\omega)`$ with
```math
\big|E_i(a)-D_i(a)\big|\le G\sqrt{\tfrac32\,(i+1)\log(i+2)}\qquad\text{for all } i\ge I(\omega)\text{ and all } a\in\mathcal A,
```
so $`E_i(a)-D_i(a)=O(\sqrt{i\log i})`$ a.s.

*Proof.* For fixed $`i,a`$, $`E_i(a)-D_i(a)=\sum_{j\le i}\kappa(i-j)(\mathbf 1[X_j=a]-\pi_j(a))`$ is a sum of $`i+1`$ independent centred terms, the $`j`$-th lying in an interval of length $`|\kappa(i-j)|\le G`$. Hoeffding's inequality gives $`\Pr(|\cdot|\ge x)\le2\exp(-2x^2/((i+1)G^2))`$. With $`x=x_i=G\sqrt{\frac32(i+1)\log(i+2)}`$ this is $`2(i+2)^{-3}`$. Summing the bounds over $`a`$ and $`i`$ gives a finite total ($`2V\sum_i(i+2)^{-3}<\infty`$), and Borel–Cantelli applies. (For different $`i`$ the sums are different weighted sums of the same variables; only the individual probabilities enter.) ∎
*Remark.* The same proof with $`\sum_j\kappa(i-j)^2`$ in place of $`(i+1)G^2`$ gives the threshold $`x_i=\sqrt{\tfrac32\log(i+2)\sum_{n\le i}\kappa(n)^2}`$, which is never larger.

*Consequence (L9.1 + L9.3).* For $`L`$-periodic $`\kappa`$, almost surely $`E_i(a)=(i+1)e_{i\bmod P}(a)+O(\sqrt{i\log i})`$.

## 2. T9a: construction and bounded offset

**T9a (PROVED; the identity in (i) is CLASSICAL, DeltaNet study D4).** Assume H-data, H-head.

1. **Comb identity.** $`D_P(n)=\frac1P\sum_{m\mid P}c_m(n)=\mathbf 1[P\mid n]`$, with weights $`\varphi(m)/P`$ that are positive and sum to $`1`$.
2. **Exact Bayes.** Take $`\kappa=D_P`$, $`W=\lambda I`$, $`b=0`$. Then $`E_i(a)=N_i(a):=\#\{j\le i:\ P\mid i-j,\ X_j=a\}`$ and $`\mathrm{softmax}(\ell_i)_a=\rho^{N_i(a)}/\sum_{a'}\rho^{N_i(a')}`$. Under H-prior this is exactly $`\Pr(\mu_{i\bmod P}=a\mid X_0,\dots,X_i)`$, for every $`i`$ and every realization.
3. **Bounded offset, expectation.** Let $`B^+:=\sup_{i,\omega,b\ne c_i}(b_i(b)-b_i(c_i))^+`$. With $`\kappa=D_P`$, $`W=\lambda I`$ and any $`b`$ obeying H-b (it may depend on the data),
   ```math
   \mathbb E\big[\mathrm{NLL}_i\big]\;\le\;(V-1)\,e^{B^+/2}\,\chi^{\,n_i}.
   ```
4. **Bounded offset, sandwich and a.s. rate.** Pointwise $`e^{-B}\,\mathrm{NLL}^{\rm Bayes}_i\le\mathrm{NLL}_i\le e^{B}\,\mathrm{NLL}^{\rm Bayes}_i`$, and almost surely
   ```math
   \lim_{i\to\infty}\frac1{n_i}\log\mathrm{NLL}_i=-(p-\nu)\log\rho ,
   ```
   the same limit as for $`b=0`$ (the Bayes NLL). With $`n_i\sim i/P`$ the rate per index is $`(p-\nu)\log\rho/P`$.
5. **Exponent of the expectation (tightness of item 3).** With $`\kappa=D_P`$, $`W=\lambda I`$ and any $`b`$ obeying H-b, put $`n=n_i`$ and $`s_Q^2=2\sqrt{p\nu}/\chi`$. Then
   ```math
   \mathbb E\big[\mathrm{NLL}_i\big]\;\ge\;\tfrac14\log\!\big(1+e^{-B}\big)\,\chi^{\,n}\,\rho^{-\sqrt{2ns_Q^2}/2},
   \qquad\text{hence}\qquad \lim_{n_i\to\infty}\frac1{n_i}\log\mathbb E\big[\mathrm{NLL}_i\big]=\log\chi .
   ```
   So the exponent $`-\log\chi`$ of item 3 cannot be improved (only the prefactor of $`\chi^{n}`$ is not claimed to be sharp).

*Proof.*
1. Group the frequencies $`h\in\mathbb Z/P`$ of $`\sum_{h}e^{2\pi ihn/P}`$ by $`m=P/\gcd(h,P)`$; the $`h`$ with a given $`m`$ are $`a P/m`$ with $`a`$ a unit mod $`m`$, so they sum to $`c_m(n)`$. The full sum is $`P\cdot\mathbf 1[P\mid n]`$ (orthogonality), and $`c_m=\varphi(m)K_m`$ with $`\sum_{m\mid P}\varphi(m)=P`$.
2. For $`j\le i`$, $`D_P(i-j)=\mathbf 1[P\mid i-j]`$, so $`E_i(a)=N_i(a)`$. For the posterior: given $`\mu`$ the tokens are independent, and the motif entries are independent a priori, so $`\Pr(\mu_r=a\mid X_{0..i})\propto\prod_{j\le i,\,j\equiv r}(p\ \text{if}\ X_j=a\ \text{else}\ \nu)=\nu^{n_i}\rho^{N_i(a)}`$ for $`r=i\bmod P`$; the other residues contribute a factor independent of $`a`$. Normalizing gives the softmax.
3. Let $`D_b=N_i(b)-N_i(c_i)`$. The logit gap is $`\ell_i(b)-\ell_i(c_i)=\lambda D_b+(b_i(b)-b_i(c_i))\le\lambda D_b+B^+`$. With $`\log(1+x)\le\sqrt x`$ and $`\sqrt{\sum y_b}\le\sum\sqrt{y_b}`$ (as in B1), $`\mathrm{NLL}_i\le e^{B^+/2}\sum_{b\ne c_i}\rho^{D_b/2}`$ pathwise, and $`\mathbb E\rho^{D_b/2}=\chi^{n_i}`$ by B1 step 4 (the votes are $`n_i`$ independent observations of the true residue). Pathwise validity means the data-dependence of $`b_i`$ is harmless.
4. Let $`x=\sum_{b\ne c_i}\rho^{D_b}`$ and $`x'=\sum e^{\lambda D_b+\beta_b}`$ with $`|\beta_b|\le B`$, so $`e^{-B}x\le x'\le e^Bx`$. The function $`f(y)=\log(1+y)`$ is concave with $`f(0)=0`$, hence $`f(sy)\le sf(y)`$ for $`s\ge1`$ and $`f(sy)\ge sf(y)`$ for $`0<s\le1`$; so $`e^{-B}f(x)\le f(x')\le e^Bf(x)`$. For the limit: by the strong law along the votes at residue $`i\bmod P`$, $`N_i(c_i)/n_i\to p`$ and $`N_i(b)/n_i\to\nu`$ a.s., so the margin $`\lambda(N_i(c_i)-N_i(b))`$ is $`\lambda(p-\nu)n_i(1+o(1))`$ for every $`b\ne c_i`$. Then $`x\to0`$ and $`f(x)\in[x/2,x]`$ eventually, so $`\frac1{n_i}\log\mathrm{NLL}^{\rm Bayes}_i\to-\lambda(p-\nu)`$; the sandwich transfers this to $`b`$.
5. Fix $`b_0\ne c_i`$ and put $`S=D_{b_0}=N_i(b_0)-N_i(c_i)=\sum_{u=1}^nZ_u`$, where $`Z_u`$ are i.i.d. with $`\Pr(Z=+1)=\nu`$, $`\Pr(Z=-1)=p`$, $`\Pr(Z=0)=(V-2)\nu`$ (the $`n`$ votes). Pathwise $`\mathrm{NLL}_i\ge\log(1+e^{\lambda S+b_i(b_0)-b_i(c_i)})\ge\log(1+e^{-B})\,\mathbf 1[S\ge0]`$, so it suffices to bound $`\Pr(S\ge0)`$ from below. The moment generating function $`\mathbb Ee^{\theta Z}=\nu e^{\theta}+pe^{-\theta}+(V-2)\nu`$ is minimized at $`e^{\theta_*}=\sqrt\rho`$ (derivative $`\nu e^\theta-pe^{-\theta}=0`$), with value $`2\sqrt{p\nu}+(V-2)\nu=\chi`$. Define the tilted law $`Q(Z=z)=\Pr(Z=z)e^{\theta_*z}/\chi`$: $`Q(\pm1)=\sqrt{p\nu}/\chi`$, $`Q(0)=(V-2)\nu/\chi`$. Under $`Q`$ the step is symmetric, centred, with variance $`s_Q^2`$, and $`\Pr(S=s)=\chi^ne^{-\theta_*s}Q(S=s)`$. For $`x>0`$, $`\Pr(S\ge0)\ge\chi^ne^{-\theta_*x}\,Q(0\le S\le x)`$. By symmetry $`Q(0\le S\le x)\ge\frac12Q(|S|\le x)`$, and by Chebyshev $`Q(|S|\le x)\ge1-ns_Q^2/x^2`$. Take $`x=\sqrt{2ns_Q^2}`$: $`Q(0\le S\le x)\ge\frac14`$ and $`e^{-\theta_*x}=\rho^{-x/2}`$. Hence $`\mathbb E\,\mathrm{NLL}_i\ge\frac14\log(1+e^{-B})\chi^n\rho^{-\sqrt{2ns_Q^2}/2}`$, and $`\frac1n\log`$ of the right side tends to $`\log\chi`$ because $`\sqrt n/n\to0`$. The matching upper bound is item 3. (The expectation is over everything, including data-dependent $`b_i`$: the lower bound holds pathwise.) ∎

*Checks.* The identity $`D_P=\sum_{m\mid P}\frac{\varphi(m)}{P}K_m=\mathbf 1[P\mid\cdot]`$ holds exactly for $`P=2,\dots,32`$, $`|n|\le3P`$, with Ramanujan kernels from `arithscope/arith.py` [`a_identity`]. Exact equality of $`\mathrm{softmax}(\log\rho\cdot\text{counts})`$ and the product-likelihood posterior, as `Fraction`s, on all sequences of length $`\le6`$ ($`V=3`$, $`\varepsilon=\frac12`$, 1093 sequences, 84 count vectors), $`\le4`$ ($`V=4`$, $`\varepsilon=\frac13`$) and $`\le3`$ ($`V=16`$, $`\varepsilon=\frac12`$, 4369 sequences); the float softmax differs by $`\le1.6\cdot10^{-15}`$ [`a_bayes_equality`, the float comparison is NUMERICAL]. The head built from the exact kernel table equals the comb count, $`P\in\{7,12,13\}`$, $`V=5`$, 90 positions [`a_head_comb_counts`]. The bound in item 3 holds in an enumeration with exact probabilities and floating-point logarithms (NUMERICAL) for $`B^+\in\{0,1,3\}`$ with $`b_{\rm true}=0,b_{\rm wrong}=B^+`$, $`m=1..6`$ ($`V=16`$) and $`m=1..8`$ ($`V=3`$) [`a_rate_bound`]. Examples at $`B^+=0`$: $`V=16`$, $`m=1`$: $`\mathbb E\,\mathrm{NLL}=1.960591`$ (bound $`10.43`$); $`m=6`$: $`0.356482`$ (bound $`1.693`$). $`V=3`$, $`m=8`$: $`0.197606`$ (bound $`0.465`$).

*Rates.* A.s. rate per comb vote, $`(p-\nu)\log\rho`$: $`1.416607`$ for $`V=16`$, $`0.693147=\log2`$ for $`V=3`$. Rate of the *bound* on the expectation, $`-\log\chi`$: $`0.363564`$ and $`0.182322`$ [`a_rate_bound`]. These differ because the expectation is dominated by rare plurality-wrong events. Item 5 shows that $`-\log\chi`$ is nevertheless the exact exponent of the expectation; only the prefactor of $`\chi^{n}`$ is left open. Numerically, with $`S_n`$ the vote walk of the proof, $`\sqrt n\,\Pr(S_n\ge0)/\chi^n`$ is $`0.818,0.843,0.853,0.859,0.862,0.863,0.864`$ for $`n=10,25,50,100,200,400,800`$ ($`V=16`$) and $`0.777,\dots,0.890`$ ($`V=3`$), the behaviour a local limit theorem would give (not proved here) [`a_tightness`, NUMERICAL]; the chain of item 5, including the exact tilting identity at $`n=30`$ and $`\mathbb E\,\mathrm{NLL}\ge\log2\cdot\Pr(S_m\ge0)`$ for $`m\le6`$, holds in that check.

*What this does not say.* It does not say a trained model reaches this; $`\rho`$ (i.e. $`\varepsilon`$) and $`P`$ are assumed known to the construction; independence of the votes is essential (T9e); H-b is a hypothesis on the rest of the network, not a consequence of the head; nothing is said about deeper interaction between $`E_i`$ and $`b_i`$ beyond boundedness. It does not contradict P3/C8: P3 bounds only $`b`$, and the point is that an unnormalized term outside the LayerNorm can carry the unbounded evidence.

## 3. T9b: expressibility

**T9b.1 (span criterion; PROVED).** Let $`M`$ be a finite set of orders and $`P\ge1`$. As functions on $`\mathbb Z`$ (equivalently on $`\mathbb Z/L`$, $`L=\mathrm{lcm}(M\cup\{P\})`$),
```math
D_P\in\mathrm{span}_{\mathbb R}\{K_m:m\in M\}\iff\text{every divisor of }P\text{ lies in }M .
```
The kernels $`K_m`$, $`m\in M`$, are linearly independent, so the coefficients are unique: $`\varphi(m)/P`$ for $`m\mid P`$ and $`0`$ otherwise.

*Proof.* $`c_m(n)=\sum_{a\in(\mathbb Z/m)^\times}e^{2\pi i(a/m)n}`$ is a sum of characters $`n\mapsto e^{2\pi ixn}`$ with $`x\in\mathbb Q/\mathbb Z`$ of exact denominator $`m`$. Distinct orders have disjoint sets of such $`x`$, each set is nonempty, and distinct characters of $`\mathbb Z`$ are linearly independent (Dedekind–Artin; on $`\mathbb Z/L`$: orthogonality). So the $`c_m`$, hence the $`K_m`$, are linearly independent. "If": T9a.1. "Only if": a representation $`D_P=\sum_{m\in M}\beta_mK_m`$ is, by independence of $`\{K_m:m\mid P\text{ or }m\in M\}`$, equal to the representation of T9a.1, so $`\beta_m=\varphi(m)/P>0`$ for every $`m\mid P`$, forcing $`m\in M`$. ∎

*Consequences.*
- $`\{P:D_P\in\mathrm{span}\}=\{P:\text{all divisors of }P\text{ in }M\}`$, the elements of the largest divisor-closed subset of $`M`$. For the default bank $`\{1,\dots,16\}`$ (divisor-closed) this is $`P\le16`$: $`P=12`$ (divisors $`1,2,3,4,6,12`$) and $`P=13`$ ($`1,13`$) and $`P=16`$ are in the span; $`P=17`$ and $`P=24`$ are not [`b_ramanujan_span`; coefficients $`\varphi(d)/12=\frac1{12},\frac1{12},\frac16,\frac16,\frac16,\frac13`$ for $`d=1,2,3,4,6,12`$].
- $`D_{13}=\frac1{13}K_1+\frac{12}{13}K_{13}`$ needs the **constant** $`K_1`$. The bank $`B_{16}`$ of `docs/MATH.md` (10 orders, no order 1) does not contain $`D_{13}`$ in its span; with order 1 added, it does [`b_ramanujan_span`]. This is the constant coordinate of DeltaNet study D4.
- **Softmax-simplex parametrization.** If the evidence kernel reuses the bias parametrization of §1 of `docs/MATH.md`, $`\gamma\sum\alpha_mK_m`$ with $`\alpha=\mathrm{softmax}(\beta)`$, then all weights are strictly positive, so by independence $`D_P`$ is reached exactly only when the bank *equals* the divisor set of $`P`$ (and $`\gamma=1`$, $`\alpha_m=\varphi(m)/P`$). In the default bank it is only approximated. Which parametrization an evidence head uses is a design choice not made here.
- For a finite context $`N<L`$ the "if" direction persists but independence (hence "only if") is not claimed.

*Sweep.* Four banks (default; $`B_{16}`$ with order 1; $`B_{16}`$ without order 1; divisors of 12), $`P=2,\dots,40`$: the criterion holds in all 156 cases and all banks are linearly independent [`b_ramanujan_span`].

**T9b.2 (bank design: counting needs a divisor-closed bank, selection only the lcm; PROVED, exact check).** Let $`M=B_{16}\cup\{1\}=\{1,2,3,4,5,7,8,9,11,13,16\}`$.
- *Counting.* $`D_{12}\notin\mathrm{span}\{K_m:m\in M\}`$, because the divisors $`6`$ and $`12`$ of $`12`$ are missing [`b_bank_design`, `b_ramanujan_span`]. The periods $`P\in\{2,\dots,40\}`$ with $`D_P`$ in the span are exactly the prime powers $`\le16`$: $`2,3,4,5,7,8,9,11,13,16`$ (every divisor of a prime power is a prime power; the divisor-closed part of $`M`$) [`b_ramanujan_span`].
- *Selection (T4).* The same bank selects, as strict softmax argmax, the comb $`\mathrm{lcm}(S)\mathbb Z`$ for every $`S\subseteq M`$. Of the periods up to 40 these include $`6,10,12,14,15,18,20,21,22,24,26,28,30,33,35,36,39,40`$, none of which is countable exactly [`b_ramanujan_span`, "selectable but not countable"].
- *Why it matters for sums.* For $`S=\{3,4\}`$: $`b=\tfrac12(K_3+K_4)`$ has $`b(n)=1`$ iff $`12\mid n`$, but on residues $`0,\dots,11`$ it equals $`1,-\frac14,-\frac34,\frac12,\frac14,-\frac14,0,-\frac14,\frac14,\frac12,-\frac34,-\frac14`$, with $`\sum_{s\ne0}|b(s)|=4`$ [`b_bank_design`]. A softmax with a large gain suppresses all off-comb mass; an unnormalized sum counts it. By T9c.5 (worst-motif criterion; the per-index slope formula, not an expectation bound), with $`V=3`$, $`\varepsilon=\frac12`$, $`W=I`$, the motif $`(0,0,0,1,1,0,0,0,1,1,0,0)`$ and query residue $`0`$ (true symbol $`0`$, competitor $`1`$) has slope $`\frac{p-\nu}{12}(1-4)=-\frac18`$ per index, in units of $`\lambda`$ (the log-odds margin of the truth is $`-\frac{\lambda}8\,i+o(i)`$ for $`W=\lambda I`$) [`b_bank_design`]: the wrong symbol wins and the NLL grows linearly.

- *No dominant kernel in the whole span.* The orders of $`M`$ that divide $`12`$ are $`\{1,2,3,4\}`$. Over all $`f=\sum_{d\in\{1,2,3,4\}}\beta_dK_d|_{\mathbb Z/12}`$ with $`f(0)=\sum\beta_d=1`$, the minimum of $`\sum_{s\ne0}|f(s)|`$ is exactly $`2`$ (attained at $`\beta=(\frac14,\frac14,0,\frac12)`$), so the best strict-dominance margin of T9c.5 is $`1-2=-1<0`$: **no** coefficient choice makes the kernel dominant, not only the CRT choice. Proof: $`f\mapsto\sum_{s\ne0}|f(s)|`$ is convex and piecewise linear on the affine space $`\{\sum\beta_d=1\}`$, whose arrangement $`\{f(s)=0\}`$ has vertices because $`K_1,\dots,K_4`$ are independent on $`\mathbb Z/12`$ (rank $`4`$); the minimum is attained at a vertex (three of the eleven $`f(s)`$ vanish), and the finite enumeration of all vertices in `Fraction` arithmetic gives $`2`$. The best convex mixture $`aK_3+(1-a)K_4`$ is $`a=\frac12`$ with off-comb mass $`4`$ (piecewise linear in $`a`$; exact evaluation at all breakpoints), so the CRT choice is the best *two-order mixture*, not the best element of the span. With all six divisors of $`12`$ the minimum is $`0`$ ($`D_{12}`$) [`b_span_dominance`].

*What this does not say.* It does not say CRT banks are bad for softmax attention (T4 stands). It says a bank that is only lcm-complete cannot be used as an exact counter for a composite $`P`$. "Dominant" is the worst-motif criterion: a non-dominant kernel can still have positive margins on almost every random motif (T9c.6).

**T9b.3 (cosine banks; PROVED, exact check).** Let $`\mathcal C=\mathrm{span}\{1,\ \cos(2\pi n/d):\ d\ge2\}`$ (fundamental cosines of *any* orders). Then for $`P\ge1`$,
```math
D_P\in\mathcal C\iff\varphi(P)\le2\iff P\in\{1,2,3,4,6\}.
```
In particular $`D_P\notin\mathrm{span}\{1,\cos(2\pi n/P)\}`$ for every prime $`P\ge5`$, and the matched bank $`\{1\}\cup\{\cos(2\pi n/d):d\mid P,\ d>1\}`$ contains $`D_P`$ only for $`P\in\{1,2,3,4,6\}`$ (the same set as C3.1). For $`P=12`$: $`D_{12}\notin\mathrm{span}\{\cos(2\pi n/d):d\mid12\}`$, because the harmonic $`a=5`$ is missing; but $`D_{12}\in\mathrm{span}\{\cos(2\pi an/12):a=0,\dots,6\}`$.

*Proof.* $`D_P=\frac1P\sum_{h\in\mathbb Z/P}e^{2\pi ihn/P}`$ has spectrum $`\{h/P\}`$, every coefficient $`\ne0`$. Elements of $`\mathcal C`$ have spectrum in $`\{0\}\cup\{\pm1/d\}`$ and any such pattern of symmetric coefficients is attainable (the coefficient of $`\cos(2\pi n/d)`$ is the coefficient at $`\pm1/d`$). By independence of characters, $`D_P\in\mathcal C`$ iff for each $`h\in\{1,\dots,P-1\}$, $`h/P\equiv\pm1/d\pmod1`$ for some $`d\ge2`$. Let $`g=\gcd(h,P)`$; $`h/P=(h/g)/(P/g)`$ is reduced with denominator $`P/g`$, so $`h/P\equiv\pm1/d`$ iff $`d=P/g`$ and $`h/g\equiv\pm1\pmod{P/g}`$, i.e. $`h\in\{g,\,P-g\}`$.
- *Harmonic criterion.* For $`1\le a\le\lfloor P/2\rfloor`$, $`\cos(2\pi an/P)`$ is a fundamental cosine of some order iff $`a\mid P`$. Indeed with $`g=\gcd(a,P)`$ we need $`a\in\{g,P-g\}`$; $`a=P-g\le P/2`$ forces $`P/g\le2`$, hence $`P=2g`$, $`a=g`$ again; so $`a=g`$, i.e. $`a\mid P`$; conversely $`a\mid P`$ gives $`\cos(2\pi n/(P/a))`$. (The constant, $`a=0`$, is always available.)
- *Units.* Taking $`h`$ coprime to $`P`$ ($`g=1`$) requires $`h\in\{1,P-1\}`$. All units mod $`P`$ lie in $`\{\pm1\}`$ iff $`\varphi(P)\le2`$. Now $`\varphi(P)\le2`$ means $`P\in\{1,2,3,4,6\}`$: each prime-power factor $`p^e`$ has $`\varphi(p^e)\le2`$, so $`p^e\in\{2,3,4\}`$, and the coprime products allowed are $`1,2,3,4`$ and $`2\cdot3=6`$ ($`4\cdot3=12`$ has $`\varphi=4`$). Conversely, for these $`P`$ the spectra are $`\{0,\frac12\}`$, $`\{0,\pm\frac13\}`$, $`\{0,\pm\frac14,\frac12\}`$, $`\{0,\pm\frac16,\pm\frac13,\frac12\}`$, all of the required form with $`d\mid P`$. For $`P=1`$, $`D_1\equiv1`$ is the constant.
- *Prime $`P\ge5`$:* $`h=2`$ is a unit in $`[2,P-2]`$. *$`P=12`$:* $`D_{12}=\frac1{12}+\frac1{12}\cos(\pi n)+\frac16\sum_{a=1}^5\cos(2\pi an/12)`$; $`a=1,2,3,4,6`$ divide 12 (orders $`12,6,4,3,2`$), $`a=5`$ does not. ∎

*Checks.* Exact rank over $`\mathbb Q`$ of exact Fourier-coefficient vectors: prime $`P=5,\dots,31`$ not in span; $`P=12`$ harmonics $`a=0,\dots,6` $: in span $`[\text{T},\text{T},\text{T},\text{T},\text{T},\textbf{F},\text{T}]`$, missing harmonic $`[5]`$, $`D_{12}`$ in the span of all harmonics $`0..6`$; sweep $`P=2..40`$ with matched cosines and with all cosines of order $`\le48`$: in span exactly for $`P\in\{2,3,4,6\}`$ [`b_cosine_span`]; harmonic criterion for $`P\le60`$ [`b_harmonic_criterion`]. The comparison with the Ramanujan bank is structural: $`K_m=\frac1{\varphi(m)}\sum_{(a,m)=1}\cos(2\pi an/m)`$ contains *all* the units' harmonics at once.

*What this does not say.* Not a statement about trained cosine banks with free frequencies (T9d); about the LP margin of C3.1 (a different, NUMERICAL, quantity); or about the best approximation of $`D_P`$ by cosines.

**T9b.4 (off-comb leakage in an unnormalized sum; PROVED, exact numbers).** Let $`\kappa`$ be $`P`$-periodic, $`\kappa=D_P+r`$ with $`r(0)=0`$ and $`r(s)=\delta`$ for $`s\ne0`$, i.e. $`\kappa=(1-\delta)D_P+\delta K_1`$ (a convex mixture; realizable in any bank containing the divisors of $`P`$). Take $`W=\lambda I`$, motif with symbol $`\alpha`$ at residue $`0`$ and symbol $`\beta\ne\alpha`$ at the other $`P-1`$ residues, query residue $`0`$. By L9.1 (here $`\kappa_P=\kappa`$) the log-odds of $`\alpha`$ over $`\beta`$ satisfy, a.s.,
```math
\frac{\ell_i(\alpha)-\ell_i(\beta)}{i}\;\longrightarrow\;\frac{\lambda}{P}\Big(\underbrace{(p-\nu)}_{\text{comb}}\;-\;\underbrace{(P-1)\,\delta\,(p-\nu)}_{\text{off-comb leakage}}\Big)=\frac{\lambda(p-\nu)}{P}\big(1-(P-1)\delta\big).
```
This is negative iff $`\delta>1/(P-1)`$. Then $`\beta`$ beats the true symbol $`\alpha`$ at that residue, and $`\mathrm{NLL}_i\ge\lambda|{\rm slope}|\,i(1+o(1))-B\to\infty`$ linearly. (For general $`r`$ the leakage term is $`\sum_sr(s)(\pi_{t-s}(a)-\pi_{t-s}(b))`$: it depends on the motif at the other residues and can be negative.)

*Exact numbers* ($`V=16`$, $`\varepsilon=\frac12`$, $`p-\nu=\frac12`$; slope shown in units of $`\lambda`$) [`b_residual_counterexample`]: $`P=3`$, $`\delta=\frac34`$: $`-\frac1{12}`$; $`P=13`$, $`\delta=\frac1{10}`$: $`-\frac1{130}`$; $`P=13`$, $`\delta=\frac1{12}=\frac1{P-1}`$: $`0`$; $`\delta=\frac1{13}`$: $`+\frac1{338}`$; $`\delta=0`$: $`+\frac1{26}`$. Direct exact expectation at $`i=1300`$ ($`P=13`$, $`\delta=\frac1{10}`$): $`\mathbb E[E_i(\alpha)-E_i(\beta)]=-\frac{19}2`$, equal to the closed form $`(p-\nu)\big((J+1)-(P-1)\delta J\big)`$ with $`J=100`$ (the number of full periods beyond the first; the letter $`K`$ is reserved for $`K_m`$).

*Contrast with softmax.* In a softmax a row-wise constant is invisible (T8a, D4: $`\mathrm{softmax}(\text{content}+\gamma f^\star_P)=\mathrm{softmax}(\text{content}+\gamma\tfrac P{P-1}D_P)`$), and a large comb bias suppresses off-comb mass. In a sum, $`K_1`$ multiplies the *symbol-dependent* indicator, so it counts the whole-context histogram.

*What this does not say.* The motif is adversarial for the chosen $`\delta`$; for random motifs the leakage has a nonzero mean but need not be negative. Nothing here says leakage occurs in a trained head.

## 4. T9c: exactly periodic kernels

**T9c (PROVED).** Assume H-data, H-head, H-b. Let $`\kappa`$ be $`L`$-periodic, $`W`$ fixed. For $`t\in\mathbb Z/P`$ and $`c=\mu_t`$, $`b\ne c`$, define
```math
\ell_t(a)=\sum_{a'}W_{aa'}e_t(a'),\qquad s_t(c,b)=\ell_t(c)-\ell_t(b),\qquad s_t:=\min_{b\ne\mu_t}s_t(\mu_t,b),
```
with $`e_t`$ from §0 (it depends only on $`\kappa_P`$). Then:

1. **(Linear rate.)** Almost surely, for every residue $`t`$ and every $`a`$, $`b`$: along $`i\equiv t\pmod P`$, $`\ \ell_i(a)-\ell_i(b)=(i+1)(\ell_t(a)-\ell_t(b))+O(\sqrt{i\log i})`$. In particular the margin of the true symbol is $`s_t(\mu_t,b)\,i+O(\sqrt{i\log i})`$.
2. **(Exact exponential rate.)** Almost surely, along $`i\equiv t\pmod P`$,
   ```math
   \lim_{i\to\infty}\frac1i\log\mathrm{NLL}_i=-\max(0,\ s_t) .
   ```
   So the NLL decays exponentially along *every* residue class iff $`s_t>0`$ for every $`t`$, i.e. iff every residue class gets a strictly positive linear margin against every competitor. If some $`s_t(\mu_t,b)<0`$ the margin of $`b`$ over the truth grows linearly and $`\mathrm{NLL}_i\ge|s_t(\mu_t,b)|\,i\,(1-o(1))`$; if $`s_t=0`$ there is no exponential rate (rate $`0`$).
3. **(Expectation.)** If $`\min_ts_t>0`$ there are $`C,c>0`$ with $`\mathbb E\,\mathrm{NLL}_i\le Ce^{-ci}`$ for all $`i`$.
4. **(Comb kernel.)** For $`\kappa=D_P`$, $`W=\lambda I`$: $`e_t(a)=\pi_t(a)/P`$, $`s_t=\lambda(p-\nu)/P`$ for every motif and every $`t`$: the Bayes rate of T9a.4 per index.
5. **(Dominance criterion, $`W=I`$.)** Let $`V\ge2`$. For $`W=I`$: $`s_t(\mu_t,b)=\frac{p-\nu}P\big(\kappa_P(0)+\sum_{s\ne0}\kappa_P(s)\,\varepsilon_s\big)`$ with $`\varepsilon_s=\mathbf 1[\mu_{t-s}=\mu_t]-\mathbf 1[\mu_{t-s}=b]\in\{-1,0,1\}`$. Consequently all margins are positive **for every motif** iff
   ```math
   \kappa_P(0)>\sum_{s\ne0}|\kappa_P(s)| .
   ```
   This is a **worst-motif** criterion: it says every motif works iff the fold dominates, and the necessity half uses a motif built from the signs of $`\kappa_P`$ (two symbols suffice). It does not say that a non-dominant fold fails on typical motifs; T9c.6 does.
6. **(Random motifs.)** Let $`\kappa_P`$ be the fold, $`k_0=\kappa_P(0)`$, $`\sigma=\sum_s\kappa_P(s)`$, and for a motif $`\mu`$ let $`M_a(t)=\sum_{r:\,\mu_r=a}\kappa_P(t-r)`$ be the $`\kappa_P`$-weighted histogram of the motif seen from residue $`t`$ ($`M_{\mu_t}(t)`$ contains $`k_0`$).
   - **(a) Characterization.** $`e_t(a)=\frac1P\big(\nu\sigma+(p-\nu)M_a(t)\big)`$, hence for $`W=\lambda I`$ (or any $`W=(d-o)I+oJ`$ with $`d>o`$, $`J`$ the all-ones matrix) $`s_t(\mu_t,b)=\frac{(p-\nu)\lambda'}{P}\big(M_{\mu_t}(t)-M_b(t)\big)`$ with $`\lambda'=\lambda`$ (resp. $`d-o`$). **All residues have a positive margin iff, for every $`t`$, $`\mu_t`$ is the strict argmax of $`M(t)`$.** For general $`W`$: $`s_t(c,b)=\frac1P\big[(p-\nu)\big((WM)_c-(WM)_b\big)+\nu\sigma\big((W\mathbf 1)_c-(W\mathbf 1)_b\big)\big]`$; the second term vanishes when $`W\mathbf 1`$ is constant or $`\sigma=0`$.
   - **(b) Exact probabilities.** Let $`q(\kappa_P,V)`$ be the probability, over a uniformly random motif ($`P`$ residues, $`V`$ symbols, i.i.d. uniform), that every residue has a positive margin. Then $`q=1`$ for every dominant fold (T9c.5), in particular for $`D_P`$; $`q=V^{1-P}`$ for the constant fold $`\kappa_P\equiv1`$ (positive margins force a constant motif); and for the leakage fold $`\kappa_P=(1-\delta)D_P+\delta K_1`$ ($`0<\delta<1`$) the event depends only on the symbol counts $`h_a`$ of the motif: it holds iff $`\delta\,(h_{\max}-h_{\min})<1-\delta`$, where $`h_{\min}`$ is the smallest count among *present* symbols. $`q`$ is then a finite sum over partitions of $`P`$ into at most $`V`$ parts.
   - **(c) Necessary collisions.** Let $`n_{\min}`$ be the least $`j`$ such that the $`j`$ largest values of $`|\kappa_P(s)|`$, $`s\ne0`$, sum to at least $`k_0`$ ($`n_{\min}=\infty`$ iff $`k_0>\sum_{s\ne0}|\kappa_P(s)|`$, i.e. iff the fold is dominant), and $`T=\Pr\big(\sum_{s\ne0}|\kappa_P(s)|B_s\ge k_0\big)`$ with $`B_s`$ i.i.d. Bernoulli$`(1/V)`$. If $`k_0>0`$, then
     ```math
     1-q\;\le\;P\,(V-1)\,T\;\le\;P\,(V-1)\,\Pr\big(\mathrm{Bin}(P-1,\tfrac1V)\ge n_{\min}\big).
     ```
     So failure needs at least $`n_{\min}`$ off-comb residues carrying one prescribed symbol each; the probability is of order $`V^{-n_{\min}}`$ times a combinatorial factor.
   - **(d) Not claimed.** A matching lower bound on $`1-q`$ is not proved; the near-certain failure at $`P=13`$ below is NUMERICAL.
7. **(Zero-mean kernels.)** If $`\sum_s\kappa_P(s)=0`$ (by L9.2: the order-1 component of $`\kappa`$ vanishes) then for a *constant* motif $`s_t=0`$ for every $`W`$; thus $`K_P`$ ($`P`$ prime) and the simplex comb $`f^\star_P`$ alone give no linear rate on constant motifs.

*Proof.* (1) L9.1 and L9.3 give $`E_i(a)=(i+1)e_t(a)+O(\sqrt{i\log i})`$ a.s. (one event for all $`a`$ and all $`i`$, hence for all $`W`$); $`\ell_i=b_i+WE_i`$ with $`b_i`$ bounded.
(2) Write $`m_b(i)=\ell_i(\mu_t)-\ell_i(b)=s_t(\mu_t,b)\,i+o(i)`$ and $`x_i=\sum_be^{-m_b(i)}`$. If $`s_t>0`$: $`x_i\to0`$, $`\log(1+x_i)\in[x_i/2,x_i]`$ eventually, and $`\frac1i\log x_i\to-\min_bs_t(\mu_t,b)=-s_t`$. If $`s_t\le0`$: let $`b^*`$ attain the minimum, so $`m_{b^*}(i)=s_ti+o(i)\le\epsilon i`$ eventually, for every $`\epsilon>0`$. Then $`\mathrm{NLL}_i\ge\log(1+e^{-m_{b^*}(i)})\ge\log(1+e^{-\epsilon i})\ge\frac12e^{-\epsilon i}`$ (using $`\log(1+y)\ge y/2`$ on $`[0,1]`$), hence $`\liminf\frac1i\log\mathrm{NLL}_i\ge0`$; and $`\mathrm{NLL}_i\le\log V+\max_b(-m_b)^+=O(i)`$ gives $`\limsup\le0`$. If $`s_t<0`$ also $`\mathrm{NLL}_i\ge-m_{b^*}(i)=|s_t|\,i\,(1-o(1))`$.
(3) Fix $`t`$, $`b`$, $`w=W_{\mu_t\cdot}-W_{b\cdot}`$. The margin is $`A_i+(b_i(\mu_t)-b_i(b))`$ with $`A_i=\sum_{j\le i}\kappa(i-j)w(X_j)`$, independent terms, each in an interval of length $`R=G(\max w-\min w)`$, and $`\mathbb EA_i\ge s_ti-C_0`$ by L9.1. Hoeffding with $`i+1`$ terms: $`\Pr(A_i\le\mathbb EA_i-s_ti/4)\le\exp\!\big(-2(s_ti/4)^2/((i+1)R^2)\big)=\exp\!\big(-\tfrac{s_t^2i^2}{8(i+1)R^2}\big)\le\exp\!\big(-s_t^2i/(16R^2)\big)`$ (the last step is $`i^2/(i+1)\ge i/2`$ for $`i\ge1`$; the exponent $`s_t^2i/(8R^2)`$ written in an earlier version is *not* implied by Hoeffding) [`c3_hoeffding_constant`]. On the complement the margin is $`\ge\frac34s_ti-C_0-B`$ for all $`b`$, so $`\mathrm{NLL}\le(V-1)e^{C_0+B-3s_ti/4}`$; on the event itself $`\mathrm{NLL}\le\log V+C_1(i+1)`$ since $`|\ell|\le C(i+1)`$. Multiply by the probabilities and sum.
(4) $`(D_P)_P=\delta_0`$ (L9.2), so $`e_t(a)=\pi_t(a)/P`$, and with $`W=\lambda I`$, $`s_t(c,b)=\lambda(\pi_t(c)-\pi_t(b))/P=\lambda(p-\nu)/P`$.
(5) $`\pi_r(c)-\pi_r(b)=(p-\nu)(\mathbf 1[\mu_r=c]-\mathbf 1[\mu_r=b])`$; the $`s=0`$ term is $`(p-\nu)\kappa_P(0)`$ because $`\mu_t=c`$. If the dominance inequality holds the bracket is $`\ge\kappa_P(0)-\sum|\kappa_P(s)|>0`$. Conversely choose $`\mu_{t-s}=b`$ if $`\kappa_P(s)>0`$, $`\mu_{t-s}=c`$ if $`\kappa_P(s)<0`$ (so $`\varepsilon_s=-\mathrm{sgn}\,\kappa_P(s)`$), which gives bracket $`=\kappa_P(0)-\sum|\kappa_P(s)|\le0`$ if the inequality fails.
(6) (a) $`\pi_r(a)=\nu+(p-\nu)\mathbf 1[\mu_r=a]`$ gives $`e_t(a)=\frac1P\sum_s\kappa_P(s)\pi_{t-s}(a)=\frac1P(\nu\sigma+(p-\nu)M_a(t))`$, and $`\ell_t=We_t`$; for $`W=\lambda'I+oJ`$, $`W\mathbf 1`$ is constant so the $`\sigma`$-term cancels in $`\ell_t(c)-\ell_t(b)`$, leaving $`\lambda'(p-\nu)(M_c-M_b)/P`$. (b) Dominant folds: T9c.5. Constant fold: $`M_a(t)=h_a`$ for every $`t`$; if $`\mu_t`$ is the strict argmax of $`h`$ for every $`t`$, all $`\mu_t`$ are the same symbol (the argmax is unique), so the motif is constant; conversely a constant motif has $`h_c=P>0=h_b`$. That is $`V`$ motifs out of $`V^P`$. Leakage fold: $`M_a(t)=(1-\delta)\mathbf 1[a=\mu_t]+\delta h_a`$, so the margin of a present symbol $`c`$ against $`b`$ is $`(1-\delta)+\delta(h_c-h_b)`$, worst for $`b`$ of maximal count among $`b\ne c`$; this is positive for all present $`c`$ iff $`\delta(h_{\max}-h_c)<1-\delta`$ for every present non-argmax $`c`$ (an argmax $`c`$ is always fine), i.e. iff $`\delta(h_{\max}-h_{\min})<1-\delta`$. The number of motifs with count multiset $`(h_a)`$ is a multinomial times the number of assignments of the counts to symbols; sum over partitions. (c) Condition on $`\mu_t=c`$ and fix $`b\ne c`$. The margin is $`\le0`$ iff $`\sum_{r\ne t}\kappa_P(t-r)(\mathbf 1[\mu_r=b]-\mathbf 1[\mu_r=c])\ge k_0`$. The left side is at most $`\sum_{r\ne t}|\kappa_P(t-r)|\,\mathbf 1[\mu_r=\sigma_r]`$ with $`\sigma_r=b`$ if $`\kappa_P(t-r)>0`$ and $`\sigma_r=c`$ otherwise; the indicators are independent Bernoulli$`(1/V)`$ because the $`\mu_r`$, $`r\ne t`$, are independent of $`\mu_t`$ and uniform. This gives probability $`\le T`$ for the pair $`(t,b)`$; the union over $`P`$ residues and $`V-1`$ competitors gives the first inequality. At least $`n_{\min}`$ indicators must equal $`1`$ (the $`j`$ largest weights are the best case), whence the binomial tail. (7) For a constant motif $`\pi_r\equiv\pi`$, $`e_t=\frac1P\sum_s\kappa_P(s)\,\pi=0`$, so $`\ell_t\equiv0`$ for any $`W`$. $`\sum_s\kappa_P(s)=P\cdot(\text{coefficient of }K_1)`$ because $`\sum_{s\in\mathbb Z/P}K_m(s)=0`$ for $`1<m\mid P`$. ∎

*Checks.* L9.1 and L9.2 as above. Dominance criterion: 120 random rational $`\kappa_P`$ for each $`P\in\{3,4\}`$ ($`V=3`$, all $`3^P`$ motifs, all $`t`$, all $`b`$): positivity for all motifs coincides with strict dominance in all 240 cases (64 of them dominant, so both outcomes occur) [`c_dominance`]. Zero-mean kernels ($`f^\star_{13}`$, $`K_{13}`$; $`V=5`$): zero slope for every constant motif and 5 random integer $`W`$ per symbol; $`D_{13}`$ has slope $`(p-\nu)/13`$ for every motif (20 random motifs plus a constant one) [`c_zero_mean_and_comb`].
*Example (the constant coordinate).* For $`\kappa=\beta_1K_1+\beta_PK_P`$, $`P`$ prime, $`\beta_1,\beta_P\ge0`$: $`\kappa_P(0)=\beta_1+\beta_P`$, $`\kappa_P(s)=\beta_1-\beta_P/(P-1)`$ for $`s\ne0`$, and strict dominance holds iff $`\beta_1>0`$ and $`\beta_P>\frac{P-2}2\beta_1`$. (Case analysis on the sign of $`(P-1)\beta_1-\beta_P`$.) $`D_P`$ itself has $`\beta_P=(P-1)\beta_1`$, inside the region. Verified over a grid of $`(\beta_1,\beta_P)`$ by brute force over all motifs: $`P=5,V=3`$ and $`P=7,V=2`$, 48 cases, no mismatch [`c_two_order_family`].

*Checks for T9c.6* [`c_random_motif`, EXACT unless marked]. (a) The slope formula of (a) equals the exact limit slope for 10 386 (motif, residue, competitor) triples with random rational $`\kappa_P`$ and random integer $`W`$ ($`V\in\{2,3\}`$, $`P\in\{4,5,6\}`$), and for $`W=(d-o)I+oJ`$. (b) Enumeration of all $`V^P`$ motifs: $`q=V^{1-P}`$ for the constant fold and $`q=1`$ for $`D_P`$ ($`(V,P)=(2,5),(3,5),(3,6),(4,5)`$); the partition formula equals the enumeration for the leakage fold ($`\delta=\frac15,\frac13,\frac12`$). (c) $`1-q\le\min(1,P(V-1)T)`$ against the enumerated failure probability for 12 random rational folds and 4 leakage folds, and dominant folds have $`n_{\min}=\infty`$ and $`1-q=0`$.

*Exact random-motif numbers, leakage fold, $`V=16`$* (partition sum, `Fraction`; $`1-q`$ is the probability that some residue has a non-positive margin):

| $`\delta`$ | $`P=12`$ | $`P=13`$ |
|---|---|---|
| $`\frac1{20}`$ | $`0`$ | $`0`$ |
| $`\frac1{10}`$ | $`8.0\cdot10^{-10}`$ | $`3.5\cdot10^{-9}`$ |
| $`\frac15`$ | $`8.3\cdot10^{-3}`$ | $`1.28\cdot10^{-2}`$ |
| $`\frac13`$ | $`0.489`$ | $`0.577`$ |

Compare T9b.4: at $`P=13`$, $`\delta=\frac1{10}`$ the *adversarial* motif (counts $`(1,12)`$) has a negative slope, yet a uniformly random $`V=16`$ motif fails with probability $`3.5\cdot10^{-9}`$, because $`\delta(h_{\max}-h_{\min})<1-\delta`$ needs a count spread of $`9`$ or more to fail.

**T9c.6-num (NUMERICAL; Monte Carlo, uniformly random motifs, $`V=16`$, fixed seed $`20261023`$ with a per-arm offset).** "Fails" is the number of sampled motifs with at least one residue whose margin is $`\le10^{-9}`$ ($`W=\lambda I`$; the same events for $`W=(d-o)I+oJ`$, $`d>o`$). "Pilot" folds are the folds computed by the audit for the development pilot (one training seed, development data seed 7), copied from `audit/consistency_out.txt` to three decimals; nothing was re-trained. $`n_{\min}`$, $`T`$ and the bound $`P(V-1)T`$ are from (c) (exact subset enumeration, floating point).

| $`P`$ | fold | $`N`$ | fails | dom. margin | $`n_{\min}`$ | bound $`P(V-1)T`$ |
|---|---|---|---|---|---|---|
| 12 | $`D_{12}`$ | 20 000 | 0 | $`+1`$ | $`\infty`$ | 0 |
| 12 | pilot `arith` | 20 000 | 0 | $`+8.58`$ | $`\infty`$ | 0 |
| 12 | pilot `cosine_full` | 20 000 | 0 | $`+8.69`$ | $`\infty`$ | 0 |
| 12 | pilot `cosine_fund` | 100 000 | 16 (Wilson 95%: $`1.0`$–$`2.6\cdot10^{-4}`$) | $`-2.23`$ | 6 | $`1.4\cdot10^{-4}`$ |
| 12 | `cosine_fund` span, $`L^\infty`$-optimal fold (max $`\lvert f(s)\rvert=0.155`$) | 100 000 | 20 | $`-0.70`$ | 7 | $`1.8\cdot10^{-4}`$ |
| 12 | `cosine_fund` span, $`L^1`$-optimal fold (dominance margin $`0`$) | 20 000 | 20 000 | $`0`$ | 1 | 1 |
| 12 | pilot `cosine_learned` (window fold, $`N=8192`$) | 20 000 | 20 000 | $`-4.69`$ | none ($`k_0<0`$) | 1 (vacuous) |
| 12 | pilot `uniform` | 20 000 | 20 000 | $`-10`$ | 1 | 1 |
| 13 | $`D_{13}`$, pilot `arith`, pilot `cosine_full` | 20 000 each | 0 | $`+1`$, $`+6.32`$, $`+7.83`$ | $`\infty`$ | 0 |
| 13 | pilot `cosine_fund` | 20 000 | 19 987 | $`-4.42`$ | 2 | 1 (vacuous) |
| 13 | `cosine_fund` span, $`L^\infty`$-optimal fold (max $`\lvert f(s)\rvert=0.890`$) | 100 000 | 99 923 | $`-6.00`$ | 2 | 1 |
| 13 | `cosine_fund` span, $`L^1`$-optimal fold | 20 000 | 19 983 | $`-4.07`$ | 2 | 1 |
| 13 | pilot `cosine_learned` (window fold) | 20 000 | 20 000 | $`-0.73`$ | 1 ($`k_0>0`$ small) | 1 |
| 13 | pilot `uniform` | 20 000 | 20 000 | $`-11`$ | 1 | 1 |

(The `cosine_fund` span at modulus $`P`$ is $`\mathrm{span}\{\cos(2\pi s/d):d\mid P\}`$ by L9.2′: $`d\in\{1,2,3,4,6,12\}`$ at $`P=12`$ and $`\{1,13\}`$ at $`P=13`$. The $`L^1`$- and $`L^\infty`$-optimal folds minimize $`\sum_{s\ne0}|f(s)|`$, resp. $`\max_{s\ne0}|f(s)|`$, subject to $`f(0)=1`$; vertex enumeration in floating point [`b_span_dominance`, NUMERICAL]; the minimal values are $`1.0000`$ and $`0.1547`$ at $`P=12`$, $`5.0739`$ and $`0.8902`$ at $`P=13`$. The audit's own independent estimate for the pilot `cosine_fund` fold at $`P=12`$ is $`10/10^5`$ (`audit/consistency_extra.txt`).)

*Reading.* This separates `cosine_fund` at $`P=12`$ from $`P=13`$, which T9c.5 does not. At $`P=12`$ the divisors of $`12`$ give the span enough freedom that the fold can keep every off-comb entry small ($`0.155\,k_0`$ at best, $`1.48/5.81=0.25\,k_0`$ for the fitted fold), so a failure needs six to seven off-comb residues to carry prescribed symbols: rare, and the rigorous bound $`1.4\cdot10^{-4}`$ (T9c.6(c)) is of the same size as the observed $`1.6\cdot10^{-4}`$ (the sampling interval, $`1.0`$–$`2.6\cdot10^{-4}`$, contains the bound). At $`P=13`$ the span is $`\{1,\cos(2\pi s/13)\}`$, a smooth bump whose neighbours of residue $`0`$ carry $`0.89\,k_0`$ at best ($`0.94`$ and $`0.79\,k_0`$ for the fitted fold), two colliding residues suffice, and the sampled motifs fail almost always. Strict dominance (T9c.5) is **sufficient** for a positive rate on every motif but is **not what predicts the pilot**: both `cosine_fund` folds fail it, the $`P=12`$ fold fails on about $`10^{-4}`$ of motifs, and the $`L^1`$-optimal fold, which is the closest to dominance (margin $`0`$, an antipodal tie $`f(6)=f(0)`$), fails on every sampled motif. The ranking of the arms by dominance margin is therefore not the ranking by random-motif success.

*What this does not say.* For kernels other than $`D_P`$ it does not say a good $`W`$ exists or is found by training; the condition involves the motif, and $`W`$ is shared. It treats $`W`$, $`\kappa`$ as fixed (not learned from data) and $`b_i`$ as bounded. The exponent $`\max(0,s_t)`$ is per query index $`i`$ and says nothing about finite $`i`$ (the $`O(\sqrt{i\log i})`$ error is not uniform in $`\kappa`$, and the deterministic error of L9.1 has a constant of order $`10^3`$–$`10^4`$ for banks with many orders, see "Size of the constant"). T9c.5 is a worst-motif condition: it is sufficient, not necessary, and does not by itself predict the behaviour on typical motifs (T9c.6); the random-motif numbers are for uniformly random motifs, not for the motifs of any particular data set.

## 5. T9d: non-resonant kernels

**T9d (ASYMPTOTIC, $`i\to\infty`$; nothing at finite $`i`$, see the finite-window remark below).** Let $`\theta\in\mathbb R^K`$ satisfy T5's joint hypothesis (no integer $`c`$ with $`c\cdot\theta\in2\pi(\mathbb Q\setminus\mathbb Z)`$), $`H`$ the closure of $`\{n\theta\}`$ in $`\mathbb T^K`$ (compact, connected), $`g:\mathbb T^K\to\mathbb R`$ continuous, $`\kappa(n)=g(n\theta)`$, and $`\bar\kappa=\int_Hg\,dm_H`$ ($`m_H`$ Haar). Assume H-data, H-head, and fix $`P`$ and $`\mu`$.

1. **(Deterministic part.)** $`\frac1i\sum_{0\le n\le i,\ n\equiv\varrho\,(P)}g(n\theta)\to\bar\kappa/P`$ for every $`\varrho\in\mathbb Z/P`$, hence $`D_i(a)/i\to\bar E(a):=\frac{\bar\kappa}P\sum_{r\in\mathbb Z/P}\pi_r(a)=\bar\kappa\big(\nu+(p-\nu)h_a/P\big)`$, where $`h_a=\#\{r:\mu_r=a\}`$, as $`i\to\infty`$ along **all** $`i`$ (no dependence on $`i\bmod P`$).
2. **(Noise.)** By L9.3, almost surely $`E_i(a)-D_i(a)=O(\sqrt{i\log i})`$.
3. **(Linear part.)** Almost surely $`E_i(a)/i\to\bar E(a)`$ for every $`a`$, along all $`i`$. The query residue does not appear in the limit: the linear-rate part of the evidence is the same for every query residue.
4. **(No exponential rate on some class.)** Suppose $`\mu_{r_1}=c_1\ne c_2=\mu_{r_2}`$. There is a probability-one event $`\Omega_0`$ (depending on $`\theta,g,\mu,P`$ but **not** on $`W`$ or $`b`$) on which, simultaneously for every $`W\in\mathbb R^{V\times V}`$ and every sequence $`b_i`$ with $`\sup_i|b_i|<\infty`$ (or merely $`o(i)`$; $`b_i`$ may depend on the data): there is a class $`r^*\in\{r_1,r_2\}`$, with true symbol $`c^*=\mu_{r^*}`$ and competitor $`c'`$ the other of $`\{c_1,c_2\}`$, such that for every $`\epsilon>0`$ eventually in $`i\equiv r^*\ (P)`$,
   ```math
   \ell_i(c^*)-\ell_i(c')\le\epsilon\,i,\qquad\text{hence}\qquad \mathrm{NLL}_i\ge\tfrac12e^{-\epsilon i}.
   ```
   So along that class $`\lim\frac1i\log\mathrm{NLL}_i=0`$ (or $`\liminf\ge0`$): no bound $`\mathrm{NLL}_i\le Ce^{-ci}`$ with $`c>0`$ holds. With $`\ell(a):=\sum_{a'}W_{aa'}\bar E(a')`$ and $`\Delta=\ell(c_1)-\ell(c_2)`$: $`r^*=r_1`$ if $`\Delta\le0`$, $`r^*=r_2`$ if $`\Delta\ge0`$; if $`\Delta<0`$ strictly, the margin of the wrong symbol grows linearly and $`\mathrm{NLL}_i\ge|\Delta|\,i\,(1-o(1))`$ on class $`r_1`$ (symmetrically for $`\Delta>0`$ on $`r_2`$).

*Proof.* (1) Fix $`\varrho`$. By T5 steps 2–3, $`(\varrho+kP)\theta`$ is equidistributed in $`H`$ with respect to $`m_H`$ ($`H`$ is connected, so multiplication by $`P`$ is onto $`H`$ and $`P\theta`$ generates a dense subgroup of $`H`$; Weyl). Hence the mean of $`g`$ over $`n=\varrho+kP\le i`$ tends to $`\bar\kappa`$, and there are $`i/P+O(1)`$ such $`n`$. There are only $`P`$ classes, so the convergence is uniform in $`\varrho`$. Now
```math
D_i(a)=\sum_{n=0}^ig(n\theta)\,\pi_{i-n}(a)=\sum_{r\in\mathbb Z/P}\pi_r(a)\sum_{\substack{0\le n\le i\\ n\equiv i-r\,(P)}}g(n\theta),
```
because $`\pi_{i-n}`$ depends on $`(i-n)\bmod P`$; each inner sum is $`(\bar\kappa/P+o(1))\,i`$ uniformly. (2) is L9.3, valid for any bounded $`\kappa`$. (3) follows. (4) On $`\Omega_0=\{E_i(a)/i\to\bar E(a)\ \forall a\}`$ the margin of $`c_1`$ over $`c_2`$ at class $`r_1`$ is $`\ell_i(c_1)-\ell_i(c_2)=(\ell(c_1)-\ell(c_2))\,i+o(i)=\Delta i+o(i)`$ (the $`b`$-term is $`O(1)=o(i)`$), and the margin of $`c_2`$ over $`c_1`$ at class $`r_2`$ is $`-\Delta i+o(i)`$. One of the two limits $`\pm\Delta`$ is $`\le0`$. For that class, $`m\le\epsilon i`$ eventually, and $`\mathrm{NLL}_i\ge\log(1+e^{-m})\ge\log(1+e^{-\epsilon i})\ge\frac12e^{-\epsilon i}`$. The upper bound $`\mathrm{NLL}_i=O(i)`$ then gives $`\lim\frac1i\log\mathrm{NLL}_i=0`$. For $`\Delta<0`$, $`\mathrm{NLL}_i\ge-m`$. ∎

*Quantifiers.* "Almost surely" refers to the token noise with $`\mu`$, $`\theta`$, $`g`$ fixed. The event $`\Omega_0`$ does not depend on $`W`$ or $`b`$, so the conclusion holds simultaneously for all $`W`$ and all bounded (even data-dependent) $`b`$. The *limits* (Weyl) are Cesàro/equidistribution statements about the deterministic sequence $`n\theta`$; the *noise* statement is the only probabilistic one and is $`O(\sqrt{i\log i})`$ a.s. No rate is claimed for the Weyl part.

*Contrast with T9c.* An $`L`$-periodic $`\kappa`$ keeps, by L9.2, exactly the components with $`m\mid P`$ after folding; for a non-resonant $`\kappa`$ the fold is the constant $`\bar\kappa`$ for every $`P`$ (item 1). Then $`s_t`$ is residue-independent and antisymmetric across residues with different symbols: [`d_algebra`] checks, in exact arithmetic ($`V=6`$, $`P=5`$, constant fold, 40 random motifs and random integer $`W`$), that the slope equals $`\ell(c)-\ell(b)`$ for every query residue and that margins at two residues with different symbols sum to $`0`$.
*Hypotheses are essential.* (i) If the motif is constant, no pair $`r_1,r_2`$ exists and the conclusion fails (a constant motif is learnable by a non-resonant kernel when $`\bar\kappa>0`$). (ii) For an exactly periodic $`\kappa`$ T5's hypothesis fails ($`H`$ finite) and T9c applies. (iii) The example $`\theta=\frac{2\pi}{13}(1+10^{-3})`$ is **not** a T9d instance: $`\theta/2\pi=\frac{1.001}{13}=\frac{77}{1000}`$ exactly, so $`\kappa`$ is $`1000`$-periodic. It is a case of T9c: $`\gcd(1000,13)=1`$, so by L9.2 its fold to modulus 13 is the constant $`\frac12`$ (numerically $`\max|\text{fold}-\frac12|=1.2\cdot10^{-14}`$ for $`\kappa=\frac12(1+\cos n\theta)`$ [`d_numeric`]), and the limits coincide with T9d's. The numerical table uses it together with two genuine T9d instances.

**Finite-window remark (closed form PROVED; the regime it describes is outside T9d).** T9d is a statement for $`i\to\infty`$. For a kernel near a resonance it says nothing until $`i`$ exceeds the drift horizon. Let $`\kappa(n)=\cos(n\theta)`$ with $`\theta=2\pi a/P+\eta`$ (every $`\theta`$ can be written so with $`|\eta|\le\pi/P`$) and define the window fold $`\hat\kappa_{P,N}(s)=\frac PN\sum_{0\le n<N,\ n\equiv s\,(P)}\kappa(n)`$. With $`U_s=\lceil(N-s)/P\rceil`$ terms in class $`s`$,
```math
\hat\kappa_{P,N}(s)=\frac PN\,\frac{\sin(P\eta U_s/2)}{\sin(P\eta/2)}\,\cos\!\Big(\frac{2\pi as}{P}+\eta\big(s+\tfrac{P(U_s-1)}2\big)\Big).
```
For $`N=PU`$: $`\hat\kappa_{P,N}(s)=D_U(P\eta)\cos\big(\tfrac{2\pi as}P+\eta(s+\tfrac{P(U-1)}2)\big)`$ with the normalized Dirichlet kernel $`D_U(x)=\frac{\sin(Ux/2)}{U\sin(x/2)}`$, $`D_U(0)=1`$, and $`|D_U(P\eta)|\le\frac{\pi}{N|\eta|}`$. *Proof.* Put $`n=s+Pu`$; then $`e^{iPu\theta}=e^{iPu\eta}`$ because $`e^{2\pi iau}=1`$, so the class sum is $`\mathrm{Re}\big[e^{i(2\pi as/P+s\eta)}\sum_{u<U_s}e^{iPu\eta}\big]`$, a geometric sum equal to $`e^{iP\eta(U_s-1)/2}\sin(P\eta U_s/2)/\sin(P\eta/2)`$; and $`|\sin(x/2)|\ge|x|/\pi`$ for $`|x|\le\pi`$. ∎ So: for $`N|\eta|\ll1`$ the window fold is $`\approx\cos(2\pi as/P)`$, the fold of the *periodic* cosine, which depends on the residue (T9c's regime); the residue-dependent part is damped by the factor $`D_U(P\eta)=O(1/(N|\eta|))`$ and the constant fold $`\bar\kappa=0`$ of T9d is reached only for $`N\gg1/|\eta|`$. A bank of such components adds linearly, so the horizon is $`1/|\eta|`$ of the *heaviest* near-resonant components. For the near-resonant kernel of the table below ($`a=1`$, $`\eta=\frac{2\pi}{13}\sqrt2\cdot10^{-3}=6.8\cdot10^{-4}`$, horizon $`1/\eta\approx1460`$) the factor $`|D_U(P\eta)|`$ is $`0.999,\,0.980,\,0.704,\,0.113,\,0.018`$ at the table's $`N=247,\,1014,\,4095,\,16380,\,65533`$. The audit's observation that the pilot's learned frequencies lie within $`\delta\le4.6\cdot10^{-3}`$ of harmonics (horizons $`217`$–$`1914`$) and that the window fold at $`N=8192`$ is still far from constant (`audit/consistency_extra.txt`) is the regime this remark describes; the remark was not fitted to it. [`d_window_fold`: the closed form against brute force for 90 random $`(P,a,\eta,N)`$, maximal deviation $`1.4\cdot10^{-12}`$, NUMERICAL.]

*What this does not say.* It concerns a single-layer evidence head on raw tokens with a fixed, content-independent kernel; it says nothing about deeper networks, content-dependent kernels, multiple heads sharing $`W`$ across several kernels, or which frequencies training finds. It says nothing at finite $`i`$ (see the table: near a resonance a drifting kernel looks like a comb for $`i\lesssim1/|\theta-2\pi/13|`$). It does not say non-resonant kernels are useless: with $`\bar\kappa\ne0`$ they count the whole-context histogram.

**T9d-num (NUMERICAL, one seed 20261023).** $`P=13`$, $`V=16`$, $`\varepsilon=\frac12`$, motif $`\mu_r=r`$ (13 distinct symbols), query residue $`0`$ (true symbol $`0`$, "other" = symbol $`1`$), one simulated sequence of length $`2^{16}+1`$; $`E_i`$ at $`i\equiv0\ (13)`$ [`d_numeric`]. "gap/i" $`=(E_i(0)-E_i(1))/i`$; "limit gap" is the difference of the two limits computed from L9.1 (periodic kernels) or T9d (the constant fold $`\bar\kappa=\frac12`$ gives the same limit for both symbols, so the gap is $`0`$).

| kernel | $`i=247`$ | $`1014`$ | $`4095`$ | $`16380`$ | $`65533`$ | limit gap |
|---|---|---|---|---|---|---|
| $`D_{13}`$ | 0.05263 | 0.04832 | 0.04151 | 0.03730 | **0.03815** | 0.03846 |
| $`\frac12(1+\cos\frac{2\pi n}{13})`$ (period 13, T9c) | −0.00574 | 0.01016 | 0.00392 | 0.00172 | 0.00276 | 0.00220 |
| $`\frac12(1+\cos n\theta)`$, $`\theta=2\pi\cdot\frac{77}{1000}`$ (period 1000, T9c) | −0.00610 | 0.00802 | −0.00258 | −0.00116 | −0.00023 | 0 |
| $`\theta=\frac{2\pi}{13}(1+\sqrt2\cdot10^{-3})`$ (irrational, T9d) | −0.00627 | 0.00706 | −0.00354 | −0.00053 | 0.00044 | 0 |
| $`\theta=2\pi\frac{\sqrt5-1}2`$ (irrational, T9d) | −0.02420 | −0.00125 | 0.00131 | −0.00113 | 0.00047 | 0 |

For $`D_{13}`$, $`E_i(0)/i\to p/13=0.04087`$ and $`E_i(1)/i\to\nu/13=0.00240`$ (columns of the run log); the three kernels with $`\kappa_P`$ constant have the same limit $`0.03486`$ for both symbols. At $`i=65533`$ the three kernels with constant fold have $`|{\rm gap}/i|=2.3,\,4.4,\,4.7\cdot10^{-4}`$ (rows 3–5), the period-13 cosine row has $`2.8\cdot10^{-3}`$ against its own limit gap $`2.2\cdot10^{-3}`$, and $`D_{13}`$ has $`0.038`$ (its limit gap is $`(p-\nu)/13=0.03846`$; the two limit columns above are rounded, $`0.04087-0.00240`$); the differences from the limits are within the sampling noise of a single seed (not estimated by the script; no statistical test is made), so only the order-of-magnitude contrast is read from the table. The exactly periodic cosine kernel has a small positive limit gap for *this* motif (it is not dominant: $`\kappa_P(0)=1<\sum_{s\ne0}\kappa_P(s)`$), which T9c.5 shows can be negative for other motifs.

## 6. T9e: dependence breaks the Bayes equality

**Setting.** At the query's residue, $`m`$ independent noisy observations $`Y_1,\dots,Y_m`$ (the channel above, true symbol $`c`$ uniform under H-prior) are each **copied $`k`$ times** at comb positions of the context, so the comb count is $`kM_a`$ with $`M_a=\#\{u:Y_u=a\}`$. The Bayes posterior given the copy structure is $`\varpi_a=\rho^{M_a}/\sum_{a'}\rho^{M_{a'}}`$ (copies carry no information beyond $`Y`$). The $`D_P`$ count head with $`W=\lambda I`$ outputs $`\mathrm{softmax}(k\lambda M)`$. Write $`D_b=M_b-M_c`$.

**T9e (PROVED; the table below is NUMERICAL).**
1. *Power law.* The head outputs $`\varpi^{\,k}/\sum_a\varpi_a^k`$. Its argmax equals the Bayes argmax, so accuracy is unchanged.
2. *Excess loss.* $`\mathbb E\,\mathrm{NLL}_{\rm head}-\mathbb E\,\mathrm{NLL}_{\rm Bayes}=\mathbb E\,\mathrm{KL}(\varpi\,\|\,\mathrm{head})\ge0`$, and the excess is strictly positive for every $`m\ge1`$, $`k\ge2`$. Hence the head is not the posterior.
3. *Bounds.* $`k\lambda\,\mathbb E[(\max_bD_b)^+]\ \le\ \mathbb E\,\mathrm{NLL}_{\rm head}\ \le\ k\,\mathbb E\,\mathrm{NLL}_{\rm Bayes}\ \le\ k(V-1)\chi^m`$. In particular, for fixed $`m\ge1`$ the expected loss is **unbounded in $`k`$** (the lower constant is positive since the plurality is wrong with positive probability), $`\mathbb E\,\mathrm{NLL}_{\rm head}/k\to\lambda\,\mathbb E[(\max_bD_b)^+]`$ with error $`\le\log V/k`$; while for fixed $`k`$ the loss still decays exponentially in the number $`m`$ of distinct observations.

*Proof.* (1) $`\rho^{kM_a}=(\rho^{M_a})^k`$; the map $`x\mapsto x^k`$ is monotone. (2) Conditional on the observations, the true symbol has law $`\varpi`$ (H-prior, symmetric channel), so $`\mathbb E[-\log h(c)\mid Y]=H(\varpi)+\mathrm{KL}(\varpi\|h)`$ for any prediction $`h`$; average over $`Y`$ (the expected NLL given $`c`$ does not depend on $`c`$ by symmetry). $`\mathrm{KL}(\varpi\|\varpi^k/Z)=0`$ iff $`\varpi^{k-1}`$ is constant iff $`\varpi`$ is uniform, i.e. all $`M_a`$ equal; for $`m\ge1`$ that has probability $`<1`$ (e.g. all $`Y_u`$ equal). (3) $`\mathrm{NLL}_{\rm head}=\log(1+\sum_{b\ne c}\rho^{kD_b})\ge\max(0,k\lambda\max_bD_b)`$. Upper: $`(1+\sum x_b)^k\ge1+\sum x_b^k`$ for $`x_b\ge0`$, so $`\log(1+\sum x_b^k)\le k\log(1+\sum x_b)`$ with $`x_b=\rho^{D_b}`$; then B1. The $`k\to\infty`$ limit: $`\mathrm{NLL}_{\rm head}\le\log V+k\lambda\max(0,\max_bD_b)`$. ∎

*Hand-checkable case* ($`V=3`$, $`\varepsilon=\frac12`$: $`p=\frac23`$, $`\nu=\frac16`$, $`\rho=4`$; $`m=1`$, $`k=2`$). The single observation is the truth w.p. $`\frac23`$. Bayes: posterior of the observed symbol $`\frac46=\frac23`$ (truth observed) or $`\frac16`$ for the truth otherwise; $`\mathbb E[\mathrm{NLL}]=\frac23\log\frac32+\frac13\log6=0.867563`$, $`\mathbb E[p_{\rm true}]=\frac12`$. Head ($`\rho^2=16`$): observed symbol gets $`\frac{16}{18}=\frac89`$; $`\mathbb E[\mathrm{NLL}]=\frac23\log\frac98+\frac13\log18=1.041979`$, $`\mathbb E[p_{\rm true}]=\frac{11}{18}`$. The argmax is correct w.p. $`\frac23`$ for both, while the head's mean top probability is $`\frac89`$ against Bayes' $`\frac23`$. These are exact fractions for this one enumeration and nothing more is read from them [`e_dependence`, headline]. Excess $`+0.174416`$.

*Enumeration* (NUMERICAL: expected NLL with probabilities as `Fraction`s and logarithms in floating point; the identities in 1–3 are asserted for every row) [`e_dependence`]:

| $`V`$ | $`m`$ | $`k`$ | Bayes | head | excess |
|---|---|---|---|---|---|
| 3 | 1 | 2 | 0.867563 | 1.041979 | +0.174416 |
| 3 | 1 | 4 | 0.867563 | 1.856175 | +0.988611 |
| 3 | 2 | 2 | 0.695406 | 0.827949 | +0.132543 |
| 3 | 2 | 4 | 0.695406 | 1.271761 | +0.576355 |
| 3 | 3 | 2 | 0.559656 | 0.698837 | +0.139181 |
| 3 | 3 | 4 | 0.559656 | 1.202580 | +0.642924 |
| 3 | 4 | 2 | 0.452504 | 0.570566 | +0.118062 |
| 3 | 4 | 4 | 0.452504 | 0.980487 | +0.527983 |
| 3 | 5 | 2 | 0.366763 | 0.467742 | +0.100980 |
| 3 | 5 | 4 | 0.366763 | 0.803343 | +0.436580 |
| 3 | 6 | 2 | 0.297945 | 0.384278 | +0.086333 |
| 3 | 6 | 4 | 0.297945 | 0.667823 | +0.369878 |
| 16 | 1 | 2 | 1.960591 | 2.706739 | +0.746147 |
| 16 | 1 | 4 | 1.960591 | 5.312455 | +3.351863 |
| 16 | 2 | 2 | 1.408892 | 1.832319 | +0.423427 |
| 16 | 2 | 4 | 1.408892 | 3.143566 | +1.734674 |
| 16 | 3 | 2 | 1.003989 | 1.288341 | +0.284352 |
| 16 | 3 | 4 | 1.003989 | 2.110715 | +1.106726 |
| 16 | 4 | 2 | 0.712113 | 0.928699 | +0.216586 |
| 16 | 4 | 4 | 0.712113 | 1.541057 | +0.828944 |

$`k\to\infty`$ check: $`V=3`$, $`m=3`$, $`k=1000`$: $`|\mathbb E\,\mathrm{NLL}/k-\lambda\,\mathbb E(\max_bD_b)^+|=1.2\cdot10^{-4}\le\log3/1000=1.1\cdot10^{-3}`$ [`e_dependence`].

*Monotonicity.* The excess is **not** monotone in $`m`$ for $`V=3`$: $`0.1744,\,0.1325,\,0.1392,\,0.1181,\,0.1010,\,0.0863`$ for $`m=1,\dots,6`$ at $`k=2`$ (and $`0.9886,\,0.5764,\,0.6429,\,0.5280,\,0.4366,\,0.3699`$ at $`k=4`$); it is decreasing in $`m`$ for $`V=16`$ in the rows computed ($`m\le4`$). No monotonicity claim is made, and no explanation of the non-monotone step ($`m=2\to3`$) is offered [`e_dependence`, `excess_strictly_decreasing_in_m`].

*What this does not say.* It is arithmetic on an explicit, idealized law (exact copies at comb positions, known $`k`$ for the Bayes reference). It makes no statement about any trained model, nor about weaker dependence (which is not analyzed); the quantities named (mean top probability, accuracy of the argmax, expected NLL) are computed for this enumeration only. It does not show the count head is unusable: the exponential decay in $`m`$ persists; what fails is the Bayes equality and a uniform-in-$`k`$ guarantee. It complements `research/metacognition/2026-10-15/MATHEMATICS.md` M6, which proves divergence as $`k\to\infty`$ for $`m=1`$.

## 7. T9f: what is not claimed

- **Learnability.** Nothing about SGD, optimization landscapes, whether training finds $`D_P`$, $`W=\lambda I`$, the period $`P`$, or $`\rho`$; no rates for trained models; no empirical claim (those belong to the E and P tracks and need a preregistered experiment).
- **Natural data.** Only the periodic denoising law of H-data (and the copy law of T9e). No natural-language, calibration, or e-value claim.
- **Speed and architecture.** No speedup, no sub-quadratic or linear-attention claim. $`E_i`$ is defined by the sum in H-head; this draft states what is computed (the evidence vector $`E_i`$ and its expectation) and nothing about the cost of computing it. Linear-attention state updates (Katharopoulos et al., 2020) are cited as context only.
- **Attribution.** Unnormalized counts of earlier tokens are classical (cache language models, Kuhn and De Mori, 1990; Grave et al., 2017; trigger features in maximum-entropy models, Rosenfeld, 1996). The kernel $`D_P`$ is this repository's object (DeltaNet study D4).
- **State of the art, deeper networks.** T9 concerns a single-layer evidence head with a fixed kernel on raw tokens plus a bounded rest-of-network. It does not extend T8/C8 to networks containing this head; the interaction of $`E_i`$ with deeper layers is OPEN.
- **OPEN.** (i) A finite-$`i`$ rate for T9d (needs Diophantine information on $`\theta`$); (ii) the polynomial prefactor of $`\chi^{n}`$ in the expected NLL (the exponent is settled, T9a.5; a local limit theorem would give $`c\,\chi^n/\sqrt n`$, not proved here); (iii) which $`(\kappa,W)`$, beyond $`D_P`$, give positive rate for *all* motifs when $`W`$ must be shared (T9c.5 settles $`W=I`$), and a matching lower bound for the random-motif failure probability of T9c.6; (iv) unknown or mixed periods; (v) the finite-context version of T9b.1 ("only if" for $`N<L`$); (vi) a finite-window statement for *sums* of near-resonant components with a rate in the amplitudes (the single-component closed form is in T9d; the bank-level statement is not).

## 8. Check index

| Statement | Check (in `check_t9.py`) | Label of the check; result in `verification/check_t9_run3.log` |
|---|---|---|
| T9a.1, identity | `a_identity` | EXACT, $`P=2..32`$ |
| T9a.2 | `a_bayes_equality`, `a_head_comb_counts` | EXACT (+ NUMERICAL float comparison) |
| T9a.3 | `a_rate_bound` | NUMERICAL (exact probabilities, floating logarithms) |
| T9a.5 | `a_tightness` | NUMERICAL (floating-point chain; $`n=30`$ distribution exact) |
| T9b.1 | `b_ramanujan_span` | EXACT, 156 cases |
| T9b.2 | `b_bank_design`, `b_ramanujan_span`, `b_span_dominance` | EXACT (cosine spans: NUMERICAL) |
| T9b.3 | `b_cosine_span`, `b_harmonic_criterion` | EXACT |
| T9b.4 | `b_residual_counterexample` | EXACT |
| L9.1 | `c_decomposition` (288 identity checks), `l91_component_bound` | EXACT ($`K_m`$); cosine cases NUMERICAL |
| L9.2 | `c_fold_projection` | EXACT |
| L9.2′ | `l92_character_fold` | NUMERICAL |
| T9c.3 constant | `c3_hoeffding_constant` | EXACT |
| T9c.4, 5, 7 | `c_dominance`, `c_zero_mean_and_comb`, `c_two_order_family` | EXACT |
| T9c.6 (a)–(c) | `c_random_motif` | EXACT |
| T9c.6-num | `c_random_motif` (`mc_V16`) | NUMERICAL (seed 20261023) |
| T9d algebra | `d_algebra` | EXACT |
| T9d window fold | `d_window_fold` | NUMERICAL |
| T9d-num | `d_numeric` | NUMERICAL |
| T9e | `e_dependence` | NUMERICAL (exact probabilities, floating logarithms) |

Unit tests (`test_check_t9.py`, 35 tests) re-assert the exact items with small parameters; output in `verification/unittest_run3.log`. Earlier logs (`check_t9_run1.log`, `check_t9_run2.log`, `unittest.log`) are kept as the record of rounds 1 and 2.

## References

Context and sources of tools; none is used as evidence for a claim.

- `docs/MATH.md`: L1, T3, C3.1, T4, P1, T5, T6, P3, T8, C8, T8a, B1. DeltaNet study `research/deltanet_hilbert/2026-10-15/MATHEMATICS.md` D4, D6. Metacognition study `research/metacognition/2026-10-15/MATHEMATICS.md` M5, M6.
- W. Hoeffding, *Probability inequalities for sums of bounded random variables*, JASA 58 (1963). H. Weyl (1916), as in `docs/MATH.md`. Dedekind–Artin independence of characters: classical.
- R. Kuhn and R. De Mori, *A cache-based natural language model for speech recognition*, IEEE Trans. Pattern Analysis and Machine Intelligence 12(6), 1990 (page numbers unverified).
- R. Rosenfeld, *A maximum entropy approach to adaptive statistical language modelling*, Computer Speech and Language 10(3), 1996 (page numbers unverified).
- E. Grave, A. Joulin, N. Usunier, *Improving neural language models with a continuous cache*, ICLR 2017 (venue details unverified).
- A. Katharopoulos et al., *Transformers are RNNs: fast autoregressive transformers with linear attention*, ICML 2020 (details unverified).
