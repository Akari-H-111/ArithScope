# ArithScope mathematics

This file states every mathematical claim the project makes, with its evidence level.

| Label | Meaning |
|---|---|
| **PROVED** | Complete proof below; statement also checked exhaustively on a finite range with exact integer/rational arithmetic (`tests/test_arith.py`). |
| **CLASSICAL** | Standard result, cited; finite exact check provided. |
| **ASYMPTOTIC** | Limit statement with a complete proof; no finite-$`N`$ rate is claimed. |
| **NUMERICAL** | Floating-point computation (linear program, simulation). Supporting evidence, not a proof. |
| **EMPIRICAL** | Measured in trained models; see [`RESULTS.md`](RESULTS.md). |
| **OPEN** | Not established. |

No statement here was checked by a proof assistant.

## 0. Notation

- $`c_m(n)=\sum_{1\le a\le m,\ \gcd(a,m)=1} e^{2\pi i a n/m}`$ is the Ramanujan sum (an integer).
- $`\varphi`$ is Euler's totient and $`\mu`$ the Möbius function.
- $`K_m(n)=c_m(n)/\varphi(m)`$ is the **normalized Ramanujan kernel** of **order** $`m`$. An order is a modulus, never the attention query $`q`$.
- A **bank** is a finite set of orders $`M`$. For attention, $`n=i-j`$ is the integer offset between query position $`i`$ and key position $`j`$.
- The **$`P`$-comb** is $`P\mathbb{Z}`$. For a $`P`$-periodic score $`f`$ with $`f(0)=1`$, the **comb margin** is

```math
\Delta(f) \;=\; 1-\max_{P\nmid n} f(n).
```

## 1. The ArithScope bias

For head $`h`$, with masks applied after the bias and before the softmax:

```math
\mathrm{logit}_h(i,j)=\frac{\langle q_i,k_j\rangle}{\sqrt{d}}\;+\;\gamma_h\sum_{m\in M}\alpha_{h,m}\,K_m(i-j),
\qquad \alpha_h=\mathrm{softmax}(\beta_h),\quad \gamma_h\in\mathbb{R},\ \gamma_h^{(0)}=0 .
```

$`q`$ and $`k`$ may be rotated by RoPE. With $`\gamma=0`$ the bias is exactly zero, so the model equals its base model; the tests check this bitwise, for outputs and gradients.

## 2. Kernel facts

**L1 (closed form; CLASSICAL, von Sterneck 1902, Hölder 1936).** With $`q=m/\gcd(m,n)`$,

```math
c_m(n)=\mu(q)\,\frac{\varphi(m)}{\varphi(q)},\qquad K_m(n)=\frac{\mu(q)}{\varphi(q)} .
```

So $`K_m`$ is even and $`m`$-periodic, depends only on $`\gcd(n,m)`$, and $`K_m(n)=1 \iff m\mid n`$.
*Check:* divisor formula = closed form = handoff oracle for $`m<120`$, $`|n|\le150`$.

**L1′ (PSD, unit diagonal, bounded; CLASSICAL).** Let $`\psi_m(t)=\big(e^{2\pi i a t/m}\big)_{\gcd(a,m)=1}/\sqrt{\varphi(m)}`$. Then $`K_m(i-j)=\langle\psi_m(i),\psi_m(j)\rangle`$. Consequently:

- every Gram matrix $`[K_m(t_r-t_s)]`$ is positive semidefinite;
- the diagonal is 1, and $`|K_m|\le1`$ by Cauchy–Schwarz;
- convex mixtures keep all three properties.

A negative $`\gamma`$ makes the *bias* negative semidefinite, which is allowed. **Not claimed:** anything about the attention matrix after softmax.
*Check:* the circulant spectrum is $`\frac{m}{\varphi(m)}[\gcd(h,m)=1]\ge0`$ ($`m<40`$); random mixtures have minimum eigenvalue $`\ge-10^{-10}`$.

**L2 (comb gap; PROVED).** Let $`m\ge2`$ and $`m\nmid n`$.

- $`K_m(n)\le\tfrac12`$, with the value $`\tfrac12`$ attained iff $`6\mid m`$; otherwise $`K_m(n)\le\tfrac14`$.
- If $`m=p^e`$ is a prime power, then $`K_m(n)\le0`$.
- For $`m=p`$ prime, $`\Delta(K_p)=p/(p-1)`$.

*Proof.* By L1, $`K_m(n)=\mu(q)/\varphi(q)`$, where $`q=m/\gcd(m,n)`$ is a divisor of $`m`$ with $`q>1`$. Every divisor $`q>1`$ occurs (take $`n=m/q`$). If $`\mu(q)\le0`$ the value is $`\le0`$. If $`\mu(q)=1`$, then $`q`$ is squarefree with an even number $`r\ge2`$ of prime factors, and $`\varphi(q)=\prod(p_i-1)`$:

- for $`r=2`$, $`\varphi(q)\ge(2-1)(3-1)=2`$, with equality iff $`q=6`$; every other pair gives $`\varphi(q)\ge4`$;
- for $`r\ge4`$, $`\varphi(q)\ge1\cdot2\cdot4\cdot6`$.

So $`\mu(q)/\varphi(q)\le\tfrac12`$, with equality iff $`q=6`$, and $`\le\tfrac14`$ otherwise. For $`m=p^e`$, the divisors $`q>1`$ are $`p^j`$, with $`\mu(p)=-1`$ and $`\mu(p^j)=0`$ for $`j\ge2`$. ∎
*Check:* exact for all $`m<400`$, including the $`\tfrac14`$ refinement.

## 3. Optimal periodic attention combs

**T3 (optimal positive-definite comb; PROVED).** Let $`P\ge2`$. Among all $`P`$-periodic positive-definite $`f:\mathbb{Z}\to\mathbb{R}`$ with $`f(0)=1`$,

```math
\Delta(f)\;\le\;\frac{P}{P-1},\qquad\text{with equality iff}\qquad f=f^\star_P:=\frac{P\,[P\mid n]-1}{P-1}=\sum_{d\mid P,\ d>1}\frac{\varphi(d)}{P-1}\,K_d .
```

The unique optimum is thus a **convex mixture of ArithScope kernels**, which an ArithScope head whose bank contains the divisors of $`P`$ can represent. For prime $`P`$, $`K_P`$ alone is optimal.

*Proof.* Write $`f(n)=\sum_{h\bmod P}\hat f(h)e^{2\pi i h n/P}`$.

1. *Upper bound.* Positive-definiteness on $`\mathbb{Z}/P\mathbb{Z}`$ is equivalent to $`\hat f\ge0`$, and $`f(0)=\sum_h\hat f(h)=1`$. The mean of $`f`$ over a period is $`\hat f(0)\ge0`$, so $`\sum_{n=1}^{P-1}f(n)=P\hat f(0)-1\ge-1`$. The maximum over the $`P-1`$ off-comb residues is at least their mean, hence at least $`-1/(P-1)`$, which gives $`\Delta(f)\le P/(P-1)`$.
2. *Uniqueness.* Equality forces $`\hat f(0)=0`$ and every off-comb value to equal $`-1/(P-1)`$, i.e. $`f=f^\star_P`$.
3. *Attainment.* $`f^\star_P=\frac{1}{P-1}\sum_{h=1}^{P-1}e^{2\pi i h n/P}`$ has nonnegative coefficients, so it is positive-definite.
4. *Ramanujan form.* Grouping the harmonics $`h`$ by $`d=P/\gcd(h,P)`$, i.e. by primitive $`d`$-th roots of unity, gives $`\sum_{h=1}^{P-1}e^{2\pi i h n/P}=\sum_{d\mid P,d>1}c_d(n)`$. Since $`c_d=\varphi(d)K_d`$ and $`\sum_{d\mid P,d>1}\varphi(d)=P-1`$, the weights form a convex combination. ∎

*Check:* the identity holds exactly for $`P<80`$. Independently, a linear program over all PSD $`P`$-periodic functions returns $`P/(P-1)`$ to $`4\cdot10^{-16}`$ for $`P\le36`$ (NUMERICAL, `arithscope/theory.py`).

**C3.1 (matched cosine banks are weaker).** The natural "same periods, same parameter count" Fourier control replaces each $`K_m`$ by its fundamental cosine $`\cos(2\pi n/m)`$.

- *Prime $`P`$ (PROVED).* A convex mixture that equals 1 on the $`P`$-comb can only use $`\cos(2\pi n/P)`$ and the constant. Its best margin is $`1-\cos(2\pi/P)=2\sin^2(\pi/P)`$. The ratio to the optimum, $`\frac{P/(P-1)}{2\sin^2(\pi/P)}`$, grows like $`P^2/(2\pi^2)`$.
- *Composite $`P`$ (NUMERICAL, LP).* The two banks coincide only for $`P\in\{2,3,4,6\}`$.

| period $`P`$ | 5 | 7 | 11 | 12 | 13 | 24 |
|---|---|---|---|---|---|---|
| optimum $`P/(P-1)`$ (ArithScope bank) | 1.250 | 1.167 | 1.100 | 1.091 | 1.083 | 1.043 |
| matched fundamental-cosine bank (LP) | 0.691 | 0.377 | 0.159 | 0.987 | 0.115 | 0.838 |
| single kernel $`K_P`$ | 1.250 | 1.167 | 1.100 | 0.500 | 1.083 | 0.500 |
| single cosine $`\cos(2\pi n/P)`$ | 0.691 | 0.377 | 0.159 | 0.134 | 0.115 | 0.034 |

The margin multiplies the logit scale $`\gamma`$: at a fixed logit budget, a larger margin means sharper attention on the comb.

**T4 (CRT comb composition; PROVED).** Let $`S`$ be a nonempty set of orders and $`\alpha`$ a probability vector supported on $`S`$. Then $`b=\sum_{m\in S}\alpha_mK_m`$ satisfies

```math
b(n)=1\iff \mathrm{lcm}(S)\mid n,\qquad b(n)\le 1-\min_{m\in S}\alpha_m\,\Delta(K_m)\ \text{ otherwise.}
```

Consequently:

1. A convex ArithScope head with bank $`M`$ can make exactly the combs $`\mathrm{lcm}(S)\,\mathbb{Z}`$, $`S\subseteq M`$, its strict argmax. These are its **selectable combs**.
2. If $`P=q_1\cdots q_k`$ with pairwise coprime prime powers $`q_i\in M`$, the uniform mixture over the $`q_i`$ selects $`P\mathbb{Z}`$ with margin $`\ge1/k`$, because L2 gives $`\Delta\ge1`$ for prime powers.

*Proof.* Each $`K_m\le1`$ and the weights are positive, so $`b(n)=1`$ iff every $`K_m(n)=1`$, iff every $`m\in S`$ divides $`n`$, iff $`\mathrm{lcm}(S)\mid n`$. Otherwise some $`m\in S`$ does not divide $`n`$, and then $`b(n)\le(1-\alpha_m)+\alpha_m(1-\Delta(K_m))`$. ∎
*Check:* exact for all $`P<300`$.

*Horizon of a bank.* Every $`K_m`$ with $`m\in M`$ is $`\mathrm{lcm}(M)`$-periodic. The prime-power bank $`B_X=\{p^e\le X\}`$ has $`\mathrm{lcm}(B_X)=\mathrm{lcm}(1,\dots,X)=e^{\psi(X)}`$ (Chebyshev). For example, $`B_{16}`$ has 10 orders, horizon 720 720, and folded rank 55. With ten parameters per head it selects every comb whose prime-power components are at most 16.

**P1 (alias partition; PROVED, CLASSICAL core).** Define the signature $`\sigma_M(n)=(K_m(n))_{m\in M}`$.

- Every bias in $`\mathrm{span}\{K_m:m\in M\}`$ is constant on each class $`\{n:\sigma_M(n)=\sigma\}`$.
- If $`M`$ is divisor-closed, $`\mathrm{span}\{K_m\}=\mathrm{span}\{[d\mid n]:d\in M\}`$, by Möbius inversion: $`c_m=\sum_{d\mid m}d\,\mu(m/d)[d\mid\cdot]`$ and $`[d\mid\cdot]=\frac1d\sum_{e\mid d}c_e`$.
- For $`M=\mathrm{divisors}(L)`$ the classes are exactly the gcd classes $`\{n:\gcd(n,L)=g\}`$: Cohen's even functions mod $`L`$ (E. Cohen, 1955–59).

Equivalently, an ArithScope head is invariant under $`n\mapsto un`$ for every unit $`u \bmod \mathrm{lcm}(M)`$. That invariance is its inductive bias.
*Check:* gcd classes for $`L\in\{12,18,24,30\}`$. For the E2 alias banks ([`PREREGISTRATION.md`](PREREGISTRATION.md)), the lcm-6 banks put offset 6 in the class of 0 and the lcm-12 banks do not. The P = 24 pilot banks are checked the same way.

## 4. Why exact periods are necessary for long-range periodic attention

**T5 (residue-blindness of incommensurate frequency banks; ASYMPTOTIC).** Let $`\theta\in\mathbb{R}^K`$, and let $`H`$ be the closure of $`\{n\theta \bmod 2\pi\}`$ in the torus $`\mathbb{T}^K`$. Assume that no integer vector $`c`$ has $`c\cdot\theta\in2\pi(\mathbb{Q}\setminus\mathbb{Z})`$; equivalently, $`H`$ is connected. Then for every continuous $`g:\mathbb{T}^K\to\mathbb{R}`$, every $`P\ge1`$ and every residue $`r`$:

```math
\lim_{N\to\infty}\frac{\sum_{0\le n<N,\ n\equiv r\ (P)} e^{g(n\theta)}}{\sum_{0\le n<N} e^{g(n\theta)}}=\frac1P .
```

*Application to RoPE.* With fixed content, a RoPE head's logit at offset $`n`$ is $`g(n\theta)=\sum_k\mathrm{Re}\big(z_ke^{in\theta_k}\big)`$ with $`\theta_k=b^{-2k/d}`$.

- For integer $`b\ge2`$ every $`\theta_k`$ is algebraic, so every $`c\cdot\theta`$ is algebraic.
- If $`c\cdot\theta=2\pi q`$ for some rational $`q\ne0`$, then $`\pi`$ would be algebraic, contradicting Lindemann (1882). So the hypothesis holds for every standard RoPE configuration.

**In the long-context limit, the attention mass a RoPE head puts on any residue class mod $`P`$ tends to the uniform share $`1/P`$.**

*Learned frequencies (corrected 2026-10-15).* T5 applies to any frequency vector that satisfies its **joint** hypothesis. It is not enough that each $`\theta_k/2\pi`$ is irrational; an earlier version of this file said so, and an external audit refuted it ([`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md), F3).

- *Counterexample.* Take $`\theta=(\sqrt2,\ \sqrt2+\pi)`$. Neither frequency is a rational multiple of $`2\pi`$, but $`\theta_2-\theta_1=\pi=2\pi\cdot\tfrac12`$. With $`g(n)=\cos n\theta_1+\cos n\theta_2=(1+(-1)^n)\cos n\sqrt2`$ and $`P=2`$, the even offsets carry weight $`e^{2\cos n\sqrt2}`$ (mean $`I_0(2)`$ by equidistribution) and the odd offsets weight 1, so the even-offset mass tends to $`I_0(2)/(I_0(2)+1)=0.69508\ldots`$, not $`1/2`$. *Check:* `tests/test_theory_notes.py` evaluates the sum over 200 000 offsets.
- *How often the hypothesis fails.* The frequency vectors that violate it lie on the hyperplanes $`c\cdot\theta=2\pi q`$, $`c\in\mathbb{Z}^K\setminus\{0\}`$, $`q\in\mathbb{Q}\setminus\mathbb{Z}`$: a countable union, hence a Lebesgue-null set. Every vector of rational numbers satisfies the hypothesis, because $`\pi`$ is irrational; in particular every floating-point frequency vector does, taken exactly. Nothing is claimed about which frequencies training produces.
- *What matters at finite length: near-resonances.* If $`c\cdot\theta=2\pi a/b+\varepsilon`$ with $`b\ge2`$ and small $`\varepsilon`$, then $`e^{inc\cdot\theta}`$ looks $`b`$-periodic for $`n\ll1/|\varepsilon|`$, and residue classes mod $`b`$ can keep unequal mass up to such offsets before the limit of T5 takes over. The drift remark below is the case $`K=1`$, $`c=1`$, $`a/b=1/P`$.

*Proof.*

1. *Connectedness.* $`H`$ is a compact abelian group with dual $`\mathbb{Z}^K/\mathrm{Ann}`$, where $`\mathrm{Ann}=\{c:c\cdot\theta\in2\pi\mathbb{Z}\}`$. A compact abelian group is connected iff its dual is torsion-free. Here that means $`jc\in\mathrm{Ann}\Rightarrow c\in\mathrm{Ann}`$, which is exactly the hypothesis.
2. *Equidistribution.* The sequence $`n\theta`$ is equidistributed in $`H`$ with respect to Haar measure (Weyl). Every nontrivial character $`\chi`$ of the monothetic group $`H`$ has $`\chi(\theta)\ne1`$, so the Cesàro means of $`\chi(n\theta)`$ vanish. Linear combinations of characters are uniformly dense in $`C(H)`$.
3. *Arithmetic progressions.* Let $`H_P`$ be the closed subgroup generated by $`P\theta`$; it contains $`PH`$. Since $`H`$ is a torus, multiplication by $`P`$ is onto, so $`H_P=H`$. Hence $`(r+kP)\theta`$ is equidistributed in $`r\theta+H=H`$.
4. *Limit.* Apply steps 2 and 3 to the positive continuous function $`e^{g}`$. Both normalized sums converge to the same Haar integral, and the count ratio tends to $`1/P`$. ∎

*Scope.*

- **Fixed content only.** T5 concerns the **positional channel with fixed content**. Content correlated with position (e.g. a periodic motif in the input) is outside its hypotheses. For i.i.d. content independent of position the same limit holds almost surely; this is a proof sketch, not given in full.
- **No rate, and the limit can be far away.** NUMERICAL (`results/summary/theory.json`):
  - Random RoPE coefficients give comb mass 0.0769 = 1/13 at $`N=2^{18}`$ for $`d=16`$, but still 0.109 for $`d=64`$, whose slowest frequencies have periods near 47 000 tokens.
  - A $`d=64`$ RoPE head optimized on a 64-token window with logit budget 32 keeps 0.984 comb mass up to $`N=2^{14}`$, then decays: 0.672 at $`2^{16}`$, 0.122 at $`2^{20}`$.
  - At budget 8, the same fit reaches only 0.286 even inside its window, while $`8K_{13}`$ holds 0.998 at every length.
- **Drift of learned frequencies.** For $`\omega=2\pi/P+\varepsilon`$ the comb phase drifts by $`n\varepsilon`$, so selectivity survives only for $`n\ll1/|\varepsilon|`$. NUMERICAL: $`\varepsilon=10^{-3}`$ loses most of the comb mass by $`n=1024`$.

**T6 (length-invariant comb mass; PROVED, elementary).** If the positional scores $`s`$ are $`P`$-periodic, then for every window $`N=kP`$ the comb mass equals $`e^{s(0)}/\sum_{r<P}e^{s(r)}`$, independent of $`k`$. This covers any ArithScope bank, and any fixed-period cosine bank, whose orders divide $`P`$. Exact periods turn positional extrapolation into an identity. They do not make content processing length-invariant; that part is EMPIRICAL.

**P3 (bounded confidence; PROVED, elementary; added 2026-10-15).** Let a model end in a LayerNorm (gain $`\gamma`$, bias $`\beta`$, width $`d`$) followed by a linear readout with rows $`w_j`$ and no bias, as `TinyDecoder` does. Then for every input and every target $`c`$,

```math
\mathrm{NLL}_c\;\ge\;\log\Big(1+\sum_{j\ne c}e^{-G_{cj}}\Big)\;>\;0,\qquad G_{cj}=\sqrt d\,\big\|\Pi\big((w_c-w_j)\odot\gamma\big)\big\|+(w_c-w_j)\cdot\beta,
```

where $`\Pi`$ subtracts the mean of a vector.
*Proof.* LayerNorm returns $`\gamma\odot z+\beta`$ with $`\sum_i z_i=0`$ and $`\|z\|<\sqrt d`$. Because $`z`$ is orthogonal to the all-ones vector, the logit gap $`(w_c-w_j)\cdot(\gamma\odot z+\beta)`$ equals $`\Pi((w_c-w_j)\odot\gamma)\cdot z+(w_c-w_j)\cdot\beta\le G_{cj}`$ (Cauchy–Schwarz). Finally, $`\mathrm{NLL}_c=\log\big(1+\sum_{j\ne c}e^{-(\ell_c-\ell_j)}\big)`$. ∎

In periodic denoising the Bayes NLL tends to 0 as votes accumulate, so no such model can follow it: every variant, with or without RoPE, has a positive NLL floor. *Check:* `experiments/posthoc_readout.py` finds the actual minimum over LayerNorm outputs for each diagnosed checkpoint (NUMERICAL; [`RESULTS.md`](RESULTS.md)).

### Finite-class convergence and the Bayes rate (added 2026-10-15)

T8, T8a, and B1 were proposed in an external review (the repository owner's ChatGPT workspace, 2026-10-15). They were re-derived here: T8's proof is written out in full, and B1's constant is tightened by a factor of 2. They replace the earlier proof sketch that followed P3.

**Setting.**
- The model is a `TinyDecoder` of finite depth with fixed weights.
- It has **no RoPE, no absolute positions, and no key windows**, and every additive bias depends on the offset only through $`(i-j)\bmod L`$ for a fixed $`L`$. For an ArithScope bank, $`L=\mathrm{lcm}(M)`$, since $`K_m`$ is $`m`$-periodic.
- LayerNorm has $`\epsilon>0`$, and attention is causal by token index.
- The data are periodic denoising with period $`P`$: given the motif $`\mu`$, the tokens $`X_0,X_1,\dots`$ are independent, and the law of $`X_j`$ depends only on $`j\bmod P`$.
- Let $`\Lambda=\mathrm{lcm}(L,P)`$. The **type** of position $`j`$ is $`\tau_j=(j\bmod\Lambda,\,X_j)`$, an element of a finite set $`\mathcal T`$.
- Let $`h^\ell_t`$ be the residual stream after block $`\ell`$ at position $`t`$ ($`h^0_t`$ is the embedding of $`X_t`$).

**T8 (finite-class convergence; ASYMPTOTIC).** Fix $`\mu`$. For every layer $`\ell`$ there is a function $`h_\ell:\mathcal T\to\mathbb R^d`$ such that, almost surely, $`h^\ell_t-h_\ell(\tau_t)\to0`$ as $`t\to\infty`$. Hence the logits, the predictive distribution, and the NLL at position $`t`$ approach functions of $`\tau_t`$. The mean NLL over positions $`[T_0,T)`$ converges almost surely, as $`T\to\infty`$ with $`T_0/T`$ fixed in $`[0,1)`$, to a constant $`\bar\ell(\mu)=\sum_\tau\pi_\mu(\tau)\,g_\mu(\tau)`$. Here $`\pi_\mu(\tau)`$ is the density of type $`\tau`$, and $`g_\mu(\tau)`$ is the limiting NLL of a position of that type.

*Proof.*
1. **Bounded states.** LayerNorm outputs are bounded ($`\|\gamma\odot z+\beta\|\le\sqrt d\,\|\gamma\|_\infty+\|\beta\|`$), so every query, key, value, attention logit (the biases take finitely many values), attention output, and MLP output is bounded uniformly in $`t`$. Hence all $`h^\ell_t`$ lie in a fixed compact set $`C`$. LayerNorm with $`\epsilon>0`$, the linear maps, GELU, and the softmax ratio below are continuous, hence uniformly continuous on compact sets.
2. **Induction.** The claim holds for $`\ell=0`$, since $`h^0_t`$ depends on $`X_t`$ only. Assume it for $`\ell-1`$. Fix one head of block $`\ell`$, and write $`u=\mathrm{LN}_1`$. Its logit for query $`t`$ and key $`j\le t`$ is
   ```math
   s_{tj}=\langle W_q u(h^{\ell-1}_t),W_k u(h^{\ell-1}_j)\rangle/\sqrt d+b\big((t-j)\bmod L\big),
   ```
   which is bounded by some $`S`$. The bias term depends on $`(t\bmod\Lambda,\,j\bmod\Lambda)`$ only. Let $`s^*(\tau,\sigma)`$ and $`v^*(\sigma)`$ be the same expressions with $`h_{\ell-1}`$ in place of $`h^{\ell-1}`$.
3. **Approximation.** Given $`\delta>0`$, the induction hypothesis and uniform continuity give a random $`J`$ such that $`|s_{tj}-s^*(\tau_t,\tau_j)|<\delta`$ and $`\|v_j-v^*(\tau_j)\|<\delta`$ for all $`t\ge j\ge J`$. The keys $`j<J`$ carry total attention weight at most $`Je^{2S}/(t+1)\to0`$.
4. **Limit of the average.** For the other keys, multiply the numerator and denominator of the softmax average by $`1/(t+1)`$ and group the keys by type. Each sum becomes $`\sum_\sigma\frac{N_\sigma(t)}{t+1}e^{s^*(\tau_t,\sigma)}(\cdot)`$, up to factors $`e^{\pm\delta}`$ and $`O(\delta)`$. Here $`N_\sigma(t)`$ counts the keys of type $`\sigma`$ in $`[J,t]`$. On each residue class mod $`\Lambda`$ the tokens are i.i.d., so by the strong law of large numbers $`N_\sigma(t)/(t+1)\to\pi_\mu(\sigma)`$ almost surely. The limiting denominator is at least $`e^{-S}>0`$, and $`\delta`$ is arbitrary. Therefore the head's output minus
   ```math
   A^*(\tau_t)=\frac{\sum_\sigma\pi_\mu(\sigma)e^{s^*(\tau_t,\sigma)}v^*(\sigma)}{\sum_\sigma\pi_\mu(\sigma)e^{s^*(\tau_t,\sigma)}}
   ```
   tends to 0. This is uniform over the finitely many query types.
5. **Rest of the block.** The output projection, residual additions, LayerNorm, and MLP are continuous, which completes the induction.
6. **Mean NLL.** The readout is continuous, and the target $`\mu_{t\bmod P}`$ is a function of $`\tau_t`$. So $`\mathrm{NLL}_t-g_\mu(\tau_t)\to0`$. The Cesàro mean of $`g_\mu(\tau_t)`$ converges by the strong law on each residue class. ∎

*What T8 says and does not say.*
- **Convergence per type, not to one constant.** The limit depends on the query's residue and on its own visible token, so the NLL sequence does not converge pointwise to a single constant; only its mean does.
- **Rates.** T8 gives no rate. A Hoeffding bound gives $`O_P(\sqrt{\log(|\mathcal T|/\delta)/t})`$ for the first layer, but deeper layers have dependent keys. Rates are OPEN.
- **Assumptions.** RoPE violates the assumption, because its logits depend on the offset itself, not on a residue. So do absolute positions.

**C8 (the floor; ASYMPTOTIC, from T8, P3, B1).** Under T8's setting, the mean NLL tends to $`\bar\ell(\mu)`$, which is at least the P3 floor and hence strictly positive. The Bayes NLL tends to 0 (B1). No such model's mean NLL follows the Bayes NLL at long lengths, whatever its weights.

This is a statement about a model class. It does not identify which part of a trained model sets the value of its floor; that is EMPIRICAL (E7, E8 in [`RESULTS.md`](RESULTS.md)).

*Remark (key windows; ASYMPTOTIC).* Suppose the last layer is limited to its last $`W`$ keys, as in the two-layer models of E6–E8, while the layers below it are unlimited.
- **The limit exists.** The layers below converge as in T8. The windowed layer's output at $`t`$ then approaches a continuous function $`G(t\bmod\Lambda,\,X_{t-W+1},\dots,X_t)`$. These variables are $`W`$-dependent and periodic in law, so splitting the positions into $`W\Lambda`$ interleaved subsequences of independent terms shows that the mean NLL again converges almost surely, to a limit $`F_\mu(W)`$.
- **No monotonicity is claimed.** For fixed weights, adding a key to a softmax average can raise or lower the NLL, so whether $`F_\mu(W)`$ is monotone in $`W`$ is OPEN. The windowed limit also lies above the P3 floor.

**T8a (softmax count non-identifiability; PROVED, elementary).** Fix class scores $`s_a`$ and values $`v_a`$, $`a=1,\dots,k`$. For counts $`c\in\mathbb R^k_{\ge0}`$ with $`\sum_ac_ae^{s_a}>0`$, let
```math
F(c)=\frac{\sum_a c_a e^{s_a}v_a}{\sum_a c_a e^{s_a}} .
```
Then $`F(\lambda c)=F(c)`$ for every $`\lambda>0`$, and $`DF(c)\,c=0`$.

*Proof.* The numerator and the denominator are both linear in $`c`$, so the factor $`\lambda`$ cancels. Differentiating $`F(\lambda c)`$ in $`\lambda`$ at $`\lambda=1`$ (Euler's identity for degree 0) gives the second statement. ∎

In T8's limit, every attention output has this form, with $`c`$ the type densities. So the output depends on class *proportions*, and the total count is a direction the aggregation cannot see. For example, the counts $`(8,2,0)`$ and $`(80,20,0)`$ give the same output, although the second carries ten times the evidence.

T8a concerns this aggregation map. It does not show that no Transformer can count: other paths, other heads, or an explicit length signal could carry the count. It does say what a remedy must add, namely a signal of evidence mass beside the normalized average.

**B1 (Bayes posterior and exponential NLL bound for periodic denoising; PROVED, elementary).**
- **Noise model.** Each visible token equals the motif symbol with probability $`1-\varepsilon`$ and is otherwise drawn uniformly from the $`V`$ symbols, so $`p=1-\varepsilon+\varepsilon/V`$ (correct symbol) and $`q=\varepsilon/V`$ (any given wrong symbol). Let $`\rho=p/q`$; with $`V=16`$ and $`\varepsilon=1/2`$, $`p=17/32`$, $`q=1/32`$, and $`\rho=17`$.
- **Posterior.** Under a uniform prior on the motif symbol, with $`n`$ comb observations and counts $`c_a`$ of each symbol,
  ```math
  \Pr(\mu_r=a\mid c)=\frac{\rho^{c_a}}{\sum_b\rho^{c_b}},\qquad\log\frac{\Pr(\mu_r=a\mid c)}{\Pr(\mu_r=b\mid c)}=(c_a-c_b)\log\rho .
  ```
  The log-odds grow with the count difference, not with the proportions.
- **Bound.** If the true symbol is $`c`$, then
  ```math
  \mathbb E\big[\mathrm{NLL}_{\mathrm{Bayes}}\big]\;\le\;(V-1)\,\kappa^{\,n},\qquad\kappa=1-(\sqrt p-\sqrt q)^2=2\sqrt{pq}+(V-2)q<1 .
  ```

*Proof.*
1. **Posterior.** The likelihood of the observations given $`\mu_r=a`$ is $`p^{c_a}q^{n-c_a}`$; normalize.
2. **NLL.** With $`D_b=N_b-N_c`$, the Bayes NLL is $`\log\big(1+\sum_{b\ne c}\rho^{D_b}\big)`$.
3. **Two inequalities.** $`\log(1+x)\le\sqrt x`$ for $`x\ge0`$ (the difference vanishes at 0 and has derivative $`(1-\sqrt x)^2/(2\sqrt x(1+x))\ge0`$), and $`\sqrt{\sum y_b}\le\sum\sqrt{y_b}`$. Hence $`\mathrm{NLL}\le\sum_{b\ne c}\rho^{D_b/2}`$.
4. **Expectation.** Each observation contributes the factor $`\rho^{-1/2}`$ with probability $`p`$, $`\rho^{1/2}`$ with probability $`q`$, and $`1`$ otherwise. So $`\mathbb E\rho^{D_b/2}=(\sqrt{pq}+\sqrt{pq}+(V-2)q)^n=\kappa^n`$, using $`p+(V-1)q=1`$. ∎

*Use.*
- **Values.** For $`V=16`$ and $`\varepsilon=1/2`$, $`\kappa=(14+2\sqrt{17})/32\approx0.6952`$. A query at position $`t`$ has $`n=\lfloor t/P\rfloor+1`$ comb observations, its own token included. At $`t=1{,}024`$ and $`P=13`$, the bound is below $`10^{-11}`$, which is why the Bayes NLL is 0 to machine precision there ([`RESULTS.md`](RESULTS.md)).
- **What the bound caps.** It also caps the evidence a window of $`W`$ keys gives up: about $`\lfloor W/P\rfloor`$ comb votes remain. It concerns the posterior's NLL, not plurality accuracy.

*Check:* `tests/test_theory_notes.py` computes the expected Bayes NLL exactly, by exact enumeration over count vectors, and compares it with the bound (B1). It also checks T8a's identities on random inputs. `tests/test_convergence.py` runs random ArithScope-only models with a bank of lcm 12 on 4,096-token sequences. It checks that the spread of same-type logits shrinks as $`t`$ grows (T8, NUMERICAL illustration). With the full bank $`\{1,\dots,16\}`$, $`\Lambda`$ is a multiple of 720,720, so a test of feasible length never sees a type twice.

### Evidence accumulation in an unnormalized readout (T9; added 2026-10-23)

T8a says a normalized average cannot see the total count, and P3 caps the confidence of a LayerNorm readout. T9 asks what an **unnormalized** sum of kernel-weighted token counts, added to the logits *after* the final LayerNorm, can do. Provenance: drafted by Claude Sonnet workers, audited adversarially by a separate review, revised (round 3), 2026-10-23. The full draft, with every proof and table, is [`research/evidence_readout/2026-10-23/theory/T9_DRAFT.md`](../research/evidence_readout/2026-10-23/theory/T9_DRAFT.md) (cited below as *the draft*); its checks are `theory/check_t9.py` (log `theory/verification/check_t9_run3.log`, all OK) and `theory/test_check_t9.py` (35 tests). The audit is `research/evidence_readout/2026-10-23/audit/REVIEW_T9.md`. Nothing here is a statement about a trained model.

**Setting and notation.**
- *Channel* (B1 with two letters renamed, because $`q`$ is the attention query and $`\kappa`$ is the kernel here): $`|\mathcal A|=V`$, $`p=1-\varepsilon+\varepsilon/V`$, $`\nu=\varepsilon/V`$ (B1's $`q`$), $`\rho=p/\nu`$, $`\lambda=\log\rho`$, $`\chi=1-(\sqrt p-\sqrt\nu)^2=2\sqrt{p\nu}+(V-2)\nu<1`$ (B1's $`\kappa`$). For $`V=16`$, $`\varepsilon=\tfrac12`$: $`p-\nu=\tfrac12`$, $`\rho=17`$, $`\chi=(14+2\sqrt{17})/32=0.695194`$.
- *Data (H-data).* A motif $`\mu\in\mathcal A^P`$; the tokens $`X_j`$ are independent with $`\Pr(X_j=a)=\pi_{j\bmod P}(a)`$, $`\pi_r(a)=p`$ if $`a=\mu_r`$, else $`\nu`$. The target of a query at index $`i`$ is $`c_i=\mu_{i\bmod P}`$, and it has $`n_i=\lfloor i/P\rfloor+1`$ **comb votes**. Where a posterior is mentioned the motif is also i.i.d. uniform (H-prior).
- *Head (H-head).* For a kernel $`\kappa:\mathbb Z\to\mathbb R`$ and $`W\in\mathbb R^{V\times V}`$,
  ```math
  E_i(a)=\sum_{0\le j\le i}\kappa(i-j)\,\mathbf 1[X_j=a],\qquad \ell_i=b_i+WE_i .
  ```
  $`b_i`$ is the rest of the network. **(H-b):** $`B=\sup|b_i(a)-b_i(a')|<\infty`$; by P3 this holds when the rest of the network ends in LayerNorm and a bias-free readout. The architectural hypothesis is that $`WE_i`$ is added *after* the last LayerNorm, so P3 does not bound it. Whether such a network is trainable is not addressed.
- *Comb and fold.* $`D_P(n)=\sum_{m\mid P}\frac{\varphi(m)}P K_m(n)`$. For an $`L`$-periodic $`\kappa`$ and $`\Lambda=\mathrm{lcm}(L,P)`$, the **fold** $`\kappa_P(s)`$ is the mean of $`\kappa`$ over each residue class $`s \bmod P`$, and $`e_t(a)=\frac1P\sum_s\kappa_P(s)\pi_{t-s}(a)`$.

**Lemmas (PROVED; proofs in the draft, §1).**
- **L9.1 (expected evidence).** For $`\kappa=\sum_c\beta_c\psi_c`$ with components of exact period $`m_c`$, $`\sup|\psi_c|\le G_c`$: $`|\mathbb E E_i(a)-(i+1)e_{i\bmod P}(a)|\le2\sum_c|\beta_c|G_c\,\mathrm{lcm}(m_c,P)`$. *Key step:* each component's sum over $`0\le n\le i`$ is full $`\mathrm{lcm}(m_c,P)`$-periods plus fewer than one period of bounded terms. *Size:* the constant is $`\le 26`$ for the comb itself ($`V=16`$) but $`1632`$ ($`P=12`$) and $`3224`$ ($`P=13`$) for the default bank with all orders at weight 1, so the lemma has no finite-$`i`$ content below roughly $`10^3`$–$`10^5`$ there; the true deviations are far smaller (at most $`7.6`$ in the checks). T9c.1, T9c.2 and T9d are limit statements and do not use the constant.
- **L9.2 (fold of a Ramanujan kernel; CLASSICAL core).** $`(K_m)_P=K_m`$ if $`m\mid P`$ and $`0`$ if $`m\nmid P`$. So $`(D_P)_P=\delta_0`$, and only the part of $`\kappa`$ that is $`P`$-periodic contributes to the linear rate. *Proof:* if $`m\nmid P`$ and $`g=\gcd(m,P)`$, summing $`c_m`$ over the coset $`s+g\mathbb Z/m`$ leaves $`\frac mg\sum_{a:\,(m/g)\mid a}e^{2\pi ias/m}`$, and no unit $`a`$ is divisible by $`m/g>1`$. The same holds for $`e^{2\pi ian/m}`$ and $`\cos(2\pi an/m)`$ with $`\gcd(a,m)=1`$ (L9.2′, a geometric sum). *Check:* exact for $`P\le24`$, $`m\le16`$; the right condition is $`m\nmid P`$, not coprimality ($`K_4`$ folds to $`0`$ at $`P=2`$).
- **L9.3 (noise).** For bounded $`\kappa`$, almost surely $`|E_i(a)-\mathbb EE_i(a)|\le G\sqrt{\tfrac32(i+1)\log(i+2)}`$ for all large $`i`$ and all $`a`$ (Hoeffding with $`x_i`$ giving probability $`2(i+2)^{-3}`$, then Borel–Cantelli). Hence, for $`L`$-periodic $`\kappa`$, $`E_i(a)=(i+1)e_{i\bmod P}(a)+O(\sqrt{i\log i})`$ a.s.

**T9a (count head with the comb kernel; PROVED; item 1 is CLASSICAL, DeltaNet study D4).** Take $`\kappa=D_P`$, $`W=\lambda I`$.
1. *Comb identity.* $`D_P=\mathbf 1[P\mid n]`$, with positive weights $`\varphi(m)/P`$ summing to $`1`$ (this is T3's identity before the shift by $`-1/(P-1)`$).
2. *Exact Bayes.* With $`b=0`$, $`E_i(a)=N_i(a)`$, the comb count of symbol $`a`$, and $`\mathrm{softmax}(\ell_i)_a=\rho^{N_i(a)}/\sum_{a'}\rho^{N_i(a')}`$. Under H-prior this is exactly $`\Pr(\mu_{i\bmod P}=a\mid X_{0..i})`$ for every $`i`$ and every realization.
3. *Bounded offset, expectation.* For any $`b`$ obeying H-b, with $`B^+=\sup(b_i(b)-b_i(c_i))^+`$: $`\mathbb E\,\mathrm{NLL}_i\le(V-1)e^{B^+/2}\chi^{\,n_i}`$.
4. *Sandwich and almost-sure rate.* Pointwise $`e^{-B}\mathrm{NLL}^{\rm Bayes}_i\le\mathrm{NLL}_i\le e^{B}\mathrm{NLL}^{\rm Bayes}_i`$, and almost surely $`\frac1{n_i}\log\mathrm{NLL}_i\to-(p-\nu)\log\rho`$, the Bayes rate, for every bounded $`b`$.
5. *Exponent of the expectation.* $`\mathbb E\,\mathrm{NLL}_i\ge\frac14\log(1+e^{-B})\,\chi^{n}\rho^{-\sqrt{2ns_Q^2}/2}`$ with $`s_Q^2=2\sqrt{p\nu}/\chi`$, hence $`\frac1{n_i}\log\mathbb E\,\mathrm{NLL}_i\to\log\chi`$. Only the prefactor of $`\chi^n`$ is not claimed sharp.

*Proof (key steps; full argument in the draft, §2).* (1) Group the harmonics $`h\in\mathbb Z/P`$ by $`m=P/\gcd(h,P)`$ to get $`c_m(n)`$; orthogonality gives $`P\,\mathbf 1[P\mid n]`$. (2) The likelihood of the votes at residue $`r`$ given $`\mu_r=a`$ is $`\nu^{n_i}\rho^{N_i(a)}`$; other residues contribute a factor free of $`a`$. (3) With $`D_b=N_i(b)-N_i(c_i)`$, $`\log(1+x)\le\sqrt x`$ and $`\sqrt{\sum y}\le\sum\sqrt y`$ give $`\mathrm{NLL}_i\le e^{B^+/2}\sum_b\rho^{D_b/2}`$ pathwise (so data-dependence of $`b_i`$ is harmless), and $`\mathbb E\rho^{D_b/2}=\chi^{n_i}`$ as in B1. (4) $`f(y)=\log(1+y)`$ is concave with $`f(0)=0`$, so $`e^{-B}f(x)\le f(x')\le e^Bf(x)`$ when $`e^{-B}x\le x'\le e^Bx`$; the strong law gives the Bayes rate. (5) Tilt the vote walk $`S=N(b_0)-N(c_i)`$ at $`e^{\theta_*}=\sqrt\rho`$ (the minimizer of the moment generating function, with value $`\chi`$): the tilted step is symmetric with variance $`s_Q^2`$, and Chebyshev plus symmetry give $`Q(0\le S\le\sqrt{2ns_Q^2})\ge\frac14`$. ∎
*Check:* the identity for $`P=2,\dots,32`$ and the exact Bayes equality as `Fraction`s on short sequences, in `tests/test_t9.py`. In the draft, the item-3 bound is checked in an enumeration with exact probabilities and floating-point logarithms (NUMERICAL), e.g. $`V=16`$, $`B^+=0`$: $`\mathbb E\,\mathrm{NLL}=0.356482`$ at $`n=6`$ against the bound $`1.693`$. The a.s. rate per vote is $`1.416607`$ ($`V=16`$), the exponent of the expectation $`-\log\chi=0.363564`$: they differ because the expectation is dominated by rare plurality-wrong events.
*What it does not say.* It does not say a trained model reaches this; $`\rho`$ and $`P`$ are assumed known to the construction; H-b is a hypothesis on the rest of the network, not a consequence of the head; independence of the votes is essential (T9e). It does not contradict P3/C8: P3 bounds only $`b`$, and an unnormalized term outside the LayerNorm can carry unbounded evidence.

**T9b (expressibility; PROVED).**
- **T9b.1 (span criterion).** $`D_P\in\mathrm{span}_{\mathbb R}\{K_m:m\in M\}\iff`$ every divisor of $`P`$ lies in $`M`$. The $`K_m`$ are linearly independent, so the coefficients are unique ($`\varphi(m)/P`$ on divisors, $`0`$ elsewhere). *Proof:* $`c_m`$ is a sum of characters $`e^{2\pi ixn}`$ with $`x`$ of exact denominator $`m`$; different orders have disjoint nonempty character sets, and distinct characters are independent (Dedekind–Artin). "If" is T9a.1. "Only if": the representation is unique, so $`\beta_m=\varphi(m)/P>0`$ forces $`m\in M`$. ∎
  *Consequences.* For the default bank $`\{1,\dots,16\}`$ the reachable periods are $`P\le16`$ ($`P=12`$, $`13`$ yes; $`17`$, $`24`$ no). $`D_{13}=\frac1{13}K_1+\frac{12}{13}K_{13}`$ needs the **constant** $`K_1`$, so $`B_{16}`$ (10 orders, no order 1) does not contain $`D_{13}`$ and $`B_{16}\cup\{1\}`$ does. If the evidence kernel reuses the softmax-simplex weights of §1 (all strictly positive), then $`D_P`$ is reached exactly only when the bank equals the divisor set of $`P`$; which parametrization an evidence head uses is not decided here. For a finite context $`N<L`$ the "if" direction persists and "only if" is not claimed.
  *Check:* the criterion and independence for four banks and $`P=1,\dots,40`$ by exact rational rank, and a rank check on evaluated kernels over $`\mathbb Z/12`$ (`tests/test_t9.py`; the draft's sweep has 156 cases).
- **T9b.2 (bank design: exact counting needs a divisor-closed bank; selection needs only the lcm).** For $`M=B_{16}\cup\{1\}`$, $`D_P`$ is in the span exactly for the prime powers $`\le16`$ among $`P\le40`$: $`2,3,4,5,7,8,9,11,13,16`$; $`D_{12}`$ is not (divisors $`6`$, $`12`$ are missing). T4 nevertheless selects the comb $`\mathrm{lcm}(S)\mathbb Z`$ for every $`S\subseteq M`$ (so $`P=12`$ is *selectable* but not *countable*). The reason it matters: for $`S=\{3,4\}`$, $`b=\frac12(K_3+K_4)`$ equals $`1`$ only on $`12\mid n`$ but has off-comb mass $`\sum_{s\ne0}|b(s)|=4`$ on residues $`1..11`$; a softmax with large gain suppresses it, an unnormalized sum counts it. By T9c.5 (below) with $`V=3`$, $`\varepsilon=\frac12`$, $`W=\lambda I`$, the motif $`(0,0,0,1,1,0,0,0,1,1,0,0)`$ and query residue $`0`$ has slope $`\frac{p-\nu}{12}(1-4)=-\frac18`$ per index in units of $`\lambda`$: the wrong symbol wins and the NLL grows linearly. More strongly, over all of $`\mathrm{span}\{K_1,K_2,K_3,K_4\}`$ (the orders of $`M`$ dividing $`12`$) with $`f(0)=1`$ the minimum of $`\sum_{s\ne0}|f(s)|`$ is exactly $`2>1`$ (exact vertex enumeration, draft §3), so no coefficient choice is dominant; the CRT choice is the best two-order mixture, not the best element of the span. *Proof:* T9b.1 plus the values listed. *Check:* the off-comb values on residues $`0..11`$, their $`\ell^1`$ mass $`4`$, and the slope $`-\frac18`$, exactly, in `tests/test_t9.py`; the vertex enumeration is in the draft's check (the cosine spans in it are NUMERICAL).
  *What it does not say.* CRT banks are not bad for softmax attention (T4 stands). "Dominant" is a worst-motif criterion; a non-dominant kernel can still be positive on almost every random motif (T9c.6).
- **T9b.3 (cosine banks).** Let $`\mathcal C=\mathrm{span}\{1,\cos(2\pi n/d):d\ge2\}`$ (fundamental cosines of any orders). Then $`D_P\in\mathcal C\iff\varphi(P)\le2\iff P\in\{1,2,3,4,6\}`$. So $`D_P\notin\mathrm{span}\{1,\cos(2\pi n/P)\}`$ for every prime $`P\ge5`$, and the matched bank $`\{1\}\cup\{\cos(2\pi n/d):d\mid P,d>1\}`$ contains $`D_P`$ only for $`P\in\{1,2,3,4,6\}`$ (the set of C3.1). For $`P=12`$ the harmonic $`a=5`$ is missing, although $`D_{12}\in\mathrm{span}\{\cos(2\pi an/12):a=0,\dots,6\}`$. *Proof:* $`D_P`$ has spectrum $`\{h/P\}`$ with every coefficient nonzero, elements of $`\mathcal C`$ have spectrum in $`\{0\}\cup\{\pm1/d\}`$, and characters are independent; so each $`h/P`$ must be $`\pm1/d`$, which for $`g=\gcd(h,P)`$ means $`h\in\{g,P-g\}`$. Units ($`g=1`$) must lie in $`\{\pm1\}`$, i.e. $`\varphi(P)\le2`$, and each prime-power factor then has $`p^e\in\{2,3,4\}`$, giving $`P\in\{1,2,3,4,6\}`$; for these $`P`$ all spectra have the required form. ∎ The Ramanujan kernel $`K_m=\varphi(m)^{-1}\sum_{(a,m)=1}\cos(2\pi an/m)`$ contains all unit harmonics at once. *Check:* exact rank of Fourier-coefficient vectors, $`P=1,\dots,40`$, orders up to $`48`$, and the $`P=12`$ harmonics, in `tests/test_t9.py` (the draft adds the harmonic criterion for $`P\le60`$). *What it does not say.* Nothing about trained cosine banks with free frequencies (T9d), about the LP margin of C3.1 (a different, NUMERICAL, quantity), or about the best approximation of $`D_P`$ by cosines.
- **T9b.4 (off-comb leakage in an unnormalized sum).** Let $`\kappa=(1-\delta)D_P+\delta K_1`$ ($`P`$-periodic, realizable in any bank containing the divisors of $`P`$), $`W=\lambda I`$, the motif with symbol $`\alpha`$ at residue $`0`$ and $`\beta\ne\alpha`$ at the other $`P-1`$ residues, query residue $`0`$. Almost surely
  ```math
  \frac{\ell_i(\alpha)-\ell_i(\beta)}{i}\to\frac{\lambda(p-\nu)}{P}\big(1-(P-1)\delta\big),
  ```
  negative iff $`\delta>1/(P-1)`$; then $`\beta`$ beats the truth and $`\mathrm{NLL}_i`$ grows linearly. *Proof:* L9.1 with $`\kappa_P=\kappa`$: the comb contributes $`(p-\nu)`$ and each of the $`P-1`$ off-comb residues $`-\delta(p-\nu)`$. ∎ *Exact numbers* ($`V=16`$, $`\varepsilon=\frac12`$, units of $`\lambda`$): $`P=13`$, $`\delta=\frac1{10}`$: $`-\frac1{130}`$; $`\delta=\frac1{12}`$: $`0`$; $`\delta=\frac1{13}`$: $`+\frac1{338}`$; $`\delta=0`$: $`+\frac1{26}`$; $`P=3`$, $`\delta=\frac34`$: $`-\frac1{12}`$. The exact expected gap at $`i=1300`$ ($`P=13`$, $`\delta=\frac1{10}`$) is $`-\frac{19}2`$. *Contrast with softmax:* a row-wise constant is invisible there (T8a), but in a sum $`K_1`$ multiplies the symbol-dependent indicator and counts the whole-context histogram. *Check:* all of the above, exactly, in `tests/test_t9.py`. *What it does not say.* The motif is adversarial; for random motifs the leakage has a nonzero mean but need not be negative (T9c.6). Nothing here says leakage occurs in a trained head.

**T9c (exactly periodic kernels; PROVED).** Assume H-data, H-head, H-b; $`\kappa`$ is $`L`$-periodic, $`W`$ fixed. For a residue $`t`$ with $`c=\mu_t`$ let $`\ell_t=We_t`$, $`s_t(c,b)=\ell_t(c)-\ell_t(b)`$ and $`s_t=\min_{b\ne\mu_t}s_t(\mu_t,b)`$.
1. *Linear rate.* Almost surely, along $`i\equiv t\ (P)`$, $`\ell_i(a)-\ell_i(b)=(i+1)(\ell_t(a)-\ell_t(b))+O(\sqrt{i\log i})`$.
2. *Exact exponential rate.* Almost surely $`\frac1i\log\mathrm{NLL}_i\to-\max(0,s_t)`$ along that class. So the NLL decays exponentially along every class iff $`s_t>0`$ for every $`t`$; if some $`s_t(\mu_t,b)<0`$ then $`\mathrm{NLL}_i\ge|s_t(\mu_t,b)|\,i\,(1-o(1))`$.
3. *Expectation.* If $`\min_ts_t>0`$, then $`\mathbb E\,\mathrm{NLL}_i\le Ce^{-ci}`$ for all $`i`$ (Hoeffding with exponent $`s_t^2i/(16R^2)`$; the constants inherit L9.1's size).
4. *Comb kernel.* For $`D_P`$, $`W=\lambda I`$: $`s_t=\lambda(p-\nu)/P`$ for every motif and residue, the Bayes rate of T9a.4 per index.
5. *Dominance criterion ($`W=I`$, $`V\ge2`$).* $`s_t(\mu_t,b)=\frac{p-\nu}P\big(\kappa_P(0)+\sum_{s\ne0}\kappa_P(s)\varepsilon_s\big)`$, $`\varepsilon_s=\mathbf 1[\mu_{t-s}=\mu_t]-\mathbf 1[\mu_{t-s}=b]`$. All margins are positive **for every motif** iff $`\kappa_P(0)>\sum_{s\ne0}|\kappa_P(s)|`$. This is a **worst-motif** criterion, sufficient for a positive rate on every motif; it does not say a non-dominant fold fails on typical motifs (T9c.6). *Example:* for $`\kappa=\beta_1K_1+\beta_PK_P`$, $`P`$ prime, strict dominance holds iff $`\beta_1>0`$ and $`\beta_P>\frac{P-2}2\beta_1`$ (so the constant coordinate is needed).
6. *Random motifs.* Let $`k_0=\kappa_P(0)`$, $`\sigma=\sum_s\kappa_P(s)`$ and $`M_a(t)=\sum_{r:\mu_r=a}\kappa_P(t-r)`$.
   - (a) $`e_t(a)=\frac1P(\nu\sigma+(p-\nu)M_a(t))`$. For $`W=\lambda I`$, or $`W=(d-o)I+oJ`$ with $`d>o`$, all residues have a positive margin iff each $`\mu_t`$ is the strict argmax of $`M(t)`$.
   - (b) Over uniformly random motifs the probability $`q`$ that every residue is positive is: $`1`$ for a dominant fold (in particular $`D_P`$); $`V^{1-P}`$ for the constant fold $`\kappa_P\equiv1`$; and, for the leakage fold $`(1-\delta)D_P+\delta K_1`$, the event $`\delta(h_{\max}-h_{\min})<1-\delta`$ on the symbol counts $`h_a`$ of the motif ($`h_{\min}`$ over present symbols), a finite sum over partitions of $`P`$. Exact values of $`1-q`$ for $`V=16`$ (draft §4, `Fraction` partition sum): $`\delta=\frac1{10}`$: $`8.0\cdot10^{-10}`$ ($`P=12`$), $`3.5\cdot10^{-9}`$ ($`P=13`$); $`\delta=\frac15`$: $`8.3\cdot10^{-3}`$, $`1.28\cdot10^{-2}`$; $`\delta=\frac13`$: $`0.489`$, $`0.577`$; $`\delta=\frac1{20}`$: $`0`$. Compare T9b.4: at $`P=13`$, $`\delta=\frac1{10}`$ the adversarial motif has a negative slope, yet a random $`V=16`$ motif fails with probability $`3.5\cdot10^{-9}`$.
   - (c) *Necessary collisions.* Let $`n_{\min}`$ be the least $`j`$ such that the $`j`$ largest $`|\kappa_P(s)|`$, $`s\ne0`$, sum to at least $`k_0`$ ($`\infty`$ iff dominant) and $`T=\Pr(\sum_{s\ne0}|\kappa_P(s)|B_s\ge k_0)`$ with $`B_s`$ i.i.d. Bernoulli$`(1/V)`$. If $`k_0>0`$, $`1-q\le P(V-1)T\le P(V-1)\Pr(\mathrm{Bin}(P-1,\tfrac1V)\ge n_{\min})`$.
   - (d) A matching lower bound on $`1-q`$ is not proved.
7. *Zero-mean kernels.* If $`\sum_s\kappa_P(s)=0`$ (by L9.2 the order-1 component vanishes), a constant motif has $`s_t=0`$ for every $`W`$; so $`K_P`$ ($`P`$ prime) and the simplex comb $`f^\star_P`$ alone give no linear rate there.

*Proof (key steps; full argument in the draft, §4).* (1) L9.1 and L9.3, one almost-sure event for all $`a`$ and $`W`$, with $`b_i`$ bounded. (2) If $`s_t>0`$, $`\log(1+x)\in[x/2,x]`$ for the small $`x=\sum_be^{-m_b}`$ and $`\frac1i\log x\to-s_t`$; if $`s_t\le0`$, $`\mathrm{NLL}_i\ge\log(1+e^{-\epsilon i})\ge\frac12e^{-\epsilon i}`$ for every $`\epsilon>0`$, and $`\mathrm{NLL}_i=O(i)`$. (3) Hoeffding on the margin, bound on its complement, sum. (4) $`(D_P)_P=\delta_0`$. (5) $`\pi_r(c)-\pi_r(b)=(p-\nu)(\mathbf 1[\mu_r=c]-\mathbf 1[\mu_r=b])`$; sufficiency is the triangle inequality, and for necessity choose $`\mu_{t-s}=b`$ if $`\kappa_P(s)>0`$ and $`c`$ otherwise (two symbols suffice). (6a) $`\pi_r(a)=\nu+(p-\nu)\mathbf 1[\mu_r=a]`$; $`W\mathbf 1`$ constant cancels the $`\sigma`$ term. (6b) Dominant: (5). Constant fold: $`M_a=h_a`$ for all $`t`$, so all $`\mu_t`$ are the unique argmax. Leakage fold: $`M_a(t)=(1-\delta)\mathbf 1[a=\mu_t]+\delta h_a`$. (6c) Condition on $`\mu_t=c`$; the margin against $`b`$ is $`\le0`$ only if the independent uniform symbols at the other residues realize signs worth at least $`k_0`$; union over $`P`$ residues and $`V-1`$ competitors. (7) For a constant motif $`\pi_r\equiv\pi`$, so $`e_t=\frac1P\sum_s\kappa_P(s)\pi=0`$. ∎
*Check:* `tests/test_t9.py` covers the comb identity (T9c.4 rests on it) and the slope formula in the T9b.4 and T9b.2 cases. In the draft's check, exact: 240 random rational folds ($`V=3`$, $`P\in\{3,4\}`$, all $`3^P`$ motifs): positivity for all motifs coincides with strict dominance, with both outcomes occurring; the slope formula of 6(a) on 10 386 triples; the enumeration $`q=V^{1-P}`$, $`q=1`$ and the partition formula; the bound 6(c) against the enumerated failure probability; the two-order family of 5, 48 cases.

**T9c.6-num (NUMERICAL; Monte Carlo over uniformly random motifs, $`V=16`$, fixed seed $`20261023`$).** The "pilot" folds are the development pilot's folds as computed by the audit (one training seed; nothing re-trained). Failures of the sampled motifs: $`D_{12}`$, pilot `arith` and `cosine_full` at $`P=12`$ and at $`P=13`$: $`0`$ of $`20\,000`$; pilot `cosine_fund`: $`16`$ of $`100\,000`$ at $`P=12`$ (rigorous bound of 6(c): $`1.4\cdot10^{-4}`$; the Wilson interval contains it) against $`19\,987`$ of $`20\,000`$ at $`P=13`$ (bound vacuous); pilot `uniform` and `cosine_learned`: $`20\,000`$ of $`20\,000`$ at both periods. The $`L^1`$-optimal `cosine_fund` fold at $`P=12`$ (dominance margin exactly $`0`$, an antipodal tie) fails on every sampled motif, so the best dominance margin is not the best fold for random motifs. Full table in the draft, §4. This separates `cosine_fund` at $`P=12`$ from $`P=13`$, which T9c.5 does not.

**T9d (non-resonant kernels; ASYMPTOTIC as $`i\to\infty`$; nothing at finite $`i`$).** Let $`\theta\in\mathbb R^K`$ satisfy T5's joint hypothesis, $`H`$ the closure of $`\{n\theta\}`$, $`g`$ continuous on $`\mathbb T^K`$, $`\kappa(n)=g(n\theta)`$, $`\bar\kappa=\int_Hg\,dm_H`$. Fix $`P`$ and $`\mu`$.
1. *Deterministic part.* $`\frac1i\sum_{n\le i,\,n\equiv\varrho\,(P)}g(n\theta)\to\bar\kappa/P`$ for every class $`\varrho`$, so $`\mathbb E E_i(a)/i\to\bar E(a)=\bar\kappa(\nu+(p-\nu)h_a/P)`$ along **all** $`i`$, $`h_a=\#\{r:\mu_r=a\}`$.
2. *Linear part.* Almost surely $`E_i(a)/i\to\bar E(a)`$ (with L9.3); the query residue does not appear in the limit.
3. *No exponential rate on some class.* If the motif has two distinct symbols $`\mu_{r_1}\ne\mu_{r_2}`$, there is a probability-one event, independent of $`W`$ and of $`b`$, on which for **every** $`W`$ and every $`b_i=o(i)`$ (possibly data-dependent) some class $`r^*\in\{r_1,r_2\}`$ has $`\ell_i(c^*)-\ell_i(c')\le\epsilon i`$ eventually for every $`\epsilon>0`$, hence $`\mathrm{NLL}_i\ge\frac12e^{-\epsilon i}`$: no bound $`Ce^{-ci}`$, $`c>0`$, holds on that class. With $`\Delta=\ell(c_1)-\ell(c_2)`$, $`\ell=W\bar E`$: $`r^*=r_1`$ if $`\Delta\le0`$, $`r_2`$ if $`\Delta\ge0`$; if $`\Delta<0`$ the wrong symbol's margin grows linearly.

*Proof.* By T5 steps 2–3, $`(\varrho+kP)\theta`$ is equidistributed in the connected $`H`$ (Weyl), so each class has mean $`\bar\kappa`$ and density $`1/P`$; rewrite $`\mathbb EE_i(a)=\sum_r\pi_r(a)\sum_{n\equiv i-r}g(n\theta)`$. The margins at $`r_1`$ and $`r_2`$ are $`\pm\Delta i+o(i)`$, one of which is $`\le0`$; then use $`\mathrm{NLL}_i\ge\log(1+e^{-m})`$. ∎ (Full proof: draft, §5.)
*Quantifiers and hypotheses.* "Almost surely" refers to the token noise with $`\mu,\theta,g`$ fixed; Weyl gives no rate. A constant motif is learnable by a non-resonant kernel when $`\bar\kappa>0`$, so the two-symbol hypothesis is essential. An exactly periodic $`\kappa`$ is outside T5's hypothesis and falls under T9c: for example $`\theta=\frac{2\pi}{13}(1+10^{-3})`$ has $`\theta/2\pi=\frac{77}{1000}`$, so $`\kappa`$ is $`1000`$-periodic and its fold to modulus $`13`$ is the constant $`\frac12`$ (L9.2), which is not a T9d instance.
*Finite-window remark (closed form PROVED; the regime it describes is outside T9d).* For $`\kappa(n)=\cos(n\theta)`$, $`\theta=2\pi a/P+\eta`$, $`N=PU`$, the window fold is $`\hat\kappa_{P,N}(s)=D_U(P\eta)\cos\!\big(\frac{2\pi as}P+\eta(s+\frac{P(U-1)}2)\big)`$ with the normalized Dirichlet kernel $`D_U`$, and $`|D_U(P\eta)|\le\pi/(N|\eta|)`$ (geometric sum over each class). So for $`N|\eta|\ll1`$ the window fold is that of the periodic cosine (T9c's regime) and the constant fold of T9d is reached only for $`N\gg1/|\eta|`$, the *drift horizon*; a bank adds linearly, so the horizon is that of the heaviest near-resonant component. Checked against brute force for 90 random cases (NUMERICAL, draft). The bank-level statement with rates in the amplitudes is OPEN.
*What it does not say.* A single-layer evidence head on raw tokens with a fixed, content-independent kernel; nothing about deeper networks, content-dependent kernels, shared $`W`$ across several kernels, or which frequencies training finds; nothing at finite $`i`$. It does not say non-resonant kernels are useless: with $`\bar\kappa\ne0`$ they count the whole-context histogram.

**T9d-num (NUMERICAL; one seed, no sampling-noise estimate).** $`P=13`$, $`V=16`$, a 13-symbol motif, one sequence of length $`2^{16}+1`$, evidence gap $`(E_i(0)-E_i(1))/i`$ at query residue $`0`$. At $`i=65\,533`$ the comb $`D_{13}`$ has gap $`0.038`$ (limit $`(p-\nu)/13=0.03846`$), while the three constant-fold kernels (period 1000, and two irrational frequencies) have $`|{\rm gap}/i|=2.3,\,4.4,\,4.7\cdot10^{-4}`$ against a limit of $`0`$. Only the order-of-magnitude contrast is read from the table (draft, §5).

**T9e (dependence breaks the Bayes equality; PROVED; the table in the draft is NUMERICAL).** At the query's residue, $`m`$ independent noisy observations $`Y_u`$ of the true symbol are each **copied $`k`$ times** at comb positions, so the comb count is $`kM_a`$, $`M_a=\#\{u:Y_u=a\}`$. The Bayes posterior is $`\varpi_a\propto\rho^{M_a}`$; the $`D_P`$ count head with $`W=\lambda I`$ outputs $`\varpi^{\,k}/\sum_a\varpi_a^k`$. Write $`D_b=M_b-M_c`$.
1. *Power law.* The head has the Bayes argmax, so accuracy is unchanged.
2. *Excess loss.* $`\mathbb E\,\mathrm{NLL}_{\rm head}-\mathbb E\,\mathrm{NLL}_{\rm Bayes}=\mathbb E\,\mathrm{KL}(\varpi\,\|\,\mathrm{head})`$, strictly positive for every $`m\ge1`$, $`k\ge2`$.
3. *Bounds.* $`k\lambda\,\mathbb E[(\max_bD_b)^+]\le\mathbb E\,\mathrm{NLL}_{\rm head}\le k\,\mathbb E\,\mathrm{NLL}_{\rm Bayes}\le k(V-1)\chi^m`$. For fixed $`m`$ the expected loss is unbounded in $`k`$ (with $`\mathbb E\,\mathrm{NLL}_{\rm head}/k\to\lambda\,\mathbb E[(\max_bD_b)^+]`$, error $`\le\log V/k`$), while for fixed $`k`$ it still decays exponentially in $`m`$.

*Proof.* (1) $`x\mapsto x^k`$ is monotone. (2) Given $`Y`$ the truth has law $`\varpi`$, so $`\mathbb E[-\log h(c)\mid Y]=H(\varpi)+\mathrm{KL}(\varpi\|h)`$; the KL vanishes iff $`\varpi^{k-1}`$ is constant iff $`\varpi`$ is uniform, which has probability $`<1`$. (3) $`\mathrm{NLL}_{\rm head}=\log(1+\sum_b\rho^{kD_b})\ge\max(0,k\lambda\max_bD_b)`$; and $`(1+\sum x_b)^k\ge1+\sum x_b^k`$ gives $`\log(1+\sum x_b^k)\le k\log(1+\sum x_b)`$, then B1. ∎
*Check, hand case* ($`V=3`$, $`\varepsilon=\frac12`$, $`m=1`$, $`k=2`$; $`p=\frac23`$, $`\nu=\frac16`$, $`\rho=4`$): Bayes $`\mathbb E\,\mathrm{NLL}=\frac23\log\frac32+\frac13\log6=0.867563`$, head $`\frac23\log\frac98+\frac13\log18=1.041979`$, excess $`+0.174416`$; the argmax is correct with probability $`\frac23`$ for both, the head's mean top probability is $`\frac89`$ against Bayes' $`\frac23`$ (facts about this enumeration only). Reproduced to $`10^{-6}`$, with excess $`=\mathbb E\,\mathrm{KL}`$, in `tests/test_t9.py`. The draft tabulates 20 rows ($`V\in\{3,16\}`$); the excess is not monotone in $`m`$ for $`V=3`$ and no explanation or claim is offered.
*What it does not say.* Arithmetic on an explicit idealized law (exact copies at comb positions, known $`k`$ for the reference); nothing about a trained model or weaker dependence; not a calibration statement. It does not show the count head is unusable: what fails is the Bayes equality and a uniform-in-$`k`$ guarantee.

**What T9 does not say.**
- Nothing about learnability: whether training finds $`D_P`$, $`W=\lambda I`$, $`P`$ or $`\rho`$, or any trained rate. That belongs to the preregistered experiments.
- Only the periodic denoising law of H-data and the copy law of T9e; no natural-data claim.
- No speedup, sub-quadratic or linear-attention claim, and no statement about the cost of computing $`E_i`$.
- It does not extend T8/C8 to networks containing this head; the rest of the network is only assumed bounded (H-b).
- Unnormalized counts of earlier tokens are classical (cache language models, trigger features); the kernel $`D_P`$ is this repository's object.

*Empirical counterpart:* E10 ([`PREREGISTRATION.md`](PREREGISTRATION.md) §13; [`RESULTS.md`](RESULTS.md)). The measured numbers are produced by `experiments/analyze.py` and are not repeated here.

## 5. Implementation identities

**T7 (exact low-rank folding; PROVED).** Define real features as follows:

- $`u_1(t)=1`$ and $`u_2(t)=(-1)^t`$;
- for $`m>2`$, $`u_m(t)=\sqrt{2/\varphi(m)}\,\big(\cos\tfrac{2\pi a t}{m},\ \sin\tfrac{2\pi a t}{m}\big)_{\gcd(a,m)=1,\ a<m/2}`$.

Then $`K_m(i-j)=\langle u_m(i),u_m(j)\rangle`$, hence

```math
\frac{\langle q_i,k_j\rangle}{\sqrt d}+\gamma_h\sum_m\alpha_{h,m}K_m(i-j)=\frac{1}{\sqrt d}\Big\langle\big[q_i,\ \sqrt d\,(\gamma_h\alpha_{h,m}u_m(i))_m\big],\ \big[k_j,\ (u_m(j))_m\big]\Big\rangle .
```

The bias therefore runs inside a fused attention kernel. It adds $`\sum_{m\in M}\varphi(m)`$ dimensions to queries and keys (80 for $`M=\{1,\dots,16\}`$, 55 for $`B_{16}`$) and never builds an $`N\times N`$ bias tensor. The rank $`\sum\varphi(m)`$ is minimal once $`N\ge\mathrm{lcm}(M)`$, because distinct reduced fractions $`a/m`$ give linearly independent characters. Phases use the exact residue $`(a\,(t\bmod m))\bmod m`$.
*Check:* folded and dense biases agree to $`10^{-12}`$ in float64, including positions near $`10^{15}`$. The explicit, SDPA, and folded backends agree to $`10^{-10}`$, with and without padding.
*Cost:* compute stays $`O\big(N^2(d+\sum\varphi)\big)`$. **This is not sub-quadratic attention.** Only the $`O(HN^2)`$ bias memory is removed.
*Implementation:* values are zero-padded to the widened query/key dimension, because fused kernels need one head dimension; without this, CPU SDPA fell back to materializing the attention matrix ([`SPEC.md`](SPEC.md), [`EXECUTION_LEDGER.md`](EXECUTION_LEDGER.md)).

**P2 (exact shift invariance; implementation property).** The dense path indexes a rational table by $`(i-j)\bmod m`$, computed in int64. It is bitwise invariant under any common shift of positions: tested to $`2^{61}`$, and to $`10^{15}`$ for a full model. RoPE instead evaluates $`t\theta`$ in floating point, so its relative-phase error at absolute position $`t`$ grows like $`t\,\theta\,\epsilon_{\text{mach}}`$, a known long-context failure mode (Wang et al., 2024; Liu, 2026).

**B (handoff theorem B, phase identifiability modulus; PROVED in the handoff, re-checked).** $`F_H(t)=(e^{2\pi i h t/L})_{h\in H}`$ is injective exactly modulo $`L/\gcd(L,\gcd H)`$. Equivalently, frequencies $`h/L`$ have periods $`L/\gcd(h,L)`$, and the joint period is their lcm. For $`L=120`$: $`H=\{6,10\}`$ gives 60, and $`H=\{6,10,15\}`$ gives 120. The full phase bank resolves $`n\bmod\mathrm{lcm}`$, while the scalar trace kernel $`K_m`$ resolves only $`\gcd(n,m)`$ (P1). These are different representations, and the project never conflates them.

## 6. Not claimed / open

- **Not claimed:** sub-quadratic attention, speedups, natural-language gains, PSD attention matrices, or any statement about pretrained LLMs.
- **OPEN:** finite-$`N`$ rates for T5 as a function of RoPE base and head dimension; finite-$`t`$ rates for T8 beyond the first layer; monotonicity of the windowed limit $`F_\mu(W)`$ in $`W`$; optimal bank choice under a feature budget; data-dependent (query-conditioned) order routing; behaviour beyond the synthetic tasks in [`PREREGISTRATION.md`](PREREGISTRATION.md).
- **OPEN (T9, evidence readout):**
  - a finite-$`i`$ rate for T9d (needs Diophantine information on $`\theta`$; Weyl gives only $`o(i)`$), and a bank-level finite-window statement for sums of near-resonant components;
  - the rest-of-network term $`b_i`$: T9 assumes only that it is bounded (H-b), and its interaction with $`E_i`$ is not modeled;
  - unknown or mixed periods for the head ($`P`$ and $`\rho`$ are assumed known to the construction);
  - deeper networks containing the head (T8/C8 are not extended to them);
  - dependence models beyond exact copies at comb positions (T9e);
  - natural data;
  - also listed in the draft: the polynomial prefactor of $`\chi^n`$ in T9a.3 (the exponent is T9a.5), which $`(\kappa,W)`$ beyond $`D_P`$ give a positive rate on all motifs with a shared $`W`$ (T9c.5 settles $`W=I`$), a matching lower bound for the T9c.6 failure probability, and the "only if" half of T9b.1 for finite contexts $`N<\mathrm{lcm}`$.

## References

- S. Ramanujan, *On certain trigonometrical sums and their applications in the theory of numbers*, Trans. Cambridge Philos. Soc. 22 (1918).
- R. D. von Sterneck (1902); O. Hölder, *Zur Theorie der Kreisteilungsgleichung*, Prace Mat.-Fiz. 43 (1936).
- E. Cohen, *A class of arithmetical functions*, PNAS 41 (1955); *Representations of even functions (mod r)*, Duke Math. J. 25–26 (1958–59).
- P. J. McCarthy, *Introduction to Arithmetical Functions*, Springer (1986).
- P. P. Vaidyanathan, *Ramanujan sums in the context of signal processing, Parts I–II*, IEEE Trans. Signal Process. 62 (2014).
- H. Weyl, *Über die Gleichverteilung von Zahlen mod. Eins*, Math. Ann. 77 (1916); F. Lindemann, *Über die Zahl π*, Math. Ann. 20 (1882).
- J. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021); O. Press et al., *Train Short, Test Long (ALiBi)*, ICLR 2022; C. Raffel et al., *T5*, JMLR 2020.
- H. Wang et al., *When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training*, TMLR 2025 (arXiv:2411.13476); F. Liu, *Rotary Positional Embeddings as Phase Modulation* (arXiv:2602.10959, 2026).
