# ArithScope mathematics

This file states every mathematical claim the project makes, with its evidence
level. Labels:

| Label | Meaning |
|---|---|
| **PROVED** | Complete proof below; statement also checked exhaustively on a finite range with exact integer/rational arithmetic (`tests/test_arith.py`). |
| **CLASSICAL** | Standard result, cited; finite exact check provided. |
| **ASYMPTOTIC** | Limit statement with a complete proof; no finite-`N` rate is claimed. |
| **NUMERICAL** | Floating-point computation (linear program, simulation). Supporting evidence, not a proof. |
| **EMPIRICAL** | Measured in trained models; see [`RESULTS.md`](RESULTS.md). |
| **OPEN** | Not established. |

No statement here was checked by a proof assistant.

## 0. Notation

- `c_m(n) = sum_{1<=a<=m, gcd(a,m)=1} exp(2 pi i a n / m)` is the Ramanujan sum (an integer); `phi` is Euler's totient, `mu` the Moebius function.
- `K_m(n) = c_m(n) / phi(m)` is the **normalized Ramanujan kernel** of **order** `m`. (An order is a modulus. It is never the attention query `Q`.)
- A **bank** is a finite set of orders `M`. For attention, `n = i - j` is the integer offset between query position `i` and key position `j`.
- The **`P`-comb** is the set of offsets `P Z`. For a `P`-periodic score `f` with `f(0) = 1`, the **comb margin** is `Delta(f) = 1 - max_{P does not divide n} f(n)`.

## 1. The ArithScope bias

For head `h`,

```
logits[h,i,j] = <q_i, k_j> / sqrt(d_head)  (+ RoPE rotation, if used)
                + gamma_h * sum_{m in M} alpha_{h,m} K_m(i - j)
alpha_h = softmax(order_logits_h)  (convex weights),   gamma_h in R, initialized to 0
```

Masks are applied after the bias and before the softmax. With `gamma = 0` the bias is exactly zero, so the model is identical to its base model (bitwise in our tests, for outputs and gradients).

## 2. Kernel facts

**L1 (closed form, CLASSICAL; von Sterneck 1902, Hoelder 1936).** With `q = m / gcd(m, n)`,
`c_m(n) = mu(q) phi(m) / phi(q)`, hence `K_m(n) = mu(q) / phi(q)`. So `K_m` is even, `m`-periodic, depends only on `gcd(n, m)`, and `K_m(n) = 1` iff `m | n`.
*Check:* divisor formula = closed form = handoff oracle for `m < 120`, `|n| <= 150`.

**L1' (positive-semidefinite, unit diagonal, bounded; CLASSICAL).** `K_m(i - j) = <psi_m(i), psi_m(j)>` with `psi_m(t) = (exp(2 pi i a t / m))_{a in U_m} / sqrt(phi(m))`. So every Gram matrix `[K_m(t_r - t_s)]` is PSD, the diagonal is 1, and `|K_m| <= 1` by Cauchy-Schwarz. Convex mixtures keep all three properties. A negative `gamma` makes the *bias* negative semidefinite, which is allowed. **Not claimed:** anything about the PSD-ness of the attention matrix after softmax.
*Check:* the circulant spectrum is `(m / phi(m)) [gcd(h, m) = 1] >= 0` (`m < 40`); the minimum eigenvalue of random mixtures is ≥ −1e−10 (`tests/test_bias.py`).

**L2 (comb gap; PROVED).** Let `m >= 2`. For `m` not dividing `n`, `K_m(n) <= 1/2`. The value `1/2` is attained iff `6 | m`; otherwise the bound improves to `K_m(n) <= 1/4`. If `m = p^e` is a prime power, then `K_m(n) <= 0` off the comb. For `m = p` prime, the gap is `Delta(K_p) = p / (p - 1)`.

*Proof.* By L1, `K_m(n) = mu(q)/phi(q)` with `q = m/gcd(m,n)` a divisor of `m`, and `q > 1` exactly when `m` does not divide `n`. Every divisor `q > 1` occurs (take `n = m/q`). If `mu(q) <= 0` the value is `<= 0`. If `mu(q) = 1`, then `q` is squarefree with an even number `r >= 2` of prime factors, and `phi(q) = prod (p_i - 1)`. For `r = 2` this is `>= (2-1)(3-1) = 2`, with equality iff `q = 6`; any other pair gives `phi >= 4`. For `r >= 4`, `phi >= 1*2*4*6`. So `mu(q)/phi(q) <= 1/2`, with equality iff `q = 6`, and `<= 1/4` otherwise. For `m = p^e` the divisors `q > 1` are `p^j`, where `mu(p) = -1` and `mu(p^j) = 0` for `j >= 2`. QED.
*Check:* exact, all `m < 400`.

## 3. Optimal periodic attention combs

**T3 (optimal positive-definite comb; PROVED).** Let `P >= 2`. Among all `P`-periodic positive-definite `f: Z -> R` with `f(0) = 1`,

```
Delta(f) <= P / (P - 1),   with equality iff   f = f*_P := (P [P | n] - 1) / (P - 1).
```

Moreover, `f*_P = sum_{d | P, d > 1} (phi(d) / (P - 1)) K_d`. This is a convex mixture of ArithScope kernels, so an ArithScope head whose bank contains the divisors of `P` represents the unique optimum. For prime `P`, `K_P` alone is optimal.

*Proof.* Expand `f(n) = sum_{h mod P} fhat(h) exp(2 pi i h n / P)`. Positive-definiteness on `Z/PZ` is equivalent to `fhat >= 0`, and `f(0) = sum fhat = 1`. The mean over a period is `fhat(0) >= 0`, so `sum_{n=1}^{P-1} f(n) = P fhat(0) - 1 >= -1`. The maximum over the `P - 1` off-comb residues is at least their mean, so it is `>= -1/(P-1)`, and `Delta(f) <= 1 + 1/(P-1)`. Equality forces `fhat(0) = 0` and every off-comb value to equal `-1/(P-1)`, so `f = f*_P`. Conversely, `f*_P = (1/(P-1)) sum_{h=1}^{P-1} exp(2 pi i h n / P)` has nonnegative coefficients. Group the harmonics `h` by `d = P / gcd(h, P)`, i.e. by the primitive `d`-th roots of unity. This gives `sum_{h=1}^{P-1} exp(...) = sum_{d | P, d > 1} c_d(n)`, and `c_d = phi(d) K_d`. The weights sum to `sum_{d|P, d>1} phi(d) / (P - 1) = 1`. QED.
*Check:* the identity holds exactly for `P < 80`. Independently, a linear program over all PSD `P`-periodic functions returns `P/(P-1)` to 4e−16 for `P <= 36` (NUMERICAL, `arithscope/theory.py`).

**C3.1 (matched cosine banks are weaker).** The natural "same periods, same parameter count" Fourier control replaces each `K_m` by its fundamental cosine `cos(2 pi n / m)`. For prime `P`, a convex mixture that equals 1 on the `P`-comb can only use `cos(2 pi n / P)` and the constant. Its best margin is therefore `1 - cos(2 pi / P) = 2 sin^2(pi / P)`. The ratio to the ArithScope optimum is `(P/(P-1)) / (2 sin^2(pi/P))`, which grows like `P^2 / (2 pi^2)`. The two banks are equal only for `P in {2, 3, 4, 6}` (PROVED for primes; NUMERICAL LP for composites):

| period `P` | 5 | 7 | 11 | 12 | 13 | 24 |
|---|---|---|---|---|---|---|
| optimum `P/(P-1)` (ArithScope bank) | 1.250 | 1.167 | 1.100 | 1.091 | 1.083 | 1.043 |
| matched fundamental-cosine bank (LP) | 0.691 | 0.377 | 0.159 | 0.987 | 0.115 | 0.838 |
| single kernel `K_P` | 1.250 | 1.167 | 1.100 | 0.500 | 1.083 | 0.500 |
| single cosine `cos(2 pi n/P)` | 0.691 | 0.377 | 0.159 | 0.134 | 0.115 | 0.034 |

The margin multiplies the logit scale `gamma`. At a fixed logit budget, a larger margin means sharper attention on the comb.

**T4 (CRT comb composition; PROVED).** Let `S` be a nonempty set of orders and `alpha` a probability vector with support `S`. Then `b = sum_{m in S} alpha_m K_m` satisfies `b(n) = 1` iff `lcm(S) | n`. Otherwise `b(n) <= 1 - min_{m in S} alpha_m Delta(K_m)`. Consequently:

1. A convex ArithScope head with bank `M` can make exactly the combs `lcm(S) Z`, `S ⊆ M`, its strict argmax set. These are its **selectable combs**.
2. If `P = q_1 ... q_k` with pairwise coprime prime powers `q_i ∈ M`, the uniform mixture over the `q_i` selects `P Z` with margin `>= 1/k`, because L2 gives `Delta >= 1` for prime powers.

*Proof.* Each `K_m <= 1` and the weights are positive, so `b(n) = 1` iff every `K_m(n) = 1`, iff every `m in S` divides `n`, iff `lcm(S) | n`. If not, some `m in S` does not divide `n`. Then `b(n) <= (1 - alpha_m) + alpha_m (1 - Delta(K_m))`. QED.
*Check:* exact, all `P < 300`.

*Horizon of a bank.* Every `K_m`, `m in M`, is `lcm(M)`-periodic. The prime-power bank `B_X = {p^e <= X}` has `lcm(B_X) = lcm(1..X) = exp(psi(X))` (Chebyshev). For example, `B_16` has 10 orders, horizon 720,720, and folded rank 55. With ten parameters per head it can select every comb whose prime-power components are at most 16.

**P1 (alias partition; PROVED, CLASSICAL core).** Let the signature of an offset be `sigma_M(n) = (K_m(n))_{m in M}`. Every bias in `span{K_m : m in M}` is constant on each class `{n : sigma_M(n) = sigma}`. If `M` is divisor-closed, then `span{K_m} = span{[d | n] : d in M}`. This follows from Moebius inversion, `c_m = sum_{d|m} d mu(m/d) [d | .]` and `[d | .] = (1/d) sum_{e|d} c_e`. For `M = divisors(L)` the classes are exactly the gcd classes `{n : gcd(n, L) = g}`. This is Cohen's space of even functions mod `L` (E. Cohen, 1955/1959). Equivalently, an ArithScope head is invariant under `n -> u n` for every unit `u mod lcm(M)`, which is precisely its inductive bias.
*Check:* gcd classes for `L in {12, 18, 24, 30}`. The alias fixture (Section 5) checks that banks of lcm 12 put offset 12 in the comb class of 0, while `{3, 8}` and `{8, 12}` do not.

## 4. Why exact periods are necessary for long-range periodic attention

**T5 (residue-blindness of incommensurate frequency banks; ASYMPTOTIC).** Let `theta in R^K` and let `H` be the closure of `{n theta mod 2 pi}` in the torus `T^K`. Assume no integer vector `c` has `c . theta in 2 pi (Q \ Z)`, which is equivalent to `H` being connected. Then for every continuous `g: T^K -> R`, every `P >= 1`, and every residue `r`,

```
lim_{N -> inf}  sum_{0<=n<N, n = r mod P} exp(g(n theta))  /  sum_{0<=n<N} exp(g(n theta))  =  1/P.
```

*Application to RoPE.* With fixed content, a RoPE head's logit at offset `n` is `g(n theta) = sum_k Re(z_k exp(i n theta_k))`, where `theta_k = b^(-2k/d)`. For integer `b >= 2` every `theta_k` is algebraic, so every `c . theta` is algebraic. Then `c . theta = 2 pi q` with rational `q != 0` would make `pi` algebraic, contradicting Lindemann (1882). So the hypothesis holds for every standard RoPE configuration. In the long-context limit, the attention mass a RoPE head puts on any residue class mod `P` tends to the uniform share `1/P`: **the RoPE positional channel cannot keep a periodic preference at long range.** The same holds for learned real frequencies unless a frequency is *exactly* a rational multiple of `2 pi`, and gradient descent does not produce exact rationals.

*Proof.*
1. `H` is a compact abelian group whose dual is `Z^K / Ann`, where `Ann = {c : c . theta in 2 pi Z}`. A compact abelian group is connected iff its dual is torsion-free. Here that means `j c in Ann` implies `c in Ann`, i.e. no `c` has `c . theta in 2 pi (Q \ Z)`.
2. The sequence `n theta` is equidistributed in `H` with respect to Haar measure (Weyl). Every nontrivial character `chi` of the monothetic group `H` has `chi(theta) != 1`, so the Cesaro means of `chi(n theta)` vanish, and continuous functions are uniform limits of characters' linear combinations.
3. Let `H_P` be the closed subgroup generated by `P theta`. It contains `P H`. Since `H` is a torus, multiplication by `P` is onto, so `H_P = H`. Hence `(r + kP) theta = r theta + k (P theta)` is equidistributed in `r theta + H = H`.
4. Apply 2 and 3 to the positive continuous function `exp(g)`. Both sums divided by their term counts converge to the same Haar integral, and the count ratio tends to `1/P`. QED.

*Scope.*
- This is a statement about the **positional channel with fixed content**. Content that is itself correlated with position (e.g. a periodic motif in the input) is outside its hypotheses. With i.i.d. content independent of position the same limit holds almost surely; this is a proof sketch, not given in full.
- No convergence rate is claimed. NUMERICAL: random RoPE coefficients give comb mass 0.0769 = 1/13 at `N = 2^18` for `d = 16`, but still 0.109 for `d = 64`, whose slowest frequencies have periods near 47,000 tokens.
- For a frequency `omega = 2 pi / P + eps`, the comb phase drifts by `n eps`, so selectivity survives only for `n << 1/|eps|` (NUMERICAL: `eps = 1e-3` loses most of the comb mass by `n = 1024`).

**T6 (length-invariant comb mass; PROVED, elementary).** If the positional scores are `P`-periodic, as for any ArithScope bank or fixed-period cosine bank whose orders divide `P`, then for every window `N = kP` the comb mass equals `exp(s(0)) / sum_{r<P} exp(s(r))`, independent of `k`. Exact periods turn the "positional extrapolation" problem into an identity. They do not make content processing length-invariant; that part is EMPIRICAL.

## 5. Implementation identities

**T7 (exact low-rank folding; PROVED).** Define real features `u_1(t) = 1` and `u_2(t) = (-1)^t`. For `m > 2`, set `u_m(t) = sqrt(2/phi(m)) (cos(2 pi a t/m), sin(2 pi a t/m))` over `a in U_m, a < m/2`. Then `K_m(i - j) = <u_m(i), u_m(j)>`, so

```
<q_i, k_j>/sqrt(d) + bias_h(i - j) = < [q_i, sqrt(d) * (gamma_h alpha_{h,m} u_m(i))_m],  [k_j, (u_m(j))_m] > / sqrt(d).
```

The ArithScope bias can therefore run inside a fused attention kernel. It adds `sum_{m in M} phi(m)` feature dimensions to queries and keys (80 for `M = {1..16}`, 55 for `B_16`) and never builds an `[N, N]` bias tensor. The rank `sum phi(m)` is minimal once `N >= lcm(M)`, because distinct reduced fractions `a/m` give linearly independent characters. Phases use the exact residue `(a (t mod m)) mod m`.
*Check:* the folded and dense biases agree to 1e−12 in float64, including positions near 1e15. The explicit, SDPA, and folded attention backends agree to 1e−10, with and without padding (`tests/`).
*Cost:* compute stays `O(N^2 (d + sum phi))`. **This is not sub-quadratic attention.** Only the `O(H N^2)` bias memory is removed.

**P2 (exact shift invariance; implementation property).** The dense path indexes a rational table by `(i - j) mod m`, computed in int64. It is bitwise invariant under any common shift of positions (tested up to `2^61`, and for a full model up to `10^15`). By contrast, RoPE evaluates `t * theta` in floating point. The relative phase error at absolute position `t` grows like `t * theta * eps_mach`, a known long-context failure mode (Wang et al., 2024; Liu, 2026).

**B (handoff theorem B, phase identifiability modulus; PROVED in the handoff, re-checked).** `F_H(t) = (exp(2 pi i h t / L))_{h in H}` is injective exactly modulo `L / gcd(L, gcd(H))`. Equivalently, frequencies `h / L` have periods `L / gcd(h, L)`, and the joint period is their lcm (e.g. `L = 120`: `H = {6, 10}` gives 60, `H = {6, 10, 15}` gives 120). The full phase bank resolves `n mod lcm`. The scalar trace kernel `K_m` only resolves `gcd(n, m)` (P1). These are different representations, and the project never conflates them.

## 6. Not claimed / open

- **Not claimed:** sub-quadratic attention, speedups, natural-language gains, PSD attention matrices, or any statement about pretrained LLMs.
- **OPEN:** finite-`N` rates for T5 as a function of RoPE base and head dimension; optimal bank choice under a feature budget; data-dependent (query-conditioned) order routing; behaviour beyond the synthetic tasks in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## References

- S. Ramanujan, *On certain trigonometrical sums and their applications in the theory of numbers*, Trans. Cambridge Philos. Soc. 22 (1918).
- R. D. von Sterneck (1902); O. Hoelder, *Zur Theorie der Kreisteilungsgleichung*, Prace Mat.-Fiz. 43 (1936).
- E. Cohen, *A class of arithmetical functions*, PNAS 41 (1955); *Representations of even functions (mod r)*, Duke Math. J. 25/26 (1958-59).
- P. J. McCarthy, *Introduction to Arithmetical Functions*, Springer (1986).
- P. P. Vaidyanathan, *Ramanujan sums in the context of signal processing, Parts I-II*, IEEE Trans. Signal Process. 62 (2014).
- H. Weyl, *Ueber die Gleichverteilung von Zahlen mod. Eins*, Math. Ann. 77 (1916); F. Lindemann, *Ueber die Zahl pi*, Math. Ann. 20 (1882).
- J. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021); O. Press et al., *Train Short, Test Long (ALiBi)*, ICLR 2022; C. Raffel et al., *T5*, JMLR 2020.
- H. Wang et al., *When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training*, TMLR 2025 (arXiv:2411.13476); F. Liu, *Rotary Positional Embeddings as Phase Modulation* (arXiv:2602.10959, 2026).
