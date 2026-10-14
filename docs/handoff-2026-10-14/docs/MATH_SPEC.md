# Mathematical kernel / observability specifications

Status legend: **EXACT DERIVATION** (standard mathematics; reproducible below), **SOURCE-REPORTED** (under assumptions in originating work), **CANDIDATE** (new model adaptation), **OPEN** (requires study).

## A. Ramanujan character kernel [EXACT DERIVATION]

For integer order `m >= 1`, let `U_m = {a in {1,...,m} : gcd(a,m)=1}`, with the convention `U_1={1}`, `phi(1)=1`. Define

`c_m(n) = sum_{a in U_m} exp(2π i a n/m) = sum_{d | gcd(m,n)} d μ(m/d)`.

For integer token positions `t,s`, set

`K_m(t,s) = c_m(t-s)/phi(m)`.

Feature map `psi_m(t) = (exp(2π i a t/m))_(a∈U_m) / sqrt(phi(m))` gives `K_m(t,s)=<psi_m(t),psi_m(s)>`. Thus every finite Gram matrix is Hermitian PSD (in fact real symmetric); `K_m(t,t)=1` and `|K_m(t,s)| <= 1`.

For a **finite** M and `alpha_m >=0`, `sum alpha_m=1`, the mixture `K_alpha = sum alpha_m K_m` is PSD, diagonal 1 and bounded in absolute value by 1. Learning the `alpha` by softmax preserves these properties, **not** the positivity of attention logits.

Important: **kernel PSD is not a proof that softmax(K) is PSD or efficient**, and its boundedness does not imply complete Transformer numerical stability. A real-valued `K_m` loses information about phase orientation relative to the full complex character vector; never conflate two different representation-identifiability claims.

## B. Position identifiability theorem [EXACT DERIVATION]

Let `L >= 2`, finite nonempty positive harmonic labels `H={h_1,...,h_k}`, and `F_H(t) = (exp(2π i h_j t/L))_j` on integer positions. Put `g = gcd(H)`. Then:

`F_H(t) = F_H(s)  iff  t-s ≡ 0 mod [L/gcd(L,g)]`.

Proof: equality iff `L | h_j (t-s)` for all j. Bézout combines the `h_j` into `g`, so this is equivalent to `L | g(t-s)`, equivalent to the displayed modulus.

This is an **exact identifiability horizon for the stated integer Fourier feature bank only**. It does not mean that an untrained neural projection, kernel-only relative bias, combined positional/content head or finite-precision learned mapping retains that identifiability.

For L=120, H={6,10} yields period 60; H={6,10,15} yields period 120. The source AHR saturated lattice concerns an **additional multi-phase orbit quotient** and cannot be substituted for this simpler proposition without preserving its hypotheses.

## C. Cyclotomic-descendant prior [EXACT DERIVATION]

For a positive order m and integer D >= 2, `Desc_D(m)={r>=1: r/gcd(r,D)=m}`. The finite probability vector

`P_D(r | m) = phi(r)/(D phi(m))`, for r in Desc_D(m), 0 otherwise,

sums to 1. Proof: the D-th power map on roots of unity has D preimages for each point; over all primitive m-th roots there are D phi(m) preimages, partitioned by primitive source orders r. This defines a *counting distribution*, **not** a probabilistic model for all LLM routing tasks.

Define `p_r = P_D(r|m) exp(ell_r)/sum_s P_D(s|m) exp(ell_s)` on allowed r. With equal logits p returns to the prior. Hard-zeros mean only admitted branches are eligible. Candidate use: hierarchical memory/expert routing *if tokens/experts have a justified order-group assignment*.

## D. Finite normalized arithmetic features [EXACT POINTWISE IDENTITY]

For dimensionless torus coordinates theta in R^d and finite harmonic bank Q, let `a_0(theta)=1`, `a_(q,j)(theta)=exp(2π i q theta_j)-1`, `A(theta)=1+sum_(q,j)|a_(q,j)|^2`, `phi_r = a_r/sqrt(A)`. Then `sum_r |phi_r(theta)|^2 = 1` pointwise.

In the **original SAMT continuum setting**, the paper composes these multipliers with a unitary Haar-Mellin transform and obtains a Parseval g-frame. A tokenwise phase map does **not** automatically inherit that full Hilbert-space theorem or task robustness. If theta_j is an integer and q integer, `a_(q,j)=0`: choose a nontrivial, dimensionless `theta` map.

## E. Source AHR Bézout branch reconstruction [SOURCE-REPORTED / RE-DERIVABLE]

If normalized harmonic phase observations satisfy `U_h = exp(i h Delta_phi)`, coefficients b with `sum_h b_h h = g = gcd(H)` yield `prod_h U_h^{b_h} = exp(i g Delta_phi)`. Under independent equal small phase error variance sigma^2, the *linearized* error variance scales as `sigma^2 sum b_h^2`. For H=(6,10,15): (-14,7,1) has squared norm 246, (1,1,-1) has squared norm 3; the ratio is 82 for that specified model. This does NOT establish robustness of the learned attention layer.

Use existing AHR's full `phase_support_geometry/v0_01/MATHEMATICS.md` for the stronger deterministic error-box bound, conditions, and held-out confidence work.

## F. Evidence mixture / softmax [ALGEBRA EXACT; CONDITIONAL STATISTICS]

For normalized nonnegative priors pi_r and candidate log evidence s_r, define `F(s)=log sum_r pi_r exp(s_r)`. Then `∂F/∂s_r = softmax_r(log pi_r+s_r)`. This is a purely algebraic identity.

If `exp(s_r)` is a valid null likelihood ratio measured on independent or appropriately conditionally controlled held-out observations and pi_r is training-measurable, `E=sum_r pi_r exp(s_r)` is a unit-mean null e-value, and `Pr_0(E>=1/alpha)<=alpha` by Markov. Ordinary attention score logits **need not be likelihood ratios**; applying this null error statement to them without a fresh proof is forbidden.

## G. Marked response closure [SOURCE-REPORTED / STANDARD LINEAR ALGEBRA]

For fixed finite-dimensional V, subspace N_0 and fixed endomorphisms G_j, define `N_(r+1) = N_r + sum_j G_j N_r`. Once `N_(r+1)=N_r`, subsequent states are equal. At most `dim V-dim N_0` strict dimension-growth steps occur. Application to learned recurrent memory is **CANDIDATE** and needs floating-rank thresholds and resource comparison.

## H. Metric-stable frame candidate [CONDITIONAL EXACT]

If `T=S Λ S^{-1}` with invertible real S and real `|Λ_j| <=1`, choose positive diagonal W and `H=S^{-T} W S^{-1}`. Then `T^T H T <= H`. If also a finite analysis operator F satisfies `F^*F=I`, then `A=F H^(1/2)` obeys `||ATx|| <= ||Ax||`. This is an operator bound under *specified* S,Λ,W,F, not a learned LLM performance or stability guarantee.

## I. Implementation default [CANDIDATE]

For each head h, logits

`L_(h,i,j) = dot(q_(h,i),k_(h,j))/sqrt(d_head) + gamma_h sum_(m in M) alpha_(h,m) K_m(i,j)`.

Apply causal and padding masks **after** forming logits and **before** softmax. `alpha_(h,m)>=0`, `sum_m alpha_(h,m)=1`, `gamma` independent. For gamma=0, outputs/gradients must match the same underlying attention baseline. Keep exact order lookup defined for integer displacements; no rounding of position scalars prior to gcd computations. Full dense N×N bias remains O(N²).
