# Delta writes, arithmetic Hilbert geometry, and evidence counts

CLASSICAL labels established identities. PROVED means the displayed argument
establishes the stated conditional application, not a novelty claim. EXACT
FINITE and NUMERICAL checks verify bounded cases. DEVELOPMENT observations do
not upgrade the main ledger. Integration into a learned model remains OPEN.

## D1. What the Hilbert-space interpretation supplies

Let H be a real key Hilbert space and E the value space. A memory is a linear
operator S:H->E. For finite dimensions, its Hilbert-Schmidt norm is the
Frobenius norm. With column key k, query q, and value v, DeltaNet uses

    S_t = S_(t-1) + beta_t (v_t - S_(t-1) k_t) k_t^*,
    output_t = S_t q_t.

This is the established online regression/fast-weight recurrence, with a
rank-one operator `(v-Sk) k^*`. It updates the recurrent state, not the model's
global QKV projection parameters through an optimizer. Finite Euclidean QKV
vectors already form a Hilbert space. The useful addition is an explicit
inner product, feature map, and operator analysis, rather than the name alone.

An RKHS interpretation supplies a feature inner product equal to a PSD kernel.
An infinite-dimensional kernel is not free fixed-size storage: a generic
kernelized update needs a growing dictionary or a finite approximation.
ArithScope supplies finite residue features with a known rank, so this issue
can be audited directly. Its current additive score and folded SDPA backend
are not thereby replaced by a DeltaNet layer.

## D2. Projection and stability of one write

**CLASSICAL / PROVED conditional application.** For nonzero k, the closest
operator to S in Hilbert-Schmidt norm satisfying `S_new k = v` is

    S_new = S + (v-Sk) k^* / ||k||^2.

Indeed any correction D obeys `||D||_HS >= ||Dk||/||k||` and the displayed
rank-one correction attains equality. For unit k, beta=1 is this affine
projection; it fixes the current key exactly while preserving directions
orthogonal to k.

If a reference S_star satisfies `S_star k = v`, let E=S-S_star. Then

    ||E_new||_HS^2 = ||E||_HS^2
                    - beta (2-beta ||k||^2) ||Ek||^2.

Expand `(I-beta kk^*)^2` to prove the identity. The step is nonexpansive for
`0 <= beta ||k||^2 <= 2`. PSD alone does not normalize k or guarantee this
condition. A scalar key k=2 with beta=1 multiplies the squared error by 9.
This is a same-target contraction, not a proof that noisy writes always reduce
prediction risk or that a common S_star exists for inconsistent associations.

## D3. Interference and the causal QKV matrix

For any read query q, a write changes its output by

    Delta output(q) = beta (v-Sk) <k,q>.

Thus exact non-interference between independently writable addresses requires
orthogonality. Negative correlations can change an unrelated read in the
opposite direction; a PSD Gram matrix can have negative off-diagonal entries.

For initial state zero, stack row queries Q, keys K and values V. Let B be the
diagonal matrix of gains, G=KK^T, and L=I+B strictLower(G). The innovations U
and outputs O satisfy

    U = L^(-1) B V,
    O = Lower(QK^T) U.

Proof: `U_t=beta_t(v_t-sum_(j<t) U_j <k_j,k_t>)`; solve the unit lower-triangular
system, then read the sum of past innovations. The resulting influence matrix
can be signed and need not have row sums one. For two keys with inner product
-1/2, unit gains, and final query equal to the first key, the two value
coefficients are `(3/4,-1/2)`. They are not attention probabilities or confidence
weights. The dense form in this study is an independent small verification
oracle, not a fast parallel kernel.

## D4. A direct arithmetic bridge: restore the constant coordinate

The parent's T3 optimum is the simplex kernel

    f*_P(n) = (P 1_(P|n)-1)/(P-1)
            = sum_(m|P,m>1) phi(m)/(P-1) K_m(n).

Its Gram matrix on the P residues has rank P-1, unit diagonal, off-diagonal
-1/(P-1), and zero row sums. Any feature realization obeys
`sum_r Phi(r)=0`, because the squared norm of that sum is the sum of the Gram
entries. Therefore every plain linear state readout obeys
`sum_r S Phi(r)=0`. It cannot store arbitrary independently chosen values at
all P addresses; even a common nonzero constant value is impossible.

This is a restriction on this linear memory readout, not on a complete neural
network with residuals, nonlinearities, or an extra affine value bias. Such an
extra bias can itself supply the missing constant degree of freedom and must
be included in the state/feature accounting.

The obstruction concerns using the T3-optimal period subbank as the sole key
geometry. It is not a rank statement about the parent's entire order bank,
content projections, or full Transformer. Those can have additional dimensions.

**CLASSICAL identity / PROVED application.** Add the constant character:

    D_P(n) = [1+(P-1) f*_P(n)]/P
           = sum_(m|P) phi(m)/P K_m(n)
           = 1_(P|n).

The identity follows from the T3 formula and K_1=1. Its feature realization is

    Phi_plus(r) = (1/sqrt(P), sqrt((P-1)/P) Phi_simplex(r)).

Then `<Phi_plus(r),Phi_plus(s)> = 1_(r=s mod P)`. These are P orthonormal
addresses, with minimal effective dimension P. More generally exact retrieval
of arbitrary values at R linear addresses requires their feature matrix to have
rank R: otherwise some nonzero linear relation between keys also constrains
every possible readout.

In softmax attention, the two kernels are equivalent after rescaling:

    softmax(content + gamma f*_P)
      = softmax(content + gamma P/(P-1) D_P),

since the remaining term is a row-wise constant. At a fixed gamma budget their
margins differ, as T3 states; this equation includes the scale change. Recurrent
memory has no corresponding softmax invariance. The constant coordinate
changes rank and cross-address interference. This is the repository-specific
bridge, not a claim that the divisor identity itself is new.

The sum construction requires all divisor orders of P. The default bank contains
those needed for P=12 and P=13; the finite check through P=32 does not assert
that every such kernel is available in the unchanged default bank. For coprime
a,b, multiplying orthogonal kernels D_a D_b gives D_(ab), but its tensor-product
feature space still needs ab independent phase dimensions. CRT comb selection
by a small additive bank is not the same as independent storage at every
residue of its lcm horizon.

The prototype reuses exact-residue `key_features` and multiplies both Q and K
features by square roots of the nonnegative mixture weights. The parent's
asymmetric `query_features` and positional gamma are not substituted for this
Hilbert map. Positional gamma is not the positive delta write gain beta.

## D5. Content and phase should be combined according to the desired geometry

Concatenation gives an additive kernel:

    <(psi(x),Phi(r)),(psi(x'),Phi(s))>
      = <psi(x),psi(x')> + <Phi(r),Phi(s)>.

Even orthogonal phase features leave cross-phase content interference. A tensor
product instead gives

    <psi(x) tensor Phi_plus(r), psi(x') tensor Phi_plus(s)>
      = <psi(x),psi(x')> 1_(r=s).

This is a proved way to isolate phase blocks while allowing content-dependent
reads and writes inside each block. With d content features and P phases it
has dP features and a value-dimension-times-dP state; it is not a free capacity
increase. Queries and keys may have separate content projections, but their
phase factors must still align. No learned content tensor-product model is
trained here. The numerical counterexample only checks the product identity.

## D6. What DeltaNet does and does not accumulate

For orthonormal phase keys, alpha=1, a phase's nth visit, and observed-symbol
one-hot value v_n, choose beta_n=1/n. The corresponding read obeys

    h_n = (1-1/n) h_(n-1) + v_n/n = (sum_(i<=n) v_i)/n.

Induction proves the running mean; other phases are unchanged. Means still
discard count. Keep n separately and place the scale after any normalization.
Under known period, uniform clean-symbol prior, independent replacement noise
epsilon and vocabulary V, the exact posterior is

    p(clean=c | observations) = softmax_c(lambda n h_n),
    lambda = log((1-epsilon+epsilon/V)/(epsilon/V)).

For V=16 and epsilon=1/2, lambda=log(17). This is an engineered oracle for the
specified law, not a learned calibration result or a valid formula for arbitrary
hidden-state vectors. Approximate content-dependent shares need new analysis.

**Necessary simpler control.** With the same orthogonal keys, additive memory
`A_new=A+v k^*` already has `A k_r=counts_r`. Its readout `softmax(lambda A k_r)`
is the same exact posterior, without a separate counter being mathematically
necessary. Delta mean plus count has no accuracy or probability advantage over
that additive reference in the ideal stationary law. Delta's distinct purpose
is selective correction, bounded adaptive summaries, and possible tracking of
changing associations, not the invention of cumulative evidence.

The simulation zero-pads simplex features to P and allocates the same VP fast
weight scalars and P count registers to every mechanism. Effective ranks still
differ. There is no learned parameter tuning; fixed readout scale is a declared
reference, not an optimized baseline. No performance claim against trained
DeltaNet or the parent Transformer follows from these scores.

## D7. Forgetting and confidence are separate design choices

For constant beta in (0,1) at one address, stationary independent observations
receive exponential weights. Their limiting variance is
`beta/(2-beta) * Var(v)`; the variance-equivalent independent sample count is
`(2-beta)/beta`. At beta=1/4 this is 7, regardless of the total number of visits.
This is a variance identity, not a posterior uncertainty certificate. Scaling
such an adaptive state by total raw count can create unjustified confidence.

The reverse tradeoff occurs after a change point. After 16 zeros and 4 ones,
the all-history mean is 1/5 while gain-1/4 memory is 175/256, closer to the new
value 1. The example shows a tracking/aggregation tradeoff, not a general
optimal forgetting schedule. Correlated observations also invalidate the
independent count interpretation, as the previous branch demonstrates.

Gated DeltaNet's established update is

    S_new = alpha S (I-beta kk^*) + beta v k^*.

It decays the state before computing the residual write. A global alpha<1 also
decays non-addressed phase blocks; the exact running-mean/count proof above
requires alpha=1. Coupling gates to uncertainty needs an explicit statistical
model. The Kalman Delta Networks preprint already studies covariance/uncertainty
tracking in a linear-Gaussian memory model. That is related prior work, and its
Gaussian optimality is not the categorical posterior statement proved here.

## D8. Scope of possible assistance to ArithScope

The useful conditional construction is a phase-addressed memory, with a count
or uncertainty state that distinguishes averaging from evidence accumulation.
The kernel/rank calculation says how to avoid one specific interference source.
The projection calculation says how to constrain write stability. Neither
establishes learned period discovery, mixed/unknown-period routing, natural
data benefit, general calibration, or a faster implementation.

The parent's fixed LayerNorm plus linear readout bound still applies if a new
memory is routed through that same bounded final representation. An unbounded
count-derived scale must affect logits after normalization to follow the
ideal posterior to zero NLL. A large arithmetic lcm horizon also does not imply
that a small finite-rank state stores independent arbitrary values at all of
those residues. These are required integration checks, not solved claims.

