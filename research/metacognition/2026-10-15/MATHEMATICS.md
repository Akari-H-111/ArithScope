# Information available to a metacognitive controller

Evidence labels: PROVED means the displayed argument proves the stated
conditional statement. CLASSICAL identifies established algebra/statistics,
not a novelty claim. EXACT FINITE CHECK and NUMERICAL CHECK are verification
levels, not substitutes for a proof. DEVELOPMENT and OPEN do not amend the
main claim ledger. None of the propositions below establishes a new general
theory of machine metacognition.

## M1. The decision problem

Let X be the observed causal prefix, Y its unknown clean target, theta fixed
model parameters, and S a specified monitoring state. An action a returns a
distribution p_a(X), or abstains at a specified cost. A useful controller chooses

    a*(S) = argmin_a { E[-log p_a(X)[Y] | S] + eta Cost(a) }.

This is a decision-theoretic formulation, not an implemented policy. Error
prediction, probability quality, and action value are distinct targets.
If abstention is allowed, coverage must be reported; omitting difficult samples
does not establish improved loss at full coverage. Cost includes extra
forwards, gradient extraction, updates, and monitoring-model training.

## M2. A deterministic gradient does not enlarge a sufficient forward state

**PROVED, elementary.** If G = F(S) almost surely, every policy pi(S,G) can be
implemented as pi(S,F(S)). Therefore observing G cannot improve the optimal
decision risk over unrestricted policies based on S. Equivalently,
I(Y;G | S) = 0 whenever this conditional information is defined.

This is relative to the stated S. It does not assert I(Y;G | final hidden
state) = 0 for all network gradients. It does not rule out useful feature
engineering, finite-capacity learning effects, or a different computational
representation of the same information.

**CLASSICAL exact algebra, applied here.** Let h be one example's final hidden
vector, W a linear readout, p = softmax(Wh), u a fixed uniform target, and
L = KL(u || p), at temperature one. Then

    dL/dW = (p-u) h^T,
    ||dL/dW||_F = ||p-u||_2 ||h||_2.

Proof: dL/d(Wh) = p-u, followed by the chain rule. The Frobenius norm of an
outer product factors. Entrywise L1 norms factor as well. Thus the entire
last-layer gradient, not just its norm, is a deterministic function of (p,h).
The forward control must include both quantities and their interaction, rather
than entropy alone. This is not a replication of GradNorm's OOD benchmark.

For a stopped pseudo-label yhat, replacing u by one_hot(yhat) gives the same
identity. A confidently wrong pseudo-label can yield a small gradient. Neither
small nor large gradient has an unconditional correctness interpretation.

## M3. Replication can hide count from every parameter gradient

**PROVED, conditional.** Suppose a differentiable model satisfies
f_theta(R_k X) = f_theta(X) for every theta in an open neighborhood, where R_k
replicates all observations k times. For any differentiable proxy loss that
depends only on f_theta(X) and a replication-invariant target,

    grad_theta L(f_theta(R_k X)) = grad_theta L(f_theta(X)).

Proof: differentiate the functional identity. A mean of shared per-token
features followed by an arbitrary differentiable readout meets this condition.
Consequently even upstream gradients cannot recover replication count in this
class. The numerical probe checks all encoder and readout gradients.

**Counterexample to a stronger assertion.** Equality at one parameter value is
insufficient. Let binary logits be (2 + theta log n, 0), with theta = 0. Counts
n = 3 and n = 30 give the same output. However the uniform-KL derivative is
(sigmoid(2)-1/2) log n, which differs. The computational graph already contains
n; the gradient exposes its local sensitivity. A forward state that explicitly
includes n can access that information without this backward pass.

ArithScope has causal masks, residuals, and position-dependent computation.
Neither global replication invariance nor the complete mechanism of its
observed NLL plateau is proved here. M3 is a counterexample to universal claims
about gradient monitoring, not a theorem about every ArithScope checkpoint.

## M4. Exact evidence and a constructive readout

**CLASSICAL Bayes calculation; PROVED under the registered generating law.**
Assume known period P, independent uniform motif symbols, and independent
replacement noise epsilon in (0,1). Vocabulary size is V. At a query index t,
let v_c count observed symbols c at j <= t with j = t mod P. Set

    r = 1-epsilon+epsilon/V, s = epsilon/V, lambda = log(r/s).

For target c the likelihood is r^(v_c) s^(n-v_c), with n = sum_c v_c.
The uniform prior and common s^n factor cancel, so

    pi_c = exp(lambda v_c) / sum_d exp(lambda v_d).

Counts are sufficient under these assumptions. Shares h_c = v_c/n together
with n are also sufficient, and the readout softmax(lambda n h) is exactly pi.
This supplies a positive construction for the toy task, using the generating
period and noise. It is an oracle reference, not a trained ArithScope result.

With V = 16 and epsilon = 1/2, r/s = 17. The count vectors (2,1,0,...,0)
and (20,10,0,...,0) have the same shares and different posteriors. No function
of shares alone can equal both. M2 and M3 identify gradients that cannot repair
this collision.

For two equally weighted contexts with posteriors pi and rho and a common
prediction z, the mean excess Brier risk is

    (||z-pi||^2 + ||z-rho||^2)/2
      = ||z-(pi+rho)/2||^2 + ||pi-rho||^2/4.

Thus the strictly positive lower bound is ||pi-rho||^2/4. The analogous minimum
excess log loss is the Jensen-Shannon divergence, attained by the same mean
distribution. This is an identifiability limitation independent of training.

The parent's P3 bounds logits for fixed LayerNorm plus linear readout. Routing
among finitely many such fixed readouts still has a positive global NLL floor:
take the minimum of their finitely many positive bounds. To follow Bayes NLL
to zero, a count-dependent scale must be permitted after normalization, or the
readout must change. A finite menu can still lower the practical plateau.

## M5. Extra independent evidence helps Bayes prediction on average

**CLASSICAL proper-score identity.** For the true posterior pi and any positive
prediction p,

    E[-log p(Y) | X] = H(pi) + KL(pi || p).

For nested observation sets, the expected optimal log-loss difference is
H(Y|X_1:n)-H(Y|X_1:n+1) = I(Y;X_(n+1)|X_1:n) >= 0. The new symbol can increase
uncertainty on a particular realized history; the inequality averages over
the new observation. The expected optimal Brier loss likewise cannot increase,
by conditional expectation and squared-error projection.

Therefore an improvement from windowing a fixed learned model cannot by itself
show that distant evidence is inherently harmful. It can reflect misspecified
aggregation or readout. This statement does not prove which mechanism caused
E6-E8. For known pi, the benefit of switching full to windowed prediction is
KL(pi||p_full)-KL(pi||p_window); a clean realized label is unnecessary for that
oracle calculation. The probe uses it only for retrospective scoring.

## M6. Repeated observations are not repeated independent evidence

**PROVED counterexample.** Draw one noisy symbol S from the channel above, then
copy it k times. Conditional on the observed copies, the true posterior remains
P(Y|S), independent of k. A readout that incorrectly applies the independent
count formula predicts a distribution proportional to (r/s)^k for S, and 1
for each other symbol. As k grows it tends to a point mass, while the true
conditional error probability is positive. Its expected log loss diverges;
the correct posterior has constant finite risk. The exact finite checks
quantify the Brier regret at fixed k.

This does not refute count evidence under M4. It shows why independence,
dependence structure, and possible change points matter for imperfect periodic
data. An attention-derived effective sample size is not automatically an
independent count or a calibrated posterior; that requires a model and proof.

## M7. Conditions for a worthwhile gradient branch

The remaining OPEN hypothesis is specific: some gradient from a richer
computational graph adds predictive value for intervention benefit beyond a
matched forward monitoring state, after compute is counted. It requires all
of the following in a future experiment:

- a proxy available from the causal prefix, with no clean-target leakage;
- a forward baseline that includes the exact closed-form last-layer features;
- independent development, validation, and evaluation observations;
- comparison with the best fixed window and a count-aware forward baseline;
- improvement in full-coverage proper loss with an accuracy guard, or an
  explicit selective-risk/coverage comparison if abstention is the task;
- independent sequence and model-seed replication, including a dependence or
  nonstationarity stress case where raw count can fail.

A possible self-supervised noisy-symbol likelihood is
(1-epsilon) p_theta(clean_symbol=c | held-out context) + epsilon/V. Its score
must exclude the very observation being predicted. The current denoiser sees
its own noisy token, so directly fitting that token is not automatically a
valid held-out objective. No adaptation or routing policy is trained here.

