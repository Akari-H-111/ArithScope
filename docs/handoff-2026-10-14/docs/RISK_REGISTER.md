# Risk register and corrective tests

1. **Trivial integer phasors:** `exp(2π i q t)=1` for integer q,t. Test nontrivial t/L or dimensionless torus coordinates; default to integer-displacement Ramanujan kernel lookup instead.
2. **Phase and kernel identifiability conflated:** full feature map may separate more states than scalar `K_m(t,s)` viewed at one relative displacement. Specify which representation the theorem concerns.
3. **Kernel rank / aliasing:** finite harmonic/order banks have collisions; test exact gcd horizon, modular aliases, held-out periods and noise; don't infer capacity from PSD.
4. **Numerical normalization:** very large |positions| passed to trig routines can lose exact integer modular semantics. Prefer exact modulo before float features; verify finite precision and gradient.
5. **Misleading speed claims:** full dense attention plus additive N² kernel remains quadratic; benchmark host/device-specific runtime and allocation.
6. **Bad comparison:** an arithmetic bias may only win against a weak no-position baseline. Include matched Fourier, RoPE and ALiBi controls; train equally.
7. **Prior/likelihood misuse:** cyclotomic counting is not evidence of task probabilities; AHR root-mixture e-values require held-out conditional null. Keep as hypotheses until transported.
8. **Train/test leakage:** hyperparameters, learned phase prior and model selection must not use held-out task labels. Freeze before test and retain consumed-development distinctions.
9. **Operator-theory overreach:** SAMT continuum Parseval and metric contraction do not automatically prove a learned attention module's stability. Give separate finite-token theorem or empirical bound.
10. **False derivation from number fields:** Galois/cyclotomic structures and CY power maps share algebra, not a canonical correspondence to tokens. Define the engineered mapping.
11. **External code and rights:** public repositories include separately licensed and possibly all-rights-reserved work. Do not copy sources or figures into a new published repository unless rights are clear; cite and reimplement minimal mathematics independently.
12. **Scope explosion:** start with H1/H2 PyTorch logit bias. Root-mixture, rank memory, browser visualization and RH-inspired operators are conditional studies, not a reason to delay reproducible model-level tests.
