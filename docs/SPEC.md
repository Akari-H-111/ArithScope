# ArithScope module specification (v0.1)

This is the normative contract for the public API. Tests in `tests/` check each
clause; the clause tags in brackets name the test.

## 1. `arithscope.arith` (no torch dependency)

All functions are exact: they return `int` or `fractions.Fraction`, never floats.
Inputs that must be positive integers reject `bool`, `float`, `0`, and negatives with
`ValueError` [`test_input_validation`].

| Function | Contract |
|---|---|
| `ramanujan_sum(m, n)` | `c_m(n)` by the divisor formula, for any int `n` (negative allowed). Equals `ramanujan_sum_closed(m, n)` and the handoff oracle [`test_divisor_formula_matches_closed_form_and_handoff_oracle`]. |
| `kernel(m, n)` | `K_m(n) = c_m(n) / phi(m) = mu(q) / phi(q)`, `q = m / gcd(m, n)` [`test_kernel_closed_form_L1`]. |
| `kernel_table(m)` | `(K_m(0), …, K_m(m - 1))`. |
| `comb_gap(m)` | `1 - max_{m does not divide n} K_m(n)`; `≥ 1/2`, `= 1/2` iff `6 | m`; `≥ 1` for prime powers [`test_comb_gap_lemma_L2`]. |
| `optimal_comb_weights(P)` | `{d: phi(d)/(P-1) : d | P, d > 1}`; the mixture equals `(P [P|n] - 1)/(P - 1)` [`test_optimal_positive_definite_comb_T3`]. |
| `prime_power_components(n)` | Pairwise coprime prime powers with product `n` [`test_crt_comb_composition_T4`]. |
| `alias_classes(orders, offsets)` | Partition of `offsets` by the signature `(K_m(n))_{m in orders}` [`test_alias_classes_*`]. |
| `folded_rank(orders)` | `sum_m phi(m)`. |
| `identifiability_modulus(L, H)` | `L / gcd(L, gcd(H))` (handoff theorem B). |

## 2. Position tensors

- **Type.** Positions are integer tensors (`int32` or `int64`). Floating positions raise `TypeError` [`test_rejects_float_positions_and_bad_orders`].
- **Shapes.** `[N]` (shared by the batch) or `[B, N]` (per sequence, e.g. left padding or a KV-cache offset).
- **Offsets.** `i - j` is computed in int64. Every offset-only bias is a function of the exact integer offset.
- **Absolute values.** Positions may be arbitrarily large within int64 (tested to `2^61`). Only differences matter for every mechanism except RoPE's floating-point angle computation (see §5).

## 3. Bias modules (`arithscope.bias`)

Every bias is an `OffsetBias`. The call `bias(q_pos [..., Nq], k_pos [..., Nk])` returns `[..., H, Nq, Nk]`, where `out[..., h, i, j] = b_h(q_pos[i] - k_pos[j])`. Leading dimensions broadcast. `Nq != Nk` is supported, e.g. queries after a KV cache [`test_dense_shapes_offsets_and_symmetry`].

### 3.1 `RamanujanBias(num_heads, orders=(1..16), gamma_init=0.0, learn_mixture=True)`

- **Value.** `b_h(n) = gamma_h * sum_m alpha_{h,m} K_m(n)`, with `alpha_h = softmax(mixture_logits_h)`. `K_m` comes from an exact rational table indexed by `n mod m`, rounded once to float64 and then to the parameter dtype [`test_matches_exact_mixture`].
- **Parameters.** `mixture_logits [H, |M|]` (init 0, i.e. uniform `alpha`) and `gamma [H]` (init `gamma_init`; default 0).
- **Orders.** `orders` must be distinct positive ints, otherwise `ValueError`.
- **`gamma = 0`.** The output is exactly `0.0`. The gradient with respect to `mixture_logits` is exactly 0 (expected, not a bug); the gradient with respect to `gamma` is generally nonzero [`test_gamma_zero_is_exact_zero_and_alpha_gradient_trap`].
- **Kernel properties.** With `gamma = 1`, the matrix over any positions is symmetric PSD with unit diagonal and `|.| <= 1` [`test_kernel_mixture_is_psd_with_unit_diagonal`].
- **Shift invariance.** Bitwise invariant under any common shift of `q_pos` and `k_pos` [`test_exact_shift_invariance`]. A shift leaves the set of offsets, and hence the whole computation, unchanged.
- **Rounding across calls.** The value at a given offset is computed as `gamma * (alpha @ K[:, offsets])`. In calls whose offset *ranges* differ, it can differ by floating-point rounding (observed: 1 ulp in float64 with the pip CPU build of PyTorch). Equality across such calls holds to 1e−12 in float64, not bitwise [`test_dense_shapes_offsets_and_symmetry`].
- **Folded form.** `key_features(pos) -> [..., N, F]` and `query_features(pos) -> [..., H, N, F]`, with `F = feature_dim = sum_m phi(m)`. They satisfy `query_features(p) @ key_features(p).T == bias(p, p)` up to floating-point rounding: 1e−12 in float64, tested to positions near 1e15 [`test_folded_features_reproduce_dense_bias`].

### 3.2 Controls

| Module | Value | Init |
|---|---|---|
| `CosineBankBias(H, components)` | `gamma_h * sum_c alpha_{h,c} cos(2 pi a_c n / m_c)`, the same gated-mixture form as 3.1 | `gamma = 0` |
| `CosineBankBias.fundamental(orders)` | components `(m, 1)`; same periods and parameter count as ArithScope | |
| `CosineBankBias.full(orders)` | all reduced `(m, a)`, `a <= m/2` (41 for `M = 1..16`); its span contains every `K_m` [`test_ramanujan_span_is_inside_full_cosine_span`] | |
| `LearnedFrequencyBias(H, F)` | `sum_f w_{h,f} cos(omega_{h,f} n)`, free `omega`, float64 phases | `w = 0` |
| `RelativeBias(H, D)` | learned table, offset clipped to `[-D, D]` | 0 |
| `ALiBiBias(H)` | `-2^(-8(h+1)/H) * abs(n)`, fixed | – |

## 4. Attention (`arithscope.attention.CausalSelfAttention`)

```
forward(x [B, N, D], positions [N] | [B, N], key_padding_mask [B, N] (True = valid) | None,
        backend in {"explicit", "sdpa", "folded", "flex"}, return_attention=False)
```

**Logits.** `<q, k>/sqrt(d_head)`, with q and k rotated by RoPE when enabled, plus the sum of all configured biases.

**Mask.** Key `j` is allowed for query `i` iff `pos_j <= pos_i` and key `j` is valid. The mask is applied after the bias and before the softmax. A query row with no allowed key outputs exactly 0 and never NaN [`test_key_padding_and_fully_masked_rows`].

**Causality.** Outputs at positions `<= t` do not depend on inputs at positions `> t` [`test_causality`].

**Backends compute the same function.** They agree to 1e−10 in float64, with and without padding and with per-sequence positions [`test_backends_agree*`].
- `explicit` is the reference. It is the only backend that can return attention weights. Softmax runs in float32 for half-precision inputs, otherwise in the input dtype.
- `sdpa` passes the summed bias as `attn_mask`.
- `folded` requires the bias set to be exactly `{"arith"}` (RoPE allowed). It concatenates `sqrt(d_head) * query_features` to q and `key_features` to k, zero-pads q, k, and v to one head dimension (the next multiple of 8), calls SDPA with `scale = 1/sqrt(d_head)`, and drops the padded output columns. Zero columns change no dot product, so the padding is exact [`test_folded_backend_pads_head_dims_exactly`]. Without padding and with shared 1-D positions it uses `is_causal=True` and builds no `[N, N]` tensor of any kind. Equal head dimensions matter in practice: with q/k wider than v, CPU SDPA falls back to a path that materializes the attention matrix (measured in `experiments/bench_overhead.py`).
- `flex` uses FlexAttention (`torch.nn.attention.flex_attention`; module `arithscope.flex`). The summed bias is evaluated once on the offsets `−s … s`, where `s` is the largest position span of any sequence in the batch, and a `score_mod` reads it by offset; a `mask_mod` applies the mask rule above. It needs a PyTorch build that provides FlexAttention. Eager mode is the default; `arithscope.flex.compile_kernels()` switches to `torch.compile(flex_attention)`. On CPU only the forward pass is available (call it under `torch.no_grad()`). Its output matches `explicit` to 1e−12 in float64 in eager mode, with padding and per-sequence positions [`test_flex_backend_matches_explicit`], and to 1e−5 in float32 with compiled kernels [`test_compiled_flex_backend_matches_explicit`, opt-in via `ARITHSCOPE_TEST_COMPILED_FLEX=1`]. Gradients and GPU kernels are not tested in this repository.

**Zero-init equivalence.** If every additive channel is at its zero init, a `rope+X` model with the same shared weights produces bitwise-identical outputs and gradients to `rope` [`GammaZeroEquivalenceTests`]. The same holds for `arith` vs `nope` and `alibi+arith` vs `alibi`.

## 5. Known numerical boundaries

- **RoPE angles.** RoPE computes `pos * inv_freq` in `rope_angle_dtype` (default float32, as in common LLM code). Relative-position exactness therefore degrades at large absolute positions. This is a property of RoPE, not a bug of this package; the post-hoc shift test at the training length measures it (`experiments/posthoc_shift.py`, [`RESULTS.md`](RESULTS.md)).
- **Feature dimension.** `feature_dim` grows as `sum_m phi(m)`. The folded backend trades `O(H N^2)` bias memory for `F` extra q/k dimensions. Compute stays quadratic in `N`.

## 6. Model and harness

- **`TinyDecoder(ModelConfig)`.** Pre-LN, no absolute position embedding, untied head.
- **`count_parameters()`.** Splits parameters into positional and backbone.
- **`arithscope.train.run(RunSpec, out_dir)`.** Writes exactly one JSON file (schema `arithscope-run-1`), named `<experiment>__<variant>__s<seed>__<sha256(spec)[:12]>.json`. The file is written atomically. Existing files are never overwritten, so runs are resumable and immutable.
