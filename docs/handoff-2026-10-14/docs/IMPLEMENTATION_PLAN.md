# Implementation plan: smallest verifiable increment first

## Stage 0: recovery and environment (gate G0)

- Re-run 13 included reference tests, capture interpreter version, OS, git status and command output.
- Inspect pinned sources in `SOURCES_MANIFEST.json`; read minimal relevant math and code. Do not mutate them. Record unreadable sources.
- Initialize new ArithScope working tree locally **only**, using independent venv and explicit optional PyTorch version when actually installed. Do not assume CUDA or Apple MPS.
- Output: environment report, exact method scopes, source access ledger.

## Stage 1: mathematical oracle + attention bias (G1)

- Preserve `reference/` as a no-dependency oracle. Produce PyTorch implementation `arithscope/arithmetic_bias.py`, with configurable positive order bank `orders`, `heads`, optional causal mode, base positions and dtype/device handling.
- Build exact lookup for `c_m(d)` over a finite integer displacement range; store as registered buffers, normalized by `phi(m)`, with clearly defined caching and extension on longer sequences. Do not convert huge unbounded sequence lengths into O(N²) storage unknowingly.
- `alpha=softmax(order_logits,dim=-1)`. Per-head `gamma` separate, default zero. `bias[H,Nq,Nk]` or batch-repeated as needed. Non-negative `alpha` is for PSD of the **kernel mixture**; gamma can be negative, making the *bias* non-PSD, which is acceptable and should not be confused.
- Public API example (adjust only after checking framework):
  ```python
  class ArithmeticBias(torch.nn.Module):
      def forward(self, query_positions: torch.LongTensor,
                  key_positions: torch.LongTensor) -> torch.Tensor:
          """Return real [heads, Nq, Nk] additive logit bias."""
  ```
- Construct `arithscope/attention.py` using stable attention primitive if feature integration allows; start with explicit `QK^T + bias` baseline for transparency and correctness, then examine `scaled_dot_product_attention` compatibility.
- Required tests: exact reference integer values, real/complex parity not needed in tensor path, kernel symmetry, Nq≠Nk, position offsets (KV cache), Causal mask, key padding, all-legal rows, gradients on both `gamma` and `order_logits` (with gamma nonzero), exact gamma=0 control, fp32/fp64 CPU. If CUDA/MPS present, run their applicable tests and label tolerances.
- Example expected fail/trap: with gamma=0, order-logit gradients can be zero; do not count that as a differentiability bug. Check order-logit gradients at nonzero gamma.
- Gate: all tests passing and no semantic regression. No training claim.

## Stage 2: small real Transformer + baselines (G2)

- Minimal pre-norm decoder-only or encoder stack (choose and document), real learnable Q/K/V, FFN, LN, residual, token embeddings, output logits, training loss and generation/test routine. Do not implement both architectures initially.
- Matched controls with identical seed/model size/training budget: base positional scheme, learned sinusoidal/Fourier bias with same budget, RoPE, ALiBi; FoPE secondary if reproducible licensed source and implementation verified.
- Avoid comparing a positional-bias-enhanced candidate only against a model with no position signal. For each baseline report parameter count, FLOPs or estimated math ops, kernel backend, actual runtime and memory.
- Gate: reproducible training script; model trains on a tiny overfit case; no catastrophic masks/NaNs; gamma=0 reproduces control path.

## Stage 3: controlled tasks (G3)

- Frozen synthetic tasks and seed lists from `docs/EXPERIMENT_PROTOCOL.md`.
- Record matched paired performance across multiple seeds and lengths. Incorporate known-hard `H={6,10}` vs `{6,10,15}` and adversarial alias cases. Include **aperiodic sampling** as a separate experimental control, not secretly as part of the arithmetic model.
- Gate: test-set measurement with persistent per-run JSON, negative cases retained, checkpoint provenance.

## Stage 4: first public research report candidate (G4)

- Create plots from run logs only; avoid hand-edited numbers.
- Re-evaluate H1/H2 against strongest baselines, overhead, extrapolation and robustness. If negative, publish an accurately scoped finding only upon authorization. Only after a real benchmark propose a LinkedIn description stronger than 'in progress'.

## Optional, isolated follow-up experiments (never block G1–G3)

1. Saturated-lattice coordinate bank and short noise-aware Bézout certificates (AHR). Compare against random or learned Fourier bank of equal width.
2. Cyclotomic descendant prior applied to a justified tree of experts or memory pages. Without interpretable order labels, treat assignment as engineered and ablate it.
3. Evidence-calibrated root routing with training-only root priors and independent held-out evidence; logsumexp identities are not a surrogate for valid likelihood ratios.
4. SAMT-inspired frame analysis and H-symmetrized dynamics; test actual sample-space norms and convergence.
5. Marked-response rank closure, retrieval augmentation and progressive memory packets; do not confuse this with a formal neural memory capacity theorem.

## Engineering disciplines

- Small diffs; pure math separated from tensor operations; public APIs documented; no duplicate reconstruction engines.
- Defensive input validation for integer orders, shapes, device, dtype, sequence lengths, masks and no legal key positions.
- Avoid dense per-head complex embeddings if real integer-kernel lookup suffices; record allocation and timing.
- Version outputs; freeze source and benchmark configs; do not rewrite prior experiments.
