# Codex: begin the ArithScope research implementation

You are working on **ArithScope: Identifiability-Aware Arithmetic Attention for Transformers**, a model-internals and computational mathematics research project.

## Hard scope

1. Work in a **new isolated ArithScope workspace**; do not modify any source research repository, sealed artifact, checked-in result, paper, or private historical data. Do not automatically create a public GitHub repo or publish results.
2. Read `README.md`, `AGENTS.md`, `docs/MATH_SPEC.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/CLAIM_LEDGER.md`, and `docs/SOURCE_PROVENANCE.md` before altering any supplied file.
3. Treat the supplied `reference/` module and 13 tests as **finite mathematical reference only**, not as a trained attention implementation or experimental demonstration. Run them first and report the actual result. If there is a failure, diagnose it before development.
4. The objective is to develop **an actual learnable PyTorch Transformer attention component** and compare it against matched baselines. Avoid a mathematics-only decorative demo.
5. Prioritize **H1/H2**: exact integer-phase identifiability and cyclotomic/Ramanujan positional bias. Evidence-aware root-mixture, frame memory, rank-aware memory, and visualization are **future stages** and should not delay a minimal integrated model.
6. If external source files are inaccessible, identify their exact missing paths; continue with the self-contained mathematical specification. Never invent repository contents or claim a verifier ran when it did not.

## First action sequence (do it, not merely plan it)

**Stage 0: preflight**
- Check project working tree, runtime, Python/PyTorch versions and device support; do not overwrite unrelated files.
- Run `python3 -m unittest discover -s tests -v` and record environment and result.
- Create a concise `docs/EXECUTION_LEDGER.md` listing completed/not-run/failed scopes; do not copy old AHR performance values into ArithScope result rows.
- Read GitHub source paths and pinned SHAs from `SOURCES_MANIFEST.json`, choosing the smallest needed subset; do not ingest entire historical archives.

**Stage 1: implement the minimal computation layer**
- Build `arithscope/` as a small Python package; use PyTorch if installed, otherwise keep dependency discovery separate and explain the blocker.
- Implement `ArithmeticBias(nn.Module)` with integer positions and finite orders `M`, precomputed exact integer `c_m(shift)` lookups plus floating normalization by `phi(m)`. Make `alpha` nonnegative and normalized (e.g. softmax of learned order logits), `gamma` separately learnable and freely signed unless an explicit bound is selected.
- Integrate it into a causal Transformer attention layer **without changing the Q/K/V or mask semantics**; provide a disabled/baseline path with the same weights.
- Check `gamma=0` produces same output and gradients as the control within dtype/backend tolerance. Ensure causal/padding masks are applied consistently and no NaNs appear on legal attention rows.
- Include finite mathematical tests from `reference/` as independent oracles, plus tensor-shape, causal, mask, batch, gradient, CPU, non-contiguous and long-position tests.

**Stage 2: controlled experiments**
- Implement baselines: (a) ordinary attention with matched position handling, (b) a same-budget sinusoidal/Fourier additive bias, (c) RoPE, (d) ALiBi. FoPE can follow if verified implementation is available. Keep architectural differences and trainable parameter counts in a metadata table.
- Implement **first** two pre-registered synthetic tasks: modular delayed-copy retrieval and conflicting-period retrieval with distractors. Include a nonperiodic control and hard aliases.
- Separate token-position identifiability tests from genuine end-to-end train/eval benchmarks. Pre-register data generation and freezes; hold out test periods/lengths and avoid test-set tuning.
- Report accuracy, token/sequence loss (as appropriate), train/inference runtime, peak device memory, sequence length, mask type, head configuration, device/dtype, seed, data hash, and paired baseline deltas. Preserve failures.

**Stage 3: decide whether to expand**
- Only after tests and matched baselines, consider AHR's saturated lattice/Bézout phase certificates to choose order banks, and its independent train/held-out split protocol.
- Treat optional evidence-aware routing as a **different experimental hypothesis**, not an e-value guarantee for ordinary attention weights.
- Only after reproducible model-level evidence consider adding long-context language modeling, root-mixture routing, framed memory, and public presentation.

## Required first milestone deliverables

A small runnable PyTorch attention module and minimal Transformer, tests for exact reference match + masks + gradients + zero-bias equivalence, synthetic benchmark generator and baseline, version/seed metadata, status ledger, and a candid numerical result or documented failure. Keep complexity O(N²) if constructing full logits; never advertise a linear-attention speedup by algebraic analogy.

## Independent reviewer questions you must answer

- What new computation did we add to **QK attention logits** or routing, instead of just changing token preprocessing?
- Which theorems are exact identities and which claims require data or training?
- In what precise regime is the positional feature injective; are we mistakenly equating `K_m` with the full phase-feature map?
- Do the outputs and gradients match the baseline when `gamma=0`?
- Does the method beat a matched Fourier bias rather than just outperforming a weaker no-position control?
- Are the claims substantiated by reproducible files from **this** new experiment?

Proceed to execution without asking the user to repeat already stated requirements. Do not request permission to do ordinary read-only or local work. Ask before actions that publish, overwrite other repositories, incur material paid costs, or require credentials.
