# Novelty and nearest-neighbor literature screen

Research status: a scoped reconnaissance, **not an exhaustive 2026 novelty or patent search**. Existing facts may change as research is published. Keep priority claims tentative.

## Important nearby literature

- Brunner et al. (2019), *On Identifiability in Transformers*, https://arxiv.org/abs/1908.04211. Studies attention-weight and contextual-embedding identifiability. **Different object:** ArithScope's first exact theorem concerns injection/collisions of a specified integer Fourier **position feature bank**, not identifiability of a learned attention weight matrix.
- Dufter, Schmitt & Schütze (2022), *Position Information in Transformers: An Overview*, https://aclanthology.org/2022.cl-3.7/. Position mechanisms and fair comparisons are long-established.
- Chen et al. (EMNLP 2021), *A Simple and Effective Positional Encoding for Transformers*, https://aclanthology.org/2021.emnlp-main.236/. Adding position information at the attention layer is **not new**.
- Cheng et al. (AAAI 2024), *Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning*, https://doi.org/10.1609/aaai.v38i10.29033. AMFormer is a named arithmetic-interaction Transformer, for tabular data; don't market all 'arithmetic attention' as new or identical.
- Hua et al. (ICML 2025), *FoPE* (Fourier Position Embedding), https://proceedings.mlr.press/v267/hua25b.html. Verify details before coding comparisons. Never cite as a defeated baseline without actually running it.
- Ramanujan subspaces/sums for periodic signal representation: original signal-processing literature exists, e.g. https://authors.library.caltech.edu/records/3c8eh-xc275 . Ramanujan sums are **classical**; potential novelty is transformer-specific compositional use, explicit collision testing and measured comparisons.
- Singer, angular synchronization from noisy phase observations, https://arxiv.org/abs/0905.3174; see also AHR's cited multi-frequency phase-unwrapping references. Bézout and phase reconstruction are not exclusive inventions of this project.
- Performer (positive random features) and efficient attention are independent prior directions; PSD arithmetic bias does **not** entail Performer-like linear complexity.
- Sparse attention has current 2026 hardware-aware research, e.g. *TileSparse* https://proceedings.mlr.press/v306/wang26lx.html. Real runtime cost must be measured rather than inferred from sparse algebra.

## Differentiation targets (proposed, not proven novel)

1. Source-to-model pipeline that couples **exact position-identifiability certificates** with trainable arithmetic attention bias and held-out alias benchmarks.
2. A proper comparison of the **full phase bank vs. compressed real character-trace kernel** under noise, mixed periods and long-length generalization.
3. Noise-conditioned Bézout bank selection coupled to actual attention, with matched learned Fourier controls.
4. Optional evidence-aware route selection with valid held-out likelihood ratio interpretation and explicit statistical gates, not confidence theatre.
5. Independent symbolic/reference verifier plus model-level/gradient/performance evidence and negative-result preservation.

Novelty decision is conditional on a final literature sweep immediately before public paper claims.

## Name screen

Working project **ArithScope** was chosen instead of `CycloFormer` (name already used by other Transformer research) or ambiguous `Arithmetic Attention` (present in several works). A quick general web query did not surface a clearly dominant identical *ArithScope Transformer research* title. This is **not** a trademark/registry/URL availability assertion.
