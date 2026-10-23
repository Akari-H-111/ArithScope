# Primary sources and version boundaries

Accessed 2026-10-15. No exhaustive novelty claim is made.

1. Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim,
   *Parallelizing Linear Transformers with the Delta Rule over Sequence
   Length*, NeurIPS 2024. [Conference PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf);
   [final arXiv version v6, 2025-01-15](https://arxiv.org/abs/2406.06484v6).
   Section 2.2 supplies the delta update, its online squared-regression
   interpretation, and the fast-weight reading rule. Its original DeltaNet
   attribution is to earlier work by Schlag and colleagues. This study does
   not reproduce the paper's language-model training or hardware kernels.
2. Songlin Yang, Jan Kautz, Ali Hatamizadeh, *Gated Delta Networks: Improving
   Mamba2 with Delta Rule*, ICLR 2025.
   [Conference PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf).
   Equation 10 is the gated recurrence used in the mathematical discussion;
   it decays the previous state before the residual update. The paper separates
   memory clearing from key-specific corrections. No hybrid-model performance
   claim is transferred to ArithScope.
3. Ngoc Bui, Tinglin Huang, Rex Ying, *Kalman Delta Networks: Uncertainty-aware
   Associative Memory*, arXiv preprint, v2 dated 2026-09-29.
   [Pinned version](https://arxiv.org/abs/2609.07816v2);
   [full HTML](https://arxiv.org/html/2609.07816v2).
   Reformulates associative memory under a linear-Gaussian state-space model,
   tracks uncertainty, and derives covariance-dependent residual gains with
   practical approximations. This is direct prior work for uncertainty-aware
   delta memory; the idea is not claimed as new here. Its assumptions and
   evaluations differ from this study's categorical periodic denoising law.

## Raw repository material

- `arithscope/arith.py`: exact normalized kernels, divisor weights, and totients.
- `arithscope/bias.py`: integer-residue real primitive features. The symmetric
  square-root construction here is distinct from the asymmetric score folding.
- `docs/MATH.md` T3, T7, and P3: existing comb optimum, feature factorization,
  and bounded final-readout argument. New conditional deductions stay local.
- `arithscope/tasks.py`: registered noisy motif law. All new observations are
  generated independently in this branch and saved in its numerical JSON.
- The previous metacognition branch supplies a caution about correlated count
  evidence; its files are protected by this study's initial hash snapshot.
- `inputs/manifest.json` pins repository source bytes and the paper versions.
  Parent E9 inputs, checkpoints, and scores are not consumed.

