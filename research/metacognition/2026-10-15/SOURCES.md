# Primary sources and their limits

Accessed 2026-10-15. These are conceptual and methodological inputs. None
establishes an ArithScope result. No claim of an exhaustive novelty search.

1. Stephen M. Fleming and Nathaniel D. Daw (2017), *Self-evaluation of
   decision-making: A general Bayesian framework for metacognitive
   computation*. [Author institution's record and abstract](https://collaborate.princeton.edu/en/publications/self-evaluation-of-decision-making-a-general-bayesian-framework-f/).
   Models confidence and error detection as second-order inference about a
   decision system. Provides the monitoring interpretation, not a neural
   backward-pass mechanism or a controller guarantee.
2. Rui Huang, Andrew Geng, and Yixuan Li (2021), *On the Importance of Gradients
   for Detecting Distributional Shifts in the Wild*.
   [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/063e26c670d07bb7c4d30e6fc69fe056-Paper.pdf).
   GradNorm uses gradients of KL from a uniform target, without a true class
   label. Its analysis relates last-layer gradient scores to features and
   outputs. Its reported task is OOD detection, not periodic denoising,
   long-context calibration, or an output correctness certificate.
3. Yu Sun et al. (2020), *Test-Time Training with Self-Supervision for
   Generalization under Distribution Shifts*.
   [ICML paper](https://proceedings.mlr.press/v119/sun20b.html).
   Constructs a self-supervised task from an unlabeled test sample and updates
   parameters before prediction. Motivates distinguishing monitoring from
   actual adaptation. The reported improvements are in image classification.
4. Yonatan Geifman and Ran El-Yaniv (2019), *SelectiveNet: A Deep Neural
   Network with an Integrated Reject Option*.
   [ICML paper](https://proceedings.mlr.press/v97/geifman19a.html).
   Jointly learns prediction and rejection; motivates reporting the
   risk-coverage tradeoff when an output gate abstains.
5. Nils Gruenefeld, Jes Frellsen, and Christian Hardmeier (2026), *An Isotropic
   Approach to Efficient Uncertainty Quantification with Gradient Norms*.
   [Probabilistic ML proceedings](https://proceedings.mlr.press/v327/grunefeld26a.html).
   Uses a first-order approximation and isotropic parameter-covariance
   assumption to relate uncertainty to gradient norms. Reports benchmark
   dependence in answer-correctness discrimination. It does not establish a
   universal correctness interpretation for a gradient norm.

## Repository raw sources

- `arithscope/tasks.py`: exact registered replacement-noise law and causal vote
  groups. `experiments/posthoc_nll.py`: existing Bayes posterior computation.
- `arithscope/model.py`, `attention.py`, `bias.py`, and `train.py`: the actual
  network, proxy-gradient sites, and checkpoint schema. Parent source hashes
  are pinned in `inputs/manifest.json`.
- Two copied E6 seed-1 ArithScope-only checkpoint tensors and original run JSON
  files, identified in that manifest. The original run JSON does not contain a
  checkpoint SHA-256: the link is checked through filename and full run spec,
  and the copied tensor bytes are sealed for the first time by this study.
- The four plus four fresh sequence records in `results/gradient_v1.json` are
  raw inputs for this branch. Main E9 observations are never consumed.

