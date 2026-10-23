# A deeper research request

## 1. What a specialist would add

A specialist would first separate three hypotheses: a backward pass provides
information, that information predicts the value of an intervention, and the
intervention improves predictions after its cost is counted. They would test
identifiability and failure cases before adding a controller. They would not
treat error detection, probability estimation, abstention, and adaptation as
interchangeable outcomes.

## 2. The raw material

Start from the token-generating law, individual observed symbols, hidden motif
labels used only for scoring, the actual forward/loss code, frozen parameter
tensors, original run JSON files, and primary research papers. E6 checkpoint
copies are available here; the local checkout does not contain E8 checkpoints.
Main-project summaries establish context, not the evidence for this branch.

## 3. Methods and tools

Use sufficient statistics and conditional information, proper scoring rules,
analytic differentiation, replication counterexamples, exact rational posterior
enumeration, independent direct-likelihood calculations, PyTorch float64
autograd, central finite differences, deterministic fresh observations, unit
tests, SHA-256 manifests, and replay. Use the existing interpreter and standard
library; install nothing. Restrict neural probes to two frozen checkpoints and
short causal prefixes, with one thread and low priority.

## 4. What could establish correctness independently

An exact Bayes oracle, independent likelihood enumeration, a second derivative
implementation, counterexamples, saved observations and hashes, deterministic
replay, and a reviewer checking the assumptions can establish the bounded
results. A successful implementation check cannot establish controller benefit.
That requires a new prospective experiment with unseen data, matched budgets,
independent sequences and seeds, and a fixed decision rule.

## Rewritten prompt

In an isolated subdirectory of ArithScope, investigate whether label-free
backward-pass monitoring can add information beyond forward evidence for
controlling long-sequence predictions. Make the task deeper by identifying the
precise statistical question, rather than expanding the architecture or the
benchmark list. Start from the generating law, raw observations, frozen weights,
and primary papers. Derive sufficient statistics, redundancy results, and
counterexamples; test them using exact enumeration and independent derivative
checks. Distinguish monitoring from adaptation and from selective output.
Determine what evidence quantity a useful controller would need, and when that
quantity fails under dependence. Run a small frozen-checkpoint diagnostic only
where it answers this question. Preserve source hashes, observations, results,
failures, and replay commands. Label proof, assumptions, finite checks,
numerical checks, development observations, and open claims separately. Leave
a concrete handoff for another chat to judge possible integration; do not
change, retrain, or reinterpret the running main experiment.

