"""Synthetic tasks with exact Bayes oracles (see docs/PREREGISTRATION.md).

All samplers take an explicit ``torch.Generator``; the same seed gives the same
batch on every machine, and test sets are hashed into the run logs.

* :class:`PeriodicDenoising` -- the target at position ``t`` is the hidden motif
  symbol ``mu[t mod P]``; inputs are noisy copies. The Bayes-optimal causal
  predictor is a plurality vote over positions ``j <= t`` with ``j = t (mod P)``:
  an attention head must place its mass on the arithmetic comb ``{t - kP}``.
* :class:`TaggedDenoising` -- the aperiodic control: identical vote structure,
  but group membership is a random content tag, not an arithmetic residue.
* :class:`AssociativeRecall` -- multi-query associative recall (MQAR, Arora et
  al., 2023), the standard synthetic probe of attention's recall ability.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import torch


@dataclass
class Batch:
    tokens: torch.Tensor      # [B, N] int64
    targets: torch.Tensor     # [B, N] int64
    loss_mask: torch.Tensor   # [B, N] bool, positions that carry a target
    oracle: torch.Tensor      # [B, N] float, Bayes-optimal probability of being correct (ties split)
    group: torch.Tensor       # [B] int64, per-sequence group (period for periodic tasks, else 0)
    observed: torch.Tensor | None = None  # [B, N] bool, positions whose token carries evidence (None = all)

    def digest(self) -> str:
        h = hashlib.sha256()
        for t in (self.tokens, self.targets, self.loss_mask):
            h.update(t.contiguous().numpy().tobytes())
        return h.hexdigest()


def _plurality_oracle(symbols: torch.Tensor, groups: torch.Tensor, truth: torch.Tensor, vocab: int,
                      observed: torch.Tensor | None = None) -> torch.Tensor:
    """Expected accuracy of a causal plurality vote within groups (uniform tie-breaking).

    ``symbols, groups, truth``: ``[B, N]``. For each ``t`` the vote counts ``symbols[j]``
    over ``j <= t`` with ``groups[j] == groups[t]`` and ``observed[j]`` (default: all ``j``).
    """
    b, n = symbols.shape
    num_groups = int(groups.max()) + 1
    onehot = torch.zeros(b, n, num_groups, vocab)
    weight = torch.ones(b, n) if observed is None else observed.float()
    onehot[torch.arange(b)[:, None], torch.arange(n)[None, :], groups, symbols.clamp(max=vocab - 1)] = weight
    counts = onehot.cumsum(dim=1)
    own = counts[torch.arange(b)[:, None], torch.arange(n)[None, :], groups]  # [B, N, V]
    best = own.max(dim=-1, keepdim=True).values
    winners = own == best
    hit = winners.gather(-1, truth.unsqueeze(-1)).squeeze(-1).float()
    return hit / winners.sum(-1).float()


@dataclass(frozen=True)
class PeriodicDenoising:
    """Periodic denoising; with ``query_rate > 0`` it becomes periodic *retrieval* with query tokens.

    With ``query_rate = q``, each position is independently a query (probability ``q``) showing
    the uninformative token ``QUERY = vocab_size``. Only queries are scored, and the target is
    still ``mu[t mod P]``. A query's own token then carries no evidence, so the comb vote over the
    *observed* positions is the exact information ceiling for a positional aggregator
    (used for the alias fixture, docs/PREREGISTRATION.md).
    """

    periods: tuple[int, ...] = (12,)
    vocab_size: int = 16
    noise: float = 0.5
    query_rate: float = 0.0

    name = "periodic"

    @property
    def model_vocab(self) -> int:
        return self.vocab_size + (1 if self.query_rate > 0 else 0)

    def sample(self, batch: int, length: int, gen: torch.Generator) -> Batch:
        periods = torch.tensor(self.periods)[torch.randint(len(self.periods), (batch,), generator=gen)]
        max_p = max(self.periods)
        motif = torch.randint(self.vocab_size, (batch, max_p), generator=gen)
        t = torch.arange(length)
        residue = t[None, :] % periods[:, None]
        truth = motif.gather(1, residue)
        noise_tok = torch.randint(self.vocab_size, (batch, length), generator=gen)
        flip = torch.rand(batch, length, generator=gen) < self.noise
        tokens = torch.where(flip, noise_tok, truth)
        observed, scored = None, torch.ones(batch, length, dtype=torch.bool)
        if self.query_rate > 0:
            scored = torch.rand(batch, length, generator=gen) < self.query_rate
            observed = ~scored
            tokens = torch.where(scored, torch.full_like(tokens, self.vocab_size), tokens)
        oracle = _plurality_oracle(tokens, t[None, :] % periods[:, None], truth, self.vocab_size, observed)
        return Batch(tokens, truth, scored, oracle, periods, observed)

    def vote_oracle(self, batch: Batch, vote_period: int) -> torch.Tensor:
        """Expected accuracy of a plurality vote over the ``vote_period``-comb instead of the true one.

        With ``vote_period = gcd(P, lcm(bank))`` this is the best a positional aggregator
        restricted to an ArithScope bank can do when the bank aliases the true period
        (docs/MATH.md, P1/T4).
        """
        t = torch.arange(batch.tokens.shape[1])
        groups = (t[None, :] % vote_period).expand_as(batch.tokens)
        return _plurality_oracle(batch.tokens, groups, batch.targets, self.vocab_size, batch.observed)


@dataclass(frozen=True)
class TaggedDenoising:
    num_tags: int = 12
    vocab_size: int = 16
    noise: float = 0.5

    name = "tagged"

    @property
    def model_vocab(self) -> int:
        return self.num_tags * self.vocab_size

    def sample(self, batch: int, length: int, gen: torch.Generator) -> Batch:
        tags = torch.randint(self.num_tags, (batch, length), generator=gen)
        motif = torch.randint(self.vocab_size, (batch, self.num_tags), generator=gen)
        truth = motif.gather(1, tags)
        noise_tok = torch.randint(self.vocab_size, (batch, length), generator=gen)
        flip = torch.rand(batch, length, generator=gen) < self.noise
        symbols = torch.where(flip, noise_tok, truth)
        tokens = tags * self.vocab_size + symbols
        oracle = _plurality_oracle(symbols, tags, truth, self.vocab_size)
        return Batch(tokens, truth, torch.ones(batch, length, dtype=torch.bool), oracle,
                     torch.zeros(batch, dtype=torch.long))


@dataclass(frozen=True)
class AssociativeRecall:
    """``k1 v1 ... kn vn  q1 a1 ... qm am`` with ``n = m = length / 4``; targets at query keys."""

    num_keys: int = 512
    num_values: int = 64

    name = "mqar"

    @property
    def model_vocab(self) -> int:
        return self.num_keys + self.num_values

    def sample(self, batch: int, length: int, gen: torch.Generator) -> Batch:
        if length % 4:
            raise ValueError("length must be a multiple of 4")
        pairs = length // 4
        if pairs > self.num_keys:
            raise ValueError("not enough distinct keys for this length")
        keys = torch.argsort(torch.rand(batch, self.num_keys, generator=gen), dim=1)[:, :pairs]
        values = torch.randint(self.num_values, (batch, pairs), generator=gen)
        query_idx = torch.randint(pairs, (batch, pairs), generator=gen)
        q_keys, q_vals = keys.gather(1, query_idx), values.gather(1, query_idx)
        val_tok = self.num_keys + values
        tokens = torch.stack([torch.cat([keys, q_keys], 1), torch.cat([val_tok, self.num_keys + q_vals], 1)], -1)
        tokens = tokens.reshape(batch, length)
        targets = torch.zeros(batch, length, dtype=torch.long)
        mask = torch.zeros(batch, length, dtype=torch.bool)
        query_pos = torch.arange(2 * pairs, length, 2)
        targets[:, query_pos] = self.num_keys + q_vals
        mask[:, query_pos] = True
        return Batch(tokens, targets, mask, mask.float(), torch.zeros(batch, dtype=torch.long))


TASKS = {"periodic": PeriodicDenoising, "tagged": TaggedDenoising, "mqar": AssociativeRecall}


def make_task(kind: str, **kwargs):
    if kind not in TASKS:
        raise ValueError(f"unknown task {kind!r}; expected one of {sorted(TASKS)}")
    if "periods" in kwargs:
        kwargs["periods"] = tuple(kwargs["periods"])
    return TASKS[kind](**kwargs)


def task_dict(task) -> dict:
    return {"kind": task.name, **asdict(task)}
