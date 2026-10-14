"""Numerical companions to docs/MATH.md (no training involved).

* :func:`comb_margin_lp` -- best comb margin attainable by a convex mixture of
  fixed periodic components (a linear program; Theorem T3 predicts the optimum).
* :func:`comb_mass` -- fraction of softmax attention on the comb ``{n : P | n}``
  for a positional score over offsets ``0..N-1``.
* :func:`fit_rope_comb` -- the best positional-only RoPE score for a ``P``-comb on a
  training window under a logit budget; used to show residue-blindness (T5) at
  longer windows.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import arith


def comb_margin_lp(components: Sequence[Sequence[float]]) -> tuple[float, np.ndarray]:
    """Maximize ``1 - max_{n=1..P-1} sum_j a_j f_j(n)`` over ``a >= 0, sum a = 1``.

    Every component must be ``P``-periodic with ``f_j(0) = 1`` and is given on one period.
    """
    from scipy.optimize import linprog

    feats = np.asarray(components, dtype=float)
    if not np.allclose(feats[:, 0], 1.0):
        raise ValueError("components must equal 1 at offset 0")
    j, p = feats.shape
    cost = np.zeros(j + 1)
    cost[-1] = -1.0  # maximize t
    a_ub = np.hstack([feats[:, 1:].T, np.ones((p - 1, 1))])
    a_eq = np.hstack([np.ones((1, j)), np.zeros((1, 1))])
    res = linprog(cost, A_ub=a_ub, b_ub=np.ones(p - 1), A_eq=a_eq, b_eq=[1.0],
                  bounds=[(0, None)] * j + [(None, None)], method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(-res.fun), res.x[:j]


def margin_row(period: int) -> dict:
    """Comb margins for one period across bias families (all PSD, unit diagonal)."""
    offs = range(period)
    ramanujan = [[float(arith.kernel(d, n)) for n in offs] for d in arith.divisors(period)]
    fundamental = [[math.cos(2 * math.pi * n / d) for n in offs] for d in arith.divisors(period)]
    return {
        "period": period,
        "optimal_pd": period / (period - 1),
        "ramanujan_bank": comb_margin_lp(ramanujan)[0],
        "fundamental_cosine_bank": comb_margin_lp(fundamental)[0],
        "single_ramanujan": float(arith.comb_gap(period)),
        "single_cosine": 1 - math.cos(2 * math.pi / period),
    }


def comb_mass(scores: np.ndarray, period: int) -> np.ndarray:
    """Comb attention mass for every window prefix: ``out[N-1] = mass on {n < N : P | n}``."""
    w = np.exp(scores - scores.max())
    comb = np.where(np.arange(len(scores)) % period == 0, w, 0.0)
    return np.cumsum(comb) / np.cumsum(w)


def rope_frequencies(head_dim: int, base: float = 10000.0) -> np.ndarray:
    return base ** (-np.arange(0, head_dim, 2) / head_dim)


def rope_scores(coef: np.ndarray, thetas: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Positional-only RoPE logit ``sum_k Re(coef_k exp(i n theta_k))`` (float64 phases)."""
    phase = np.outer(offsets.astype(np.float64), thetas)
    return (np.cos(phase) * coef.real - np.sin(phase) * coef.imag).sum(axis=1)


def fit_rope_comb(period: int, window: int, head_dim: int, budget: float,
                  base: float = 10000.0, steps: int = 3000, seed: int = 0) -> np.ndarray:
    """Coefficients maximizing the log comb mass on offsets ``0..window-1``.

    The logit budget ``sum_k |coef_k| <= budget`` bounds ``|score| <= budget``, the
    same bound that ``budget * K_P`` satisfies, so the comparison is at equal scale.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    thetas = torch.tensor(rope_frequencies(head_dim, base))
    offsets = torch.arange(window, dtype=torch.float64)
    phase = offsets[:, None] * thetas[None, :]
    cos, sin = torch.cos(phase), torch.sin(phase)
    comb = (torch.arange(window) % period == 0)
    re = torch.randn(len(thetas), generator=gen, dtype=torch.float64).mul_(0.1).requires_grad_()
    im = torch.randn(len(thetas), generator=gen, dtype=torch.float64).mul_(0.1).requires_grad_()
    opt = torch.optim.Adam([re, im], lr=0.05)
    for _ in range(steps):
        scores = cos @ re - sin @ im
        loss = -(torch.logsumexp(scores[comb], 0) - torch.logsumexp(scores, 0))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # project onto the budget ball (l1 norm of moduli)
            mod = torch.sqrt(re**2 + im**2)
            total = mod.sum()
            if total > budget:
                re.mul_(budget / total)
                im.mul_(budget / total)
    return (re.detach() + 1j * im.detach()).numpy()


def residue_mass_limit_check(head_dim: int, period: int, length: int, seed: int = 0) -> np.ndarray:
    """Comb mass for random RoPE coefficients; Theorem T5 predicts convergence to ``1 / period``."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(size=head_dim // 2) + 1j * rng.normal(size=head_dim // 2)
    scores = rope_scores(coef, rope_frequencies(head_dim), np.arange(length))
    return comb_mass(scores, period)
