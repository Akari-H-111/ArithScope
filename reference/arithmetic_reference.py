"""ArithScope: tiny dependency-free exact/finite numerical reference.

This is a mathematical oracle and smoke-test substrate, NOT a trained
Transformer, a CUDA attention kernel, or evidence of benchmark improvement.
"""
from __future__ import annotations

from math import cos, exp, gcd, isfinite, log, pi, sin, sqrt
from typing import Iterable, Sequence


def _positive(n: int, label: str) -> None:
    if type(n) is not int or n < 1:
        raise ValueError(f"{label} must be a positive integer")


def divisors(n: int) -> list[int]:
    _positive(n, "n")
    lower, upper = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            lower.append(d)
            if d * d != n:
                upper.append(n // d)
        d += 1
    return lower + upper[::-1]


def mobius(n: int) -> int:
    _positive(n, "n")
    remaining, parity, p = n, 0, 2
    while p * p <= remaining:
        if remaining % p == 0:
            remaining //= p
            parity ^= 1
            if remaining % p == 0:
                return 0
            while remaining % p == 0:
                remaining //= p
        p += 1
    if remaining > 1:
        parity ^= 1
    return -1 if parity else 1


def euler_phi(n: int) -> int:
    _positive(n, "n")
    remaining, result, p = n, n, 2
    while p * p <= remaining:
        if remaining % p == 0:
            result -= result // p
            while remaining % p == 0:
                remaining //= p
        p += 1
    if remaining > 1:
        result -= result // remaining
    return result


def ramanujan_sum(m: int, displacement: int) -> int:
    """Exact integer c_m(n), including negative n and m=1."""
    _positive(m, "m")
    if type(displacement) is not int:
        raise ValueError("displacement must be an integer")
    return sum(d * mobius(m // d) for d in divisors(gcd(m, displacement)))


def normalized_kernel(m: int, t: int, s: int) -> float:
    _positive(m, "m")
    if type(t) is not int or type(s) is not int:
        raise ValueError("positions must be integers")
    return ramanujan_sum(m, t - s) / euler_phi(m)


def primitive_feature(m: int, t: int) -> tuple[complex, ...]:
    _positive(m, "m")
    if type(t) is not int:
        raise ValueError("position must be integer")
    scale = sqrt(euler_phi(m))
    return tuple(complex(cos(2*pi*a*t/m), sin(2*pi*a*t/m)) / scale
                 for a in range(1, m+1) if gcd(a, m) == 1)


def identifiable_modulus(length: int, harmonic_labels: Sequence[int]) -> int:
    """Smallest positive position period for F_H(t) on Z/LZ."""
    _positive(length, "length")
    if not harmonic_labels:
        raise ValueError("nonempty harmonic labels required")
    g = 0
    for h in harmonic_labels:
        _positive(h, "harmonic label")
        g = gcd(g, h)
    return length // gcd(length, g)


def phase_vector(length: int, harmonic_labels: Sequence[int], t: int) -> tuple[complex, ...]:
    _positive(length, "length")
    return tuple(complex(cos(2*pi*((h*t) % length)/length),
                         sin(2*pi*((h*t) % length)/length)) for h in harmonic_labels)


def descendants(D: int, m: int) -> tuple[int, ...]:
    _positive(D, "D")
    _positive(m, "m")
    if D < 2:
        raise ValueError("D must be >= 2")
    return tuple(r for r in range(1, D*m + 1) if r // gcd(r, D) == m)


def descendant_prior(D: int, m: int) -> dict[int, float]:
    denom = D * euler_phi(m)
    return {r: euler_phi(r) / denom for r in descendants(D, m)}


def _logsumexp(scores: Sequence[float]) -> float:
    if not scores:
        raise ValueError("at least one score needed")
    if any(not isfinite(v) for v in scores):
        raise ValueError("all scores must be finite")
    shift = max(scores)
    return shift + log(sum(exp(v - shift) for v in scores))


def prior_softmax(D: int, m: int, logits: dict[int, float]) -> dict[int, float]:
    """Normalized exp(log prior + content logit) over admissible descendants."""
    prior = descendant_prior(D, m)
    if set(logits) != set(prior):
        raise ValueError("logits keys must exactly match admissible descendants")
    keys = list(prior)
    scores = [log(prior[r]) + float(logits[r]) for r in keys]
    z = _logsumexp(scores)
    return {r: exp(value - z) for r, value in zip(keys, scores)}


def normalized_white_features(theta: Sequence[float], harmonic_labels: Sequence[int]) -> tuple[complex, ...]:
    """Finite POINTWISE Parseval identity, not a sampled continuum g-frame theorem."""
    if not theta or not harmonic_labels:
        raise ValueError("nonempty coordinates and labels required")
    raw: list[complex] = [1+0j]
    for q in harmonic_labels:
        _positive(q, "harmonic label")
        for v in theta:
            if not isfinite(v):
                raise ValueError("phase coordinates must be finite")
            raw.append(complex(cos(2*pi*q*v)-1, sin(2*pi*q*v)))
    scale = sqrt(sum(abs(v)**2 for v in raw))
    return tuple(v / scale for v in raw)


def mixture_responsibilities(log_priors: Sequence[float], scores: Sequence[float]) -> tuple[float, ...]:
    """Gradient of logsumexp(log_prior_r + score_r) wrt score_r."""
    if len(log_priors) != len(scores) or not scores:
        raise ValueError("shape mismatch or empty mixture")
    vals = [a + b for a,b in zip(log_priors, scores)]
    z = _logsumexp(vals)
    return tuple(exp(v - z) for v in vals)
