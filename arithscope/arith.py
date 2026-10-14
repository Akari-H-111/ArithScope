"""Exact integer arithmetic behind ArithScope (no torch, no floats).

Every function here returns integers or ``fractions.Fraction`` values. The
PyTorch modules in :mod:`arithscope.bias` are tested against these functions,
and the theorems in ``docs/MATH.md`` are checked exhaustively with them.

Notation
--------
``c_m(n)``  Ramanujan sum, ``sum_{1<=a<=m, gcd(a,m)=1} exp(2*pi*i*a*n/m)`` (an integer).
``K_m(n)``  normalized Ramanujan kernel ``c_m(n) / phi(m)``; ``K_m(0) = 1``.
``order``   the modulus ``m`` of a kernel. It is never the attention query ``Q``.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd
from typing import Iterable, Sequence


def _require_positive_int(value: int, label: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive int, got {value!r}")


def _require_int(value: int, label: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{label} must be an int, got {value!r}")


@lru_cache(maxsize=None)
def factorize(n: int) -> tuple[tuple[int, int], ...]:
    """Prime factorization ``((p1, e1), (p2, e2), ...)`` with increasing primes."""
    _require_positive_int(n, "n")
    factors, remaining, p = [], n, 2
    while p * p <= remaining:
        if remaining % p == 0:
            exponent = 0
            while remaining % p == 0:
                remaining //= p
                exponent += 1
            factors.append((p, exponent))
        p += 1
    if remaining > 1:
        factors.append((remaining, 1))
    return tuple(factors)


@lru_cache(maxsize=None)
def divisors(n: int) -> tuple[int, ...]:
    """All positive divisors of ``n`` in increasing order."""
    _require_positive_int(n, "n")
    small, large, d = [], [], 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d * d != n:
                large.append(n // d)
        d += 1
    return tuple(small + large[::-1])


def mobius(n: int) -> int:
    """Moebius function ``mu(n)``."""
    factors = factorize(n)
    if any(e > 1 for _, e in factors):
        return 0
    return -1 if len(factors) % 2 else 1


def euler_phi(n: int) -> int:
    """Euler's totient ``phi(n)``."""
    result = n
    for p, _ in factorize(n):
        result -= result // p
    return result


def lcm(values: Iterable[int]) -> int:
    """Least common multiple of positive integers (``lcm(()) = 1``)."""
    out = 1
    for v in values:
        _require_positive_int(v, "lcm argument")
        out = out * v // gcd(out, v)
    return out


def units(m: int) -> tuple[int, ...]:
    """Reduced residues ``{1 <= a <= m : gcd(a, m) = 1}`` (``units(1) == (1,)``)."""
    _require_positive_int(m, "m")
    return tuple(a for a in range(1, m + 1) if gcd(a, m) == 1)


def ramanujan_sum(m: int, n: int) -> int:
    """``c_m(n)`` via the divisor formula ``sum_{d | gcd(m,n)} d * mu(m/d)``."""
    _require_positive_int(m, "m")
    _require_int(n, "n")
    return sum(d * mobius(m // d) for d in divisors(gcd(m, n)))


def ramanujan_sum_closed(m: int, n: int) -> int:
    """``c_m(n)`` via von Sterneck/Hoelder: ``mu(m/g) * phi(m) / phi(m/g)``, ``g = gcd(m, n)``.

    Kept separate from :func:`ramanujan_sum` so that each can test the other.
    """
    _require_positive_int(m, "m")
    _require_int(n, "n")
    q = m // gcd(m, n)
    return mobius(q) * (euler_phi(m) // euler_phi(q))


def kernel(m: int, n: int) -> Fraction:
    """Normalized Ramanujan kernel ``K_m(n) = c_m(n) / phi(m) = mu(q) / phi(q)``, ``q = m / gcd(m, n)``."""
    _require_positive_int(m, "m")
    _require_int(n, "n")
    q = m // gcd(m, n)
    return Fraction(mobius(q), euler_phi(q))


@lru_cache(maxsize=None)
def kernel_table(m: int) -> tuple[Fraction, ...]:
    """``(K_m(0), ..., K_m(m-1))``; ``K_m`` is ``m``-periodic and even."""
    return tuple(kernel(m, r) for r in range(m))


def comb_gap(m: int) -> Fraction:
    """``1 - max_{n not divisible by m} K_m(n)`` (``2`` by convention for ``m = 1``).

    Lemma (docs/MATH.md, L2): the gap is at least ``1/2`` for every ``m``,
    equals ``1/2`` exactly when ``6 | m``, and is at least ``1`` for prime powers.
    """
    _require_positive_int(m, "m")
    if m == 1:
        return Fraction(2)
    return 1 - max(kernel_table(m)[1:])


def prime_power_components(n: int) -> tuple[int, ...]:
    """Pairwise coprime prime powers whose product is ``n`` (CRT components)."""
    return tuple(p**e for p, e in factorize(n))


def is_prime_power(n: int) -> bool:
    return n > 1 and len(factorize(n)) == 1


def optimal_comb_weights(period: int) -> dict[int, Fraction]:
    """Convex weights ``alpha_d = phi(d) / (P - 1)`` on orders ``d | P, d > 1``.

    Theorem (docs/MATH.md, T3): ``sum_d alpha_d K_d(n) = (P [P | n] - 1) / (P - 1)``,
    the unique ``P``-periodic positive-definite function with unit diagonal that
    maximizes the comb margin ``1 - max_{P does not divide n} f(n)``; the optimum is ``P / (P - 1)``.
    """
    _require_positive_int(period, "period")
    if period < 2:
        raise ValueError("period must be >= 2")
    return {d: Fraction(euler_phi(d), period - 1) for d in divisors(period) if d > 1}


def mixture_value(weights: dict[int, Fraction], n: int) -> Fraction:
    """``sum_m w_m K_m(n)`` exactly."""
    return sum((w * kernel(m, n) for m, w in weights.items()), Fraction(0))


def comb_margin(values: Sequence[Fraction], period: int) -> Fraction:
    """Margin ``min_{comb} f - max_{off-comb} f`` of a function given on one full period."""
    if len(values) != period:
        raise ValueError("values must cover exactly one period")
    comb = values[0]
    return comb - max(values[1:]) if period > 1 else Fraction(0)


def folded_rank(orders: Iterable[int]) -> int:
    """Real feature dimension of the exact low-rank factorization: ``sum_m phi(m)``."""
    return sum(euler_phi(m) for m in orders)


def alias_signature(orders: Sequence[int], n: int) -> tuple[Fraction, ...]:
    """Vector ``(K_m(n))_{m in orders}``; equal signatures => every bias in the span ties."""
    return tuple(kernel(m, n) for m in orders)


def alias_classes(orders: Sequence[int], offsets: Iterable[int]) -> list[list[int]]:
    """Partition ``offsets`` into classes that no bias in ``span{K_m : m in orders}`` separates.

    Offsets ``n, n'`` are tied for every such bias iff their signatures agree
    (docs/MATH.md, P1). Classes are returned in order of first appearance.
    """
    classes: dict[tuple[Fraction, ...], list[int]] = {}
    for n in offsets:
        classes.setdefault(alias_signature(orders, n), []).append(n)
    return list(classes.values())


def identifiability_modulus(length: int, harmonics: Sequence[int]) -> int:
    """Handoff theorem B: ``F_H(t) = (exp(2 pi i h t / L))_h`` is injective exactly modulo ``L / gcd(L, gcd(H))``."""
    _require_positive_int(length, "length")
    if not harmonics:
        raise ValueError("harmonics must be nonempty")
    for h in harmonics:
        _require_positive_int(h, "harmonic")
    return length // gcd(length, reduce(gcd, harmonics, 0))
