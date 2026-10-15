"""Additive relative-position attention biases, including the ArithScope bias.

Every module here maps integer query/key positions to an additive logit bias
``bias[..., h, i, j] = b_h(q_pos[i] - k_pos[j])`` of shape ``[..., heads, Nq, Nk]``.
Offsets are formed with exact int64 subtraction, so a bias that depends only on
``offset mod m`` (ArithScope, the fixed-period Fourier banks) is bitwise invariant
under any common shift of all positions.

The ArithScope bias (``RamanujanBias``) is

    b_h(n) = gamma_h * sum_{m in orders} alpha_{h,m} * c_m(n) / phi(m),
    alpha_h = softmax(order_logits_h),  gamma_h in R (initialized to 0),

where ``c_m`` is the Ramanujan sum. ``gamma = 0`` gives an exactly zero bias.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from . import arith

DEFAULT_ORDERS: tuple[int, ...] = tuple(range(1, 17))


POSITION_LIMIT = 2**62  # |position| < 2^62 keeps every int64 offset q_pos - k_pos exact (no overflow)


def _check_positions(q_pos: torch.Tensor, k_pos: torch.Tensor) -> None:
    for name, pos in (("q_pos", q_pos), ("k_pos", k_pos)):
        if pos.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be an integer tensor, got {pos.dtype}")
        if pos.dim() < 1:
            raise ValueError(f"{name} must have a sequence dimension")
        # min/max, not abs(): abs(-2**63) overflows to a negative number in int64
        if pos.numel() and (int(pos.min()) <= -POSITION_LIMIT or int(pos.max()) >= POSITION_LIMIT):
            raise ValueError(f"{name} must satisfy |position| < 2**62 so that offsets fit in int64")


class OffsetBias(nn.Module):
    """Base class: an additive bias that depends only on the integer offset ``i - j``.

    Subclasses implement :meth:`offset_values`, mapping a 1-D int64 tensor of
    offsets ``[R]`` to per-head values ``[heads, R]``, and :meth:`score_fn`, the same
    bias evaluated pointwise (used by the FlexAttention backend). :meth:`forward`
    evaluates ``offset_values`` once per offset and gathers the dense bias.

    Memory is bounded by the number of query/key pairs, never by the numeric span of
    the positions: contiguous positions use the offset range ``[lo, hi]`` (at most
    ``Nq + Nk - 1`` values), and sparse positions use only the offsets that occur.
    """

    num_heads: int

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def score_fn(self):
        """Return ``fn(h, delta) -> bias`` for integer tensors of head indices and offsets (same shape)."""
        raise NotImplementedError

    def forward(self, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
        """``q_pos [..., Nq]`` and ``k_pos [..., Nk]`` (broadcastable leading dims) -> ``[..., H, Nq, Nk]``."""
        _check_positions(q_pos, k_pos)
        diff = q_pos.long().unsqueeze(-1) - k_pos.long().unsqueeze(-2)
        lo, hi = int(diff.min()), int(diff.max())
        if hi - lo < diff.numel():  # the offset range is no larger than the number of pairs
            offsets = torch.arange(lo, hi + 1, device=diff.device, dtype=torch.long)
            return self.offset_values(offsets)[:, diff - lo].movedim(0, -3)
        offsets, index = torch.unique(diff, return_inverse=True)  # sparse positions
        return self.offset_values(offsets)[:, index].movedim(0, -3)


def _residue_table(orders: Sequence[int], entries: Sequence[Sequence[float]]) -> torch.Tensor:
    """Pad per-order lookup rows (row ``f`` has ``orders[f]`` entries) into one float64 table."""
    width = max(orders)
    table = torch.zeros(len(entries), width, dtype=torch.float64)
    for f, row in enumerate(entries):
        table[f, : len(row)] = torch.tensor(row, dtype=torch.float64)
    return table


def _lookup(table: torch.Tensor, row_orders: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``table[f, offsets mod row_orders[f]]`` for every row ``f``: ``[F, R]`` (exact integer indexing)."""
    residues = torch.remainder(offsets.unsqueeze(0), row_orders.unsqueeze(1))
    return torch.gather(table, 1, residues)


class _GatedMixtureBias(OffsetBias):
    """Shared parameterization ``gamma_h * sum_c softmax(logits_h)_c * phi_c(n)`` over fixed components.

    ``phi_c`` is ``row_orders[c]``-periodic and given by a lookup table over residues,
    so the bias is exactly periodic and exactly shift invariant.
    """

    def __init__(self, num_heads: int, row_orders: Sequence[int], table: torch.Tensor,
                 gamma_init: float = 0.0, learn_mixture: bool = True) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.num_heads = num_heads
        self.register_buffer("row_orders", torch.tensor(list(row_orders), dtype=torch.long), persistent=False)
        self.register_buffer("table", table, persistent=False)
        logits = torch.zeros(num_heads, len(row_orders))
        self.mixture_logits = nn.Parameter(logits, requires_grad=learn_mixture)
        self.gamma = nn.Parameter(torch.full((num_heads,), float(gamma_init)))

    @property
    def alpha(self) -> torch.Tensor:
        """Convex mixture weights ``[heads, components]`` (nonnegative, rows sum to 1)."""
        return torch.softmax(self.mixture_logits, dim=-1)

    def component_values(self, offsets: torch.Tensor) -> torch.Tensor:
        """``[components, R]`` values of each fixed component at the given offsets."""
        return _lookup(self.table, self.row_orders, offsets).to(self.gamma.dtype)

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        mix = self.alpha @ self.component_values(offsets)  # [H, R]
        return self.gamma.unsqueeze(-1) * mix

    def score_fn(self):
        weights = self.gamma.unsqueeze(-1) * self.alpha  # [H, C]
        table = self.table.to(weights.dtype)
        orders = [int(m) for m in self.row_orders]

        def fn(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
            out = weights[h, 0] * table[0, torch.remainder(delta, orders[0])]
            for c in range(1, len(orders)):
                out = out + weights[h, c] * table[c, torch.remainder(delta, orders[c])]
            return out

        return fn


class RamanujanBias(_GatedMixtureBias):
    """ArithScope bias: ``gamma_h * sum_m alpha_{h,m} K_m(i - j)``, ``K_m = c_m / phi(m)``.

    ``K_m(n)`` is stored as an exact rational table over residues ``n mod m`` and
    rounded once to float64 (then to the parameter dtype). Properties used in the
    tests and in ``docs/MATH.md``: each ``K_m`` and every convex mixture is a
    positive semidefinite Toeplitz kernel with unit diagonal and ``|K| <= 1``;
    ``K_m(n) = 1`` iff ``m | n``.

    The same bias has an exact rank-``sum_m phi(m)`` factorization,
    ``K_m(i - j) = <u_m(i), u_m(j)>``, exposed by :meth:`key_features` and
    :meth:`query_features` so it can run inside fused attention kernels without
    an ``[Nq, Nk]`` bias tensor (see :mod:`arithscope.attention`).
    """

    def __init__(self, num_heads: int, orders: Sequence[int] = DEFAULT_ORDERS,
                 gamma_init: float = 0.0, learn_mixture: bool = True) -> None:
        orders = tuple(orders)
        if not orders or len(set(orders)) != len(orders):
            raise ValueError("orders must be a nonempty sequence of distinct positive ints")
        rows = []
        for m in orders:
            arith._require_positive_int(m, "order")
            rows.append([float(v) for v in arith.kernel_table(m)])
        super().__init__(num_heads, orders, _residue_table(orders, rows), gamma_init, learn_mixture)
        self.orders = orders
        self._build_feature_tables()

    # ----- exact low-rank factorization -------------------------------------------------
    def _build_feature_tables(self) -> None:
        feat_orders, feat_group, rows = [], [], []
        for g, m in enumerate(self.orders):
            if m <= 2:  # K_1 = 1 and K_2(n) = (-1)^n: one real feature each
                pairs = [(1, "cos", 1.0)]
            else:  # pair units a < m/2 with m - a: K_m = (2/phi(m)) sum cos(2 pi a n / m)
                scale = math.sqrt(2.0 / arith.euler_phi(m))
                pairs = [(a, kind, scale) for a in arith.units(m) if 2 * a < m for kind in ("cos", "sin")]
            for a, kind, scale in pairs:
                trig = math.cos if kind == "cos" else math.sin
                rows.append([scale * trig(2.0 * math.pi * ((a * r) % m) / m) for r in range(m)])
                feat_orders.append(m)
                feat_group.append(g)
        self.register_buffer("feature_orders", torch.tensor(feat_orders, dtype=torch.long), persistent=False)
        self.register_buffer("feature_group", torch.tensor(feat_group, dtype=torch.long), persistent=False)
        self.register_buffer("feature_table", _residue_table(feat_orders, rows), persistent=False)

    @property
    def feature_dim(self) -> int:
        """``sum_m phi(m)``: real dimension of the exact factorization."""
        return int(self.feature_orders.numel())

    def key_features(self, pos: torch.Tensor) -> torch.Tensor:
        """Parameter-free features ``u(t)``: ``pos [..., N] -> [..., N, feature_dim]``.

        Phases use the exact residue ``(a * (t mod m)) mod m``, so the features take
        finitely many values and do not lose precision at large positions.
        """
        if pos.dtype not in (torch.int32, torch.int64):
            raise TypeError("positions must be an integer tensor")
        flat = pos.long().reshape(-1)
        values = _lookup(self.feature_table, self.feature_orders, flat)  # [F, N*]
        return values.t().reshape(*pos.shape, -1).to(self.gamma.dtype)

    def query_features(self, pos: torch.Tensor) -> torch.Tensor:
        """``gamma_h * alpha_{h,m(f)} * u_f(t)``: ``pos [..., N] -> [..., H, N, feature_dim]``."""
        weights = self.gamma.unsqueeze(-1) * self.alpha[:, self.feature_group]  # [H, F]
        return self.key_features(pos).unsqueeze(-3) * weights.unsqueeze(-2)


class CosineBankBias(_GatedMixtureBias):
    """Matched Fourier control: ``gamma_h * sum_c alpha_{h,c} cos(2 pi a_c n / m_c)``.

    ``components`` is a list of ``(m, a)`` pairs. Two standard banks:

    * ``fundamental(orders)``: one cosine ``cos(2 pi n / m)`` per order -- the same
      periods and the same parameter count as :class:`RamanujanBias`.
    * ``full(orders)``: every reduced harmonic ``a / m`` (``a <= m / 2``); its span
      strictly contains the Ramanujan span, with more mixture parameters.
    """

    def __init__(self, num_heads: int, components: Sequence[tuple[int, int]],
                 gamma_init: float = 0.0, learn_mixture: bool = True) -> None:
        components = tuple((int(m), int(a)) for m, a in components)
        if not components:
            raise ValueError("components must be nonempty")
        rows = [[math.cos(2.0 * math.pi * ((a * r) % m) / m) for r in range(m)] for m, a in components]
        super().__init__(num_heads, [m for m, _ in components], _residue_table([m for m, _ in components], rows),
                         gamma_init, learn_mixture)
        self.components = components

    @staticmethod
    def fundamental(orders: Sequence[int] = DEFAULT_ORDERS) -> list[tuple[int, int]]:
        return [(m, 1) for m in orders]

    @staticmethod
    def full(orders: Sequence[int] = DEFAULT_ORDERS) -> list[tuple[int, int]]:
        return [(m, a) for m in orders for a in arith.units(m) if 2 * a <= m or m == 1]


class LearnedFrequencyBias(OffsetBias):
    """Learnable-frequency Fourier control: ``sum_f w_{h,f} cos(omega_{h,f} n)``.

    Frequencies are free real parameters (initialized at periods log-spaced in
    ``[min_period, max_period]``); amplitudes start at zero, so the initial bias is 0.
    Phases are computed in float64.
    """

    def __init__(self, num_heads: int, num_frequencies: int = 16,
                 min_period: float = 2.0, max_period: float = 32.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), num_frequencies)
        omega = (2.0 * math.pi / periods).repeat(num_heads, 1)
        self.omega = nn.Parameter(omega)
        self.amplitude = nn.Parameter(torch.zeros(num_heads, num_frequencies))

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        phase = self.omega.double().unsqueeze(-1) * offsets.double()  # [H, F, R]
        return (self.amplitude.double().unsqueeze(-1) * torch.cos(phase)).sum(1).to(self.amplitude.dtype)

    def score_fn(self):
        omega, amplitude, dtype = self.omega.double(), self.amplitude.double(), self.amplitude.dtype

        def fn(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
            n = delta.double()
            out = amplitude[h, 0] * torch.cos(omega[h, 0] * n)
            for f in range(1, omega.shape[1]):
                out = out + amplitude[h, f] * torch.cos(omega[h, f] * n)
            return out.to(dtype)

        return fn


class RelativeBias(OffsetBias):
    """Learned per-offset table (Shaw/T5-style without log buckets), clipped at ``max_distance``.

    The strongest in-range control: it can represent any bias on ``|n| <= max_distance``
    exactly, and is constant beyond it.
    """

    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        if max_distance < 1:
            raise ValueError("max_distance must be positive")
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.table = nn.Parameter(torch.zeros(num_heads, 2 * max_distance + 1))

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        idx = offsets.clamp(-self.max_distance, self.max_distance) + self.max_distance
        return self.table[:, idx]

    def score_fn(self):
        table, d = self.table, self.max_distance

        def fn(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
            return table[h, delta.clamp(-d, d) + d]

        return fn


class ALiBiBias(OffsetBias):
    """ALiBi (Press et al., 2022): fixed ``-slope_h * |n|`` with geometric slopes ``2^(-8 h / H)``."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / num_heads) for h in range(num_heads)])
        self.register_buffer("slopes", slopes, persistent=False)

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        return -self.slopes.unsqueeze(-1) * offsets.abs().to(self.slopes.dtype)

    def score_fn(self):
        slopes = self.slopes

        def fn(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
            return -slopes[h] * delta.abs().to(slopes.dtype)

        return fn
