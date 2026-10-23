"""Opt-in evidence head: an unnormalized, kernel-weighted count readout added to the logits.

    E_i      = sum_{j <= i, key j valid} kappa(pos_i - pos_j) * onehot(x_j)        [V_in]
    logits_i = head(ln_f(h_i)) + W_out @ E_i                                         [V_out]

Attention reports vote *shares*; ``E_i`` is a vote *count* (weighted by ``kappa``), so a
readout on top of it can grow with the number of votes (docs/MATH.md, T8a, P3, B1).
Causality is by token index (``j <= i``) and padded keys contribute nothing; positions
enter only through the kernel, which depends on the exact int64 offset ``pos_i - pos_j``.

Kernels (:class:`EvidenceConfig.kernel`):

* ``arith``          -- ``kappa(n) = sum_m beta_m K_m(n)``, ``K_m = c_m / phi(m)`` from the exact
  residue tables of :class:`~arithscope.bias.RamanujanBias`; ``beta`` is an unconstrained real
  vector initialized to ``1 / len(orders)`` (so ``kappa(0) = 1``).
* ``cosine_fund``    -- ``kappa(n) = sum_c a_c cos(2 pi n / m_c)``, one fixed cosine per order (matched control
  with the same number of coefficients as ``arith``); ``a`` unconstrained, initialized to ``1 / len(orders)``.
* ``cosine_full``    -- the same over every reduced harmonic (``CosineBankBias.full``); its span contains the
  comb indicator ``D_P``, with more coefficients.
* ``cosine_learned`` --``kappa(n) = sum_f a_f cos(omega_f n)``, frequencies initialized as in
  :class:`~arithscope.bias.LearnedFrequencyBias`, amplitudes to ``1 / F`` (phases in float64).
* ``uniform``        -- ``kappa(n) = 1``: a plain causal bag-of-tokens count. No parameters.

``W_out`` starts at zero, so a model with the head reproduces the model without it bitwise at
initialization. The head is built without drawing from the global RNG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import arith
from .bias import DEFAULT_ORDERS, CosineBankBias, OffsetBias, _lookup, _residue_table

KERNELS = ("arith", "cosine_fund", "cosine_full", "cosine_learned", "uniform")
BANK_KERNELS = ("arith", "cosine_fund", "cosine_full")  # fixed periodic components, free coefficients
EVIDENCE_QUERY_BLOCK = 2048  # query rows per block (exact; bounds the [rows, keys] kernel memory)


@dataclass(frozen=True)
class EvidenceConfig:
    kernel: str = "arith"
    orders: tuple[int, ...] = DEFAULT_ORDERS  # arith kernel only
    num_frequencies: int = 16                 # cosine_learned only
    min_period: float = 2.0
    max_period: float = 32.0

    def __post_init__(self) -> None:
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown evidence kernel {self.kernel!r}; expected one of {KERNELS}")
        if self.kernel in BANK_KERNELS and (not self.orders or len(set(self.orders)) != len(self.orders)):
            raise ValueError("orders must be a nonempty sequence of distinct positive ints")


class _ArithKernel(OffsetBias):
    """``sum_m beta_m K_m(n)`` with exact residue tables (``beta`` unconstrained)."""

    def __init__(self, orders: tuple[int, ...]) -> None:
        super().__init__()
        self.num_heads = 1
        self.orders = tuple(orders)
        rows = []
        for m in self.orders:
            arith._require_positive_int(m, "order")
            rows.append([float(v) for v in arith.kernel_table(m)])
        self.register_buffer("row_orders", torch.tensor(self.orders, dtype=torch.long), persistent=False)
        self.register_buffer("table", _residue_table(self.orders, rows), persistent=False)
        self.beta = nn.Parameter(torch.full((len(self.orders),), 1.0 / len(self.orders)))

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        values = _lookup(self.table, self.row_orders, offsets).to(self.beta.dtype)  # [M, R]
        return (self.beta @ values).unsqueeze(0)


class _CosineBankKernel(OffsetBias):
    """``sum_c a_c cos(2 pi a n / m)`` over fixed components ``(m, a)``; coefficients ``coef`` unconstrained.

    The rows are built as in :class:`~arithscope.bias.CosineBankBias`, from the exact residue ``(a * r) % m``.
    ``fundamental`` uses one cosine per order (the matched control of ``arith``); ``full`` every reduced harmonic.
    """

    def __init__(self, components: tuple[tuple[int, int], ...]) -> None:
        super().__init__()
        self.num_heads = 1
        self.components = tuple((int(m), int(a)) for m, a in components)
        rows = [[math.cos(2.0 * math.pi * ((a * r) % m) / m) for r in range(m)] for m, a in self.components]
        orders = [m for m, _ in self.components]
        self.register_buffer("row_orders", torch.tensor(orders, dtype=torch.long), persistent=False)
        self.register_buffer("table", _residue_table(orders, rows), persistent=False)
        self.coef = nn.Parameter(torch.full((len(self.components),), 1.0 / len(self.components)))

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        values = _lookup(self.table, self.row_orders, offsets).to(self.coef.dtype)  # [C, R]
        return (self.coef @ values).unsqueeze(0)


class _CosineKernel(OffsetBias):
    """``sum_f a_f cos(omega_f n)``; phases in float64."""

    def __init__(self, num_frequencies: int, min_period: float, max_period: float) -> None:
        super().__init__()
        self.num_heads = 1
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), num_frequencies)
        self.omega = nn.Parameter(2.0 * math.pi / periods)
        self.amplitude = nn.Parameter(torch.full((num_frequencies,), 1.0 / num_frequencies))

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        phase = self.omega.double().unsqueeze(-1) * offsets.double()  # [F, R]
        return (self.amplitude.double().unsqueeze(-1) * torch.cos(phase)).sum(0).to(self.amplitude.dtype).unsqueeze(0)


class _UniformKernel(OffsetBias):
    """``kappa = 1``; the buffer only carries the dtype and device."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 1
        self.register_buffer("one", torch.ones(()), persistent=False)

    def offset_values(self, offsets: torch.Tensor) -> torch.Tensor:
        return self.one.expand(1, offsets.numel())


def make_kernel(cfg: EvidenceConfig) -> OffsetBias:
    if cfg.kernel == "arith":
        return _ArithKernel(cfg.orders)
    if cfg.kernel == "cosine_fund":
        return _CosineBankKernel(CosineBankBias.fundamental(cfg.orders))
    if cfg.kernel == "cosine_full":
        return _CosineBankKernel(CosineBankBias.full(cfg.orders))
    if cfg.kernel == "cosine_learned":
        return _CosineKernel(cfg.num_frequencies, cfg.min_period, cfg.max_period)
    return _UniformKernel()


class EvidenceHead(nn.Module):
    """Kernel-weighted causal token counts followed by a zero-initialized linear map ``W_out``."""

    def __init__(self, cfg: EvidenceConfig, vocab_in: int, vocab_out: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.kernel = make_kernel(cfg)
        self.w_out = nn.Parameter(torch.zeros(vocab_out, vocab_in))

    def positional_parameters(self) -> list[nn.Parameter]:
        return list(self.kernel.parameters())

    def counts(self, tokens: torch.Tensor, positions: torch.Tensor,
               key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``E [B, N, V_in]`` for ``tokens [B, N]``, integer ``positions [N]`` or ``[B, N]``."""
        n = tokens.shape[1]
        onehot = F.one_hot(tokens, self.w_out.shape[1]).to(self.w_out.dtype)
        if key_padding_mask is not None:
            onehot = onehot * key_padding_mask.to(onehot.dtype).unsqueeze(-1)
        out = []
        for start in range(0, n, EVIDENCE_QUERY_BLOCK):
            stop = min(start + EVIDENCE_QUERY_BLOCK, n)
            kappa = self.kernel(positions[..., start:stop], positions[..., :stop]).squeeze(-3)  # [(B,) rows, stop]
            causal = torch.arange(stop, device=tokens.device) <= torch.arange(start, stop, device=tokens.device)[:, None]
            out.append(kappa.masked_fill(~causal, 0.0) @ onehot[:, :stop])
        return torch.cat(out, dim=1)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.counts(tokens, positions, key_padding_mask) @ self.w_out.t()
