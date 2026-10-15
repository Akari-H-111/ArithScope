"""Causal self-attention with pluggable relative-position mechanisms.

Logit semantics (all backends):

    logits[b,h,i,j] = <q_i, k_j> / sqrt(d_head)  (q, k optionally rotated by RoPE)
                      + sum of additive biases b_h(pos_i - pos_j)
    allowed[b,i,j]  = (j <= i) and key_valid[b,j]       (token order, not position values)
    attn            = softmax over allowed keys; rows with no allowed key output 0

Causality is by token index. Positions enter only the biases and RoPE, so they may
repeat, restart (packed sequences), or carry offsets without letting a token see a
later one.

Backends compute the same function:

* ``explicit`` -- materializes logits; the reference path (used for training).
* ``sdpa``     -- ``torch.nn.functional.scaled_dot_product_attention`` with the
  summed bias passed as ``attn_mask``.
* ``folded``   -- ArithScope only: the Ramanujan bias is folded into extra q/k
  feature dimensions (exact rank-``sum phi(m)`` factorization), so no ``[N, N]``
  bias tensor is formed and fused causal kernels can be used.
* ``flex``     -- FlexAttention: every bias becomes a ``score_mod`` that reads a
  per-head table of offset values (:mod:`arithscope.flex`; eager by default,
  fused after ``flex.compile_kernels()``). Forward only on CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from .bias import (ALiBiBias, CosineBankBias, DEFAULT_ORDERS, LearnedFrequencyBias, OffsetBias,
                   RamanujanBias, RelativeBias)
from .rope import RotaryEmbedding

EXPLICIT_QUERY_BLOCK = 2048  # query rows per block in the explicit backend (exact; bounds memory)
BIAS_KINDS = ("arith", "fourier_fund", "fourier_full", "fourier_learned", "relbias", "alibi")


@dataclass(frozen=True)
class PositionConfig:
    """Which relative-position mechanisms an attention layer uses.

    Named variants (see :meth:`from_name`): ``nope``, ``rope``, ``alibi``, ``arith``,
    and ``rope+<bias>`` / ``alibi+arith`` combinations.
    """

    rope: bool = False
    rope_base: float = 10000.0
    rope_angle_dtype: str = "float32"
    bias: tuple[str, ...] = ()
    orders: tuple[int, ...] = DEFAULT_ORDERS
    relbias_max_distance: int = 64
    learned_frequencies: int = 16

    def __post_init__(self) -> None:
        unknown = set(self.bias) - set(BIAS_KINDS)
        if unknown:
            raise ValueError(f"unknown bias kinds {sorted(unknown)}; expected {BIAS_KINDS}")

    @classmethod
    def from_name(cls, name: str, **overrides) -> "PositionConfig":
        parts = [p for p in name.split("+") if p]
        if not parts:
            raise ValueError("empty position variant name")
        rope = "rope" in parts
        biases = tuple(p for p in parts if p not in ("rope", "nope"))
        if "nope" in parts and (rope or biases):
            raise ValueError("'nope' cannot be combined with other mechanisms")
        return replace(cls(rope=rope, bias=biases), **overrides)

    @property
    def name(self) -> str:
        parts = (["rope"] if self.rope else []) + list(self.bias)
        return "+".join(parts) if parts else "nope"


def make_bias(kind: str, num_heads: int, cfg: PositionConfig) -> OffsetBias:
    if kind == "arith":
        return RamanujanBias(num_heads, cfg.orders)
    if kind == "fourier_fund":
        return CosineBankBias(num_heads, CosineBankBias.fundamental(cfg.orders))
    if kind == "fourier_full":
        return CosineBankBias(num_heads, CosineBankBias.full(cfg.orders))
    if kind == "fourier_learned":
        return LearnedFrequencyBias(num_heads, cfg.learned_frequencies, max_period=float(max(cfg.orders)) * 2)
    if kind == "relbias":
        return RelativeBias(num_heads, cfg.relbias_max_distance)
    if kind == "alibi":
        return ALiBiBias(num_heads)
    raise ValueError(f"unknown bias kind {kind!r}")


def allowed_mask(n_q: int, n_k: int, key_padding_mask: torch.Tensor | None = None, q_offset: int = 0,
                 device: torch.device | str | None = None) -> torch.Tensor:
    """Boolean ``[B or 1, Nq, Nk]``: key index ``j <= q_offset + i`` (causal by token order) and ``key_padding_mask``.

    ``q_offset`` is the token index of the first query row: the block start in blocked
    evaluation, or ``Nk - Nq`` when the queries are the last tokens of the keys.
    """
    q_idx = torch.arange(q_offset, q_offset + n_q, device=device)
    allowed = (torch.arange(n_k, device=device).unsqueeze(0) <= q_idx.unsqueeze(1)).unsqueeze(0)
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask.bool().unsqueeze(-2)
    return allowed


def _allowed_and_row_ok(n: int, key_padding_mask: torch.Tensor | None, device):
    """Mask ``[B|1, 1, N, N]`` for the fused backends, and whether each query row has any allowed key."""
    allowed = allowed_mask(n, n, key_padding_mask, device=device).unsqueeze(1)
    return allowed, allowed.any(-1, keepdim=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos: PositionConfig) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pos_cfg = pos
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, pos.rope_base, pos.rope_angle_dtype) if pos.rope else None
        self.biases = nn.ModuleDict({kind: make_bias(kind, num_heads, pos) for kind in pos.bias})

    def _qkv(self, x: torch.Tensor, positions: torch.Tensor):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q, positions), self.rope(k, positions)
        return q, k, v

    def total_bias(self, q_pos: torch.Tensor, k_pos: torch.Tensor | None = None) -> torch.Tensor | None:
        """Sum of additive biases, ``[B or (), H, Nq, Nk]``, or ``None`` when there are none."""
        k_pos = q_pos if k_pos is None else k_pos
        out = None
        for module in self.biases.values():
            term = module(q_pos, k_pos)
            out = term if out is None else out + term
        return out

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None, backend: str = "explicit",
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``x [B, N, D]``, integer ``positions [N]`` or ``[B, N]``, ``key_padding_mask [B, N]`` (True = valid)."""
        b, n, d = x.shape
        q, k, v = self._qkv(x, positions)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = None
        if backend == "flex":
            if return_attention:
                raise ValueError("return_attention requires backend='explicit'")
            from . import flex

            out = flex.attention(q, k, v, list(self.biases.values()), positions, key_padding_mask, scale)
            return self.proj(out.transpose(1, 2).reshape(b, n, d))
        if backend == "explicit":
            out, attn = self._explicit(q, k, v, positions, key_padding_mask, scale, return_attention)
            out = self.proj(out.transpose(1, 2).reshape(b, n, d))
            return (out, attn) if return_attention else out
        if backend == "sdpa":
            bias = self.total_bias(positions)
            if bias is None and key_padding_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
            else:
                allowed, row_ok = _allowed_and_row_ok(n, key_padding_mask, x.device)
                zero = torch.zeros((), dtype=q.dtype)
                mask = torch.where(allowed, zero if bias is None else bias, float("-inf"))
                mask = torch.where(row_ok, mask, zero)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale) * row_ok
        elif backend == "folded":
            out = self._folded(q, k, v, positions, key_padding_mask, scale)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        if return_attention:
            raise ValueError("return_attention requires backend='explicit'")
        return self.proj(out.transpose(1, 2).reshape(b, n, d))

    def _explicit(self, q, k, v, positions, key_padding_mask, scale, keep_attention):
        """Reference path. Long inputs are processed in query blocks of ``EXPLICIT_QUERY_BLOCK`` rows.

        Blocking is exact: every query row's softmax only involves that row's logits.
        """
        n = q.shape[-2]
        block = n if keep_attention else min(n, EXPLICIT_QUERY_BLOCK)
        outs, attn = [], None
        for start in range(0, n, block):
            q_pos = positions[..., start:start + block]
            logits = (q[..., start:start + block, :] @ k.transpose(-1, -2)) * scale
            bias = self.total_bias(q_pos, positions)
            if bias is not None:
                logits = logits + bias
            allowed = allowed_mask(q_pos.shape[-1], n, key_padding_mask, q_offset=start,
                                   device=q.device).unsqueeze(1)  # [B|1, 1, Bq, N]
            row_ok = allowed.any(-1, keepdim=True)
            logits = logits.masked_fill(~allowed, float("-inf"))
            logits = torch.where(row_ok, logits, torch.zeros((), dtype=logits.dtype))
            low_precision = logits.dtype in (torch.float16, torch.bfloat16)
            attn = torch.softmax(logits, dim=-1, dtype=torch.float32 if low_precision else logits.dtype)
            attn = attn.to(q.dtype) * row_ok
            outs.append(attn @ v)
        return torch.cat(outs, dim=-2), (attn if keep_attention else None)

    def _folded(self, q, k, v, positions, key_padding_mask, scale):
        kinds = set(self.biases.keys())
        if kinds != {"arith"}:
            raise ValueError(f"folded backend supports only the 'arith' bias (optionally with RoPE), got {sorted(kinds)}")
        arith_bias: RamanujanBias = self.biases["arith"]  # type: ignore[assignment]
        # logits = scale * (<q,k> + sqrt(d) <gamma*alpha*u(i), u(j)>) = <q,k>/sqrt(d) + bias
        qf = arith_bias.query_features(positions) * math.sqrt(self.head_dim)  # [B|.., H, N, F]
        kf = arith_bias.key_features(positions).unsqueeze(-3)  # [B|.., 1, N, F]
        q2 = torch.cat((q, qf.expand(*q.shape[:-1], qf.shape[-1]).to(q.dtype)), dim=-1)
        k2 = torch.cat((k, kf.expand(*k.shape[:-1], kf.shape[-1]).to(k.dtype)), dim=-1)
        # Fused SDPA kernels need one head dimension for q, k and v (a multiple of 8 on some devices).
        # Zero columns change no dot product and no output column we keep, so padding is exact.
        width = -(-max(q2.shape[-1], v.shape[-1]) // 8) * 8
        q2, k2 = F.pad(q2, (0, width - q2.shape[-1])), F.pad(k2, (0, width - k2.shape[-1]))
        v2 = F.pad(v, (0, width - v.shape[-1]))
        if key_padding_mask is None:
            return F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=scale)[..., : v.shape[-1]]
        allowed, row_ok = _allowed_and_row_ok(q.shape[-2], key_padding_mask, q.device)
        mask = torch.where(allowed & row_ok, 0.0, float("-inf")).to(q.dtype)
        mask = torch.where(row_ok, mask, torch.zeros((), dtype=q.dtype))
        out = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask, scale=scale) * row_ok
        return out[..., : v.shape[-1]]
