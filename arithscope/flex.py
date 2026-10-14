"""FlexAttention backend: every additive bias as a ``score_mod`` over a per-head offset table.

FlexAttention (``torch.nn.attention.flex_attention``) applies a user function to each
attention score inside the attention kernel. All biases in :mod:`arithscope.bias`
depend only on the integer offset ``pos_i - pos_j``, so their sum is evaluated once
per possible offset (a ``[heads, 2N - 1]`` table, from the same ``offset_values`` the
explicit backend uses) and read back by index inside the kernel:

    score_mod(score, b, h, i, j) = score + table[h, pos[b, i] - pos[b, j] - lo]
    mask_mod(b, h, i, j)         = pos[b, j] <= pos[b, i] and key_valid[b, j]

Uncompiled (the default), FlexAttention materializes the scores. After
:func:`compile_kernels`, calls go through ``torch.compile(flex_attention)``, which fuses
the lookup into the attention kernel without materializing scores or an ``[N, N]`` bias.

**What is verified here, on CPU and for the forward pass only.** Eager mode matches the
explicit backend to 1e-12 in float64, and compiled kernels match it to 1e-5 in float32
(``tests/test_attention.py``; the compiled test is opt-in). FlexAttention has no CPU
backward, so call it under ``torch.no_grad()`` on CPU. Gradients and GPU kernels are
**not tested** in this repository.
"""
from __future__ import annotations

import functools

import torch

_COMPILE = False


def compile_kernels(enabled: bool = True) -> None:
    """Route ``backend="flex"`` through ``torch.compile(flex_attention)`` (fused kernels) or back to eager mode.

    Compiled kernels do not materialize the scores. On CPU they need inductor's C++ toolchain; with
    conda-forge PyTorch builds the vector-ISA probe can fail on an undefined SLEEF symbol, which
    preloading the environment's ``libsleef.so`` (``LD_PRELOAD``) works around.
    """
    global _COMPILE
    _COMPILE = bool(enabled)


@functools.lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False)


def available() -> bool:
    """Whether this PyTorch build provides FlexAttention."""
    try:
        from torch.nn.attention import flex_attention  # noqa: F401
    except ImportError:
        return False
    return True


def offset_table(biases, positions: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    """Summed bias values ``[H, R]`` for every offset ``lo .. lo + R - 1`` that ``positions [B, N]`` can form."""
    span = positions.max(-1).values - positions.min(-1).values
    lo, hi = -int(span.max()), int(span.max())
    offsets = torch.arange(lo, hi + 1, device=positions.device, dtype=torch.long)
    table = None
    for module in biases:
        values = module.offset_values(offsets)
        table = values if table is None else table + values
    return table, lo


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, biases, positions: torch.Tensor,
              key_padding_mask: torch.Tensor | None, scale: float) -> torch.Tensor:
    """``q, k, v [B, H, N, d]``; integer ``positions [N]`` or ``[B, N]``. Same semantics as the explicit backend.

    Rows with no allowed key return 0, as in the other backends.
    """
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    b, _, n, _ = q.shape
    pos = (positions if positions.dim() == 2 else positions.unsqueeze(0).expand(b, n)).long()
    valid = (torch.ones(b, n, dtype=torch.bool, device=q.device) if key_padding_mask is None
             else key_padding_mask.bool())
    table, lo = offset_table(biases, pos)
    if table is not None:
        table = table.to(q.dtype)

    def score_mod(score, bi, h, qi, ki):
        if table is None:
            return score
        return score + table[h, pos[bi, qi] - pos[bi, ki] - lo]

    def mask_mod(bi, h, qi, ki):
        return (pos[bi, ki] <= pos[bi, qi]) & valid[bi, ki]

    block_mask = create_block_mask(mask_mod, B=b, H=None, Q_LEN=n, KV_LEN=n, device=q.device.type,
                                   _compile=_COMPILE)
    attend = _compiled_flex_attention() if _COMPILE else flex_attention
    return attend(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale)


__all__ = ["attention", "available", "compile_kernels", "offset_table"]
