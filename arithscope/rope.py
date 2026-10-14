"""Rotary position embedding (Su et al., 2021), rotate-half convention."""
from __future__ import annotations

import torch
from torch import nn

_ANGLE_DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}


class RotaryEmbedding(nn.Module):
    """Rotates ``x[..., i]`` and ``x[..., i + d/2]`` by ``pos * base^(-2i/d)``.

    ``angle_dtype`` controls the precision of ``pos * inv_freq`` -- the step where
    RoPE loses relative-position exactness at large absolute positions. ``float32``
    matches common LLM implementations.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, angle_dtype: str = "float32") -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if angle_dtype not in _ANGLE_DTYPES:
            raise ValueError(f"angle_dtype must be one of {sorted(_ANGLE_DTYPES)}")
        self.head_dim = head_dim
        self.angle_dtype = _ANGLE_DTYPES[angle_dtype]
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        self.register_buffer("inv_freq", base ** (-exponent), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``x [..., H, N, d]``, integer ``pos [..., N]`` (leading dims broadcastable)."""
        angles = pos.to(self.angle_dtype).unsqueeze(-1) * self.inv_freq.to(self.angle_dtype)
        cos = angles.cos().to(x.dtype).unsqueeze(-3)  # [..., 1, N, d/2]
        sin = angles.sin().to(x.dtype).unsqueeze(-3)
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
