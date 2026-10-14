"""A minimal pre-LayerNorm decoder-only Transformer used for all experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .attention import CausalSelfAttention, PositionConfig


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 256
    pos: PositionConfig = field(default_factory=PositionConfig)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["pos"]["name"] = self.pos.name
        return out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.num_heads, cfg.pos)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, positions, key_padding_mask=None, backend="explicit"):
        x = x + self.attn(self.ln1(x), positions, key_padding_mask, backend)
        return x + self.mlp(self.ln2(x))


class TinyDecoder(nn.Module):
    """Token embedding -> ``num_layers`` pre-LN blocks -> LayerNorm -> vocabulary logits.

    There is no absolute position embedding: all position information enters
    through the attention layer's :class:`PositionConfig`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None, backend: str = "explicit") -> torch.Tensor:
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, positions, key_padding_mask, backend)
        return self.head(self.ln_f(x))

    def positional_parameters(self) -> list[nn.Parameter]:
        return [p for blk in self.blocks for p in blk.attn.biases.parameters()]

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        positional = sum(p.numel() for p in self.positional_parameters())
        return {"total": total, "positional": positional, "backbone": total - positional}
