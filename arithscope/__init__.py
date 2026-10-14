"""ArithScope: exact arithmetic relative-position channels for Transformer attention."""
from importlib.util import find_spec

from . import arith

__all__ = ["arith"]
__version__ = "0.1.0"

if find_spec("torch") is not None:  # the exact arithmetic core stays dependency-free
    from .attention import CausalSelfAttention, PositionConfig
    from .bias import ALiBiBias, CosineBankBias, LearnedFrequencyBias, RamanujanBias, RelativeBias
    from .model import ModelConfig, TinyDecoder

    __all__ += ["CausalSelfAttention", "PositionConfig", "ALiBiBias", "CosineBankBias",
                "LearnedFrequencyBias", "RamanujanBias", "RelativeBias", "ModelConfig", "TinyDecoder"]
