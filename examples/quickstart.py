"""Minimal usage of the ArithScope bias (runs in a few seconds on CPU).

    python examples/quickstart.py
"""
import torch

from arithscope import RamanujanBias, arith
from arithscope.attention import CausalSelfAttention, PositionConfig

torch.manual_seed(0)

# 1) The bias as a standalone module: gamma_h * sum_m alpha_{h,m} c_m(i - j) / phi(m)
bias = RamanujanBias(num_heads=2, orders=range(1, 17))      # gamma = 0 at init -> exactly zero bias
pos = torch.arange(32)
assert torch.equal(bias(pos, pos), torch.zeros(2, 32, 32))

with torch.no_grad():                                        # put a head on the provably optimal 12-comb (Thm T3)
    weights = arith.optimal_comb_weights(12)                 # {d: phi(d)/11 for d | 12, d > 1}
    logits = torch.full((16,), -30.0)
    for d, w in weights.items():
        logits[d - 1] = torch.log(torch.tensor(float(w)))
    bias.mixture_logits[0] = logits
    bias.gamma[0] = 6.0
b = bias(pos, pos)[0]
print("bias row for query 24 (offsets 24..0):", [round(v, 2) for v in b[24, :25].tolist()])
print("comb margin of this mixture:", float(arith.comb_margin([arith.mixture_value(weights, n) for n in range(12)], 12)))

# 2) Inside attention: a drop-in channel next to RoPE, with three equivalent backends
layer = CausalSelfAttention(d_model=64, num_heads=4, pos=PositionConfig.from_name("rope+arith"))
with torch.no_grad():
    layer.biases["arith"].gamma.fill_(2.0)
x = torch.randn(2, 128, 64)
p = torch.arange(128)
ref = layer(x, p, backend="explicit")                       # materialized logits
fused = layer(x, p, backend="folded")                       # bias folded into q/k features, no [N, N] tensor
print("explicit vs folded max |diff|:", (ref - fused).abs().max().item())
print("folded feature dim (sum of phi(m), m = 1..16):", layer.biases["arith"].feature_dim)
