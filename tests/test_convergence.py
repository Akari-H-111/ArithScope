"""T8 (MATH.md, section 4): in an ArithScope-only model without RoPE, logits converge per position type.

NUMERICAL illustration, not a proof. A bank of lcm 12 with period 12 keeps the number of types
(residue mod 12, visible token) small enough for a 4,096-token sequence to repeat each one often.
"""
import unittest
from importlib.util import find_spec

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from arithscope.attention import PositionConfig
    from arithscope.model import ModelConfig, TinyDecoder
    from arithscope.tasks import make_task


def within_type_spread(logits, types, start: int, end: int) -> float:
    """Mean squared distance of centered logits from their type's mean, over positions [start, end)."""
    z, ty = logits[start:end], types[start:end]
    total, count = 0.0, 0
    for k in ty.unique():
        sel = z[ty == k]
        if len(sel) > 1:
            total += float(((sel - sel.mean(0)) ** 2).sum())
            count += len(sel)
    return total / count


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class FiniteClassConvergenceTests(unittest.TestCase):
    def test_same_type_logits_draw_together(self):
        length, period = 4096, 12
        for seed in (0, 1):
            torch.manual_seed(seed)
            pos = PositionConfig.from_name("arith", orders=(1, 2, 3, 4, 6, 12))
            model = TinyDecoder(ModelConfig(vocab_size=16, pos=pos)).double().eval()
            with torch.no_grad():
                for p in model.parameters():  # random weights, ArithScope gates included
                    p.normal_(0, 0.3)
            batch = make_task("periodic", periods=[period]).sample(1, length, torch.Generator().manual_seed(seed))
            with torch.no_grad():
                logits = model(batch.tokens)[0]
            logits = logits - logits.mean(-1, keepdim=True)
            types = (torch.arange(length) % period) * 16 + batch.tokens[0]
            early = within_type_spread(logits, types, 512, 1024)
            late = within_type_spread(logits, types, length - 512, length)
            self.assertLess(late, 0.1 * early, seed)


if __name__ == "__main__":
    unittest.main()
