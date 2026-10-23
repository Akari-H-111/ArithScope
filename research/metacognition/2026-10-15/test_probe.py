"""Derivative, causal-input, and no-update checks, separate from main tests."""
import unittest
from importlib.util import find_spec

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import sys
    from pathlib import Path

    import torch

    from probe import analytic_checks, jacobian_collision, measure, replication_checks, state_digest, window_prediction

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from arithscope.attention import PositionConfig
    from arithscope.model import ModelConfig, TinyDecoder


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class GradientChecks(unittest.TestCase):
    def test_two_derivative_implementations(self):
        self.assertLess(analytic_checks()["maximum_central_difference_error"], 1e-7)

    def test_replication_invariance(self):
        self.assertLess(replication_checks()["maximum_all_parameter_gradient_difference"], 1e-12)

    def test_forward_collision_does_not_force_gradient_collision(self):
        rows = jacobian_collision()
        self.assertEqual(rows[0]["top_probability"], rows[1]["top_probability"])
        self.assertNotEqual(rows[0]["gradient"], rows[1]["gradient"])

    def test_monitor_uses_only_prefix_and_preserves_weights(self):
        torch.set_num_threads(1)
        torch.manual_seed(9113)
        cfg = ModelConfig(vocab_size=3, d_model=8, num_heads=2, d_ff=16,
                          pos=PositionConfig.from_name("arith", orders=(1, 2, 3)))
        model = TinyDecoder(cfg).double().eval()
        tokens = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.int64)
        changed = tokens.clone()
        changed[:, 4:] = 0
        before = state_digest(model)
        a, b = measure(model, tokens, 3), measure(model, changed, 3)
        self.assertEqual(a, b)
        self.assertEqual(a["probability"], window_prediction(model, tokens, 3, 64))
        self.assertEqual(before, state_digest(model))
        self.assertIsNone(model.blocks[-1].attn.window)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
