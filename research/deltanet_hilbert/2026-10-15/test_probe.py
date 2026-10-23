import unittest
from importlib.util import find_spec

HAS_TORCH = find_spec('torch') is not None
if HAS_TORCH:
    import numpy as np
    import torch

    from probe import delta_step, feature_checks, operator_checks, recurrent, tensor_counterexample, triangular


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable')
class OperatorTests(unittest.TestCase):
    def test_sgd_and_matrix_oracle(self):
        self.assertLess(operator_checks()['recurrent_triangular_error'], 1e-10)

    def test_nonunit_key_can_break_contraction(self):
        state = torch.tensor([[1.]], dtype=torch.float64)
        after = delta_step(state, torch.tensor([2.], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64), 1.)
        self.assertEqual(float(after.square().sum()), 9.)

    def test_gated_update_decays_before_residual(self):
        state = torch.eye(2, dtype=torch.float64)
        key = torch.tensor([1., 0.], dtype=torch.float64)
        value = torch.tensor([2., 3.], dtype=torch.float64)
        after = delta_step(state, key, value, 1., decay=0.5)
        self.assertTrue(torch.equal(after @ key, value))
        self.assertTrue(torch.equal(after[:, 1], torch.tensor([0., 0.5], dtype=torch.float64)))

    def test_tensor_product_blocks_cross_phase_content(self):
        self.assertEqual(tensor_counterexample()['tensor_cross_residue_inner_product'], 0.)

    def test_primitive_features_match_both_grams(self):
        rows = feature_checks()
        self.assertTrue(all(r['period_shift_bitwise'] for r in rows))
        self.assertEqual([r['effective_rank'] for r in rows], [11, 12, 12, 13])

    def test_recurrence_has_no_future_access(self):
        generator = torch.Generator().manual_seed(9204)
        k = torch.eye(3, dtype=torch.float64).repeat(2, 1)
        v = torch.randn(6, 2, generator=generator, dtype=torch.float64)
        b = torch.full((6,), 0.25, dtype=torch.float64)
        before = recurrent(k, k, v, b)
        changed = v.clone()
        changed[4:] = 50
        after = recurrent(k, k, changed, b)
        self.assertTrue(torch.equal(before[:4], after[:4]))
        self.assertTrue(np.allclose(before.numpy(), triangular(k, k, v, b).numpy(), atol=1e-12))


if __name__ == '__main__':
    unittest.main()
