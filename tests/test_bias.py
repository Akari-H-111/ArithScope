"""PyTorch bias modules versus the exact arithmetic core."""
import math
import unittest
from importlib.util import find_spec

from arithscope import arith

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from arithscope.bias import (ALiBiBias, CosineBankBias, LearnedFrequencyBias, RamanujanBias,
                                 RelativeBias)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class RamanujanBiasTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.orders = tuple(range(1, 17))
        self.bias = RamanujanBias(3, self.orders).double()
        with torch.no_grad():
            self.bias.mixture_logits.normal_()
            self.bias.gamma.copy_(torch.tensor([1.5, -0.7, 3.0]))

    def test_matches_exact_mixture(self):
        alpha, gamma = self.bias.alpha.detach(), self.bias.gamma.detach()
        offsets = torch.arange(-200, 201)
        got = self.bias.offset_values(offsets).detach()
        for h in range(3):
            for idx, n in enumerate(offsets.tolist()):
                exact = sum(float(alpha[h, g]) * float(arith.kernel(m, n)) for g, m in enumerate(self.orders))
                self.assertAlmostEqual(got[h, idx].item(), float(gamma[h]) * exact, places=12)

    def test_dense_shapes_offsets_and_symmetry(self):
        q_pos = torch.arange(40, 47)  # e.g. 7 new queries after a 40-token KV cache
        k_pos = torch.arange(0, 47)
        out = self.bias(q_pos, k_pos)
        self.assertEqual(out.shape, (3, 7, 47))
        # Values for a given offset agree across calls with different offset ranges up to
        # floating-point rounding (the mixture matmul may round differently by width); calls
        # with identical offset sets are bitwise identical (see test_exact_shift_invariance).
        values = self.bias.offset_values(torch.arange(-46, 47))
        for i, qi in enumerate(q_pos.tolist()):
            for j, kj in enumerate(k_pos.tolist()):
                self.assertTrue(torch.allclose(out[:, i, j], values[:, qi - kj + 46], atol=1e-12, rtol=0))
        sq = self.bias(k_pos, k_pos)
        self.assertTrue(torch.allclose(sq, sq.transpose(-1, -2), atol=1e-12, rtol=0))
        batched = self.bias(torch.stack([q_pos, q_pos + 5]), torch.stack([k_pos, k_pos + 5]))
        self.assertEqual(batched.shape, (2, 3, 7, 47))
        self.assertTrue(torch.equal(batched[0], batched[1]))

    def test_exact_shift_invariance(self):
        pos = torch.arange(0, 64)
        base = self.bias(pos, pos)
        for shift in (1, 10**6, 10**12, 2**61):
            self.assertTrue(torch.equal(base, self.bias(pos + shift, pos + shift)))

    def test_folded_features_reproduce_dense_bias(self):
        self.assertEqual(self.bias.feature_dim, arith.folded_rank(self.orders))
        for shift in (0, 12345, 10**15):
            pos = torch.arange(0, 50) + shift
            qf, kf = self.bias.query_features(pos), self.bias.key_features(pos)
            folded = qf @ kf.transpose(-1, -2)
            self.assertTrue(torch.allclose(folded, self.bias(pos, pos), atol=1e-12, rtol=0))

    def test_kernel_mixture_is_psd_with_unit_diagonal(self):
        unit = RamanujanBias(2, self.orders).double()
        with torch.no_grad():
            unit.mixture_logits.normal_()
            unit.gamma.fill_(1.0)
        pos = torch.arange(0, 90)
        gram = unit(pos, pos)
        self.assertTrue(torch.allclose(torch.diagonal(gram, dim1=-2, dim2=-1), torch.ones(2, 90, dtype=torch.float64)))
        self.assertGreaterEqual(torch.linalg.eigvalsh(gram).min().item(), -1e-10)
        self.assertLessEqual(gram.abs().max().item(), 1 + 1e-12)

    def test_gamma_zero_is_exact_zero_and_alpha_gradient_trap(self):
        fresh = RamanujanBias(2, self.orders)
        pos = torch.arange(0, 32)
        out = fresh(pos, pos)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        out.sum().backward()
        self.assertTrue(torch.equal(fresh.mixture_logits.grad, torch.zeros_like(fresh.mixture_logits)))
        self.assertNotEqual(fresh.gamma.grad.abs().sum().item(), 0.0)

    def test_gradients_reach_mixture_when_gamma_nonzero(self):
        pos = torch.arange(0, 32)
        loss = (self.bias(pos, pos) * torch.randn(3, 32, 32, dtype=torch.float64)).sum()
        loss.backward()
        self.assertGreater(self.bias.mixture_logits.grad.abs().sum().item(), 0)
        self.assertGreater(self.bias.gamma.grad.abs().sum().item(), 0)

    def test_rejects_float_positions_and_bad_orders(self):
        with self.assertRaises(TypeError):
            self.bias(torch.arange(4.0), torch.arange(4.0))
        with self.assertRaises(ValueError):
            RamanujanBias(2, (3, 3))
        with self.assertRaises(ValueError):
            RamanujanBias(2, (0, 2))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class BaselineBiasTests(unittest.TestCase):
    def test_cosine_banks(self):
        self.assertEqual(CosineBankBias.fundamental((1, 2, 5)), [(1, 1), (2, 1), (5, 1)])
        full = CosineBankBias.full(range(1, 17))
        self.assertEqual(len(full), 41)
        self.assertIn((12, 5), full)
        self.assertNotIn((12, 7), full)  # cos is even: a and m - a are the same component
        bank = CosineBankBias(1, full).double()
        offsets = torch.arange(-30, 31)
        comps = bank.component_values(offsets)
        for c, (m, a) in enumerate(full):
            expected = torch.cos(2 * math.pi * a * offsets.double() / m)
            self.assertTrue(torch.allclose(comps[c], expected, atol=1e-12))

    def test_ramanujan_span_is_inside_full_cosine_span(self):
        # K_m = (2 / phi(m)) * sum_{a in U_m, a < m/2} cos(2 pi a n / m) for m > 2.
        offsets = torch.arange(-40, 41)
        full = CosineBankBias.full(range(1, 17))
        comps = CosineBankBias(1, full).double().component_values(offsets)
        for m in range(3, 17):
            rows = [c for c, (mm, _) in enumerate(full) if mm == m]
            recon = comps[rows].sum(0) * 2 / arith.euler_phi(m)
            exact = torch.tensor([float(arith.kernel(m, n)) for n in offsets.tolist()], dtype=torch.float64)
            self.assertTrue(torch.allclose(recon, exact, atol=1e-12))

    def test_learned_frequency_relative_alibi(self):
        pos = torch.arange(0, 20)
        lf = LearnedFrequencyBias(2, 8)
        self.assertTrue(torch.equal(lf(pos, pos), torch.zeros(2, 20, 20)))
        rel = RelativeBias(2, max_distance=5)
        with torch.no_grad():
            rel.table.copy_(torch.arange(11.0).repeat(2, 1))
        out = rel(pos, pos)
        self.assertEqual(out[0, 19, 0].item(), 10.0)  # offset 19 clipped to +5 -> index 10
        self.assertEqual(out[0, 3, 1].item(), 7.0)    # offset 2 -> index 7
        alibi = ALiBiBias(4)
        out = alibi(pos, pos)
        self.assertAlmostEqual(out[0, 10, 3].item(), -7 * 2 ** -2)
        self.assertAlmostEqual(out[3, 10, 3].item(), -7 * 2 ** -8)


if __name__ == "__main__":
    unittest.main()
