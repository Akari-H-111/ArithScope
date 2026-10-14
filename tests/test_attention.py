"""Attention/model semantics: gamma=0 equivalence, backends, masks, gradients, training."""
import os
import unittest
import warnings
from importlib.util import find_spec

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from arithscope.attention import CausalSelfAttention, PositionConfig
    from arithscope.model import ModelConfig, TinyDecoder


def _pair(base_name: str, cand_name: str, dtype=None, seed: int = 0):
    """Baseline and candidate models whose shared parameters are identical."""
    torch.manual_seed(seed)
    base = TinyDecoder(ModelConfig(vocab_size=11, pos=PositionConfig.from_name(base_name)))
    cand = TinyDecoder(ModelConfig(vocab_size=11, pos=PositionConfig.from_name(cand_name)))
    missing, unexpected = cand.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(".biases." in k for k in missing), (missing, unexpected)
    if dtype is not None:
        base, cand = base.to(dtype), cand.to(dtype)
    return base, cand


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class GammaZeroEquivalenceTests(unittest.TestCase):
    def _check(self, base_name, cand_name):
        base, cand = _pair(base_name, cand_name)
        tokens = torch.randint(0, 11, (3, 40))
        out_b, out_c = base(tokens), cand(tokens)
        self.assertTrue(torch.equal(out_b, out_c), f"{cand_name} at gamma=0 differs from {base_name}")
        target = torch.randint(0, 11, (3, 40))
        torch.nn.functional.cross_entropy(out_b.flatten(0, 1), target.flatten()).backward()
        torch.nn.functional.cross_entropy(out_c.flatten(0, 1), target.flatten()).backward()
        grads_c = dict(cand.named_parameters())
        for name, p in base.named_parameters():
            self.assertTrue(torch.equal(p.grad, grads_c[name].grad), name)

    def test_rope_plus_arith_equals_rope(self):
        self._check("rope", "rope+arith")

    def test_arith_equals_nope(self):
        self._check("nope", "arith")

    def test_alibi_plus_arith_equals_alibi(self):
        self._check("alibi", "alibi+arith")

    def test_zero_initialized_controls_equal_rope(self):
        for name in ("rope+fourier_fund", "rope+fourier_full", "rope+fourier_learned", "rope+relbias"):
            self._check("rope", name)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class BackendAndMaskTests(unittest.TestCase):
    def _attn(self, name, seed=1):
        torch.manual_seed(seed)
        layer = CausalSelfAttention(32, 4, PositionConfig.from_name(name)).double()
        for module in layer.biases.values():
            with torch.no_grad():
                for p in module.parameters():
                    p.normal_()
        return layer

    def test_backends_agree(self):
        x = torch.randn(2, 37, 32, dtype=torch.float64)
        pos = torch.arange(37)
        for name in ("arith", "rope+arith"):
            layer = self._attn(name)
            ref = layer(x, pos, backend="explicit")
            for backend in ("sdpa", "folded"):
                self.assertTrue(torch.allclose(ref, layer(x, pos, backend=backend), atol=1e-10), (name, backend))
        for name in ("rope", "alibi", "rope+fourier_full", "rope+relbias"):
            layer = self._attn(name)
            self.assertTrue(torch.allclose(layer(x, pos, backend="explicit"), layer(x, pos, backend="sdpa"), atol=1e-10))

    def test_backends_agree_with_padding_and_batched_positions(self):
        layer = self._attn("rope+arith")
        x = torch.randn(2, 20, 32, dtype=torch.float64)
        pos = torch.stack([torch.arange(20), torch.arange(20) + 1000])
        valid = torch.ones(2, 20, dtype=torch.bool)
        valid[0, :3] = False  # left padding in the first sequence
        ref = layer(x, pos, valid, backend="explicit")
        for backend in ("sdpa", "folded"):
            got = layer(x, pos, valid, backend=backend)
            self.assertTrue(torch.allclose(ref, got, atol=1e-10), backend)

    def test_flex_backend_matches_explicit(self):
        from arithscope import flex

        if not flex.available():
            self.skipTest("FlexAttention is not available in this torch build")
        x = torch.randn(2, 37, 32, dtype=torch.float64)
        pos = torch.stack([torch.arange(37), torch.arange(37) + 1000])
        valid = torch.ones(2, 37, dtype=torch.bool)
        valid[0, :3] = False  # rows 0-2 of the first sequence have no valid key
        for name in ("rope+arith", "arith", "rope+fourier_full", "rope+fourier_learned", "rope+relbias", "alibi",
                     "rope", "nope"):
            layer = self._attn(name)
            with torch.no_grad(), warnings.catch_warnings():  # FlexAttention has no CPU backward
                warnings.simplefilter("ignore", UserWarning)  # eager (uncompiled) FlexAttention warns
                ref = layer(x, pos, valid, backend="explicit")
                got = layer(x, pos, valid, backend="flex")
            self.assertTrue(torch.allclose(ref, got, atol=1e-12, rtol=0), name)

    @unittest.skipUnless(os.environ.get("ARITHSCOPE_TEST_COMPILED_FLEX") == "1",
                         "opt-in: set ARITHSCOPE_TEST_COMPILED_FLEX=1 to compile FlexAttention kernels (slow)")
    def test_compiled_flex_backend_matches_explicit(self):
        from arithscope import flex

        layer = self._attn("rope+arith").float()
        x = torch.randn(2, 300, 32)
        pos = torch.stack([torch.arange(300), torch.arange(300) + 10**6])
        valid = torch.ones(2, 300, dtype=torch.bool)
        valid[0, :5] = False
        try:
            flex.compile_kernels(True)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                got = layer(x, pos, valid, backend="flex")
                ref = layer(x, pos, valid, backend="explicit")
        finally:
            flex.compile_kernels(False)
        self.assertTrue(torch.allclose(ref, got, atol=1e-5, rtol=0))
        self.assertTrue(torch.equal(got[0, :5], torch.zeros(5, 32)))

    def test_explicit_query_blocking_is_exact(self):
        import arithscope.attention as attention_module

        x = torch.randn(2, 45, 32, dtype=torch.float64)
        pos = torch.stack([torch.arange(45), torch.arange(45) + 7])
        valid = torch.ones(2, 45, dtype=torch.bool)
        valid[1, :5] = False
        for name in ("rope+arith", "rope+relbias", "alibi"):
            layer = self._attn(name)
            ref = layer(x, pos, valid)  # 45 <= default block: unblocked
            saved = attention_module.EXPLICIT_QUERY_BLOCK
            try:
                attention_module.EXPLICIT_QUERY_BLOCK = 7  # 7 blocks, last one partial
                blocked = layer(x, pos, valid)
            finally:
                attention_module.EXPLICIT_QUERY_BLOCK = saved
            self.assertTrue(torch.allclose(ref, blocked, atol=1e-12, rtol=0), name)

    def test_causality(self):
        layer = self._attn("rope+arith")
        x = torch.randn(1, 30, 32, dtype=torch.float64)
        pos = torch.arange(30)
        y = layer(x, pos)
        x2 = x.clone()
        x2[:, 20:] = torch.randn(1, 10, 32, dtype=torch.float64)
        self.assertTrue(torch.equal(y[:, :20], layer(x2, pos)[:, :20]))

    def test_key_padding_and_fully_masked_rows(self):
        layer = self._attn("rope+arith")
        x = torch.randn(1, 12, 32, dtype=torch.float64)
        pos = torch.arange(12)
        valid = torch.ones(1, 12, dtype=torch.bool)
        valid[0, :4] = False
        y = layer(x, pos, valid, backend="explicit")
        self.assertTrue(torch.isfinite(y).all())
        # Rows 0..3 have no valid key: attention output is zero, so the block output is proj(0) = 0.
        self.assertTrue(torch.equal(y[0, :4], torch.zeros(4, 32, dtype=torch.float64)))
        # Changing padded keys does not change valid rows.
        x2 = x.clone()
        x2[:, :4] = torch.randn(1, 4, 32, dtype=torch.float64)
        self.assertTrue(torch.allclose(y[0, 4:], layer(x2, pos, valid)[0, 4:], atol=1e-12))

    def test_arith_only_model_is_exactly_shift_invariant(self):
        torch.manual_seed(3)
        model = TinyDecoder(ModelConfig(vocab_size=11, pos=PositionConfig.from_name("arith")))
        for blk in model.blocks:
            with torch.no_grad():
                blk.attn.biases["arith"].gamma.fill_(2.0)
                blk.attn.biases["arith"].mixture_logits.normal_()
        tokens = torch.randint(0, 11, (2, 48))
        pos = torch.arange(48)
        y = model(tokens, pos)
        for shift in (10**4, 10**9, 10**15):
            self.assertTrue(torch.equal(y, model(tokens, pos + shift)), shift)

    def test_non_contiguous_input_and_long_positions(self):
        layer = self._attn("rope+arith").float()
        x = torch.randn(2, 64, 32).transpose(0, 1).contiguous().transpose(0, 1)[:, ::2]
        self.assertFalse(x.is_contiguous())
        pos = torch.arange(32) + 5_000_000
        y = layer(x, pos)
        self.assertTrue(torch.isfinite(y).all())


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TrainingSmokeTests(unittest.TestCase):
    def test_tiny_model_learns_a_periodic_sequence(self):
        torch.manual_seed(0)
        model = TinyDecoder(ModelConfig(vocab_size=8, d_model=32, num_layers=1, num_heads=2, d_ff=64,
                                        pos=PositionConfig.from_name("arith")))
        opt = torch.optim.Adam(model.parameters(), lr=3e-2)
        period, n = 5, 40
        losses = []
        for _ in range(150):
            motif = torch.randint(0, 8, (16, period))
            tokens = motif.repeat(1, n // period)
            noisy = torch.where(torch.rand(16, n) < 0.3, torch.randint(0, 8, (16, n)), tokens)
            loss = torch.nn.functional.cross_entropy(model(noisy).flatten(0, 1), tokens.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertTrue(all(map(lambda v: v == v, losses)))
        self.assertLess(sum(losses[-10:]) / 10, 0.6 * sum(losses[:10]) / 10)
        gammas = [blk.attn.biases["arith"].gamma.abs().max().item() for blk in model.blocks]
        self.assertGreater(max(gammas), 0.1)


if __name__ == "__main__":
    unittest.main()
