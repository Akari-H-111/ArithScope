"""Opt-in evidence head: default semantics, zero-init equivalence, causality, shift invariance, exact Bayes construction."""
import json
import math
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch

    import arithscope.evidence as evidence_module
    from arithscope import arith
    from arithscope.attention import PositionConfig
    from arithscope.bias import CosineBankBias, _lookup
    from arithscope.evidence import KERNELS, EvidenceConfig, EvidenceHead
    from arithscope.model import ModelConfig, TinyDecoder
    from arithscope.tasks import PeriodicDenoising, make_task
    from arithscope.train import RunSpec, _optimizer, build_model, run
    from experiments.suites import SUITES

RUNS = Path(__file__).resolve().parent.parent / "results" / "runs"
VOCAB = 11


def _model(pos="arith", kernel=None, seed=0):
    torch.manual_seed(seed)
    ev = None if kernel is None else EvidenceConfig(kernel)
    return TinyDecoder(ModelConfig(vocab_size=VOCAB, pos=PositionConfig.from_name(pos), evidence=ev))


def _randomize(head, seed=3):
    """Nontrivial, reproducible kernel parameters and ``W_out`` (float64 heads)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64))
    return head


def _reference_counts(head, tokens, positions, valid):
    """Brute force ``E_i = sum_{j <= i, valid} kappa(pos_i - pos_j) onehot(x_j)``, loop by loop."""
    b, n = tokens.shape
    out = torch.zeros(b, n, head.w_out.shape[1], dtype=torch.float64)
    pos = positions.expand(b, n) if positions.dim() == 1 else positions
    for bi in range(b):
        for i in range(n):
            for j in range(i + 1):
                if valid is None or valid[bi, j]:
                    kappa = head.kernel.offset_values(torch.tensor([int(pos[bi, i] - pos[bi, j])]))[0, 0]
                    out[bi, i, tokens[bi, j]] += kappa
    return out


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class DefaultSemanticsTests(unittest.TestCase):
    def test_no_head_leaves_the_model_untouched(self):
        model = _model("rope+arith")
        self.assertIsNone(model.evidence)
        self.assertFalse(any(name.startswith("evidence") for name, _ in model.named_modules()))
        self.assertFalse(any(key.startswith("evidence") for key in model.state_dict()))
        self.assertNotIn("evidence", model.cfg.to_dict())
        self.assertEqual(model.count_parameters()["positional"],
                         sum(p.numel() for blk in model.blocks for p in blk.attn.biases.parameters()))

    def test_earlier_run_ids_and_model_records_are_unchanged(self):
        checked = 0
        for suite in ("e1_periodic", "e2_alias", "e3_tagged", "e5_mixed", "e6_calibration", "e7_floor",
                      "e8_power_retest", "e9_paired_window"):
            specs = SUITES[suite]()
            for spec in specs:
                self.assertTrue((RUNS / suite / spec.filename).exists(), (suite, spec.filename))
                checked += 1
            spec = specs[len(specs) // 2]
            stored = json.loads((RUNS / suite / spec.filename).read_text())["model"]
            vocab = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"}).model_vocab
            current = json.loads(json.dumps(build_model(spec, vocab).cfg.to_dict()))
            self.assertEqual({k: current[k] for k in stored}, stored)
            self.assertLessEqual(set(current) - set(stored), {"attn_windows"})  # E1-E5 predate that field
            self.assertNotIn("evidence", current)
        self.assertGreaterEqual(checked, 360)

    def test_evidence_changes_the_run_id_only_when_set(self):
        spec = SUITES["e1_periodic"]()[0]
        self.assertEqual(RunSpec(**{**spec.__dict__, "evidence": None}).filename, spec.filename)
        with_head = RunSpec(**{**spec.__dict__, "evidence": {"kernel": "arith"}})
        self.assertNotEqual(with_head.run_id, spec.run_id)
        self.assertNotEqual(with_head.run_id, RunSpec(**{**spec.__dict__, "evidence": {"kernel": "uniform"}}).run_id)

    def test_rejects_unknown_kernels(self):
        with self.assertRaises(ValueError):
            EvidenceConfig("softmax")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ZeroInitEquivalenceTests(unittest.TestCase):
    def _check(self, pos, kernel):
        base, cand = _model(pos), _model(pos, kernel)
        sd_c = cand.state_dict()
        self.assertTrue(all(torch.equal(v, sd_c[k]) for k, v in base.state_dict().items()))
        self.assertEqual(set(sd_c) - set(base.state_dict()), {k for k in sd_c if k.startswith("evidence.")})
        tokens = torch.randint(0, VOCAB, (3, 40))
        out_b, out_c = base(tokens), cand(tokens)
        self.assertTrue(torch.equal(out_b, out_c), (pos, kernel))
        target = torch.randint(0, VOCAB, (3, 40))
        torch.nn.functional.cross_entropy(out_b.flatten(0, 1), target.flatten()).backward()
        torch.nn.functional.cross_entropy(out_c.flatten(0, 1), target.flatten()).backward()
        grads_c = dict(cand.named_parameters())
        for name, p in base.named_parameters():
            self.assertTrue(torch.equal(p.grad, grads_c[name].grad), (pos, kernel, name))

    def test_logits_and_base_gradients_are_bitwise_identical(self):
        for kernel in KERNELS:
            for pos in ("nope", "rope", "arith", "rope+arith"):
                self._check(pos, kernel)

    def test_construction_draws_nothing_from_the_global_rng(self):
        torch.manual_seed(5)
        TinyDecoder(ModelConfig(vocab_size=VOCAB))
        reference = torch.get_rng_state()
        for kernel in KERNELS:
            torch.manual_seed(5)
            TinyDecoder(ModelConfig(vocab_size=VOCAB, evidence=EvidenceConfig(kernel)))
            self.assertTrue(torch.equal(torch.get_rng_state(), reference), kernel)

    def test_initial_values(self):
        arith_head = _model("arith", "arith").evidence
        self.assertTrue(torch.equal(arith_head.w_out, torch.zeros_like(arith_head.w_out)))
        self.assertTrue(torch.allclose(arith_head.kernel.beta, torch.full((16,), 1 / 16)))
        cosine = _model("arith", "cosine_learned").evidence.kernel
        periods = 2 * math.pi / cosine.omega.detach()
        self.assertAlmostEqual(float(periods[0]), 2.0, places=5)
        self.assertAlmostEqual(float(periods[-1]), 32.0, places=4)
        self.assertTrue(torch.allclose(cosine.amplitude, torch.full((16,), 1 / 16)))
        self.assertEqual(list(_model("arith", "uniform").evidence.kernel.parameters()), [])
        for kernel, size in (("cosine_fund", 16), ("cosine_full", len(CosineBankBias.full(range(1, 17))))):
            coef = _model("arith", kernel).evidence.kernel.coef
            self.assertTrue(torch.allclose(coef, torch.full((size,), 1 / size)), kernel)
        self.assertEqual(len(_model("arith", "cosine_fund").evidence.kernel.coef), len(_model("arith", "arith").evidence.kernel.beta))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class CountSemanticsTests(unittest.TestCase):
    def _head(self, kernel):
        return _randomize(EvidenceHead(EvidenceConfig(kernel), VOCAB, VOCAB).double())

    def test_counts_match_a_brute_force_reference(self):
        gen = torch.Generator().manual_seed(1)
        tokens = torch.randint(0, VOCAB, (2, 30), generator=gen)
        valid = torch.rand(2, 30, generator=gen) > 0.25
        for kernel in KERNELS:
            head = self._head(kernel)
            for positions in (torch.arange(30), torch.arange(30) + 10**12,
                              torch.stack([torch.arange(30) * 3 + 5, torch.randperm(30, generator=gen)])):
                for mask in (None, valid):
                    got = head.counts(tokens, positions, mask)
                    ref = _reference_counts(head, tokens, positions, mask)
                    self.assertTrue(torch.allclose(got, ref, atol=1e-11, rtol=0), (kernel, positions.dim(), mask is None))

    def test_causality_is_by_token_index_not_position_value(self):
        gen = torch.Generator().manual_seed(2)
        tokens = torch.randint(0, VOCAB, (2, 36), generator=gen)
        later = tokens.clone()
        later[:, 20:] = (later[:, 20:] + 1) % VOCAB
        for kernel in KERNELS:
            head = self._head(kernel)
            for positions in (torch.arange(36), torch.randperm(36, generator=gen), 100 - torch.arange(36)):
                a, b = head.counts(tokens, positions), head.counts(later, positions)
                self.assertTrue(torch.equal(a[:, :20], b[:, :20]), kernel)
                self.assertFalse(torch.equal(a[:, 20:], b[:, 20:]), kernel)

    def test_padded_keys_contribute_nothing(self):
        gen = torch.Generator().manual_seed(4)
        tokens = torch.randint(0, VOCAB, (2, 30), generator=gen)
        valid = torch.rand(2, 30, generator=gen) > 0.3
        other = torch.where(valid, tokens, (tokens + 5) % VOCAB)
        positions = torch.arange(30)
        for kernel in KERNELS:
            head = self._head(kernel)
            self.assertTrue(torch.equal(head.counts(tokens, positions, valid), head.counts(other, positions, valid)))

    def test_query_blocking_is_exact(self):
        tokens = torch.randint(0, VOCAB, (2, 50), generator=torch.Generator().manual_seed(6))
        positions = torch.arange(50)
        saved = evidence_module.EVIDENCE_QUERY_BLOCK
        try:
            for kernel in KERNELS:
                head = self._head(kernel)
                full = head.counts(tokens, positions)
                evidence_module.EVIDENCE_QUERY_BLOCK = 7
                self.assertTrue(torch.allclose(full, head.counts(tokens, positions), atol=1e-12, rtol=0), kernel)
                evidence_module.EVIDENCE_QUERY_BLOCK = saved
        finally:
            evidence_module.EVIDENCE_QUERY_BLOCK = saved

    def test_kernels_are_exactly_shift_invariant(self):
        tokens = torch.randint(0, VOCAB, (2, 64), generator=torch.Generator().manual_seed(7))
        for kernel in KERNELS:
            head = self._head(kernel)
            base = head.counts(tokens, torch.arange(64))
            self.assertTrue(torch.equal(base, head.counts(tokens, torch.arange(64) + 10**12)), kernel)

    def test_arith_model_with_a_trained_head_is_shift_invariant(self):
        model = _model("arith", "arith")
        with torch.no_grad():
            model.evidence.w_out.copy_(torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(8)))
        tokens = torch.randint(0, VOCAB, (2, 48), generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(model(tokens, torch.arange(48)), model(tokens, torch.arange(48) + 10**12)))

    def test_head_moves_the_logits_once_w_out_is_nonzero(self):
        model = _model("rope", "uniform")
        tokens = torch.randint(0, VOCAB, (1, 20))
        before = model(tokens)
        with torch.no_grad():
            model.evidence.w_out.fill_(0.1)
        self.assertFalse(torch.equal(before, model(tokens)))
        # a uniform kernel counts every earlier token: the first position sees one token, the last twenty
        counts = model.evidence.counts(tokens, torch.arange(20))
        self.assertTrue(torch.equal(counts.sum(-1), torch.arange(1.0, 21.0).expand(1, 20)))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ExactBayesConstructionTests(unittest.TestCase):
    """beta_m = phi(m) / P on divisors, W_out = log(rho) I, zero base logits: softmax is the Bayes posterior (B1)."""

    def _check(self, period, orders):
        task = PeriodicDenoising(periods=(period,))
        rho = (1 - task.noise + task.noise / task.vocab_size) / (task.noise / task.vocab_size)
        self.assertAlmostEqual(rho, 17.0, places=12)
        batch = task.sample(2, 100, torch.Generator().manual_seed(period))
        pos = PositionConfig.from_name("nope", orders=orders)
        torch.manual_seed(0)
        model = TinyDecoder(ModelConfig(vocab_size=16, pos=pos, evidence=EvidenceConfig("arith", orders))).double()
        with torch.no_grad():
            model.head.weight.zero_()
            model.evidence.kernel.beta.copy_(torch.tensor(
                [arith.euler_phi(m) / period if period % m == 0 else 0.0 for m in orders], dtype=torch.float64))
            model.evidence.w_out.copy_(math.log(rho) * torch.eye(16, dtype=torch.float64))
        probs = torch.softmax(model(batch.tokens), dim=-1)
        for b in range(2):
            for t in range(100):
                counts = torch.zeros(16, dtype=torch.float64)
                for j in range(t % period, t + 1, period):  # j <= t, j = t mod P, own token included
                    counts[batch.tokens[b, j]] += 1
                posterior = rho ** counts / (rho ** counts).sum()
                self.assertTrue(torch.allclose(probs[b, t], posterior, atol=1e-9, rtol=0), (period, b, t))

    def test_period_12_default_bank(self):
        self._check(12, tuple(range(1, 17)))

    def test_period_12_divisor_bank(self):
        self._check(12, (1, 2, 3, 4, 6, 12))

    def test_period_13_default_bank(self):
        self._check(13, tuple(range(1, 17)))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class CombSpanTests(unittest.TestCase):
    """D_P = 1[P | n] is in the span of the arith and full-cosine kernels, not of the fundamental cosines."""

    def _relative_residual(self, kernel, period, length=20160):
        head = EvidenceHead(EvidenceConfig(kernel), VOCAB, VOCAB).double()
        offsets = torch.arange(length)
        columns = _lookup(head.kernel.table, head.kernel.row_orders, offsets).t()  # [R, components]
        target = (offsets % period == 0).double()
        fit = torch.linalg.lstsq(columns, target.unsqueeze(1)).solution
        residual = target - (columns @ fit).squeeze(1)
        return float(residual.pow(2).mean().sqrt() / target.pow(2).mean().sqrt())

    def test_residuals(self):
        for period in (12, 13):
            self.assertLess(self._relative_residual("arith", period), 1e-10, period)
            self.assertLess(self._relative_residual("cosine_full", period), 1e-10, period)
            self.assertGreater(self._relative_residual("cosine_fund", period), 0.1, period)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class ReportBayesNllTests(unittest.TestCase):
    def test_matches_a_brute_force_posterior(self):
        from experiments.pilot_evidence_report import bayes_nll

        for period in (5, 12):
            task = PeriodicDenoising(periods=(period,))
            batch = task.sample(2, 47, torch.Generator().manual_seed(period))  # length not a multiple of P
            rho = 17.0
            got = bayes_nll(batch.tokens, batch.targets, period, rho, 16)
            for b in range(2):
                for t in range(47):
                    counts = torch.zeros(16, dtype=torch.float64)
                    for j in range(t % period, t + 1, period):
                        counts[batch.tokens[b, j]] += 1
                    posterior = rho ** counts / (rho ** counts).sum()
                    self.assertAlmostEqual(float(got[b, t]), -math.log(float(posterior[batch.targets[b, t]])), places=9)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TrainingIntegrationTests(unittest.TestCase):
    SPEC = dict(experiment="t_evidence", task={"kind": "periodic", "periods": [5]}, variant="arith", seed=0,
                d_model=16, num_layers=1, num_heads=2, d_ff=32, batch_size=4, train_length=16, steps=4, warmup=1,
                val_every=2, eval_lengths=(16,), eval_tokens=64, eval_tokens_long=64)

    def test_optimizer_groups(self):
        spec = RunSpec(**self.SPEC, evidence={"kernel": "arith"})
        model = build_model(spec, 16)
        groups = _optimizer(model, spec).param_groups
        in_group = {id(p): g["lr_mult"] for g in groups for p in g["params"]}
        self.assertEqual(in_group[id(model.evidence.kernel.beta)], spec.pos_lr_mult)
        self.assertEqual(in_group[id(model.evidence.w_out)], 1.0)
        for kernel, name in (("cosine_learned", "omega"), ("cosine_learned", "amplitude")):
            model = build_model(RunSpec(**self.SPEC, evidence={"kernel": kernel}), 16)
            self.assertIn(id(getattr(model.evidence.kernel, name)), {id(p) for p in model.positional_parameters()})

    def test_record_has_a_separate_evidence_key_and_unchanged_positional_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = run(RunSpec(**self.SPEC), Path(tmp) / "a")
            with_head = run(RunSpec(**self.SPEC, evidence={"kernel": "arith"}), Path(tmp) / "b")
            cosine = run(RunSpec(**self.SPEC, evidence={"kernel": "cosine_learned"}), Path(tmp) / "c")
            uniform = run(RunSpec(**self.SPEC, evidence={"kernel": "uniform"}), Path(tmp) / "d")
            fund = run(RunSpec(**self.SPEC, evidence={"kernel": "cosine_fund"}), Path(tmp) / "e")
            full = run(RunSpec(**self.SPEC, evidence={"kernel": "cosine_full"}), Path(tmp) / "f")
        self.assertEqual(set(fund["evidence"]), {"kernel", "orders", "components", "coef", "w_out"})
        self.assertEqual(len(fund["evidence"]["coef"]), 16)
        self.assertEqual(len(full["evidence"]["coef"]), len(full["evidence"]["components"]))
        self.assertGreater(len(full["evidence"]["coef"]), 16)
        self.assertNotIn("evidence", plain)
        self.assertNotIn("evidence", plain["model"])
        self.assertEqual(set(with_head["positional"][0]), set(plain["positional"][0]))
        self.assertNotIn("evidence", with_head["positional"][0])
        ev = with_head["evidence"]
        self.assertEqual(ev["kernel"], "arith")
        self.assertEqual(len(ev["beta"]), 16)
        self.assertEqual(ev["w_out"]["shape"], [16, 16])
        self.assertGreater(ev["w_out"]["frobenius"], 0.0)  # W_out moved off zero
        self.assertEqual(len(ev["w_out"]["diag"]), 16)
        self.assertEqual(set(cosine["evidence"]), {"kernel", "omega", "amplitude", "w_out"})
        self.assertEqual(set(uniform["evidence"]), {"kernel", "w_out"})
        self.assertEqual(with_head["model"]["evidence"]["kernel"], "arith")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class PilotSuiteTests(unittest.TestCase):
    def test_arms_and_no_filename_collisions(self):
        specs = SUITES["pilot_evidence"]()
        self.assertEqual(len(specs), 20)
        arms = {(s.task["periods"][0], s.variant, (s.evidence or {}).get("kernel")) for s in specs}
        self.assertEqual(len(arms), 20)
        for period in (12, 13):
            self.assertEqual({(v, k) for p, v, k in arms if p == period},
                             {("arith", k) for k in (None, "arith", "cosine_fund", "cosine_full", "cosine_learned", "uniform")} |
                             {("rope", None), ("rope", "arith"), ("rope+arith", None), ("rope+arith", "arith")})
        self.assertTrue(all(s.data_seed == 7 and s.seed == 0 and s.eval_lengths == (64, 256, 1024, 4096, 8192)
                            for s in specs))
        mine = {s.filename for s in specs}
        self.assertEqual(len(mine), 20)
        others = {s.filename for name, fn in SUITES.items() if name != "pilot_evidence" for s in fn()}
        self.assertFalse(mine & others)
        self.assertFalse({s.experiment for s in specs} & {s.experiment for name, fn in SUITES.items()
                                                          if name != "pilot_evidence" for s in fn()})


if __name__ == "__main__":
    unittest.main()
