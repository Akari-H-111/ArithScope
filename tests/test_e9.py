"""E9 machinery (docs/PREREGISTRATION.md, Section 12): window evaluation, seed-count rule, verdicts.

Synthetic inputs and a tiny model only; nothing here is E9 data.
"""
import random
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

HAS_TORCH = find_spec("torch") is not None and find_spec("scipy") is not None
if HAS_TORCH:
    from arithscope.train import RunSpec, evaluate, run
    from experiments import e9_power, e9_verdicts, e9_window_eval
    from experiments.suites import E9_WINDOWS

TIMING = ("eval_seconds", "tokens_per_second")


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class WindowEvaluationTests(unittest.TestCase):
    def test_windows_reuse_one_checkpoint_and_the_full_window_is_the_plain_evaluation(self):
        spec = RunSpec(experiment="t9", task={"kind": "periodic", "periods": [4]}, variant="arith", seed=0,
                       orders=(1, 2, 4), d_model=16, num_heads=2, d_ff=32, train_length=8, batch_size=4, steps=3,
                       warmup=1, val_every=3, eval_lengths=(8,), eval_tokens=64, eval_tokens_long=64)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run(spec, tmp / "runs", checkpoint_dir=tmp / "ckpt")
            ckpt = tmp / "ckpt" / (spec.filename[:-5] + ".pt")
            written = e9_window_eval.evaluate_windows(ckpt, (8, 16, 32), 32, tmp / "eval")
            self.assertEqual(len(written), 3)
            self.assertEqual(e9_window_eval.evaluate_windows(ckpt, (8, 16, 32), 32, tmp / "eval"), [])  # resumes
            import json

            recs = {r["window"]: r for r in (json.loads(p.read_text()) for p in written)}
            self.assertEqual(len({r["checkpoint_sha256"] for r in recs.values()}), 1)
            self.assertEqual(len({r["test"][0]["digest"] for r in recs.values()}), 1)
            # W = N is the ordinary unwindowed evaluation on the same test set, bit for bit
            _, task, model = e9_window_eval.load_checkpoint(ckpt)
            self.assertTrue(all(b.attn.window is None for b in model.blocks))
            from arithscope.train import task_dict

            parts = ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), 32)
            plain = evaluate(model, task, 32, 2, parts, spec.train_length)
            strip = lambda d: {k: v for k, v in d.items() if k not in TIMING}  # noqa: E731
            self.assertEqual(strip(recs[32]["test"][0]), strip(json.loads(json.dumps(plain))))
            self.assertNotEqual(recs[8]["test"][0]["all"]["nll"], recs[32]["test"][0]["all"]["nll"])

    def test_suite_trains_without_a_window_and_windows_cover_the_sequence(self):
        from experiments.suites import E9_LENGTH, e9_paired_window

        specs = e9_paired_window()
        self.assertTrue(all(s.attn_windows == () and s.variant == "arith" for s in specs))
        self.assertEqual(E9_WINDOWS[-1], E9_LENGTH)
        self.assertEqual(E9_WINDOWS[0], specs[0].train_length)


def _metrics(curve, seeds=range(27, 43), noise=0.002, acc=1.0, rng=None):
    rng = rng or random.Random(0)
    return {(p, s, w): {"far_nll": curve(w) + rng.gauss(0, noise), "far_acc": acc}
            for p in (12, 13) for s in seeds for w in E9_WINDOWS} if HAS_TORCH else {}


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class VerdictTests(unittest.TestCase):
    seeds = {12: list(range(27, 43)), 13: list(range(27, 43))}

    def test_slope(self):
        self.assertGreater(e9_verdicts.slope([1, 2, 3, 4]), 0)
        self.assertLess(e9_verdicts.slope([4, 3, 2, 1]), 0)
        self.assertAlmostEqual(e9_verdicts.slope([0.0] * 4), 0.0)

    def test_rising_curve_is_supported(self):
        out = e9_verdicts.decide(_metrics(lambda w: 0.01 + 0.05 * (w > 64)), self.seeds)
        self.assertEqual(out["verdicts"]["R1_window_dose_response"], "SUPPORTED")
        self.assertTrue(out["verdicts"]["S10_near_monotone"])
        self.assertTrue(out["verdicts"]["S11_w64_below_full"])

    def test_flat_or_falling_curve_is_falsified(self):
        for curve in (lambda w: 0.05, lambda w: 0.05 - 0.01 * (w > 64)):
            out = e9_verdicts.decide(_metrics(curve), self.seeds)
            self.assertTrue(out["verdicts"]["R1_window_dose_response"].startswith("FALSIFIED"))

    def test_accuracy_guard(self):
        m = _metrics(lambda w: 0.01 + 0.05 * (w > 64))
        for (_, _, w), v in m.items():
            if w == 64:
                v["far_acc"] = 0.9
        out = e9_verdicts.decide(m, self.seeds)
        self.assertIn("accuracy guard", out["verdicts"]["R1_window_dose_response"])

    def test_non_monotone_step_fails_s10_only(self):
        m = _metrics(lambda w: {64: 0.01, 256: 0.08, 1024: 0.05, 8192: 0.06}[w])
        out = e9_verdicts.decide(m, self.seeds)
        self.assertEqual(out["verdicts"]["R1_window_dose_response"], "SUPPORTED")
        self.assertFalse(out["verdicts"]["S10_near_monotone"])

    def test_sanity_detects_mixed_weights_and_test_sets(self):
        def rec(sha, digest, commit="c"):
            return {"checkpoint_sha256": sha, "test": [{"digest": digest}], "environment": {"git_commit": commit}}
        seeds = {12: [27]}
        good = {(12, 27, w): rec("a", "d") for w in E9_WINDOWS}
        training = {(12, 27): {"status": "COMPLETED"}}
        self.assertTrue(all(v for k, v in e9_verdicts.sanity(good, training, seeds).items() if k != "code_commits"))
        bad = dict(good)
        bad[(12, 27, 256)] = rec("b", "d")
        self.assertFalse(e9_verdicts.sanity(bad, training, seeds)["same_weights_across_windows"])
        bad[(12, 27, 256)] = rec("a", "e")
        self.assertFalse(e9_verdicts.sanity(bad, training, seeds)["one_test_set_per_period"])


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class SeedCountRuleTests(unittest.TestCase):
    def test_plan(self):
        rng = random.Random(1)
        strong = {p: [-0.05 + rng.gauss(0, 0.02) for _ in range(21)] for p in (12, 13)}
        out = e9_power.plan(strong)
        self.assertIsNotNone(out["n_seeds"])
        self.assertGreaterEqual(out["joint_power_at_n"], 0.90)
        self.assertGreaterEqual(out["n_seeds"], e9_power.N_MIN)
        if out["n_seeds"] > e9_power.N_MIN:  # minimality: no smaller n reaches the target
            ds = [c["d"] for c in out["periods"].values()]
            self.assertIsNone(e9_power.smallest_n(ds, cap=out["n_seeds"] - 1))

    def test_no_reduction_stops(self):
        out = e9_power.plan({12: [0.01, -0.005, 0.02, 0.0], 13: [-0.05, -0.04, -0.06, -0.05]})
        self.assertIsNone(out["n_seeds"])
        self.assertIn("no reduction", out["stop"])

    def test_noisy_effect_exceeds_the_budget(self):
        rng = random.Random(2)
        weak = {p: [-0.005 + rng.gauss(0, 0.05) for _ in range(21)] for p in (12, 13)}
        weak = {p: [v - (sum(vs) / len(vs)) - 0.005 for v in vs] for p, vs in weak.items()}  # mean exactly -0.005
        out = e9_power.plan(weak)
        self.assertIsNone(out["n_seeds"])
        self.assertIn("decide", out["stop"])

    def test_power_increases_with_seeds(self):
        self.assertLess(e9_power.power(0.8, 5), e9_power.power(0.8, 20))


if __name__ == "__main__":
    unittest.main()
