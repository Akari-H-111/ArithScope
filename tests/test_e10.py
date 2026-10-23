"""E10 machinery (research/evidence_readout/2026-10-23/E10_SPEC.md): copied law, Bayes NLL, suites, copy evaluation,
seed-count rule, verdicts. Synthetic inputs and tiny models only; nothing here is E10 data."""
import json
import math
import random
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

HAS_TORCH = find_spec("torch") is not None and find_spec("scipy") is not None
if HAS_TORCH:
    import torch

    from arithscope.tasks import CopiedPeriodicDenoising, PeriodicDenoising, make_task, periodic_bayes_nll, task_dict
    from arithscope.train import RunSpec, run
    from experiments import e10_copy_eval, e10_power, e10_verdicts
    from experiments.suites import (E10_ARMS, E10_DATA_SEED, E10_FIRST_SEED, E10_SEEDS, E10_SPREAD_SEEDS, SUITES,
                                    e10_arm)

LOG17 = math.log(17)


def _brute_force_nll(batch, task):
    """-log of the product-likelihood posterior over the true symbol, by explicit loops (uniform prior)."""
    v, eps = task.vocab_size, task.noise
    hit, miss = 1 - eps + eps / v, eps / v
    b, n = batch.tokens.shape
    out = torch.zeros(b, n, dtype=torch.float64)
    for bi in range(b):
        period = int(batch.group[bi])
        for t in range(n):
            loglik = torch.zeros(v, dtype=torch.float64)
            for j in range(t % period, t + 1, period):
                if batch.observed is not None and not batch.observed[bi, j]:
                    continue
                for s in range(v):
                    loglik[s] += math.log(hit if int(batch.tokens[bi, j]) == s else miss)
            out[bi, t] = -(loglik[int(batch.targets[bi, t])] - torch.logsumexp(loglik, 0))
    return out


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class CopiedLawTests(unittest.TestCase):
    def _batches(self, copies, periods=(12,), length=100):
        gen = lambda: torch.Generator().manual_seed(11)  # noqa: E731
        base = PeriodicDenoising(periods=periods).sample(3, length, gen())
        copied = CopiedPeriodicDenoising(periods=periods, copies=copies).sample(3, length, gen())
        return base, copied

    def test_one_copy_is_the_independent_law_bitwise(self):
        for periods in ((12,), (5, 7)):
            base, copied = self._batches(1, periods)
            for name in ("tokens", "targets", "loss_mask", "oracle", "group"):
                self.assertTrue(torch.equal(getattr(base, name), getattr(copied, name)), name)
            self.assertTrue(bool(copied.observed.all()))
            self.assertEqual(base.digest(), copied.digest())

    def test_four_copies_equal_their_leader_and_keep_the_marginal_law(self):
        for periods in ((12,), (13,), (5, 7)):
            base, copied = self._batches(4, periods)
            self.assertTrue(torch.equal(base.targets, copied.targets))
            for b in range(3):
                p = int(copied.group[b])
                for t in range(100):
                    leader = (t % p) + p * ((t // p) - (t // p) % 4)
                    self.assertEqual(int(copied.tokens[b, t]), int(copied.tokens[b, leader]))
                    self.assertEqual(int(copied.tokens[b, t]), int(base.tokens[b, leader]))  # the k = 1 token there
                    self.assertEqual(bool(copied.observed[b, t]), leader == t)
                    self.assertEqual(int(copied.targets[b, t]), int(base.targets[b, t]))
            # leaders keep the independent draw, so the fraction of correct leader tokens is the noise-law rate
            self.assertTrue(torch.equal(copied.tokens[copied.observed], base.tokens[copied.observed]))

    def test_registration_and_the_independent_law_is_unchanged(self):
        task = make_task("periodic_copied", periods=[12], copies=4)
        self.assertEqual(task_dict(task), {"kind": "periodic_copied", "periods": (12,), "vocab_size": 16,
                                           "noise": 0.5, "copies": 4})
        self.assertEqual(task_dict(PeriodicDenoising()), {"kind": "periodic", "periods": (12,), "vocab_size": 16,
                                                           "noise": 0.5, "query_rate": 0.0})
        with self.assertRaises(ValueError):
            CopiedPeriodicDenoising(copies=0)


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class BayesNllTests(unittest.TestCase):
    def test_matches_a_brute_force_product_likelihood_posterior(self):
        gen = lambda: torch.Generator().manual_seed(5)  # noqa: E731
        for task in (PeriodicDenoising(periods=(5,)), PeriodicDenoising(periods=(4, 6)),
                     CopiedPeriodicDenoising(periods=(5,), copies=3), CopiedPeriodicDenoising(periods=(4, 6), copies=2)):
            batch = task.sample(3, 37, gen())
            got, ref = periodic_bayes_nll(batch, task), _brute_force_nll(batch, task)
            self.assertTrue(torch.allclose(got, ref, atol=1e-9, rtol=0), task)

    def test_independent_law_is_the_oracle_log_odds_form(self):
        task = PeriodicDenoising(periods=(12,))
        batch = task.sample(2, 60, torch.Generator().manual_seed(2))
        # one vote for the truth at t = 0: logsumexp([log rho, 0 x 15]) - log rho
        expected = math.log(17 + 15) - LOG17 if int(batch.tokens[0, 0]) == int(batch.targets[0, 0]) else math.log(17 + 15)
        self.assertAlmostEqual(float(periodic_bayes_nll(batch, task)[0, 0]), expected, places=9)

    def test_copied_law_counts_leaders_only(self):
        task = CopiedPeriodicDenoising(periods=(12,), copies=4)
        batch = task.sample(1, 4 * 12, torch.Generator().manual_seed(3))  # one block per residue: one vote everywhere
        independent = PeriodicDenoising(periods=(12,))
        nll = periodic_bayes_nll(batch, task)
        batch.observed = None  # the same tokens, wrongly treated as independent: up to four votes
        self.assertFalse(torch.allclose(nll, periodic_bayes_nll(batch, independent)))
        self.assertTrue(torch.allclose(nll, nll[:, :12].repeat(1, 4), atol=1e-12))  # no new evidence after visit 0


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class SuiteTests(unittest.TestCase):
    def test_arms_runs_and_settings(self):
        specs = SUITES["e10_evidence"]()
        self.assertEqual(len(specs), 2 * len(E10_ARMS) * len(E10_SEEDS))
        self.assertEqual(E10_FIRST_SEED, 42)
        self.assertEqual(E10_SEEDS[0], 42)
        self.assertEqual(E10_DATA_SEED, 20261020)
        self.assertEqual({e10_arm(s) for s in specs}, set(E10_ARMS))
        self.assertTrue(all(s.data_seed == 20261020 and s.eval_lengths == (64, 256, 1024, 8192)
                            and s.eval_tokens == 65536 == s.eval_tokens_long and s.steps == 2000 and s.train_length == 64
                            and s.task["kind"] == "periodic" and s.evidence == E10_ARMS[e10_arm(s)].get("evidence")
                            for s in specs))
        self.assertEqual({s.experiment for s in specs}, {"e10_p12", "e10_p13"})
        spread = SUITES["pilot_e10_spread"]()
        self.assertEqual(len(spread), 2 * len(E10_ARMS) * 4)
        self.assertEqual(sorted({s.seed for s in spread}), [1000, 1001, 1002, 1003])
        self.assertEqual(E10_SPREAD_SEEDS, (1000, 1001, 1002, 1003))
        self.assertTrue(all(s.data_seed == 7 for s in spread))
        self.assertEqual({s.experiment for s in spread}, {"pilot_e10_p12", "pilot_e10_p13"})

    def test_pairing_and_no_collisions(self):
        specs = SUITES["e10_evidence"]()
        for period in (12, 13):
            for seed in E10_SEEDS:
                cell = [s for s in specs if s.task["periods"] == [period] and s.seed == seed]
                self.assertEqual(len({s.experiment for s in cell}), 1)  # one experiment name: shared init and batches
        for name in ("e10_evidence", "pilot_e10_spread"):
            mine = {s.filename for s in SUITES[name]()}
            self.assertEqual(len(mine), len(SUITES[name]()))
            others = {s.filename for other, fn in SUITES.items() if other != name for s in fn()}
            self.assertFalse(mine & others)
            self.assertFalse({s.experiment for s in SUITES[name]()} &
                             {s.experiment for other, fn in SUITES.items() if other != name for s in fn()})

    def test_hyperparameters_are_the_periodic_ones(self):
        for s in SUITES["e10_evidence"]():
            self.assertEqual((s.lr, s.pos_lr_mult), (3e-3, 10.0))


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class CopyEvaluationTests(unittest.TestCase):
    def test_records_are_complete_resumable_and_use_the_copied_law(self):
        import hashlib

        spec = RunSpec(experiment="t10", task={"kind": "periodic", "periods": [5]}, variant="arith", seed=0,
                       d_model=16, num_layers=1, num_heads=2, d_ff=32, train_length=16, batch_size=4, steps=3, warmup=1,
                       val_every=3, eval_lengths=(16,), eval_tokens=64, eval_tokens_long=64, data_seed=9,
                       evidence={"kernel": "arith"})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run(spec, tmp / "runs", checkpoint_dir=tmp / "ckpt")
            ckpt = tmp / "ckpt" / (spec.filename[:-5] + ".pt")
            written = e10_copy_eval.evaluate_checkpoint(ckpt, (16, 32), tmp / "eval", copies=2)
            self.assertEqual([p.name for p in written], [f"{ckpt.stem}__N16.json", f"{ckpt.stem}__N32.json"])
            self.assertEqual(e10_copy_eval.evaluate_checkpoint(ckpt, (16, 32), tmp / "eval", copies=2), [])
            rec = json.loads(written[0].read_text())
            self.assertEqual(rec["checkpoint_sha256"], hashlib.sha256(ckpt.read_bytes()).hexdigest())
            self.assertEqual(rec["copied_task"], {"kind": "periodic_copied", "periods": [5], "vocab_size": 16,
                                                  "noise": 0.5, "copies": 2})
            self.assertEqual(rec["source_run"], spec.filename)
            self.assertEqual(rec["spec"]["evidence"], {"kernel": "arith"})
            self.assertIn("git_commit", rec["environment"])
            self.assertEqual(rec["test"][0]["length"], 16)
            task = CopiedPeriodicDenoising(periods=(5,), copies=2)
            parts = ("test", 9, json.dumps(task_dict(task), sort_keys=True), 16)
            from arithscope.train import _generator

            batch = task.sample(4, 16, _generator(*parts))  # 64 // 16 sequences
            self.assertEqual(rec["test"][0]["digest"], batch.digest())
            self.assertAlmostEqual(rec["bayes_nll"]["all"], float(periodic_bayes_nll(batch, task).mean()), places=12)


def _rand(rng):
    return rng.gauss(0, 1e-4)


def _metrics(mods=None, seeds=None):
    """Synthetic per-run metrics for which every claim and secondary prediction holds; ``mods`` overrides cells."""
    seeds = E10_SEEDS if seeds is None else seeds
    rng = random.Random(0)
    base = {"arith": {"far_nll": 0.06, "far_acc": 1.0, "excess_1024": 0.10, "indep_nll_256": 0.20, "copy_nll_256": 0.50},
            "arith_ev": {"far_nll": 0.0005, "far_acc": 1.0, "excess_1024": 0.001, "indep_nll_256": 0.15,
                         "copy_nll_256": 0.60, "cos_dp": 0.995, "lambda_eff": LOG17 * 1.02},
            "arith_evfund": {"far_nll": 0.04, "far_acc": 0.80, "excess_1024": 0.02, "indep_nll_256": 0.17,
                             "copy_nll_256": 0.60},
            "rope": {"far_nll": 0.50, "far_acc": 0.50, "excess_1024": 0.30, "indep_nll_256": 0.30, "copy_nll_256": 0.70},
            "rope_ev": {"far_nll": 0.001, "far_acc": 1.0, "excess_1024": 0.002, "indep_nll_256": 0.16,
                        "copy_nll_256": 0.80, "cos_dp": 0.995, "lambda_eff": LOG17 * 0.98}}
    m = {}
    for p in (12, 13):
        for arm, cell in base.items():
            for s in seeds:
                m[(p, arm, s)] = {k: v + (_rand(rng) if k not in ("cos_dp", "lambda_eff") else 0.0) for k, v in cell.items()}
        for s in seeds:  # at P = 12 the matched cosines are as accurate as the exact basis (S12)
            if p == 12:
                m[(p, "arith_evfund", s)]["far_acc"] = 1.0 + _rand(rng)
    for (p, arm, key), value in (mods or {}).items():
        for s in seeds:
            if p is None:
                for q in (12, 13):
                    m[(q, arm, s)][key] = value
            else:
                m[(p, arm, s)][key] = value
    return m


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class VerdictTests(unittest.TestCase):
    def verdicts(self, mods=None):
        return e10_verdicts.decide(_metrics(mods))["verdicts"]

    def test_everything_supported(self):
        v = self.verdicts()
        for name in ("V1_head_lowers_far_nll", "V2_follows_bayes_at_1024", "V3_exact_basis_matters_at_13",
                     "V4_restores_rope_extrapolation", "B1_copied_observations_harm"):
            self.assertEqual(v[name], "SUPPORTED", name)
        for name in ("S12_no_failure_at_12", "S13_sgd_finds_construction", "S14_bayes_scale", "S15_crossover"):
            self.assertTrue(v[name], name)

    def test_each_claim_can_be_falsified_and_each_guard_fails(self):
        cases = [
            ("V1_head_lowers_far_nll", {(13, "arith_ev", "far_nll"): 0.06}, "far NLL"),
            ("V1_head_lowers_far_nll", {(12, "arith_ev", "far_acc"): 0.95}, "accuracy guard"),
            ("V2_follows_bayes_at_1024", {(13, "arith_ev", "excess_1024"): 0.01}, "excess"),
            ("V3_exact_basis_matters_at_13", {(13, "arith_evfund", "far_nll"): 0.0001}, "far NLL"),
            ("V3_exact_basis_matters_at_13", {(13, "arith_evfund", "far_acc"): 0.95}, "+10 pp"),
            ("V4_restores_rope_extrapolation", {(12, "rope_ev", "far_acc"): 0.65}, "+20 pp"),
            ("V4_restores_rope_extrapolation", {(13, "rope_ev", "far_nll"): 0.6}, "far NLL"),
            ("B1_copied_observations_harm", {(12, "arith_ev", "copy_nll_256"): 0.45}, "copy NLL"),
        ]
        for claim, mods, reason in cases:
            mods = {(p, arm, key): val for (p, arm, key), val in mods.items()}
            verdict = self.verdicts(mods)[claim]
            self.assertTrue(verdict.startswith("FALSIFIED"), (claim, verdict))
            self.assertIn(reason, verdict)

    def test_claims_are_independent(self):
        v = self.verdicts({(13, "arith_ev", "excess_1024"): 0.01})
        self.assertEqual(v["V1_head_lowers_far_nll"], "SUPPORTED")
        self.assertTrue(v["V2_follows_bayes_at_1024"].startswith("FALSIFIED"))

    def test_v3_looks_only_at_p13_and_s12_only_at_p12(self):
        v = self.verdicts({(12, "arith_evfund", "far_acc"): 0.5, (12, "arith_evfund", "far_nll"): 0.0001})
        self.assertEqual(v["V3_exact_basis_matters_at_13"], "SUPPORTED")
        self.assertFalse(v["S12_no_failure_at_12"])
        v = self.verdicts({(13, "arith_evfund", "far_acc"): 1.0})
        self.assertTrue(v["V3_exact_basis_matters_at_13"].startswith("FALSIFIED"))
        self.assertTrue(v["S12_no_failure_at_12"])

    def test_secondary_predictions(self):
        self.assertFalse(self.verdicts({(12, "rope_ev", "cos_dp"): 0.9})["S13_sgd_finds_construction"])
        self.assertFalse(self.verdicts({(13, "arith_ev", "lambda_eff"): LOG17 * 1.3})["S14_bayes_scale"])
        self.assertFalse(self.verdicts({(13, "arith_ev", "lambda_eff"): LOG17 * 0.85})["S14_bayes_scale"])
        self.assertFalse(self.verdicts({(12, "arith_ev", "indep_nll_256"): 0.25})["S15_crossover"])
        self.assertEqual(self.verdicts({(12, "rope_ev", "cos_dp"): 0.9})["V1_head_lowers_far_nll"], "SUPPORTED")

    def test_a_dropped_seed_is_dropped_only_from_comparisons_that_use_it(self):
        m = _metrics()
        for key in [k for k in m if k[1] == "rope_ev" and k[2] == E10_SEEDS[0] and k[0] == 12]:
            del m[key]
        out = e10_verdicts.decide(m)
        self.assertEqual(len(out["periods"]["12"]["v4_far_acc"]["seeds"]), len(E10_SEEDS) - 1)
        self.assertEqual(len(out["periods"]["12"]["v1_far_nll"]["seeds"]), len(E10_SEEDS))
        self.assertEqual(len(out["periods"]["13"]["v4_far_acc"]["seeds"]), len(E10_SEEDS))

    def test_holm_covers_the_interval_tests(self):
        out = e10_verdicts.decide(_metrics())
        self.assertEqual(len(out["holm"]), 14)
        self.assertTrue(all(0 <= v["p_holm"] <= 1 for v in out["holm"].values()))


def _loaded(commit="c", digest="d", sha="a", seeds=(1, 2)):
    runs, copies, expected = {}, {}, []
    for p in (12, 13):
        for arm in ("arith", "arith_ev"):
            for s in seeds:
                key = (p, arm, s)
                expected.append(key)
                runs[key] = {"status": "COMPLETED", "environment": {"git_commit": commit},
                             "test": [{"length": 256, "digest": f"{digest}{p}"}]}
                for n in (256, 1024):
                    copies[(*key, n)] = {"source_run": f"{p}_{arm}_{s}.json", "checkpoint_sha256": sha,
                                         "environment": {"git_commit": commit}, "test": [{"digest": f"c{p}{n}"}]}
    files = {f"{p}_{arm}_{s}.json": sha for p in (12, 13) for arm in ("arith", "arith_ev") for s in seeds}
    return {"runs": runs, "copies": copies, "failed": [], "expected": expected}, files


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class SanityTests(unittest.TestCase):
    def ok(self, loaded, files):
        return {k: v for k, v in e10_verdicts.sanity(loaded, files).items() if k != "code_commits"}

    def test_good_records_pass(self):
        self.assertTrue(all(self.ok(*_loaded()).values()))

    def test_mixed_commits_digests_missing_and_mismatched_checkpoints_are_detected(self):
        loaded, files = _loaded()
        loaded["runs"][(12, "arith", 1)]["environment"]["git_commit"] = "other"
        self.assertFalse(self.ok(loaded, files)["single_code_commit"])
        loaded, files = _loaded()
        loaded["copies"][(13, "arith_ev", 2, 256)]["environment"]["git_commit"] = "other"
        self.assertFalse(self.ok(loaded, files)["single_code_commit"])
        loaded, files = _loaded()
        loaded["runs"][(12, "arith", 1)]["test"][0]["digest"] = "different"
        self.assertFalse(self.ok(loaded, files)["one_test_set_per_period_and_length"])
        loaded, files = _loaded()
        loaded["copies"][(12, "arith", 1, 1024)]["test"][0]["digest"] = "different"
        self.assertFalse(self.ok(loaded, files)["one_test_set_per_period_and_length"])
        loaded, files = _loaded()
        del loaded["runs"][(13, "arith_ev", 2)]
        self.assertFalse(self.ok(loaded, files)["all_records_accounted_for"])
        loaded["failed"].append((13, "arith_ev", 2, "training failed"))  # reported, not missing
        self.assertTrue(self.ok(loaded, files)["all_records_accounted_for"])
        loaded, files = _loaded()
        files["12_arith_1.json"] = "other-sha"
        self.assertFalse(self.ok(loaded, files)["copy_checkpoints_match_suite_runs"])
        loaded, files = _loaded()
        del files["12_arith_1.json"]
        self.assertFalse(self.ok(loaded, files)["copy_checkpoints_match_suite_runs"])
        loaded, files = _loaded()
        loaded["runs"][(12, "arith", 1)]["status"] = "FAILED"
        self.assertFalse(self.ok(loaded, files)["training_runs_completed"])


def _spread(mods=None):
    return _metrics(mods, E10_SPREAD_SEEDS)


def _far_alone(mean=0.06, sd=0.03, n=36, seed=0):
    rng = random.Random(seed)
    return {p: [mean + rng.gauss(0, sd) for _ in range(n)] for p in (12, 13)}


@unittest.skipUnless(HAS_TORCH, "torch/scipy not installed")
class SeedCountRuleTests(unittest.TestCase):
    def test_power_is_monotone_and_the_smallest_n_is_minimal(self):
        powers = [e10_power.power(0.9, n) for n in range(5, 30)]
        self.assertEqual(powers, sorted(powers))
        ds = [1.2, 1.5, 1.1]
        n = e10_power.smallest_n(ds)
        self.assertGreaterEqual(e10_power.joint_power(ds, n), 0.90)
        if n > e10_power.N_MIN:
            self.assertLess(e10_power.joint_power(ds, n - 1), 0.90)
        self.assertGreaterEqual(n, 5)
        self.assertGreater(e10_power.joint_power([1.0], 12), e10_power.joint_power([1.0, 1.0], 12))
        self.assertIsNone(e10_power.smallest_n([0.1, 0.1]))

    def test_plan_takes_the_max_over_claims(self):
        out = e10_power.plan(_far_alone(), _spread())
        self.assertEqual(out["stop_reasons"], [])
        ns = [c["n_needed"] for c in out["claims"].values()]
        self.assertEqual(out["n_seeds"], max(ns))
        self.assertGreaterEqual(out["n_seeds"], 5)
        for entry in out["claims"].values():
            self.assertGreaterEqual(entry["joint_power_at_n"], 0.90 - 1e-12 if entry["n_needed"] == out["n_seeds"] else 0.0)
        self.assertEqual(set(out["claims"]), {"V1", "V2", "V3", "V4", "B1"})
        self.assertEqual([len(c["parts"]) for c in out["claims"].values()], [2, 2, 2, 4, 2])
        self.assertIn("point_sd", out["sensitivity"])

    def test_stop_conditions(self):
        far = _far_alone()
        cases = {
            "V1 assumption": {(12, "arith_ev", "far_nll"): 0.01},
            "V2 mean": {(13, "arith_ev", "excess_1024"): 0.006},
            "V3 nll": {(13, "arith_evfund", "far_nll"): 0.0},
            "V3 acc": {(13, "arith_evfund", "far_acc"): 0.95},
            "V4 acc": {(12, "rope_ev", "far_acc"): 0.65},
            "V4 nll": {(13, "rope_ev", "far_nll"): 0.6},
            "B1": {(12, "arith_ev", "copy_nll_256"): 0.4},
        }
        for label, mods in cases.items():
            out = e10_power.plan(far, _spread(mods))
            self.assertIsNone(out["n_seeds"], label)
            self.assertTrue(out["stop_reasons"], label)
            self.assertIn("stop", out)

    def test_a_claim_needing_more_than_twenty_seeds_sets_no_count(self):
        m = _spread()
        rng = random.Random(4)
        for p in (12, 13):
            for s in E10_SPREAD_SEEDS:
                m[(p, "arith_ev", s)]["copy_nll_256"] = m[(p, "arith", s)]["copy_nll_256"] + 0.01 + rng.gauss(0, 0.5)
        out = e10_power.plan(_far_alone(), m)
        self.assertIsNone(out["n_seeds"])
        self.assertIsNone(out["claims"]["B1"]["n_needed"])
        self.assertTrue(any("more than 20" in r for r in out["stop_reasons"]))

    def test_zero_spread_means_unit_power(self):
        pt = e10_power.part("x", 12, [0.3, 0.3, 0.3, 0.3], 1, 0.2)
        self.assertTrue(math.isinf(pt["d"]))
        self.assertEqual(e10_power.smallest_n([pt["d"]]), 5)

    def test_inputs_are_read_from_the_e8_and_e9_summaries(self):
        root = Path(__file__).resolve().parent.parent / "results" / "summary"
        values = e10_power.far_alone_from_summaries(root / "e8.json", root / "e9.json")
        self.assertEqual({p: len(v) for p, v in values.items()}, {12: 36, 13: 36})  # 21 (E8) + 15 (E9) seeds


if __name__ == "__main__":
    unittest.main()
