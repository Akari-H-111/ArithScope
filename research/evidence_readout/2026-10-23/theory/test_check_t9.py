"""Unit tests for the exact items of T9 (standard library only; total run time well under 60 s).

    /usr/bin/python3 -B -m unittest -v test_check_t9.py

The heavy sweeps of check_t9.py are run with small parameters here; the full run is `check_t9.py --json`.
"""
import math
import random
import unittest
from fractions import Fraction

import check_t9 as c


class TestConstruction(unittest.TestCase):
    """(a)"""

    def test_dp_identity_small(self):
        r = c.check_a_identity(pmax=14)
        self.assertTrue(r["ok"], r["failures"])
        self.assertTrue(r["weights_sum_to_one"])

    def test_dp_period_one_is_constant(self):
        # P = 1: D_1 = K_1 = 1, every offset is on the comb
        self.assertTrue(all(c.d_p_value(1, n) == 1 for n in range(-5, 6)))

    def test_bayes_equality_exact(self):
        r = c.check_a_bayes_equality(cases=((3, Fraction(1, 2), 4), (4, Fraction(1, 3), 3)))
        self.assertTrue(r["ok"])
        for case in r["cases"]:
            self.assertEqual(case["exact_mismatches"], 0)

    def test_head_equals_comb_counts(self):
        r = c.check_a_head_comb_counts(Ps=(7, 12), length=40)
        self.assertTrue(r["ok"], r["failures"])

    def test_noise_parameters_v16(self):
        p, nu, rho = c.noise_params(16, Fraction(1, 2))
        self.assertEqual((p, nu, rho), (Fraction(17, 32), Fraction(1, 32), Fraction(17)))
        # chi = 1 - (sqrt p - sqrt nu)^2 = (14 + 2 sqrt 17)/32
        self.assertAlmostEqual(c.chi_value(16, p, nu), (14 + 2 * math.sqrt(17)) / 32, places=12)

    def test_rate_bound_with_bounded_offset(self):
        r = c.check_a_rate_bound(cases=((3, Fraction(1, 2), 5),), B_values=(0.0, 2.0))
        self.assertTrue(r["ok"])
        case = r["cases"][0]
        # a.s. slope per vote (p - nu) log rho for V = 3, eps = 1/2 is log 2
        self.assertAlmostEqual(case["as_slope_per_vote_(p-nu)log(rho)"], math.log(2), places=12)


class TestExpressibility(unittest.TestCase):
    """(b)"""

    def test_span_iff_divisors_in_bank(self):
        default = c.BANKS["default {1..16}"]
        for P, expected in ((12, True), (13, True), (16, True), (17, False), (24, False)):
            r = c.span_test_ramanujan(default, P)
            self.assertEqual(r["D_P_in_span"], expected, P)
            self.assertEqual(r["D_P_in_span"], r["all_divisors_of_P_in_bank"], P)
            self.assertTrue(r["bank_linearly_independent"])

    def test_b16_with_order_one_misses_p12(self):
        bank = c.BANKS["B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}"]
        self.assertFalse(c.span_test_ramanujan(bank, 12)["D_P_in_span"])
        self.assertTrue(c.span_test_ramanujan(bank, 13)["D_P_in_span"])
        # exact counting periods of B16 + {1}: prime powers <= 16 (a divisor-closed subset of the bank)
        counted = [P for P in range(2, 19) if c.span_test_ramanujan(bank, P)["D_P_in_span"]]
        self.assertEqual(counted, [2, 3, 4, 5, 7, 8, 9, 11, 13, 16])
        no_one = c.BANKS["B16 as in MATH.md, order 1 excluded (10 orders)"]
        self.assertFalse(c.span_test_ramanujan(no_one, 13)["D_P_in_span"])

    def test_divisor_closed_bank_sweep(self):
        bank = [1, 2, 3, 4, 6, 12]
        for P in range(2, 25):
            r = c.span_test_ramanujan(bank, P)
            self.assertTrue(r["iff_holds"], P)

    def test_bank_design_crt_comb_is_not_counting(self):
        r = c.check_b_bank_design()
        self.assertTrue(r["ok"])
        self.assertEqual(r["divisors_of_P_missing_from_bank"], [6, 12])
        self.assertEqual(r["sum_abs_offcomb"], 4)
        self.assertEqual(r["worst_motif_slope_V3 (query residue 0, W=I)"], Fraction(-1, 8))

    def test_cosine_span(self):
        r = c.check_b_cosine(pmax=14, big_bank=16)
        self.assertTrue(r["ok"])
        self.assertEqual(r["P=12"]["missing_harmonics"], [5])
        self.assertFalse(r["P=12"]["D_12_in_span_matched_cosines"])
        self.assertTrue(r["P=12"]["D_12_in_span_of_all_harmonics_0..6"])
        self.assertEqual(r["criterion_sweep"]["P_values_in_span"], [2, 3, 4, 6])

    def test_harmonic_criterion(self):
        r = c.check_b_harmonic_criterion(pmax=24)
        self.assertTrue(r["ok"], r["failures"])
        # phi(P) <= 2 characterizes the periods whose combs are cosine-expressible
        phi_small = [P for P in range(1, 40) if c.arith.euler_phi(P) <= 2]
        self.assertEqual(phi_small, [1, 2, 3, 4, 6])

    def test_residual_counterexample(self):
        r = c.check_b_residual_counterexample()
        self.assertTrue(r["ok"])
        self.assertEqual([x["slope_without_log_rho (W=I)"] for x in r["cases"]],
                         [Fraction(-1, 12), Fraction(-1, 130), 0, Fraction(1, 338), Fraction(1, 26)])
        # threshold delta > 1/(P-1)
        for P in (3, 5, 13):
            for delta in (Fraction(1, 2 * (P - 1)), Fraction(1, P - 1), Fraction(2, P - 1)):
                p, nu, _ = c.noise_params(16, Fraction(1, 2))
                kP = [Fraction(1)] + [delta] * (P - 1)
                pit = c.pi_table([0] + [1] * (P - 1), 16, p, nu)
                sl = c.slope(kP, c.identity(16), pit, 0, 0, 1)
                self.assertEqual(sl < 0, delta > Fraction(1, P - 1))


class TestPositiveRate(unittest.TestCase):
    """(c)"""

    def test_fold_projection(self):
        r = c.check_c_fold_projection(pmax=12, mmax=12)
        self.assertTrue(r["ok"], r["failures"])

    def test_fold_kills_orders_not_dividing_p(self):
        # m = 4 does not divide P = 2 and is not coprime to it; it still folds to zero
        self.assertEqual(c.fold(lambda n: c.arith.kernel(4, n), 4, 2), [Fraction(0), Fraction(0)])
        # m = 2 divides P = 6: folded kernel is K_2 itself
        self.assertEqual(c.fold(lambda n: c.arith.kernel(2, n), 2, 6), [c.arith.kernel(2, s) for s in range(6)])

    def test_expected_count_decomposition(self):
        r = c.check_c_decomposition()
        self.assertTrue(r["ok"], r["failures"])

    def test_dominance_criterion(self):
        r = c.check_c_dominance(P_list=(3,), V=3, trials=30)
        self.assertTrue(r["ok"], r["mismatches"])

    def test_two_order_family(self):
        r = c.check_c_two_order_family(cases_PV=((5, 3),))
        self.assertTrue(r["ok"], r["mismatches"])

    def test_comb_slope_is_motif_independent(self):
        V, P = 4, 5
        p, nu, _ = c.noise_params(V, Fraction(1, 2))
        delta = [Fraction(1)] + [Fraction(0)] * (P - 1)
        rng = random.Random(1)
        for _ in range(10):
            motif = [rng.randrange(V) for _ in range(P)]
            pit = c.pi_table(motif, V, p, nu)
            for t in range(P):
                for b in range(V):
                    if b != motif[t]:
                        self.assertEqual(c.slope(delta, c.identity(V), pit, t, motif[t], b), (p - nu) / P)


class TestNonResonant(unittest.TestCase):
    """(d): algebra only; the limits are theorems, the table in check_t9.py is NUMERICAL."""

    def test_residue_blind_algebra(self):
        r = c.check_d_algebra()
        self.assertTrue(r["ok"], r["failures"])

    def test_drift_kernel_is_exactly_period_1000(self):
        r = c.check_d_numeric(kmin=8, kmax=10)
        self.assertLess(r["drift_fold_at_13_max_abs_dev_from_1/2"], 1e-12)
        # theta = 2 pi/13 * 1.001 = 2 pi * 77/1000
        self.assertEqual(Fraction(1001, 1000) / 13, Fraction(77, 1000))

    def test_limit_values_for_comb_kernel(self):
        r = c.check_d_numeric(kmin=8, kmax=9)
        row = [x for x in r["rows"] if x["kernel"].startswith("D_13")][0]
        p, nu, _ = c.noise_params(16, Fraction(1, 2))
        self.assertAlmostEqual(row["limit_true"], float(p) / 13, places=12)
        self.assertAlmostEqual(row["limit_other"], float(nu) / 13, places=12)


class TestDependence(unittest.TestCase):
    """(e)"""

    def test_headline_numbers(self):
        r = c.check_e_dependence(cases=((3, Fraction(1, 2), 3), (16, Fraction(1, 2), 2)))
        self.assertTrue(r["ok"])
        h = r["headline_V3_eps1/2_k2_m1"]
        self.assertEqual(h["E_prob_true_head"], Fraction(11, 18))
        self.assertEqual(h["E_prob_true_bayes"], Fraction(1, 2))
        self.assertEqual(h["accuracy_of_argmax (both heads)"], Fraction(2, 3))
        self.assertEqual(h["E_max_prob_head"], Fraction(8, 9))
        self.assertEqual(h["E_max_prob_bayes"], Fraction(2, 3))
        self.assertAlmostEqual(h["E_NLL_head"], 2 / 3 * math.log(9 / 8) + 1 / 3 * math.log(18), places=12)
        self.assertAlmostEqual(h["E_NLL_bayes"], 2 / 3 * math.log(3 / 2) + 1 / 3 * math.log(6), places=12)

    def test_excess_is_expected_kl_and_positive(self):
        r = c.dependence_case(3, Fraction(1, 2), 2, (1, 2, 4))
        self.assertTrue(r["ok"])
        for k in (2, 4):
            self.assertGreater(r["per_k"][k]["excess_NLL"], 0)
            self.assertLessEqual(r["per_k"][k]["E_NLL_head"], k * r["per_k"][1]["E_NLL_head"] + 1e-12)

    def test_power_inequality(self):
        # log(1 + sum x_b^k) <= k log(1 + sum x_b) for x_b >= 0, k >= 1
        rng = random.Random(7)
        for _ in range(200):
            xs = [math.exp(rng.uniform(-8, 8)) for _ in range(rng.randint(1, 6))]
            k = rng.randint(1, 6)
            self.assertLessEqual(math.log(1 + sum(x ** k for x in xs)), k * math.log(1 + sum(xs)) + 1e-12)


class TestRoundThree(unittest.TestCase):
    """Items added after the audit (L9.1', L9.2', T9b.2 span, T9c.3 constant, T9c.6, T9a.5, T9d window fold)."""

    def test_component_bound_and_old_bound(self):
        r = c.check_l91_component_bound(imax=120)
        self.assertTrue(r["ok"], r["failures"])
        q = r["quoted"]["D_P, P=12"]
        self.assertEqual(q["bound_per_component"], 24.0)  # 2 * sum phi(m)/12 * lcm(m,12) = 2 * 12
        self.assertEqual(r["quoted"]["beta_m=1, m=1..16, P=12"]["bound_2LambdaG"], 2 * 720720 * 16)

    def test_character_fold(self):
        r = c.check_fold_character(pmax=10, mmax=12)
        self.assertTrue(r["ok"], r["failures"])

    def test_hoeffding_constant(self):
        r = c.check_hoeffding_constant(imax=500)
        self.assertTrue(r["ok"])

    def test_no_dominant_kernel_in_crt_span(self):
        r = c.check_span_dominance()
        self.assertTrue(r["ok"])
        self.assertEqual(r["span_K1_K2_K3_K4"]["min_off_comb_mass_with_f(0)=1"], 2)
        self.assertEqual(r["mixtures_K3_K4"]["best_off_comb_mass"], 4)
        self.assertEqual(r["span_all_divisors_of_12"]["min_off_comb_mass"], 0)

    def test_random_motif_uniform_fold_closed_form(self):
        for V, P in ((2, 4), (3, 4), (4, 3)):
            self.assertEqual(c.prob_all_positive_enum([Fraction(1)] * P, V), Fraction(1, V ** (P - 1)))
            self.assertEqual(c.prob_all_positive_enum([Fraction(1)] + [Fraction(0)] * (P - 1), V), 1)

    def test_random_motif_partition_method_matches_enumeration(self):
        for V, P, delta in ((3, 5, Fraction(1, 3)), (4, 4, Fraction(1, 5)), (2, 6, Fraction(1, 2))):
            self.assertEqual(c.prob_all_positive_enum(c.leakage_fold(P, delta), V),
                             c.prob_all_positive_leakage(P, V, delta))

    def test_failure_bound_dominates_exact_failure(self):
        rng = random.Random(3)
        for V, P in ((3, 5), (2, 6)):
            for _ in range(3):
                kP = [Fraction(rng.randint(2, 9), 2)] + [Fraction(rng.randint(-3, 3), 2) for _ in range(P - 1)]
                fail = 1 - c.prob_all_positive_enum(kP, V)
                self.assertLessEqual(fail, min(Fraction(1), P * (V - 1) * c.pair_tail(kP, V)))
        # strict dominance <=> no failure and n_min is None
        kP = [Fraction(5), Fraction(1), Fraction(-2), Fraction(1)]
        self.assertIsNone(c.n_min(kP))
        self.assertEqual(c.prob_all_positive_enum(kP, 3), 1)

    def test_tightness_of_chi_exponent(self):
        r = c.check_a_tightness()
        self.assertTrue(r["ok"])

    def test_window_fold_closed_form(self):
        r = c.check_d_window_fold()
        self.assertTrue(r["ok"])
        # exactly periodic limit: eta -> 0 gives the periodic fold cos(2 pi a s/P)
        f = c.window_fold_closed(2, 1e-13, 12, 12 * 50)
        for s in range(12):
            self.assertAlmostEqual(f[s], math.cos(2 * math.pi * 2 * s / 12), places=6)

    def test_excess_not_monotone_in_m(self):
        r = c.check_e_dependence(cases=((3, Fraction(1, 2), 3),), ks=(1, 2))
        self.assertFalse(r["excess_strictly_decreasing_in_m"]["V=3,k=2"])


if __name__ == "__main__":
    unittest.main()
