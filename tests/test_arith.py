"""Exact checks of the arithmetic lemmas in docs/MATH.md (stdlib only, no floats unless noted)."""
import cmath
import unittest
from fractions import Fraction
from math import gcd

from arithscope import arith
from reference import arithmetic_reference as oracle


class ExactArithmeticTests(unittest.TestCase):
    def test_divisor_formula_matches_closed_form_and_handoff_oracle(self):
        for m in range(1, 120):
            for n in range(-150, 151):
                c = arith.ramanujan_sum(m, n)
                self.assertEqual(c, arith.ramanujan_sum_closed(m, n))
                self.assertEqual(c, oracle.ramanujan_sum(m, n))
            self.assertEqual(arith.euler_phi(m), oracle.euler_phi(m))
            self.assertEqual(arith.mobius(m), oracle.mobius(m))

    def test_kernel_closed_form_L1(self):
        for m in range(1, 150):
            for n in range(-60, 61):
                q = m // gcd(m, n)
                self.assertEqual(arith.kernel(m, n), Fraction(arith.mobius(q), arith.euler_phi(q)))
                self.assertEqual(arith.kernel(m, n), Fraction(arith.ramanujan_sum(m, n), arith.euler_phi(m)))

    def test_comb_gap_lemma_L2(self):
        for m in range(1, 400):
            table = arith.kernel_table(m)
            self.assertEqual(table[0], 1)
            self.assertTrue(all(v < 1 for v in table[1:]))
            gap = arith.comb_gap(m)
            self.assertGreaterEqual(gap, Fraction(1, 2))
            self.assertEqual(gap == Fraction(1, 2), m % 6 == 0)
            if m % 6 and m > 1:
                self.assertLessEqual(max(table[1:]), Fraction(1, 4))  # refinement stated in L2
            if arith.is_prime_power(m):
                self.assertTrue(all(v <= 0 for v in table[1:]))
                self.assertGreaterEqual(gap, 1)

    def test_optimal_positive_definite_comb_T3(self):
        for period in range(2, 80):
            weights = arith.optimal_comb_weights(period)
            self.assertEqual(sum(weights.values()), 1)
            self.assertTrue(all(w > 0 for w in weights.values()))
            values = [arith.mixture_value(weights, n) for n in range(period)]
            self.assertEqual(values[0], 1)
            self.assertTrue(all(v == Fraction(-1, period - 1) for v in values[1:]))
            self.assertEqual(arith.comb_margin(values, period), Fraction(period, period - 1))

    def test_prime_period_single_kernel_is_optimal(self):
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
            self.assertEqual(arith.comb_gap(p), Fraction(p, p - 1))

    def test_crt_comb_composition_T4(self):
        for period in range(2, 300):
            parts = arith.prime_power_components(period)
            self.assertEqual(arith.lcm(parts), period)
            k = len(parts)
            for n in range(0, 2 * period):
                b = sum(arith.kernel(q, n) for q in parts) / k
                if n % period == 0:
                    self.assertEqual(b, 1)
                else:
                    self.assertLessEqual(b, 1 - Fraction(1, k))

    def test_ramanujan_subspace_identities(self):
        for period in range(1, 90):
            for n in range(-period, 2 * period):
                total = sum(arith.ramanujan_sum(d, n) for d in arith.divisors(period))
                self.assertEqual(total, period if n % period == 0 else 0)
                indicator = Fraction(total, period)
                self.assertEqual(indicator, 1 if n % period == 0 else 0)

    def test_kernel_circulant_spectrum_is_nonnegative(self):
        # Eigenvalues of the m x m circulant of K_m: (m/phi(m)) [gcd(h, m) = 1] (floating check).
        for m in range(1, 40):
            table = arith.kernel_table(m)
            for h in range(m):
                lam = sum(complex(table[n]) * cmath.exp(-2j * cmath.pi * h * n / m) for n in range(m))
                expected = m / arith.euler_phi(m) if gcd(h, m) == 1 else 0.0
                self.assertAlmostEqual(lam.real, expected, places=9)
                self.assertAlmostEqual(lam.imag, 0.0, places=9)

    def test_alias_classes_divisor_closed_bank_are_gcd_classes_P1(self):
        for top in (12, 18, 24, 30):
            bank = arith.divisors(top)
            offsets = range(0, 3 * top)
            classes = arith.alias_classes(bank, offsets)
            by_gcd = {}
            for n in offsets:
                by_gcd.setdefault(gcd(n, top), []).append(n)
            self.assertEqual(sorted(map(tuple, classes)), sorted(map(tuple, by_gcd.values())))

    def test_alias_fixture_banks(self):
        # Target comb 24: the comb class must not contain 12 (mod 24) for a resolving bank.
        def comb_class(bank):
            for cls in arith.alias_classes(bank, range(0, 96)):
                if 0 in cls:
                    return set(cls)
        def comb_class12(bank):
            for cls in arith.alias_classes(bank, range(0, 48)):
                if 0 in cls:
                    return set(cls)
        for bank in ((2, 3), (1, 2, 3, 6)):                  # E2 aliased banks: lcm 6
            self.assertIn(6, comb_class12(bank))
        for bank in ((3, 4), (4, 6), (12,), (1, 2, 3, 4, 6, 12)):  # E2 resolving banks: lcm 12
            self.assertEqual(comb_class12(bank), set(range(0, 48, 12)))
        self.assertIn(12, comb_class((2, 3, 4, 6, 12)))   # lcm 12: aliased despite five orders
        self.assertIn(12, comb_class((4, 6)))             # lcm 12
        self.assertNotIn(12, comb_class((3, 8)))          # lcm 24 via CRT: resolved with two orders
        self.assertNotIn(12, comb_class((8, 12)))         # lcm 24
        self.assertEqual(comb_class((3, 8)), set(range(0, 96, 24)))

    def test_folded_rank_and_identifiability_modulus(self):
        self.assertEqual(arith.folded_rank(range(1, 17)), 80)
        self.assertEqual(arith.identifiability_modulus(120, [6, 10]), 60)
        self.assertEqual(arith.identifiability_modulus(120, [6, 10, 15]), 120)
        for L, H in ((120, [6, 10]), (120, [6, 10, 15]), (210, [6, 10, 15]), (77, [7, 14])):
            self.assertEqual(arith.identifiability_modulus(L, H), oracle.identifiable_modulus(L, H))

    def test_input_validation(self):
        for bad in (0, -3, 2.0, True):
            with self.assertRaises(ValueError):
                arith.euler_phi(bad)
        with self.assertRaises(ValueError):
            arith.kernel(5, 1.0)
        with self.assertRaises(ValueError):
            arith.optimal_comb_weights(1)
        with self.assertRaises(ValueError):
            arith.identifiability_modulus(12, [0, 4])


if __name__ == "__main__":
    unittest.main()
