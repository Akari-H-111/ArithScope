"""Exact checks for T9 (evidence accumulation in an unnormalized readout), docs/MATH.md section 4.

Standard library plus ``arithscope.arith``; no torch. Sources: the identities and criteria are those of
``research/evidence_readout/2026-10-23/theory/T9_DRAFT.md`` (this file does not import from there).
"""
import itertools
import math
import unittest
from fractions import Fraction

from arithscope.arith import divisors, euler_phi, kernel, units


def rank(rows):
    """Exact rank over Q of a list of dicts {column: Fraction} (Gaussian elimination)."""
    rows = [dict(r) for r in rows]
    rk = 0
    while rows:
        row = rows.pop()
        row = {c: v for c, v in row.items() if v != 0}
        if not row:
            continue
        pivot = min(row)
        rk += 1
        for other in rows:
            if pivot in other:
                f = other[pivot] / row[pivot]
                for c, v in row.items():
                    other[c] = other.get(c, Fraction(0)) - f * v
    return rk


def in_span(target, basis):
    return rank(basis + [target]) == rank(basis)


def ramanujan_fourier(m):
    """K_m as a vector of exponential coefficients, indexed by x in Q/Z (exponent e^{2 pi i x n})."""
    return {Fraction(a, m) % 1: Fraction(1, euler_phi(m)) for a in units(m)}


def comb_fourier(period):
    """D_P = (1/P) sum_{h mod P} e^{2 pi i h n / P}."""
    return {Fraction(h, period): Fraction(1, period) for h in range(period)}


def cosine_basis_vector(d):
    """cos(2 pi n / d) in the basis {cos(2 pi x n) : 0 <= x <= 1/2}; d = 1 is the constant."""
    return {Fraction(1, d) if d > 1 else Fraction(0): Fraction(1)}


def comb_cosine_vector(period):
    """D_P = (1/P)(1 + sum_{h=1}^{P-1} cos(2 pi h n / P)) in the same basis."""
    vec = {}
    for h in range(period):
        x = Fraction(min(h, period - h), period)
        vec[x] = vec.get(x, Fraction(0)) + Fraction(1, period)
    return vec


def is_comb_in_ramanujan_span(period, bank):
    return in_span(comb_fourier(period), [ramanujan_fourier(m) for m in bank])


def divisor_closed_contains(period, bank):
    return all(d in bank for d in divisors(period))


B16 = (2, 3, 4, 5, 7, 8, 9, 11, 13, 16)


class CombIdentityTests(unittest.TestCase):
    """T9a.1: D_P = sum_{m | P} (phi(m)/P) K_m is the indicator of P | n, with convex weights."""

    def test_identity_for_p_2_to_32(self):
        for period in range(2, 33):
            weights = {m: Fraction(euler_phi(m), period) for m in divisors(period)}
            self.assertEqual(sum(weights.values()), 1)
            for n in range(-3 * period, 3 * period + 1):
                value = sum(w * kernel(m, n) for m, w in weights.items())
                self.assertEqual(value, 1 if n % period == 0 else 0, (period, n))

    def test_fourier_vectors_reproduce_the_kernels(self):
        for m in range(1, 13):
            vec = ramanujan_fourier(m)
            for n in range(24):
                re = sum(float(c) * math.cos(2 * math.pi * float(x) * n) for x, c in vec.items())
                self.assertAlmostEqual(re, float(kernel(m, n)), places=12)


class BayesEqualityTests(unittest.TestCase):
    """T9a.2: softmax(log(rho) * counts) is the product-likelihood posterior, as exact fractions."""

    def check(self, vocab, eps, max_len):
        p = 1 - eps + eps / vocab
        nu = eps / vocab
        rho = p / nu
        self.assertEqual(p + (vocab - 1) * nu, 1)
        for length in range(1, max_len + 1):
            for seq in itertools.product(range(vocab), repeat=length):
                counts = [seq.count(a) for a in range(vocab)]
                lik = []
                for a in range(vocab):
                    prod = Fraction(1)
                    for x in seq:
                        prod *= p if x == a else nu
                    lik.append(prod)
                post = [v / sum(lik) for v in lik]
                soft = [rho**c / sum(rho**cc for cc in counts) for c in counts]
                self.assertEqual(post, soft, seq)

    def test_v3_eps_half(self):
        self.check(3, Fraction(1, 2), 5)

    def test_v4_eps_third(self):
        self.check(4, Fraction(1, 3), 3)

    def test_v16_constants(self):
        p, nu = 1 - Fraction(1, 2) + Fraction(1, 32), Fraction(1, 32)
        self.assertEqual((p, nu, p / nu, p - nu), (Fraction(17, 32), Fraction(1, 32), 17, Fraction(1, 2)))


class SpanCriterionTests(unittest.TestCase):
    """T9b.1 / T9b.2: D_P is in span{K_m : m in M} iff every divisor of P is in M."""

    def test_criterion_on_several_banks(self):
        banks = {
            "default 1..16": tuple(range(1, 17)),
            "B16 + 1": (1,) + B16,
            "B16": B16,
            "divisors of 12": (1, 2, 3, 4, 6, 12),
        }
        for name, bank in banks.items():
            basis = [ramanujan_fourier(m) for m in bank]
            self.assertEqual(rank(basis), len(bank), name)  # K_m independent
            for period in range(1, 41):
                self.assertEqual(
                    is_comb_in_ramanujan_span(period, bank),
                    divisor_closed_contains(period, bank),
                    (name, period),
                )

    def test_default_bank_reaches_exactly_p_up_to_16(self):
        bank = tuple(range(1, 17))
        reached = [P for P in range(1, 41) if is_comb_in_ramanujan_span(P, bank)]
        self.assertEqual(reached, list(range(1, 17)))

    def test_bank_design_cases(self):
        bank = tuple(range(1, 17))
        self.assertTrue(is_comb_in_ramanujan_span(12, bank))
        self.assertTrue(is_comb_in_ramanujan_span(13, bank))
        self.assertFalse(is_comb_in_ramanujan_span(17, bank))
        self.assertFalse(is_comb_in_ramanujan_span(24, bank))
        with_one = (1,) + B16
        self.assertFalse(is_comb_in_ramanujan_span(12, with_one))  # divisors 6 and 12 missing
        self.assertTrue(is_comb_in_ramanujan_span(13, with_one))  # D_13 = K_1/13 + 12 K_13/13
        self.assertFalse(is_comb_in_ramanujan_span(13, B16))  # no order 1
        prime_powers = [P for P in range(2, 41) if is_comb_in_ramanujan_span(P, with_one)]
        self.assertEqual(prime_powers, [2, 3, 4, 5, 7, 8, 9, 11, 13, 16])

    def test_span_by_evaluation_on_a_small_window(self):
        """Independent of the Fourier vectors: rank of K_m evaluated on Z/L, L = lcm(M, P)."""

        def evaluated(bank, period):
            length = math.lcm(period, *bank)
            return [{n: kernel(m, n) for n in range(length)} for m in bank], {
                n: Fraction(1) if n % period == 0 else Fraction(0) for n in range(length)
            }

        full, comb = evaluated((1, 2, 3, 4, 6, 12), 12)
        self.assertEqual(rank(full), 6)
        self.assertTrue(in_span(comb, full))
        short, comb = evaluated((1, 2, 3, 4), 12)
        self.assertEqual(rank(short), 4)
        self.assertFalse(in_span(comb, short))
        no_one, comb = evaluated((2, 13), 13)
        self.assertFalse(in_span(comb, no_one))

    def test_crt_choice_off_comb_values_at_p12(self):
        """T9b.2: b = (K_3 + K_4)/2 is 1 only on 12 | n, but its off-comb mass is 4; worst-motif slope -1/8."""
        values = [(kernel(3, s) + kernel(4, s)) / 2 for s in range(12)]
        self.assertEqual(
            values,
            [Fraction(x) for x in ("1", "-1/4", "-3/4", "1/2", "1/4", "-1/4", "0", "-1/4", "1/4", "1/2", "-3/4", "-1/4")],
        )
        self.assertEqual(sum(abs(v) for v in values[1:]), 4)
        p_minus_nu = Fraction(1, 2)  # V = 3, eps = 1/2
        motif = (0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0)
        truth, rival = motif[0], 1
        total = values[0]
        for s in range(1, 12):
            sym = motif[(0 - s) % 12]
            total += values[s] * ((sym == truth) - (sym == rival))
        self.assertEqual(p_minus_nu / 12 * total, Fraction(-1, 8))


class CosineCriterionTests(unittest.TestCase):
    """T9b.3: D_P is in the span of fundamental cosines (any orders) iff phi(P) <= 2."""

    def test_criterion_for_p_up_to_40(self):
        orders = range(1, 49)
        basis = [cosine_basis_vector(d) for d in orders]
        self.assertEqual(rank(basis), len(basis))
        for period in range(1, 41):
            self.assertEqual(in_span(comb_cosine_vector(period), basis), euler_phi(period) <= 2, period)
        self.assertEqual([P for P in range(1, 41) if euler_phi(P) <= 2], [1, 2, 3, 4, 6])

    def test_matched_bank_and_harmonics_at_p12(self):
        matched = [cosine_basis_vector(d) for d in (1, 2, 3, 4, 6, 12)]
        self.assertFalse(in_span(comb_cosine_vector(12), matched))  # harmonic 5 is missing
        harmonics = [{Fraction(a, 12): Fraction(1)} for a in range(7)]
        self.assertTrue(in_span(comb_cosine_vector(12), harmonics))
        self.assertTrue(all(in_span(h, matched) for h in harmonics if next(iter(h)) != Fraction(5, 12)))
        self.assertFalse(in_span({Fraction(5, 12): Fraction(1)}, matched))

    def test_prime_p_is_not_in_the_one_cosine_span(self):
        for period in (5, 7, 11, 13, 17, 19, 23, 29, 31):
            basis = [cosine_basis_vector(1), cosine_basis_vector(period)]
            self.assertFalse(in_span(comb_cosine_vector(period), basis), period)


class LeakageTests(unittest.TestCase):
    """T9b.4: kappa = (1 - delta) D_P + delta K_1 has a negative linear slope iff delta > 1/(P - 1)."""

    P_MINUS_NU = Fraction(1, 2)  # V = 16, eps = 1/2

    def slope(self, period, delta):
        """(l_i(alpha) - l_i(beta)) / (lambda i) along i = 0 mod P, motif alpha at residue 0, beta elsewhere."""
        kappa = [(1 - delta) * (1 if s == 0 else 0) + delta for s in range(period)]
        # e_0(alpha) - e_0(beta) = (1/P) sum_s kappa(s) (pi_{-s}(alpha) - pi_{-s}(beta))
        gap = sum(kappa[s] * (self.P_MINUS_NU if s == 0 else -self.P_MINUS_NU) for s in range(period))
        return gap / period

    def test_exact_values(self):
        self.assertEqual(self.slope(3, Fraction(3, 4)), Fraction(-1, 12))
        self.assertEqual(self.slope(13, Fraction(1, 10)), Fraction(-1, 130))
        self.assertEqual(self.slope(13, Fraction(1, 12)), 0)
        self.assertEqual(self.slope(13, Fraction(1, 13)), Fraction(1, 338))
        self.assertEqual(self.slope(13, 0), Fraction(1, 26))

    def test_sign_flips_at_one_over_p_minus_one(self):
        for period in (3, 5, 12, 13, 20):
            for num in range(0, 40):
                delta = Fraction(num, 40)
                s = self.slope(period, delta)
                self.assertEqual(s < 0, delta > Fraction(1, period - 1), (period, delta))
                self.assertEqual(s, self.P_MINUS_NU / period * (1 - (period - 1) * delta))

    def test_expected_gap_at_i_1300(self):
        period, delta, i = 13, Fraction(1, 10), 1300
        gap = Fraction(0)
        for n in range(i + 1):
            kap = (1 - delta) * (1 if n % period == 0 else 0) + delta
            gap += kap * (self.P_MINUS_NU if (i - n) % period == 0 else -self.P_MINUS_NU)
        self.assertEqual(gap, Fraction(-19, 2))


class DependenceTests(unittest.TestCase):
    """T9e: exact copies make the count head differ from the Bayes posterior; excess = E KL > 0."""

    def enumerate_losses(self, vocab, eps, m, k):
        p = 1 - eps + eps / vocab
        nu = eps / vocab
        rho = p / nu
        bayes = head = kl = 0.0
        for obs in itertools.product(range(vocab), repeat=m):
            prob = math.prod(float(p if y == 0 else nu) for y in obs)  # true symbol is 0 by symmetry
            counts = [obs.count(a) for a in range(vocab)]
            w = [float(rho) ** c for c in counts]
            post = [x / sum(w) for x in w]
            wk = [float(rho) ** (k * c) for c in counts]
            hd = [x / sum(wk) for x in wk]
            bayes += prob * -math.log(post[0])
            head += prob * -math.log(hd[0])
            kl += prob * sum(a * math.log(a / b) for a, b in zip(post, hd))
        return bayes, head, kl

    def test_headline_case(self):
        bayes, head, kl = self.enumerate_losses(3, Fraction(1, 2), 1, 2)
        self.assertAlmostEqual(bayes, 0.867563, delta=1e-6)
        self.assertAlmostEqual(head, 1.041979, delta=1e-6)
        self.assertAlmostEqual(head - bayes, kl, delta=1e-12)
        self.assertAlmostEqual(bayes, 2 / 3 * math.log(3 / 2) + 1 / 3 * math.log(6), delta=1e-12)
        self.assertAlmostEqual(head, 2 / 3 * math.log(9 / 8) + 1 / 3 * math.log(18), delta=1e-12)

    def test_more_rows_and_positive_excess(self):
        for m, k, bayes_ref, head_ref in ((2, 2, 0.695406, 0.827949), (3, 2, 0.559656, 0.698837), (1, 4, 0.867563, 1.856175)):
            bayes, head, kl = self.enumerate_losses(3, Fraction(1, 2), m, k)
            self.assertAlmostEqual(bayes, bayes_ref, delta=1e-6)
            self.assertAlmostEqual(head, head_ref, delta=1e-6)
            self.assertGreater(kl, 0)
        bayes, head, kl = self.enumerate_losses(3, Fraction(1, 2), 2, 1)  # k = 1: no copies, no excess
        self.assertAlmostEqual(head, bayes, delta=1e-12)
        self.assertAlmostEqual(kl, 0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
