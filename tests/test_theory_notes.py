"""Numerical regression notes for docs/MATH.md (standard library only)."""
import math
import unittest


def bessel_i0(x: float, terms: int = 40) -> float:
    """Modified Bessel function I_0 by its power series sum_k ((x / 2)^k / k!)^2."""
    return sum(((x / 2) ** k / math.factorial(k)) ** 2 for k in range(terms))


class LearnedFrequencyResonanceTests(unittest.TestCase):
    """T5 needs joint non-resonance; individual irrationality is not enough (MATH.md, section 4)."""

    def test_two_frequency_counterexample_keeps_residue_structure(self):
        theta1, theta2 = math.sqrt(2.0), math.sqrt(2.0) + math.pi  # theta2 - theta1 = 2 pi * (1/2)
        limit = bessel_i0(2.0) / (bessel_i0(2.0) + 1.0)
        self.assertAlmostEqual(limit, 0.6950834, places=6)
        even = odd = 0.0
        for n in range(200_000):
            weight = math.exp(math.cos(n * theta1) + math.cos(n * theta2))
            if n % 2:
                odd += weight
            else:
                even += weight
        mass = even / (even + odd)
        self.assertAlmostEqual(mass, limit, delta=1e-3)
        self.assertGreater(mass, 0.6)  # far from the uniform share 1/2 that T5 gives under its hypothesis


if __name__ == "__main__":
    unittest.main()
