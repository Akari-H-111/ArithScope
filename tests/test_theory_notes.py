"""Numerical regression notes for docs/MATH.md (standard library only)."""
import math
import random
import unittest


def bessel_i0(x: float, terms: int = 40) -> float:
    """Modified Bessel function I_0 by its power series sum_k ((x / 2)^k / k!)^2."""
    return sum(((x / 2) ** k / math.factorial(k)) ** 2 for k in range(terms))


def layer_norm(x, gain, bias, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [g * (v - mean) / math.sqrt(var + eps) + b for v, g, b in zip(x, gain, bias)]


class BoundedConfidenceTests(unittest.TestCase):
    """P3: LayerNorm + linear readout keeps the NLL above log(1 + sum_j exp(-G_cj)) (MATH.md, section 4)."""

    def test_nll_never_beats_the_p3_bound(self):
        rng = random.Random(20261015)
        d, vocab = 12, 5
        for _ in range(20):
            w = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(vocab)]
            gain = [rng.gauss(1, 0.5) for _ in range(d)]
            bias = [rng.gauss(0, 0.3) for _ in range(d)]
            for c in range(vocab):
                gaps, directions = [], []
                for j in range(vocab):
                    if j != c:
                        u = [(a - b) * g for a, b, g in zip(w[c], w[j], gain)]
                        u = [x - sum(u) / d for x in u]  # the projection Pi
                        gaps.append(math.sqrt(d) * math.sqrt(sum(x * x for x in u))
                                    + sum((a - b) * s for a, b, s in zip(w[c], w[j], bias)))
                        directions.append(u)
                bound = math.log1p(sum(math.exp(-g) for g in gaps))
                # random inputs, and inputs along each gap's maximizing direction (where the bound is nearly tight)
                inputs = [[rng.gauss(0, 1) * rng.uniform(0.1, 30) for _ in range(d)] for _ in range(30)]
                inputs += [[1e3 * x for x in u] for u in directions]
                for x in inputs:
                    h = layer_norm(x, gain, bias)
                    logits = [sum(a * b for a, b in zip(row, h)) for row in w]
                    nll = math.log(sum(math.exp(v - logits[c]) for v in logits))
                    self.assertGreaterEqual(nll, bound - 1e-12)


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
