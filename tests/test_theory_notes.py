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


class CountNonIdentifiabilityTests(unittest.TestCase):
    """T8a: a softmax average over classes is homogeneous of degree 0 in the class counts (MATH.md, section 4)."""

    def test_scaling_counts_leaves_the_average_unchanged(self):
        rng = random.Random(8)
        for _ in range(200):
            k = rng.randint(2, 6)
            scores = [rng.gauss(0, 2) for _ in range(k)]
            values = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(k)]
            counts = [rng.choice((0, rng.randint(1, 50))) for _ in range(k)]
            counts[rng.randrange(k)] = rng.randint(1, 50)  # at least one class present
            weights = [c * math.exp(s) for c, s in zip(counts, scores)]
            z = sum(weights)
            out = [sum(w * v[i] for w, v in zip(weights, values)) / z for i in range(3)]
            for lam in (2, 10, 1000):
                w2 = [lam * c * math.exp(s) for c, s in zip(counts, scores)]
                out2 = [sum(w * v[i] for w, v in zip(w2, values)) / sum(w2) for i in range(3)]
                for a, b in zip(out, out2):
                    self.assertAlmostEqual(a, b, places=12)
            # Euler: the Jacobian applied to the count vector, sum_a c_a e^{s_a} (v_a - F) / Z, vanishes
            for i in range(3):
                radial = sum(c * math.exp(s) * (v[i] - out[i]) for c, s, v in zip(counts, scores, values)) / z
                self.assertAlmostEqual(radial, 0.0, places=12)


def _compositions(n: int, parts: int):
    """All count vectors of `parts` non-negative integers summing to n."""
    if parts == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in _compositions(n - first, parts - 1):
            yield (first,) + rest


class BayesNllBoundTests(unittest.TestCase):
    """B1: exact expected Bayes NLL of periodic denoising against (V - 1) kappa^n (MATH.md, section 4)."""

    def _check(self, vocab: int, noise: float, max_votes: int):
        p, q = 1 - noise + noise / vocab, noise / vocab
        rho = p / q
        kappa = 1 - (math.sqrt(p) - math.sqrt(q)) ** 2
        self.assertAlmostEqual(kappa, 2 * math.sqrt(p * q) + (vocab - 2) * q, places=14)
        for n in range(max_votes + 1):
            expected = 0.0
            for counts in _compositions(n, vocab):  # the true symbol is 0
                prob = math.factorial(n) * p ** counts[0] * q ** (n - counts[0])
                for c in counts:
                    prob /= math.factorial(c)
                logw = [c * math.log(rho) for c in counts]
                top = max(logw)
                expected += prob * (top + math.log(sum(math.exp(w - top) for w in logw)) - logw[0])
            self.assertLessEqual(expected, (vocab - 1) * kappa ** n + 1e-12, (vocab, n))

    def test_bound_for_the_experiment_setting(self):
        self._check(vocab=16, noise=0.5, max_votes=5)

    def test_bound_for_small_vocabularies_and_more_votes(self):
        for vocab, noise in ((2, 0.5), (3, 0.8), (4, 0.5)):
            self._check(vocab, noise, max_votes=12)

    def test_posterior_is_the_normalized_likelihood(self):
        p, q, rho = 17 / 32, 1 / 32, 17.0
        for counts in ((3, 1, 0, 2), (0, 0, 5, 1), (7, 7, 0, 0)):
            n = sum(counts)
            like = [p ** c * q ** (n - c) for c in counts]
            closed = [rho ** c / sum(rho ** b for b in counts) for c in counts]
            for a, b in zip(like, closed):
                self.assertAlmostEqual(a / sum(like), b, places=14)
