"""Exact finite checks for the isolated MC information audit; stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path


def channel(symbol: int, truth: int, vocab: int, noise: Fraction) -> Fraction:
    return noise / vocab + (1 - noise if symbol == truth else 0)


def posterior(counts: tuple[int, ...], noise: Fraction = Fraction(1, 2)) -> tuple[Fraction, ...]:
    """Uniform-prior posterior for independent replacement-noise observations."""
    if len(counts) < 2 or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("counts must be nonnegative integers for at least two classes")
    if not 0 < noise < 1:
        raise ValueError("this audit assumes replacement noise strictly between zero and one")
    vocab = len(counts)
    ratio = (1 - noise + noise / vocab) / (noise / vocab)
    weights = tuple(ratio ** n for n in counts)
    total = sum(weights)
    return tuple(w / total for w in weights)


def likelihood_posterior(history: tuple[int, ...], vocab: int, noise: Fraction) -> tuple[Fraction, ...]:
    """Independent oracle: multiply the channel likelihood of each raw observation."""
    weights = tuple(math.prod(channel(x, y, vocab, noise) for x in history) for y in range(vocab))
    total = sum(weights)
    return tuple(Fraction(w, total) for w in weights)


def brier(prediction: tuple[Fraction, ...], truth: int) -> Fraction:
    return sum((p - int(c == truth)) ** 2 for c, p in enumerate(prediction))


def exact_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def enumerate_risks(vocab: int = 3, max_length: int = 6) -> list[dict]:
    noise = Fraction(1, 2)
    rows = []
    for length in range(max_length + 1):
        mass = Fraction(0)
        expected_brier = Fraction(0)
        nll_terms = []
        histories = 0
        for history in itertools.product(range(vocab), repeat=length):
            counts = tuple(history.count(c) for c in range(vocab))
            p = posterior(counts, noise)
            assert p == likelihood_posterior(history, vocab, noise)
            assert sum(p) == 1
            histories += 1
            for truth in range(vocab):
                joint = Fraction(1, vocab) * math.prod(channel(x, truth, vocab, noise) for x in history)
                mass += joint
                expected_brier += joint * brier(p, truth)
                nll_terms.append(float(joint) * -math.log(float(p[truth])))
        assert mass == 1
        rows.append({"length": length, "histories": histories,
                     "probability_mass_exact": exact_string(mass),
                     "bayes_brier_exact": exact_string(expected_brier),
                     "bayes_brier": float(expected_brier), "bayes_nll": math.fsum(nll_terms)})
    assert all(Fraction(b["bayes_brier_exact"]) <= Fraction(a["bayes_brier_exact"])
               for a, b in zip(rows, rows[1:]))
    assert all(b["bayes_nll"] <= a["bayes_nll"] + 1e-14 for a, b in zip(rows, rows[1:]))
    return rows


def share_collision() -> dict:
    small = (2, 1) + (0,) * 14
    large = tuple(10 * n for n in small)
    shares_small = tuple(Fraction(n, sum(small)) for n in small)
    shares_large = tuple(Fraction(n, sum(large)) for n in large)
    assert shares_small == shares_large
    a, b = posterior(small), posterior(large)
    common_optimum = tuple((x + y) / 2 for x, y in zip(a, b))
    regret = sum((x - y) ** 2 for x, y in zip(a, b)) / 4
    direct = (sum((q - x) ** 2 for q, x in zip(common_optimum, a))
              + sum((q - y) ** 2 for q, y in zip(common_optimum, b))) / 2
    assert regret == direct > 0
    return {"same_shares_exact": True, "small_counts": list(small), "large_counts": list(large),
            "small_top_p_exact": exact_string(a[0]), "small_top_p": float(a[0]),
            "large_top_p_exact": exact_string(b[0]), "large_top_p": float(b[0]),
            "large_remaining_mass_exact": exact_string(1 - b[0]),
            "large_remaining_mass": float(1 - b[0]),
            "common_prediction_minimum_mean_brier_regret_exact": exact_string(regret),
            "common_prediction_minimum_mean_brier_regret": float(regret)}


def correlated_copies() -> list[dict]:
    """The actual law is one independent draw copied k times, not k fresh draws."""
    vocab, noise = 3, Fraction(1, 2)
    rows = []
    for copies in (1, 2, 4, 8, 32):
        proper, naive, excess = Fraction(0), Fraction(0), Fraction(0)
        proper_nll, naive_nll = [], []
        for symbol in range(vocab):
            one = tuple(int(c == symbol) for c in range(vocab))
            repeated = tuple(copies * n for n in one)
            pi, p = posterior(one, noise), posterior(repeated, noise)
            predictive_mass = sum(Fraction(1, vocab) * channel(symbol, y, vocab, noise) for y in range(vocab))
            excess += predictive_mass * sum((q - r) ** 2 for q, r in zip(p, pi))
            for truth in range(vocab):
                joint = Fraction(1, vocab) * channel(symbol, truth, vocab, noise)
                proper += joint * brier(pi, truth)
                naive += joint * brier(p, truth)
                proper_nll.append(float(joint) * -math.log(float(pi[truth])))
                naive_nll.append(float(joint) * -math.log(float(p[truth])))
        assert naive - proper == excess
        assert (excess == 0) == (copies == 1)
        rows.append({"copies": copies, "proper_brier_exact": exact_string(proper),
                     "naive_brier_exact": exact_string(naive), "excess_brier_exact": exact_string(excess),
                     "excess_brier": float(excess), "proper_nll": math.fsum(proper_nll),
                     "naive_nll": math.fsum(naive_nll)})
    return rows


def payload() -> dict:
    rows = enumerate_risks()
    return {"suite": "mc_information_v1", "status": "EXACT FINITE CHECKS; LOG LOSSES NUMERICAL",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),
            "independent_histories_checked": sum(row["histories"] for row in rows),
            "iid_risks": rows, "share_collision": share_collision(), "correlated_copies": correlated_copies()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = payload()
    if args.verify:
        assert result == json.loads(args.verify.read_text()), "exact replay differs"
        print("Exact replay: PASS")
    else:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Exact checks: PASS; {result['independent_histories_checked']} histories")


if __name__ == "__main__":
    main()

