"""Exact arithmetic and memory-geometry checks; standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))
from arithscope import arith


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> dict:
    manifest = json.loads((BASE / "inputs/manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if relative.startswith("arithscope/"):
            assert sha(ROOT / relative) == expected, f"parent source drift: {relative}"
    return manifest


def gram(period: int, kind: str) -> list[list[Fraction]]:
    if period < 2 or kind not in ("simplex", "orthogonal"):
        raise ValueError("need period >= 2 and a registered geometry")
    off = Fraction(-1, period - 1) if kind == "simplex" else Fraction(0)
    return [[Fraction(1) if i == j else off for j in range(period)] for i in range(period)]


def rank(matrix: list[list[Fraction]]) -> int:
    a = [list(row) for row in matrix]
    row = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(row, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        scale = a[row][col]
        a[row] = [x / scale for x in a[row]]
        for i in range(row + 1, len(a)):
            scale = a[i][col]
            a[i] = [x - scale * y for x, y in zip(a[i], a[row])]
        row += 1
        if row == len(a):
            break
    return row


def kernel_identity() -> dict:
    checked = 0
    for period in range(2, 33):
        weights = {m: Fraction(sum(math.gcd(a, m) == 1 for a in range(1, m + 1)), period)
                   for m in range(1, period + 1) if period % m == 0}
        assert sum(weights.values()) == 1
        for residue in range(period):
            value = arith.mixture_value(weights, residue)
            optimum = arith.mixture_value(arith.optimal_comb_weights(period), residue)
            assert value == int(residue == 0)
            assert value == (1 + (period - 1) * optimum) / period
            checked += 1
    return {"period_min": 2, "period_max": 32, "residues_checked": checked}


def write(coefficients: list[Fraction], g: list[list[Fraction]], address: int,
          value: Fraction, gain: Fraction) -> None:
    prediction = sum(c * row[address] for c, row in zip(coefficients, g))
    coefficients[address] += gain * (value - prediction)


def read(coefficients: list[Fraction], g: list[list[Fraction]]) -> list[Fraction]:
    return [sum(c * row[address] for c, row in zip(coefficients, g)) for address in range(len(g))]


def constant_recall() -> list[dict]:
    rows = []
    for period in (3, 12, 13):
        for kind in ("simplex", "orthogonal"):
            g = gram(period, kind)
            coefficients = [Fraction(0)] * period
            for address in range(period):
                write(coefficients, g, address, Fraction(1), Fraction(1))
            predictions = read(coefficients, g)
            mse = sum((x - 1) ** 2 for x in predictions) / period
            assert rank(g) == period - int(kind == "simplex")
            if kind == "simplex":
                assert sum(predictions) == 0 and mse >= 1
            else:
                assert predictions == [Fraction(1)] * period and mse == 0
            rows.append({"period": period, "geometry": kind, "rank": rank(g),
                         "readouts_exact": [str(x) for x in predictions],
                         "mean_squared_error_exact": str(mse), "mean_squared_error": float(mse)})
    return rows


def mean_identity() -> dict:
    period, vocab = 3, 4
    g = gram(period, "orthogonal")
    coefficients = [[Fraction(0)] * period for _ in range(vocab)]
    additive = [[Fraction(0)] * period for _ in range(vocab)]
    counts = [[0] * vocab for _ in range(period)]
    tokens = (0, 1, 2, 1, 1, 3, 0, 2, 2, 3, 0, 1)
    for t, symbol in enumerate(tokens):
        address = t % period
        counts[address][symbol] += 1
        n = sum(counts[address])
        for c in range(vocab):
            write(coefficients[c], g, address, Fraction(c == symbol), Fraction(1, n))
            additive[c][address] += Fraction(c == symbol)
        for r in range(period):
            total = sum(counts[r])
            for c in range(vocab):
                expected = Fraction(counts[r][c], total) if total else Fraction(0)
                assert read(coefficients[c], g)[r] == expected
                assert read(additive[c], g)[r] == counts[r][c]
    return {"tokens_checked": len(tokens), "period": period, "vocab": vocab,
            "every_prefix_exact": True, "additive_counts_exact": True}


def order_and_drift() -> dict:
    def running(values, schedule):
        state = Fraction(0)
        for n, value in enumerate(values, 1):
            gain = Fraction(1, n) if schedule == "mean" else Fraction(1, 4)
            state += gain * (value - state)
        return state

    order_a, order_b = running((0, 1), "fixed"), running((1, 0), "fixed")
    assert order_a != order_b
    assert running((0, 1), "mean") == running((1, 0), "mean")
    values = (0,) * 16 + (1,) * 4
    mean, fixed = running(values, "mean"), running(values, "fixed")
    assert abs(1 - fixed) < abs(1 - mean)
    beta = Fraction(1, 4)
    return {"same_multiset_fixed_a_exact": str(order_a), "same_multiset_fixed_b_exact": str(order_b),
            "change_point_mean_exact": str(mean), "change_point_fixed_exact": str(fixed),
            "fixed_gain": str(beta), "limiting_independent_sample_equivalent_exact": str((2 - beta) / beta)}


def payload() -> dict:
    verify_sources()
    return {"suite": "delta_hilbert_geometry_v1", "status": "EXACT FINITE CHECKS",
            "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(BASE / "PROTOCOL.md"),
            "kernel_identity": kernel_identity(), "constant_recall": constant_recall(),
            "running_mean": mean_identity(), "order_and_drift": order_and_drift()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    data = payload()
    if args.verify:
        assert data == json.loads(args.verify.read_text()), "exact replay differs"
        print("Exact geometry replay: PASS")
    else:
        with args.output.open("x") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Exact geometry: PASS; {data['kernel_identity']['residues_checked']} residues")


if __name__ == "__main__":
    main()
