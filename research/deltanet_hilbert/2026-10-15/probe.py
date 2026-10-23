"""Numerical operator identities and untrained known-period memory mechanisms."""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

from geometry import BASE, ROOT, arith, sha, verify_sources

sys.path.insert(0, str(ROOT))
from arithscope.bias import RamanujanBias

METHODS = ("simplex_constant", "orthogonal_constant", "orthogonal_mean", "orthogonal_mean_count",
           "orthogonal_additive_count", "bayes_oracle")


def delta_step(state, key, value, gain, decay=1.0):
    previous = decay * state
    return previous + gain * torch.outer(value - previous @ key, key)


def recurrent(query, key, value, gain):
    state = torch.zeros(value.shape[1], key.shape[1], dtype=key.dtype)
    outputs = []
    for q, k, v, beta in zip(query, key, value, gain):
        state = delta_step(state, k, v, beta)
        outputs.append(state @ q)
    return torch.stack(outputs)


def triangular(query, key, value, gain):
    length = len(gain)
    identity = torch.eye(length, dtype=key.dtype)
    lower = identity + gain[:, None] * (key @ key.T).tril(-1)
    innovations = torch.linalg.solve_triangular(lower, gain[:, None] * value, upper=False)
    return (query @ key.T).tril() @ innovations


def operator_checks() -> dict:
    generator = torch.Generator().manual_seed(9201)
    key = torch.randn(5, generator=generator, dtype=torch.float64)
    key /= key.norm()
    state = torch.randn(3, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    value = torch.randn(3, generator=generator, dtype=torch.float64)
    loss = (state @ key - value).square().sum() / 2
    gradient, = torch.autograd.grad(loss, (state,))
    gain = 0.4
    error = float((state.detach() - gain * gradient - delta_step(state.detach(), key, value, gain)).abs().max())
    assert error < 1e-10
    projection = delta_step(state.detach(), key, value, 1.0)
    projection_error = float((projection @ key - value).abs().max())
    e = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    energy_errors = []
    for beta in (0., 0.25, 1., 1.8, 2.):
        after = e @ (torch.eye(5, dtype=torch.float64) - beta * torch.outer(key, key))
        predicted = e.square().sum() - beta * (2 - beta * key.square().sum()) * (e @ key).square().sum()
        energy_errors.append(float((after.square().sum() - predicted).abs()))
        assert after.square().sum() <= e.square().sum() + 1e-10
    gen = torch.Generator().manual_seed(9202)
    k = torch.randn(12, 5, generator=gen, dtype=torch.float64)
    k /= k.norm(dim=-1, keepdim=True)
    q = torch.randn(12, 5, generator=gen, dtype=torch.float64)
    v = torch.randn(12, 3, generator=gen, dtype=torch.float64)
    beta = torch.rand(12, generator=gen, dtype=torch.float64)
    matrix_error = float((recurrent(q, k, v, beta) - triangular(q, k, v, beta)).abs().max())
    assert matrix_error < 1e-10 and projection_error < 1e-10 and max(energy_errors) < 1e-10
    signed_keys = torch.tensor([[1., 0.], [-0.5, math.sqrt(0.75)]], dtype=torch.float64)
    signed_query = signed_keys[[0]]
    transform = torch.eye(2, dtype=torch.float64) + (signed_keys @ signed_keys.T).tril(-1)
    weights = signed_query @ signed_keys.T @ torch.linalg.inv(transform)
    assert float(weights[0, 1]) < 0
    return {"sgd_update_error": error, "projection_constraint_error": projection_error,
            "maximum_energy_identity_error": max(energy_errors), "recurrent_triangular_error": matrix_error,
            "signed_final_query_weights": weights.tolist(),
            "unnormalized_key_norm_squared": 4, "gain_one_error_energy_multiplier": 9}


def phase_features(period: int, kind: str):
    orders = arith.divisors(period)
    if kind == "simplex":
        orders = tuple(m for m in orders if m > 1)
    elif kind != "orthogonal":
        raise ValueError("unknown feature geometry")
    module = RamanujanBias(1, orders=orders).double()
    denominator = period - 1 if kind == "simplex" else period
    per_order = torch.tensor([arith.euler_phi(m) / denominator for m in orders], dtype=torch.float64)
    weights = per_order[module.feature_group].sqrt()
    positions = torch.arange(period, dtype=torch.int64)
    features = module.key_features(positions) * weights
    shifted = module.key_features(positions + 10**12 * period) * weights
    assert torch.equal(features, shifted)
    if kind == "simplex":
        features = torch.cat((features, torch.zeros(period, 1, dtype=torch.float64)), dim=-1)
    assert features.shape == (period, period)
    return features


def feature_checks() -> list[dict]:
    rows = []
    for period in (12, 13):
        for kind in ("simplex", "orthogonal"):
            features = phase_features(period, kind)
            actual = features @ features.T
            expected = torch.eye(period, dtype=torch.float64)
            if kind == "simplex":
                expected = (period * expected - torch.ones_like(expected)) / (period - 1)
            error = float((actual - expected).abs().max())
            assert error < 1e-10
            rows.append({"period": period, "geometry": kind, "allocated_key_dimension": period,
                         "effective_rank": int(torch.linalg.matrix_rank(features)),
                         "gram_max_error": error, "period_shift_bitwise": True})
    return rows


def tensor_counterexample() -> dict:
    content = torch.tensor([1., 0.], dtype=torch.float64)
    phases = torch.eye(2, dtype=torch.float64)
    concatenated = [torch.cat((content, phase)) / math.sqrt(2) for phase in phases]
    tensor = [torch.kron(phase, content) for phase in phases]
    a, b = float(concatenated[0] @ concatenated[1]), float(tensor[0] @ tensor[1])
    assert abs(a - 0.5) < 1e-12 and b == 0
    return {"concatenation_cross_residue_inner_product": a, "tensor_cross_residue_inner_product": b}


def score(logits: np.ndarray, truth: int) -> tuple[float, float]:
    best = int(logits.argmax())
    peak = float(logits[best])
    other_mass = math.fsum(math.exp(float(x) - peak) for c, x in enumerate(logits) if c != best)
    nll = math.log1p(other_mass) + (peak - float(logits[truth]))
    winners = np.flatnonzero(np.abs(logits - peak) <= 1e-9)
    return nll, float(truth in winners) / len(winners)


def observation(period: int, sequence: int) -> dict:
    seed = f"2026101502:{period}:{sequence}"
    rng = random.Random(seed)
    motif = [rng.randrange(16) for _ in range(period)]
    tokens = [rng.randrange(16) if rng.random() < 0.5 else motif[t % period] for t in range(1024)]
    return {"period": period, "sequence": sequence, "seed": seed, "motif_for_scoring": motif, "tokens": tokens}


def simulate(raw: dict) -> tuple[list[dict], float]:
    period = raw["period"]
    simplex = phase_features(period, "simplex").detach().numpy()
    orthogonal = phase_features(period, "orthogonal").detach().numpy()
    states = [np.zeros((16, period), dtype=np.float64) for _ in range(3)]
    additive = np.zeros((16, period), dtype=np.float64)
    counts = np.zeros((period, 16), dtype=np.int64)
    metrics = {name: [] for name in METHODS}
    maximum_error = 0.0
    for t, symbol in enumerate(raw["tokens"]):
        residue = t % period
        counts[residue, symbol] += 1
        n = int(counts[residue].sum())
        value = np.eye(16, dtype=np.float64)[symbol]
        for state, features, gain in zip(states, (simplex, orthogonal, orthogonal), (0.25, 0.25, 1 / n)):
            key = features[residue]
            state += gain * np.outer(value - state @ key, key)
        additive += np.outer(value, orthogonal[residue])
        reads = [states[0] @ simplex[residue], states[1] @ orthogonal[residue], states[2] @ orthogonal[residue]]
        logits = [math.log(17) * x for x in reads]
        count_logits = math.log(17) * n * reads[2]
        additive_logits = math.log(17) * (additive @ orthogonal[residue])
        oracle_logits = math.log(17) * counts[residue]
        maximum_error = max(maximum_error, float(np.max(np.abs(count_logits - oracle_logits))),
                            float(np.max(np.abs(additive_logits - oracle_logits))))
        assert maximum_error < 1e-9
        # Clean labels are accessed only after all state writes and predictions.
        truth = raw["motif_for_scoring"][residue]
        for name, prediction in zip(METHODS, (*logits, count_logits, additive_logits, oracle_logits)):
            metrics[name].append(score(prediction, truth))
    rows = []
    for length in (128, 256, 512, 1024):
        for name in METHODS:
            selected = metrics[name][3 * length // 4:length]
            rows.append({"period": period, "sequence": raw["sequence"], "length": length,
                         "method": name, "positions": len(selected),
                         "nll": math.fsum(x[0] for x in selected) / len(selected),
                         "accuracy": math.fsum(x[1] for x in selected) / len(selected)})
    return rows, maximum_error


def payload() -> dict:
    verify_sources()
    torch.set_num_threads(1)
    raw = [observation(period, sequence) for period in (12, 13) for sequence in range(8)]
    rows, errors = [], []
    for stream in raw:
        output, error = simulate(stream)
        rows.extend(output)
        errors.append(error)
    summary = []
    for period in (12, 13):
        for length in (128, 256, 512, 1024):
            for method in METHODS:
                selected = [r for r in rows if r["period"] == period and r["length"] == length and r["method"] == method]
                summary.append({"period": period, "length": length, "method": method, "sequences": len(selected),
                                "mean_nll": math.fsum(r["nll"] for r in selected) / len(selected),
                                "mean_accuracy": math.fsum(r["accuracy"] for r in selected) / len(selected)})
    return {"suite": "delta_hilbert_geometry_v1", "status": "NUMERICAL CHECKS; UNTRAINED MECHANISM DIAGNOSTICS",
            "source_sha256": sha(Path(__file__)), "exact_source_sha256": sha(BASE / "geometry.py"),
            "protocol_sha256": sha(BASE / "PROTOCOL.md"),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "platform": platform.platform(), "threads": torch.get_num_threads()},
            "operators": operator_checks(), "features": feature_checks(), "tensor_product": tensor_counterexample(),
            "raw_observations": raw, "sequence_metrics": rows, "summary": summary,
            "count_logits_max_error": max(errors),
            "state_budget_per_mechanism": {str(p): {"fast_weight_scalars": 16 * p, "count_registers": p}
                                           for p in (12, 13)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    data = payload()
    if args.verify:
        assert data == json.loads(args.verify.read_text()), "numerical replay differs"
        print("Delta-Hilbert numerical replay: PASS")
    else:
        with args.output.open("x") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Delta-Hilbert numerical checks: PASS; {len(data['raw_observations'])} fresh streams")


if __name__ == "__main__":
    main()
