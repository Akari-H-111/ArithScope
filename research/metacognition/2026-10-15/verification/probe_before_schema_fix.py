"""Label-free gradient diagnostics on small functions and frozen checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from fractions import Fraction
from pathlib import Path

import torch

from study import posterior

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def uniform_kl(logits):
    return -logits.log_softmax(-1).mean() - math.log(logits.numel())


def analytic_checks() -> dict:
    analytic_errors, difference_errors = [], []
    for seed in range(5):
        gen = torch.Generator().manual_seed(9101 + seed)
        weight = torch.randn(3, 8, generator=gen, dtype=torch.float64, requires_grad=True)
        hidden = torch.randn(8, generator=gen, dtype=torch.float64)
        logits = weight @ hidden
        gradient, = torch.autograd.grad(uniform_kl(logits), (weight,))
        probability = logits.detach().softmax(-1)
        closed = torch.outer(probability - 1 / 3, hidden)
        analytic_errors.append(float((gradient - closed).abs().max()))
        step = 1e-6
        plus, minus = weight.detach().clone(), weight.detach().clone()
        plus[0, 0] += step
        minus[0, 0] -= step
        central = (uniform_kl(plus @ hidden) - uniform_kl(minus @ hidden)) / (2 * step)
        difference_errors.append(float((central - gradient[0, 0]).abs()))
    assert max(analytic_errors) < 1e-10
    assert max(difference_errors) < 1e-7
    return {"cases": 5, "maximum_analytic_autograd_error": max(analytic_errors),
            "maximum_central_difference_error": max(difference_errors), "difference_step": 1e-6}


def replication_checks() -> dict:
    gen = torch.Generator().manual_seed(9110)
    encoder = torch.randn(4, 2, generator=gen, dtype=torch.float64, requires_grad=True)
    readout = torch.randn(3, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    observations = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)

    def compute(values):
        hidden = (values @ encoder.T).tanh().mean(0)
        logits = readout @ hidden
        gradients = torch.autograd.grad(uniform_kl(logits), (encoder, readout))
        return logits.detach().softmax(-1), torch.cat([g.flatten() for g in gradients])

    p1, g1 = compute(observations)
    p2, g2 = compute(observations.repeat(10, 1))
    prediction_error = float((p1 - p2).abs().max())
    gradient_error = float((g1 - g2).abs().max())
    assert prediction_error < 1e-12 and gradient_error < 1e-12
    return {"replication_factor": 10, "maximum_prediction_difference": prediction_error,
            "maximum_all_parameter_gradient_difference": gradient_error}


def jacobian_collision() -> list[dict]:
    rows = []
    for count in (3, 30):
        theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        logits = torch.stack((2 + theta * math.log(count), theta * 0))
        gradient, = torch.autograd.grad(uniform_kl(logits), (theta,))
        rows.append({"count": count, "top_probability": float(logits.detach().softmax(0)[0]),
                     "gradient": float(gradient)})
    assert rows[0]["top_probability"] == rows[1]["top_probability"]
    assert rows[0]["gradient"] != rows[1]["gradient"]
    return rows


def measure(model, tokens, query: int) -> dict:
    """Accept observed tokens only. A clean target is deliberately not an input."""
    if not 0 <= query < tokens.shape[1]:
        raise ValueError("query must be a valid token index")
    prefix = tokens[:, :query + 1]
    captured = []
    hook = model.ln_f.register_forward_hook(lambda _module, _args, output: captured.append(output))
    try:
        logits = model(prefix, backend="explicit")[0, -1]
    finally:
        hook.remove()
    loss = uniform_kl(logits)
    gate = model.blocks[-1].attn.biases["arith"].gamma
    head_gradient, gate_gradient = torch.autograd.grad(loss, (model.head.weight, gate))
    hidden = captured[0][0, -1].detach()
    probability = logits.detach().softmax(-1)
    closed = torch.outer(probability - 1 / probability.numel(), hidden)
    error = float((head_gradient - closed).abs().max())
    assert error < 1e-10
    formula_norm = float(torch.linalg.vector_norm(probability - 1 / probability.numel())
                         * torch.linalg.vector_norm(hidden))
    actual_norm = float(torch.linalg.vector_norm(head_gradient))
    assert abs(formula_norm - actual_norm) < 1e-10
    return {"probability": probability.tolist(), "proxy_uniform_kl": float(loss.detach()),
            "entropy": float(-(probability * probability.log()).sum()),
            "hidden_norm": float(torch.linalg.vector_norm(hidden)),
            "head_gradient_norm": actual_norm, "head_norm_from_forward": formula_norm,
            "head_gradient_max_error": error,
            "second_layer_gate_gradient_norm": float(torch.linalg.vector_norm(gate_gradient))}


def window_prediction(model, tokens, query: int, window: int) -> list[float]:
    attention = model.blocks[-1].attn
    before = attention.window
    try:
        attention.window = window
        with torch.no_grad():
            return model(tokens[:, :query + 1], backend="explicit")[0, -1].softmax(-1).tolist()
    finally:
        attention.window = before


def sample(period: int, sequence: int, vocab: int, noise: float, length: int = 256) -> dict:
    seed = f"2026101501:{period}:{sequence}"
    rng = random.Random(seed)
    motif = [rng.randrange(vocab) for _ in range(period)]
    tokens = [rng.randrange(vocab) if rng.random() < noise else motif[t % period] for t in range(length)]
    return {"period": period, "sequence": sequence, "seed": seed, "motif": motif, "tokens": tokens}


def checkpoint_checks() -> dict:
    manifest_path = BASE / "inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["source_sha256"].items():
        assert sha(ROOT / relative) == expected, f"parent source drift: {relative}"
    sys.path.insert(0, str(ROOT))
    from arithscope.train import RunSpec, build_model

    records, observations, digests = [], [], []
    for entry in manifest["checkpoints"]:
        path, run_path = BASE / entry["copy"], BASE / entry["record_copy"]
        assert sha(path) == entry["sha256"] and sha(run_path) == entry["record_sha256"]
        blob = torch.load(path, map_location="cpu", weights_only=True)
        original = json.loads(run_path.read_text())
        assert json.loads(json.dumps(blob["spec"])) == original["spec"]
        spec = RunSpec(**blob["spec"])
        assert spec.variant == "arith" and spec.seed == 1
        period, vocab, noise = entry["period"], spec.task["vocab_size"], spec.task["noise"]
        assert tuple(spec.task["periods"]) == (period,) and noise == 0.5
        model = build_model(spec, vocab).double().eval()
        model.load_state_dict({name: value.double() if value.is_floating_point() else value
                               for name, value in blob["state_dict"].items()})
        before = state_digest(model)
        for sequence in range(4):
            raw = sample(period, sequence, vocab, noise)
            observations.append(raw)
            tokens = torch.tensor([raw["tokens"]], dtype=torch.int64)
            for query in (63, 127, 255):
                monitor = measure(model, tokens, query)
                limited = window_prediction(model, tokens, query, 64)
                if query == 63:
                    assert monitor["probability"] == limited
                observed = [raw["tokens"][t] for t in range(query + 1) if t % period == query % period]
                counts = tuple(observed.count(c) for c in range(vocab))
                pi = posterior(counts, Fraction(1, 2))
                truth = raw["motif"][query % period]
                full = monitor["probability"]
                risk_full = math.fsum(-float(p) * math.log(q) for p, q in zip(pi, full))
                risk_limited = math.fsum(-float(p) * math.log(q) for p, q in zip(pi, limited))
                entropy = math.fsum(-float(p) * math.log(float(p)) for p in pi)
                records.append({"period": period, "sequence": sequence, "query": query,
                                "counts": list(counts), "count": sum(counts), "truth_for_scoring": truth,
                                **monitor, "window_probability": limited,
                                "full_nll": -math.log(full[truth]), "window_nll": -math.log(limited[truth]),
                                "full_correct": int(max(range(vocab), key=full.__getitem__) == truth),
                                "window_correct": int(max(range(vocab), key=limited.__getitem__) == truth),
                                "oracle_expected_full_nll": risk_full,
                                "oracle_expected_window_nll": risk_limited,
                                "oracle_full_kl": risk_full - entropy,
                                "oracle_window_kl": risk_limited - entropy,
                                "oracle_window_benefit": risk_full - risk_limited})
        after = state_digest(model)
        assert before == after and all(p.grad is None for p in model.parameters())
        digests.append({"period": period, "state_before": before, "state_after": after})
    summary = []
    for period in (12, 13):
        rows = [row for row in records if row["period"] == period and row["query"] > 63]
        summary.append({"period": period, "rows": len(rows), "independent_sequences": 4,
                        "mean_full_nll": math.fsum(r["full_nll"] for r in rows) / len(rows),
                        "mean_window_nll": math.fsum(r["window_nll"] for r in rows) / len(rows),
                        "mean_oracle_window_benefit": math.fsum(r["oracle_window_benefit"] for r in rows) / len(rows),
                        "full_correct": sum(r["full_correct"] for r in rows),
                        "window_correct": sum(r["window_correct"] for r in rows)})
    return {"input_manifest_sha256": sha(manifest_path), "raw_observations": observations,
            "checkpoint_state_digests": digests, "records": records, "summary": summary,
            "maximum_head_gradient_error": max(r["head_gradient_max_error"] for r in records),
            "maximum_forward_norm_error": max(abs(r["head_gradient_norm"] - r["head_norm_from_forward"])
                                              for r in records)}


def payload() -> dict:
    torch.set_num_threads(1)
    return {"suite": "mc_information_v1", "status": "NUMERICAL CHECKS AND DEVELOPMENT DIAGNOSTICS",
            "source_sha256": sha(Path(__file__)), "exact_source_sha256": sha(BASE / "study.py"),
            "protocol_sha256": sha(BASE / "PROTOCOL.md"),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "platform": platform.platform(), "threads": torch.get_num_threads(), "dtype": "float64"},
            "linear_gradient": analytic_checks(), "replication": replication_checks(),
            "pointwise_collision": jacobian_collision(), "checkpoints": checkpoint_checks()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = payload()
    if args.verify:
        assert result == json.loads(args.verify.read_text()), "numerical replay differs"
        print("Gradient/checkpoint replay: PASS")
    else:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Gradient checks: PASS; {len(result['checkpoints']['records'])} frozen-checkpoint queries")


if __name__ == "__main__":
    main()

