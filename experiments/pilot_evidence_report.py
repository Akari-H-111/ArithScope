"""Report for the evidence-head development pilot (suite ``pilot_evidence``). Development pilot, not confirmatory.

    python experiments/pilot_evidence_report.py [--runs results/runs/pilot_evidence]
                                                [--checkpoints checkpoints/pilot_evidence]
                                                [--summary results/summary/pilot_evidence.json]

Reads the run records, regenerates each test set deterministically (the generator parts of
``arithscope.train.run``; the stored digest is checked), and computes the exact Bayes NLL per
position (docs/MATH.md, B1):

    NLL_t = logsumexp_b(c_b log rho) - c_true log rho,    c_b = #{j <= t : j = t mod P, x_j = b},

with the own token included and ``rho`` the likelihood ratio of a clean over a noisy-miss token.
Writes the JSON summary and a markdown table next to it. "Far" means the second half of the longest
evaluation length (positions 4,096-8,191 at N = 8,192). Per-bucket numbers come from the run's
position buckets; per-position statistics (maximum NLL, positions above 1 nat) need the run's
checkpoint and are skipped (null) when it is missing. The learned kernel is rebuilt from the
coefficients stored in the run record.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope import arith  # noqa: E402
from arithscope.evidence import EvidenceConfig, make_kernel  # noqa: E402
from arithscope.tasks import make_task, task_dict  # noqa: E402
from arithscope.train import RunSpec, _eval_chunk, _generator, build_model  # noqa: E402

LABEL = "development pilot, not confirmatory"
VARIANT_ORDER = ("arith", "rope", "rope+arith")
KERNEL_ORDER = ("none", "arith", "cosine_fund", "cosine_full", "cosine_learned", "uniform")
FIT_RANGE = 720        # offsets 0 .. 719 for the fit of the learned kernel to c * D_P
BAYES_FLOOR = 1e-6     # NLL / Bayes ratios are reported only where the Bayes NLL exceeds this
OVERCONFIDENT = 1.0    # nats
BUCKET_LENGTHS = (1024, 8192)
COEFFICIENTS = ("beta", "coef", "omega", "amplitude")


def bayes_nll(tokens: torch.Tensor, truth: torch.Tensor, period: int, rho: float, vocab: int) -> torch.Tensor:
    """Exact Bayes NLL ``[B, N]`` of the true symbol given the same-residue votes up to and including ``t``."""
    b, n = tokens.shape
    pad = (-n) % period
    onehot = F.one_hot(F.pad(tokens, (0, pad)), vocab).double().view(b, -1, period, vocab)
    counts = onehot.cumsum(dim=1).view(b, -1, vocab)[:, :n]  # votes of residue class t mod P up to t
    log_rho = math.log(rho)
    own = counts.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
    return torch.logsumexp(counts * log_rho, dim=-1) - own * log_rho


def _task(spec: dict):
    task = make_task(spec["task"]["kind"], **{k: v for k, v in spec["task"].items() if k != "kind"})
    if task.name != "periodic" or len(task.periods) != 1 or task.query_rate > 0:
        raise ValueError("the Bayes readout here covers single-period PeriodicDenoising without queries")
    return task


def _bucket_mean(buckets: list[dict], start: int, key: str) -> float | None:
    rows = [b for b in buckets if b["start"] >= start and b["n"]]
    n = sum(b["n"] for b in rows)
    return sum(b[key] * b["n"] for b in rows) / n if n else None


def _buckets(test: dict, bayes: torch.Tensor) -> list[dict]:
    """Per position bucket: model NLL and accuracy (stored), Bayes NLL, the difference, and the ratio where defined."""
    rows = []
    for b in test["by_position"]:
        bayes_b = float(bayes[:, b["start"]:b["end"]].mean())
        rows.append({"start": b["start"], "end": b["end"], "nll": b["nll"], "bayes_nll": bayes_b,
                     "diff": b["nll"] - bayes_b, "ratio": b["nll"] / bayes_b if bayes_b > BAYES_FLOOR else None,
                     "acc": b["acc"], "oracle_acc": b["oracle_acc"]})
    return rows


@torch.no_grad()
def _far_stats(checkpoint: Path, batch, bayes: torch.Tensor, half: int, stored_nll: float, vocab: int) -> dict:
    """Per-position model NLL in the far region, from the run's checkpoint."""
    blob = torch.load(checkpoint, weights_only=False)
    fields = dict(blob["spec"])
    for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
        fields[key] = tuple(fields[key])
    model = build_model(RunSpec(**fields), vocab)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    n = batch.tokens.shape[1]
    positions = torch.arange(n)
    nll = torch.zeros(batch.tokens.shape)
    chunk = _eval_chunk(n)
    for i in range(0, batch.tokens.shape[0], chunk):
        logits = model(batch.tokens[i:i + chunk], positions)
        nll[i:i + chunk] = F.cross_entropy(logits.transpose(1, 2), batch.targets[i:i + chunk], reduction="none")
    far, bayes_far = nll[:, half:], bayes[:, half:]
    return {"far_positions": far.numel(), "far_max_nll": float(far.max()),
            "far_over_1nat": int((far > OVERCONFIDENT).sum()), "far_over_1nat_fraction": float((far > OVERCONFIDENT).float().mean()),
            "bayes_far_max_nll": float(bayes_far.max()), "bayes_far_over_1nat": int((bayes_far > OVERCONFIDENT).sum()),
            "recomputed_mean_nll_minus_stored": float(nll.mean()) - stored_nll}


def _learned_values(ev: dict, length: int) -> torch.Tensor:
    """The learned kernel ``kappa(n)``, n = 0 .. length-1, rebuilt in float64 from the stored coefficients."""
    kwargs = {"orders": tuple(ev["orders"])} if "orders" in ev else {}
    kernel = make_kernel(EvidenceConfig(ev["kernel"], **kwargs)).double()
    with torch.no_grad():
        for name in COEFFICIENTS:
            if name in ev:
                getattr(kernel, name).copy_(torch.tensor(ev[name], dtype=torch.float64))
        return kernel.offset_values(torch.arange(length))[0]


def _kernel_summary(ev: dict, period: int) -> dict:
    values = _learned_values(ev, FIT_RANGE)
    comb = (torch.arange(FIT_RANGE) % period == 0).double()
    c = float(values @ comb / (comb @ comb))
    return {"row_0_to_2P_minus_1": values[:2 * period].tolist(), "kappa_0": float(values[0]),
            "max_abs_off_comb_1_to_P_minus_1": float(values[1:period].abs().max()),
            "fit_c": c, "fit_relative_residual": float((values - c * comb).norm() / values.norm()),
            "fit_range": FIT_RANGE}


def _d_p_coefficients(ev: dict, period: int) -> dict:
    orders, beta = ev["orders"], ev["beta"]
    target = [arith.euler_phi(m) / period if period % m == 0 else 0.0 for m in orders]
    dot = sum(x * y for x, y in zip(beta, target))
    norms = math.sqrt(sum(x * x for x in beta)) * math.sqrt(sum(y * y for y in target))
    return {"orders": orders, "beta": beta, "d_p": target,
            "divisor_rows": [{"m": m, "beta": x, "d_p": y} for m, x, y in zip(orders, beta, target) if period % m == 0],
            "nondivisor_max_abs": max([abs(x) for m, x in zip(orders, beta) if period % m] or [0.0]),
            "l1_to_d_p": sum(abs(x - y) for x, y in zip(beta, target)),
            "cosine_to_d_p": dot / norms if norms else None}


def _cosine_harmonics(ev: dict, period: int) -> list[dict]:
    omega, amplitude = ev["omega"], ev["amplitude"]
    periods = torch.logspace(math.log10(2.0), math.log10(32.0), len(omega))  # as at initialization
    init = (2.0 * math.pi / periods).tolist()
    rows = []
    for a in range(1, period // 2 + 1):  # omega nearest 2 pi a / P
        target = 2.0 * math.pi * a / period
        f = min(range(len(omega)), key=lambda i: abs(omega[i] - target))
        rows.append({"harmonic": a, "target_omega": target, "index": f, "omega": omega[f], "omega_init": init[f],
                     "drift": omega[f] - init[f], "epsilon": omega[f] - target, "amplitude": amplitude[f]})
    return rows


def summarize(path: Path, checkpoints: Path | None = None) -> dict:
    record = json.loads(path.read_text())
    spec = record["spec"]
    task = _task(spec)
    period = task.periods[0]
    noise_hit = task.noise / task.vocab_size
    rho = (1.0 - task.noise + noise_hit) / noise_hit
    kernel = (spec.get("evidence") or {}).get("kernel", "none")
    longest = max(t["length"] for t in record["test"])
    lengths, buckets, digest_ok, far = [], {}, True, None
    for test in record["test"]:
        length = test["length"]
        budget = spec["eval_tokens"] if length <= 2048 else spec["eval_tokens_long"]
        parts = ("test", spec["data_seed"], json.dumps(task_dict(task), sort_keys=True), length)
        batch = task.sample(max(2, budget // length), length, _generator(*parts))
        digest_ok &= batch.digest() == test["digest"]
        nll = bayes_nll(batch.tokens, batch.targets, period, rho, task.vocab_size)
        half = torch.linspace(0, length, 9).long().tolist()[4]
        lengths.append({"length": length, "acc": test["all"]["acc"], "nll": test["all"]["nll"],
                        "oracle_acc": test["all"]["oracle_acc"], "bayes_nll": float(nll.mean()),
                        "far_start": half, "far_nll": _bucket_mean(test["by_position"], half, "nll"),
                        "far_acc": _bucket_mean(test["by_position"], half, "acc"),
                        "bayes_far_nll": float(nll[:, half:].mean())})
        buckets[str(length)] = _buckets(test, nll)
        checkpoint = None if checkpoints is None else checkpoints / (path.stem + ".pt")
        if length == longest and checkpoint is not None and checkpoint.exists():
            far = _far_stats(checkpoint, batch, nll, half, test["all"]["nll"], task.model_vocab)
    out = {"file": path.name, "period": period, "variant": spec["variant"], "kernel": kernel, "seed": spec["seed"],
           "rho": rho, "digest_ok": digest_ok, "lengths": lengths, "buckets": buckets, "far_stats": far}
    ev = record.get("evidence")
    if ev:
        w = ev["w_out"]
        out["w_out"] = {k: v for k, v in w.items() if k != "diag"}
        if w["offdiag_mean"] is not None:
            out["w_out"]["diag_minus_offdiag"] = w["diag_mean"] - w["offdiag_mean"]
        if kernel != "uniform":
            out["learned_kernel"] = _kernel_summary(ev, period)
        if kernel == "arith":
            out["d_p_coefficients"] = _d_p_coefficients(ev, period)
        elif kernel == "cosine_learned":
            out["harmonics"] = _cosine_harmonics(ev, period)
    return out


def _fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]


def markdown(rows: list[dict]) -> str:
    cap = f"*{LABEL}*"
    lines = [f"# Evidence head: {LABEL}", "",
             "Development data (seed 7), one training seed per arm; no intervals, no verdicts. "
             "Far = second half of the longest length. Bayes = exact B1 NLL on the same positions. "
             f"Reference: log 17 = {math.log(17):.3f} for the Bayes construction at rho = 17.", ""]
    for period in sorted({r["period"] for r in rows}):
        group = sorted((r for r in rows if r["period"] == period),
                       key=lambda r: (VARIANT_ORDER.index(r["variant"]), KERNEL_ORDER.index(r["kernel"])))
        name = lambda r: f"{r['variant']} + {r['kernel']}"  # noqa: E731
        lengths = [e["length"] for e in group[0]["lengths"]]
        lines += [f"## P = {period}", "", f"Accuracy and NLL by length. {cap}", ""]
        body = [[name(r)] + [_fmt(e["acc"]) for e in r["lengths"]] + [_fmt(e["nll"]) for e in r["lengths"]] for r in group]
        body.append(["Bayes (oracle acc / exact NLL)"] + [_fmt(e["oracle_acc"]) for e in group[0]["lengths"]] +
                    [_fmt(e["bayes_nll"], 4) for e in group[0]["lengths"]])
        lines += _table(["arm"] + [f"acc {n}" for n in lengths] + [f"NLL {n}" for n in lengths], body)
        last = [r["lengths"][-1] for r in group]
        lines += ["", f"Far NLL (positions {last[0]['far_start']}-{last[0]['length'] - 1} at N = {last[0]['length']}); "
                  f"per-position maximum and positions above {OVERCONFIDENT:g} nat from the checkpoint. {cap}", ""]
        body = []
        for r, e in zip(group, last):
            fs = r["far_stats"] or {}
            gap = None if e["far_nll"] is None else e["far_nll"] - e["bayes_far_nll"]
            body.append([name(r), _fmt(e["far_nll"]), _fmt(e["bayes_far_nll"], 4), _fmt(gap), _fmt(e["far_acc"]),
                         _fmt(fs.get("far_max_nll")), "n/a" if not fs else f"{fs['far_over_1nat']} / {fs['far_positions']}",
                         "n/a" if not fs else f"{fs['bayes_far_over_1nat']} (max {fs['bayes_far_max_nll']:.2f})"])
        lines += _table(["arm", "far NLL", "Bayes far NLL", "gap", "far acc", "max NLL", f"# > {OVERCONFIDENT:g} nat",
                         "Bayes # > 1 nat"], body)
        for n in BUCKET_LENGTHS:
            if str(n) not in group[0]["buckets"]:
                continue
            ref = group[0]["buckets"][str(n)]
            edges = [f"{b['start']}-{b['end'] - 1}" for b in ref]
            lines += ["", f"N = {n}: NLL - Bayes NLL (ratio NLL / Bayes where Bayes > {BAYES_FLOOR:g}) per position bucket. {cap}", ""]
            body = [["Bayes NLL"] + [_fmt(b["bayes_nll"], 5) for b in ref]]
            for r in group:
                body.append([name(r)] + [f"{_fmt(b['diff'], 4)} ({_fmt(b['ratio'], 2) if b['ratio'] is not None else '-'})"
                                         for b in r["buckets"][str(n)]])
            lines += _table(["arm"] + edges, body)
        if "1024" in group[0]["buckets"]:
            ref = group[0]["buckets"]["1024"]
            lines += ["", f"N = 1024: accuracy (plurality oracle accuracy) per position bucket. {cap}", ""]
            body = [[name(r)] + [f"{_fmt(b['acc'])} ({_fmt(b['oracle_acc'])})" for b in r["buckets"]["1024"]] for r in group]
            lines += _table(["arm"] + [f"{b['start']}-{b['end'] - 1}" for b in ref], body)
        lines += ["", f"W_out (diagonal mean, off-diagonal mean, their difference; compare log 17 = {math.log(17):.3f}). {cap}", ""]
        lines += _table(["arm", "diag mean", "off-diag mean", "diag - off-diag", "off-diag abs mean"],
                        [[name(r), _fmt(w.get("diag_mean")), _fmt(w.get("offdiag_mean")), _fmt(w.get("diag_minus_offdiag")),
                          _fmt(w.get("offdiag_abs_mean"))] for r in group if (w := r.get("w_out"))])
        lines += ["", f"Learned kernel kappa(n): kappa(0), largest |kappa(n)| for n = 1..{period - 1}, and the least-squares fit to "
                  f"c * D_P on n in [0, {FIT_RANGE}) (relative residual = |kappa - c D_P| / |kappa|). {cap}", ""]
        lines += _table(["arm", "kappa(0)", "max off-comb |kappa|", "fit c", "relative residual"],
                        [[name(r), _fmt(k["kappa_0"]), _fmt(k["max_abs_off_comb_1_to_P_minus_1"]), _fmt(k["fit_c"]),
                          _fmt(k["fit_relative_residual"])] for r in group if (k := r.get("learned_kernel"))])
        lines += ["", f"kappa(n) for n = 0 .. {2 * period - 1}:", ""]
        lines += [f"- {name(r)}: " + ", ".join(f"{v:.3f}" for v in r["learned_kernel"]["row_0_to_2P_minus_1"])
                  for r in group if r.get("learned_kernel")]
        for r in group:
            if r["kernel"] == "arith" and r.get("d_p_coefficients"):
                d = r["d_p_coefficients"]
                lines += ["", f"Learned arith coefficients, {r['variant']} (beta vs D_P = phi(m)/P on divisors of {period}, 0 elsewhere): "
                          f"cosine to D_P {_fmt(d['cosine_to_d_p'])}, L1 distance {_fmt(d['l1_to_d_p'])}, "
                          f"max |beta| off the divisors {_fmt(d['nondivisor_max_abs'])}. {cap}", ""]
                lines += _table(["m", "beta", "D_P"], [[str(x["m"]), _fmt(x["beta"], 4), _fmt(x["d_p"], 4)] for x in d["divisor_rows"]])
            elif r["kernel"] == "cosine_learned" and r.get("harmonics"):
                lines += ["", f"Learned cosine frequencies, {r['variant']}: omega nearest 2 pi a / {period}; "
                          f"epsilon = omega - target, drift = omega - initial. {cap}", ""]
                lines += _table(["a", "target", "index", "omega", "initial", "epsilon", "drift", "amplitude"],
                                [[str(h["harmonic"]), _fmt(h["target_omega"], 4), str(h["index"]), _fmt(h["omega"], 4),
                                  _fmt(h["omega_init"], 4), _fmt(h["epsilon"], 4), _fmt(h["drift"], 4), _fmt(h["amplitude"], 4)]
                                 for h in r["harmonics"]])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=str(ROOT / "results" / "runs" / "pilot_evidence"))
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints" / "pilot_evidence"))
    parser.add_argument("--summary", default=str(ROOT / "results" / "summary" / "pilot_evidence.json"))
    args = parser.parse_args()
    files = sorted(Path(args.runs).glob("*.json"))
    if not files:
        raise SystemExit(f"no run records in {args.runs}")
    rows = [summarize(p, Path(args.checkpoints)) for p in files]
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": LABEL, "runs": rows}, indent=1))
    table = markdown(rows)
    out.with_suffix(".md").write_text(table)
    print(table)
    if not all(r["digest_ok"] for r in rows):
        raise SystemExit("test-set digest mismatch: regenerated data differs from the stored records")


if __name__ == "__main__":
    main()
