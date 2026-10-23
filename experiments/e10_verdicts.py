"""E10 verdicts (research/evidence_readout/2026-10-23/E10_SPEC.md), frozen with the preregistration.

    python experiments/e10_verdicts.py

Reads the training runs in results/runs/e10_evidence/ and the copied-law evaluation records in
results/runs/e10_evidence_copy_eval/ (experiments/e10_copy_eval.py) and writes results/summary/e10.json.
Far NLL and far accuracy are E6's (positions 4,096-8,191 of the N = 8,192 test, ``e6_verdicts.metrics``,
imported unchanged); intervals are 95% t intervals over seeds (``e6_verdicts.ci95``), paired per seed within a
period. The module also provides the loaders and per-run metrics used by experiments/e10_power.py.

Claims: V1 (the head lowers ArithScope's far NLL at no accuracy cost), V2 (excess NLL at 1,024 below 0.005 nats),
V3 (at P = 13 the exact basis beats the matched fundamental cosines), V4 (the head restores RoPE's
extrapolation), B1 (boundary: worse than no head on copied observations). Secondary: S12-S15. A run or
evaluation that failed (an ``.error.txt`` file) is reported and its seed is dropped from every comparison of
that period that uses it, never replaced. A record that is neither present nor failed stops the script.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope import arith  # noqa: E402
from arithscope.tasks import make_task, periodic_bayes_nll  # noqa: E402
from arithscope.train import RunSpec, _generator, task_dict  # noqa: E402
from experiments.e6_verdicts import ACC_MARGIN, FAR, LENGTH, PERIODS, ci95, holm, metrics  # noqa: E402
from experiments.suites import E10_COPY_LENGTHS, E10_SEEDS, SUITES, e10_arm  # noqa: E402

SUITE = "e10_evidence"
ARMS = ("arith", "arith_ev", "arith_evfund", "rope", "rope_ev")
EXCESS_LENGTH, INDEP_LENGTH, COPY_LENGTH = 1024, 256, 256
V2_MARGIN = 0.005   # nats
V3_ACC = 0.10       # +10 pp
V4_ACC = 0.20       # +20 pp
S12_BAND = 0.01     # +-1 pp
S13_COSINE = 0.99
S14_BAND = 0.10     # lambda_eff within +-10% of log 17
LOG_RHO = math.log(17)

_BAYES: dict = {}


def spec_of(rec: dict) -> RunSpec:
    fields = dict(rec["spec"])
    for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
        fields[key] = tuple(fields[key])
    return RunSpec(**fields)


def excess_nll(rec: dict, length: int = EXCESS_LENGTH) -> float:
    """NLL over all positions of the test at ``length`` minus the exact Bayes NLL on the same (regenerated) batch."""
    spec = spec_of(rec)
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    test = next(t for t in rec["test"] if t["length"] == length)
    num = max(2, (spec.eval_tokens if length <= 2048 else spec.eval_tokens_long) // length)
    parts = ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), length)
    if parts + (num,) not in _BAYES:
        batch = task.sample(num, length, _generator(*parts))
        _BAYES[parts + (num,)] = (batch.digest(), float(periodic_bayes_nll(batch, task).mean()))
    digest, bayes = _BAYES[parts + (num,)]
    if digest != test["digest"]:
        raise ValueError(f"regenerated test batch differs from the record's digest: {spec.filename}")
    return test["all"]["nll"] - bayes


def kernel_quantities(rec: dict) -> dict:
    """cosine(beta, D_P coefficients) and lambda_eff = (sum_{m | P} beta_m) * (diag mean - off-diagonal mean of W_out)."""
    ev, period = rec["evidence"], spec_of(rec).task["periods"][0]
    orders, beta = ev["orders"], ev["beta"]
    target = [arith.euler_phi(m) / period if period % m == 0 else 0.0 for m in orders]
    dot = sum(x * y for x, y in zip(beta, target))
    norm = math.sqrt(sum(x * x for x in beta)) * math.sqrt(sum(y * y for y in target))
    scale = sum(b for m, b in zip(orders, beta) if period % m == 0) * (ev["w_out"]["diag_mean"] - ev["w_out"]["offdiag_mean"])
    return {"cos_dp": dot / norm if norm else float("nan"), "lambda_eff": scale}


def run_metrics(rec: dict, copy_records: dict | None = None) -> dict:
    """Per-run metrics of E10_SPEC.md; ``copy_records`` maps length -> that checkpoint's copy-evaluation record."""
    out = metrics(rec)
    out["excess_1024"] = excess_nll(rec)
    out["indep_nll_256"] = next(t for t in rec["test"] if t["length"] == INDEP_LENGTH)["all"]["nll"]
    if rec.get("evidence") and rec["evidence"]["kernel"] == "arith":  # the matched cosines have no D_P coefficients
        out.update(kernel_quantities(rec))
    if copy_records:
        out["copy_nll_256"] = copy_records[COPY_LENGTH]["test"][0]["all"]["nll"]
        out["copy_bayes_nll_256"] = copy_records[COPY_LENGTH]["bayes_nll"]["all"]
        out["copy_nll_1024"] = copy_records[1024]["test"][0]["all"]["nll"]
        out["copy_bayes_nll_1024"] = copy_records[1024]["bayes_nll"]["all"]
    return out


def load(suite: str = SUITE, root: Path = ROOT, with_copy: bool = True) -> dict:
    """Runs and copy evaluations of a suite: ``{runs, copies, failed, expected}``; stops on a missing record."""
    runs_dir, copy_dir = root / "results" / "runs" / suite, root / "results" / "runs" / (suite + "_copy_eval")
    runs, copies, failed, expected = {}, {}, [], []
    for spec in SUITES[suite]():
        key = (spec.task["periods"][0], e10_arm(spec), spec.seed)
        expected.append(key)
        if (runs_dir / (spec.filename + ".error.txt")).exists():
            failed.append((*key, "training failed"))
            continue
        if not (runs_dir / spec.filename).exists():
            sys.exit(f"training run missing and not failed: {spec.filename}")
        runs[key] = json.loads((runs_dir / spec.filename).read_text())
        if not with_copy:
            continue
        stem = spec.filename[:-5]
        if (copy_dir / (stem + ".error.txt")).exists():
            failed.append((*key, "copy evaluation failed"))
            continue
        for n in E10_COPY_LENGTHS:
            path = copy_dir / f"{stem}__N{n}.json"
            if not path.exists():
                sys.exit(f"copy-evaluation record missing and not failed: {path.name}")
            rec = json.loads(path.read_text())
            assert rec["length"] == n and rec["source_run"] == spec.filename, path.name
            copies[(*key, n)] = rec
    return {"runs": runs, "copies": copies, "failed": failed, "expected": expected}


def collect(loaded: dict) -> dict:
    """{(period, arm, seed): metrics}; a seed whose copy evaluation failed has no copy metrics."""
    m = {}
    for key, rec in loaded["runs"].items():
        copy = {n: loaded["copies"][(*key, n)] for n in E10_COPY_LENGTHS if (*key, n) in loaded["copies"]}
        m[key] = run_metrics(rec, copy if len(copy) == len(E10_COPY_LENGTHS) else None)
    return m


def checkpoint_hashes(suite: str, loaded: dict, root: Path = ROOT) -> dict:
    """SHA-256 of the checkpoint file of each loaded run, by run file name (None when the file is missing)."""
    out = {}
    for spec in SUITES[suite]():
        key = (spec.task["periods"][0], e10_arm(spec), spec.seed)
        if key in loaded["runs"]:
            path = root / "checkpoints" / suite / (spec.filename[:-5] + ".pt")
            out[spec.filename] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


def sanity(loaded: dict, checkpoint_sha: dict) -> dict:
    """Completed runs, one code commit, one test digest per period and length, checkpoint hashes, all records accounted for."""
    runs, copies = loaded["runs"], loaded["copies"]
    failed = {tuple(f[:3]) for f in loaded["failed"]}
    commits = {r["environment"]["git_commit"] for r in list(runs.values()) + list(copies.values())}
    digests: dict = {}
    for (p, _, _), rec in runs.items():
        for t in rec["test"]:
            digests.setdefault(("run", p, t["length"]), set()).add(t["digest"])
    for (p, _, _, n), rec in copies.items():
        digests.setdefault(("copy", p, n), set()).add(rec["test"][0]["digest"])
    files = set(checkpoint_sha)
    return {"all_records_accounted_for": all(k in runs or k in failed for k in loaded["expected"]),
            "training_runs_completed": all(r["status"] == "COMPLETED" for r in runs.values()),
            "single_code_commit": len(commits) == 1, "code_commits": sorted(c for c in commits if c),
            "one_test_set_per_period_and_length": all(len(v) == 1 for v in digests.values()),
            "copy_checkpoints_match_suite_runs": all(
                r["source_run"] in files and checkpoint_sha[r["source_run"]] == r["checkpoint_sha256"]
                for r in copies.values())}


def paired(m: dict, period: int, arm: str, other: str, key: str, seeds) -> dict:
    """95% interval of ``arm - other`` over the seeds where both runs (and the metric) exist."""
    used = [s for s in seeds if key in m.get((period, arm, s), {}) and key in m.get((period, other, s), {})]
    out = ci95([m[(period, arm, s)][key] - m[(period, other, s)][key] for s in used])
    out["seeds"] = used
    return out


def single(m: dict, period: int, arm: str, key: str, seeds) -> dict:
    used = [s for s in seeds if key in m.get((period, arm, s), {})]
    out = ci95([m[(period, arm, s)][key] for s in used])
    out["seeds"] = used
    return out


def _verdict(failures: list[str]) -> str:
    return "SUPPORTED" if not failures else "FALSIFIED (" + "; ".join(failures) + ")"


def decide(m: dict, seeds=E10_SEEDS) -> dict:
    """Paired statistics and verdicts from {(period, arm, seed): metrics}."""
    cells = {}
    for p in PERIODS:
        cells[p] = {
            "v1_far_nll": paired(m, p, "arith_ev", "arith", "far_nll", seeds),
            "v1_far_acc": paired(m, p, "arith_ev", "arith", "far_acc", seeds),
            "v2_excess_1024": single(m, p, "arith_ev", "excess_1024", seeds),
            "v3_far_nll": paired(m, p, "arith_evfund", "arith_ev", "far_nll", seeds),
            "v3_far_acc": paired(m, p, "arith_ev", "arith_evfund", "far_acc", seeds),
            "v4_far_acc": paired(m, p, "rope_ev", "rope", "far_acc", seeds),
            "v4_far_nll": paired(m, p, "rope_ev", "rope", "far_nll", seeds),
            "b1_copy_nll_256": paired(m, p, "arith_ev", "arith", "copy_nll_256", seeds),
            "s15_indep_nll_256": paired(m, p, "arith_ev", "arith", "indep_nll_256", seeds),
            "means": {arm: {k: sum(m[(p, arm, s)][k] for s in seeds if (p, arm, s) in m) /
                            max(1, sum((p, arm, s) in m for s in seeds))
                            for k in ("far_nll", "far_acc", "excess_1024", "indep_nll_256")}
                      for arm in ARMS},
            "kernel": {arm: {k: [m[(p, arm, s)][k] for s in seeds if (p, arm, s) in m] for k in ("cos_dp", "lambda_eff")}
                       for arm in ("arith_ev", "rope_ev")},
        }
    c12, c13 = cells[12], cells[13]
    v1, v2, v4, b1 = [], [], [], []
    for p, c in cells.items():
        if c["v1_far_nll"]["hi"] >= 0:
            v1.append(f"P = {p}: far NLL interval does not lie below 0")
        if c["v1_far_acc"]["lo"] <= -ACC_MARGIN:
            v1.append(f"P = {p}: accuracy guard fails")
        if c["v2_excess_1024"]["hi"] >= V2_MARGIN:
            v2.append(f"P = {p}: excess NLL upper bound reaches {V2_MARGIN}")
        if c["v4_far_acc"]["lo"] <= V4_ACC:
            v4.append(f"P = {p}: far accuracy lower bound not above +20 pp")
        if c["v4_far_nll"]["hi"] >= 0:
            v4.append(f"P = {p}: far NLL interval does not lie below 0")
        if c["b1_copy_nll_256"]["lo"] <= 0:
            b1.append(f"P = {p}: copy NLL interval reaches 0")
    v3 = []
    if c13["v3_far_nll"]["lo"] <= 0:
        v3.append("P = 13: far NLL interval does not lie above 0")
    if c13["v3_far_acc"]["lo"] <= V3_ACC:
        v3.append("P = 13: far accuracy lower bound not above +10 pp")
    kernel = [v for c in cells.values() for arm in ("arith_ev", "rope_ev") for v in zip(c["kernel"][arm]["cos_dp"],
                                                                                      c["kernel"][arm]["lambda_eff"])]
    out = {"periods": {str(p): c for p, c in cells.items()}, "verdicts": {
        "V1_head_lowers_far_nll": _verdict(v1), "V2_follows_bayes_at_1024": _verdict(v2),
        "V3_exact_basis_matters_at_13": _verdict(v3), "V4_restores_rope_extrapolation": _verdict(v4),
        "B1_copied_observations_harm": _verdict(b1),
        "S12_no_failure_at_12": c12["v3_far_acc"]["lo"] >= -S12_BAND and c12["v3_far_acc"]["hi"] <= S12_BAND,
        "S13_sgd_finds_construction": bool(kernel) and all(c >= S13_COSINE for c, _ in kernel),
        "S14_bayes_scale": bool(kernel) and all(abs(lam / LOG_RHO - 1) <= S14_BAND for _, lam in kernel),
        "S15_crossover": all(c["s15_indep_nll_256"]["hi"] < 0 for c in cells.values())}}
    # Sensitivity only. holm() tests "mean < 0"; every "X > c" test is entered as c - X, every "X < c" as X - c.
    tests = {}
    for p, c in cells.items():
        tests[f"V1_far_nll_p{p}"] = c["v1_far_nll"]["values"]
        tests[f"V1_far_acc_guard_p{p}"] = [-ACC_MARGIN - v for v in c["v1_far_acc"]["values"]]
        tests[f"V2_excess_p{p}"] = [v - V2_MARGIN for v in c["v2_excess_1024"]["values"]]
        tests[f"V4_far_acc_p{p}"] = [V4_ACC - v for v in c["v4_far_acc"]["values"]]
        tests[f"V4_far_nll_p{p}"] = c["v4_far_nll"]["values"]
        tests[f"B1_copy_nll_p{p}"] = [-v for v in c["b1_copy_nll_256"]["values"]]
    tests["V3_far_nll_p13"] = [-v for v in c13["v3_far_nll"]["values"]]
    tests["V3_far_acc_p13"] = [V3_ACC - v for v in c13["v3_far_acc"]["values"]]
    out["holm"] = holm(tests)
    return out


def main() -> None:
    plan_path = ROOT / "results" / "summary" / "e10_power.json"
    plan = json.loads(plan_path.read_text())
    if plan.get("n_seeds") is None or len(E10_SEEDS) != plan["n_seeds"]:
        sys.exit(f"E10_SEEDS has {len(E10_SEEDS)} seeds but e10_power.json plans {plan.get('n_seeds')}")
    loaded = load()
    m = collect(loaded)
    checks = sanity(loaded, checkpoint_hashes(SUITE, loaded))
    out = {"status": "E10 (research/evidence_readout/2026-10-23/E10_SPEC.md)", "length": LENGTH, "far": FAR,
           "planned_seeds": list(E10_SEEDS), "failed": loaded["failed"], "sanity": checks, **decide(m)}
    if not all(v for k, v in checks.items() if k != "code_commits"):
        out["verdicts"] = {"E10": "NOT REPORTED (a sanity check failed)"}
    dest = ROOT / "results" / "summary" / "e10.json"
    dest.write_text(json.dumps(out, indent=1))
    if loaded["failed"]:
        print(f"{len(loaded['failed'])} failed records (reported, not replaced): {loaded['failed']}")
    for name, verdict in out["verdicts"].items():
        print(f"{name}: {verdict}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
