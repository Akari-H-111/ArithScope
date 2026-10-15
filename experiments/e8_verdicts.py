"""E8 verdicts (docs/PREREGISTRATION.md, Section 11), frozen with the preregistration.

    python experiments/e8_verdicts.py

Reads results/runs/e8_power_retest/*.json and writes results/summary/e8.json. Metrics, regions,
intervals, and the accuracy guard are E6's (experiments/e6_verdicts.py, imported unchanged):

* far NLL / far accuracy: positions 4,096-8,191 of the N = 8,192 test records (buckets 4-7);
* growth: NLL on positions 7,168-8,191 minus NLL on positions 1,024-2,047.

Paired differences are taken per seed within a period; intervals are 95% t intervals over seeds. A run
that failed (an ``.error.txt`` file in place of its record) is reported and never replaced: its seed is
left out of that period's paired analysis. A run that is neither recorded nor failed stops the script.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope.train import RunSpec  # noqa: E402
from experiments.e6_verdicts import ACC_MARGIN, EARLY, FAR, FLAT, LATE, LENGTH, PERIODS, ci95, holm, metrics  # noqa: E402
from experiments.suites import E8_ARMS, E8_SEEDS, e8_arm, e8_power_retest  # noqa: E402

SUITE = "e8_power_retest"
ARMS = tuple(E8_ARMS)  # reference, arith_window
SPREAD_RATIO = 3.0     # S8: seed-to-seed sd of far NLL, reference / arith_window


def load() -> tuple[dict, list]:
    """({(period, arm, seed): run record}, [(period, arm, seed) of failed runs]); fails on a missing or duplicate run."""
    runs, failed = {}, []
    directory = ROOT / "results" / "runs" / SUITE
    for path in sorted(directory.glob("*.json")):
        rec = json.loads(path.read_text())
        fields = dict(rec["spec"])
        for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
            fields[key] = tuple(fields[key])
        spec = RunSpec(**fields)
        key = (spec.task["periods"][0], e8_arm(spec), spec.seed)
        assert key not in runs, key
        runs[key] = rec
    for spec in e8_power_retest():
        key = (spec.task["periods"][0], e8_arm(spec), spec.seed)
        if key not in runs:
            if not (directory / (spec.filename + ".error.txt")).exists():
                sys.exit(f"run missing and not failed: {spec.filename}; finish the suite first")
            failed.append(key)
    return runs, failed


def paired(m: dict, period: int, seeds: list[int], arm: str, other: str, key: str) -> dict:
    out = ci95([m[(period, arm, s)][key] - m[(period, other, s)][key] for s in seeds])
    out["lower"] = sum(v < 0 for v in out["values"])  # seeds where `arm` is lower than `other`
    out["seeds"] = seeds
    return out


def sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def training_identical(runs: dict, period: int, seed: int) -> bool:
    """The window never binds at the training length, so arith_window and reference must train bitwise alike."""
    a, b = runs[(period, "arith_window", seed)], runs[(period, "reference", seed)]
    return (a["train"]["loss_curve"] == b["train"]["loss_curve"] and a["train"]["val_curve"] == b["train"]["val_curve"]
            and a["positional"] == b["positional"])


def main() -> None:
    plan = json.loads((ROOT / "results" / "summary" / "e8_power.json").read_text())
    assert len(E8_SEEDS) == plan["n_seeds"], (len(E8_SEEDS), plan["n_seeds"])
    runs, failed = load()
    m = {k: metrics(r) for k, r in runs.items()}
    out = {"status": "PREREGISTERED (docs/PREREGISTRATION.md, Section 11)", "length": LENGTH, "far": FAR,
           "early": EARLY, "late": LATE, "planned_seeds": list(E8_SEEDS), "failed_runs": failed,
           "periods": {}, "verdicts": {}}
    for p in PERIODS:
        seeds = [s for s in E8_SEEDS if all((p, a, s) in runs for a in ARMS)]
        cell = {"seeds": seeds,
                "means": {arm: {k: sum(m[(p, arm, s)][k] for s in seeds) / len(seeds) for k in m[(p, arm, seeds[0])]}
                          for arm in ARMS},
                "sanity_arith_window_training_identical": all(training_identical(runs, p, s) for s in seeds),
                "arith_window_vs_reference": {key: paired(m, p, seeds, "arith_window", "reference", key)
                                              for key in ("far_nll", "far_acc", "growth")},
                "far_nll": {arm: ci95([m[(p, arm, s)]["far_nll"] for s in seeds]) for arm in ARMS},
                "far_nll_sd": {arm: sd([m[(p, arm, s)]["far_nll"] for s in seeds]) for arm in ARMS},
                "growth": {arm: ci95([m[(p, arm, s)]["growth"] for s in seeds]) for arm in ARMS}}
        out["periods"][str(p)] = cell

    cells = [out["periods"][str(p)] for p in PERIODS]
    if any(c["arith_window_vs_reference"]["far_nll"]["hi"] >= 0 for c in cells):
        w2 = "FALSIFIED (far NLL not lower than ArithScope alone at both periods)"
    elif any(c["arith_window_vs_reference"]["far_acc"]["lo"] <= -ACC_MARGIN for c in cells):
        w2 = "FALSIFIED (lower far NLL, but the accuracy guard fails)"
    else:
        w2 = "SUPPORTED"
    out["holm"] = holm({f"arith_window_p{p}": out["periods"][str(p)]["arith_window_vs_reference"]["far_nll"]["values"]
                        for p in PERIODS})
    out["verdicts"] = {
        "W2_arith_window_below_reference": w2,
        "S8_spread_in_reference": all(c["far_nll_sd"]["reference"] >= SPREAD_RATIO * c["far_nll_sd"]["arith_window"]
                                      for c in cells),
        "S9_arith_window_flat": all(abs(c["growth"]["arith_window"]["mean"]) < FLAT for c in cells),
        "sanity_arith_window_training_identical": all(c["sanity_arith_window_training_identical"] for c in cells),
    }
    dest = ROOT / "results" / "summary" / "e8.json"
    dest.write_text(json.dumps(out, indent=1))
    if failed:
        print(f"{len(failed)} failed runs (reported, not replaced): {failed}")
    for name, verdict in out["verdicts"].items():
        print(f"{name}: {verdict}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
