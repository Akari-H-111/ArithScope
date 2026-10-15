"""E7 verdicts (docs/PREREGISTRATION.md, Section 10), frozen with the preregistration.

    python experiments/e7_verdicts.py

Reads results/runs/e7_floor/*.json and writes results/summary/e7.json. Metrics, regions, intervals,
and the accuracy guard are E6's (experiments/e6_verdicts.py, imported unchanged):

* far NLL / far accuracy: positions 4,096-8,191 of the N = 8,192 test records (buckets 4-7);
* growth: NLL on positions 7,168-8,191 minus NLL on positions 1,024-2,047.

Paired differences are taken per seed within a period; intervals are 95% t intervals over seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope.train import RunSpec  # noqa: E402
from experiments.e6_verdicts import (ACC_MARGIN, EARLY, FAR, LATE, LENGTH, PERIODS, SEEDS,  # noqa: E402
                                     ci95, holm, metrics)
from experiments.suites import E7_ARMS, e7_arm  # noqa: E402

SUITE = "e7_floor"
ARMS = tuple(E7_ARMS)  # reference, arith_window, window


def load() -> dict:
    """{(period, arm, seed): run record}; fails on a missing or duplicate run."""
    runs = {}
    for path in sorted((ROOT / "results" / "runs" / SUITE).glob("*.json")):
        rec = json.loads(path.read_text())
        fields = dict(rec["spec"])
        for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
            fields[key] = tuple(fields[key])
        spec = RunSpec(**fields)
        key = (spec.task["periods"][0], e7_arm(spec), spec.seed)
        assert key not in runs, key
        runs[key] = rec
    return runs


def paired(m: dict, period: int, arm: str, other: str, key: str) -> dict:
    out = ci95([m[(period, arm, s)][key] - m[(period, other, s)][key] for s in SEEDS])
    out["lower"] = sum(v < 0 for v in out["values"])  # seeds where `arm` is lower than `other`
    return out


def training_identical(runs: dict, period: int, seed: int) -> bool:
    """The window never binds at the training length, so arith_window and reference must train bitwise alike."""
    a, b = runs[(period, "arith_window", seed)], runs[(period, "reference", seed)]
    return (a["train"]["loss_curve"] == b["train"]["loss_curve"] and a["train"]["val_curve"] == b["train"]["val_curve"]
            and a["positional"] == b["positional"])


def main() -> None:
    runs = load()
    missing = [(p, a, s) for p in PERIODS for a in ARMS for s in SEEDS if (p, a, s) not in runs]
    if missing:
        sys.exit(f"{len(missing)} runs missing, e.g. {missing[:3]}; a failed run is reported, never replaced")
    m = {k: metrics(r) for k, r in runs.items()}
    out = {"status": "PREREGISTERED (docs/PREREGISTRATION.md, Section 10)", "length": LENGTH, "far": FAR,
           "early": EARLY, "late": LATE, "periods": {}, "verdicts": {}}
    for p in PERIODS:
        cell = {"means": {arm: {k: sum(m[(p, arm, s)][k] for s in SEEDS) / len(SEEDS) for k in m[(p, arm, 1)]}
                          for arm in ARMS},
                "sanity_arith_window_training_identical": all(training_identical(runs, p, s) for s in SEEDS),
                "arith_window_vs_reference": {key: paired(m, p, "arith_window", "reference", key)
                                              for key in ("far_nll", "far_acc", "growth")},
                "arith_window_vs_window": {"far_nll": paired(m, p, "arith_window", "window", "far_nll")},
                "window_vs_reference": {"far_nll": paired(m, p, "window", "reference", "far_nll")},
                "growth": {arm: ci95([m[(p, arm, s)]["growth"] for s in SEEDS]) for arm in ARMS}}
        # Descriptive only: the share of the reference-to-window far-NLL gap that the window on ArithScope
        # alone closes, from the period means (no decision depends on it).
        means = cell["means"]
        gap = means["reference"]["far_nll"] - means["window"]["far_nll"]
        cell["gap_closed_share"] = (means["reference"]["far_nll"] - means["arith_window"]["far_nll"]) / gap \
            if gap != 0 else None
        out["periods"][str(p)] = cell

    cells = [out["periods"][str(p)] for p in PERIODS]
    if any(c["arith_window_vs_reference"]["far_nll"]["hi"] >= 0 for c in cells):
        w1 = "FALSIFIED (far NLL not lower than ArithScope alone at both periods)"
    elif any(c["arith_window_vs_reference"]["far_acc"]["lo"] <= -ACC_MARGIN for c in cells):
        w1 = "FALSIFIED (lower far NLL, but the accuracy guard fails)"
    else:
        w1 = "SUPPORTED"
    out["holm"] = holm({f"arith_window_p{p}": out["periods"][str(p)]["arith_window_vs_reference"]["far_nll"]["values"]
                        for p in PERIODS})
    out["verdicts"] = {
        "W1_arith_window_below_reference": w1,
        "S6_arith_window_above_window": all(c["arith_window_vs_window"]["far_nll"]["lo"] > 0 for c in cells),
        "S7_window_below_reference": all(c["window_vs_reference"]["far_nll"]["hi"] < 0 for c in cells),
        "sanity_arith_window_training_identical": all(c["sanity_arith_window_training_identical"] for c in cells),
    }
    dest = ROOT / "results" / "summary" / "e7.json"
    dest.write_text(json.dumps(out, indent=1))
    for name, verdict in out["verdicts"].items():
        print(f"{name}: {verdict}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
