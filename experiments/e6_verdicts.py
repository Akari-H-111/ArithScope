"""E6 verdicts (docs/PREREGISTRATION.md, Section 9), frozen with the preregistration.

    python experiments/e6_verdicts.py

Reads results/runs/e6_calibration/*.json and writes results/summary/e6.json. Every metric comes
from the N = 8,192 test record's position buckets (1,024 positions each), so it covers the same
positions in every arm whatever its training length:

* far NLL / far accuracy: positions 4,096-8,191 (buckets 4-7);
* growth: NLL on positions 7,168-8,191 minus NLL on positions 1,024-2,047.

Paired differences are taken per seed within a period; intervals are 95% t intervals over seeds.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope.train import RunSpec  # noqa: E402
from experiments.suites import e6_arm  # noqa: E402

SUITE = "e6_calibration"
LENGTH = 8192
FAR = (4096, 8192)
EARLY, LATE = (1024, 2048), (7168, 8192)
ACC_MARGIN = 0.01   # L1/L2 accuracy guard: lower CI bound of the paired difference above -1 pp
FLAT = 0.02         # S3: |mean growth| below 0.02 nats counts as flat
PERIODS = (12, 13)
SEEDS = (1, 2, 3, 4, 5)


def ci95(values: list[float]) -> dict:
    from scipy import stats

    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half, "values": values}


def holm(diffs: dict[str, list[float]]) -> dict:
    """Sensitivity only: Holm-adjusted one-sided p-values for "mean difference < 0"."""
    from scipy import stats

    raw = {}
    for name, d in diffs.items():
        c = ci95(d)
        se = (c["hi"] - c["mean"]) / float(stats.t.ppf(0.975, c["n"] - 1))
        raw[name] = float(stats.t.cdf(c["mean"] / se, c["n"] - 1)) if se > 0 else float(c["mean"] >= 0)
    adjusted, running = {}, 0.0
    for i, (name, pv) in enumerate(sorted(raw.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(raw) - i) * pv))
        adjusted[name] = {"p": pv, "p_holm": running}
    return adjusted


def region(run: dict, start: int, end: int, key: str) -> float:
    """n-weighted mean of ``key`` over the N = 8,192 position buckets inside [start, end)."""
    test = next(t for t in run["test"] if t["length"] == LENGTH)
    cells = [b for b in test["by_position"] if start <= b["start"] and b["end"] <= end]
    assert cells and cells[0]["start"] == start and cells[-1]["end"] == end, (start, end)
    n = sum(b["n"] for b in cells)
    return sum(b[key] * b["n"] for b in cells) / n


def metrics(run: dict) -> dict:
    return {"far_nll": region(run, *FAR, "nll"), "far_acc": region(run, *FAR, "acc"),
            "far_oracle_acc": region(run, *FAR, "oracle_acc"),
            "growth": region(run, *LATE, "nll") - region(run, *EARLY, "nll")}


def load() -> dict:
    """{(period, arm, seed): run record}; fails on a missing or duplicate run."""
    runs = {}
    for path in sorted((ROOT / "results" / "runs" / SUITE).glob("*.json")):
        rec = json.loads(path.read_text())
        fields = dict(rec["spec"])
        for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
            fields[key] = tuple(fields[key])
        spec = RunSpec(**fields)
        key = (spec.task["periods"][0], e6_arm(spec), spec.seed)
        assert key not in runs, key
        runs[key] = rec
    return runs


def paired(m: dict, period: int, arm: str, other: str, key: str) -> dict:
    out = ci95([m[(period, arm, s)][key] - m[(period, other, s)][key] for s in SEEDS])
    out["lower"] = sum(v < 0 for v in out["values"])  # seeds where `arm` is lower than `other`
    return out


def training_identical(runs: dict, period: int, seed: int) -> bool:
    """The window never binds at the training length, so window and baseline must train bitwise alike."""
    a, b = runs[(period, "window", seed)], runs[(period, "baseline", seed)]
    return (a["train"]["loss_curve"] == b["train"]["loss_curve"] and a["train"]["val_curve"] == b["train"]["val_curve"]
            and a["positional"] == b["positional"])


def main() -> None:
    runs = load()
    missing = [(p, a, s) for p in PERIODS for a in ("reference", "baseline", "window", "long_train") for s in SEEDS
               if (p, a, s) not in runs]
    if missing:
        sys.exit(f"{len(missing)} runs missing, e.g. {missing[:3]}; a failed run is reported, never replaced")
    m = {k: metrics(r) for k, r in runs.items()}
    out = {"status": "PREREGISTERED (docs/PREREGISTRATION.md, Section 9)", "length": LENGTH, "far": FAR,
           "early": EARLY, "late": LATE, "periods": {}, "verdicts": {}}
    for p in PERIODS:
        cell = {"means": {arm: {k: sum(m[(p, arm, s)][k] for s in SEEDS) / len(SEEDS) for k in m[(p, arm, 1)]}
                          for arm in ("reference", "baseline", "window", "long_train")},
                "sanity_window_training_identical": all(training_identical(runs, p, s) for s in SEEDS)}
        for arm in ("window", "long_train"):
            cell[arm] = {"far_nll_vs_baseline": paired(m, p, arm, "baseline", "far_nll"),
                         "far_acc_vs_baseline": paired(m, p, arm, "baseline", "far_acc"),
                         "growth_vs_baseline": paired(m, p, arm, "baseline", "growth"),
                         "far_nll_vs_reference": paired(m, p, arm, "reference", "far_nll"),
                         "growth": ci95([m[(p, arm, s)]["growth"] for s in SEEDS])}
        cell["baseline_growth"] = ci95([m[(p, "baseline", s)]["growth"] for s in SEEDS])
        out["periods"][str(p)] = cell

    def claim(arm: str) -> str:
        cells = [out["periods"][str(p)][arm] for p in PERIODS]
        if any(c["far_nll_vs_baseline"]["hi"] >= 0 for c in cells):
            return "FALSIFIED (far NLL not lower than baseline at both periods)"
        if any(c["far_acc_vs_baseline"]["lo"] <= -ACC_MARGIN for c in cells):
            return "FALSIFIED (lower far NLL, but the accuracy guard fails)"
        return "SUPPORTED"

    cells = {p: out["periods"][str(p)] for p in PERIODS}
    out["holm"] = holm({f"{arm}_p{p}": cells[p][arm]["far_nll_vs_baseline"]["values"]
                        for arm in ("window", "long_train") for p in PERIODS})
    out["verdicts"] = {
        "L1_window": claim("window"),
        "L2_long_train": claim("long_train"),
        "S3_window_flat": all(abs(c["window"]["growth"]["mean"]) < FLAT for c in cells.values()),
        "S4_window_above_reference": all(c["window"]["far_nll_vs_reference"]["lo"] > 0 for c in cells.values()),
        "S5_long_train_still_grows": all(c["long_train"]["growth"]["mean"] > 0 for c in cells.values()),
        "sanity_window_training_identical": all(c["sanity_window_training_identical"] for c in cells.values()),
    }
    dest = ROOT / "results" / "summary" / "e6.json"
    dest.write_text(json.dumps(out, indent=1))
    for name, verdict in out["verdicts"].items():
        print(f"{name}: {verdict}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
