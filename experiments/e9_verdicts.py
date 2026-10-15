"""E9 verdicts (docs/PREREGISTRATION.md, Section 12), frozen with the preregistration.

    python experiments/e9_verdicts.py

Reads the window-evaluation records in results/runs/e9_paired_window_eval/ (one per checkpoint and
window, written by experiments/e9_window_eval.py) and writes results/summary/e9.json. Far NLL and far
accuracy are E6's (positions 4,096-8,191 of the N = 8,192 test, from experiments/e6_verdicts.py,
imported unchanged). Statistics follow the repository's convention: 95% t intervals over seeds, two-sided.

* R1 (claim): per seed, the least-squares slope of far NLL on log2(W) over the windows. Supported if the
  95% interval of the mean slope lies above 0 at both periods, and the 95% lower bound of far accuracy
  (W = 64 minus W = 8,192) is above -1 pp at both periods.
* S10 (secondary): adjacent windows, near-monotone. Each one-sided lower bound at level 1 - 0.05/6 (three
  steps, two periods) of far NLL(next W) - far NLL(previous W) is above -0.005 nats.
* S11 (secondary): far NLL at W = 64 minus W = 8,192 has a 95% interval below 0 at both periods.

A seed whose training run or window evaluation failed (an ``.error.txt`` file) is reported and left out of
that period's analysis, never replaced. A record that is neither present nor failed stops the script.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.e6_verdicts import ACC_MARGIN, FAR, LENGTH, PERIODS, ci95, holm, metrics  # noqa: E402
from experiments.suites import E9_LENGTH, E9_SEEDS, E9_WINDOWS, e9_paired_window  # noqa: E402

SUITE, EVAL_SUITE = "e9_paired_window", "e9_paired_window_eval"
MONOTONE_MARGIN = 0.005  # S10: nats of decrease tolerated per step
S10_ALPHA = 0.05 / 6     # three adjacent steps x two periods


def slope(ys: list[float], windows: tuple[int, ...] = E9_WINDOWS) -> float:
    """Least-squares slope of ys on log2(window)."""
    xs = [math.log2(w) for w in windows]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def lower_one_sided(values: list[float], alpha: float) -> float:
    from scipy import stats

    n = len(values)
    mean = sum(values) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)
    return mean - float(stats.t.ppf(1 - alpha, n - 1)) * se


def load(root: Path = ROOT) -> tuple[dict, list, dict]:
    """({(period, seed, window): record}, [(period, seed, why) failed], {(period, seed): training run})."""
    assert E9_LENGTH == LENGTH, (E9_LENGTH, LENGTH)
    runs_dir, eval_dir = root / "results" / "runs" / SUITE, root / "results" / "runs" / EVAL_SUITE
    records, failed, training = {}, [], {}
    for spec in e9_paired_window():
        p, s, stem = spec.task["periods"][0], spec.seed, spec.filename[:-5]
        if (runs_dir / (stem + ".json.error.txt")).exists():
            failed.append((p, s, "training failed"))
            continue
        if not (runs_dir / spec.filename).exists():
            sys.exit(f"training run missing and not failed: {spec.filename}")
        training[(p, s)] = json.loads((runs_dir / spec.filename).read_text())
        if (eval_dir / (stem + ".error.txt")).exists():
            failed.append((p, s, "window evaluation failed"))
            continue
        for w in E9_WINDOWS:
            path = eval_dir / f"{stem}__W{w}.json"
            if not path.exists():
                sys.exit(f"window record missing and not failed: {path.name}")
            rec = json.loads(path.read_text())
            assert rec["window"] == w and rec["source_run"] == spec.filename, path.name
            records[(p, s, w)] = rec
    return records, failed, training


def sanity(records: dict, training: dict, seeds_by_period: dict) -> dict:
    """Same weights across windows, one test set per period, one code commit, completed training runs."""
    same_weights = all(len({records[(p, s, w)]["checkpoint_sha256"] for w in E9_WINDOWS}) == 1
                       for p, seeds in seeds_by_period.items() for s in seeds)
    one_test_set = all(len({records[(p, s, w)]["test"][0]["digest"] for s in seeds for w in E9_WINDOWS}) == 1
                       for p, seeds in seeds_by_period.items())
    commits = {rec["environment"]["git_commit"] for rec in records.values()}
    completed = all(training[(p, s)]["status"] == "COMPLETED" for p, seeds in seeds_by_period.items() for s in seeds)
    return {"same_weights_across_windows": same_weights, "one_test_set_per_period": one_test_set,
            "single_code_commit": len(commits) == 1, "code_commits": sorted(c for c in commits if c),
            "training_runs_completed": completed}


def decide(m: dict, seeds_by_period: dict) -> dict:
    """Paired statistics and verdicts from {(period, seed, window): metrics}."""
    out = {"periods": {}}
    for p, seeds in seeds_by_period.items():
        slopes = [slope([m[(p, s, w)]["far_nll"] for w in E9_WINDOWS]) for s in seeds]
        acc = [m[(p, s, E9_WINDOWS[0])]["far_acc"] - m[(p, s, E9_WINDOWS[-1])]["far_acc"] for s in seeds]
        ends = [m[(p, s, E9_WINDOWS[0])]["far_nll"] - m[(p, s, E9_WINDOWS[-1])]["far_nll"] for s in seeds]
        steps = [[m[(p, s, b)]["far_nll"] - m[(p, s, a)]["far_nll"] for s in seeds]
                 for a, b in zip(E9_WINDOWS[:-1], E9_WINDOWS[1:])]
        out["periods"][str(p)] = {
            "seeds": list(seeds),
            "far_nll_by_window": {str(w): ci95([m[(p, s, w)]["far_nll"] for s in seeds]) for w in E9_WINDOWS},
            "far_acc_by_window": {str(w): ci95([m[(p, s, w)]["far_acc"] for s in seeds]) for w in E9_WINDOWS},
            "slope": ci95(slopes),
            "acc_w64_minus_full": ci95(acc),
            "nll_w64_minus_full": ci95(ends),
            "steps": [{"from": a, "to": b, **ci95(d), "lower_s10": lower_one_sided(d, S10_ALPHA)}
                      for (a, b), d in zip(zip(E9_WINDOWS[:-1], E9_WINDOWS[1:]), steps)],
        }
    cells = list(out["periods"].values())
    if any(c["slope"]["lo"] <= 0 for c in cells):
        r1 = "FALSIFIED (the slope interval reaches 0 at a period)"
    elif any(c["acc_w64_minus_full"]["lo"] <= -ACC_MARGIN for c in cells):
        r1 = "FALSIFIED (positive slope, but the accuracy guard fails)"
    else:
        r1 = "SUPPORTED"
    out["verdicts"] = {
        "R1_window_dose_response": r1,
        "S10_near_monotone": all(st["lower_s10"] > -MONOTONE_MARGIN for c in cells for st in c["steps"]),
        "S11_w64_below_full": all(c["nll_w64_minus_full"]["hi"] < 0 for c in cells),
    }
    # Sensitivity only: Holm over the two one-sided slope tests ("slope > 0"; holm() tests "< 0", so negate).
    out["holm"] = holm({f"slope_p{p}": [-v for v in c["slope"]["values"]] for p, c in out["periods"].items()})
    return out


def main() -> None:
    plan = json.loads((ROOT / "results" / "summary" / "e9_power.json").read_text())
    if len(E9_SEEDS) != plan["n_seeds"]:
        sys.exit(f"E9_SEEDS has {len(E9_SEEDS)} seeds but e9_power.json plans {plan['n_seeds']}")
    records, failed, training = load()
    seeds_by_period = {p: [s for s in E9_SEEDS if all((p, s, w) in records for w in E9_WINDOWS)] for p in PERIODS}
    m = {k: metrics(r) for k, r in records.items()}
    out = {"status": "PREREGISTERED (docs/PREREGISTRATION.md, Section 12)", "length": LENGTH, "far": FAR,
           "windows": list(E9_WINDOWS), "planned_seeds": list(E9_SEEDS), "failed": failed,
           "sanity": sanity(records, training, seeds_by_period), **decide(m, seeds_by_period)}
    if not all(v for k, v in out["sanity"].items() if k != "code_commits"):
        out["verdicts"] = {"R1_window_dose_response": "NOT REPORTED (a sanity check failed; see Section 12.8)"}
    dest = ROOT / "results" / "summary" / "e9.json"
    dest.write_text(json.dumps(out, indent=1))
    if failed:
        print(f"{len(failed)} failed seeds (reported, not replaced): {failed}")
    for name, verdict in out["verdicts"].items():
        print(f"{name}: {verdict}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
