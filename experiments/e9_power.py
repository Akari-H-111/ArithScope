"""E9 seed count (docs/PREREGISTRATION.md, Section 12), from E8's results, before any E9 run.

    python experiments/e9_power.py

This rule was written before E8 finished, and is run once E8's summary exists. It reads
results/summary/e8.json and writes results/summary/e9_power.json.

* **Input.** E8's per-seed paired differences D = far NLL(window 64) - far NLL(no window) of ArithScope
  alone, per period (`arith_window_vs_reference.far_nll.values`).
* **Planning model.** Within a seed, the four windows share the weights and the test data, so most of a
  seed's variation is the unwindowed floor, which every window above 64 shares. The plan takes the
  per-seed slope over log2(W) to be c * (-D) with c > 0 the same for every seed. Then the standardized
  effect of the slope equals that of D, whatever c is, and the shape of the curve between the end points
  does not enter. Noise from the intermediate windows is not modelled; this is the plan's main
  simplification, and the reason for the margin below.
* **Spread and effect.** Per period, the standard deviation of D raised to its one-sided 80% upper
  confidence limit (as in E8), and the absolute mean of D. If the mean of D is not negative at a period,
  there is no effect to plan for and the script stops.
* **Power.** The probability that the 95% t interval of the mean slope lies above 0 is the noncentral t
  probability P(T' > t_{0.975, n-1}) with T' ~ t(n - 1, d sqrt(n)). The two periods are independent
  models, so the joint power is the product. **n is the smallest number of seeds, at least 5, with joint
  power at least 0.90.**
* **Budget.** If n exceeds 40 seeds per period (80 trainings, about E8's 84), the script records that and
  sets no seed count; the design then needs a decision before any preregistration.

The accuracy guard is not part of the plan (far accuracy was 100% in every arm and seed of E6 and E7).
A sensitivity table for other choices is written next to the plan; it informs no decision.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PERIODS = (12, 13)
TARGET = 0.90
UCL = 0.80
ALPHA = 0.025  # one side of the 95% t interval
N_MIN, N_CAP = 5, 40


def sd(values: list[float]) -> float:
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def power(d: float, n: int) -> float:
    """P(lower bound of the 95% t interval > 0) for n paired values with mean d sigma, d > 0."""
    return float(stats.nct.sf(stats.t.ppf(1 - ALPHA, n - 1), n - 1, d * math.sqrt(n)))


def smallest_n(ds: list[float], target: float = TARGET, cap: int = N_CAP) -> int | None:
    for n in range(N_MIN, cap + 1):
        if math.prod(power(d, n) for d in ds) >= target:
            return n
    return None


def plan(diffs: dict[int, list[float]]) -> dict:
    """The plan from E8's per-seed differences {period: [D_s]}."""
    out = {"status": "PREREGISTERED (docs/PREREGISTRATION.md, Section 12)", "target_joint_power": TARGET,
           "ucl": UCL, "alpha_one_sided": ALPHA, "n_min": N_MIN, "n_cap": N_CAP, "periods": {}}
    ds = []
    for p in PERIODS:
        values = diffs[p]
        mean, s = sum(values) / len(values), sd(values)
        if mean >= 0:
            out["n_seeds"] = None
            out["stop"] = f"no reduction to plan for at P = {p} (mean D = {mean:+.4f})"
            return out
        df = len(values) - 1
        spread = s * math.sqrt(df / stats.chi2.ppf(1 - UCL, df))
        d = -mean / spread
        ds.append(d)
        out["periods"][str(p)] = {"n_e8": len(values), "mean_d": mean, "sd_d": s, "spread_ucl": spread, "d": d}
    n = smallest_n(ds)
    out["n_seeds"] = n
    if n is None:
        out["stop"] = f"more than {N_CAP} seeds per period needed; decide before preregistering"
    else:
        out["power_at_n"] = {str(p): power(d, n) for p, d in zip(PERIODS, ds)}
        out["joint_power_at_n"] = math.prod(power(d, n) for d in ds)
    point = [-out["periods"][str(p)]["mean_d"] / out["periods"][str(p)]["sd_d"] for p in PERIODS]
    out["sensitivity"] = {
        "point_sd": smallest_n(point, cap=500),
        "target_0.80": smallest_n(ds, 0.80, cap=500),
        "effect_halved": smallest_n([d / 2 for d in ds], cap=500),
    }
    return out


def main() -> None:
    e8 = json.loads((ROOT / "results" / "summary" / "e8.json").read_text())
    diffs = {p: e8["periods"][str(p)]["arith_window_vs_reference"]["far_nll"]["values"] for p in PERIODS}
    out = plan(diffs)
    dest = ROOT / "results" / "summary" / "e9_power.json"
    dest.write_text(json.dumps(out, indent=1))
    for p, c in out["periods"].items():
        print(f"P = {p}: E8 mean D {c['mean_d']:+.4f}, sd {c['sd_d']:.4f}, 80% upper limit {c['spread_ucl']:.4f}, "
              f"d = {c['d']:.3f}")
    if out["n_seeds"] is None:
        print(f"STOP: {out['stop']}")
    else:
        print(f"n = {out['n_seeds']} seeds per period (joint power {out['joint_power_at_n']:.3f})")
    print(f"sensitivity (n): {out['sensitivity']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
