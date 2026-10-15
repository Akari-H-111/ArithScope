"""E8 seed count (docs/PREREGISTRATION.md, Section 11), computed before any E8 run.

    python experiments/e8_power.py

Reads the frozen E6 and E7 summaries and writes results/summary/e8_power.json. E8 repeats E7's claim
W1 (far NLL of `arith_window` minus `reference`, decided by a 95% t interval over seeds at both
periods) with more seeds. The rule, fixed here before any E8 run:

* **Spread.** The per-seed paired differences that measure the seed-to-seed spread of ArithScope
  alone's far-NLL floor: E7's `arith_window - reference` (the W1 quantity itself) and E6's
  `window - reference` (RoPE + ArithScope with the same window, which E7 found within about 0.005 nats of
  `arith_window`; in E7 the two differences have nearly equal spread because the `reference` floor
  dominates both). Per period, the two sample variances are pooled (8 degrees of freedom) and the
  pooled standard deviation is raised to its one-sided 80% upper confidence limit (Browne 1995), so that
  a pilot spread that happened to be small does not shrink the plan.
* **Effect.** Per period, the smaller of the two mean differences in absolute value.
* **Power.** For n seeds, the probability that the upper bound of the 95% t interval lies below 0 is the
  noncentral t probability P(T' < -t_{0.975, n-1}) with T' ~ t(n - 1, -d sqrt(n)), d = effect / spread,
  under a normal model of the paired differences. The two periods are independent models, so the joint
  probability is the product. **n is the smallest number of seeds with joint power at least 0.90.**

The accuracy guard is not part of the plan: far accuracy was 100% in every arm and seed of E6 and E7
(every paired accuracy difference was exactly 0). A sensitivity table for other choices is written next
to the plan; it informs no decision.
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
TARGET = 0.90       # joint power over both periods
UCL = 0.80          # one-sided confidence level of the upper limit on the spread
ALPHA = 0.025       # one side of the 95% t interval
N_MAX = 200


def sd(values: list[float]) -> float:
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def power(d: float, n: int) -> float:
    """P(upper bound of the 95% t interval < 0) for n paired differences with mean -d sigma."""
    return float(stats.nct.cdf(-stats.t.ppf(1 - ALPHA, n - 1), n - 1, -d * math.sqrt(n)))


def smallest_n(ds: list[float], target: float = TARGET) -> int:
    for n in range(3, N_MAX + 1):
        if math.prod(power(d, n) for d in ds) >= target:
            return n
    raise ValueError("no n up to N_MAX reaches the target")


def main() -> None:
    e6 = json.loads((ROOT / "results" / "summary" / "e6.json").read_text())
    e7 = json.loads((ROOT / "results" / "summary" / "e7.json").read_text())
    out = {"status": "PREREGISTERED (docs/PREREGISTRATION.md, Section 11)", "target_joint_power": TARGET,
           "ucl": UCL, "alpha_one_sided": ALPHA, "periods": {}}
    plan_d = []
    for p in PERIODS:
        direct = e7["periods"][str(p)]["arith_window_vs_reference"]["far_nll"]["values"]
        proxy = e6["periods"][str(p)]["window"]["far_nll_vs_reference"]["values"]
        df = len(direct) - 1 + len(proxy) - 1
        pooled = math.sqrt(((len(direct) - 1) * sd(direct) ** 2 + (len(proxy) - 1) * sd(proxy) ** 2) / df)
        spread = pooled * math.sqrt(df / stats.chi2.ppf(1 - UCL, df))
        effect = min(abs(sum(direct) / len(direct)), abs(sum(proxy) / len(proxy)))
        d = effect / spread
        plan_d.append(d)
        out["periods"][str(p)] = {
            "e7_arith_window_minus_reference": {"mean": sum(direct) / len(direct), "sd": sd(direct), "values": direct},
            "e6_window_minus_reference": {"mean": sum(proxy) / len(proxy), "sd": sd(proxy), "values": proxy},
            "pooled_sd": pooled, "df": df, "spread_ucl": spread, "effect": effect, "d": d}
    n = smallest_n(plan_d)
    out["n_seeds"] = n
    out["power_at_n"] = {str(p): power(d, n) for p, d in zip(PERIODS, plan_d)}
    out["joint_power_at_n"] = math.prod(power(d, n) for d in plan_d)

    # Sensitivity only (no decision depends on it): n under other planning choices.
    cells = out["periods"]
    alt = {  # name: (d per period, target joint power)
        "e7_only_point": ([abs(cells[str(p)]["e7_arith_window_minus_reference"]["mean"])
                           / cells[str(p)]["e7_arith_window_minus_reference"]["sd"] for p in PERIODS], TARGET),
        "pooled_point_sd": ([cells[str(p)]["effect"] / cells[str(p)]["pooled_sd"] for p in PERIODS], TARGET),
        "plan_target_0.80": (plan_d, 0.80),
        "plan_effect_halved": ([d / 2 for d in plan_d], TARGET),
    }
    out["sensitivity"] = {name: {"d": ds, "target": target, "n": smallest_n(ds, target)}
                          for name, (ds, target) in alt.items()}
    out["sensitivity"]["e7_design_n5_joint_power"] = math.prod(power(d, 5) for d in plan_d)

    dest = ROOT / "results" / "summary" / "e8_power.json"
    dest.write_text(json.dumps(out, indent=1))
    for p, d in zip(PERIODS, plan_d):
        c = cells[str(p)]
        print(f"P = {p}: effect {c['effect']:.4f}, pooled sd {c['pooled_sd']:.4f} (df {c['df']}), "
              f"80% upper limit {c['spread_ucl']:.4f}, d = {d:.3f}, power at n = {n}: {out['power_at_n'][str(p)]:.3f}")
    print(f"n = {n} seeds per arm and period (joint power {out['joint_power_at_n']:.3f}; "
          f"E7's 5 seeds: {out['sensitivity']['e7_design_n5_joint_power']:.3f})")
    for name, s in out["sensitivity"].items():
        if isinstance(s, dict):
            print(f"  sensitivity {name}: d = {[round(x, 3) for x in s['d']]}, n = {s['n']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
