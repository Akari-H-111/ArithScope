"""E10 seed count (research/evidence_readout/2026-10-23/E10_SPEC.md), fixed before the spread pilot is run.

    python experiments/e10_power.py

Reads results/summary/e8.json and e9.json (per-seed far NLL of ArithScope alone), the spread-pilot run
records (results/runs/pilot_e10_spread/) and their copy evaluations (results/runs/pilot_e10_spread_copy_eval/),
and writes results/summary/e10_power.json. For each claim, a standardized effect d per period part; the
power of one part with n seeds is the noncentral t probability P(T' > t_{0.975, n-1}), T' ~ t(n-1, d sqrt(n))
(``e9_power.power``, unchanged); the joint power of a claim is the product over its parts. The spread is the
sample SD raised to its one-sided 80% upper confidence limit, s sqrt(df / chi2_{0.20, df}).

* V1 (far-NLL part only; the accuracy guard is not planned): D = -(far NLL of ArithScope alone), the 21 seeds of
  E8's ``reference`` and the 15 seeds of E9's W = 8,192 per period (confirmatory data of the same setup);
  d = |mean D| / spread. Assumption check: in the spread pilot ``arith_ev``'s far NLL is <= 0.001 in every
  seed and period; otherwise stop.
* V2: excess NLL at 1,024 of ``arith_ev`` in the spread pilot; d = (0.005 - mean) / spread; stop if mean >= 0.005.
* V3 (P = 13): far NLL (``arith_evfund - arith_ev``), d = mean / spread; far accuracy (``arith_ev - arith_evfund``),
  d = (mean - 0.10) / spread; stop if a mean is on the wrong side.
* V4 (both periods): far accuracy (``rope_ev - rope``), d = (mean - 0.20) / spread; far NLL (``rope_ev - rope``),
  d = -mean / spread.
* B1 (both periods): copy NLL at 256 (``arith_ev - arith``), d = mean / spread; stop if mean <= 0.
* n = the maximum over claims of the smallest n >= 5 with joint power >= 0.90. If a claim needs more than 20
  seeds per period, that is recorded and no seed count is set (the design then needs a decision).

A sensitivity table (point SD instead of the upper limit, target power 0.80, effects halved) is written next
to the plan; no decision uses it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.e9_power import ALPHA, N_MIN, TARGET, UCL, power, sd  # noqa: E402
from experiments.e10_verdicts import (V2_MARGIN, V3_ACC, V4_ACC, collect, load, paired, single)  # noqa: E402
from experiments.suites import E10_SPREAD_SEEDS  # noqa: E402

PERIODS = (12, 13)
N_CAP = 20
N_SENSITIVITY_CAP = 500
ASSUMPTION_FAR_NLL = 0.001


def joint_power(ds: list[float], n: int) -> float:
    return math.prod(1.0 if math.isinf(d) and d > 0 else power(d, n) for d in ds)


def smallest_n(ds: list[float], target: float = TARGET, cap: int = N_CAP) -> int | None:
    for n in range(N_MIN, cap + 1):
        if joint_power(ds, n) >= target:
            return n
    return None


def spread_ucl(values: list[float]) -> float:
    """Sample SD raised to its one-sided 80% upper confidence limit."""
    df = len(values) - 1
    return sd(values) * math.sqrt(df / stats.chi2.ppf(1 - UCL, df))


def part(name: str, period: int, values: list[float], sign: int, offset: float, absolute: bool = False) -> dict:
    """One standardized effect: d = sign * (mean - offset) / spread (|mean| / spread for ``absolute``)."""
    mean, s, spread = sum(values) / len(values), sd(values), spread_ucl(values)
    effect = abs(mean) if absolute else sign * (mean - offset)

    def ratio(denominator: float) -> float:
        return effect / denominator if denominator > 0 else (math.inf if effect > 0 else -math.inf)

    return {"name": name, "period": period, "n_values": len(values), "mean": mean, "sd": s, "spread_ucl": spread,
            "d": ratio(spread), "d_point_sd": ratio(s),
            "stop": None if effect > 0 else f"{name} at P = {period}: mean {mean:+.5f} is on the wrong side of {offset:g}"}


def claim_parts(far_alone: dict, m: dict, seeds) -> dict:
    """The parts of each claim from the all-period inputs: far NLL of ArithScope alone, and the spread-pilot metrics."""
    claims: dict = {"V1": [], "V2": [], "V3": [], "V4": [], "B1": []}
    for p in PERIODS:
        claims["V1"].append(part("far NLL of ArithScope alone (D = -far NLL)", p, [-v for v in far_alone[p]], -1, 0.0,
                                 absolute=True))
        claims["V2"].append(part("arith_ev excess NLL at 1,024", p, single(m, p, "arith_ev", "excess_1024", seeds)["values"],
                                 -1, V2_MARGIN))
        claims["V4"].append(part("far accuracy rope_ev - rope", p,
                                 paired(m, p, "rope_ev", "rope", "far_acc", seeds)["values"], 1, V4_ACC))
        claims["V4"].append(part("far NLL rope_ev - rope", p,
                                 paired(m, p, "rope_ev", "rope", "far_nll", seeds)["values"], -1, 0.0))
        claims["B1"].append(part("copy NLL at 256 arith_ev - arith", p,
                                 paired(m, p, "arith_ev", "arith", "copy_nll_256", seeds)["values"], 1, 0.0))
    claims["V3"].append(part("far NLL arith_evfund - arith_ev", 13,
                             paired(m, 13, "arith_evfund", "arith_ev", "far_nll", seeds)["values"], 1, 0.0))
    claims["V3"].append(part("far accuracy arith_ev - arith_evfund", 13,
                             paired(m, 13, "arith_ev", "arith_evfund", "far_acc", seeds)["values"], 1, V3_ACC))
    return claims


def plan(far_alone: dict, m: dict, seeds=E10_SPREAD_SEEDS) -> dict:
    """The plan from {period: far NLL of ArithScope alone per seed} and the spread-pilot metrics."""
    out = {"status": "E10 seed-count plan (research/evidence_readout/2026-10-23/E10_SPEC.md)",
           "target_joint_power": TARGET, "ucl": UCL, "alpha_one_sided": ALPHA, "n_min": N_MIN, "n_cap": N_CAP,
           "spread_seeds": list(seeds), "stop_reasons": []}
    far_ev = {p: [m[(p, "arith_ev", s)]["far_nll"] for s in seeds if (p, "arith_ev", s) in m] for p in PERIODS}
    out["assumption_arith_ev_far_nll"] = {"threshold": ASSUMPTION_FAR_NLL, "values": {str(p): v for p, v in far_ev.items()},
                                          "holds": all(v <= ASSUMPTION_FAR_NLL for vs in far_ev.values() for v in vs)}
    if not out["assumption_arith_ev_far_nll"]["holds"]:
        out["stop_reasons"].append(f"arith_ev far NLL exceeds {ASSUMPTION_FAR_NLL} in the spread pilot (V1 assumption)")
    claims = claim_parts(far_alone, m, seeds)
    out["claims"] = {}
    for name, parts in claims.items():
        stops = [pt["stop"] for pt in parts if pt["stop"]]
        out["stop_reasons"] += stops
        ds = [pt["d"] for pt in parts]
        n = None if stops else smallest_n(ds)
        entry = {"parts": parts, "n_needed": n}
        if not stops and n is None:
            out["stop_reasons"].append(f"{name} needs more than {N_CAP} seeds per period; decide before preregistering")
        out["claims"][name] = entry
    ns = [c["n_needed"] for c in out["claims"].values()]
    out["n_seeds"] = None if out["stop_reasons"] or any(n is None for n in ns) else max(ns)
    if out["n_seeds"] is not None:
        n = out["n_seeds"]
        for entry in out["claims"].values():
            entry["power_at_n"] = [power(pt["d"], n) if not math.isinf(pt["d"]) else 1.0 for pt in entry["parts"]]
            entry["joint_power_at_n"] = math.prod(entry["power_at_n"])
    else:
        out["stop"] = "; ".join(out["stop_reasons"]) or "a claim needs more than the seed cap"
    out["sensitivity"] = sensitivity(out["claims"])
    return out


def sensitivity(claims: dict) -> dict:
    """Overall n (and per claim) under other choices; None where a claim's effect has the wrong sign."""
    def table(key: str, target: float = TARGET, scale: float = 1.0) -> dict:
        per = {}
        for name, entry in claims.items():
            ds = [pt[key] * scale for pt in entry["parts"]]
            per[name] = smallest_n(ds, target, N_SENSITIVITY_CAP) if all(d > 0 for d in ds) else None
        ok = [v for v in per.values()]
        return {"overall": None if any(v is None for v in ok) else max(ok), "per_claim": per}

    return {"point_sd": table("d_point_sd"), "target_0.80": table("d", 0.80), "effect_halved": table("d", scale=0.5)}


def far_alone_from_summaries(e8_path: Path, e9_path: Path) -> dict:
    """Per period: E8 ``reference`` (21 seeds) and E9 W = 8,192 (15 seeds) far NLL, read from the summaries."""
    e8, e9 = json.loads(e8_path.read_text()), json.loads(e9_path.read_text())
    return {p: e8["periods"][str(p)]["far_nll"]["reference"]["values"] +
            e9["periods"][str(p)]["far_nll_by_window"]["8192"]["values"] for p in PERIODS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e8", default=str(ROOT / "results" / "summary" / "e8.json"))
    parser.add_argument("--e9", default=str(ROOT / "results" / "summary" / "e9.json"))
    parser.add_argument("--spread-root", default=str(ROOT), help="directory holding results/runs/pilot_e10_spread*")
    parser.add_argument("--out", default=str(ROOT / "results" / "summary" / "e10_power.json"))
    args = parser.parse_args()
    loaded = load("pilot_e10_spread", Path(args.spread_root))
    if loaded["failed"]:
        sys.exit(f"spread-pilot records failed: {loaded['failed']}; a failed run is reported, never replaced")
    out = plan(far_alone_from_summaries(Path(args.e8), Path(args.e9)), collect(loaded))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    for name, entry in out["claims"].items():
        print(f"{name}: n needed {entry['n_needed']}; " + "; ".join(
            f"P={pt['period']} d={pt['d']:.2f}" for pt in entry["parts"]))
    print(f"STOP: {out['stop']}" if out["n_seeds"] is None else
          f"n = {out['n_seeds']} seeds per period")
    print(f"sensitivity (overall n): { {k: v['overall'] for k, v in out['sensitivity'].items()} }")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
