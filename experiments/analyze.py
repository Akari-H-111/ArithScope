"""Regenerate every results table and figure from run JSON files.

    python experiments/analyze.py

Reads ``results/runs/<suite>/*.json`` and writes ``results/summary/summary.json``,
``results/summary/tables.md``, and ``results/figures/*.png``. It then rewrites the
auto-generated blocks in ``docs/RESULTS.md`` and ``README.md``, between
``<!-- AUTO:<name>:BEGIN -->`` and ``<!-- AUTO:<name>:END -->`` markers.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope import arith  # noqa: E402
from experiments.plotstyle import LABELS, color, plt, save, themed  # noqa: E402

# Overridable for dry runs on other data (e.g. pilots); defaults are the committed results tree.
RUNS = Path(os.environ.get("ARITHSCOPE_RUNS", ROOT / "results" / "runs"))
_OUT = Path(os.environ.get("ARITHSCOPE_OUT", ROOT / "results"))
FIG = _OUT / "figures"
SUMMARY = _OUT / "summary"
DOCS_REWRITE = "ARITHSCOPE_OUT" not in os.environ
STANDALONE = ("nope", "alibi", "rope", "arith")
ROPE_FAMILY = ("rope", "rope+arith", "rope+fourier_fund", "rope+fourier_full", "rope+fourier_learned", "rope+relbias")


# ----------------------------------------------------------------------------- loading
def load(suite: str) -> list[dict]:
    path = RUNS / suite
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))] if path.exists() else []


def metric(run: dict, length: int, region: str = "auto", key: str = "acc") -> float | None:
    for t in run["test"]:
        if t["length"] == length:
            use = t["extrapolated"] if (region == "auto" and t["extrapolated"]) else t["all"]
            if region == "all":
                use = t["all"]
            return use.get(key)
    return None


def ci95(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "lo": None, "hi": None, "half": None}
    half = float(sstats.t.ppf(0.975, n - 1) * a.std(ddof=1) / math.sqrt(n))
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half, "half": half}


def group(runs: list[dict], *keys) -> dict:
    out = defaultdict(list)
    for r in runs:
        out[tuple(_key(r, k) for k in keys)].append(r)
    return out


def _key(run: dict, key: str):
    if key == "period":
        return tuple(run["spec"]["task"].get("periods", ()))
    if key == "orders":
        return tuple(run["spec"]["orders"])
    if key == "task":
        return run["spec"]["task"]["kind"]
    return run["spec"][key]


def paired(runs_a: list[dict], runs_b: list[dict], length: int, region: str = "auto") -> dict:
    a = {r["spec"]["seed"]: metric(r, length, region) for r in runs_a}
    b = {r["spec"]["seed"]: metric(r, length, region) for r in runs_b}
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    out = ci95(diffs)
    out.update({"seeds": seeds, "diffs": diffs, "wins": sum(d > 0 for d in diffs)})
    return out


def lengths_of(runs: list[dict]) -> list[int]:
    return sorted({t["length"] for r in runs for t in r["test"]})


def fmt(x: float | None, pct: bool = True, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f}" if pct else f"{x:.{digits}f}"


def fmt_ci(c: dict, digits: int = 1) -> str:
    if c["half"] is None:
        return fmt(c["mean"], digits=digits)
    return f"{100 * c['mean']:.{digits}f} ± {100 * c['half']:.{digits}f}"


def fmt_diff(p: dict) -> str:
    if p["half"] is None:
        return "–"
    return (f"{100 * p['mean']:+.1f} pp [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}] "
            f"({p['wins']}/{len(p['diffs'])})")


# ----------------------------------------------------------------------------- tables
def accuracy_table(runs: list[dict], variants: tuple[str, ...], title: str) -> tuple[str, dict]:
    by_var = group(runs, "variant")
    lens = lengths_of(runs)
    rows, data = [], {}
    header = "| variant | " + " | ".join(f"N={L}" for L in lens) + " |"
    rows.append(header)
    rows.append("|---|" + "---|" * len(lens))
    for v in variants:
        rs = by_var.get((v,), [])
        if not rs:
            continue
        cells, data[v] = [], {}
        for L in lens:
            c = ci95([metric(r, L) for r in rs if metric(r, L) is not None])
            data[v][L] = c
            cells.append(fmt_ci(c))
        rows.append(f"| {LABELS.get(v, v)} | " + " | ".join(cells) + " |")
    oracle = {L: float(np.mean([metric(r, L, key="oracle_acc") for r in runs])) for L in lens}
    rows.append("| *Bayes oracle* | " + " | ".join(fmt(oracle[L]) for L in lens) + " |")
    data["oracle"] = oracle
    note = (f"**{title}.** Accuracy (%) as mean ± 95% CI half-width over seeds. For N > 64 only "
            "positions t ≥ 64 (never seen in training) are scored.")
    return note + "\n\n" + "\n".join(rows), data


# ----------------------------------------------------------------------------- figures
def plot_extrapolation(data: dict, title: str, stem: str, mode: str) -> Path:
    with themed(mode) as th:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
        lens = sorted(data["oracle"])
        for ax, family, sub in ((axes[0], STANDALONE, "Standalone position mechanisms"),
                                (axes[1], ROPE_FAMILY, "RoPE + additive channel")):
            ax.plot(lens, [data["oracle"][L] for L in lens], color=th["muted"], linewidth=1.2)
            ax.annotate("Bayes oracle", (lens[-1], data["oracle"][lens[-1]]), textcoords="offset points",
                        xytext=(-4, 6), ha="right", color=th["ink2"], fontsize=8.5)
            for v in family:
                if v not in data:
                    continue
                hero = v in ("arith", "rope+arith")  # drawn on top: equal-valued controls would hide it
                mean = [data[v][L]["mean"] for L in lens]
                ax.plot(lens, mean, color=color(th, v), label=LABELS[v], marker="o", markersize=5 if hero else 4.5,
                        markeredgecolor=th["surface"], markeredgewidth=1.0, zorder=5 if hero else 3,
                        linewidth=2.6 if hero else 2.0)
                lo = [data[v][L]["lo"] if data[v][L]["lo"] is not None else data[v][L]["mean"] for L in lens]
                hi = [data[v][L]["hi"] if data[v][L]["hi"] is not None else data[v][L]["mean"] for L in lens]
                ax.fill_between(lens, lo, hi, color=color(th, v), alpha=0.12, linewidth=0)
            ax.axvline(64, color=th["axis"], linewidth=1.0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(lens)
            ax.set_xticklabels([str(L) for L in lens])
            ax.set_xlabel("evaluation length N (trained at 64)")
            ax.set_title(sub, loc="left", fontsize=10)
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1, fontsize=8.5)
        axes[0].set_ylabel("accuracy on unseen positions")
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, stem, mode)


HERO_VARIANTS = ("rope+arith", "rope+fourier_learned", "rope+fourier_fund", "rope+relbias", "rope")


def _theory_margins() -> dict[int, tuple[float, float]]:
    """Exact best comb margins (ArithScope optimum, matched cosine bank) from results/summary/theory.json."""
    path = ROOT / "results" / "summary" / "theory.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())["margins"]
    return {r["period"]: (r["optimal_pd"], r["fundamental_cosine_bank"]) for r in rows}


MARGINS = _theory_margins()


def plot_hero(panels: dict, mode: str) -> Path:
    """P = 12 next to P = 13: where theory says the matched cosine bank should and should not work."""
    with themed(mode) as th:
        periods = sorted(panels)
        fig, axes = plt.subplots(1, len(periods), figsize=(10.5, 3.9), sharey=True, squeeze=False)
        for ax, p in zip(axes[0], periods):
            data = panels[p]
            lens = sorted(data["oracle"])
            ax.plot(lens, [data["oracle"][L] for L in lens], color=th["muted"], linewidth=1.2, zorder=2)
            ax.annotate("Bayes oracle", (lens[-1], data["oracle"][lens[-1]]), textcoords="offset points",
                        xytext=(-4, 6), ha="right", color=th["ink2"], fontsize=8.5)
            for v in HERO_VARIANTS[::-1]:
                if v not in data:
                    continue
                hero = v == "rope+arith"
                mean = [data[v][L]["mean"] for L in lens]
                lo = [data[v][L]["lo"] if data[v][L]["lo"] is not None else data[v][L]["mean"] for L in lens]
                hi = [data[v][L]["hi"] if data[v][L]["hi"] is not None else data[v][L]["mean"] for L in lens]
                ax.fill_between(lens, lo, hi, color=color(th, v), alpha=0.12, linewidth=0)
                ax.plot(lens, mean, color=color(th, v), label=LABELS[v], marker="o", markersize=5 if hero else 4.5,
                        markeredgecolor=th["surface"], markeredgewidth=1.0, zorder=5 if hero else 3,
                        linewidth=2.6 if hero else 2.0)
            ax.axvline(64, color=th["axis"], linewidth=1.0)
            if "rope+fourier_fund" in data and "rope+arith" in data:
                gap = max(abs(data["rope+fourier_fund"][L]["mean"] - data["rope+arith"][L]["mean"]) for L in lens)
                if gap < 0.01:  # below the plateau and above RoPE's curve, where no line runs
                    ax.text(0.30, 0.62, "matched cosine bank ≈ ArithScope\n(their lines overlap)",
                            transform=ax.transAxes, ha="left", va="center", color=th["ink2"], fontsize=8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(lens)
            ax.set_xticklabels([str(L) for L in lens], fontsize=8)
            ax.set_xlabel("evaluation length N (trained at 64)")
            kind = "prime" if arith.factorize(p) == ((p, 1),) else "composite"
            m = MARGINS.get(p)
            margin = f": comb margin {m[1]:.3f} (cosine bank) vs {m[0]:.3f} (ArithScope)" if m else ""
            ax.set_title(f"P = {p}, {kind}{margin}", loc="left", fontsize=9.5)
        axes[0][0].set_ylabel("accuracy on unseen positions")
        handles, labels = axes[0][0].get_legend_handles_labels()
        order = [labels.index(LABELS[v]) for v in HERO_VARIANTS if LABELS[v] in labels]
        fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                   bbox_to_anchor=(0.06, -0.01), ncol=3, fontsize=8.5, frameon=False)
        any_p = panels[periods[0]]
        n_seeds = any_p["rope+arith"][min(any_p["oracle"])]["n"] if "rope+arith" in any_p else 0
        fig.suptitle(f"Periodic denoising: trained on 64 tokens, tested on up to {max(any_p['oracle'])} "
                     f"({n_seeds} seed{'s' if n_seeds != 1 else ''} per model, mean and 95% CI)", x=0.01, ha="left", fontweight="bold",
                     fontsize=11, color=th["ink"])
        return save(fig, FIG, "hero", mode)


def plot_alias(rows: list[dict], period: int, mode: str) -> Path:
    with themed(mode) as th:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        xs = np.arange(len(rows))
        for x, row in zip(xs, rows):
            resolved = row["lcm"] % period == 0
            c = th["series"][0] if resolved else th["series"][7]
            ax.scatter([x] * len(row["seed_acc"]), row["seed_acc"], color=c, s=26, zorder=3,
                       edgecolors=th["surface"], linewidths=1.0)
            ax.plot([x - 0.28, x + 0.28], [row["mean"]] * 2, color=c, linewidth=2.4, zorder=4)
            ax.plot([x - 0.36, x + 0.36], [row["bayes"]] * 2, color=th["muted"], linewidth=1.0, zorder=2)
            ax.plot([x - 0.36, x + 0.36], [row["alias"]] * 2, color=th["ink2"], linewidth=1.0,
                    linestyle=(0, (2, 2)), zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{{{','.join(map(str, r['bank']))}}}\nlcm {r['lcm']}" for r in rows], fontsize=8)
        ax.set_ylabel("accuracy at N = 1024 (unseen positions)")
        ax.set_title(f"Alias fixture (period {period}, ArithScope as the only position signal)", loc="left")
        handles = [plt.Line2D([], [], color=th["series"][0], linewidth=2.4, label=f"bank resolves {period} ({period} | lcm)"),
                   plt.Line2D([], [], color=th["series"][7], linewidth=2.4, label=f"bank aliases {period}"),
                   plt.Line2D([], [], color=th["muted"], linewidth=1.0, label="Bayes oracle"),
                   plt.Line2D([], [], color=th["ink2"], linewidth=1.0, linestyle=(0, (2, 2)),
                              label="alias oracle (vote over the gcd(P, lcm)-comb)")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, fontsize=8.5)
        return save(fig, FIG, "e2_alias", mode)


def plot_shift(data: dict, mode: str, title: str = "Absolute-position shift at float32 (relative structure unchanged)",
               ylabel: str = "accuracy, N = 256") -> Path:
    with themed(mode) as th:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for v, curve in data.items():
            xs = sorted(curve)
            ax.plot([max(x, 1) for x in xs], [curve[x]["mean"] for x in xs], color=color(th, v), label=LABELS[v],
                    marker="o", markersize=4.5, markeredgecolor=th["surface"], markeredgewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("constant shift added to every position (0 plotted at 1)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, fontsize=8.5)
        return save(fig, FIG, "e1_shift", mode)


def plot_alpha(alpha: dict, mode: str) -> Path:
    with themed(mode) as th:
        periods = sorted(alpha)
        fig, axes = plt.subplots(1, len(periods), figsize=(5.4 * len(periods), 3.4), squeeze=False, sharey=True)
        for ax, p in zip(axes[0], periods):
            d = alpha[p]
            orders = d["orders"]
            xs = np.arange(len(orders))
            ax.bar(xs, d["mean"], width=0.62, color=th["series"][0], label="learned α (layer-0 heads, mean ± sd)",
                   yerr=d["sd"], error_kw={"ecolor": th["ink2"], "elinewidth": 1.0, "capsize": 0})
            ax.scatter(xs, d["optimal"], marker="D", s=30, color=th["series"][1], zorder=4,
                       edgecolors=th["surface"], linewidths=1.0, label="Theorem T3 optimum φ(d)/(P−1)")
            ax.set_xticks(xs)
            ax.set_xticklabels(orders, fontsize=8)
            ax.set_xlabel("order m")
            ax.set_title(f"P = {p}  (n = {d['heads']} heads; TV distance {d['tv']:.2f})", loc="left", fontsize=10)
            ax.set_ylim(0, 1.05)
        axes[0][0].set_ylabel("mixture weight")
        axes[0][0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2, fontsize=8.5)
        fig.suptitle("Learned ArithScope mixtures vs the provably optimal comb", x=0.01, ha="left",
                     fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, "e1_alpha", mode)


# ----------------------------------------------------------------------------- suites
def analyze_e1(summary: dict, tables: list[str], suite: str = "e1_periodic", label: str = "E1") -> None:
    runs = load(suite)
    if not runs:
        return
    for (period,), rs in sorted(group(runs, "period").items()):
        ptxt = "/".join(map(str, period))
        table, data = accuracy_table(rs, STANDALONE + ROPE_FAMILY[1:],
                                     f"{label} periodic denoising, P = {ptxt}")
        tables.append(table)
        by_var = group(rs, "variant")
        prim = {}
        for other in ("rope", "rope+fourier_fund", "rope+fourier_full", "rope+fourier_learned", "rope+relbias"):
            if (other,) in by_var and ("rope+arith",) in by_var:
                prim[other] = {L: paired(by_var[("rope+arith",)], by_var[(other,)], L) for L in lengths_of(rs)}
        summary[f"{label}_P{ptxt}"] = {"table": data, "paired_vs": {
            o: {str(L): {k: v for k, v in d.items()} for L, d in per.items()} for o, per in prim.items()}}
        lines = [f"**{label} paired differences, P = {ptxt}** (RoPE + ArithScope minus control; mean, 95% CI, "
                 "seeds won):", "", "| control | N=64 | N=256 | N=1024 | N=2048 |", "|---|---|---|---|---|"]
        for other, per in prim.items():
            cells = [fmt_diff(per[L]) if L in per else "–" for L in (64, 256, 1024, 2048)]
            lines.append(f"| {LABELS[other]} | " + " | ".join(cells) + " |")
        tables.append("\n".join(lines))
        for mode in ("light", "dark"):
            plot_extrapolation(data, f"Periodic denoising, P = {ptxt}: trained at N = 64, tested up to N = "
                                     f"{max(data['oracle'])}", f"{suite}_P{ptxt.replace('/', '-')}", mode)
        if suite == "e1_periodic" and len(period) == 1:
            summary.setdefault("_hero_panels", {})[period[0]] = data
    if suite == "e1_periodic" and summary.get("_hero_panels"):
        for mode in ("light", "dark"):
            plot_hero(summary["_hero_panels"], mode)


def analyze_shift(summary: dict, tables: list[str]) -> None:
    runs = [r for r in load("e1_periodic") if r.get("shift_test")]
    if not runs:
        return
    out = {}
    for (period,), rs in sorted(group(runs, "period").items()):
        curves = {}
        for (v,), vr in group(rs, "variant").items():
            per_shift = defaultdict(list)
            for r in vr:
                for t in r["shift_test"]:
                    per_shift[t["position_shift"]].append(t["all"]["acc"])
            curves[v] = {s: ci95(vals) for s, vals in sorted(per_shift.items())}
        out[period] = curves
    summary["E1_shift"] = {str(p): {v: {str(s): c for s, c in cur.items()} for v, cur in curves.items()}
                           for p, curves in out.items()}
    period = sorted(out)[-1]
    curves = out[period]
    keep = {v: curves[v] for v in ("rope+arith", "arith", "rope", "rope+fourier_fund", "rope+relbias", "alibi")
            if v in curves}
    shifts = sorted(next(iter(keep.values())))
    lines = [f"**E1 position-shift robustness, P = {'/'.join(map(str, period))}, N = 256** (accuracy %, mean over seeds):",
             "", "| variant | " + " | ".join(f"{s:.0e}" if s else "0" for s in shifts) + " |",
             "|---|" + "---|" * len(shifts)]
    for v, cur in keep.items():
        lines.append(f"| {LABELS[v]} | " + " | ".join(fmt(cur[s]["mean"]) for s in shifts) + " |")
    tables.append("\n".join(lines))


def analyze_posthoc_shift(summary: dict, tables: list[str]) -> None:
    """POST-HOC (not preregistered): shift test at the training length, from results/posthoc/shift_n64.json."""
    path = ROOT / "results" / "posthoc" / "shift_n64.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    shifts = data["shifts"]
    by = defaultdict(lambda: defaultdict(list))
    for run in data["runs"]:
        period = run["experiment"].split("_p")[-1]
        for row in run["rows"]:
            by[(period, run["variant"])][row["shift"]].append(row["acc"])
    out = {}
    for (period, v), cur in by.items():
        out.setdefault(period, {})[v] = {sh: ci95(vals) for sh, vals in cur.items()}
    summary["posthoc_shift_n64"] = {p: {v: {str(sh): c for sh, c in cur.items()} for v, cur in vs.items()}
                                    for p, vs in out.items()}
    order = ("rope+arith", "arith", "rope", "rope+fourier_fund", "rope+fourier_full", "rope+fourier_learned",
             "rope+relbias", "alibi", "nope")
    for period in sorted(out):
        lines = [f"**Post-hoc (not preregistered): absolute-position shift at the training length, P = {period}, "
                 f"N = {data['length']}** (accuracy %, mean over seeds; RoPE angles in float32):", "",
                 "| variant | " + " | ".join("0" if sh == 0 else f"10^{len(str(sh)) - 1}" for sh in shifts) + " |",
                 "|---|" + "---|" * len(shifts)]
        for v in order:
            if v in out[period]:
                lines.append(f"| {LABELS[v]} | " + " | ".join(fmt(out[period][v][sh]["mean"]) for sh in shifts) + " |")
        tables.append("\n".join(lines))
    period = "13" if "13" in out else sorted(out)[-1]
    keep = {v: out[period][v] for v in ("rope+arith", "arith", "rope", "rope+fourier_fund", "rope+relbias", "alibi")
            if v in out[period]}
    for mode in ("light", "dark"):
        plot_shift(keep, mode, title=f"Post-hoc: shift every position by a constant (P = {period}, N = 64, float32 RoPE)",
                   ylabel="accuracy at the training length")


def analyze_alpha(summary: dict, tables: list[str]) -> None:
    """Compare learned layer-0 mixtures of RoPE + ArithScope with the T3 optimum (exploratory)."""
    runs = [r for r in load("e1_periodic") if r["spec"]["variant"] in ("rope+arith", "arith")]
    if not runs:
        return
    alpha, lines = {}, ["**Learned mixtures vs Theorem T3** (layer-0 heads, all seeds; exploratory):", "",
                        "| P | variant | heads | mass on divisors of P | TV distance to optimum | "
                        "comb margin of mean α (optimum P/(P−1)) |", "|---|---|---|---|---|---|"]
    for (period, variant), rs in sorted(group(runs, "period", "variant").items()):
        p = period[0]
        orders = rs[0]["positional"][0]["arith"]["orders"]
        mats = np.asarray([layer["arith"]["alpha"][h] for r in rs for layer in r["positional"] if layer["layer"] == 0
                           for h in range(len(layer["arith"]["gamma"]))])
        mean, sd = mats.mean(0), mats.std(0)
        opt = arith.optimal_comb_weights(p)
        optimal = np.asarray([float(opt.get(m, 0)) for m in orders])
        tv = 0.5 * float(np.abs(mean - optimal).sum())
        on_div = float(sum(mean[i] for i, m in enumerate(orders) if p % m == 0))
        values = [sum(mean[i] * float(arith.kernel(m, n)) for i, m in enumerate(orders)) for n in range(p)]
        margin = values[0] - max(values[1:])
        summary[f"E1_alpha_P{p}_{variant}"] = {"orders": orders, "mean": mean.tolist(), "sd": sd.tolist(),
                                              "optimal": optimal.tolist(), "tv": tv, "mass_on_divisors": on_div,
                                              "margin": margin, "optimal_margin": p / (p - 1), "heads": len(mats)}
        lines.append(f"| {p} | {LABELS[variant]} | {len(mats)} | {on_div:.3f} | {tv:.3f} | {margin:.3f} ({p / (p - 1):.3f}) |")
        if variant == "rope+arith":
            alpha[p] = {"orders": orders, "mean": mean.tolist(), "sd": sd.tolist(), "optimal": optimal.tolist(),
                        "tv": tv, "heads": len(mats)}
    tables.append("\n".join(lines))
    if alpha:
        for mode in ("light", "dark"):
            plot_alpha(alpha, mode)
    analyze_cosine_mixtures(summary, tables)


DRIFT_MAX = 8192


def _log_bins(blocks: int) -> list[tuple[int, int]]:
    """Block-index bins [1, 2), [2, 4), [4, 8), ... covering 1 .. blocks - 1."""
    bins, a = [], 1
    while a < blocks:
        bins.append((a, min(2 * a, blocks)))
        a *= 2
    return bins


def analyze_drift(summary: dict, tables: list[str]) -> None:
    """POST-HOC mechanism check for S1: comb selectivity of trained learned-frequency heads versus offset.

    Each head of ``rope+fourier_learned`` (both layers) has the positional bias
    ``b(n) = sum_f a_f cos(omega_f n)`` (float64, as in the model). In the period block
    ``[jP, (j+1)P)`` its comb margin is ``b(jP) - max_{0<r<P} b(jP + r)``. A *comb head* has a
    positive margin in every block inside the training window and a block-0 margin of at least
    25% of its block-0 range. Its margin divided by the block-0 margin is averaged over
    log-spaced bins of blocks up to N = 8192. An exactly periodic bias (ArithScope, the fixed
    cosine banks) has ratio 1 at every offset by construction.
    """
    runs = [r for r in load("e1_periodic") if r["spec"]["variant"] == "rope+fourier_learned"]
    if not runs:
        return
    marks = (64, 1024, 4096, DRIFT_MAX)
    out, lines = {}, ["**Post-hoc (not preregistered): comb selectivity of learned-frequency heads** (S1 mechanism). "
                      "Comb margin divided by the margin at offset 0, per comb head, averaged over the block bin "
                      "that contains the offset; median over heads:", "",
                      "| P | comb heads / all heads | n ≈ 64 | n ≈ 1,024 | n ≈ 4,096 | n ≈ 8,192 | comb lost (ratio ≤ 0) near 8,192 |",
                      "|---|---|---|---|---|---|---|"]
    for (period,), rs in sorted(group(runs, "period").items()):
        p = period[0]
        blocks = DRIFT_MAX // p
        n = np.arange(blocks * p, dtype=np.float64)
        window = -(-rs[0]["spec"]["train_length"] // p)  # blocks that start inside the training window
        bins = _log_bins(blocks)
        per_head, total = [], 0
        for r in rs:
            for layer in r["positional"]:
                omega = np.asarray(layer["fourier_learned"]["omega"], dtype=np.float64)
                amp = np.asarray(layer["fourier_learned"]["amplitude"], dtype=np.float64)
                for h in range(omega.shape[0]):
                    total += 1
                    b = (amp[h][:, None] * np.cos(omega[h][:, None] * n[None, :])).sum(0).reshape(blocks, p)
                    margin = b[:, 0] - b[:, 1:].max(1)
                    if margin[:window].min() > 0 and margin[0] >= 0.25 * (b[0].max() - b[0].min()):
                        ratio = margin / margin[0]
                        per_head.append([ratio[a:c].mean() for a, c in bins])
        if not per_head:
            continue
        per_head = np.asarray(per_head)  # [heads, bins]
        centers = [p * math.sqrt(a * (c - 1)) for a, c in bins]
        at = {L: min(range(len(bins)), key=lambda i: abs(math.log(centers[i] / L))) for L in marks}
        q = np.quantile(per_head, [0.1, 0.5, 0.9], axis=0)
        lost = int((per_head[:, at[DRIFT_MAX]] <= 0).sum())
        out[p] = {"offsets": centers, "q10": q[0].tolist(), "median": q[1].tolist(), "q90": q[2].tolist(),
                  "heads": len(per_head), "all_heads": total, "lost_near_max": lost,
                  "median_at": {str(L): float(q[1][at[L]]) for L in marks}}
        lines.append(f"| {p} | {len(per_head)} / {total} | " + " | ".join(f"{q[1][at[L]]:.2f}" for L in marks)
                     + f" | {lost} of {len(per_head)} |")
    if not out:
        return
    summary["posthoc_drift"] = {str(p): {k: v for k, v in d.items() if k not in ("offsets", "q10", "median", "q90")}
                                for p, d in out.items()}
    tables.append("\n".join(lines))
    for mode in ("light", "dark"):
        plot_drift(out, mode)


def plot_drift(out: dict, mode: str) -> Path:
    with themed(mode) as th:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.axhline(1.0, color=color(th, "rope+arith"), linewidth=2.6, label="ArithScope (exactly periodic: ratio 1 at every offset)")
        ax.axhline(0.0, color=th["muted"], linewidth=1.0)
        dashes = {13: (0, (5, 2))}  # same entity color for both periods; the line style tells them apart
        for p, d in sorted(out.items()):
            c = color(th, "rope+fourier_learned")
            ax.fill_between(d["offsets"], d["q10"], d["q90"], color=c, alpha=0.10, linewidth=0)
            ax.plot(d["offsets"], d["median"], color=c, linewidth=2.0, linestyle=dashes.get(p, "-"), marker="o",
                    markersize=4.5, markeredgecolor=th["surface"], markeredgewidth=1.0,
                    label=f"learned frequencies, P = {p} (median of {d['heads']} comb heads, 10–90% band)")
        ax.axvline(64, color=th["axis"], linewidth=1.0)
        ax.text(64, 0.03, " training window ends", transform=ax.get_xaxis_transform(), color=th["ink2"], fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("offset n = i − j (bins of period blocks)")
        ax.set_ylabel("comb margin / margin at n = 0")
        ax.set_title("Post-hoc: trained learned-frequency heads lose their comb with distance", loc="left", fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=1, fontsize=8.5)
        return save(fig, FIG, "e1_drift", mode)


def analyze_cosine_mixtures(summary: dict, tables: list[str]) -> None:
    """Where the matched cosine bank puts its layer-0 mixture mass (exploratory mechanism check for C2)."""
    runs = [r for r in load("e1_periodic") if r["spec"]["variant"] == "rope+fourier_fund"]
    if not runs:
        return
    lines = ["**Matched cosine bank, layer-0 mixtures** (exploratory; mean over heads and seeds). Mass on periods that "
             "divide P keeps the bias exactly P-periodic; mass elsewhere approximates the comb only inside the training "
             "window:", "", "| P | heads | mass on periods dividing P | largest components |", "|---|---|---|---|"]
    for (period,), rs in sorted(group(runs, "period").items()):
        p = period[0]
        comps = rs[0]["positional"][0]["fourier_fund"]["components"]
        mats = np.asarray([layer["fourier_fund"]["alpha"][h] for r in rs for layer in r["positional"]
                           if layer["layer"] == 0 for h in range(len(layer["fourier_fund"]["gamma"]))])
        mean = mats.mean(0)
        on_div = float(sum(mean[i] for i, (m, _) in enumerate(comps) if p % m == 0))
        top = np.argsort(-mean)[:4]
        summary[f"E1_cosine_mix_P{p}"] = {"mass_on_divisor_periods": on_div, "mean": mean.tolist(),
                                          "components": comps, "heads": len(mats)}
        lines.append(f"| {p} | {len(mats)} | {on_div:.3f} | " +
                     ", ".join(f"cos(2πn/{comps[i][0]}): {mean[i]:.2f}" for i in top) + " |")
    tables.append("\n".join(lines))


def analyze_alias(summary: dict, tables: list[str]) -> None:
    runs = load("e2_alias")
    if not runs:
        return
    period = runs[0]["spec"]["task"]["periods"][0]
    rows = []
    for (bank,), rs in sorted(group(runs, "orders").items(), key=lambda kv: (arith.lcm(kv[0][0]) % period == 0,
                                                                           arith.lcm(kv[0][0]), len(kv[0][0]))):
        L = 1024
        accs = [metric(r, L) for r in rs]
        ext = [next(t for t in r["test"] if t["length"] == L)["extrapolated"] for r in rs]
        lcm = arith.lcm(bank)
        alias = float(np.mean([e.get("alias_oracle_acc", e["oracle_acc"]) for e in ext]))
        rows.append({"bank": list(bank), "lcm": lcm, "orders": len(bank), "mean": float(np.mean(accs)),
                     "ci": ci95(accs), "seed_acc": accs, "bayes": float(np.mean([e["oracle_acc"] for e in ext])),
                     "alias": alias})
    summary["E2_alias"] = {"period": period, "rows": rows}
    lines = [f"**E2 alias fixture** (period {period}, query tokens, ArithScope only; accuracy % on unseen positions "
             "at N = 1024):", "",
             "| bank | orders | lcm | model | alias oracle | Bayes oracle | preregistered check |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["lcm"] % period == 0:
            ok = r["mean"] >= r["bayes"] - 0.05
            check = f"within 5 pp of Bayes: {'yes' if ok else 'NO'}"
        else:
            ok = r["mean"] <= r["alias"] + 0.02
            check = f"at most alias + 2 pp: {'yes' if ok else 'NO'}"
        lines.append(f"| {{{', '.join(map(str, r['bank']))}}} | {r['orders']} | {r['lcm']} | {fmt_ci(r['ci'])} | "
                     f"{fmt(r['alias'])} | {fmt(r['bayes'])} | {check} |")
    tables.append("\n".join(lines))
    for mode in ("light", "dark"):
        plot_alias(rows, period, mode)


def analyze_controls(summary: dict, tables: list[str]) -> None:
    for suite, title in (("e3_tagged", "E3 tagged denoising (aperiodic control)"), ("e4_mqar", "E4 MQAR")):
        runs = load(suite)
        if not runs:
            continue
        table, data = accuracy_table(runs, ("nope", "alibi", "rope", "arith", "rope+arith", "rope+fourier_fund",
                                            "rope+relbias"), title)
        tables.append(table)
        by_var = group(runs, "variant")
        p = paired(by_var[("rope+arith",)], by_var[("rope",)], 64, "all") if ("rope",) in by_var else None
        summary[suite] = {"table": data, "c4_rope_arith_minus_rope_N64": p}
        if p:
            verdict = ("no harm (CI lower bound > −1 pp)" if p["lo"] is not None and p["lo"] > -0.01
                       else "inconclusive or harmful (CI lower bound ≤ −1 pp)")
            tables.append(f"*{title} — C4 check:* RoPE + ArithScope minus RoPE at N = 64: {fmt_diff(p)} → {verdict}.")


def analyze_meta(summary: dict, tables: list[str]) -> None:
    runs = load("e1_periodic")
    if not runs:
        return
    lines = ["**Matched-baseline metadata** (E1; parameters from the run files, CPU time per training step):", "",
             "| variant | total params | positional params | ms / step (mean ± sd) |", "|---|---|---|---|"]
    for (v,), rs in sorted(group(runs, "variant").items()):
        step = [1000 * r["train"]["mean_step_seconds"] for r in rs]
        lines.append(f"| {LABELS[v]} | {rs[0]['parameters']['total']:,} | {rs[0]['parameters']['positional']:,} | "
                     f"{np.mean(step):.1f} ± {np.std(step):.1f} |")
    tables.append("\n".join(lines))


MODE_LABELS = {"baseline_sdpa": "RoPE only, fused SDPA (floor)", "arith_explicit": "ArithScope, explicit logits",
               "arith_sdpa_mask": "ArithScope, SDPA + dense bias mask", "arith_folded": "ArithScope, folded into q/k (fused)",
               "arith_flex": "ArithScope, compiled FlexAttention (forward only)"}
MODE_SLOT = {"baseline_sdpa": 1, "arith_explicit": 6, "arith_sdpa_mask": 2, "arith_folded": 0, "arith_flex": 3}


def analyze_overhead(summary: dict, tables: list[str]) -> None:
    path = ROOT / "results" / "overhead" / "overhead.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = [r for r in data["rows"] if "median_seconds" in r]
    summary["overhead"] = data
    for config in sorted({r["config"] for r in rows}):
        sub = [r for r in rows if r["config"] == config]
        lens = sorted({r["length"] for r in sub})
        for backward in (False, True):
            cell = {(r["mode"], r["length"]): r for r in sub if r["backward"] == backward}
            if not cell:
                continue
            head = "| backend | " + " | ".join(f"N={L}" for L in lens) + " |"
            lines = [f"**Overhead, {config}, {'forward+backward' if backward else 'forward'}** "
                     f"(median ms; peak RSS increase MB in parentheses; one layer, batch 1, CPU, 4 threads):", "",
                     head, "|---|" + "---|" * len(lens)]
            for mode in MODE_LABELS:
                vals = []
                for L in lens:
                    r = cell.get((mode, L))
                    vals.append("skipped" if r is None else f"{1000 * r['median_seconds']:.1f} ({r['peak_rss_increase_mb']:.0f})")
                lines.append(f"| {MODE_LABELS[mode]} | " + " | ".join(vals) + " |")
            tables.append("\n".join(lines))
    for mode in ("light", "dark"):
        plot_overhead(rows, mode)


def plot_overhead(rows: list[dict], mode: str) -> Path:
    with themed(mode) as th:
        configs = sorted({r["config"] for r in rows})
        fig, axes = plt.subplots(1, 2 * len(configs), figsize=(4.0 * 2 * len(configs), 3.4), squeeze=False)
        for c_idx, config in enumerate(configs):
            sub = [r for r in rows if r["config"] == config and not r["backward"]]
            for k, (key, label) in enumerate((("median_seconds", "forward time (s)"),
                                              ("peak_rss_increase_mb", "peak RSS increase (MB)"))):
                ax = axes[0][2 * c_idx + k]
                for m in MODE_LABELS:
                    pts = sorted((r["length"], r[key]) for r in sub if r["mode"] == m)
                    if pts:
                        ax.plot([x for x, _ in pts], [max(y, 1e-3) for _, y in pts], color=th["series"][MODE_SLOT[m]],
                                label=MODE_LABELS[m], marker="o", markersize=4.5, markeredgecolor=th["surface"],
                                markeredgewidth=1.0)
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xlabel("sequence length N")
                ax.set_title(f"{config}: {label}", loc="left", fontsize=9.5)
        axes[0][0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2, fontsize=8.5)
        fig.suptitle("Cost of the ArithScope bias by backend (forward pass, CPU)", x=0.01, ha="left",
                     fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, "overhead", mode)


def headline(summary: dict) -> str:
    """README headline table: E1 accuracy on unseen positions for the key variants."""
    rows = []
    keys = [k for k in summary if k.startswith("E1_P") and "/" not in k]
    if not keys:
        return "(results pending)"
    lens = [L for L in (1024, 8192) if all(L in summary[k]["table"]["oracle"] for k in keys)]
    head = "| period | model | " + " | ".join(f"N = {L} ({L // 64}× train length)" for L in lens) + " |"
    rows += [head, "|---|---|" + "---|" * len(lens)]
    for k in sorted(keys):
        tab, p = summary[k]["table"], k.split("_P")[1]
        for v in ("rope+arith", "rope", "rope+fourier_fund", "rope+relbias", "rope+fourier_learned"):
            if v in tab:
                rows.append(f"| {p} | {LABELS[v]} | " + " | ".join(fmt_ci(tab[v][L]) for L in lens) + " |")
        rows.append(f"| {p} | *Bayes oracle* | " + " | ".join(fmt(tab['oracle'][L]) for L in lens) + " |")
    seeds = sorted({summary[k]["table"]["rope+arith"][lens[0]]["n"] for k in keys if "rope+arith" in summary[k]["table"]})
    return "\n".join(rows) + (f"\n\nAccuracy (%) on positions never seen in training (t ≥ 64), mean ± 95% CI over "
                              f"{'/'.join(map(str, seeds))} seeds; generated by `experiments/analyze.py`.")


def tldr(summary: dict) -> str:
    """README summary bullets. Every number is read from the summary; wording does not depend on outcomes."""
    out = []
    t13, t12 = summary.get("E1_P13", {}).get("table", {}), summary.get("E1_P12", {}).get("table", {})
    if t13.get("rope+arith") and t13.get("rope"):
        L = max(t13["oracle"])
        p12 = (f" At period 12: {fmt(t12['rope+arith'][L]['mean'])}% vs {fmt(t12['rope'][L]['mean'])}% "
               f"(Bayes {fmt(t12['oracle'][L])}%)." if t12.get("rope+arith") and t12.get("rope") and L in t12["oracle"] else "")
        out.append(f"- **Length extrapolation.** Trained on 64 tokens and tested on {L:,} ({L // 64}× longer), "
                   f"RoPE + ArithScope scores {fmt(t13['rope+arith'][L]['mean'])}% on unseen positions at period 13; "
                   f"the Bayes-optimal predictor scores {fmt(t13['oracle'][L])}% and RoPE alone "
                   f"{fmt(t13['rope'][L]['mean'])}%.{p12}")
    if t13.get("rope+fourier_fund") and t12.get("rope+fourier_fund") and 1024 in t13["oracle"]:
        out.append(f"- **The basis matters where theory says it should.** A cosine bank with the same periods and parameter "
                   f"count scores {fmt(t13['rope+fourier_fund'][1024]['mean'])}% at period 13 (N = 1,024) against "
                   f"{fmt(t13['rope+arith'][1024]['mean'])}% for ArithScope. At period 12, where the two channels' "
                   f"optimal comb margins nearly coincide (0.987 vs 1.091), it scores "
                   f"{fmt(t12['rope+fourier_fund'][1024]['mean'])}% against {fmt(t12['rope+arith'][1024]['mean'])}%.")
    a13 = summary.get("E1_alpha_P13_rope+arith")
    if a13:
        w13 = a13["mean"][a13["orders"].index(13)]
        out.append(f"- **Gradient descent finds the provably optimal comb.** Offered 16 orders, layer-0 heads trained at "
                   f"period 13 put {fmt(w13)}% of their mixture weight on order 13, the unique optimum of Theorem T3 "
                   f"(total-variation distance {a13['tv']:.3f}, {a13['heads']} heads).")
    e2 = summary.get("E2_alias")
    if e2:
        period, rows = e2["period"], e2["rows"]
        alias = [r for r in rows if r["lcm"] % period]
        good = [r for r in rows if r["lcm"] % period == 0]
        if alias and good:
            gap_alias = max(r["mean"] - r["alias"] for r in alias)
            gap_bayes = max(r["bayes"] - r["mean"] for r in good)
            out.append(f"- **Identifiability theory predicts which banks fail.** In the alias fixture (period {period}), "
                       f"banks with lcm {', '.join(sorted({str(r['lcm']) for r in alias}))} end at most "
                       f"{100 * gap_alias:+.1f} pp from their alias ceiling, whatever their size; every bank with "
                       f"{period} | lcm ends within {100 * gap_bayes:.1f} pp of Bayes, including two-order banks.")
    c4 = summary.get("e3_tagged", {}).get("c4_rope_arith_minus_rope_N64")
    if c4 and c4["lo"] is not None:
        out.append(f"- **No measurable cost on aperiodic data.** On tagged denoising, adding ArithScope to RoPE changes "
                   f"accuracy at the training length by {100 * c4['mean']:+.2f} pp (95% CI [{100 * c4['lo']:+.2f}, "
                   f"{100 * c4['hi']:+.2f}]).")
    sh = summary.get("posthoc_shift_n64", {}).get("13", {})
    if sh.get("arith") and sh.get("rope"):
        top = max(int(s) for s in sh["arith"])
        out.append(f"- **Exact at any absolute position (post hoc).** Shifting every position by "
                   f"10<sup>{len(str(top)) - 1}</sup> leaves "
                   f"ArithScope-only models at {fmt(sh['arith'][str(top)]['mean'])}% (unshifted "
                   f"{fmt(sh['arith']['0']['mean'])}%); RoPE models, whose angles are float32, drop from "
                   f"{fmt(sh['rope']['0']['mean'])}% to {fmt(sh['rope'][str(top)]['mean'])}%.")
    return "\n".join(out) if out else "(results pending)"


def overhead_md(summary: dict) -> str:
    """README cost paragraph from results/overhead/overhead.json (empty when the benchmark has not run)."""
    data = summary.get("overhead")
    if not data:
        return ""
    rows = [r for r in data["rows"] if "median_seconds" in r and not r["backward"]]
    config = "llm_d64_B16"
    cell = {(r["mode"], r["length"]): r for r in rows if r["config"] == config}
    need = ("baseline_sdpa", "arith_explicit", "arith_folded")
    common = [L for L in sorted({L for _, L in cell}) if all((m, L) in cell for m in need)]
    if not common:
        return ""
    L = common[-1]
    modes = need + (("arith_flex",) if ("arith_flex", L) in cell else ())
    ms = {m: 1000 * cell[(m, L)]["median_seconds"] for m in modes}
    mb = {m: cell[(m, L)]["peak_rss_increase_mb"] for m in modes}
    names = {"arith_folded": "folded", "arith_flex": "compiled FlexAttention", "arith_explicit": "explicit bias",
             "baseline_sdpa": "RoPE alone"}
    order = [m for m in ("arith_folded", "arith_flex", "arith_explicit") if m in modes]
    longest = max(Lf for (m, Lf) in cell if m == "arith_folded")
    times = ", ".join(f"{ms[m]:.1f} ms ({names[m]})" for m in order)
    mems = ", ".join(f"{mb[m]:.0f} MB ({names[m]})" for m in order + ["baseline_sdpa"])
    flex_floor = ""
    if "arith_flex" in modes:
        L0 = min(Lf for (m, Lf) in cell if m == "arith_flex")
        flex_floor = (f" The compiled FlexAttention figure includes one-time compilation; at N = {L0:,} it already "
                      f"adds {cell[('arith_flex', L0)]['peak_rss_increase_mb']:.0f} MB.")
    return "\n".join([
        "### Cost", "",
        '<p align="center">', "  <picture>",
        '    <source media="(prefers-color-scheme: dark)" srcset="results/figures/overhead_dark.png">',
        '    <img alt="Forward time and peak memory of one attention layer versus sequence length, by backend" '
        'src="results/figures/overhead_light.png" width="920">',
        "  </picture>", "</p>", "",
        f"One attention layer, inference forward pass, batch 1, CPU with 4 threads (`experiments/bench_overhead.py`). "
        f"In the LLM-sized configuration (8 heads of dimension 64, prime-power bank `B_16`, 55 folded dimensions) at "
        f"N = {L:,}, ArithScope takes {times}, against {ms['baseline_sdpa']:.1f} ms for RoPE alone in fused SDPA. "
        f"Peak memory grows by {mems}.{flex_floor} The folded path runs up to N = {longest:,}. These are "
        "single-machine CPU measurements, not a speed claim; full tables are in [`docs/RESULTS.md`](docs/RESULTS.md).",
    ])


def _pd(summary: dict, period: str, other: str, length: int) -> dict | None:
    return summary.get(f"E1_P{period}", {}).get("paired_vs", {}).get(other, {}).get(str(length))


def findings(summary: dict) -> str:
    """Preregistered verdicts (docs/PREREGISTRATION.md) evaluated mechanically from the summary."""
    out = []

    def ci_txt(p: dict) -> str:
        return (f"{100 * p['mean']:+.1f} pp, 95% CI [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}], "
                f"{p['wins']}/{len(p['diffs'])} seeds")

    # C1
    c1 = [(P, _pd(summary, P, "rope", 1024)) for P in ("12", "13")]
    if all(p and p["lo"] is not None for _, p in c1):
        ok = all(p["lo"] > 0 for _, p in c1)
        detail = "; ".join(f"P = {P}: {ci_txt(p)}" for P, p in c1)
        out.append(f"- **C1 (periodic extrapolation vs RoPE): {'supported' if ok else 'NOT supported'}.** "
                   f"RoPE + ArithScope minus RoPE, unseen positions at N = 1024: {detail}.")
    # C2
    p13, p12 = _pd(summary, "13", "rope+fourier_fund", 1024), _pd(summary, "12", "rope+fourier_fund", 1024)
    if p13 and p13["lo"] is not None:
        ok = p13["lo"] > 0
        extra = f" At P = 12, where theory predicts no meaningful gap: {ci_txt(p12)}." if p12 and p12["lo"] is not None else ""
        out.append(f"- **C2 (prime period vs the matched cosine bank): {'supported' if ok else 'NOT supported'}.** "
                   f"P = 13, N = 1024: {ci_txt(p13)}.{extra}")
    # C3
    e2 = summary.get("E2_alias")
    if e2:
        period, rows = e2["period"], e2["rows"]
        aliased = [r for r in rows if r["lcm"] % period]
        resolved = [r for r in rows if r["lcm"] % period == 0]
        ok = all(r["mean"] <= r["alias"] + 0.02 for r in aliased) and all(r["mean"] >= r["bayes"] - 0.05 for r in resolved)
        a_txt = ", ".join(f"{{{','.join(map(str, r['bank']))}}} {100 * r['mean']:.1f}% (ceiling {100 * r['alias']:.1f}%)"
                          for r in aliased)
        r_txt = ", ".join(f"{{{','.join(map(str, r['bank']))}}} {100 * r['mean']:.1f}% (Bayes {100 * r['bayes']:.1f}%)"
                          for r in resolved)
        out.append(f"- **C3 (identifiability / alias fixture, P = {period}): {'supported' if ok else 'NOT supported'}.** "
                   f"Aliased banks: {a_txt}. Resolving banks: {r_txt}.")
    # C4
    e3 = summary.get("e3_tagged", {}).get("c4_rope_arith_minus_rope_N64")
    if e3 and e3["lo"] is not None:
        verdict = "supported" if e3["lo"] > -0.01 else "inconclusive / not supported"
        out.append(f"- **C4 (no harm on the aperiodic control): {verdict}.** Tagged denoising, RoPE + ArithScope minus "
                   f"RoPE at N = 64: {ci_txt(e3)} (margin: lower bound > −1 pp).")
    # S1
    s1 = []
    for P in ("12", "13"):
        tab = summary.get(f"E1_P{P}", {}).get("table", {})
        if "rope+fourier_learned" in tab and 8192 in tab["rope+fourier_learned"] and 1024 in tab["rope+fourier_learned"]:
            lf = tab["rope+fourier_learned"]
            ar = tab["rope+arith"]
            s1.append((P, lf[1024]["mean"], lf[8192]["mean"], ar[1024]["mean"], ar[8192]["mean"],
                       tab["oracle"][1024], tab["oracle"][8192]))
    if s1:
        drift = any(b < a for _, a, b, *_ in s1)
        tight = all(abs(a1 - o1) <= 0.01 and abs(a8 - o8) <= 0.01 for _, _, _, a1, a8, o1, o8 in s1)
        txt = "; ".join(f"P = {P}: learned frequencies {100 * a:.1f}% → {100 * b:.1f}%, ArithScope {100 * c:.1f}% → "
                        f"{100 * d:.1f}% (Bayes {100 * o1:.1f}% → {100 * o8:.1f}%)" for P, a, b, c, d, o1, o8 in s1)
        out.append(f"- **S1 (drift of learned frequencies, N = 1024 → 8192): {'observed' if drift else 'not observed'}; "
                   f"ArithScope within 1 pp of Bayes at both lengths: {'yes' if tight else 'no'}.** {txt}.")
    # S2
    for P in (12, 13):
        a = summary.get(f"E1_alpha_P{P}_rope+arith")
        if a:
            top = int(np.argmax(a["mean"]))
            out.append(f"- **S2 (learned mixtures), P = {P}:** layer-0 heads put {100 * a['mass_on_divisors']:.1f}% of their "
                       f"mixture mass on divisors of {P} (largest weight: order {a['orders'][top]}, "
                       f"{100 * a['mean'][top]:.1f}%); total-variation distance to the Theorem T3 optimum {a['tv']:.3f}; "
                       f"comb margin {a['margin']:.3f} vs optimum {a['optimal_margin']:.3f} ({a['heads']} heads).")
    return "\n".join(out) if out else "(results pending)"


def rewrite_block(path: Path, name: str, content: str) -> None:
    if not path.exists():
        return
    text = path.read_text()
    pattern = re.compile(rf"(<!-- AUTO:{name}:BEGIN -->)(.*?)(<!-- AUTO:{name}:END -->)", re.S)
    if pattern.search(text):
        path.write_text(pattern.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(3), text))


def main() -> None:
    summary: dict = {}
    tables: list[str] = []
    analyze_e1(summary, tables)
    analyze_shift(summary, tables)
    analyze_posthoc_shift(summary, tables)
    analyze_alpha(summary, tables)
    analyze_drift(summary, tables)
    analyze_alias(summary, tables)
    analyze_controls(summary, tables)
    analyze_e1(summary, tables, suite="e5_mixed", label="E5")
    analyze_meta(summary, tables)
    analyze_overhead(summary, tables)
    summary.pop("_hero_panels", None)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    md = "\n\n".join(tables)
    print(findings(summary))
    (SUMMARY / "tables.md").write_text(md + "\n")
    if DOCS_REWRITE:
        rewrite_block(ROOT / "docs" / "RESULTS.md", "tables", md)
        rewrite_block(ROOT / "README.md", "headline", headline(summary))
        rewrite_block(ROOT / "README.md", "tldr", tldr(summary))
        rewrite_block(ROOT / "README.md", "overhead", overhead_md(summary))
        rewrite_block(ROOT / "README.md", "findings", findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "findings", findings(summary))
    print(md)


if __name__ == "__main__":
    main()
