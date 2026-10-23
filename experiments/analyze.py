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


def paired(runs_a: list[dict], runs_b: list[dict], length: int, region: str = "auto", key: str = "acc") -> dict:
    a = {r["spec"]["seed"]: metric(r, length, region, key) for r in runs_a}
    b = {r["spec"]["seed"]: metric(r, length, region, key) for r in runs_b}
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
def _band(cells: dict, lens: list[int]) -> tuple[list[float], list[float]]:
    """95% CI band for plotting, clipped to [0, 1] (an accuracy cannot leave that range; with few seeds the t
    interval can)."""
    lo = [cells[L]["lo"] if cells[L]["lo"] is not None else cells[L]["mean"] for L in lens]
    hi = [cells[L]["hi"] if cells[L]["hi"] is not None else cells[L]["mean"] for L in lens]
    return [max(x, 0.0) for x in lo], [min(x, 1.0) for x in hi]


def plot_extrapolation(data: dict, title: str, stem: str, mode: str, oracle_label: str = "Bayes oracle") -> Path:
    with themed(mode) as th:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
        lens = sorted(data["oracle"])
        for ax, family, sub in ((axes[0], STANDALONE, "Standalone position mechanisms"),
                                (axes[1], ROPE_FAMILY, "RoPE + additive channel")):
            ax.plot(lens, [data["oracle"][L] for L in lens], color=th["muted"], linewidth=1.2)
            ax.annotate(oracle_label, (lens[-1], data["oracle"][lens[-1]]), textcoords="offset points",
                        xytext=(-4, 6), ha="right", color=th["ink2"], fontsize=8.5)
            for v in family:
                if v not in data:
                    continue
                hero = v in ("arith", "rope+arith")  # drawn on top: equal-valued controls would hide it
                mean = [data[v][L]["mean"] for L in lens]
                ax.plot(lens, mean, color=color(th, v), label=LABELS[v], marker="o", markersize=5 if hero else 4.5,
                        markeredgecolor=th["surface"], markeredgewidth=1.0, zorder=5 if hero else 3,
                        linewidth=2.6 if hero else 2.0)
                lo, hi = _band(data[v], lens)
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
        fig, axes = plt.subplots(1, len(periods), figsize=(10.5, 4.2), sharey=True, squeeze=False)
        notes: dict[int, list[str]] = {}
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
                lo, hi = _band(data[v], lens)
                ax.fill_between(lens, lo, hi, color=color(th, v), alpha=0.12, linewidth=0)
                ax.plot(lens, mean, color=color(th, v), label=LABELS[v], marker="o", markersize=5 if hero else 4.5,
                        markeredgecolor=th["surface"], markeredgewidth=1.0, zorder=5 if hero else 3,
                        linewidth=2.6 if hero else 2.0)
            ax.axvline(64, color=th["axis"], linewidth=1.0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(lens)
            ax.set_xticklabels([str(L) for L in lens], fontsize=8)
            ax.set_xlabel("evaluation length N (trained at 64)")
            m = MARGINS.get(p)
            note = [f"best comb margin: cosine bank {m[1]:.3f}, ArithScope {m[0]:.3f}"] if m else []
            if "rope+fourier_fund" in data and "rope+arith" in data:
                gap = max(abs(data["rope+fourier_fund"][L]["mean"] - data["rope+arith"][L]["mean"]) for L in lens)
                if gap < 0.01:
                    note.append("the cosine-bank curve lies under ArithScope's")
            notes[p] = note
        lines = max(len(n) for n in notes.values())
        for ax, p in zip(axes[0], periods):
            kind = "prime" if arith.factorize(p) == ((p, 1),) else "composite"
            ax.set_title(f"Period P = {p} ({kind})", loc="left", fontsize=10, pad=8 + 12 * lines)
            ax.text(0.0, 1.015, "\n".join(notes[p]), transform=ax.transAxes, ha="left", va="bottom",
                    color=th["ink2"], fontsize=8.5, linespacing=1.3)
        axes[0][0].set_ylabel("accuracy on unseen positions")
        handles, labels = axes[0][0].get_legend_handles_labels()
        order = [labels.index(LABELS[v]) for v in HERO_VARIANTS if LABELS[v] in labels]
        fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                   bbox_to_anchor=(0.06, -0.01), ncol=3, fontsize=8.5, frameon=False)
        any_p = panels[periods[0]]
        n_seeds = any_p["rope+arith"][min(any_p["oracle"])]["n"] if "rope+arith" in any_p else 0
        fig.suptitle(f"Periodic denoising: trained on 64 tokens, tested on up to {max(any_p['oracle'])} "
                     f"({n_seeds} seed{'s' if n_seeds != 1 else ''} per model, mean and 95% CI)", x=0.01, y=1.0 + 0.035 * lines,
                     ha="left", fontweight="bold", fontsize=11, color=th["ink"])
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
        hero = ("rope+arith", "arith")
        order = [v for v in data if v not in hero] + [v for v in hero if v in data]  # ArithScope drawn last, on top
        for v in order:
            curve = data[v]
            xs = sorted(curve)
            on_top = v in hero
            ax.plot([max(x, 1) for x in xs], [curve[x]["mean"] for x in xs], color=color(th, v), label=LABELS[v],
                    marker="o", markersize=4.5, markeredgecolor=th["surface"], markeredgewidth=1.0,
                    linewidth=2.6 if on_top else 2.0, zorder=5 if on_top else 3,
                    linestyle=(0, (4, 2)) if v == "arith" else "-")
        if "rope+arith" in data:
            ref = data["rope+arith"]
            same = [LABELS[v] for v in data if v != "rope+arith"
                    and all(abs(data[v][x]["mean"] - ref[x]["mean"]) < 0.0015 for x in ref)]
            if same:
                y = min(ref[x]["mean"] for x in ref)
                ax.annotate("same accuracy at every shift (lines overlap):\nRoPE + ArithScope, " + ", ".join(same),
                            (1, y), xytext=(4, -8), textcoords="offset points", va="top", ha="left",
                            color=th["ink2"], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("constant shift added to every position (0 plotted at 1)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        want = [LABELS[v] for v in data]  # legend in the caller's order, not the drawing order
        idx = [labels.index(w) for w in want if w in labels]
        ax.legend([handles[i] for i in idx], [labels[i] for i in idx], loc="upper left", bbox_to_anchor=(0.0, -0.18),
                  ncol=2, fontsize=8.5)
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
        mixed = len(period) > 1
        what = (f"Mixed periods {', '.join(map(str, period))}, unlabeled" if mixed else f"Periodic denoising, P = {ptxt}")
        for mode in ("light", "dark"):
            plot_extrapolation(data, f"{what}: trained at N = 64, tested up to N = {max(data['oracle'])}",
                               f"{suite}_P{ptxt.replace('/', '-')}", mode,
                               oracle_label="oracle (told the period)" if mixed else "Bayes oracle")
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
    summary["E1_shift"] = {"/".join(map(str, p)): {v: {str(s): c for s, c in cur.items()} for v, cur in curves.items()}
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


def analyze_gates(summary: dict, tables: list[str]) -> None:
    """Exploratory: how far training opens the ArithScope gate gamma, by suite and variant."""
    lines = ["**ArithScope gate magnitudes** (exploratory; largest |γ| over the heads of each layer, all layers and "
             "seeds; γ starts at 0):", "", "| suite | variant | layers | median | min | max |", "|---|---|---|---|---|---|"]
    out = {}
    for suite in ("e1_periodic", "e2_alias", "e3_tagged", "e5_mixed"):
        for (variant,), rs in sorted(group(load(suite), "variant").items()):
            gates = [float(np.abs(layer["arith"]["gamma"]).max()) for r in rs for layer in r["positional"]
                     if "arith" in layer]
            if not gates:
                continue
            out[f"{suite}/{variant}"] = {"median": float(np.median(gates)), "min": min(gates), "max": max(gates),
                                         "layers": len(gates)}
            lines.append(f"| {suite} | {LABELS[variant]} | {len(gates)} | {np.median(gates):.2f} | {min(gates):.2f} | "
                         f"{max(gates):.2f} |")
    if out:
        summary["gates"] = out
        tables.append("\n".join(lines))


def analyze_nll_runs(summary: dict, tables: list[str]) -> None:
    """E1 negative log-likelihood from the run files. Recorded for every run; not a preregistered decision metric."""
    runs = load("e1_periodic")
    if not runs:
        return
    out = {}
    for (period,), rs in sorted(group(runs, "period").items()):
        p = period[0]
        lens = lengths_of(rs)
        by_var = group(rs, "variant")
        lines = [f"**E1 NLL, P = {p}** (nats per scored position; unseen positions for N > 64; mean ± 95% CI over seeds. "
                 "NLL is recorded for every run but was not a preregistered decision metric):", "",
                 "| variant | " + " | ".join(f"N={L}" for L in lens) + " |", "|---|" + "---|" * len(lens)]
        table = {}
        for v in STANDALONE + ROPE_FAMILY[1:]:
            if (v,) not in by_var:
                continue
            cells = {L: ci95([metric(r, L, key="nll") for r in by_var[(v,)]]) for L in lens}
            table[v] = cells
            lines.append(f"| {LABELS[v]} | " + " | ".join(f"{c['mean']:.3f} ± {c['half']:.3f}" for c in cells.values()) + " |")
        pairs = {}
        if ("rope+arith",) in by_var and ("arith",) in by_var:
            pairs = {L: paired(by_var[("rope+arith",)], by_var[("arith",)], L, key="nll") for L in lens}
            lines += ["", "Paired NLL difference, RoPE + ArithScope minus ArithScope only (positive: RoPE + ArithScope is "
                      "worse): " + "; ".join(f"N = {L}: {p_['mean']:+.3f} [{p_['lo']:+.3f}, {p_['hi']:+.3f}] "
                                            f"({p_['wins']}/{len(p_['diffs'])} seeds higher)"
                                            for L, p_ in pairs.items() if L >= 1024) + "."]
        out[str(p)] = {"table": {v: {str(L): c for L, c in cells.items()} for v, cells in table.items()},
                       "rope_arith_minus_arith": {str(L): d for L, d in pairs.items()}}
        tables.append("\n".join(lines))
    summary["E1_nll"] = out


def analyze_posthoc_nll(summary: dict, tables: list[str]) -> None:
    """POST-HOC: NLL by position against the Bayes NLL, a recent-keys intervention, and attention off the comb."""
    path = ROOT / "results" / "posthoc" / "nll_diagnostics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    window = data["window"]
    by = defaultdict(list)
    for run in data["runs"]:
        by[(run["period"], run["variant"])].append(run)
    out = {}
    for (p, v), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        buckets = []
        for i, b in enumerate(rs[0]["buckets"]):
            xs = [r["buckets"][i] for r in rs]
            row = {k: float(np.mean([x[k] for x in xs])) for k in POSTHOC_SCALARS}
            for k in ("acc_window_layer", "nll_window_layer", "nll_seq", "nll_window_seq"):
                row[k] = np.mean([x[k] for x in xs], axis=0).tolist()
            row["nll_quantiles"] = {q: float(np.mean([x["nll_quantiles"][q] for x in xs])) for q in b["nll_quantiles"]}
            for k in ("spearman_nll_share", "spearman_nll_bayes"):
                defined = [x[k] for x in xs if x[k] is not None]
                row[k] = float(np.mean(defined)) if len(defined) == len(xs) else None
            # per seed: the NLL removed by limiting one layer, as a fraction of what limiting every layer removes
            drops = [(x["nll"] - x["nll_window"], [x["nll"] - y for y in x["nll_window_layer"]]) for x in xs]
            row["layer_share_range"] = ([[float(min(d[j] / a for a, d in drops)), float(max(d[j] / a for a, d in drops))]
                                         for j in range(len(b["nll_window_layer"]))]
                                        if all(a > 1e-3 for a, _ in drops) else None)
            row["seed_nll"], row["seed_nll_window"] = [x["nll"] for x in xs], [x["nll_window"] for x in xs]
            buckets.append({"start": b["start"], "end": b["end"], **row})
        depth = len(rs[0]["attention"])
        attention = [{k: float(np.mean([np.mean(r["attention"][j][k]) for r in rs])) for k in ("off_comb", "far_off_comb")}
                     for j in range(depth)]
        gates = [float(np.mean([max(r["gates"][j]) for r in rs])) for j in range(depth)]
        # per seed: change of NLL from the first bucket beyond the window to the last one, with and without limits
        first = next(i for i, b in enumerate(rs[0]["buckets"]) if b["start"] >= window)

        def change(get, rs=rs, first=first):
            return [float(get(r["buckets"][-1]) - get(r["buckets"][first])) for r in rs]

        rise = {"no_limit": change(lambda b: b["nll"]), "all_layers": change(lambda b: b["nll_window"]),
                **{f"layer_{j}": change(lambda b, j=j: b["nll_window_layer"][j]) for j in range(depth)}}
        out.setdefault(str(p), {})[v] = {"seeds": [r["seed"] for r in rs], "buckets": buckets, "attention": attention,
                                        "gates": gates, "rise": rise, "ece": float(np.mean([r["ece"] for r in rs])),
                                        "brier": float(np.mean([r["brier"] for r in rs]))}
    # RoPE + ArithScope minus ArithScope only, paired by seed, and per test sequence (averaged over seeds)
    pairs = {}
    for p, vs in out.items():
        if {"arith", "rope+arith"} <= set(vs):
            a = {r["seed"]: r for r in by[(int(p), "arith")]}
            b = {r["seed"]: r for r in by[(int(p), "rope+arith")]}
            seeds = sorted(set(a) & set(b))
            pairs[p] = [{"start": bk["start"], "end": bk["end"],
                         "seeds": ci95([b[s]["buckets"][i]["nll"] - a[s]["buckets"][i]["nll"] for s in seeds]),
                         "per_sequence": np.mean([np.subtract(b[s]["buckets"][i]["nll_seq"], a[s]["buckets"][i]["nll_seq"])
                                                  for s in seeds], axis=0).tolist()}
                        for i, bk in enumerate(vs["arith"]["buckets"])]
    head = (f"{data['sequences']} fresh sequences of {data['length']:,} tokens per checkpoint, data seed "
            f"{data['data_seed']}; mean over 5 seeds")
    periods = sorted(out, key=int)
    nll = [f"**Post-hoc (not preregistered): NLL by query position** ({head}). Median, 99th percentile, and \"top 1%\" "
           "(the share of the mean carried by the worst 1% of positions) describe the per-position distribution; "
           "\"from errors\" is the part of the mean contributed by wrong predictions, the rest comes from correct ones:", "",
           "| P | variant | positions | accuracy | NLL | median | 99th pct. | top 1% | from errors | Bayes NLL |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    win = [f"**Post-hoc: limiting attention to the last {window:,} keys** (same sequences; positions below {window:,} "
           "are unaffected by construction). NLL, then accuracy:", "",
           "| P | variant | positions | NLL, no limit | all layers | layer 0 only | layer 1 only | accuracy, no limit | "
           "all layers |", "|---|---|---|---|---|---|---|---|---|"]
    att = ["**Post-hoc: attention off the comb** (attention mass on keys whose offset is not a multiple of P, averaged "
           f"over heads and the query positions {data['query_rows'][0]:,}, {data['query_rows'][1]:,}, …, "
           f"{data['query_rows'][-1]:,}) and the largest learned gate |γ| per layer:", "",
           "| P | variant | layer | off the comb, all keys | off the comb, beyond the last "
           f"{window:,} keys | max \\|γ\\| |", "|---|---|---|---|---|---|"]
    for p in periods:
        for v, d in out[p].items():
            for b in d["buckets"]:
                span = f"{b['start']:,}–{b['end']:,}"
                q = b["nll_quantiles"]
                bayes = f"{b['bayes_nll']:.4f}" if b["bayes_nll"] >= 5e-5 else "< 0.0001"
                nll.append(f"| {p} | {LABELS[v]} | {span} | {100 * b['acc']:.2f}% | {b['nll']:.3f} | {q['50']:.3f} | "
                           f"{q['99']:.3f} | {100 * b['nll_top1pct_share']:.1f}% | {b['nll_from_errors']:.3f} | {bayes} |")
                if b["start"] >= window:
                    win.append(f"| {p} | {LABELS[v]} | {span} | {b['nll']:.3f} | {b['nll_window']:.3f} | "
                               + " | ".join(f"{x:.3f}" for x in b["nll_window_layer"])
                               + f" | {100 * b['acc']:.2f}% | {100 * b['acc_window']:.2f}% |")
            for j, a in enumerate(d["attention"]):
                att.append(f"| {p} | {LABELS[v]} | {j} | {100 * a['off_comb']:.1f}% | {100 * a['far_off_comb']:.1f}% | "
                           f"{d['gates'][j]:.2f} |")
    notes = ["Calibration over positions t ≥ 64 (top-label ECE with 15 bins, Brier score; mean over seeds): " + "; ".join(
        f"P = {p}, {LABELS[v]}: ECE {d['ece']:.3f}, Brier {d['brier']:.4f}" for p in periods for v, d in out[p].items()) + "."]
    if pairs:
        notes.append("Paired NLL difference, RoPE + ArithScope minus ArithScope only, on the same sequences (seed-paired "
                     f"mean with 95% CI; then each of the {data['sequences']} test sequences, averaged over seeds): "
                     + "; ".join(
                         f"P = {p}, {r['start']:,}–{r['end']:,}: {r['seeds']['mean']:+.3f} "
                         f"[{r['seeds']['lo']:+.3f}, {r['seeds']['hi']:+.3f}], sequences "
                         + ", ".join(f"{x:+.3f}" for x in r["per_sequence"])
                         for p in sorted(pairs, key=int) for r in pairs[p] if r["start"] >= window) + ".")
    last = {p: {v: d["buckets"][-1] for v, d in out[p].items()} for p in periods}
    spans = [_span(next(iter(out[periods[0]].values()))["buckets"][i]) for i in (0, -1)]
    notes.append(f"Rank correlation of per-position NLL with the true symbol's vote share on the comb, positions {spans[0]} "
                 f"/ {spans[1]} (an attention average sees vote shares; the Bayes posterior depends on vote counts): "
                 + "; ".join(f"P = {p}, {LABELS[v]}: ρ = {d['buckets'][0]['spearman_nll_share']:+.2f} / "
                             f"{d['buckets'][-1]['spearman_nll_share']:+.2f}" for p in periods for v, d in out[p].items()
                             if d["buckets"][0]["spearman_nll_share"] is not None
                             and d["buckets"][-1]["spearman_nll_share"] is not None)
                 + ". The rank correlation with the Bayes NLL, defined only where the Bayes NLL is not exactly 0, is "
                 + ", ".join(f"{d['buckets'][0]['spearman_nll_bayes']:+.2f}" for p in periods for d in out[p].values()
                             if d["buckets"][0]["spearman_nll_bayes"] is not None)
                 + f" in the same order at positions {spans[0]}.")
    own = [(p, v, _own_share(b)) for p in periods for v, b in last[p].items() if _own_share(b) is not None]
    if own:
        fracs = sorted({f"{100 * b['own_token_wrong']:.0f}%" for p in periods for b in last[p].values()})
        notes.append(f"Positions whose own token is corrupted make up {'–'.join(fracs)} of positions {_last_span(out)}. "
                     "Their share of the NLL that the window removes (shown where the window removes more than 0.01 "
                     "nats): " + "; ".join(f"P = {p}, {LABELS[v]}: {100 * s:.0f}%" for p, v, s in own) + ".")
    depth = len(next(iter(out[periods[0]].values()))["attention"])
    rise = [f"**Post-hoc: change of NLL with query position, per seed** (seeds in order; mean NLL at positions "
            f"{spans[1]} minus positions {_span(next(b for b in next(iter(out[periods[0]].values()))['buckets'] if b['start'] >= window))}"
            f"; keys limited to the last {window:,}):", "",
            "| P | variant | no limit | all layers limited | " + " | ".join(f"layer {j} only" for j in range(depth)) + " |",
            "|---|---|---|---|" + "---|" * depth]
    for p in periods:
        for v, d in out[p].items():
            cells = [d["rise"]["no_limit"], d["rise"]["all_layers"]] + [d["rise"][f"layer_{j}"] for j in range(depth)]
            rise.append(f"| {p} | {LABELS[v]} | " + " | ".join(", ".join(f"{x:+.3f}" for x in c) for c in cells) + " |")
    readout = ROOT / "results" / "posthoc" / "readout_floor.json"
    if readout.exists():
        floors = defaultdict(list)
        for r in json.loads(readout.read_text())["runs"]:
            floors[(str(r["period"]), r["variant"])].append(r["max_floor"])
        for (p, v), xs in floors.items():
            if v in out.get(p, {}):
                out[p][v]["readout_floor"] = float(np.mean(xs))
        notes[0] += ("\n\nReadout limit (`experiments/posthoc_readout.py`, weights only): the lowest NLL that the final "
                     "LayerNorm and linear readout can produce for the hardest target symbol, searched over all "
                     "LayerNorm outputs (mean over seeds): " + "; ".join(
                         f"P = {p}, {LABELS[v]}: {d['readout_floor']:.4f}" for p in periods for v, d in out[p].items()
                         if "readout_floor" in d) + ".")
    summary["posthoc_nll"] = out
    summary["posthoc_nll_pairs"] = pairs
    tables += ["\n".join(nll) + "\n\n" + notes[0], "\n".join(win), "\n".join(rise) + "\n\n" + "\n\n".join(notes[1:]),
               "\n".join(att)]
    for mode in ("light", "dark"):
        plot_posthoc_nll(out, window, data["length"], mode)


POSTHOC_SCALARS = ("acc", "nll", "nll_from_errors", "acc_window", "nll_window", "nll_window_from_errors", "bayes_nll",
                   "own_token_wrong", "nll_from_own_token_wrong", "nll_window_from_own_token_wrong", "nll_top1pct_share")


def _span(b: dict) -> str:
    return f"{b['start']:,}–{b['end']:,}"


def _last_span(out: dict) -> str:
    return _span(next(iter(next(iter(out.values())).values()))["buckets"][-1])


def _own_share(b: dict) -> float | None:
    """Fraction of the window's NLL reduction that comes from positions whose own token is corrupted."""
    drop = b["nll"] - b["nll_window"]
    if drop <= 0.01:
        return None
    return (b["nll_from_own_token_wrong"] - b["nll_window_from_own_token_wrong"]) / drop


def plot_posthoc_nll(out: dict, window: int, length: int, mode: str) -> Path:
    with themed(mode) as th:
        periods = sorted(out, key=int)
        fig, axes = plt.subplots(1, len(periods), figsize=(10.5, 3.9), sharey=True, squeeze=False)
        top = 0.0
        # ArithScope with and without RoPE share a color across all figures; marker and dash tell them apart
        styles = {"rope+fourier_full": ("-", "D", "RoPE + full Fourier bank"),
                  "arith": ((0, (4, 2)), "s", "ArithScope only"), "rope+arith": ("-", "o", "RoPE + ArithScope")}
        for ax, p in zip(axes[0], periods):
            vs = out[p]
            any_v = next(iter(vs.values()))
            x = [math.sqrt(b["start"] * b["end"]) for b in any_v["buckets"]]
            ax.plot(x, [b["bayes_nll"] for b in any_v["buckets"]], color=th["muted"], linewidth=1.2, label="Bayes oracle")
            for v in ("rope+fourier_full", "arith", "rope+arith"):
                if v not in vs:
                    continue
                ls, marker, label = styles[v]
                hero = v == "rope+arith"
                ys = [b["nll"] for b in vs[v]["buckets"]]
                top = max(top, *ys)
                ax.plot(x, ys, color=color(th, v), linestyle=ls, label=label, marker=marker,
                        markersize=5.5 if hero else 5, markeredgecolor=th["surface"], markeredgewidth=1.0,
                        linewidth=2.6 if hero else 2.0, zorder=5 if hero else 3)
            if "rope+arith" in vs:
                ax.plot(x, [b["nll_window"] for b in vs["rope+arith"]["buckets"]], color=color(th, "rope+arith"),
                        linestyle=(0, (1, 1.5)), marker="o", markersize=5, markerfacecolor=th["surface"],
                        markeredgewidth=1.4, linewidth=2.0, zorder=4,
                        label=f"RoPE + ArithScope, keys limited to the last {window:,}")
            ax.axvline(window, color=th["axis"], linewidth=1.0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b['start']:,}–\n{b['end']:,}" for b in any_v["buckets"]], fontsize=8)
            ax.minorticks_off()
            ax.set_xlabel("query position t (bucket)")
            ax.set_title(f"Period P = {p}", loc="left", fontsize=10)
        for ax in axes[0]:  # after every panel is drawn, so that no curve is clipped
            ax.set_ylim(0, 1.12 * top)
            ax.text(window * 1.06, 1.08 * top, f"limit binds beyond t = {window:,}", fontsize=7.5, color=th["ink2"],
                    va="top")
        axes[0][0].set_ylabel("NLL (nats per position)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, -0.05), ncol=3, fontsize=8.5, frameon=False)
        fig.suptitle(f"Post-hoc: NLL by query position at {length:,} tokens, with and without a recent-keys limit "
                     "(trained at 64)",
                     x=0.01, ha="left", fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, "e1_nll", mode)


def analyze_multiplicity(summary: dict, tables: list[str]) -> None:
    """POST-HOC sensitivity: the CI-based confirmatory comparisons with Bonferroni and Holm adjustment."""
    comps = {"C1, P = 12": (_pd(summary, "12", "rope", 1024), 0.0), "C1, P = 13": (_pd(summary, "13", "rope", 1024), 0.0),
             "C2, P = 13": (_pd(summary, "13", "rope+fourier_fund", 1024), 0.0),
             "C4 (margin −1 pp)": (summary.get("e3_tagged", {}).get("c4_rope_arith_minus_rope_N64"), 0.01)}
    if any(p is None or p.get("half") is None for p, _ in comps.values()):
        return
    k = len(comps)
    rows = []
    for name, (p, margin) in comps.items():
        d = np.asarray(p["diffs"])
        n = len(d)
        se = d.std(ddof=1) / math.sqrt(n)
        pval = float(sstats.t.sf((d.mean() + margin) / se, n - 1))  # one-sided, in the claimed direction
        half = float(sstats.t.ppf(1 - 0.05 / (2 * k), n - 1) * se)
        rows.append({"claim": name, "mean": float(d.mean()), "p": pval, "lo": float(d.mean() - half),
                     "hi": float(d.mean() + half), "margin": margin})
    order = sorted(range(k), key=lambda i: rows[i]["p"])
    holm_ok, running = {}, True
    for rank, i in enumerate(order):
        running = running and rows[i]["p"] <= 0.05 / (k - rank)
        holm_ok[i] = running
    lines = ["**Post-hoc sensitivity to multiplicity** (the preregistration used unadjusted 95% intervals; C3 uses "
             "fixed thresholds and is not included). One-sided paired t tests in the claimed direction:", "",
             "| comparison | mean difference | one-sided p | Bonferroni 98.75% interval | holds after Bonferroni | "
             "holds after Holm |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        bonf = r["lo"] > -r["margin"]
        lines.append(f"| {r['claim']} | {100 * r['mean']:+.2f} pp | {r['p']:.1e} | [{100 * r['lo']:+.2f}, "
                     f"{100 * r['hi']:+.2f}] pp | {'yes' if bonf else 'NO'} | {'yes' if holm_ok[i] else 'NO'} |")
        r.update({"bonferroni": bonf, "holm": holm_ok[i]})
    summary["multiplicity"] = rows
    tables.append("\n".join(lines))


def analyze_replay(summary: dict, tables: list[str]) -> None:
    """Replays of committed runs with the current code (experiments/replay.py)."""
    out, lines = {}, []
    for mode, what in (("eval", "E1 checkpoints re-scored on the preregistered test sets at N = 1,024 and 8,192"),
                       ("train", "confirmatory runs retrained from scratch")):
        path = ROOT / "results" / "replay" / f"{mode}.json"
        if path.exists():
            d = json.loads(path.read_text())
            out[mode] = {"identical": d["identical"], "total": d["total"]}
            lines.append(f"- {what}: {d['identical']} of {d['total']} bitwise identical to the committed records.")
    if out:
        summary["replay"] = out
        tables.append("**Replays with the current code** (`experiments/replay.py`):\n\n" + "\n".join(lines))


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
CONFIG_LABELS = {"llm_d64_B16": "8 heads × 64 dims, bank B16", "tiny_d16_M1-16": "4 heads × 16 dims, bank 1–16"}


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
                ax.set_title(f"{CONFIG_LABELS.get(config, config)}: {label}", loc="left", fontsize=9.5)
        axes[0][0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2, fontsize=8.5)
        fig.suptitle("Cost of the ArithScope bias by backend (one layer, inference forward pass, CPU)", x=0.01, ha="left",
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
    if t13.get("rope+arith") and t13.get("rope") and t12.get("rope+arith") and t12.get("rope"):
        L = max(t13["oracle"])
        gap = max(abs(t["rope+arith"][n]["mean"] - t["oracle"][n]) for t in (t12, t13) for n in t["oracle"])
        gap_pp = math.ceil(1000 * gap) / 10  # rounded up to 0.1 pp
        out.append(f"- **Length extrapolation.** Trained on 64 tokens, RoPE + ArithScope stays within {gap_pp:.1f} pp "
                   f"of the Bayes-optimal predictor on unseen positions at every tested length up to {L:,} tokens "
                   f"({L // 64}× the training length), at both periods. At {L:,} tokens RoPE alone scores "
                   f"{fmt(t13['rope'][L]['mean'])}% (period 13) and {fmt(t12['rope'][L]['mean'])}% (period 12), "
                   + (f"against {fmt(t13['oracle'][L])}% for the Bayes oracle."
                      if fmt(t13["oracle"][L]) == fmt(t12["oracle"][L]) else
                      f"against {fmt(t13['oracle'][L])}% and {fmt(t12['oracle'][L])}% for the Bayes oracle."))
        nll = summary.get("E1_nll", {}).get("13", {}).get("table", {})
        if str(L) in nll.get("rope+arith", {}) and str(L) in nll.get("arith", {}):
            out[-1] += (f" These are accuracies. The NLL at {L:,} tokens (period 13) is "
                        f"{nll['rope+arith'][str(L)]['mean']:.3f} nats for RoPE + ArithScope and "
                        f"{nll['arith'][str(L)]['mean']:.3f} for ArithScope alone; see "
                        "[Calibration](#calibration-post-hoc).")
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
            below = [r["alias"] - r["mean"] for r in alias]
            gap_bayes = math.ceil(1000 * max(r["bayes"] - r["mean"] for r in good)) / 10  # rounded up to 0.1 pp
            lcms = ", ".join(sorted({str(r["lcm"]) for r in alias}))
            where = (f"stay below their alias ceiling ({100 * min(below):.1f}–{100 * max(below):.1f} pp below)"
                     if min(below) > 0 else f"end at most {100 * -min(below):.1f} pp above their alias ceiling")
            out.append(f"- **Identifiability theory predicts which banks fail.** In the alias fixture (period {period}), "
                       f"banks with lcm {lcms} {where}, whatever their size; every bank with {period} | lcm ends within "
                       f"{gap_bayes:.1f} pp of the Bayes oracle, including one- and two-order banks.")
    c4 = summary.get("e3_tagged", {}).get("c4_rope_arith_minus_rope_N64")
    if c4 and c4["lo"] is not None:
        out.append(f"- **No measurable cost on aperiodic data.** On tagged denoising, adding ArithScope to RoPE changes "
                   f"accuracy at the training length by {100 * c4['mean']:+.2f} pp (95% CI [{100 * c4['lo']:+.2f}, "
                   f"{100 * c4['hi']:+.2f}]).")
    sh = summary.get("posthoc_shift_n64", {}).get("13", {})
    if sh.get("arith") and sh.get("rope") and sh.get("rope+arith"):
        top = str(max(int(k) for k in sh["arith"]))
        out.append(f"- **Exact at any absolute position (post hoc).** At period 13 and the training length, shifting "
                   f"every position by 10<sup>{len(top) - 1}</sup> leaves RoPE + ArithScope at "
                   f"{fmt(sh['rope+arith'][top]['mean'])}% and ArithScope alone at {fmt(sh['arith'][top]['mean'])}% "
                   f"(unshifted: {fmt(sh['rope+arith']['0']['mean'])}% and {fmt(sh['arith']['0']['mean'])}%). "
                   f"RoPE alone, whose rotary angles are float32, drops from {fmt(sh['rope']['0']['mean'])}% to "
                   f"{fmt(sh['rope'][top]['mean'])}%.")
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
    train = {(r["mode"], r["length"]): r for r in data["rows"] if r["config"] == config and r["backward"]}
    train_txt = ""
    if ("arith_folded", L) in train and "median_seconds" in train[("arith_folded", L)]:
        tf, tb = train[("arith_folded", L)], train.get(("baseline_sdpa", L), {})
        skipped = train.get(("arith_explicit", L), {}).get("skipped", "")
        train_txt = (f" A training step (forward and backward) at N = {L:,} takes {1000 * tf['median_seconds']:.1f} ms "
                     f"and {tf['peak_rss_increase_mb']:.0f} MB with the folded path"
                     + (f", against {1000 * tb['median_seconds']:.1f} ms and {tb['peak_rss_increase_mb']:.0f} MB for "
                        "RoPE alone" if "median_seconds" in tb else "")
                     + (f"; the explicit and dense-mask paths were not run there ({skipped})." if skipped else "."))
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
        f"Peak memory grows by {mems}.{flex_floor}{train_txt} These are single-machine CPU measurements, not a "
        "speed claim; full tables are in [`docs/RESULTS.md`](docs/RESULTS.md).",
    ])


def calibration_md(summary: dict) -> str:
    """README calibration paragraph from the post-hoc NLL diagnosis (empty until it has run)."""
    diag, runs = summary.get("posthoc_nll", {}), summary.get("E1_nll", {})
    if "13" not in diag or "13" not in runs or not {"arith", "rope+arith"} <= set(diag["13"]):
        return ""
    t = runs["13"]["table"]
    ra, ar = diag["13"]["rope+arith"], diag["13"]["arith"]
    beyond = next(b for b in ra["buckets"] if b["start"] >= 1024)  # the first bucket the window affects
    last, ar_last = ra["buckets"][-1], ar["buckets"][-1]
    depth = len(ra["attention"])
    # the layer whose limit alone reproduces the all-layer effect, per model; named only if it is the same everywhere
    rope = [d for vs in diag.values() for v, d in vs.items() if v.startswith("rope")]
    culprits = {int(np.argmin([abs(np.mean(d["rise"][f"layer_{j}"])) for j in range(depth)])) for d in rope}
    j = culprits.pop() if len(culprits) == 1 else None
    rises = [x for d in rope for x in d["rise"]["no_limit"]]
    flat = max(abs(x) for d in rope for x in d["rise"][f"layer_{j}" if j is not None else "all_layers"])
    plain = max(abs(x) for vs in diag.values() for v, d in vs.items() if not v.startswith("rope") for x in d["rise"]["no_limit"])
    ordinal = {0: "first", 1: "second"}.get(j, f"layer-{j}")
    where = (f"to distant keys in the {ordinal} attention layer" if j is not None else "to distant keys")
    limit = (f"limiting the {ordinal} layer to the last 1,024 keys holds it at {last['nll_window_layer'][j]:.3f} with "
             f"accuracy {100 * last['acc_window_layer'][j]:.0f}%, while limiting the "
             + " and ".join({0: "first", 1: "second"}.get(k, f"layer-{k}") for k in range(depth) if k != j)
             + " layer changes nothing" if j is not None else
             f"limiting every layer to the last 1,024 keys holds it at {last['nll_window']:.3f} with accuracy "
             f"{100 * last['acc_window']:.0f}%")
    floor = (f" while its readout alone could go as low as {ar['readout_floor']:.3f}" if "readout_floor" in ar else "")
    limited = last["nll_window_layer"][j] if j is not None else last["nll_window"]
    above = " RoPE + ArithScope stays above it even with the limit." if limited > ar_last["nll"] else ""
    return "\n".join([
        "### Calibration (post hoc)", "",
        '<p align="center">', "  <picture>",
        '    <source media="(prefers-color-scheme: dark)" srcset="results/figures/e1_nll_dark.png">',
        '    <img alt="NLL by query position at length 8192 for ArithScope only, RoPE + ArithScope, and RoPE + full '
        'Fourier bank, with the Bayes NLL and RoPE + ArithScope with attention limited to recent keys" '
        'src="results/figures/e1_nll_light.png" width="920">',
        "  </picture>", "</p>", "",
        f"Accuracy is not the whole story. At N = 8,192 and period 13, RoPE + ArithScope and ArithScope alone both match "
        f"the oracle's accuracy, yet their mean NLL on unseen positions is {t['rope+arith']['8192']['mean']:.3f} and "
        f"{t['arith']['8192']['mean']:.3f} nats (committed run files, 5 seeds). A post-hoc diagnosis on fresh sequences "
        f"traces the part of the gap that grows with length {where}. Beyond 1,024 tokens, RoPE + ArithScope's NLL rises with the query position, from "
        f"{beyond['nll']:.3f} at positions {_span(beyond)} to {last['nll']:.3f} at {_span(last)}, at "
        f"{100 * last['acc']:.0f}% accuracy; {limit}. Across both periods, the NLL rises by more than 0.01 nats in "
        f"{sum(x > 0.01 for x in rises)} of the {len(rises)} models with RoPE (RoPE + ArithScope and RoPE + full Fourier "
        f"bank), and the limit leaves a change of at most {flat:.3f} nats in every one of them. Over the same positions, "
        f"ArithScope alone changes by at most {plain:.3f} nats without any limit, although its second layer also attends "
        f"to distant keys.{above} ArithScope alone has "
        f"a floor of its own: {ar_last['nll']:.3f} nats where the Bayes NLL is 0,{floor}. That fits a softmax average, "
        "which reports the share of votes on the comb rather than their number ([`MATH.md`](docs/MATH.md), P3, T8, "
        "and T8a). This was found after an external audit and is not preregistered; details in "
        "[`docs/RESULTS.md`](docs/RESULTS.md).",
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


E6_ARM_LABELS = {"reference": "ArithScope only", "baseline": "RoPE + ArithScope",
                 "window": "RoPE + ArithScope, 2nd layer limited to 64 keys",
                 "long_train": "RoPE + ArithScope, trained at length 256"}


def analyze_e6(summary: dict, tables: list[str]) -> None:
    """E6 (preregistered, PREREGISTRATION.md Section 9). Verdicts and paired intervals come from
    results/summary/e6.json, written by the frozen experiments/e6_verdicts.py; per-arm intervals and the
    NLL-by-position figure are computed here from the same run files with that script's metric functions."""
    path = ROOT / "results" / "summary" / "e6.json"
    if not path.exists() or not (RUNS / "e6_calibration").exists():
        return
    import experiments.e6_verdicts as e6

    frozen = json.loads(path.read_text())
    runs = e6.load()
    if len(runs) != len(e6.PERIODS) * len(E6_ARM_LABELS) * len(e6.SEEDS):
        return  # incomplete suite: no table until every run exists
    m = {k: e6.metrics(r) for k, r in runs.items()}
    out = {"verdicts": frozen["verdicts"], "holm": frozen["holm"], "periods": {}}
    pct = lambda c: f"{100 * c['mean']:.1f} ± {100 * (c['hi'] - c['mean']):.1f}"  # noqa: E731
    nats = lambda c: f"{c['mean']:.3f} ± {c['hi'] - c['mean']:.3f}"  # noqa: E731
    diff = lambda c: f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"  # noqa: E731
    lines = ["**E6, long-length calibration (preregistered, Section 9).** N = 8,192; far region = positions "
             "4,096–8,191; growth = NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. Mean ± 95% CI "
             "half-width over 5 seeds.", "",
             "| P | arm | far NLL (nats) | far accuracy (%) | growth (nats) |", "|---|---|---|---|---|"]
    curves = {}
    for p in e6.PERIODS:
        cell, frozen_p = {}, frozen["periods"][str(p)]
        for arm, label in E6_ARM_LABELS.items():
            stats = {k: e6.ci95([m[(p, arm, s)][k] for s in e6.SEEDS]) for k in ("far_nll", "far_acc", "growth")}
            cell[arm] = stats
            lines.append(f"| {p} | {label} | {nats(stats['far_nll'])} | {pct(stats['far_acc'])} | "
                         f"{nats(stats['growth'])} |")
            test = [next(t for t in runs[(p, arm, s)]["test"] if t["length"] == e6.LENGTH) for s in e6.SEEDS]
            curves.setdefault(str(p), {})[arm] = {
                "buckets": [(b["start"], b["end"]) for b in test[0]["by_position"]],
                "nll": [e6.ci95([t["by_position"][i]["nll"] for t in test]) for i in range(len(test[0]["by_position"]))]}
        out["periods"][str(p)] = {"arms": cell, **{arm: frozen_p[arm] for arm in ("window", "long_train")}}
    lines += ["", "Paired differences against RoPE + ArithScope (mean [95% CI], seeds lower / 5):", ""]
    for p in e6.PERIODS:
        for arm in ("window", "long_train"):
            c = frozen["periods"][str(p)][arm]
            lines.append(f"- P = {p}, {E6_ARM_LABELS[arm]}: far NLL {diff(c['far_nll_vs_baseline'])} "
                         f"({c['far_nll_vs_baseline']['lower']}/5); far accuracy "
                         f"{100 * c['far_acc_vs_baseline']['mean']:+.2f} pp [{100 * c['far_acc_vs_baseline']['lo']:+.2f}, "
                         f"{100 * c['far_acc_vs_baseline']['hi']:+.2f}]; growth {diff(c['growth_vs_baseline'])}; far NLL "
                         f"minus ArithScope only {diff(c['far_nll_vs_reference'])}.")
    lines += ["", "Holm-adjusted one-sided p-values for the four far-NLL comparisons (sensitivity only): "
              + "; ".join(f"{k.replace('_p', ', P = ')}: {v['p_holm']:.4f}" for k, v in
                          sorted(frozen["holm"].items())) + "."]
    tables.append("\n".join(lines))
    out["curves"] = curves
    summary["E6"] = out
    for mode in ("light", "dark"):
        plot_e6(curves, mode)


def plot_e6(curves: dict, mode: str) -> Path:
    with themed(mode) as th:
        periods = sorted(curves, key=int)
        fig, axes = plt.subplots(1, len(periods), figsize=(10.5, 3.9), sharey=True, squeeze=False)
        # ArithScope keeps its color in every figure; the remedies use free slots, with marker and dash as well
        styles = {"reference": (color(th, "arith"), (0, (4, 2)), "s"), "baseline": (color(th, "rope+arith"), "-", "o"),
                  "window": (th["series"][6], "-", "^"), "long_train": (th["series"][7], "-", "D")}
        for ax, p in zip(axes[0], periods):
            any_arm = next(iter(curves[p].values()))
            x = [(a + b) / 2 for a, b in any_arm["buckets"]]  # equal-width buckets: a linear axis
            for arm, label in E6_ARM_LABELS.items():
                c, ls, marker = styles[arm]
                ys = [v["mean"] for v in curves[p][arm]["nll"]]
                ax.fill_between(x, [v["lo"] for v in curves[p][arm]["nll"]], [v["hi"] for v in curves[p][arm]["nll"]],
                                color=c, alpha=0.15, linewidth=0)
                ax.plot(x, ys, color=c, linestyle=ls, marker=marker, markersize=5, markeredgecolor=th["surface"],
                        markeredgewidth=1.0, label=label, linewidth=2.6 if arm == "baseline" else 2.0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a // 1024}–{b // 1024}k" for a, b in any_arm["buckets"]], fontsize=8)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("query position t (bucket)")
            ax.set_title(f"Period P = {p}", loc="left", fontsize=10)
        axes[0][0].set_ylabel("NLL (nats per position)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, -0.05), ncol=2, fontsize=8.5, frameon=False)
        fig.suptitle("E6 (preregistered): NLL by query position at 8,192 tokens, trained at 64 unless stated "
                     "(mean and 95% CI over 5 seeds)", x=0.01, ha="left", fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, "e6_nll", mode)


def e6_findings(summary: dict) -> str:
    """E6 verdicts (PREREGISTRATION.md Section 9), as computed by experiments/e6_verdicts.py."""
    e = summary.get("E6")
    if not e:
        return "(results pending)"
    v, ps = e["verdicts"], e["periods"]

    def far(arm: str) -> str:
        return "; ".join(f"P = {p}: far NLL {c[arm]['far_nll_vs_baseline']['mean']:+.3f}, 95% CI "
                         f"[{c[arm]['far_nll_vs_baseline']['lo']:+.3f}, {c[arm]['far_nll_vs_baseline']['hi']:+.3f}], "
                         f"lower in {c[arm]['far_nll_vs_baseline']['lower']}/5 seeds; accuracy lower bound "
                         f"{100 * c[arm]['far_acc_vs_baseline']['lo']:+.2f} pp" for p, c in ps.items())

    def growth(arm: str) -> str:
        return ", ".join(f"P = {p}: {c['arms'][arm]['growth']['mean']:+.4f}" for p, c in ps.items())

    ref = "; ".join(f"P = {p}: {c['window']['far_nll_vs_reference']['mean']:+.3f}, 95% CI "
                    f"[{c['window']['far_nll_vs_reference']['lo']:+.3f}, {c['window']['far_nll_vs_reference']['hi']:+.3f}]"
                    for p, c in ps.items())
    return "\n".join([
        f"- **Sanity check (window and baseline train bitwise alike): "
        f"{'passed' if v['sanity_window_training_identical'] else 'FAILED'}.**",
        f"- **L1 (second-layer window equal to the training length): {v['L1_window'].lower()}.** Window minus "
        f"RoPE + ArithScope: {far('window')}.",
        f"- **L2 (training at length 256, same tokens per step): {v['L2_long_train'].lower()}.** Longer training minus "
        f"RoPE + ArithScope: {far('long_train')}.",
        f"- **S3 (window flat, mean growth within ±0.02 nats): {'as predicted' if v['S3_window_flat'] else 'not as predicted'}.** "
        f"Growth {growth('window')} (RoPE + ArithScope: {growth('baseline')}).",
        f"- **S4 (window stays above ArithScope only): {'as predicted' if v['S4_window_above_reference'] else 'not as predicted'}.** "
        f"Window minus ArithScope only, far NLL: {ref}.",
        f"- **S5 (longer training still grows): {'as predicted' if v['S5_long_train_still_grows'] else 'not as predicted'}.** "
        f"Growth {growth('long_train')}.",
    ])


E7_ARM_LABELS = {"reference": "ArithScope only", "arith_window": "ArithScope only, 2nd layer limited to 64 keys",
                 "window": "RoPE + ArithScope, 2nd layer limited to 64 keys"}


def analyze_e7(summary: dict, tables: list[str]) -> None:
    """E7 (preregistered, PREREGISTRATION.md Section 10). Verdicts and paired intervals come from
    results/summary/e7.json, written by the frozen experiments/e7_verdicts.py; per-arm intervals and the
    NLL-by-position figure are computed here from the same run files with E6's metric functions."""
    path = ROOT / "results" / "summary" / "e7.json"
    if not path.exists() or not (RUNS / "e7_floor").exists():
        return
    import experiments.e6_verdicts as e6
    import experiments.e7_verdicts as e7

    frozen = json.loads(path.read_text())
    runs = e7.load()
    if len(runs) != len(e6.PERIODS) * len(E7_ARM_LABELS) * len(e6.SEEDS):
        return  # incomplete suite: no table until every run exists
    m = {k: e6.metrics(r) for k, r in runs.items()}
    out = {"verdicts": frozen["verdicts"], "holm": frozen["holm"], "periods": {}}
    pct = lambda c: f"{100 * c['mean']:.1f} ± {100 * (c['hi'] - c['mean']):.1f}"  # noqa: E731
    nats = lambda c: f"{c['mean']:.4f} ± {c['hi'] - c['mean']:.4f}"  # noqa: E731
    diff = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    lines = ["**E7, the NLL floor of ArithScope alone (preregistered, Section 10).** N = 8,192; far region = "
             "positions 4,096–8,191; growth = NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. "
             "Mean ± 95% CI half-width over 5 seeds.", "",
             "| P | arm | far NLL (nats) | far accuracy (%) | growth (nats) |", "|---|---|---|---|---|"]
    curves = {}
    for p in e6.PERIODS:
        cell = {}
        for arm, label in E7_ARM_LABELS.items():
            stats = {k: e6.ci95([m[(p, arm, s)][k] for s in e6.SEEDS]) for k in ("far_nll", "far_acc", "growth")}
            cell[arm] = stats
            lines.append(f"| {p} | {label} | {nats(stats['far_nll'])} | {pct(stats['far_acc'])} | "
                         f"{nats(stats['growth'])} |")
            test = [next(t for t in runs[(p, arm, s)]["test"] if t["length"] == e6.LENGTH) for s in e6.SEEDS]
            curves.setdefault(str(p), {})[arm] = {
                "buckets": [(b["start"], b["end"]) for b in test[0]["by_position"]],
                "nll": [e6.ci95([t["by_position"][i]["nll"] for t in test]) for i in range(len(test[0]["by_position"]))]}
        out["periods"][str(p)] = {"arms": cell, **{k: v for k, v in frozen["periods"][str(p)].items() if k != "means"}}
    lines += ["", "Paired differences in far NLL (mean [95% CI], seeds lower / 5):", ""]
    for p in e6.PERIODS:
        c = frozen["periods"][str(p)]
        aw = c["arith_window_vs_reference"]
        lines.append(f"- P = {p}: ArithScope only, limited minus unlimited: {diff(aw['far_nll'])} "
                     f"({aw['far_nll']['lower']}/5); far accuracy {100 * aw['far_acc']['mean']:+.2f} pp "
                     f"[{100 * aw['far_acc']['lo']:+.2f}, {100 * aw['far_acc']['hi']:+.2f}]. "
                     f"ArithScope only limited minus RoPE + ArithScope limited: {diff(c['arith_window_vs_window']['far_nll'])} "
                     f"({c['arith_window_vs_window']['far_nll']['lower']}/5). RoPE + ArithScope limited minus ArithScope "
                     f"only: {diff(c['window_vs_reference']['far_nll'])} ({c['window_vs_reference']['far_nll']['lower']}/5). "
                     f"Share of the gap between ArithScope only and RoPE + ArithScope limited that the limit closes on "
                     f"ArithScope only (from means, descriptive): {100 * c['gap_closed_share']:.0f}%.")
    lines += ["", "Holm-adjusted one-sided p-values for W1's two far-NLL comparisons (sensitivity only): "
              + "; ".join(f"P = {k.rsplit('_p', 1)[1]}: {v['p_holm']:.4f}" for k, v in sorted(frozen["holm"].items())) + "."]
    tables.append("\n".join(lines))
    out["curves"] = curves
    summary["E7"] = out
    for mode in ("light", "dark"):
        plot_e7(curves, mode)


def plot_e7(curves: dict, mode: str, labels: dict | None = None, name: str = "e7_nll", title: str | None = None) -> Path:
    labels = labels or E7_ARM_LABELS
    with themed(mode) as th:
        periods = sorted(curves, key=int)
        fig, axes = plt.subplots(1, len(periods), figsize=(10.5, 3.9), sharey=True, squeeze=False)
        # ArithScope keeps its color; its limited version shares it with a solid line; E6's window arm keeps its slot
        styles = {"reference": (color(th, "arith"), (0, (4, 2)), "s"), "arith_window": (color(th, "arith"), "-", "v"),
                  "window": (th["series"][6], "-", "^")}
        for ax, p in zip(axes[0], periods):
            any_arm = next(iter(curves[p].values()))
            x = [(a + b) / 2 for a, b in any_arm["buckets"]]  # equal-width buckets: a linear axis
            for arm, label in labels.items():
                c, ls, marker = styles[arm]
                ax.fill_between(x, [v["lo"] for v in curves[p][arm]["nll"]], [v["hi"] for v in curves[p][arm]["nll"]],
                                color=c, alpha=0.15, linewidth=0)
                ax.plot(x, [v["mean"] for v in curves[p][arm]["nll"]], color=c, linestyle=ls, marker=marker,
                        markersize=5, markeredgecolor=th["surface"], markeredgewidth=1.0, label=label, linewidth=2.0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a // 1024}–{b // 1024}k" for a, b in any_arm["buckets"]], fontsize=8)
            ax.set_xlabel("query position t (bucket)")
            ax.set_title(f"Period P = {p}", loc="left", fontsize=10)
        axes[0][0].set_ylim(bottom=0)  # after every panel is drawn, so the shared top fits all of them
        axes[0][0].set_ylabel("NLL (nats per position)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, -0.05), ncol=2, fontsize=8.5, frameon=False)
        fig.suptitle(title or "E7 (preregistered): NLL by query position at 8,192 tokens, trained at 64 "
                     "(mean and 95% CI over 5 seeds)", x=0.01, ha="left", fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, name, mode)


def e7_findings(summary: dict) -> str:
    """E7 verdicts (PREREGISTRATION.md Section 10), as computed by experiments/e7_verdicts.py."""
    e = summary.get("E7")
    if not e:
        return "(results pending)"
    v, ps = e["verdicts"], e["periods"]

    def far(key: str, acc: bool = False) -> str:
        parts = []
        for p, c in ps.items():
            d = c[key]["far_nll"]
            s = f"P = {p}: {d['mean']:+.4f}, 95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}], negative in {d['lower']}/5 seeds"
            if acc:
                s += f"; accuracy lower bound {100 * c[key]['far_acc']['lo']:+.2f} pp"
            parts.append(s)
        return "; ".join(parts)

    short = {"reference": "unlimited", "arith_window": "limited", "window": "RoPE + ArithScope limited"}
    means = "; ".join(f"P = {p}: " + ", ".join(f"{short[a]} {c['arms'][a]['far_nll']['mean']:.4f}" for a in short)
                      for p, c in ps.items())
    w1_word, w1_why = (v["W1_arith_window_below_reference"].split(" ", 1) + [""])[:2]
    share = ", ".join(f"P = {p}: {100 * c['gap_closed_share']:.0f}%" for p, c in ps.items())
    return "\n".join([
        f"- **Sanity check (limited and unlimited ArithScope only train bitwise alike): "
        f"{'passed' if v['sanity_arith_window_training_identical'] else 'FAILED'}.**",
        f"- **W1 (ArithScope only, second layer limited to 64 keys, has lower far NLL): "
        f"{w1_word.lower()}** {w1_why}. Limited minus unlimited: {far('arith_window_vs_reference', True)}.",
        f"- **S6 (the limit closes only part of the gap to RoPE + ArithScope limited): "
        f"{'as predicted' if v['S6_arith_window_above_window'] else 'not as predicted'}.** ArithScope only limited minus "
        f"RoPE + ArithScope limited: {far('arith_window_vs_window')}.",
        f"- **S7 (E6's surprise replicates: RoPE + ArithScope limited ends below ArithScope only): "
        f"{'as predicted' if v['S7_window_below_reference'] else 'not as predicted'}.** {far('window_vs_reference')}.",
        f"- **Descriptive.** Mean far NLL of ArithScope only, unlimited and limited, and of RoPE + ArithScope limited: {means}. Share of the gap closed by the limit on ArithScope only: {share}.",
    ])


E8_ARM_LABELS = {k: E7_ARM_LABELS[k] for k in ("reference", "arith_window")}


def analyze_e8(summary: dict, tables: list[str]) -> None:
    """E8 (preregistered, PREREGISTRATION.md Section 11), a powered retest of E7's W1. Verdicts and paired
    intervals come from results/summary/e8.json, written by the frozen experiments/e8_verdicts.py; per-arm
    intervals and the NLL-by-position figure are computed here from the same run files with E6's metric
    functions, over the seeds that the verdict script used."""
    path = ROOT / "results" / "summary" / "e8.json"
    if not path.exists() or not (RUNS / "e8_power_retest").exists():
        return
    import experiments.e6_verdicts as e6
    import experiments.e8_verdicts as e8

    frozen = json.loads(path.read_text())
    plan = json.loads((ROOT / "results" / "summary" / "e8_power.json").read_text())
    runs, failed = e8.load()
    m = {k: e6.metrics(r) for k, r in runs.items()}
    out = {"verdicts": frozen["verdicts"], "holm": frozen["holm"], "failed_runs": failed,
           "planned_seeds": frozen["planned_seeds"], "plan": {k: plan[k] for k in ("n_seeds", "joint_power_at_n")},
           "periods": {}}
    pct = lambda c: f"{100 * c['mean']:.1f} ± {100 * (c['hi'] - c['mean']):.1f}"  # noqa: E731
    nats = lambda c: f"{c['mean']:.4f} ± {c['hi'] - c['mean']:.4f}"  # noqa: E731
    diff = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    counts = sorted({len(frozen["periods"][str(p)]["seeds"]) for p in e6.PERIODS})
    lines = ["**E8, a powered retest of E7's W1 (preregistered, Section 11).** N = 8,192; far region = positions "
             "4,096–8,191; growth = NLL on positions 7,168–8,191 minus NLL on positions 1,024–2,047. Mean ± 95% CI "
             f"half-width over {' or '.join(map(str, counts))} seeds (seeds {frozen['planned_seeds'][0]}–"
             f"{frozen['planned_seeds'][-1]}; {len(failed)} failed runs). Seed-to-seed SD of far NLL in the last column.",
             "", "| P | arm | far NLL (nats) | far accuracy (%) | growth (nats) | SD of far NLL |", "|---|---|---|---|---|---|"]
    curves = {}
    for p in e6.PERIODS:
        c = frozen["periods"][str(p)]
        seeds, cell = c["seeds"], {}
        for arm, label in E8_ARM_LABELS.items():
            stats = {k: e6.ci95([m[(p, arm, s)][k] for s in seeds]) for k in ("far_nll", "far_acc", "growth")}
            cell[arm] = stats
            lines.append(f"| {p} | {label} | {nats(stats['far_nll'])} | {pct(stats['far_acc'])} | "
                         f"{nats(stats['growth'])} | {c['far_nll_sd'][arm]:.4f} |")
            test = [next(t for t in runs[(p, arm, s)]["test"] if t["length"] == e6.LENGTH) for s in seeds]
            curves.setdefault(str(p), {})[arm] = {
                "buckets": [(b["start"], b["end"]) for b in test[0]["by_position"]],
                "nll": [e6.ci95([t["by_position"][i]["nll"] for t in test]) for i in range(len(test[0]["by_position"]))]}
        out["periods"][str(p)] = {"arms": cell, **{k: v for k, v in c.items() if k != "means"}}
    lines += ["", "Paired differences, limited minus unlimited (mean [95% CI], seeds lower / seeds):", ""]
    for p in e6.PERIODS:
        aw = frozen["periods"][str(p)]["arith_window_vs_reference"]
        n = aw["far_nll"]["n"]
        lines.append(f"- P = {p}: far NLL {diff(aw['far_nll'])} ({aw['far_nll']['lower']}/{n}); far accuracy "
                     f"{100 * aw['far_acc']['mean']:+.2f} pp [{100 * aw['far_acc']['lo']:+.2f}, "
                     f"{100 * aw['far_acc']['hi']:+.2f}]; growth {diff(aw['growth'])} ({aw['growth']['lower']}/{n}).")
    lines += ["", "Holm-adjusted one-sided p-values for W2's two far-NLL comparisons (sensitivity only): "
              + "; ".join(f"P = {k.rsplit('_p', 1)[1]}: {v['p_holm']:.2g}" for k, v in sorted(frozen["holm"].items())) + "."]
    tables.append("\n".join(lines))
    out["curves"] = curves
    summary["E8"] = out
    for mode in ("light", "dark"):
        plot_e7(curves, mode, labels=E8_ARM_LABELS, name="e8_nll",
                title=f"E8 (preregistered): NLL by query position at 8,192 tokens, trained at 64 "
                      f"(mean and 95% CI over {' or '.join(map(str, counts))} seeds)")


def e8_findings(summary: dict) -> str:
    """E8 verdicts (PREREGISTRATION.md Section 11), as computed by experiments/e8_verdicts.py."""
    e = summary.get("E8")
    if not e:
        return "(results pending)"
    v, ps = e["verdicts"], e["periods"]
    w2_word, w2_why = (v["W2_arith_window_below_reference"].split(" ", 1) + [""])[:2]

    def far() -> str:
        parts = []
        for p, c in ps.items():
            d, a = c["arith_window_vs_reference"]["far_nll"], c["arith_window_vs_reference"]["far_acc"]
            parts.append(f"P = {p}: {d['mean']:+.4f}, 95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}], negative in "
                         f"{d['lower']}/{d['n']} seeds; accuracy lower bound {100 * a['lo']:+.2f} pp")
        return "; ".join(parts)

    sds = "; ".join(f"P = {p}: unlimited {c['far_nll_sd']['reference']:.4f}, limited {c['far_nll_sd']['arith_window']:.4f} "
                    f"(ratio {c['far_nll_sd']['reference'] / c['far_nll_sd']['arith_window']:.1f})" for p, c in ps.items())
    growth = "; ".join(f"P = {p}: {c['growth']['arith_window']['mean']:+.4f}" for p, c in ps.items())
    means = "; ".join(f"P = {p}: unlimited {c['arms']['reference']['far_nll']['mean']:.4f}, limited "
                      f"{c['arms']['arith_window']['far_nll']['mean']:.4f}" for p, c in ps.items())
    failed = (f"{len(e['failed_runs'])} failed runs, reported and not replaced" if e["failed_runs"] else "no failed runs")
    return "\n".join([
        f"- **Design.** {e['plan']['n_seeds']} fresh seeds ({e['planned_seeds'][0]}–{e['planned_seeds'][-1]}) per arm "
        f"and period, from the preregistered power rule (planned joint power {e['plan']['joint_power_at_n']:.2f}); "
        f"{failed}.",
        f"- **Sanity check (limited and unlimited ArithScope only train bitwise alike): "
        f"{'passed' if v['sanity_arith_window_training_identical'] else 'FAILED'}.**",
        f"- **W2 (E7's W1 retested: ArithScope only, second layer limited to 64 keys, has lower far NLL): "
        f"{w2_word.lower()}**{f' {w2_why}' if w2_why else ''}. Limited minus unlimited: {far()}.",
        f"- **S8 (seed-to-seed SD of far NLL at least 3× larger without the limit): "
        f"{'as predicted' if v['S8_spread_in_reference'] else 'not as predicted'}.** {sds}.",
        f"- **S9 (limited model flat, mean growth within ±0.02 nats): "
        f"{'as predicted' if v['S9_arith_window_flat'] else 'not as predicted'}.** Growth {growth}.",
        f"- **Descriptive.** Mean far NLL: {means}.",
    ])


def analyze_e9(summary: dict, tables: list[str]) -> None:
    """E9 (preregistered, PREREGISTRATION.md Section 12), a paired window dose-response. Verdicts, slopes, and
    step intervals come from results/summary/e9.json, written by the frozen experiments/e9_verdicts.py; the
    per-seed curves for the figure are read from the same window records with E6's metric functions."""
    path = ROOT / "results" / "summary" / "e9.json"
    if not path.exists() or not (RUNS / "e9_paired_window_eval").exists():
        return
    import experiments.e6_verdicts as e6
    import experiments.e9_verdicts as e9

    frozen = json.loads(path.read_text())
    plan = json.loads((ROOT / "results" / "summary" / "e9_power.json").read_text())
    records, failed, _ = e9.load()
    m = {k: e6.metrics(r) for k, r in records.items()}
    windows = frozen["windows"]
    out = {"verdicts": frozen["verdicts"], "holm": frozen["holm"], "failed": failed, "windows": windows,
           "planned_seeds": frozen["planned_seeds"], "plan": {k: plan[k] for k in ("n_seeds", "joint_power_at_n")},
           "sanity": frozen["sanity"], "periods": {}}
    nats = lambda c: f"{c['mean']:.4f} ± {c['hi'] - c['mean']:.4f}"  # noqa: E731
    diff = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    counts = sorted({len(frozen["periods"][str(p)]["seeds"]) for p in e6.PERIODS})
    lines = ["**E9, a paired window dose-response (preregistered, Section 12).** ArithScope only, trained once per "
             "seed; the same weights evaluated at N = 8,192 with the last layer limited to its last W keys. Far NLL = "
             f"mean NLL on positions 4,096–8,191. Mean ± 95% CI half-width over {' or '.join(map(str, counts))} seeds "
             f"(seeds {frozen['planned_seeds'][0]}–{frozen['planned_seeds'][-1]}; {len(failed)} failed).", "",
             "| P | " + " | ".join(f"W = {w:,}" for w in windows) + " | slope per doubling of W |",
             "|---|" + "---|" * (len(windows) + 1)]
    curves = {}
    for p in e6.PERIODS:
        c = frozen["periods"][str(p)]
        lines.append(f"| {p} | " + " | ".join(nats(c["far_nll_by_window"][str(w)]) for w in windows)
                     + f" | {diff(c['slope'])} |")
        curves[str(p)] = {"windows": windows, "mean": [c["far_nll_by_window"][str(w)] for w in windows],
                          "seeds": [[m[(p, s, w)]["far_nll"] for w in windows] for s in c["seeds"]]}
        out["periods"][str(p)] = c
    lines += ["", "Steps between adjacent windows, far NLL (mean [95% CI]); W = 64 minus the whole sequence; far "
              "accuracy, W = 64 minus the whole sequence:", ""]
    for p in e6.PERIODS:
        c = frozen["periods"][str(p)]
        steps = "; ".join(f"{st['from']:,}→{st['to']:,} {diff(st)}" for st in c["steps"])
        lines.append(f"- P = {p}: {steps}. W = 64 minus whole: {diff(c['nll_w64_minus_full'])}. Accuracy: "
                     f"{100 * c['acc_w64_minus_full']['mean']:+.2f} pp [{100 * c['acc_w64_minus_full']['lo']:+.2f}, "
                     f"{100 * c['acc_w64_minus_full']['hi']:+.2f}].")
    lines += ["", "Holm-adjusted one-sided p-values for the two slope tests (sensitivity only): "
              + "; ".join(f"P = {k.rsplit('_p', 1)[1]}: {v['p_holm']:.2g}" for k, v in sorted(frozen["holm"].items())) + "."]
    tables.append("\n".join(lines))
    out["curves"] = curves
    summary["E9"] = out
    for mode in ("light", "dark"):
        plot_e9(curves, mode)


def plot_e9(curves: dict, mode: str) -> Path:
    with themed(mode) as th:
        periods = sorted(curves, key=int)
        fig, axes = plt.subplots(1, len(periods), figsize=(10.5, 3.9), sharey=True, squeeze=False)
        c = color(th, "arith")
        for ax, p in zip(axes[0], periods):
            cv = curves[p]
            x = [math.log2(w) for w in cv["windows"]]
            for ys in cv["seeds"]:  # one thin line per seed: the four windows share its weights
                ax.plot(x, ys, color=th["muted"], linewidth=0.8, alpha=0.45)
            ax.fill_between(x, [v["lo"] for v in cv["mean"]], [v["hi"] for v in cv["mean"]], color=c, alpha=0.18,
                            linewidth=0)
            ax.plot(x, [v["mean"] for v in cv["mean"]], color=c, marker="s", markersize=5.5,
                    markeredgecolor=th["surface"], markeredgewidth=1.0, linewidth=2.6, label="mean and 95% CI")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{w:,}" for w in cv["windows"]])
            ax.minorticks_off()
            ax.set_xlabel("last-layer window W (keys; log scale)")
            ax.set_title(f"Period P = {p}", loc="left", fontsize=10)
        top = max(y for cv in curves.values() for ys in cv["seeds"] for y in ys)
        axes[0][0].set_ylim(0, 1.05 * top)  # shared axis: after every panel, so that no seed is clipped
        axes[0][0].set_ylabel("far NLL, positions 4,096–8,191 (nats)")
        axes[0][0].plot([], [], color=th["muted"], linewidth=0.8, label="one seed")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, -0.05), ncol=2, fontsize=8.5, frameon=False)
        n = len(curves[periods[0]]["seeds"])
        fig.suptitle(f"E9 (preregistered): far NLL of ArithScope only against its last layer's window, N = 8,192 "
                     f"({n} seeds per period)", x=0.01, ha="left", fontweight="bold", fontsize=11, color=th["ink"])
        return save(fig, FIG, "e9_dose", mode)


def e9_findings(summary: dict) -> str:
    """E9 verdicts (PREREGISTRATION.md Section 12), as computed by experiments/e9_verdicts.py."""
    e = summary.get("E9")
    if not e:
        return "(results pending)"
    v, ps = e["verdicts"], e["periods"]
    r1_word, r1_why = (v["R1_window_dose_response"].split(" ", 1) + [""])[:2]
    slopes = "; ".join(f"P = {p}: {c['slope']['mean']:+.4f} nats per doubling, 95% CI [{c['slope']['lo']:+.4f}, "
                       f"{c['slope']['hi']:+.4f}], positive in {sum(x > 0 for x in c['slope']['values'])}/"
                       f"{c['slope']['n']} seeds; accuracy lower bound {100 * c['acc_w64_minus_full']['lo']:+.2f} pp"
                       for p, c in ps.items())
    worst = "; ".join(f"P = {p}: smallest step bound {min(st['lower_s10'] for st in c['steps']):+.4f}"
                      for p, c in ps.items())
    ends = "; ".join(f"P = {p}: {c['nll_w64_minus_full']['mean']:+.4f}, 95% CI [{c['nll_w64_minus_full']['lo']:+.4f}, "
                     f"{c['nll_w64_minus_full']['hi']:+.4f}]" for p, c in ps.items())
    means = "; ".join(f"P = {p}: " + ", ".join(f"{c['far_nll_by_window'][str(w)]['mean']:.4f}" for w in e["windows"])
                      for p, c in ps.items())
    sane = all(x for k, x in e["sanity"].items() if k != "code_commits")
    failed = f"{len(e['failed'])} failed, reported and not replaced" if e["failed"] else "no failures"
    return "\n".join([
        f"- **Design.** {e['plan']['n_seeds']} fresh seeds ({e['planned_seeds'][0]}–{e['planned_seeds'][-1]}) per "
        f"period from the preregistered power rule (planned joint power {e['plan']['joint_power_at_n']:.2f}), each "
        f"trained once and evaluated with windows {', '.join(f'{w:,}' for w in e['windows'])}; {failed}.",
        f"- **Sanity checks (one checkpoint per seed across windows, one test set per period, one code commit): "
        f"{'passed' if sane else 'FAILED'}.**",
        f"- **R1 (far NLL rises with log2 of the last layer's window): {r1_word.lower()}**"
        f"{f' {r1_why}' if r1_why else ''}. Slope: {slopes}.",
        f"- **S10 (near-monotone, every step's lower bound above −0.005 nats): "
        f"{'as predicted' if v['S10_near_monotone'] else 'not as predicted'}.** {worst}.",
        f"- **S11 (W = 64 below the whole sequence; E8 replicates): "
        f"{'as predicted' if v['S11_w64_below_full'] else 'not as predicted'}.** {ends}.",
        f"- **Descriptive.** Mean far NLL at W = {', '.join(f'{w:,}' for w in e['windows'])}: {means}.",
    ])


E10_ARM_LABELS = {"arith": "ArithScope only", "arith_ev": "ArithScope + evidence head",
                  "arith_evfund": "ArithScope + head, fundamental cosines", "rope": "RoPE",
                  "rope_ev": "RoPE + evidence head"}
E10_ARM_SLOT = {"arith": 0, "arith_ev": 2, "arith_evfund": 3, "rope": 1, "rope_ev": 4}
E10_FLOOR = 1e-12  # plotting floor of the log axis: later buckets of the exact Bayes NLL and of the head underflow


def _e10_bayes(rec: dict, length: int, copy_rec: dict | None = None) -> dict:
    """Exact Bayes NLL of the regenerated test batch of ``rec`` (``copy_rec``: its copied-law batch): the mean over
    all positions and over the record's position buckets, after checking the batch digest against the stored record."""
    import experiments.e10_verdicts as e10
    from arithscope.tasks import make_task, periodic_bayes_nll
    from arithscope.train import _generator, task_dict
    from experiments.e10_copy_eval import copied_task

    spec = e10.spec_of(rec)
    if copy_rec is None:
        task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
        test = next(t for t in rec["test"] if t["length"] == length)
    else:
        task, test = copied_task(spec, copy_rec["copied_task"]["copies"]), copy_rec["test"][0]
    num = max(2, (spec.eval_tokens if length <= 2048 else spec.eval_tokens_long) // length)
    parts = ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), length)
    batch = task.sample(num, length, _generator(*parts))
    if batch.digest() != test["digest"]:
        raise ValueError(f"regenerated test batch differs from the record's digest: {spec.filename}")
    nll = periodic_bayes_nll(batch, task)
    return {"all": float(nll.mean()),
            "buckets": [float(nll[:, b["start"]:b["end"]].mean()) for b in test["by_position"]]}


def e10_mean(m: dict, p: int, arm: str, key: str, seeds) -> float | None:
    vals = [m[(p, arm, s)][key] for s in seeds if key in m.get((p, arm, s), {})]
    return sum(vals) / len(vals) if vals else None


def analyze_e10(summary: dict, tables: list[str]) -> None:
    """E10 (preregistered, PREREGISTRATION.md Section 13), an evidence-count readout. Verdicts and paired intervals
    come from results/summary/e10.json, written by the frozen experiments/e10_verdicts.py; the figure reads the same
    run and copied-law records and recomputes the exact Bayes NLL of its test batches."""
    path = ROOT / "results" / "summary" / "e10.json"
    if not path.exists() or not (RUNS / "e10_evidence_copy_eval").exists():
        return
    import experiments.e10_verdicts as e10

    frozen = json.loads(path.read_text())
    plan = json.loads((ROOT / "results" / "summary" / "e10_power.json").read_text())
    loaded = e10.load()
    m = e10.collect(loaded)
    seeds = frozen["planned_seeds"]
    out = {"verdicts": frozen["verdicts"], "holm": frozen["holm"], "failed": frozen["failed"],
           "planned_seeds": seeds,  # V1 is the claim that binds the seed count
           "plan": {"n_seeds": plan["n_seeds"], "joint_power_at_n": plan["claims"]["V1"]["joint_power_at_n"]},
           "sanity": frozen["sanity"], "periods": frozen["periods"]}
    lines = ["**E10, an evidence-count readout (preregistered, Section 13).** Mean over seeds "
             f"{seeds[0]}–{seeds[-1]} ({len(frozen['failed'])} failed). Far = positions 4,096–8,191 of N = 8,192; "
             "excess = NLL at N = 1,024 minus the exact Bayes NLL; independent = ordinary law at N = 256; "
             "copied = copied law (k = 4) at N = 256.", "",
             "| P | arm | far NLL | far accuracy | excess NLL at 1,024 | independent NLL at 256 | copied NLL at 256 |",
             "|---|---|---|---|---|---|---|"]
    bayes, curves = {}, {}
    for p in e10.PERIODS:
        rec = loaded["runs"][(p, "arith", seeds[0])]  # one test digest per period and length (sanity check)
        bayes[p] = {"buckets_1024": _e10_bayes(rec, 1024)["buckets"], "indep_256": _e10_bayes(rec, 256)["all"],
                    "copy_256": _e10_bayes(rec, 256, loaded["copies"][(p, "arith", seeds[0], e10.COPY_LENGTH)])["all"]}
        curves[p] = {}
        for arm in e10.ARMS:
            used = [s for s in seeds if (p, arm, s) in m]
            by = np.array([[b["nll"] for b in next(t for t in loaded["runs"][(p, arm, s)]["test"]
                                                   if t["length"] == 1024)["by_position"]] for s in used])
            curves[p][arm] = {"mean": by.mean(0).tolist(), "lo": by.min(0).tolist(), "hi": by.max(0).tolist(),
                              "indep_256": [m[(p, arm, s)]["indep_nll_256"] for s in used],
                              "copy_256": [m[(p, arm, s)]["copy_nll_256"] for s in used if "copy_nll_256" in m[(p, arm, s)]]}
            far, acc, exc, ind, cop = (e10_mean(m, p, arm, k, seeds)
                                       for k in ("far_nll", "far_acc", "excess_1024", "indep_nll_256", "copy_nll_256"))
            lines.append(f"| {p} | {E10_ARM_LABELS[arm]} | {far:.4f} | {100 * acc:.2f}% | {exc:.4f} | {ind:.4f} | "
                         f"{'n/a' if cop is None else f'{cop:.4f}'} |")
    lines += ["", "Exact Bayes NLL on the same test batches: "
              + "; ".join(f"P = {p}: independent law at 256 {b['indep_256']:.4f}, copied law at 256 {b['copy_256']:.4f}"
                          for p, b in bayes.items()) + ".", ""]
    d = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    pp = lambda c: f"{100 * c['mean']:+.2f} pp [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}]"  # noqa: E731
    lines.append("Paired differences (mean [95% CI]):")
    lines.append("")
    for p in e10.PERIODS:
        c = frozen["periods"][str(p)]
        lines.append(
            f"- P = {p}: V1 far NLL, head − none {d(c['v1_far_nll'])}, far accuracy {pp(c['v1_far_acc'])}. "
            f"V2 excess NLL at 1,024 {c['v2_excess_1024']['mean']:.5f} (upper bound {c['v2_excess_1024']['hi']:.5f}). "
            f"V3 far NLL, fundamental − arithmetic {d(c['v3_far_nll'])}, far accuracy, arithmetic − fundamental "
            f"{pp(c['v3_far_acc'])}. V4 (RoPE) far accuracy, head − none {pp(c['v4_far_acc'])}, far NLL "
            f"{d(c['v4_far_nll'])}. B1 copied NLL at 256, head − none {d(c['b1_copy_nll_256'])}. S15 independent NLL "
            f"at 256, head − none {d(c['s15_indep_nll_256'])}.")
    lines += ["", "Holm-adjusted one-sided p-values over the interval tests (sensitivity only): "
              + "; ".join(f"{k}: {v['p_holm']:.2g}" for k, v in sorted(frozen["holm"].items())) + "."]
    tables.append("\n".join(lines))
    out["curves"], out["bayes"] = curves, bayes
    summary["E10"] = out
    for mode in ("light", "dark"):
        plot_e10(curves, bayes, mode)


def plot_e10(curves: dict, bayes: dict, mode: str) -> Path:
    with themed(mode) as th:
        periods = sorted(curves)
        fig, axes = plt.subplots(len(periods), 2, figsize=(10.5, 7.4), gridspec_kw={"width_ratios": [1.5, 1]})
        for (ax, bx), p in zip(axes, periods):
            x = np.arange(len(bayes[p]["buckets_1024"]))
            floor = lambda v: np.maximum(v, E10_FLOOR)  # noqa: E731 - NLL below the floor (even exactly 0) sits on it
            for arm in ("rope", "arith", "arith_evfund", "arith_ev", "rope_ev"):
                cv, c = curves[p][arm], th["series"][E10_ARM_SLOT[arm]]
                ax.fill_between(x, floor(cv["lo"]), floor(cv["hi"]), color=c, alpha=0.18, linewidth=0)
                ax.plot(x, floor(cv["mean"]), color=c, linewidth=2.0, label=E10_ARM_LABELS[arm],
                        marker="o" if arm != "rope_ev" else "s", markersize=3.5 if arm != "rope_ev" else 6,
                        linestyle="-" if arm != "rope_ev" else ":", markerfacecolor=c if arm != "rope_ev" else "none")
            ax.plot(x, floor(bayes[p]["buckets_1024"]), color=th["ink"], linestyle="--", linewidth=1.6,
                    label="exact Bayes NLL")
            ax.set_yscale("log")
            ax.set_ylim(E10_FLOOR / 3, 10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(128 * i) for i in x])
            ax.set_xlabel("position bucket start (buckets of 128 tokens)")
            ax.set_ylabel(f"mean NLL (nats, log scale; floor {E10_FLOOR:.0e})")
            ax.set_title(f"P = {p}: NLL by position at N = 1,024", loc="left", fontsize=10)
            for j, key in enumerate(("indep_256", "copy_256")):
                for k, arm in enumerate(("arith", "arith_ev")):
                    vals, xs = curves[p][arm][key], j + (k - 0.5) * 0.34
                    bx.bar(xs, np.mean(vals), width=0.3, color=th["series"][E10_ARM_SLOT[arm]], alpha=0.8)
                    bx.scatter([xs] * len(vals), vals, s=10, color=th["ink"], alpha=0.55, zorder=3, linewidths=0)
                bx.hlines(bayes[p][key], j - 0.38, j + 0.38, color=th["ink"], linestyle="--", linewidth=1.6, zorder=4)
            bx.set_xticks([0, 1])
            bx.set_xticklabels(["ordinary law", "copied law (k = 4)"])
            bx.set_ylabel("NLL at N = 256 (nats)")
            bx.set_title(f"P = {p}: NLL at N = 256, by law", loc="left", fontsize=10)
            bx.grid(axis="x", visible=False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.0), ncol=3, fontsize=8.5, frameon=False)
        bx = axes[0][1]
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        bx.legend(handles=[Patch(color=th["series"][E10_ARM_SLOT["arith"]], alpha=0.8, label="ArithScope only (left bar)"),
                           Patch(color=th["series"][E10_ARM_SLOT["arith_ev"]], alpha=0.8, label="+ evidence head (right bar)"),
                           Line2D([], [], color=th["ink"], linestyle="--", label="exact Bayes NLL of the law")],
                  loc="upper left", fontsize=8, frameon=False)
        n = len(curves[periods[0]]["arith"]["indep_256"])
        fig.suptitle(f"E10 (preregistered): the evidence head against the exact Bayes NLL ({n} seeds per arm; "
                     "band = seed range, dots = seeds)", x=0.01, y=0.995, ha="left", fontweight="bold", fontsize=11,
                     color=th["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return save(fig, FIG, "e10_evidence", mode)


def e10_findings(summary: dict) -> str:
    """E10 verdicts (PREREGISTRATION.md Section 13), as computed by experiments/e10_verdicts.py."""
    e = summary.get("E10")
    if not e:
        return "(results pending)"
    v, ps = e["verdicts"], e["periods"]
    word = lambda key: v[key].lower() if v[key] == "SUPPORTED" else v[key]  # noqa: E731
    asp = lambda key: "as predicted" if v[key] else "not as predicted"  # noqa: E731
    cnt = lambda c, sign: f"{sum(x * sign > 0 for x in c['values'])}/{c['n']}"  # noqa: E731
    d = lambda c: f"{c['mean']:+.4f}, 95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    pp = lambda c: f"{100 * c['mean']:+.2f} pp, 95% CI [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}]"  # noqa: E731
    per = lambda f: "; ".join(f"P = {p}: {f(c)}" for p, c in ps.items())  # noqa: E731
    # per-period "mean, CI, seeds with the claimed sign" of one paired difference, in nats or in pp
    nats = lambda key, sign, word: per(lambda c: d(c[key]) + f", {word} in {cnt(c[key], sign)} seeds")  # noqa: E731
    pts = lambda key, sign, word: per(lambda c: pp(c[key]) + f", {word} in {cnt(c[key], sign)} seeds")  # noqa: E731
    p13 = ps["13"]
    sane = all(x for k, x in e["sanity"].items() if k != "code_commits")
    failed = f"{len(e['failed'])} failed, reported and not replaced" if e["failed"] else "no failures"
    arms = ("arith", "arith_ev", "arith_evfund", "rope", "rope_ev")
    means = "; ".join(f"P = {p}: " + ", ".join(f"{E10_ARM_LABELS[a]} {c['means'][a]['far_nll']:.4f}" for a in arms)
                      for p, c in ps.items())
    cos = [x for c in ps.values() for arm in ("arith_ev", "rope_ev") for x in c["kernel"][arm]["cos_dp"]]
    lam = [x for c in ps.values() for arm in ("arith_ev", "rope_ev") for x in c["kernel"][arm]["lambda_eff"]]
    ex = lambda c: c["v2_excess_1024"]  # noqa: E731
    return "\n".join([
        f"- **Design.** {e['plan']['n_seeds']} fresh seeds ({e['planned_seeds'][0]}–{e['planned_seeds'][-1]}) per period "
        f"and arm from the preregistered power rule (planned joint power {e['plan']['joint_power_at_n']:.2f}), five arms, "
        f"each evaluated on the ordinary law and, unchanged, on the copied law; {failed}.",
        f"- **Sanity checks (all records accounted for, one code commit, one test set per period and length, copy "
        f"checkpoints match the suite): {'passed' if sane else 'FAILED'}.**",
        f"- **V1 (the head lowers ArithScope alone's far NLL at no accuracy cost): {word('V1_head_lowers_far_nll')}**. "
        f"Far NLL, head − none: {nats('v1_far_nll', -1, 'negative')}. "
        f"Far accuracy: {per(lambda c: pp(c['v1_far_acc']))}.",
        f"- **V2 (excess NLL at 1,024 below 0.005 nats): {word('V2_follows_bayes_at_1024')}**. "
        + per(lambda c: f"mean {ex(c)['mean']:.5f}, 95% CI [{ex(c)['lo']:.5f}, {ex(c)['hi']:.5f}], below 0.005 in "
                        f"{sum(x < 0.005 for x in ex(c)['values'])}/{ex(c)['n']} seeds") + ".",
        f"- **V3 (at P = 13 the exact basis beats the matched fundamental cosines): "
        f"{word('V3_exact_basis_matters_at_13')}**. P = 13: far NLL, fundamental − arithmetic {d(p13['v3_far_nll'])}, "
        f"positive in {cnt(p13['v3_far_nll'], 1)} seeds; far accuracy, arithmetic − fundamental {pp(p13['v3_far_acc'])}, "
        f"positive in {cnt(p13['v3_far_acc'], 1)} seeds.",
        f"- **V4 (the head restores RoPE's extrapolation): {word('V4_restores_rope_extrapolation')}**. "
        f"Far accuracy, head − none: {pts('v4_far_acc', 1, 'positive')}. "
        f"Far NLL: {nats('v4_far_nll', -1, 'negative')}.",
        f"- **B1 (boundary, preregistered: on copied observations the head is worse than none): "
        f"{word('B1_copied_observations_harm')}**. Copied-law NLL at 256, head − none: "
        f"{nats('b1_copy_nll_256', 1, 'positive')}.",
        f"- **S12 (no difference at P = 12, far accuracy within ±1 pp): {asp('S12_no_failure_at_12')}.** "
        f"P = 12, arithmetic − fundamental: {pp(ps['12']['v3_far_acc'])}.",
        f"- **S13 (SGD finds D_P, cosine at least 0.99 in every seed): {asp('S13_sgd_finds_construction')}.** "
        f"Smallest cosine to D_P's coefficients over {len(cos)} head fits: {min(cos):.4f}.",
        f"- **S14 (effective scale within ±10% of log 17 in every seed): {asp('S14_bayes_scale')}.** "
        f"λ_eff range {min(lam):.3f}–{max(lam):.3f} (log 17 = {math.log(17):.3f}).",
        f"- **S15 (on the ordinary law the head lowers the NLL at 256): {asp('S15_crossover')}.** "
        f"{nats('s15_indep_nll_256', -1, 'negative')}.",
        f"- **Descriptive.** Mean far NLL: {means}.",
    ])


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
    analyze_nll_runs(summary, tables)
    analyze_posthoc_nll(summary, tables)
    analyze_gates(summary, tables)
    analyze_meta(summary, tables)
    analyze_overhead(summary, tables)
    analyze_multiplicity(summary, tables)
    analyze_replay(summary, tables)
    analyze_e6(summary, tables)
    analyze_e7(summary, tables)
    analyze_e8(summary, tables)
    analyze_e9(summary, tables)
    analyze_e10(summary, tables)
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
        rewrite_block(ROOT / "README.md", "calibration", calibration_md(summary))
        rewrite_block(ROOT / "README.md", "findings", findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "findings", findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "e6_findings", e6_findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "e7_findings", e7_findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "e8_findings", e8_findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "e9_findings", e9_findings(summary))
        rewrite_block(ROOT / "docs" / "RESULTS.md", "e10_findings", e10_findings(summary))
    print(md)


if __name__ == "__main__":
    main()
