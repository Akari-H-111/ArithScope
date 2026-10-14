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


def plot_shift(data: dict, mode: str) -> Path:
    with themed(mode) as th:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for v, curve in data.items():
            xs = sorted(curve)
            ax.plot([max(x, 1) for x in xs], [curve[x]["mean"] for x in xs], color=color(th, v), label=LABELS[v],
                    marker="o", markersize=4.5, markeredgecolor=th["surface"], markeredgewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("constant shift added to every position (0 plotted at 1)")
        ax.set_ylabel("accuracy, N = 256")
        ax.set_title("Absolute-position shift at float32 (relative structure unchanged)", loc="left")
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
    for mode in ("light", "dark"):
        plot_shift(keep, mode)


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
               "arith_sdpa_mask": "ArithScope, SDPA + dense bias mask", "arith_folded": "ArithScope, folded into q/k (fused)"}
MODE_SLOT = {"baseline_sdpa": 1, "arith_explicit": 6, "arith_sdpa_mask": 2, "arith_folded": 0}


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
    return "\n".join(rows) + ("\n\nAccuracy (%) on positions never seen in training (t ≥ 64), mean ± 95% CI over "
                              "5 seeds; generated by `experiments/analyze.py`.")


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
    analyze_alpha(summary, tables)
    analyze_alias(summary, tables)
    analyze_controls(summary, tables)
    analyze_e1(summary, tables, suite="e5_mixed", label="E5")
    analyze_meta(summary, tables)
    analyze_overhead(summary, tables)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    md = "\n\n".join(tables)
    (SUMMARY / "tables.md").write_text(md + "\n")
    if DOCS_REWRITE:
        rewrite_block(ROOT / "docs" / "RESULTS.md", "tables", md)
        rewrite_block(ROOT / "README.md", "headline", headline(summary))
    print(md)


if __name__ == "__main__":
    main()
