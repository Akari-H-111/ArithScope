"""Figures and tables for the theory (no training): comb margins and long-range comb mass.

    python experiments/theory_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope import arith, theory  # noqa: E402
from experiments.plotstyle import plt, save, themed  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
SUMMARY = ROOT / "results" / "summary"

PERIOD, WINDOW, BUDGET = 13, 64, 8.0
LONG = 2 ** 20


def margin_data() -> list[dict]:
    return [theory.margin_row(p) for p in range(2, 37)]


def comb_mass_data() -> dict:
    offs = np.arange(LONG)
    k_table = np.array([float(arith.kernel(PERIOD, r)) for r in range(PERIOD)])
    curves = {
        "ArithScope 8·K₁₃ (budget 8)": BUDGET * k_table[offs % PERIOD],
        "matched cosine 8·cos(2πn/13) (budget 8)": BUDGET * np.cos(2 * np.pi * offs / PERIOD),
        "learned frequency off by ε = 10⁻³ (budget 8)": BUDGET * np.cos((2 * np.pi / PERIOD + 1e-3) * offs),
    }
    for budget in (BUDGET, 4 * BUDGET):
        coef = theory.fit_rope_comb(PERIOD, WINDOW, head_dim=64, budget=budget, steps=4000)
        curves[f"RoPE d=64 fit on 64 tokens (budget {budget:g})"] = theory.rope_scores(
            coef, theory.rope_frequencies(64), offs)
    lengths = 2 ** np.arange(6, 21)
    return {name: {"length": lengths.tolist(), "comb_mass": theory.comb_mass(s, PERIOD)[lengths - 1].tolist(),
                   "max_abs_score": float(np.abs(s).max())} for name, s in curves.items()}


def plot_margins(rows: list[dict], mode: str) -> Path:
    with themed(mode) as th:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        p = [r["period"] for r in rows]
        series = [("ramanujan_bank", "ArithScope bank: optimum P/(P−1) (Thm T3)", 0),
                  ("fundamental_cosine_bank", "matched cosine bank, same periods (LP)", 2),
                  ("single_cosine", "single cosine cos(2πn/P)", 1)]
        for key, label, slot in series:
            y = [r[key] for r in rows]
            ax.plot(p, y, color=th["series"][slot], label=label, marker="o", markersize=4.5,
                    markeredgecolor=th["surface"], markeredgewidth=1.0)
        by_p = {r["period"]: r for r in rows}
        for period in (13, 29):
            lo, hi = by_p[period]["fundamental_cosine_bank"], by_p[period]["ramanujan_bank"]
            ax.annotate("", xy=(period, hi * 0.93), xytext=(period, lo * 1.08),
                        arrowprops={"arrowstyle": "<->", "color": th["ink2"], "linewidth": 1.0})
            ax.annotate(f"{hi / lo:.0f}×", (period, (hi * lo) ** 0.5), textcoords="offset points",
                        xytext=(5, 0), color=th["ink2"], fontsize=9, va="center")
        ax.set_yscale("log")
        ax.set_xlabel("period P")
        ax.set_ylabel("best comb margin (log scale)")
        ax.set_title("Exact comb margins of PSD periodic biases", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=1)
        return save(fig, FIG_DIR, "theory_comb_margin", mode)


def plot_comb_mass(data: dict, mode: str) -> Path:
    with themed(mode) as th:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        slots = {"ArithScope 8·K₁₃ (budget 8)": 0, f"RoPE d=64 fit on 64 tokens (budget {BUDGET:g})": 1,
                 "matched cosine 8·cos(2πn/13) (budget 8)": 2, f"RoPE d=64 fit on 64 tokens (budget {4 * BUDGET:g})": 3,
                 "learned frequency off by ε = 10⁻³ (budget 8)": 4}
        for name, slot in slots.items():
            d = data[name]
            ax.plot(d["length"], d["comb_mass"], color=th["series"][slot], label=name, marker="o", markersize=4.5,
                    markeredgecolor=th["surface"], markeredgewidth=1.0)
        ax.axhline(1 / PERIOD, color=th["muted"], linewidth=1.0)
        ax.annotate("uniform share 1/13 (limit in Thm T5)", (2 ** 6, 1 / PERIOD), textcoords="offset points",
                    xytext=(2, 4), color=th["ink2"], fontsize=8.5)
        ax.axvline(WINDOW, color=th["axis"], linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("context length N (offsets 0 … N−1)")
        ax.set_ylabel("attention mass on the 13-comb")
        ax.set_title("Attention mass on the 13-comb vs context length", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=8.5)
        return save(fig, FIG_DIR, "theory_comb_mass", mode)


def main() -> None:
    rows = margin_data()
    mass = comb_mass_data()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "theory.json").write_text(json.dumps({"margins": rows, "comb_mass": mass,
                                                     "period": PERIOD, "window": WINDOW, "budget": BUDGET}, indent=1))
    for mode in ("light", "dark"):
        print(plot_margins(rows, mode))
        print(plot_comb_mass(mass, mode))


if __name__ == "__main__":
    main()
