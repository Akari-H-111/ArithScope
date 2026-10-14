"""Shared matplotlib styling: validated categorical palette, light and dark variants.

Colors follow the entity (a variant keeps its color in every figure), in the
fixed slot order of the validated reference palette. Text never uses series
colors; identity is carried by the legend and direct end labels.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]},
}

# Fixed entity -> slot mapping (slot order = validated adjacency order).
VARIANT_SLOT = {
    "rope+arith": 0, "arith": 0,          # the ArithScope channel (with / without RoPE)
    "rope": 1,
    "rope+fourier_fund": 2,
    "rope+fourier_full": 3,
    "rope+fourier_learned": 4,
    "rope+relbias": 5,
    "alibi": 6,
    "nope": 7,
}

LABELS = {
    "rope+arith": "RoPE + ArithScope",
    "arith": "ArithScope only",
    "rope": "RoPE",
    "rope+fourier_fund": "RoPE + matched cosine bank",
    "rope+fourier_full": "RoPE + full Fourier bank",
    "rope+fourier_learned": "RoPE + learned frequencies",
    "rope+relbias": "RoPE + relative bias (clipped)",
    "alibi": "ALiBi",
    "nope": "NoPE",
}


def color(theme: dict, variant: str) -> str:
    return theme["series"][VARIANT_SLOT[variant]]


@contextmanager
def themed(mode: str):
    theme = THEMES[mode]
    rc = {
        "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"], "axes.edgecolor": theme["axis"],
        "axes.labelcolor": theme["ink2"], "axes.titlecolor": theme["ink"], "text.color": theme["ink"],
        "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "xtick.labelcolor": theme["ink2"], "ytick.labelcolor": theme["ink2"],
        "grid.color": theme["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": theme["ink2"],
    }
    with matplotlib.rc_context(rc):
        yield theme


def save(fig, out_dir: Path, stem: str, mode: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
