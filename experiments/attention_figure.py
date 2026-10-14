"""Attention profile of trained heads: weight vs offset for the last query of a long sequence.

    python experiments/attention_figure.py [--period 13] [--seed 1] [--length 1024]

Loads E1 checkpoints (``checkpoints/e1_periodic``, written by the runner) for RoPE and
RoPE + ArithScope, runs one fixed test sequence, and plots layer-0 attention of the last
query position against the offset, averaged over the four heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope.tasks import make_task  # noqa: E402
from arithscope.train import RunSpec, _generator, build_model  # noqa: E402
from experiments.plotstyle import LABELS, color, plt, save, themed  # noqa: E402

CKPT = ROOT / "checkpoints" / "e1_periodic"
FIG = ROOT / "results" / "figures"


def load(variant: str, period: int, seed: int):
    matches = sorted(CKPT.glob(f"e1_p{period}__{variant}__s{seed}__*.pt"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint for {variant} P={period} seed={seed} in {CKPT}")
    blob = torch.load(matches[0], weights_only=False)
    fields = dict(blob["spec"])
    for key in ("orders", "eval_lengths", "shift_evals"):
        fields[key] = tuple(fields[key])
    spec = RunSpec(**fields)
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    model = build_model(spec, task.model_vocab)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), task, spec


@torch.no_grad()
def layer0_attention(model, tokens: torch.Tensor) -> torch.Tensor:
    """Layer-0 attention weights ``[H, N, N]`` for a single sequence."""
    x = model.embed(tokens)
    positions = torch.arange(tokens.shape[1])
    _, attn = model.blocks[0].attn(model.blocks[0].ln1(x), positions, return_attention=True)
    return attn[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    args = parser.parse_args()
    profiles = {}
    for variant in ("rope+arith", "rope"):
        model, task, spec = load(variant, args.period, args.seed)
        batch = task.sample(1, args.length, _generator("attention-figure", args.period, args.length))
        attn = layer0_attention(model, batch.tokens)  # [H, N, N]
        row = attn[:, -1, :].mean(0).flip(0).numpy()   # offset 0 = the query itself
        comb = np.arange(args.length) % args.period == 0
        profiles[variant] = {"weights": row.tolist(), "comb_mass": float(row[comb].sum()),
                             "comb_mass_beyond_train": float(row[comb & (np.arange(args.length) >= spec.train_length)].sum()
                                                         / max(row[np.arange(args.length) >= spec.train_length].sum(), 1e-12))}
    out = ROOT / "results" / "summary" / "attention_profile.json"
    out.write_text(json.dumps({"period": args.period, "seed": args.seed, "length": args.length,
                               "profiles": {v: {"comb_mass": p["comb_mass"],
                                                "comb_mass_beyond_train": p["comb_mass_beyond_train"],
                                                "weights": [float(f"{w:.4g}") for w in p["weights"]]}
                                            for v, p in profiles.items()}}))
    for mode in ("light", "dark"):
        with themed(mode) as th:
            fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6), sharex=True)
            for ax, variant in zip(axes, ("rope+arith", "rope")):
                w = np.asarray(profiles[variant]["weights"])
                offs = np.arange(len(w))
                ax.vlines(offs, 0, w, color=color(th, variant), linewidth=0.9)
                ax.axvspan(0, 63, color=th["grid"], alpha=0.6, linewidth=0)
                ax.set_yscale("log")
                ax.set_ylim(max(w.min(), 1e-7), 1.0)
                ax.set_ylabel("weight")
                frac = profiles[variant]["comb_mass_beyond_train"]
                ax.set_title(f"{LABELS[variant]}: {100 * frac:.0f}% of attention beyond the training window lands on "
                             f"multiples of {args.period}", loc="left", fontsize=9.5)
            for ax in axes:
                ax.text(32, 0.97, "training\nwindow", transform=ax.get_xaxis_transform(), ha="center", va="top",
                        color=th["ink2"], fontsize=8)
            axes[1].set_xlabel(f"offset i − j from the last query (sequence length {args.length}, trained at 64)")
            fig.suptitle(f"Layer-0 attention of one query, period {args.period} (mean over 4 heads, log scale)",
                         x=0.01, ha="left", fontweight="bold", fontsize=11, color=th["ink"])
            print(save(fig, FIG, "attention_profile", mode))


if __name__ == "__main__":
    main()
