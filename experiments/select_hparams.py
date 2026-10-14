"""Apply the preregistered pilot rule: per variant, pick the (lr, positional multiplier) pair
with the lowest final validation NLL at the training length, pooled over pilot tasks.

    python experiments/select_hparams.py results/runs/pilot [results/runs/pilot_controls ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main(paths: list[str]) -> dict:
    pooled = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for f in sorted(Path(path).glob("*.json")):
            r = json.loads(f.read_text())
            s = r["spec"]
            if s["experiment"].startswith("pilotc_alias"):
                continue  # single-config feasibility runs, not part of the selection
            pooled[(s["task"]["kind"], s["variant"])][(s["lr"], s["pos_lr_mult"])].append(
                r["train"]["val_curve"][-1]["nll"])
    choice = {}
    print(f"{'task':10s} {'variant':22s} " + "  ".join(f"lr={lr:g},x{m:g}" for lr, m in ((1e-3, 1.0), (3e-3, 10.0)))
          + "  -> choice")
    for (task, variant), configs in sorted(pooled.items()):
        means = {cfg: sum(v) / len(v) for cfg, v in configs.items()}
        best = min(means, key=means.get)
        choice[f"{task}/{variant}"] = {"lr": best[0], "pos_lr_mult": best[1],
                                       "val_nll": {f"{k[0]:g}/x{k[1]:g}": v for k, v in means.items()}}
        cells = "  ".join(f"{means.get(cfg, float('nan')):12.4f}" for cfg in ((1e-3, 1.0), (3e-3, 10.0)))
        print(f"{task:10s} {variant:22s} {cells}  -> lr={best[0]:g}, x{best[1]:g}")
    return choice


if __name__ == "__main__":
    result = main(sys.argv[1:])
    out = Path(__file__).resolve().parent.parent / "results" / "summary" / "hparam_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")
