"""Print a compact table of run results (reads JSON logs only).

    python experiments/summarize.py results/runs/<suite> [--region all|extrapolated]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_runs(path: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--region", default="all", choices=["all", "extrapolated"])
    args = parser.parse_args()
    runs = load_runs(Path(args.path))
    table = defaultdict(dict)
    lengths = set()
    for r in runs:
        key = (r["spec"]["experiment"], r["spec"]["variant"], r["spec"]["seed"])
        for t in r["test"]:
            stats = t[args.region] if t[args.region] else t["all"]
            table[key][t["length"]] = (stats["acc"], stats["oracle_acc"])
            lengths.add(t["length"])
        table[key]["val"] = r["train"]["val_curve"][-1]["acc"]
        table[key]["gamma"] = max((max(abs(g) for g in layer["arith"]["gamma"]) for layer in r["positional"]
                                   if "arith" in layer), default=None)
    lengths = sorted(lengths)
    head = f"{'experiment':34s} {'variant':22s} {'seed':>4s} {'val':>6s} " + " ".join(f"{L:>7d}" for L in lengths)
    print(head + "   |gamma|max")
    for key in sorted(table):
        row = table[key]
        cells = " ".join(f"{row[L][0]:7.3f}" if L in row else "    -  " for L in lengths)
        gamma = f"{row['gamma']:.2f}" if row["gamma"] is not None else "-"
        print(f"{key[0]:34s} {key[1]:22s} {key[2]:4d} {row['val']:6.3f} {cells}   {gamma}")
    any_key = next(iter(table))
    print(f"{'Bayes oracle':62s}        " + " ".join(f"{table[any_key][L][1]:7.3f}" if L in table[any_key] else "    -  "
                                                     for L in lengths))


if __name__ == "__main__":
    main()
