"""Build the ArithScope Lab page from committed summaries.

    python experiments/lab/build.py [--body-out PATH] [--ref GIT_REF]

Reads results/summary/theory.json (and summary.json when present) and writes
docs/lab/index.html, a standalone page. With --body-out it also writes the page
body alone, the form the Artifact publisher expects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.analyze import findings  # noqa: E402

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}
body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}</style>
"""
VARIANTS = ("rope+arith", "rope+fourier_learned", "rope+fourier_fund", "rope+relbias", "rope")
REPO = "https://github.com/Akari-H-111/ArithScope"


def theory_data() -> dict:
    d = json.loads((ROOT / "results" / "summary" / "theory.json").read_text())
    margins = [{"P": r["period"], "opt": round(r["optimal_pd"], 4), "cos": round(r["fundamental_cosine_bank"], 4),
                "kp": round(r["single_ramanujan"], 4), "c1": round(r["single_cosine"], 4)} for r in d["margins"]]
    mass = {k: {"N": v["length"], "mass": [round(x, 4) for x in v["comb_mass"]]} for k, v in d["comb_mass"].items()}
    return {"margins": margins, "comb_mass": mass}


def results_data() -> dict | None:
    path = ROOT / "results" / "summary" / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    e1 = {}
    for p in ("12", "13"):
        tab = s.get(f"E1_P{p}", {}).get("table")
        if not tab:
            continue
        lengths = sorted(int(L) for L in tab["oracle"])
        entry = {"lengths": lengths, "oracle": [tab["oracle"][str(L)] for L in lengths]}
        for v in VARIANTS:
            if v in tab:
                cells = [tab[v][str(L)] for L in lengths]
                entry[v] = {"mean": [c["mean"] for c in cells],
                            "lo": [max(0.0, c["lo"] if c["lo"] is not None else c["mean"]) for c in cells],
                            "hi": [min(1.0, c["hi"] if c["hi"] is not None else c["mean"]) for c in cells]}
        e1[p] = entry
    if not e1:
        return None
    restored = {k: v for k, v in s.items()}
    for key in ("E1_P12", "E1_P13"):  # findings() expects integer length keys, as analyze.py builds them
        if key in restored:
            tab = restored[key]["table"]
            restored[key] = {**restored[key], "table": {v: {int(L): c for L, c in cells.items()} for v, cells in tab.items()}}
    verdicts = [line[2:].replace("**", "") for line in findings(restored).splitlines() if line.startswith("- ")]
    n = len(e1["13"]["rope+arith"]["mean"]) if "13" in e1 else 0
    intro = ("Tiny decoder-only Transformers (2 layers, d = 64) were trained on 64-token sequences and tested on positions "
             f"they never saw, up to {max(e1[next(iter(e1))]['lengths'])} tokens. Curves are means over seeds with 95% CIs; "
             "the protocol was frozen and pushed before any confirmatory run.")
    attention = None
    att_path = ROOT / "results" / "summary" / "attention_profile.json"
    if att_path.exists():
        attention = json.loads(att_path.read_text())
    alpha = {}
    for p in ("12", "13"):
        a = s.get(f"E1_alpha_P{p}_rope+arith")
        if a:
            alpha[p] = {"orders": a["orders"], "tv": round(a["tv"], 4), "heads": a["heads"],
                        **{k: [round(x, 4) for x in a[k]] for k in ("mean", "sd", "optimal")}}
    return {"intro": intro, "e1": e1, "verdicts": verdicts, "n_lengths": n, "attention": attention,
            "alpha": alpha or None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-out")
    parser.add_argument("--ref", default="main", help="git branch, tag, or commit the page's repository links point to")
    args = parser.parse_args()
    body = (Path(__file__).parent / "template.html").read_text()
    body = body.replace("__REPO__", f"{REPO}/tree/{args.ref}").replace("__BLOB__", f"{REPO}/blob/{args.ref}")
    body = body.replace("__THEORY__", json.dumps(theory_data(), separators=(",", ":")))
    body = body.replace("__RESULTS__", json.dumps(results_data(), separators=(",", ":")))
    out = ROOT / "docs" / "lab" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    split = body.index("<main>")  # title, fonts and styles go in <head> for the standalone page
    out.write_text(HEAD + body[:split] + "</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    if args.body_out:
        Path(args.body_out).write_text(body)
        print(f"wrote {args.body_out}")


if __name__ == "__main__":
    main()
