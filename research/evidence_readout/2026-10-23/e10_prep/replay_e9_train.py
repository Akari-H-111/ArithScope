"""Retrain one E9 run with the worktree library; compare with the committed record bit for bit."""
import json
import sys
from pathlib import Path

WT = Path("/Users/akari_hayami_64/ArithScope-evidence")
MAIN = Path("/Users/akari_hayami_64/ArithScope")
sys.path.insert(0, str(WT))
from arithscope.train import run  # noqa: E402
from experiments.suites import SUITES  # noqa: E402

spec = next(s for s in SUITES["e9_paired_window"]() if s.experiment == "e9_p13" and s.seed == 27)
new = run(spec, sys.argv[1], threads=1)
old = json.loads((MAIN / "results" / "runs" / "e9_paired_window" / spec.filename).read_text())
checks = {k: new["train"][k] == old["train"][k] for k in ("loss_curve", "val_curve")}
checks["test"] = [{k: t[k] for k in ("digest", "all", "by_position")} for t in new["test"]] == \
                 [{k: t[k] for k in ("digest", "all", "by_position")} for t in old["test"]]
checks["positional"] = new["positional"] == old["positional"]
checks["no_evidence_key"] = "evidence" not in new
print(spec.filename, json.dumps(checks))
