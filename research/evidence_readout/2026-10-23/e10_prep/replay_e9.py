"""Re-evaluate stored E9 checkpoints with the worktree library; compare with the committed E9 records."""
import json
import sys
from pathlib import Path

WT = Path("/Users/akari_hayami_64/ArithScope-evidence")
MAIN = Path("/Users/akari_hayami_64/ArithScope")
sys.path.insert(0, str(WT))
from experiments.e9_window_eval import evaluate_windows  # noqa: E402

out = Path(sys.argv[1])
keys = ("length", "num_seqs", "digest", "all", "extrapolated", "by_position")
results = []
for stem in ("e9_p12__arith__s27__294a22d3f72d", "e9_p13__arith__s27"):
    ckpt = next((MAIN / "checkpoints" / "e9_paired_window").glob(stem + "*.pt"))
    evaluate_windows(ckpt, (64, 8192), 8192, out)
    for w in (64, 8192):
        new = json.loads((out / f"{ckpt.stem}__W{w}.json").read_text())
        old = json.loads((MAIN / "results" / "runs" / "e9_paired_window_eval" / f"{ckpt.stem}__W{w}.json").read_text())
        same = all(new["test"][0][k] == old["test"][0][k] for k in keys) and new["checkpoint_sha256"] == old["checkpoint_sha256"]
        results.append({"checkpoint": ckpt.stem, "window": w, "bitwise_identical": same,
                        "far_nll_bucket7_new": new["test"][0]["by_position"][7]["nll"],
                        "far_nll_bucket7_old": old["test"][0]["by_position"][7]["nll"]})
print(json.dumps(results, indent=1))
