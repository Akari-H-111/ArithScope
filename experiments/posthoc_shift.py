"""POST-HOC, EXPLORATORY: absolute-position shift at the training length (not preregistered).

The preregistered shift evaluation (E1, N = 256) is confounded for RoPE, which already
fails at 4x its training length. This script re-evaluates the saved E1 checkpoints at
N = 64, where every RoPE-based model works, with all positions shifted by a constant.
The relative structure is unchanged, so exact relative mechanisms must be invariant;
any change measures floating-point position handling (RoPE angles are float32).

    python experiments/posthoc_shift.py [--workers 4]
Writes results/posthoc/shift_n64.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHIFTS = (0, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9)
LENGTH, SEQS = 64, 1024


def evaluate_checkpoint(path: str) -> dict:
    import torch

    from arithscope.tasks import make_task
    from arithscope.train import RunSpec, build_model, evaluate, task_dict

    torch.set_num_threads(1)
    blob = torch.load(path, weights_only=False)
    fields = dict(blob["spec"])
    for key in ("orders", "eval_lengths", "shift_evals"):
        fields[key] = tuple(fields[key])
    spec = RunSpec(**fields)
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    model = build_model(spec, task.model_vocab)
    model.load_state_dict(blob["state_dict"])
    parts = ("posthoc-shift", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), LENGTH)
    rows = []
    for shift in SHIFTS:
        r = evaluate(model, task, LENGTH, SEQS, parts, spec.train_length, position_shift=shift)
        rows.append({"shift": shift, "acc": r["all"]["acc"], "oracle_acc": r["all"]["oracle_acc"], "digest": r["digest"]})
    return {"experiment": spec.experiment, "variant": spec.variant, "seed": spec.seed, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ckpts = sorted(str(p) for p in (ROOT / "checkpoints" / "e1_periodic").glob("*.pt"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_checkpoint, ckpts))
    from arithscope.train import environment_info

    out = ROOT / "results" / "posthoc" / "shift_n64.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "POST-HOC EXPLORATORY (not preregistered)", "length": LENGTH,
                               "sequences": SEQS, "shifts": list(SHIFTS), "environment": environment_info(),
                               "runs": results}, indent=1))
    print(f"wrote {out} ({len(results)} checkpoints)")


if __name__ == "__main__":
    main()
