"""Replay committed runs with the current code and compare them with the stored records.

    python experiments/replay.py eval  [--workers 4]   # re-evaluate every E1 checkpoint
    python experiments/replay.py train [--workers 4]   # retrain a subset of confirmatory runs

``eval`` reloads each checkpoint in ``checkpoints/e1_periodic`` (written by the runner,
not committed) and re-scores the preregistered test sets at N = 1,024 and N = 8,192.
``train`` reruns a fixed subset of confirmatory specs from scratch (one thread each).
Both compare test-set digests, accuracies and losses with the committed run files
exactly (bitwise for floats) and write ``results/replay/<mode>.json``. Timing and
environment fields are not compared.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_LENGTHS = (1024, 8192)
TRAIN_SUBSET = [("e1_periodic", "e1_p13", v, 1) for v in ("nope", "alibi", "rope", "arith", "rope+arith",
                                                           "rope+fourier_fund", "rope+fourier_full",
                                                           "rope+fourier_learned", "rope+relbias")]
TRAIN_SUBSET += [("e2_alias", "e2_alias", "arith", 1), ("e3_tagged", "e3_tagged", "rope+arith", 1)]


def _stored(suite: str, filename: str) -> dict:
    return json.loads((ROOT / "results" / "runs" / suite / filename).read_text())


def _metrics(test: dict) -> dict:
    keep = ("n", "acc", "oracle_acc", "nll", "alias_oracle_acc")
    return {"digest": test["digest"], **{region: {k: v for k, v in (test[region] or {}).items() if k in keep}
                                         for region in ("all", "extrapolated")}}


def replay_eval(path: str) -> dict:
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
    stored = {t["length"]: t for t in _stored("e1_periodic", spec.filename)["test"]}
    rows = []
    for length in EVAL_LENGTHS:
        num = max(2, (spec.eval_tokens if length <= 2048 else spec.eval_tokens_long) // length)  # as in train.run
        parts = ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), length)
        new = _metrics(evaluate(model, task, length, num, parts, spec.train_length))
        old = _metrics(stored[length])
        rows.append({"length": length, "identical": new == old, "stored": old, "replayed": new})
    return {"run": spec.filename, "rows": rows}


def replay_train(item: tuple) -> dict:
    from experiments.suites import SUITES

    suite, experiment, variant, seed = item
    spec = next(s for s in SUITES[suite]() if s.experiment == experiment and s.variant == variant and s.seed == seed)
    from arithscope.train import run

    with tempfile.TemporaryDirectory() as tmp:
        new = run(spec, tmp, threads=1)
    old = _stored(suite, spec.filename)
    checks = {
        "loss_curve": new["train"]["loss_curve"] == old["train"]["loss_curve"],
        "val_curve": new["train"]["val_curve"] == old["train"]["val_curve"],
        "test": [_metrics(t) for t in new["test"]] == [_metrics(t) for t in old["test"]],
        "shift_test": [_metrics(t) for t in new.get("shift_test", [])] == [_metrics(t) for t in old.get("shift_test", [])],
        "positional": new["positional"] == old["positional"],
    }
    return {"run": spec.filename, "identical": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("eval", "train"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "eval":
        jobs, fn = sorted(str(p) for p in (ROOT / "checkpoints" / "e1_periodic").glob("*.pt")), replay_eval
    else:
        jobs, fn = TRAIN_SUBSET, replay_train
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fn, jobs))
    from arithscope.train import environment_info

    if args.mode == "eval":
        identical = sum(r["identical"] for res in results for r in res["rows"])
        total = sum(len(res["rows"]) for res in results)
    else:
        identical, total = sum(r["identical"] for r in results), len(results)
    out = ROOT / "results" / "replay" / f"{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "identical": identical, "total": total,
                               "environment": environment_info(), "results": results}, indent=1))
    print(f"{args.mode} replay: {identical}/{total} identical -> {out}")


if __name__ == "__main__":
    main()
