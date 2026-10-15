"""E9 window evaluations (docs/PREREGISTRATION.md, Section 12).

    python experiments/e9_window_eval.py [--workers 4]

For every E9 checkpoint (checkpoints/e9_paired_window/*.pt, written by the training runs), the unchanged
weights are evaluated at N = 8,192 on the E9 test set with the last layer limited to its last W keys,
for each W in E9_WINDOWS. The first layer is never limited. The test set is the one `train.run` would
draw for this length (same generator parts and sequence count), so W = 8,192 is the ordinary
unwindowed evaluation. Writes one record per (checkpoint, W) to results/runs/e9_paired_window_eval/;
finished records are skipped, so the script resumes after an interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUITE = "e9_paired_window"
EVAL_SUITE = "e9_paired_window_eval"


def record_name(checkpoint: Path, window: int) -> str:
    return f"{checkpoint.stem}__W{window}.json"


def load_checkpoint(path: Path):
    """(spec, model) with the stored weights; the model is in eval mode with no key window anywhere."""
    import torch

    from arithscope.tasks import make_task
    from arithscope.train import RunSpec, build_model

    blob = torch.load(path, weights_only=False)
    fields = dict(blob["spec"])
    for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
        if key in fields:
            fields[key] = tuple(fields[key])
    spec = RunSpec(**fields)
    if spec.attn_windows:
        raise ValueError(f"E9 checkpoints are trained without a window: {path.name}")
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    model = build_model(spec, task.model_vocab)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return spec, task, model


def evaluate_windows(checkpoint: Path, windows: tuple[int, ...], length: int, out_dir: Path) -> list[Path]:
    """Evaluate one checkpoint at `length` once per last-layer window; returns the records written."""
    import torch

    from arithscope.train import environment_info, evaluate, task_dict

    torch.set_num_threads(1)
    spec, task, model = load_checkpoint(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    budget = spec.eval_tokens if length <= 2048 else spec.eval_tokens_long
    num_seqs = max(2, budget // length)  # as in train.run
    parts = ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), length)  # as in train.run
    last = model.blocks[-1].attn
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for window in windows:
        dest = out_dir / record_name(checkpoint, window)
        if dest.exists():
            continue
        last.window = window
        try:
            result = evaluate(model, task, length, num_seqs, parts, spec.train_length)
        finally:
            last.window = None
        record = {"schema_version": "arithscope-window-eval-1", "status": "COMPLETED",
                  "source_run": checkpoint.stem + ".json", "checkpoint_sha256": digest, "spec": asdict(spec),
                  "window_layer": len(model.blocks) - 1, "window": window, "test": [result],
                  "environment": environment_info()}
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, dest)
        written.append(dest)
    return written


def _worker(checkpoint: str, windows: tuple[int, ...], length: int, out_dir: str) -> tuple[str, str | None, float]:
    start = time.perf_counter()
    try:
        evaluate_windows(Path(checkpoint), windows, length, Path(out_dir))
        return checkpoint, None, time.perf_counter() - start
    except Exception:  # noqa: BLE001 - failures are reported, not hidden
        return checkpoint, traceback.format_exc(), time.perf_counter() - start


def main() -> None:
    from experiments.suites import E9_LENGTH, E9_WINDOWS, e9_paired_window

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    runs_dir, out_dir = ROOT / "results" / "runs" / SUITE, ROOT / "results" / "runs" / EVAL_SUITE
    checkpoints = []
    for spec in e9_paired_window():
        ckpt = ROOT / "checkpoints" / SUITE / (spec.filename[:-5] + ".pt")
        if not (runs_dir / spec.filename).exists():
            print(f"skip {spec.filename}: no completed training run", flush=True)
            continue
        if not ckpt.exists():
            sys.exit(f"checkpoint missing for completed run {spec.filename}")
        if any(not (out_dir / record_name(ckpt, w)).exists() for w in E9_WINDOWS):
            checkpoints.append(ckpt)
    print(f"{len(checkpoints)} checkpoints to evaluate at N = {E9_LENGTH}, windows {E9_WINDOWS}", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, str(c), E9_WINDOWS, E9_LENGTH, str(out_dir)) for c in checkpoints]
        for i, fut in enumerate(as_completed(futures), 1):
            name, err, secs = fut.result()
            print(f"[{i}/{len(futures)}] {'ok' if err is None else 'FAILED'} {Path(name).name} ({secs:.0f}s)", flush=True)
            if err:
                failures += 1
                print(err, flush=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / (Path(name).stem + ".error.txt")).write_text(err)
    print(f"done: {len(futures) - failures} ok, {failures} failed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
