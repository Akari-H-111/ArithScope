"""E10 copied-law evaluations (research/evidence_readout/2026-10-23/E10_SPEC.md).

    python experiments/e10_copy_eval.py [--suite e10_evidence] [--workers 4]

For every checkpoint of the suite (checkpoints/<suite>/*.pt, written by the training runs), the unchanged
weights are evaluated on the copied-observation law (``periodic_copied``, k = E10_COPIES = 4, the training
run's periods, vocabulary and noise) at N = 256 and 1,024. The test set is the one ``train.run`` would draw
for this length from the copied task's dict (same generator parts and sequence count, the run's data seed).
The exact copied-law Bayes NLL on the same batch is recorded next to the model's metrics (description only).
Writes one record per (checkpoint, length) to results/runs/<suite>_copy_eval/; finished records are skipped,
so the script resumes after an interruption. ``--suite pilot_e10_spread`` evaluates the spread pilot.
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

SUITES = ("e10_evidence", "pilot_e10_spread")


def record_name(checkpoint: Path, length: int) -> str:
    return f"{checkpoint.stem}__N{length}.json"


def load_checkpoint(path: Path):
    """(spec, model) with the stored weights, in eval mode."""
    import torch

    from arithscope.tasks import make_task
    from arithscope.train import RunSpec, build_model

    blob = torch.load(path, weights_only=False)
    fields = dict(blob["spec"])
    for key in ("orders", "eval_lengths", "shift_evals", "attn_windows"):
        if key in fields:
            fields[key] = tuple(fields[key])
    spec = RunSpec(**fields)
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    model = build_model(spec, task.model_vocab)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return spec, model


def copied_task(spec, copies: int):
    from arithscope.tasks import make_task

    kwargs = {k: v for k, v in spec.task.items() if k not in ("kind", "query_rate")}
    return make_task("periodic_copied", copies=copies, **kwargs)


def evaluate_checkpoint(checkpoint: Path, lengths: tuple[int, ...], out_dir: Path, copies: int | None = None) -> list[Path]:
    """Evaluate one checkpoint on the copied law at each length; returns the records written."""
    import torch

    from arithscope.tasks import periodic_bayes_nll
    from arithscope.train import _generator, environment_info, evaluate, task_dict
    from experiments.suites import E10_COPIES

    copies = E10_COPIES if copies is None else copies
    torch.set_num_threads(1)
    spec, model = load_checkpoint(checkpoint)
    if spec.task["kind"] != "periodic":
        raise ValueError(f"E10 checkpoints are periodic-denoising runs: {checkpoint.name}")
    task = copied_task(spec, copies)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        dest = out_dir / record_name(checkpoint, length)
        if dest.exists():
            continue
        budget = spec.eval_tokens if length <= 2048 else spec.eval_tokens_long
        num_seqs = max(2, budget // length)  # as in train.run
        parts = ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), length)  # as in train.run
        result = evaluate(model, task, length, num_seqs, parts, spec.train_length)
        batch = task.sample(num_seqs, length, _generator(*parts))  # the batch evaluate() drew
        assert batch.digest() == result["digest"]
        bayes = periodic_bayes_nll(batch, task)
        extrapolated = bayes[:, spec.train_length:]
        record = {"schema_version": "arithscope-copy-eval-1", "status": "COMPLETED",
                  "source_run": checkpoint.stem + ".json", "checkpoint_sha256": digest, "spec": asdict(spec),
                  "copied_task": task_dict(task), "length": length, "test": [result],
                  "bayes_nll": {"all": float(bayes.mean()),
                                "extrapolated": float(extrapolated.mean()) if extrapolated.numel() else None},
                  "environment": environment_info()}
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, dest)
        written.append(dest)
    return written


def _worker(checkpoint: str, lengths: tuple[int, ...], out_dir: str) -> tuple[str, str | None, float]:
    start = time.perf_counter()
    try:
        evaluate_checkpoint(Path(checkpoint), lengths, Path(out_dir))
        return checkpoint, None, time.perf_counter() - start
    except Exception:  # noqa: BLE001 - failures are reported, not hidden
        return checkpoint, traceback.format_exc(), time.perf_counter() - start


def main() -> None:
    from experiments.suites import E10_COPY_LENGTHS, SUITES as ALL_SUITES

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="e10_evidence", choices=SUITES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    runs_dir = ROOT / "results" / "runs" / args.suite
    out_dir = ROOT / "results" / "runs" / (args.suite + "_copy_eval")
    checkpoints = []
    for spec in ALL_SUITES[args.suite]():
        ckpt = ROOT / "checkpoints" / args.suite / (spec.filename[:-5] + ".pt")
        if not (runs_dir / spec.filename).exists():
            print(f"skip {spec.filename}: no completed training run", flush=True)
            continue
        if not ckpt.exists():
            sys.exit(f"checkpoint missing for completed run {spec.filename}")
        if any(not (out_dir / record_name(ckpt, n)).exists() for n in E10_COPY_LENGTHS):
            checkpoints.append(ckpt)
    print(f"{len(checkpoints)} checkpoints to evaluate on the copied law at N = {E10_COPY_LENGTHS}", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, str(c), E10_COPY_LENGTHS, str(out_dir)) for c in checkpoints]
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
