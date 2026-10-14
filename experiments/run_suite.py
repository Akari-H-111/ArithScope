"""Run one or more named experiment suites in one worker pool (resumable: finished runs are skipped).

    python experiments/run_suite.py <suite> [<suite> ...] [--workers 4] [--out results/runs]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arithscope.train import RunSpec, run  # noqa: E402
from experiments.suites import SUITES  # noqa: E402


def _worker(spec: RunSpec, out_dir: str, ckpt_dir: str) -> tuple[str, str | None, float]:
    start = time.perf_counter()
    try:
        run(spec, out_dir, threads=1, checkpoint_dir=ckpt_dir)
        return spec.filename, None, time.perf_counter() - start
    except Exception:  # noqa: BLE001 - failures are reported, not hidden
        return spec.filename, traceback.format_exc(), time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="+", choices=sorted(SUITES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = []
    for suite in args.suites:
        out_dir = Path(args.out) / suite
        specs = SUITES[suite]()
        todo = [s for s in specs if not (out_dir / s.filename).exists()]
        print(f"suite {suite}: {len(specs)} runs, {len(todo)} to do -> {out_dir}", flush=True)
        jobs += [(s, out_dir, ROOT / "checkpoints" / suite) for s in todo]
    if args.dry_run:
        for s, out_dir, _ in jobs:
            print(" ", out_dir.name, s.filename)
        return
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, s, str(o), str(c)): o for s, o, c in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            name, err, secs = fut.result()
            status = "ok" if err is None else "FAILED"
            print(f"[{i}/{len(jobs)}] {status} {futures[fut].name}/{name} ({secs:.0f}s)", flush=True)
            if err:
                failures += 1
                print(err, flush=True)
                futures[fut].mkdir(parents=True, exist_ok=True)
                (futures[fut] / (name + ".error.txt")).write_text(err)
    print(f"done: {len(jobs) - failures} ok, {failures} failed", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
