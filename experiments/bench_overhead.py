"""Runtime and peak-memory overhead of the ArithScope bias backends (CPU).

Each measurement runs in a fresh subprocess so peak RSS is attributable to it.
Run on an otherwise idle machine; results go to results/overhead/overhead.json.

    python experiments/bench_overhead.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIGS = {
    # name: (d_model, heads, orders)
    "tiny_d16_M1-16": (64, 4, tuple(range(1, 17))),
    "llm_d64_B16": (512, 8, (2, 3, 4, 5, 7, 8, 9, 11, 13, 16)),
}
MODES = ("baseline_sdpa", "arith_explicit", "arith_sdpa_mask", "arith_folded")


def measure(config: str, mode: str, length: int, backward: bool, repeats: int) -> dict:
    import torch

    from arithscope.attention import CausalSelfAttention, PositionConfig

    torch.set_num_threads(4)
    d_model, heads, orders = CONFIGS[config]
    variant = "rope" if mode == "baseline_sdpa" else "rope+arith"
    layer = CausalSelfAttention(d_model, heads, PositionConfig.from_name(variant, orders=orders))
    if "arith" in layer.biases:
        with torch.no_grad():
            layer.biases["arith"].gamma.fill_(1.0)
    backend = {"baseline_sdpa": "sdpa", "arith_explicit": "explicit", "arith_sdpa_mask": "sdpa",
               "arith_folded": "folded"}[mode]
    x = torch.randn(1, length, d_model, requires_grad=backward)
    pos = torch.arange(length)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def step():
        out = layer(x, pos, backend=backend)
        if backward:
            out.sum().backward()

    step()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"config": config, "mode": mode, "length": length, "backward": backward, "repeats": repeats,
            "median_seconds": statistics.median(times), "min_seconds": min(times),
            "peak_rss_increase_mb": (rss_after - rss_before) / 1024.0, "torch_threads": torch.get_num_threads(),
            "head_dim": d_model // heads, "folded_extra_dims": sum(__import__("arithscope").arith.euler_phi(m) for m in orders)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "results" / "overhead" / "overhead.json"))
    parser.add_argument("--child", nargs=5, metavar=("CONFIG", "MODE", "LENGTH", "BACKWARD", "REPEATS"))
    args = parser.parse_args()
    if args.child:
        config, mode, length, backward, repeats = args.child
        print(json.dumps(measure(config, mode, int(length), backward == "1", int(repeats))))
        return
    lengths = (256, 512) if args.quick else (512, 1024, 2048, 4096, 8192)
    rows = []
    for config in CONFIGS:
        for backward in (False, True):
            for length in lengths:
                for mode in MODES:
                    heads = CONFIGS[config][1]
                    dense_gb = heads * length * length * 4 * (3 if backward else 1) / 2**30
                    if mode != "baseline_sdpa" and mode != "arith_folded" and dense_gb > 4:
                        rows.append({"config": config, "mode": mode, "length": length, "backward": backward,
                                     "skipped": f"dense [H,N,N] tensors would need ~{dense_gb:.1f} GB"})
                        continue
                    repeats = 5 if length <= 2048 else 3
                    proc = subprocess.run([sys.executable, __file__, "--child", config, mode, str(length),
                                           "1" if backward else "0", str(repeats)],
                                          capture_output=True, text=True, check=False)
                    if proc.returncode != 0:
                        rows.append({"config": config, "mode": mode, "length": length, "backward": backward,
                                     "error": proc.stderr[-2000:]})
                    else:
                        rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
                    print(json.dumps(rows[-1]), flush=True)
    from arithscope.train import environment_info

    env = environment_info()
    env["torch_threads"] = 4
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": "arithscope-overhead-1", "environment": env, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
