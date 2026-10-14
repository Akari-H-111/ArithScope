"""Training / evaluation harness. One call to :func:`run` = one JSON result file.

Protocol (docs/PREREGISTRATION.md): fresh training batches every step from a
seed-specific generator; a fixed validation set at the training length; fixed
test sets per evaluation length that are identical across variants and seeds
(their SHA-256 digests are logged), so variant comparisons are paired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from . import arith
from .attention import PositionConfig
from .bias import CosineBankBias, LearnedFrequencyBias, RamanujanBias, RelativeBias
from .model import ModelConfig, TinyDecoder
from .tasks import make_task, task_dict


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    task: dict                      # {"kind": ..., **task kwargs}
    variant: str                    # PositionConfig name, e.g. "rope+arith"
    seed: int
    orders: tuple[int, ...] = tuple(range(1, 17))
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 256
    train_length: int = 64
    batch_size: int = 32
    steps: int = 2000
    lr: float = 2e-3
    pos_lr_mult: float = 10.0
    weight_decay: float = 0.01
    warmup: int = 100
    grad_clip: float = 1.0
    eval_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    eval_tokens: int = 65536        # predictions per evaluation length (sequences = tokens // length)
    eval_tokens_long: int = 16384   # predictions per evaluation length above 2048 tokens
    shift_eval_length: int = 256    # length used for the absolute-position-shift evaluations
    shift_evals: tuple[int, ...] = ()  # constant shifts added to all positions (relative structure unchanged)
    val_every: int = 250
    data_seed: int = 20261014       # test/validation data seed, shared by all variants and seeds

    @property
    def run_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    @property
    def filename(self) -> str:
        return f"{self.experiment}__{self.variant}__s{self.seed}__{self.run_id}.json"


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") & (2**62 - 1)


def _generator(*parts) -> torch.Generator:
    return torch.Generator().manual_seed(_seed(*parts))


def environment_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent.parent, check=False).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parent.parent, check=False).stdout.strip())
    except OSError:
        commit, dirty = None, None
    cpu = None
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), None)
    except OSError:
        pass
    return {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "cpu": cpu, "torch_threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
            "git_commit": commit, "git_dirty": dirty}


def build_model(spec: RunSpec, vocab: int) -> TinyDecoder:
    pos = PositionConfig.from_name(spec.variant, orders=tuple(spec.orders),
                                   relbias_max_distance=spec.train_length)
    cfg = ModelConfig(vocab_size=vocab, d_model=spec.d_model, num_layers=spec.num_layers,
                      num_heads=spec.num_heads, d_ff=spec.d_ff, pos=pos)
    return TinyDecoder(cfg)


def _optimizer(model: TinyDecoder, spec: RunSpec) -> torch.optim.Optimizer:
    pos_ids = {id(p) for p in model.positional_parameters()}
    decay, no_decay, positional = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in pos_ids:
            positional.append(p)
        elif p.dim() >= 2 and "embed" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": spec.weight_decay, "lr_mult": 1.0},
              {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    if positional:
        groups.append({"params": positional, "weight_decay": 0.0, "lr_mult": spec.pos_lr_mult})
    return torch.optim.AdamW(groups, lr=spec.lr, betas=(0.9, 0.98))


def _lr_factor(step: int, spec: RunSpec) -> float:
    if step < spec.warmup:
        return (step + 1) / spec.warmup
    progress = (step - spec.warmup) / max(1, spec.steps - spec.warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _masked_ce(logits: torch.Tensor, batch) -> torch.Tensor:
    mask = batch.loss_mask
    return F.cross_entropy(logits[mask], batch.targets[mask])


def _eval_chunk(length: int) -> int:
    """Sequences per forward pass, keeping the explicit [chunk, H, N, N] logits near 64 MB."""
    return max(1, min(16, int(16 * (512 / length) ** 2)))


@torch.no_grad()
def evaluate(model: TinyDecoder, task, length: int, num_seqs: int, gen_parts: tuple,
             train_length: int, position_shift: int = 0, alias_vote_period: int | None = None) -> dict:
    """Accuracy/loss on a fixed test set, overall and on the extrapolated region ``t >= train_length``.

    ``position_shift`` adds a constant to every position (relative structure unchanged).
    ``alias_vote_period`` additionally reports the vote oracle over that comb (periodic tasks).
    """
    model.eval()
    batch = task.sample(num_seqs, length, _generator(*gen_parts))
    alias = task.vote_oracle(batch, alias_vote_period) if alias_vote_period else None
    correct = torch.zeros(num_seqs, length)
    nll = torch.zeros(num_seqs, length)
    positions = torch.arange(length) + position_shift
    chunk = _eval_chunk(length)
    start = time.perf_counter()
    for i in range(0, num_seqs, chunk):
        logits = model(batch.tokens[i:i + chunk], positions)
        nll[i:i + chunk] = F.cross_entropy(logits.transpose(1, 2), batch.targets[i:i + chunk], reduction="none")
        correct[i:i + chunk] = (logits.argmax(-1) == batch.targets[i:i + chunk]).float()
    elapsed = time.perf_counter() - start
    mask = batch.loss_mask.float()
    region = mask.clone()
    region[:, :train_length] = 0.0  # positions never seen during training

    def stats(weights: torch.Tensor) -> dict:
        n = float(weights.sum())
        if n == 0:
            return {"n": 0, "acc": None, "oracle_acc": None, "nll": None}
        out = {"n": int(n), "acc": float((correct * weights).sum() / n),
               "oracle_acc": float((batch.oracle * weights).sum() / n), "nll": float((nll * weights).sum() / n)}
        if alias is not None:
            out["alias_oracle_acc"] = float((alias * weights).sum() / n)
        return out

    out = {"length": length, "num_seqs": num_seqs, "position_shift": position_shift, "digest": batch.digest(),
           "all": stats(mask), "extrapolated": stats(region) if length > train_length else None,
           "eval_seconds": elapsed, "tokens_per_second": num_seqs * length / elapsed}
    groups = batch.group.unique().tolist()
    if len(groups) > 1:
        out["by_group"] = {str(g): stats(mask * (batch.group == g).float()[:, None]) for g in groups}
    buckets = 8
    edges = torch.linspace(0, length, buckets + 1).long().tolist()
    out["by_position"] = []
    for a, b in zip(edges[:-1], edges[1:]):
        w = torch.zeros_like(mask)
        w[:, a:b] = mask[:, a:b]
        out["by_position"].append({"start": a, "end": b, **stats(w)})
    model.train()
    return out


def positional_snapshot(model: TinyDecoder) -> list[dict]:
    snap = []
    for layer, blk in enumerate(model.blocks):
        entry = {"layer": layer}
        for kind, module in blk.attn.biases.items():
            if isinstance(module, RamanujanBias):
                entry[kind] = {"gamma": module.gamma.tolist(), "alpha": module.alpha.tolist(),
                               "orders": list(module.orders)}
            elif isinstance(module, CosineBankBias):
                entry[kind] = {"gamma": module.gamma.tolist(), "alpha": module.alpha.tolist(),
                               "components": [list(c) for c in module.components]}
            elif isinstance(module, LearnedFrequencyBias):
                entry[kind] = {"omega": module.omega.tolist(), "amplitude": module.amplitude.tolist()}
            elif isinstance(module, RelativeBias):
                entry[kind] = {"table": module.table.tolist(), "max_distance": module.max_distance}
        snap.append(entry)
    return snap


def run(spec: RunSpec, out_dir: str | Path, threads: int = 1, verbose: bool = False,
        checkpoint_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec.filename
    if out_path.exists():
        return json.loads(out_path.read_text())
    torch.set_num_threads(threads)
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    torch.manual_seed(_seed("init", spec.seed))
    model = build_model(spec, task.model_vocab)
    opt = _optimizer(model, spec)
    train_gen = _generator("train", spec.experiment, spec.seed)
    val_parts = ("val", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), spec.train_length)
    history, val_history = [], []
    window, step_times = [], []
    started = time.perf_counter()
    for step in range(spec.steps):
        for group in opt.param_groups:
            group["lr"] = spec.lr * group["lr_mult"] * _lr_factor(step, spec)
        batch = task.sample(spec.batch_size, spec.train_length, train_gen)
        t0 = time.perf_counter()
        loss = _masked_ce(model(batch.tokens), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        opt.step()
        step_times.append(time.perf_counter() - t0)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        window.append(loss.item())
        if (step + 1) % 50 == 0:
            history.append({"step": step + 1, "loss": sum(window) / len(window)})
            window = []
        if (step + 1) % spec.val_every == 0 or step + 1 == spec.steps:
            val = evaluate(model, task, spec.train_length, 128, val_parts, spec.train_length)
            val_history.append({"step": step + 1, "acc": val["all"]["acc"], "nll": val["all"]["nll"]})
            if verbose:
                print(f"[{spec.variant} s{spec.seed}] step {step + 1} loss {history[-1]['loss']:.4f} "
                      f"val acc {val['all']['acc']:.4f}", flush=True)
    train_seconds = time.perf_counter() - started
    alias_period = None
    if task.name == "periodic" and len(task.periods) == 1:
        candidate = math.gcd(task.periods[0], arith.lcm(spec.orders))
        alias_period = candidate if candidate != task.periods[0] else None

    def test_parts(length: int) -> tuple:
        return ("test", spec.data_seed, json.dumps(task_dict(task), sort_keys=True), length)

    def num_seqs(length: int) -> int:
        budget = spec.eval_tokens if length <= 2048 else spec.eval_tokens_long
        return max(2, budget // length)

    results = [evaluate(model, task, length, num_seqs(length), test_parts(length),
                        spec.train_length, alias_vote_period=alias_period) for length in spec.eval_lengths]
    shift_results = [evaluate(model, task, spec.shift_eval_length, max(4, spec.eval_tokens // spec.shift_eval_length),
                              test_parts(spec.shift_eval_length), spec.train_length, position_shift=shift)
                     for shift in spec.shift_evals]
    if checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir) / (spec.filename[:-5] + ".pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"spec": asdict(spec), "state_dict": model.state_dict()}, ckpt)
    record = {
        "schema_version": "arithscope-run-1",
        "status": "COMPLETED",
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "task": task_dict(task),
        "model": model.cfg.to_dict(),
        "parameters": model.count_parameters(),
        "environment": environment_info(),
        "train": {"seconds": train_seconds, "mean_step_seconds": sum(step_times) / len(step_times),
                  "tokens_seen": spec.steps * spec.batch_size * spec.train_length,
                  "loss_curve": history, "val_curve": val_history},
        "test": results,
        "shift_test": shift_results,
        "alias_vote_period": alias_period,
        "positional": positional_snapshot(model),
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, out_path)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one ArithScope run.")
    parser.add_argument("spec", help="JSON file with RunSpec fields")
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    fields = json.loads(Path(args.spec).read_text())
    for key in ("orders", "eval_lengths", "shift_evals"):
        if key in fields:
            fields[key] = tuple(fields[key])
    record = run(RunSpec(**fields), args.out, args.threads, verbose=True)
    print(json.dumps({"run_id": record["run_id"], "test": [(r["length"], r["all"]["acc"]) for r in record["test"]]}))


if __name__ == "__main__":
    main()
