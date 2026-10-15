"""POST-HOC, EXPLORATORY: why NLL rises with length for RoPE-based models while accuracy stays at the oracle.

    python experiments/posthoc_nll.py [--workers 4]

Uses the saved E1 checkpoints (P = 12 and 13; ArithScope only, RoPE + ArithScope, RoPE + full
Fourier bank; seeds 1-5) on a fresh diagnostic test set: 8 sequences of 8,192 tokens drawn with
data seed 20261015, which no earlier evaluation used. Per position bucket it records

* model NLL and accuracy next to the exact Bayes NLL (posterior over the motif symbol given the
  votes seen so far, a closed form for this task);
* the same NLL with every attention layer restricted to the last ``WINDOW`` keys, a causal test of
  whether distant keys cause the excess loss;
* the attention mass that each layer puts off the comb (offsets not divisible by P), in total and
  beyond ``WINDOW``.

It also records top-label calibration (ECE with 15 bins, Brier score) over positions t >= 64.
Writes results/posthoc/nll_diagnostics.json. Positions are 0..N-1, as in every experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LENGTH, SEQS, DATA_SEED, WINDOW = 8192, 8, 20261015, 1024
BUCKETS = ((64, 1024), (1024, 2048), (2048, 4096), (4096, 8192))
VARIANTS = ("arith", "rope+arith", "rope+fourier_full")
ROWS = tuple(range(1024 + 63, LENGTH, 512))  # query positions for the attention-mass probe


def bayes_nll(tokens, targets, period: int, vocab: int, noise: float):
    """Exact Bayes NLL per position: posterior p(mu = c) proportional to L**count_c over the comb seen so far."""
    import torch

    b, n = tokens.shape
    log_ratio = math.log((1 - noise + noise / vocab) / (noise / vocab))
    groups = torch.arange(n) % period
    onehot = torch.zeros(b, n, period, vocab, dtype=torch.float64)
    onehot[torch.arange(b)[:, None], torch.arange(n)[None, :], groups[None, :].expand(b, n), tokens] = 1.0
    counts = onehot.cumsum(1)[torch.arange(b)[:, None], torch.arange(n)[None, :], groups[None, :].expand(b, n)]
    logp = counts * log_ratio - torch.logsumexp(counts * log_ratio, dim=-1, keepdim=True)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def attention_rows(attn, x, positions, rows):
    """Attention weights ``[B, H, R, N]`` of the given query rows, as the explicit backend computes them."""
    import torch

    q, k, _ = attn._qkv(x, positions)
    logits = (q[:, :, rows] @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
    bias = attn.total_bias(positions[rows], positions)
    if bias is not None:
        logits = logits + bias
    allowed = torch.arange(positions.shape[0]).unsqueeze(0) <= rows.unsqueeze(1)
    return torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)


def diagnose(path: str) -> dict:
    import torch
    import torch.nn.functional as F

    import arithscope.attention as attention_module
    from arithscope.tasks import make_task
    from arithscope.train import RunSpec, _generator, build_model, task_dict

    torch.set_num_threads(1)
    blob = torch.load(path, weights_only=False)
    fields = dict(blob["spec"])
    for key in ("orders", "eval_lengths", "shift_evals"):
        fields[key] = tuple(fields[key])
    spec = RunSpec(**fields)
    task = make_task(spec.task["kind"], **{k: v for k, v in spec.task.items() if k != "kind"})
    model = build_model(spec, task.model_vocab)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    period = spec.task["periods"][0]
    batch = task.sample(SEQS, LENGTH, _generator("nll-diagnostic", DATA_SEED, json.dumps(task_dict(task), sort_keys=True),
                                                 LENGTH))
    positions = torch.arange(LENGTH)

    def run(window: int | None):
        original = attention_module.allowed_mask

        def windowed(n_q, n_k, key_padding_mask=None, q_offset=0, device=None):
            allowed = original(n_q, n_k, key_padding_mask, q_offset, device)
            q_idx = torch.arange(q_offset, q_offset + n_q).unsqueeze(1)
            return allowed & (q_idx - torch.arange(n_k).unsqueeze(0) < window)

        if window is not None:
            attention_module.allowed_mask = windowed
        try:
            with torch.no_grad():
                logits = torch.cat([model(batch.tokens[i:i + 1], positions) for i in range(SEQS)])
        finally:
            attention_module.allowed_mask = original
        return logits.double()

    logits = run(None)
    nll = F.cross_entropy(logits.transpose(1, 2), batch.targets, reduction="none")
    correct = (logits.argmax(-1) == batch.targets).double()
    nll_window = F.cross_entropy(run(WINDOW).transpose(1, 2), batch.targets, reduction="none")
    bayes = bayes_nll(batch.tokens, batch.targets, period, task.vocab_size, task.noise)

    probs = logits.softmax(-1)[:, 64:]
    top, pred = probs.max(-1)
    hit = (pred == batch.targets[:, 64:]).double()
    edges = torch.linspace(0, 1, 16, dtype=torch.float64)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (top > lo) & (top <= hi)
        if sel.any():
            ece += float(sel.double().mean() * (top[sel].mean() - hit[sel].mean()).abs())
    onehot = F.one_hot(batch.targets[:, 64:], probs.shape[-1]).double()
    brier = float(((probs - onehot) ** 2).sum(-1).mean())
    err = 1.0 - correct[:, 64:]
    share = float((nll[:, 64:] * err).sum() / nll[:, 64:].sum())

    buckets = []
    for a, b in BUCKETS:
        buckets.append({"start": a, "end": b, "acc": float(correct[:, a:b].mean()), "nll": float(nll[:, a:b].mean()),
                        "nll_window": float(nll_window[:, a:b].mean()), "bayes_nll": float(bayes[:, a:b].mean())})

    rows = torch.tensor(ROWS)
    offsets = rows.unsqueeze(1) - torch.arange(LENGTH).unsqueeze(0)  # [R, N]
    off_comb = ((offsets % period) != 0) & (offsets >= 0)
    far = off_comb & (offsets >= WINDOW)
    layers, x = [], model.embed(batch.tokens)
    with torch.no_grad():
        for block in model.blocks:
            att = attention_rows(block.attn, block.ln1(x), positions, rows)  # [B, H, R, N]
            layers.append({"off_comb": (att * off_comb).sum(-1).mean((0, 2)).tolist(),
                           "far_off_comb": (att * far).sum(-1).mean((0, 2)).tolist()})
            x = block(x, positions)
    gates = []
    for block in model.blocks:
        gates.append([float(g) for g in next(iter(block.attn.biases.values())).gamma.abs()])
    return {"period": period, "variant": spec.variant, "seed": spec.seed, "buckets": buckets,
            "ece": ece, "brier": brier, "nll_share_from_errors": share, "attention": layers, "gates": gates,
            "digest": batch.digest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ckpts = sorted(str(p) for p in (ROOT / "checkpoints" / "e1_periodic").glob("*.pt")
                   if p.name.split("__")[1] in VARIANTS)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(diagnose, ckpts))
    from arithscope.train import environment_info

    out = ROOT / "results" / "posthoc" / "nll_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "POST-HOC EXPLORATORY (not preregistered)", "length": LENGTH,
                               "sequences": SEQS, "data_seed": DATA_SEED, "window": WINDOW,
                               "query_rows": list(ROWS), "environment": environment_info(), "runs": results},
                              indent=1))
    print(f"wrote {out} ({len(results)} checkpoints)")


if __name__ == "__main__":
    main()
