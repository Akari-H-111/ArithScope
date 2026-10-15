"""POST-HOC, EXPLORATORY: is the long-range NLL floor a limit of the readout or of the representation?

    python experiments/posthoc_readout.py

Every model ends in ``LayerNorm -> Linear`` (no bias), so its logits are bounded and its NLL can never
reach the Bayes NLL, which tends to 0. This script measures how low the readout alone could go. For
each E1 checkpoint diagnosed by ``posthoc_nll.py`` and each target symbol ``c``, it minimizes
``-log softmax(W (gamma * z + beta))[c]`` over LayerNorm outputs: ``z`` with mean 0 and
``|z|^2 = d / (1 + eps)``, which is what LayerNorm returns for an input with variance 1. Projected
gradient descent from several random starts; the best value found is an achievable NLL, i.e. an upper
bound on the readout's minimum. Weights only, no data. Writes results/posthoc/readout_floor.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = ("arith", "rope+arith", "rope+fourier_full")
RESTARTS, STEPS, LR = 4, 300, 0.5


def readout_floor(weight, gamma, beta, eps: float, target: int, seed: int) -> float:
    """Smallest NLL of ``target`` found over LayerNorm outputs (an achievable value)."""
    import torch

    d = weight.shape[1]
    radius = math.sqrt(d / (1 + eps))
    others = [j for j in range(weight.shape[0]) if j != target]
    gen = torch.Generator().manual_seed(seed)
    best = math.inf
    for _ in range(RESTARTS):
        z = torch.randn(d, generator=gen, dtype=torch.float64)
        for _ in range(STEPS + 1):
            z = z - z.mean()
            z = (z * (radius / z.norm())).requires_grad_(True)  # a LayerNorm output: mean 0, fixed norm
            logits = weight @ (gamma * z + beta)
            # log(exp(NLL) - 1): same minimizer as the NLL, without its vanishing gradient as p -> 1
            obj = torch.logsumexp(logits[others] - logits[target], 0)
            (grad,) = torch.autograd.grad(obj, z)
            z = (z - LR * grad).detach()
        best = min(best, float(torch.nn.functional.softplus(obj.detach())))  # NLL at the last projected point
    return best


def main() -> None:
    import torch

    torch.set_num_threads(1)
    rows = []
    for path in sorted((ROOT / "checkpoints" / "e1_periodic").glob("*.pt")):
        if path.name.split("__")[1] not in VARIANTS:
            continue
        blob = torch.load(path, weights_only=False)
        sd, spec = blob["state_dict"], blob["spec"]
        weight, gamma, beta = (sd[k].double() for k in ("head.weight", "ln_f.weight", "ln_f.bias"))
        floors = [readout_floor(weight, gamma, beta, 1e-5, c, seed=c) for c in range(weight.shape[0])]
        rows.append({"period": spec["task"]["periods"][0], "variant": spec["variant"], "seed": spec["seed"],
                     "floor_by_target": floors, "max_floor": max(floors)})
    out = ROOT / "results" / "posthoc" / "readout_floor.json"
    out.write_text(json.dumps({"status": "POST-HOC EXPLORATORY (not preregistered)", "restarts": RESTARTS,
                               "steps": STEPS, "lr": LR, "runs": rows}, indent=1))
    print(f"wrote {out} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
