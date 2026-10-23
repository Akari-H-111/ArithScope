"""Extra consistency checks (referee; stdlib only).

 1. cosine_learned: window fold kappa_hat_{P,N} as N grows (T9d predicts residue-blindness only as N -> infinity;
    the pilot lives at N <= 8192) and each learned frequency's distance to the nearest 2 pi a / P.
 2. cosine_fund at P = 12: dominance fails, but how often does a random V=16 motif fail? (W = I, 100k motifs)
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json  # noqa: E402

import ptread  # noqa: E402

CK = "/Users/akari_hayami_64/ArithScope-evidence/checkpoints/pilot_evidence/"
runs = json.load(open(os.path.join(HERE, "..", "pilot", "pilot_evidence.json")))["runs"]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def get(period, variant, kernel):
    r = [x for x in runs if x["period"] == period and x["variant"] == variant and x["kernel"] == kernel][0]
    return r, ptread.load(CK + r["file"][:-5] + ".pt")["state_dict"]


say("== 1. cosine_learned: window fold vs N ==")
for P in (12, 13):
    r, sd = get(P, "arith", "cosine_learned")
    om = [float(x) for x in sd["evidence.kernel.omega"]]
    am = [float(x) for x in sd["evidence.kernel.amplitude"]]
    say("P=%d  frequency list (omega, amplitude, distance to nearest 2 pi a/P = delta, 1/delta):" % P)
    for w, a in sorted(zip(om, am)):
        d = min(abs(w - 2 * math.pi * k / P) for k in range(0, P + 1))
        say("    omega=%.5f amp=%+.4f delta=%.5f  1/delta=%.0f" % (w, a, d, 1 / d if d > 0 else float("inf")))
    for N in (8192, 65536, 262144, 1048576):
        sums = [0.0] * P
        for n in range(N):
            sums[n % P] += sum(a * math.cos(w * n) for w, a in zip(om, am))
        cnt = N / P
        fold = [x / cnt for x in sums]
        dom = fold[0] - sum(abs(x) for x in fold[1:])
        say("  N=%8d: kappa_hat(0)=%+.5f  max_{s!=0}|kappa_hat(s)|=%.5f  sum|off|=%.5f  dominance margin=%+.5f" % (
            N, fold[0], max(abs(x) for x in fold[1:]), sum(abs(x) for x in fold[1:]), dom))

say("")
say("== 2. cosine_fund P=12, W=I: failure probability over random V=16 motifs ==")
r, sd = get(12, "arith", "cosine_fund")
coef = [float(x) for x in sd["evidence.kernel.coef"]]
orders = range(1, 17)
P = 12
fold = [0.0] * P
for m, c in zip(orders, coef):
    if P % m == 0:
        for s in range(P):
            fold[s] += c * math.cos(2 * math.pi * (s % m) / m)
say("fold = %s" % [round(x, 3) for x in fold])
V = 16
rng = random.Random(20261023)
M = 100000
bad = 0
worst = 1e9
for _ in range(M):
    mot = [rng.randrange(V) for _ in range(P)]
    ok = True
    for t in range(P):
        c = mot[t]
        cnt = {}
        for s in range(P):
            a = mot[(t - s) % P]
            cnt[a] = cnt.get(a, 0.0) + fold[s]
        base = cnt[c]
        others = max([v for a, v in cnt.items() if a != c] + [0.0])
        marg = base - others
        worst = min(worst, marg)
        if marg <= 1e-9:
            ok = False
    bad += (not ok)
say("motifs with some non-positive linear-rate margin: %d / %d  (rule-of-three 95%% upper bound ~ %.1e); worst margin seen %.3f" % (bad, M, 3.0 / M if bad == 0 else float("nan"), worst))
open(os.path.join(HERE, "consistency_extra.txt"), "w").write("\n".join(OUT) + "\n")
