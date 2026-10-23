"""Theory-vs-pilot consistency for T9 (referee script; stdlib only).

Run:  /usr/bin/python3 -I audit/consistency.py          (a few minutes)

Inputs (read-only):
  pilot/pilot_evidence.json            summaries, learned kernel rows 0..2P-1, W_out summaries, beta, 6 harmonics
  ArithScope-evidence/checkpoints/...  full evidence coefficients and W_out (the JSON does not hold them); read with
                                       audit/ptread.py (zip + pickle, no torch). Cross-checked against the JSON.
  ArithScope-evidence/arithscope/arith.py  exact K_m residue tables (stdlib-only module, loaded by path)

For each evidence arm and period P in {12, 13}:
  * P-fold kappa_P(s) over a full period Lambda = lcm(L, P)           (periodic arms)
  * window fold kappa_hat_{P,N}(s) = (P/N) sum_{n<N, n=s mod P} kappa(n)    (all arms; N = 8192 is the pilot's longest length)
  * dominance margin  kappa_P(0) - sum_{s != 0} |kappa_P(s)|
  * Monte Carlo over MOTIFS random motifs (V=16): linear-rate margins s_t(c,b) (T9c) with W = I and with the learned W_out
  * noise-free finite-i predictor: argmax of W D_i with D_i = E[E_i] at i in the pilot's far region [4096, 8192)
Output: audit/consistency_out.json and audit/consistency_out.txt
"""
import importlib.util
import json
import math
import os
import random
import sys
import time
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptread  # noqa: E402

REPO = "/Users/akari_hayami_64/ArithScope-evidence"
CKPT = REPO + "/checkpoints/pilot_evidence/"
PILOT = os.path.join(HERE, "..", "pilot", "pilot_evidence.json")
SEED = 20261023
V = 16
EPS = 0.5
NU = EPS / V
PP = 1 - EPS + NU          # p
MOTIFS = 4000              # infinite-fold Monte Carlo
MOTIFS_FIN = 1500          # finite-i Monte Carlo
TOL = 1e-9
ORDERS = tuple(range(1, 17))
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


spec = importlib.util.spec_from_file_location("arith_stdlib", REPO + "/arithscope/arith.py")
arith = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arith)
KTAB = {m: [float(x) for x in arith.kernel_table(m)] for m in ORDERS}


# --------------------------------------------------------------------------- kernel objects
class Kernel:
    """kappa(n) = sum_c w_c chi_c(n); chi_c periodic with exact period m_c, or a real-frequency cosine."""

    def __init__(self, periodic, comps=None, omegas=None, amps=None):
        self.periodic = periodic
        self.comps = comps or []          # (w, kind, m, a)
        self.omegas, self.amps = omegas, amps

    def chi(self, kind, m, a, n):
        if kind == "K":
            return KTAB[m][n % m]
        return math.cos(2.0 * math.pi * ((a * n) % m) / m)

    def __call__(self, n):
        if self.periodic:
            return sum(w * self.chi(k, m, a, n) for (w, k, m, a) in self.comps)
        return sum(A * math.cos(om * n) for A, om in zip(self.amps, self.omegas))

    def table(self, N):
        if self.periodic:
            return [self(n) for n in range(N)]
        return [sum(A * math.cos(om * n) for A, om in zip(self.amps, self.omegas)) for n in range(N)]

    def fold_exact(self, P):
        """fold to Z/P: mean of kappa over n = s mod P across a full period (component-wise, each over lcm(m_c,P))."""
        out = [0.0] * P
        for (w, k, m, a) in self.comps:
            Lm = m * P // gcd(m, P)
            for s in range(P):
                tot = sum(self.chi(k, m, a, n) for n in range(s, Lm + s, P))
                out[s] += w * tot / (Lm // P)
        return out

    def fold_bruteforce(self, P, Lam):
        sums, cnt = [0.0] * P, [0] * P
        for n in range(Lam):
            sums[n % P] += self(n)
            cnt[n % P] += 1
        return [s / c for s, c in zip(sums, cnt)]


def window_fold(tab, P, N):
    sums, cnt = [0.0] * P, [0] * P
    for n in range(N):
        sums[n % P] += tab[n]
        cnt[n % P] += 1
    return [s / c for s, c in zip(sums, cnt)]


def full_components():
    return [(m, a) for m in ORDERS for a in arith.units(m) if 2 * a <= m or m == 1]


def build(run, ck):
    sd = ck["state_dict"]
    kn = run["kernel"]
    if kn == "arith":
        return Kernel(True, [(float(b), "K", m, 1) for m, b in zip(ORDERS, sd["evidence.kernel.beta"])])
    if kn == "cosine_fund":
        return Kernel(True, [(float(c), "cos", m, 1) for m, c in zip(ORDERS, sd["evidence.kernel.coef"])])
    if kn == "cosine_full":
        comps = full_components()
        coef = sd["evidence.kernel.coef"]
        assert len(comps) == len(coef) == 41
        return Kernel(True, [(float(c), "cos", m, a) for (m, a), c in zip(comps, coef)])
    if kn == "cosine_learned":
        return Kernel(False, omegas=[float(x) for x in sd["evidence.kernel.omega"]],
                      amps=[float(x) for x in sd["evidence.kernel.amplitude"]])
    if kn == "uniform":
        return Kernel(True, [(1.0, "K", 1, 1)])
    raise ValueError(kn)


# --------------------------------------------------------------------------- motif evaluation
def motif_rows(P, rng):
    return [rng.randrange(V) for _ in range(P)]


def evidence_vec(fold, mot, t, P, scale):
    """E(a) = (scale) * (1/P) sum_s fold[s] pi_{t-s}(a); fold may be a window fold times (i+1) via scale."""
    tot = sum(fold)
    e = [NU * tot] * V
    for s in range(P):
        a = mot[(t - s) % P]
        e[a] += (PP - NU) * fold[s]
    return [x * scale / P for x in e]


def logits(e, W):
    if W is None:
        return e
    return [sum(row[j] * e[j] for j in range(V)) for row in W]


def judge(l, c):
    """returns (min margin of c over others, correct-credit with ties split)."""
    others = [l[b] for b in range(V) if b != c]
    mx = max(others)
    marg = l[c] - mx
    if marg > TOL:
        credit = 1.0
    elif marg < -TOL:
        credit = 0.0
    else:
        ties = sum(1 for b in range(V) if abs(l[b] - l[c]) <= TOL)
        credit = 1.0 / ties
    return marg, credit


def mc_infinite(fold, P, W, seed):
    rng = random.Random(seed)
    all_pos = 0
    res_pos = res_neg = res_zero = 0
    acc = 0.0
    for _ in range(MOTIFS):
        mot = motif_rows(P, rng)
        ok = True
        for t in range(P):
            l = logits(evidence_vec(fold, mot, t, P, 1.0), W)
            marg, credit = judge(l, mot[t])
            acc += credit
            if marg > TOL:
                res_pos += 1
            elif marg < -TOL:
                res_neg += 1
                ok = False
            else:
                res_zero += 1
                ok = False
        all_pos += ok
    n = MOTIFS * P
    return dict(frac_motifs_all_positive=all_pos / MOTIFS, frac_res_positive=res_pos / n,
                frac_res_negative=res_neg / n, frac_res_zero=res_zero / n, pred_acc=acc / n)


def class_sums_upto(tab, P, idx):
    """S_i[s] = sum_{n<=i, n=s mod P} kappa(n) for each i in idx (idx sorted)."""
    out = {}
    S = [0.0] * P
    want = set(idx)
    for n in range(max(idx) + 1):
        S[n % P] += tab[n]
        if n in want:
            out[n] = list(S)
    return out


G0S = (0.0, 2.0, 4.0, 6.0)


def mc_finite(tab, P, W, seed, idx):
    """Noise-free predictor at finite i: argmax of W D_i, D_i = E[E_i] exactly. G0 models the rest of the network
    (T9 H-b) crudely as a constant extra logit G0 on the true symbol; G0 = 0 is the evidence head alone."""
    cs = class_sums_upto(tab, P, idx)
    rng = random.Random(seed)
    accs = {g: [] for g in G0S}
    for _ in range(MOTIFS_FIN):
        mot = motif_rows(P, rng)
        a_m = {g: 0.0 for g in G0S}
        for i in idx:
            t = i % P
            S = cs[i]
            tot = sum(S)
            e = [NU * tot] * V
            for s in range(P):
                e[mot[(t - s) % P]] += (PP - NU) * S[s]
            l = logits(e, W)
            c = mot[t]
            for g in G0S:
                l2 = list(l)
                l2[c] += g
                _, credit = judge(l2, c)
                a_m[g] += credit
        for g in G0S:
            accs[g].append(a_m[g] / len(idx))
    out = {}
    for g in G0S:
        mean = sum(accs[g]) / MOTIFS_FIN
        sd = (sum((x - mean) ** 2 for x in accs[g]) / MOTIFS_FIN) ** 0.5
        out[g] = (mean, sd)
    return out, accs


def identity_W():
    return None


def main():
    t0 = time.time()
    pil = json.load(open(PILOT))
    runs = pil["runs"]
    results = []
    say("T9 vs pilot: consistency (referee). V=%d eps=%.2f p=%.5f nu=%.5f rho=%.1f" % (V, EPS, PP, NU, PP / NU))
    say("Pilot evaluation sizes: N=8192 -> %d sequences, 4096 -> %d, 1024 -> %d (eval_tokens 65536 / eval_tokens_long 16384)" % (16384 // 8192, 16384 // 4096, 65536 // 1024))
    arms = [r for r in runs if r["kernel"] != "none"]
    # order: P, variant, kernel
    arms.sort(key=lambda r: (r["period"], r["variant"] != "arith", ["arith", "cosine_fund", "cosine_full", "cosine_learned", "uniform"].index(r["kernel"])))
    for r in arms:
        P = r["period"]
        ckp = CKPT + r["file"][:-5] + ".pt"
        ck = ptread.load(ckp)
        K = build(r, ck)
        W = ck["state_dict"]["evidence.w_out"]
        # ---- cross-checks against the JSON
        row = r["learned_kernel"]["row_0_to_2P_minus_1"] if "learned_kernel" in r else None
        if row is not None:
            tab = K.table(2 * P)
            dev = max(abs(a - b) for a, b in zip(tab, row))
        else:
            dev = float("nan")
        diag = sum(W[i][i] for i in range(V)) / V
        off = sum(W[i][j] for i in range(V) for j in range(V) if i != j) / (V * (V - 1))
        wj = r.get("w_out", {})
        wdev = max(abs(diag - wj.get("diag_mean", diag)), abs(off - wj.get("offdiag_mean", off))) if wj else float("nan")
        # ---- folds
        N = 8192
        tab = K.table(N)
        folds = {}
        if K.periodic:
            folds["exact"] = K.fold_exact(P)
        for n_win in (256, 1024, 4096, 8192):
            folds["win%d" % n_win] = window_fold(tab, P, n_win)
        main_key = "exact" if K.periodic else "win8192"
        kp = folds[main_key]
        dom = kp[0] - sum(abs(x) for x in kp[1:])
        entry = dict(period=P, variant=r["variant"], kernel=r["kernel"], json_row_dev=dev, wout_summary_dev=wdev,
                     fold_key=main_key, fold=kp, kappa_P0=kp[0], offcomb_abs_sum=sum(abs(x) for x in kp[1:]), dominance_margin=dom,
                     dominance_ratio=(dom / abs(kp[0]) if kp[0] != 0 else float("nan")),
                     win_dom={k: v[0] - sum(abs(x) for x in v[1:]) for k, v in folds.items()},
                     W_diag_mean=diag, W_offdiag_mean=off, lam_eff=kp[0] * (diag - off))
        # brute-force cross-check of the fold over a full period (two arms only; 720720 evaluations)
        if K.periodic and r["variant"] == "arith" and r["kernel"] in ("arith", "cosine_fund"):
            Lam = 720720
            bf = K.fold_bruteforce(P, Lam)
            entry["fold_bruteforce_maxdev"] = max(abs(a - b) for a, b in zip(bf, kp))
        # ---- Monte Carlo, infinite-fold (linear-rate) predictor
        entry["mc_W_identity"] = mc_infinite(kp, P, None, SEED + P)
        entry["mc_W_learned"] = mc_infinite(kp, P, W, SEED + P)
        if not K.periodic:
            pass
        # window-8192 fold version for periodic arms too
        entry["mc_W_learned_win8192"] = mc_infinite(folds["win8192"], P, W, SEED + P)
        # ---- finite-i noise-free predictor in the pilot's far region
        idx = list(range(4096, 4096 + P)) + list(range(8192 - P, 8192))
        fin, accs = mc_finite(tab, P, W, SEED + 7 * P, idx)
        mean, sd = fin[0.0]
        per = accs[0.0]
        entry["finite_i_pred_acc_mean"] = mean
        entry["finite_i_pred_acc_sd_over_motifs"] = sd
        entry["finite_i_pred_acc_by_G0"] = {str(g): fin[g][0] for g in G0S}
        entry["finite_i_pred_acc_sd_by_G0"] = {str(g): fin[g][1] for g in G0S}
        # pilot observation
        L8192 = [x for x in r["lengths"] if x["length"] == 8192][0]
        entry["obs_far_acc_8192"] = L8192["far_acc"]
        entry["obs_far_nll_8192"] = L8192["far_nll"]
        entry["obs_far_over_1nat_frac"] = r["far_stats"]["far_over_1nat_fraction"]
        # 2-sequence sampling: P(mean of 2 random motifs within 0.1 of the observed far acc)
        for g in (0.0, 4.0):
            per_g = accs[g]
            rng = random.Random(SEED)
            hit = 0
            T = 20000
            for _ in range(T):
                a_ = (per_g[rng.randrange(len(per_g))] + per_g[rng.randrange(len(per_g))]) / 2
                if abs(a_ - entry["obs_far_acc_8192"]) <= 0.1:
                    hit += 1
            entry["p_two_motif_mean_within_0.1_of_obs_G0_%g" % g] = hit / T
        results.append(entry)
        say("done P=%d %-10s %-14s  kappa_P(0)=%8.4f  dom=%9.4f  lam_eff=%6.3f  mc_all_pos(W=I)=%.3f  pred_acc(finite)=%.3f obs_far_acc=%.3f  [t=%.0fs]" % (
            P, r["variant"], r["kernel"], entry["kappa_P0"], entry["dominance_margin"], entry["lam_eff"],
            entry["mc_W_identity"]["frac_motifs_all_positive"], mean, entry["obs_far_acc_8192"], time.time() - t0))
    json.dump(results, open(os.path.join(HERE, "consistency_out.json"), "w"), indent=1)
    write_text(results)
    open(os.path.join(HERE, "consistency_out.txt"), "w").write("\n".join(LOG) + "\n")


def write_text(results):
    say("")
    say("=== Summary table (arith-backbone arms; rope variants listed after) ===")
    hdr = "%-3s %-11s %-14s %9s %9s %9s %6s | %-27s | %-27s | %s" % (
        "P", "backbone", "kernel", "kP(0)", "sum|off|", "dom.marg", "lam_ef", "MC W=I: allpos/respos/resneg", "MC W=learned: allpos/respos/resneg", "pred(acc,fin-i) obs far acc / far NLL")
    say(hdr)
    for e in results:
        a, b = e["mc_W_identity"], e["mc_W_learned"]
        say("%-3d %-11s %-14s %9.3f %9.3f %9.3f %6.3f | %.3f / %.3f / %.3f        | %.3f / %.3f / %.3f           | %.3f  obs %.3f / %.3f" % (
            e["period"], e["variant"], e["kernel"], e["kappa_P0"], e["offcomb_abs_sum"], e["dominance_margin"], e["lam_eff"],
            a["frac_motifs_all_positive"], a["frac_res_positive"], a["frac_res_negative"],
            b["frac_motifs_all_positive"], b["frac_res_positive"], b["frac_res_negative"],
            e["finite_i_pred_acc_mean"], e["obs_far_acc_8192"], e["obs_far_nll_8192"]))
    say("")
    say("=== Finite-i noise-free predicted far accuracy (mean over random motifs; SD over single motifs) with a crude base-network gap G0 ===")
    for e in results:
        say("P=%d %-10s %-14s " % (e["period"], e["variant"], e["kernel"]) + "  ".join(
            "G0=%s: %.3f(sd %.3f)" % (g, e["finite_i_pred_acc_by_G0"][g], e["finite_i_pred_acc_sd_by_G0"][g]) for g in e["finite_i_pred_acc_by_G0"]) +
            "   | obs far acc %.3f ; P(2-motif mean within 0.1 of obs): G0=0 %.2f, G0=4 %.2f" % (
            e["obs_far_acc_8192"], e["p_two_motif_mean_within_0.1_of_obs_G0_0"], e["p_two_motif_mean_within_0.1_of_obs_G0_4"]))
    say("")
    say("=== Sanity: checkpoint vs JSON ===")
    for e in results:
        say("P=%d %-10s %-14s  max|kappa(n)-json row|(n<2P)=%.2e   max|W diag/off mean - json|=%.2e   fold brute-force dev=%s" % (
            e["period"], e["variant"], e["kernel"], e["json_row_dev"], e["wout_summary_dev"], ("%.2e" % e["fold_bruteforce_maxdev"]) if "fold_bruteforce_maxdev" in e else "-"))
    say("")
    say("=== Window-fold dominance margins (all arms; N = window length) ===")
    for e in results:
        say("P=%d %-10s %-14s " % (e["period"], e["variant"], e["kernel"]) + "  ".join("%s=%.3f" % (k, v) for k, v in e["win_dom"].items()))
    say("")
    say("=== fold values kappa_P(s) (arith backbone arms) ===")
    for e in results:
        if e["variant"] == "arith":
            say("P=%d %-14s (%s): %s" % (e["period"], e["kernel"], e["fold_key"], " ".join("%.3f" % x for x in e["fold"])))


if __name__ == "__main__":
    main()
