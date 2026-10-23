#!/usr/bin/env python3
"""Exact and numerical checks for the T9 draft (exact-period evidence accumulation).

Standard library only.  Run from this directory:

    /usr/bin/python3 -B check_t9.py --json results_t9.json

Labels used in the output follow docs/MATH.md: EXACT = Fraction / integer arithmetic (a finite
computation, no floating point); NUMERICAL = anything that uses floating point or simulation,
including enumerations whose probabilities are exact Fractions but whose logarithms are floating
point.

Notation follows docs/MATH.md: arithmetic indices are *orders* ``m``; the attention
query is never called ``q`` here.  The per-wrong-symbol noise probability of B1
(called ``q`` there) is called ``nu`` here to avoid the collision.

``arithscope/arith.py`` is loaded by file path (not through the package) so that the
package ``__init__`` (which may import torch) is never executed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # .../ArithScope
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20261023


def _load_arith():
    path = ROOT / "arithscope" / "arith.py"
    spec = importlib.util.spec_from_file_location("_t9_arith", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["_t9_arith"] = module
    spec.loader.exec_module(module)
    return module


arith = _load_arith()


# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------
def jsonable(obj):
    """Recursively convert Fractions to strings so that results can be dumped as JSON."""
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    return obj


def rank_q(matrix):
    """Rank over Q of a matrix given as a list of rows (Fractions/ints), exact."""
    a = [[Fraction(x) for x in row] for row in matrix]
    if not a or not a[0]:
        return 0
    nrows, ncols = len(a), len(a[0])
    r = 0
    for c in range(ncols):
        piv = None
        for i in range(r, nrows):
            if a[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        a[r], a[piv] = a[piv], a[r]
        pv = a[r][c]
        a[r] = [x / pv for x in a[r]]
        for i in range(r + 1, nrows):
            f = a[i][c]
            if f != 0:
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        r += 1
        if r == nrows:
            break
    return r


def logsumexp(xs):
    mx = max(xs)
    return mx + math.log(sum(math.exp(x - mx) for x in xs))


def noise_params(V, eps):
    """B1 channel: p = 1-eps+eps/V (correct symbol), nu = eps/V (each wrong symbol), rho = p/nu."""
    eps = Fraction(eps)
    nu = eps / V
    p = 1 - eps + nu
    return p, nu, p / nu


def chi_value(V, p, nu):
    """chi = 1 - (sqrt p - sqrt nu)^2 = 2 sqrt(p nu) + (V-2) nu (B1's kappa; renamed: kappa is the kernel here)."""
    return 2.0 * math.sqrt(float(p * nu)) + (V - 2) * float(nu)


def identity(V):
    return [[Fraction(1 if i == j else 0) for j in range(V)] for i in range(V)]


# ----------------------------------------------------------------------------
# (a) construction
# ----------------------------------------------------------------------------
def d_p_value(P, n):
    """D_P(n) = sum_{m | P} phi(m)/P * K_m(n), computed from the exact kernels of arith.py."""
    return sum((Fraction(arith.euler_phi(m), P) * arith.kernel(m, n) for m in arith.divisors(P)), Fraction(0))


def check_a_identity(pmax=32):
    bad = []
    convex_ok = True
    for P in range(2, pmax + 1):
        divs = arith.divisors(P)
        if sum(arith.euler_phi(d) for d in divs) != P:
            convex_ok = False
        for n in range(-3 * P, 3 * P + 1):
            if d_p_value(P, n) != (1 if n % P == 0 else 0):
                bad.append((P, n))
    return {
        "label": "EXACT",
        "statement": "D_P(n) = sum_{m|P} phi(m)/P K_m(n) = 1[P | n]; weights phi(m)/P sum to 1",
        "P_range": [2, pmax],
        "offsets": "n in [-3P, 3P]",
        "weights_sum_to_one": convex_ok,
        "failures": bad,
        "ok": (not bad) and convex_ok,
    }


def posterior_product(seq, V, p, nu):
    """Posterior of the motif symbol (uniform prior) from the product likelihood, exact."""
    w = []
    for a in range(V):
        like = Fraction(1)
        for x in seq:
            like *= p if x == a else nu
        w.append(like)
    s = sum(w)
    return [x / s for x in w]


def softmax_counts_exact(counts, rho):
    """softmax(log(rho) * counts) computed exactly: exp(log(rho) * c) = rho ** c."""
    w = [rho ** c for c in counts]
    s = sum(w)
    return [x / s for x in w]


def softmax_counts_float(counts, lam):
    xs = [lam * c for c in counts]
    mx = max(xs)
    e = [math.exp(x - mx) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def check_a_bayes_equality(cases=((3, Fraction(1, 2), 6), (4, Fraction(1, 3), 4), (16, Fraction(1, 2), 3))):
    out = []
    ok_all = True
    for V, eps, nmax in cases:
        p, nu, rho = noise_params(V, eps)
        lam = math.log(float(rho))
        bad_exact = 0
        max_float_err = 0.0
        n_seq = 0
        seen = set()
        for n in range(nmax + 1):
            for seq in product(range(V), repeat=n):
                counts = [0] * V
                for x in seq:
                    counts[x] += 1
                post = posterior_product(seq, V, p, nu)
                soft = softmax_counts_exact(counts, rho)
                n_seq += 1
                if post != soft:
                    bad_exact += 1
                key = tuple(counts)
                if key not in seen:
                    seen.add(key)
                    sf = softmax_counts_float(counts, lam)
                    err = max(abs(float(x) - y) for x, y in zip(soft, sf))
                    max_float_err = max(max_float_err, err)
        ok = bad_exact == 0 and max_float_err < 1e-12
        ok_all = ok_all and ok
        out.append({
            "V": V, "eps": eps, "p": p, "nu": nu, "rho": rho, "n_max": nmax,
            "sequences_checked": n_seq, "count_vectors_float_checked": len(seen),
            "exact_mismatches": bad_exact,
            "max_abs_err_float_softmax_vs_exact (NUMERICAL)": max_float_err,
            "ok": ok,
        })
    return {"label": "EXACT (float softmax comparison: NUMERICAL)",
            "statement": "softmax(log rho * counts) == product-likelihood posterior, exactly",
            "cases": out, "ok": ok_all}


def check_a_head_comb_counts(Ps=(7, 12, 13), V=5, length=90, seed=SEED):
    """With kappa = D_P (built from exact Ramanujan kernels) E_i(a) is the comb count N_i(a)."""
    rng = random.Random(seed)
    bad = []
    for P in Ps:
        xs = [rng.randrange(V) for _ in range(length)]
        dtab = [d_p_value(P, r) for r in range(P)]
        for i in range(length):
            for a in range(V):
                e = sum(dtab[(i - j) % P] for j in range(i + 1) if xs[j] == a)
                n_comb = sum(1 for j in range(i + 1) if (i - j) % P == 0 and xs[j] == a)
                if e != n_comb:
                    bad.append((P, i, a))
    return {"label": "EXACT", "Ps": list(Ps), "length": length, "failures": bad, "ok": not bad}


def count_states(V, p, nu, m):
    """Exact law of the count state after m independent observations of one residue, true symbol c.

    State = (M_c, sorted tuple of the V-1 wrong-symbol counts, descending).  Probabilities are Fractions.
    """
    states = {(0, (0,) * (V - 1)): Fraction(1)}
    for _ in range(m):
        new = {}
        for (mc, wr), pr in states.items():
            key = (mc + 1, wr)
            new[key] = new.get(key, Fraction(0)) + pr * p
            for idx in range(V - 1):
                lst = list(wr)
                lst[idx] += 1
                key = (mc, tuple(sorted(lst, reverse=True)))
                new[key] = new.get(key, Fraction(0)) + pr * nu
        states = new
    return states


def expected_nll(states, lam, k, B=0.0):
    """E[NLL] of a head with logits k*lam*M_a + b_a, b_true = 0, b_wrong = B (worst-case oscillation B)."""
    tot = 0.0
    for (mc, wr), pr in states.items():
        terms = [0.0] + [k * lam * (w - mc) + B for w in wr]
        tot += float(pr) * logsumexp(terms)
    return tot


def check_a_rate_bound(cases=((16, Fraction(1, 2), 6), (3, Fraction(1, 2), 8)), B_values=(0.0, 1.0, 3.0)):
    out = []
    ok_all = True
    for V, eps, mmax in cases:
        p, nu, rho = noise_params(V, eps)
        lam = math.log(float(rho))
        chi = chi_value(V, p, nu)
        rows = []
        for m in range(1, mmax + 1):
            states = count_states(V, p, nu, m)
            if sum(states.values()) != 1:
                ok_all = False
            row = {"m": m, "chi_pow_m": chi ** m}
            for B in B_values:
                e = expected_nll(states, lam, 1, B)
                bound = (V - 1) * math.exp(B / 2.0) * chi ** m
                row["B=%g" % B] = {"E_NLL": e, "bound (V-1) e^{B/2} chi^m": bound, "ok": e <= bound + 1e-12,
                                   "E_NLL / chi^m": e / chi ** m}
                ok_all = ok_all and e <= bound + 1e-12
            rows.append(row)
        out.append({
            "V": V, "eps": eps, "p": p, "nu": nu, "rho": rho, "chi": chi,
            "as_slope_per_vote_(p-nu)log(rho)": float((p - nu)) * lam,
            "expectation_exponent_-log(chi)": -math.log(chi),
            "rows": rows,
        })
    return {"label": "NUMERICAL (probabilities exact Fractions, logarithms floating point); the bound itself is PROVED in T9a",
            "cases": out, "ok": ok_all}


# ----------------------------------------------------------------------------
# (b) expressibility
# ----------------------------------------------------------------------------
def span_test_ramanujan(M, P):
    """Exact test of D_P in span{K_m : m in M} over Q.

    All functions involved are functions of gcd(n, L), L = lcm(M u {P}) (K_m depends on gcd(n, m), and
    [P | n] = [P | gcd(n, L)]).  So evaluating them at the divisors g of L (one representative per gcd
    class) loses no information: the rank over Q computed here is the exact rank of the functions on Z/L.
    """
    M = sorted(set(M))
    L = arith.lcm(list(M) + [P])
    rows_n = arith.divisors(L)
    A = [[arith.kernel(m, n) for m in M] for n in rows_n]
    d = [Fraction(1) if n % P == 0 else Fraction(0) for n in rows_n]
    r_bank = rank_q(A)
    r_aug = rank_q([row + [dv] for row, dv in zip(A, d)])
    divs_in = all(dd in M for dd in arith.divisors(P))
    return {
        "bank": M, "P": P, "L": L, "gcd_classes_used": len(rows_n),
        "rank_bank": r_bank, "rank_with_D_P": r_aug,
        "bank_linearly_independent": r_bank == len(M),
        "D_P_in_span": r_aug == r_bank,
        "all_divisors_of_P_in_bank": divs_in,
        "iff_holds": (r_aug == r_bank) == divs_in,
    }


BANKS = {
    "default {1..16}": list(range(1, 17)),
    "B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}": [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16],
    "B16 as in MATH.md, order 1 excluded (10 orders)": [2, 3, 4, 5, 7, 8, 9, 11, 13, 16],
    "divisors of 12": [1, 2, 3, 4, 6, 12],
}


def check_b_ramanujan(pmax=40):
    specific = []
    for name, P in [("default {1..16}", 12), ("default {1..16}", 13), ("default {1..16}", 16),
                    ("default {1..16}", 17), ("default {1..16}", 24),
                    ("B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}", 12),
                    ("B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}", 13),
                    ("B16 as in MATH.md, order 1 excluded (10 orders)", 13),
                    ("divisors of 12", 12)]:
        r = span_test_ramanujan(BANKS[name], P)
        r["bank_name"] = name
        specific.append(r)
    mismatches = []
    nonindep = []
    checked = 0
    countable = {}
    for name, M in BANKS.items():
        countable[name] = []
        for P in range(2, pmax + 1):
            r = span_test_ramanujan(M, P)
            checked += 1
            if r["D_P_in_span"]:
                countable[name].append(P)
            if not r["iff_holds"]:
                mismatches.append((name, P))
            if not r["bank_linearly_independent"]:
                nonindep.append((name, P))
    coeffs = {}
    for P in (12, 13):
        coeffs[P] = {d: Fraction(arith.euler_phi(d), P) for d in arith.divisors(P)}
    # selectable combs (T4): lcm of a nonempty subset of the bank, up to pmax
    b16 = BANKS["B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}"]
    sel = set(b16)
    grew = True
    while grew:
        grew = False
        for x in list(sel):
            for y in b16:
                z = arith.lcm([x, y])
                if z <= pmax and z not in sel:
                    sel.add(z)
                    grew = True
    sel = sorted(x for x in sel if x >= 2)
    prime_powers = [q for q in range(2, 17) if len({f for f in range(2, q + 1) if q % f == 0 and
                                                   all(f % g for g in range(2, f))}) == 1]
    return {
        "label": "EXACT (rank over Q on gcd-class representatives)",
        "periods_with_D_P_in_span_(P in 2..pmax)": countable,
        "B16+1_selectable_periods_T4_(lcm of subsets,<=pmax)": sel,
        "B16+1_selectable_but_not_countable": [x for x in sel if x not in countable[
            "B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}"]],
        "specific_cases": specific,
        "sweep": {"banks": list(BANKS), "P_range": [2, pmax], "tests": checked,
                  "iff_mismatches": mismatches, "dependent_bank_cases": nonindep},
        "D_P_coefficients_in_K_basis": coeffs,
        "ok": (not mismatches) and (not nonindep)
              and specific[0]["D_P_in_span"] and specific[1]["D_P_in_span"]
              and (not specific[3]["D_P_in_span"]) and (not specific[4]["D_P_in_span"])
              and (not specific[5]["D_P_in_span"]) and specific[6]["D_P_in_span"]
              and (not specific[7]["D_P_in_span"])
              and countable["default {1..16}"] == list(range(2, 17))
              and countable["B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}"] == prime_powers,
    }


# --- cosine banks: exact Fourier supports (frequencies are reduced fractions mod 1) -------------
def cos_fourier(d, a=1):
    """Fourier coefficients of cos(2 pi a n / d) as {frequency in [0,1): coefficient}."""
    f = Fraction(a, d) % 1
    if f == 0:
        return {Fraction(0): Fraction(1)}
    if f == Fraction(1, 2):
        return {f: Fraction(1)}
    return {f: Fraction(1, 2), 1 - f: Fraction(1, 2)}


def d_fourier(P):
    return {Fraction(r, P): Fraction(1, P) for r in range(P)}


def fourier_matrix(funcs):
    keys = sorted(set().union(*[set(f) for f in funcs]))
    return [[f.get(k, Fraction(0)) for f in funcs] for k in keys]


def fourier_in_span(target, gens):
    r1 = rank_q(fourier_matrix(gens))
    r2 = rank_q(fourier_matrix(gens + [target]))
    return r2 == r1, r1, r2


def check_b_cosine(pmax=40, big_bank=48):
    res = {"label": "EXACT (rank over Q of exact Fourier-coefficient vectors; characters are independent)"}
    # prime P >= 5: not in span{1, cos(2 pi n/P)}
    primes = [p for p in range(5, 32) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    prime_rows = []
    for P in primes:
        gens = [cos_fourier(1), cos_fourier(P)]
        inn, r1, r2 = fourier_in_span(d_fourier(P), gens)
        prime_rows.append({"P": P, "in_span": inn, "rank_bank": r1, "rank_with_D_P": r2})
    res["prime_P_ge_5_not_in_span_{1,cos(2 pi n/P)}"] = prime_rows
    small = {}
    for P in (2, 3):
        gens = [cos_fourier(1)] + [cos_fourier(d) for d in arith.divisors(P) if d > 1]
        small[P] = fourier_in_span(d_fourier(P), gens)[0]
    res["P=2,3_in_span_of_matched_cosines"] = small

    # P = 12, harmonics a = 0..6
    gens12 = [cos_fourier(d) for d in arith.divisors(12)]
    cos_coeff = {0: Fraction(1, 12), 6: Fraction(1, 12)}
    for a in range(1, 6):
        cos_coeff[a] = Fraction(1, 6)
    recon = {}
    for a, c in cos_coeff.items():
        for k, v in cos_fourier(12, a).items():
            recon[k] = recon.get(k, Fraction(0)) + c * v
    harm = []
    for a in range(0, 7):
        inn, _, _ = fourier_in_span(cos_fourier(12, a), gens12)
        harm.append({"a": a, "coefficient_of_cos(2 pi a n/12)_in_D_12": cos_coeff[a],
                     "in_span_of_matched_cosines_d|12": inn,
                     "reduced_denominator_of_a/12": 12 // math.gcd(a, 12) if a else 1})
    full_inn, _, _ = fourier_in_span(d_fourier(12), [cos_fourier(12, a) for a in range(0, 7)])
    d12_inn, r1, r2 = fourier_in_span(d_fourier(12), gens12)
    res["P=12"] = {
        "cosine_expansion_of_D_12_matches_exact_fourier": recon == d_fourier(12),
        "harmonics": harm,
        "D_12_in_span_matched_cosines": d12_inn, "rank_bank": r1, "rank_with_D_12": r2,
        "D_12_in_span_of_all_harmonics_0..6": full_inn,
        "missing_harmonics": [h["a"] for h in harm if not h["in_span_of_matched_cosines_d|12"]],
    }

    # general criterion: D_P in span{cos(2 pi n/d): d in S} iff every divisor d>1 of P has phi(d)<=2 and lies in S
    crit_bad = []
    crit_bad_big = []
    big = [cos_fourier(1)] + [cos_fourier(d) for d in range(2, big_bank + 1)]
    for P in range(2, pmax + 1):
        gens = [cos_fourier(1)] + [cos_fourier(d) for d in arith.divisors(P) if d > 1]
        inn = fourier_in_span(d_fourier(P), gens)[0]
        pred = all(arith.euler_phi(d) <= 2 for d in arith.divisors(P))
        if inn != pred:
            crit_bad.append(P)
        inn_big = fourier_in_span(d_fourier(P), big)[0]
        if inn_big != pred:
            crit_bad_big.append(P)
    res["criterion_sweep"] = {
        "P_range": [2, pmax],
        "claim": "in span iff all divisors d of P satisfy phi(d)<=2, i.e. P in {2,3,4,6}",
        "mismatches_with_matched_cosines": crit_bad,
        "mismatches_with_all_cosines_d<=%d" % big_bank: crit_bad_big,
        "P_values_in_span": [P for P in range(2, pmax + 1)
                             if all(arith.euler_phi(d) <= 2 for d in arith.divisors(P))],
    }
    res["ok"] = (all(not r["in_span"] for r in prime_rows) and small[2] and small[3]
                 and res["P=12"]["cosine_expansion_of_D_12_matches_exact_fourier"]
                 and (not d12_inn) and full_inn and res["P=12"]["missing_harmonics"] == [5]
                 and not crit_bad and not crit_bad_big
                 and res["criterion_sweep"]["P_values_in_span"] == [2, 3, 4, 6])
    return res


# --- evidence slopes (limit of l_i differences / i) ----------------------------------------------
def pi_table(motif, V, p, nu):
    return [[p if motif[r] == a else nu for a in range(V)] for r in range(len(motif))]


def slope(kP, W, pit, t, c, b):
    """lim (l_i(c) - l_i(b)) / i along i = t mod P:  (1/P) sum_s kP(s) sum_a' (W[c][a'] - W[b][a']) pi_{t-s}(a')."""
    P = len(pit)
    V = len(pit[0])
    tot = Fraction(0)
    for s in range(P):
        r = (t - s) % P
        for ap in range(V):
            tot += kP[s] * (W[c][ap] - W[b][ap]) * pit[r][ap]
    return tot / P


def check_b_residual_counterexample():
    V = 16
    eps = Fraction(1, 2)
    p, nu, rho = noise_params(V, eps)
    W = identity(V)
    out = {"label": "EXACT", "V": V, "eps": eps, "p": p, "nu": nu, "rho": rho, "p-nu": p - nu}
    cases = []
    for P, delta in ((3, Fraction(3, 4)), (13, Fraction(1, 10)), (13, Fraction(1, 12)),
                     (13, Fraction(1, 13)), (13, Fraction(0))):
        motif = [0] + [1] * (P - 1)  # symbol 0 at residue 0 (query residue), symbol 1 elsewhere
        kP = [Fraction(1)] + [delta] * (P - 1)  # kappa = (1-delta) D_P + delta K_1
        pit = pi_table(motif, V, p, nu)
        sl = slope(kP, W, pit, 0, 0, 1)
        # direct exact expectation E[E_i(0)] - E[E_i(1)] at i = P*K, K = 100
        K = 100
        i = P * K
        direct = Fraction(0)
        for j in range(i + 1):
            kap = Fraction(1) if (i - j) % P == 0 else delta
            direct += kap * (pit[j % P][0] - pit[j % P][1])
        closed = (p - nu) * ((K + 1) - (P - 1) * delta * K)
        cases.append({
            "P": P, "delta": delta, "motif": motif, "kernel_P_fold": kP,
            "slope_without_log_rho (W=I)": sl,
            "slope_closed_form (p-nu)/P*(1-(P-1)delta)": (p - nu) / P * (1 - (P - 1) * delta),
            "negative": sl < 0,
            "exact_E[E_i(true)-E_i(other)]_at_i=%d" % i: direct,
            "closed_form": closed,
            "direct_equals_closed_form": direct == closed,
        })
    out["cases"] = cases
    out["threshold"] = "slope < 0 iff delta > 1/(P-1) for this motif"
    out["ok"] = (cases[0]["slope_without_log_rho (W=I)"] == Fraction(-1, 12)
                 and cases[1]["slope_without_log_rho (W=I)"] == Fraction(-1, 130)
                 and cases[1]["exact_E[E_i(true)-E_i(other)]_at_i=1300"] == Fraction(-19, 2)
                 and cases[2]["slope_without_log_rho (W=I)"] == 0  # boundary delta = 1/(P-1)
                 and cases[3]["slope_without_log_rho (W=I)"] > 0
                 and cases[4]["slope_without_log_rho (W=I)"] == Fraction(1, 26)
                 and all(c["direct_equals_closed_form"] for c in cases)
                 and all(c["slope_without_log_rho (W=I)"] == c["slope_closed_form (p-nu)/P*(1-(P-1)delta)"]
                         for c in cases))
    return out



def check_b_bank_design():
    """CRT comb selection (T4) versus exact counting (needs D_P): bank B16 + order 1, P = 12.

    S = {3, 4} has lcm 12, so (K_3 + K_4)/2 equals 1 exactly on 12Z (T4), but off the comb it is not zero:
    in an unnormalized sum those residues vote.  We list the values and the worst-case W = I slope over motifs.
    """
    P = 12
    bank = BANKS["B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}"]
    missing = [d for d in arith.divisors(P) if d not in bank]
    b = [(arith.kernel(3, n) + arith.kernel(4, n)) / 2 for n in range(P)]
    t4_ok = all((b[n] == 1) == (n % P == 0) for n in range(P))
    off_abs = sum((abs(x) for x in b[1:]), Fraction(0))
    V = 3
    p, nu, _ = noise_params(V, Fraction(1, 2))
    W = identity(V)
    worst = None
    for motif in product(range(V), repeat=P):
        pit = pi_table(motif, V, p, nu)
        for bad in range(V):
            if bad == motif[0]:
                continue
            sl = slope(b, W, pit, 0, motif[0], bad)
            if worst is None or sl < worst[0]:
                worst = (sl, list(motif), bad)
        if worst[0] == (p - nu) / P * (1 - off_abs):  # minimum attained: stop early
            break
    pred = (p - nu) / P * (b[0] - off_abs)
    return {"label": "EXACT", "P": P, "bank": bank, "divisors_of_P_missing_from_bank": missing,
            "b=(K_3+K_4)/2_on_residues_0..11": b, "T4_comb_selects_12Z": t4_ok,
            "sum_abs_offcomb": off_abs,
            "worst_motif_slope_V3 (query residue 0, W=I)": worst[0], "worst_motif": worst[1], "worst_wrong_symbol": worst[2],
            "predicted_(p-nu)/P*(b(0)-sum|b(s)|)": pred,
            "ok": missing == [6, 12] and t4_ok and off_abs == 4 and worst[0] == pred and pred < 0}


def check_b_harmonic_criterion(pmax=60):
    """For 1 <= a <= P/2, cos(2 pi a n/P) equals a fundamental cosine cos(2 pi n/d) of some order d iff a | P."""
    bad = []
    for P in range(2, pmax + 1):
        for a in range(1, P // 2 + 1):
            target = cos_fourier(P, a)
            is_fund = any(cos_fourier(d, 1) == target for d in range(1, 2 * P + 2))
            if is_fund != (P % a == 0):
                bad.append((P, a))
    # D_P needs every a in 1..floor(P/2); all divide P iff phi(P) <= 2
    ok_set = [P for P in range(2, pmax + 1) if all(P % a == 0 for a in range(1, P // 2 + 1))]
    return {"label": "EXACT", "P_range": [2, pmax], "failures": bad,
            "P_with_all_harmonics_fundamental": ok_set,
            "ok": (not bad) and ok_set == [2, 3, 4, 6]}


def check_c_two_order_family(cases_PV=((5, 3), (7, 2))):
    """kappa = beta1 K_1 + beta_P K_P (P prime, beta >= 0), W = I: all slopes positive for every motif iff
    beta1 > 0 and beta_P > (P-2) beta1 / 2 (strict dominance kP(0) > sum |kP(s)|)."""
    bad = []
    cases = 0
    for P, V in cases_PV:
        p, nu, _ = noise_params(V, Fraction(1, 2))
        W = identity(V)
        motifs = list(product(range(V), repeat=P))
        pits = [pi_table(m, V, p, nu) for m in motifs]
        for b1 in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
            for bP in (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3)):
                kP = [b1 + bP] + [b1 - bP / (P - 1)] * (P - 1)
                allpos = all(slope(kP, W, pit, t, mot[t], bb) > 0
                             for mot, pit in zip(motifs, pits) for t in range(P) for bb in range(V) if bb != mot[t])
                pred = b1 > 0 and bP > Fraction(P - 2, 2) * b1
                cases += 1
                if allpos != pred:
                    bad.append((P, str(b1), str(bP)))
    return {"label": "EXACT", "cases_(P,V)": [list(x) for x in cases_PV], "cases": cases, "mismatches": bad,
            "ok": not bad}


# ----------------------------------------------------------------------------
# (c) positive rate for exactly periodic kernels
# ----------------------------------------------------------------------------
def kernel_mix(weights):
    items = list(weights.items())

    def f(n):
        return sum((w * arith.kernel(m, n) for m, w in items), Fraction(0))
    return f


def fold(kfun, L, P):
    """kappa_P(s) = mean of kappa over the residue class s mod P (kappa is L-periodic)."""
    lam = arith.lcm([L, P])
    sums = [Fraction(0)] * P
    for n in range(lam):
        sums[n % P] += kfun(n)
    scale = Fraction(P, lam)
    return [x * scale for x in sums]


def check_c_fold_projection(pmax=24, mmax=16):
    bad = []
    for P in range(2, pmax + 1):
        for m in range(1, mmax + 1):
            kp = fold(lambda n, m=m: arith.kernel(m, n), m, P)
            expect = [arith.kernel(m, s) for s in range(P)] if P % m == 0 else [Fraction(0)] * P
            if kp != expect:
                bad.append((P, m))
    # D_P folded at P is the delta; D_12 folded at modulus 24 and 6 (consistency of the projection statement)
    d12 = {d: Fraction(arith.euler_phi(d), 12) for d in arith.divisors(12)}
    fold12 = fold(kernel_mix(d12), 12, 12)
    ok_delta = fold12 == [Fraction(1)] + [Fraction(0)] * 11
    fold_d12_mod24 = fold(kernel_mix(d12), 12, 24)  # period 12 divides 24: unchanged in Z/24
    fold_d12_mod5 = fold(kernel_mix(d12), 12, 5)    # orders dividing 5 among {1,2,3,4,6,12}: only m=1
    exp_mod5 = [Fraction(arith.euler_phi(1), 12)] * 5
    return {
        "label": "EXACT",
        "statement": "folded kernel of K_m at modulus P is K_m if m | P and 0 otherwise",
        "P_range": [2, pmax], "m_range": [1, mmax],
        "failures": bad,
        "D_12_folded_at_12_is_delta": ok_delta,
        "D_12_folded_at_24_equals_D_12_on_Z/24": fold_d12_mod24 == [d_p_value(12, s) for s in range(24)],
        "D_12_folded_at_5_is_constant_1/12": fold_d12_mod5 == exp_mod5,
        "ok": (not bad) and ok_delta and fold_d12_mod24 == [d_p_value(12, s) for s in range(24)]
              and fold_d12_mod5 == exp_mod5,
    }


def check_c_decomposition():
    """E[E_i(a)] = (i - t) e_t(a) + partial(t) exactly, for i = t + Lambda k, kappa a mixture of Ramanujan kernels."""
    P = 12
    weights = {5: Fraction(1), 8: Fraction(2), 12: Fraction(3), 1: Fraction(1, 2), 4: Fraction(-1)}
    kf0 = kernel_mix(weights)
    L = arith.lcm(list(weights))
    lam = arith.lcm([L, P])
    ktab = [kf0(n) for n in range(lam)]  # kappa is L-periodic, hence Lambda-periodic

    def kf(n):
        return ktab[n % lam]
    kP = fold(kf, L, P)
    expected_fold = [sum((w * arith.kernel(m, s) for m, w in weights.items() if P % m == 0), Fraction(0))
                     for s in range(P)]
    V = 4
    p, nu, _ = noise_params(V, Fraction(1, 2))
    motif = [(3 * r + 1) % V for r in range(P)]
    pit = pi_table(motif, V, p, nu)
    bad = []
    checked = 0
    for t in range(0, lam, 7):
        for k in range(0, 4):
            i = t + lam * k
            for a in range(V):
                e_t = sum((kP[s] * pit[(t - s) % P][a] for s in range(P)), Fraction(0)) / P
                det = sum((kf(i - j) * pit[j % P][a] for j in range(i + 1)), Fraction(0))
                partial = sum((kf(n) * pit[(t - n) % P][a] for n in range(t + 1)), Fraction(0))
                checked += 1
                if det != (i - t) * e_t + partial:
                    bad.append((t, k, a))
    return {"label": "EXACT", "P": P, "orders": sorted(weights), "L": L, "Lambda": lam,
            "fold_equals_projection_on_orders_dividing_P": kP == expected_fold,
            "identity_checks": checked, "failures": bad,
            "ok": (not bad) and kP == expected_fold}


def check_c_dominance(P_list=(3, 4), V=3, trials=120, seed=SEED):
    """For W = I: all slopes positive for every motif, every t, every wrong b  <=>  kP(0) > sum_{s != 0} |kP(s)|."""
    rng = random.Random(seed)
    p, nu, _ = noise_params(V, Fraction(1, 2))
    W = identity(V)
    mismatches = []
    cases = 0
    n_dominant = 0
    for P in P_list:
        motifs = list(product(range(V), repeat=P))
        pits = [pi_table(m, V, p, nu) for m in motifs]
        for trial in range(trials):
            kP = [Fraction(rng.randint(-8, 8), 8) for _ in range(P)]
            if trial % 2 == 1:
                kP[0] = sum(abs(x) for x in kP[1:]) + Fraction(rng.randint(-1, 2), 8)
            positive_all = True
            for mot, pit in zip(motifs, pits):
                for t in range(P):
                    c = mot[t]
                    for b in range(V):
                        if b != c and slope(kP, W, pit, t, c, b) <= 0:
                            positive_all = False
            dominant = kP[0] > sum(abs(x) for x in kP[1:])
            n_dominant += 1 if dominant else 0
            cases += 1
            if positive_all != dominant:
                mismatches.append((P, [str(x) for x in kP]))
    return {"label": "EXACT", "P_list": list(P_list), "V": V, "cases": cases,
            "cases_strictly_dominant": n_dominant, "mismatches": mismatches,
            "ok": (not mismatches) and n_dominant > 0 and n_dominant < cases}


def check_c_zero_mean_and_comb(seed=SEED):
    """Zero-mean folded kernels (simplex comb f*_P, K_P) have zero slope for constant motifs; D_P does not."""
    rng = random.Random(seed)
    V = 5
    P = 13
    p, nu, _ = noise_params(V, Fraction(1, 2))
    fstar = [Fraction(P * (1 if s == 0 else 0) - 1, P - 1) for s in range(P)]
    kP13 = fold(lambda n: arith.kernel(13, n), 13, 13)
    delta = [Fraction(1)] + [Fraction(0)] * (P - 1)
    bad = []
    for sym in range(V):
        motif = [sym] * P
        pit = pi_table(motif, V, p, nu)
        for _ in range(5):
            W = [[Fraction(rng.randint(-3, 3)) for _ in range(V)] for _ in range(V)]
            for t in range(P):
                for c in range(V):
                    for b in range(V):
                        for name, kp in (("f_star", fstar), ("K_13", kP13)):
                            if slope(kp, W, pit, t, c, b) != 0:
                                bad.append((name, sym, t, c, b))
    # D_P with W = I and a constant motif: positive slope (p-nu)/P for the true symbol over every wrong one
    motif = [2] * P
    pit = pi_table(motif, V, p, nu)
    comb_slopes = {slope(delta, identity(V), pit, t, 2, b) for t in range(P) for b in range(V) if b != 2}
    # arbitrary motifs: D_P slope is (p-nu)/P regardless of the motif
    motif_indep = True
    for _ in range(20):
        motif = [rng.randrange(V) for _ in range(P)]
        pit = pi_table(motif, V, p, nu)
        for t in range(P):
            for b in range(V):
                if b != motif[t] and slope(delta, identity(V), pit, t, motif[t], b) != (p - nu) / P:
                    motif_indep = False
    return {"label": "EXACT", "zero_slope_failures_for_zero_mean_kernels": bad,
            "D_P_constant_motif_slopes": sorted(comb_slopes), "(p-nu)/P": (p - nu) / P,
            "D_P_slope_is_motif_independent": motif_indep,
            "ok": (not bad) and comb_slopes == {(p - nu) / P} and motif_indep}


# ----------------------------------------------------------------------------
# (d) non-resonant kernels: exact algebra + numerical illustration
# ----------------------------------------------------------------------------
def check_d_algebra(seed=SEED):
    """With folded kernel constant (= kbar), the slope matrix is independent of the query residue and the
    margins of two residues with different symbols are negatives of each other."""
    rng = random.Random(seed)
    V, P = 6, 5
    p, nu, _ = noise_params(V, Fraction(1, 2))
    kbar = Fraction(1, 2)
    kP = [kbar] * P
    bad = []
    tested = 0
    for _ in range(40):
        motif = [rng.randrange(V) for _ in range(P)]
        pit = pi_table(motif, V, p, nu)
        W = [[Fraction(rng.randint(-4, 4)) for _ in range(V)] for _ in range(V)]
        hist = [motif.count(a) for a in range(V)]
        ell = [sum((W[a][ap] * kbar * sum(pit[r][ap] for r in range(P)) / P for ap in range(V)), Fraction(0))
               for a in range(V)]
        for t in range(P):
            for b in range(V):
                if b == motif[t]:
                    continue
                s = slope(kP, W, pit, t, motif[t], b)
                tested += 1
                if s != ell[motif[t]] - ell[b]:
                    bad.append(("slope != ell(c)-ell(b)", t, b))
        for t1 in range(P):
            for t2 in range(P):
                c1, c2 = motif[t1], motif[t2]
                if c1 != c2:
                    s1 = slope(kP, W, pit, t1, c1, c2)
                    s2 = slope(kP, W, pit, t2, c2, c1)
                    if s1 + s2 != 0:
                        bad.append(("antisymmetry", t1, t2))
        Wi = identity(V)
        for a in range(V):
            for b in range(V):
                lhs = (sum((kbar * pit[r][a] for r in range(P)), Fraction(0)) / P
                       - sum((kbar * pit[r][b] for r in range(P)), Fraction(0)) / P)
                if lhs != kbar / P * (p - nu) * (hist[a] - hist[b]):
                    bad.append(("histogram formula", a, b))
        _ = Wi
    return {"label": "EXACT (algebra of the limit formulas; the limits themselves are proved in T9 L9.1/T9d)",
            "slopes_tested": tested, "failures": bad, "ok": not bad}


def simulate_tokens(N, P, V, eps, motif, seed):
    rng = random.Random(seed)
    xs = []
    keep = 1.0 - float(eps)
    for j in range(N):
        if rng.random() < keep:
            xs.append(motif[j % P])
        else:
            xs.append(rng.randrange(V))
    return xs


def check_d_numeric(seed=SEED, kmin=8, kmax=16):
    """NUMERICAL illustration (fixed seed): E_i(true)/i versus E_i(other)/i for several kernels, P = 13.

    Query residue 0 (i a multiple of 13), true symbol 0, 'other' = symbol 1 (the motif symbol at residue 1).
    """
    P, V = 13, 16
    eps = Fraction(1, 2)
    p, nu, _ = noise_params(V, eps)
    pf, nf = float(p), float(nu)
    motif = list(range(P))
    xs = simulate_tokens(2 ** kmax + 1, P, V, eps, motif, seed)
    two_pi = 2.0 * math.pi
    th_drift = two_pi / 13.0 * (1.0 + 1e-3)
    th_irr = two_pi * (math.sqrt(5.0) - 1.0) / 2.0
    th_near = two_pi / 13.0 * (1.0 + math.sqrt(2.0) * 1e-3)
    kernels = [
        ("D_13 (exact comb)", lambda n: 1.0 if n % 13 == 0 else 0.0,
         [1.0] + [0.0] * (P - 1)),
        ("(1+cos(2 pi n/13))/2  (exact period 13)", lambda n: 0.5 * (1.0 + math.cos(two_pi * (n % 13) / 13.0)),
         [0.5 * (1.0 + math.cos(two_pi * s / 13.0)) for s in range(P)]),
        ("(1+cos(n th))/2, th = 2pi/13*1.001=2pi*77/1000", lambda n: 0.5 * (1.0 + math.cos(n * th_drift)),
         [0.5] * P),
        ("(1+cos(n th))/2, th = 2pi/13*(1+sqrt2*1e-3) (irr.)",
         lambda n: 0.5 * (1.0 + math.cos(n * th_near)), [0.5] * P),
        ("(1+cos(n th))/2, th = 2pi*(sqrt5-1)/2  (irrational)", lambda n: 0.5 * (1.0 + math.cos(n * th_irr)),
         [0.5] * P),
    ]
    checkpoints = [13 * ((2 ** k) // 13) for k in range(kmin, kmax + 1)]
    rows = []
    for name, kf, kP in kernels:
        tab = [kf(n) for n in range(checkpoints[-1] + 1)]

        def limit(a):  # (1/P) sum_s kP(s) pi_{0-s}(a)
            return sum(kP[s] * (pf if motif[(-s) % P] == a else nf) for s in range(P)) / P
        lim_true, lim_other = limit(0), limit(1)
        for i in checkpoints:
            st = so = 0.0
            for j in range(i + 1):
                x = xs[j]
                if x == 0:
                    st += tab[i - j]
                elif x == 1:
                    so += tab[i - j]
            rows.append({"kernel": name, "i": i, "E_true/i": st / i, "E_other/i": so / i,
                         "gap/i": (st - so) / i, "limit_true": lim_true, "limit_other": lim_other,
                         "limit_gap": lim_true - lim_other})
    # theta_drift / 2 pi = 1.001/13 = 77/1000 exactly: that kernel is 1000-periodic in exact arithmetic, so T5's
    # hypothesis fails for it and it is a case of (c) (gcd(1000, 13) = 1: folded kernel at modulus 13 is constant).
    fold_dev = max(abs(sum(0.5 * (1.0 + math.cos(two_pi * 77 * n / 1000.0))
                           for n in range(s, 13000, 13)) / 1000.0 - 0.5) for s in range(13))
    return {"label": "NUMERICAL (simulation, fixed seed %d)" % seed, "P": P, "V": V, "eps": eps,
            "theta_drift_over_2pi_exact": "1.001/13 = 77/1000 (period 1000; gcd(1000,13)=1)",
            "drift_fold_at_13_max_abs_dev_from_1/2": fold_dev, "theta_near": th_near,
            "motif": "mu_r = r (13 distinct symbols), query residue 0, true symbol 0, other symbol 1",
            "theta_drift": th_drift, "theta_irr": th_irr, "checkpoints": checkpoints, "rows": rows,
            "note": "limit_* for the two cos(n th) kernels with th near 2pi/13 equal (kbar/P)(p+12 nu) = 0.5(p+12 nu)/13 "
                    "(identical for symbols 0 and 1, which occur once each in the motif). The 'drift' theta is exactly "
                    "2pi*77/1000, so it is a 1000-periodic kernel (case (c), constant fold at modulus 13), not a T5 "
                    "instance; 'near' (sqrt2 perturbation) and 'irrational' are genuine (d) instances. At finite i "
                    "all three are still partly comb-like (near-resonance window ~ 1/|theta-2pi/13| ~ 1e3 steps).",
            "ok": fold_dev < 1e-12}


# ----------------------------------------------------------------------------
# (e) dependence: each noisy observation copied k times
# ----------------------------------------------------------------------------
def state_vec(mc, wr, rho, k):
    """Head output softmax(k * log(rho) * M) as exact Fractions; index 0 = true symbol."""
    M = [mc] + list(wr)
    w = [rho ** (k * (x - mc)) for x in M]
    s = sum(w)
    return [x / s for x in w]


def dependence_case(V, eps, m, ks=(1, 2, 4)):
    p, nu, rho = noise_params(V, eps)
    lam = math.log(float(rho))
    chi = chi_value(V, p, nu)
    states = count_states(V, p, nu, m)
    total = sum(states.values())
    gap = sum((pr * max(0, max(wr) - mc) for (mc, wr), pr in states.items()), Fraction(0))
    pis = {key: state_vec(key[0], key[1], rho, 1) for key in states}
    e_prob1 = sum((pr * pis[key][0] for key, pr in states.items()), Fraction(0))
    brier1 = sum((pr * (sum(x * x for x in pis[key]) - 2 * pis[key][0] + 1) for key, pr in states.items()),
                 Fraction(0))
    nll1 = expected_nll(states, lam, 1)
    per_k = {}
    flags = {"power_identity": True, "brier_identity": True, "kl_identity": True,
             "lower_bound": True, "upper_bound": True}
    for k in ks:
        zs = {key: state_vec(key[0], key[1], rho, k) for key in states}
        for key, z in zs.items():
            pw = [x ** k for x in pis[key]]
            s = sum(pw)
            if any(zz != y / s for zz, y in zip(z, pw)):
                flags["power_identity"] = False
        e_prob = sum((pr * zs[key][0] for key, pr in states.items()), Fraction(0))
        brier = sum((pr * (sum(x * x for x in zs[key]) - 2 * zs[key][0] + 1) for key, pr in states.items()),
                    Fraction(0))
        l2 = sum((pr * sum((a - b) ** 2 for a, b in zip(zs[key], pis[key])) for key, pr in states.items()),
                 Fraction(0))
        if brier - brier1 != l2:
            flags["brier_identity"] = False
        nll = expected_nll(states, lam, k)
        ekl = 0.0
        for key, pr in states.items():
            kl = sum(float(a) * (math.log(float(a)) - math.log(float(b))) for a, b in zip(pis[key], zs[key]))
            ekl += float(pr) * kl
        if abs((nll - nll1) - ekl) > 1e-9:
            flags["kl_identity"] = False
        lower = k * lam * float(gap)
        upper = k * (V - 1) * chi ** m          # NLL_k <= k NLL_1 pointwise, then B1
        if lower > nll + 1e-12:
            flags["lower_bound"] = False
        if nll > upper + 1e-12 or nll > k * nll1 + 1e-12:
            flags["upper_bound"] = False
        per_k[k] = {
            "E_NLL_head": nll, "E_NLL_bayes": nll1, "excess_NLL": nll - nll1,
            "E_KL(pi||head)": ekl,
            "E_prob_true_head": e_prob, "E_prob_true_bayes": e_prob1,
            "E_Brier_head": brier, "E_Brier_bayes": brier1, "E_Brier_excess=E||head-pi||^2": brier - brier1,
            "lower_bound k*lam*E[(max wrong - M_c)^+]": lower,
            "upper_bound k(V-1)chi^m": upper, "k*E_NLL_bayes": k * nll1,
        }
    closed = None
    if m == 1:
        closed = {k: float(p) * math.log(1 + (V - 1) * float(rho) ** (-k))
                  + (V - 1) * float(nu) * math.log(float(rho) ** k + V - 1) for k in ks}
        closed_ok = all(abs(closed[k] - per_k[k]["E_NLL_head"]) < 1e-12 for k in ks)
    else:
        closed_ok = True
    return {"V": V, "eps": eps, "m": m, "p": p, "nu": nu, "rho": rho, "chi": chi,
            "total_probability": total, "E[(max wrong M - M_c)^+]": gap,
            "per_k": per_k, "m1_closed_form_E_NLL_head": closed, "m1_closed_form_ok": closed_ok,
            "flags": flags, "ok": total == 1 and all(flags.values()) and closed_ok}


def check_e_dependence(cases=((3, Fraction(1, 2), 6), (16, Fraction(1, 2), 4)), ks=(1, 2, 4)):
    out = []
    ok_all = True
    for V, eps, mmax in cases:
        for m in range(1, mmax + 1):
            r = dependence_case(V, eps, m, ks)
            out.append(r)
            ok_all = ok_all and r["ok"]
    # k -> infinity: E NLL_head / k -> lam * E[(max wrong - M_c)^+], error <= log(V)/k
    V, eps, m, k = 3, Fraction(1, 2), 3, 1000
    p, nu, rho = noise_params(V, eps)
    lam = math.log(float(rho))
    states = count_states(V, p, nu, m)
    gap = sum((pr * max(0, max(wr) - mc) for (mc, wr), pr in states.items()), Fraction(0))
    lim_gap = abs(expected_nll(states, lam, k) / k - lam * float(gap))
    lim_ok = lim_gap <= math.log(V) / k + 1e-9
    # headline exact numbers, V = 3, eps = 1/2, k = 2, m = 1 (hand-checkable)
    head = dependence_case(3, Fraction(1, 2), 1, (2,))
    p3, nu3, rho3 = noise_params(3, Fraction(1, 2))
    st1 = count_states(3, p3, nu3, 1)
    acc = sum((pr for (mc, wr), pr in st1.items() if mc > max(wr)), Fraction(0))
    maxp_head = sum((pr * max(state_vec(mc, wr, rho3, 2)) for (mc, wr), pr in st1.items()), Fraction(0))
    maxp_bayes = sum((pr * max(state_vec(mc, wr, rho3, 1)) for (mc, wr), pr in st1.items()), Fraction(0))
    headline = {
        "E_prob_true_head": head["per_k"][2]["E_prob_true_head"],
        "E_prob_true_bayes": head["per_k"][2]["E_prob_true_bayes"],
        "accuracy_of_argmax (both heads)": acc,
        "E_max_prob_head": maxp_head, "E_max_prob_bayes": maxp_bayes,
        "E_NLL_head_closed_form": "(2/3) log(9/8) + (1/3) log 18",
        "E_NLL_bayes_closed_form": "(2/3) log(3/2) + (1/3) log 6",
        "E_NLL_head": head["per_k"][2]["E_NLL_head"], "E_NLL_bayes": head["per_k"][2]["E_NLL_bayes"],
        "check_closed_forms": (abs(head["per_k"][2]["E_NLL_head"]
                                   - (2 / 3 * math.log(9 / 8) + 1 / 3 * math.log(18))) < 1e-12
                               and abs(head["per_k"][2]["E_NLL_bayes"]
                                       - (2 / 3 * math.log(3 / 2) + 1 / 3 * math.log(6))) < 1e-12),
    }
    ok_all = ok_all and lim_ok and headline["check_closed_forms"] \
        and headline["E_prob_true_head"] == Fraction(11, 18) and headline["E_prob_true_bayes"] == Fraction(1, 2) \
        and acc == Fraction(2, 3) and maxp_head == Fraction(8, 9) and maxp_bayes == Fraction(2, 3)
    excess_by_m = {}
    for case in out:
        for k, v in case["per_k"].items():
            if k > 1:
                excess_by_m.setdefault("V=%d,k=%d" % (case["V"], k), []).append(v["excess_NLL"])
    monotone = {key: all(x > y for x, y in zip(v, v[1:])) for key, v in excess_by_m.items()}
    return {"label": "NUMERICAL (probabilities exact Fractions, logarithms floating point); the inequalities illustrated are PROVED in T9e",
            "excess_NLL_by_m (m = 1, 2, ...)": excess_by_m, "excess_strictly_decreasing_in_m": monotone,
            "cases": out, "k_to_infinity": {"V": V, "m": m, "k": k, "abs_error": lim_gap,
                                              "bound log(V)/k": math.log(V) / k, "ok": lim_ok},
            "headline_V3_eps1/2_k2_m1": headline, "ok": ok_all}


# ----------------------------------------------------------------------------
# round 3 (after the audit): L9.1 per-component bound, L9.2 character version, span dominance,
# T9c.6 random motifs, T9a.5 exponent tightness, T9d finite-window fold
# ----------------------------------------------------------------------------
def l91_component_bound(components, P):
    """sum_c 2 |beta_c| G_c lcm(m_c, P) for components = [(beta, m, G)] (beta Fraction/float, G = sup|chi_c|)."""
    return sum((2 * abs(b) * G * arith.lcm([m, P]) for b, m, G in components), 0)


def check_l91_component_bound(seed=SEED, imax=460):
    """L9.1': |D_i(a) - (i+1) e_{i mod P}(a)| <= sum_c 2 |beta_c| G_c lcm(m_c, P)  (exact for K_m, float for cosines)."""
    rng = random.Random(seed)
    V = 4
    p, nu, _ = noise_params(V, Fraction(1, 2))
    out_cases = []
    bad = []
    orders_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16]
    for P in (12, 13):
        motif = [(3 * r + 1) % V for r in range(P)]
        pit = pi_table(motif, V, p, nu)
        for trial in range(6):
            orders = sorted(rng.sample(orders_pool, 4))
            weights = {m: Fraction(rng.randint(-5, 5), rng.randint(1, 4)) for m in orders}
            tabs = {m: [arith.kernel(m, n) for n in range(m)] for m in orders}

            def kf(n):
                return sum((w * tabs[m][n % m] for m, w in weights.items()), Fraction(0))
            kP = [sum((w * arith.kernel(m, s) for m, w in weights.items() if P % m == 0), Fraction(0))
                  for s in range(P)]
            ktab = [kf(n) for n in range(imax + 1)]
            comps = [(w, m, 1) for m, w in weights.items()]
            b_new = l91_component_bound(comps, P)
            Lam = arith.lcm(list(orders) + [P])
            b_old = 2 * Lam * sum((abs(w) for w in weights.values()), Fraction(0))
            worst = Fraction(0)
            for i in range(0, imax + 1, 7):
                t = i % P
                for a in range(V):
                    e = sum((kP[s] * pit[(t - s) % P][a] for s in range(P)), Fraction(0)) / P
                    d = sum((ktab[n] * pit[(i - n) % P][a] for n in range(i + 1)), Fraction(0))
                    dev = abs(d - (i + 1) * e)
                    worst = max(worst, dev)
            if worst > b_new:
                bad.append(("K_m", P, orders, str(worst), str(b_new)))
            out_cases.append({"P": P, "orders": orders, "max_dev": float(worst), "bound_per_component": float(b_new),
                              "bound_2LambdaG": float(b_old)})
    # cosine components (float)
    cos_cases = []
    for P in (12, 13):
        motif = [(3 * r + 1) % V for r in range(P)]
        pitf = [[float(x) for x in row] for row in pi_table(motif, V, p, nu)]
        for trial in range(4):
            comps = []
            for _ in range(3):
                m = rng.choice([5, 7, 8, 9, 12, 13, 16])
                a = rng.choice([x for x in range(1, m) if math.gcd(x, m) == 1])
                comps.append((rng.uniform(-2, 2), m, a))
            ktab = [sum(b * math.cos(2 * math.pi * a * n / m) for b, m, a in comps) for n in range(imax + 1)]
            # fold of cos(2 pi a n/m) to P is cos(2 pi a s/m) if m | P else 0 (character version of L9.2)
            kP = [sum(b * math.cos(2 * math.pi * a * s / m) for b, m, a in comps if P % m == 0) for s in range(P)]
            bound = l91_component_bound([(b, m, 1.0) for b, m, a in comps], P)
            worst = 0.0
            for i in range(0, imax + 1, 7):
                t = i % P
                for aa in range(V):
                    e = sum(kP[s] * pitf[(t - s) % P][aa] for s in range(P)) / P
                    d = sum(ktab[n] * pitf[(i - n) % P][aa] for n in range(i + 1))
                    worst = max(worst, abs(d - (i + 1) * e))
            if worst > bound + 1e-9:
                bad.append(("cos", P, comps, worst, bound))
            cos_cases.append({"P": P, "components_(beta,m,a)": comps, "max_dev": worst, "bound_per_component": bound})
    # numbers quoted in the draft: default bank {1..16}, unit weights, and the comb itself
    pm = Fraction(1, 2)  # p - nu for V = 16, eps = 1/2
    quoted = {}
    for P in (12, 13):
        for name, w in (("D_P", {m: Fraction(arith.euler_phi(m), P) for m in arith.divisors(P)}),
                        ("beta_m=1, m=1..16", {m: Fraction(1) for m in range(1, 17)})):
            comps = [(b, m, 1) for m, b in w.items()]
            B = l91_component_bound(comps, P)
            Lam = arith.lcm(list(w) + [P])
            old = 2 * Lam * sum((abs(b) for b in w.values()), Fraction(0))
            k0 = sum((b for m, b in w.items() if P % m == 0), Fraction(0))
            g = pm * k0 / P  # leading per-index gap of E_i(a)-E_i(b) for a comb-dominant fold, V=16, eps=1/2
            # informative: error of the gap (<= 2 B) at most half the leading term (i+1) g
            i_inf = math.ceil(float(4 * B / g)) - 1
            quoted["%s, P=%d" % (name, P)] = {"bound_per_component": float(B), "bound_2LambdaG": float(old),
                                              "kappa_P(0)": float(k0), "gap_slope_(p-nu)kP(0)/P": float(g),
                                              "i_informative (2B <= (i+1) g/2)": i_inf}
    return {"label": "EXACT for the K_m cases (Fractions); cosine cases NUMERICAL (floating point)",
            "K_m_cases": out_cases, "cos_cases": cos_cases, "quoted": quoted, "failures": bad, "ok": not bad}


def check_fold_character(pmax=20, mmax=24):
    """L9.2 character version (NUMERICAL, floating point): the fold of cos(2 pi a n/m), gcd(a,m)=1, to modulus P
    is cos(2 pi a s/m) if m | P and 0 otherwise."""
    bad = []
    count = 0
    maxdev = 0.0
    for P in range(1, pmax + 1):
        for m in range(1, mmax + 1):
            for a in ([0] if m == 1 else [x for x in range(1, m) if math.gcd(x, m) == 1]):
                lam = arith.lcm([m, P])
                sums = [0.0] * P
                for n in range(lam):
                    sums[n % P] += math.cos(2 * math.pi * a * n / m)
                for s in range(P):
                    got = sums[s] * P / lam
                    want = math.cos(2 * math.pi * a * s / m) if P % m == 0 else 0.0
                    maxdev = max(maxdev, abs(got - want))
                    count += 1
                    if abs(got - want) > 1e-9:
                        bad.append((P, m, a, s))
    return {"label": "NUMERICAL (floating point; the statement is PROVED in L9.2')", "P_range": [1, pmax],
            "m_range": [1, mmax], "values_compared": count, "max_abs_dev": maxdev, "failures": bad[:10],
            "ok": not bad}


def check_hoeffding_constant(imax=20000):
    """T9c.3: with deviation s_t i / 4, Hoeffding gives exp(-s^2 i^2 / (8 (i+1) R^2)), which is NOT bounded by the
    earlier exp(-s^2 i / (8 R^2)) but is bounded by exp(-s^2 i / (16 R^2)); exact comparison of the exponents."""
    old_wrong = 0
    for i in range(1, imax + 1):
        e = Fraction(i * i, 8 * (i + 1))  # coefficient of s^2/R^2 in the Hoeffding exponent
        if e >= Fraction(i, 8):
            old_wrong += 1  # would mean the earlier exponent was valid
        if e < Fraction(i, 16):
            return {"label": "EXACT", "ok": False, "failing_i": i}
    return {"label": "EXACT", "i_range": [1, imax], "i_where_earlier_exponent_was_valid": old_wrong,
            "statement": "i^2/(8(i+1)) < i/8 and >= i/16 for all i >= 1", "ok": old_wrong == 0}


def _solve(A, b, tol=0):
    """Gaussian elimination (square system); works for Fractions (tol = 0) and floats.  None if singular."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) <= tol:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]


def l1_min_on_plane(cols, tol=0):
    """min over beta in R^n with sum(beta) = 1 of sum_s |sum_d beta_d cols[d][s]|, by vertex enumeration.

    The objective is convex piecewise linear and the arrangement {f(s) = 0} is pointed when the columns are
    linearly independent (the check asserts the rank), so the minimum is attained at a vertex."""
    from itertools import combinations
    n, R = len(cols), len(cols[0])
    best = None
    for zero in combinations(range(R), n - 1):
        A = [[Fraction(1)] * n] + [[cols[d][s] for d in range(n)] for s in zero]
        sol = _solve(A, [Fraction(1)] + [Fraction(0)] * (n - 1), tol)
        if sol is None:
            continue
        val = sum((abs(sum(sol[d] * cols[d][s] for d in range(n))) for s in range(R)), 0)
        if best is None or val < best[0]:
            best = (val, sol)
    return best


def linf_min_on_plane(cols, tol=1e-9):
    """min tau with |sum_d beta_d cols[d][s]| <= tau for every s, sum(beta) = 1 (floating point vertex enumeration)."""
    from itertools import combinations
    n, R = len(cols), len(cols[0])
    cons = [(s, sg) for s in range(R) for sg in (1.0, -1.0)]
    best = None
    for act in combinations(cons, n):
        A = [[1.0] * n + [0.0]] + [[sg * cols[d][s] for d in range(n)] + [-1.0] for s, sg in act]
        sol = _solve(A, [1.0] + [0.0] * n, tol)
        if sol is None:
            continue
        beta, tau = sol[:n], sol[n]
        if tau < 0 or max(abs(sum(beta[d] * cols[d][s] for d in range(n))) for s in range(R)) > tau + 1e-9:
            continue
        if best is None or tau < best[0]:
            best = (tau, beta)
    return best


def cosine_fold_from_coefficients(P, ds, coef):
    return [sum(c * math.cos(2 * math.pi * s / d) for c, d in zip(coef, ds)) for s in range(P)]


def check_span_dominance():
    """T9b.2 strengthening: no kernel in span{K_d : d | 12, d in B16+{1}} = span{K_1,K_2,K_3,K_4} is strictly
    dominant at P = 12, and the best convex mixture of K_3, K_4 has off-comb mass 4 (EXACT); cosine spans (NUMERICAL)."""
    P = 12
    res = {"label": "EXACT for the Ramanujan spans; cosine spans NUMERICAL (floating point)"}
    # (1) Ramanujan span of the divisors of 12 that lie in B16+{1}: {1,2,3,4}
    cols = [[arith.kernel(d, s) for s in range(1, P)] for d in (1, 2, 3, 4)]
    rank = rank_q([[arith.kernel(d, s) for d in (1, 2, 3, 4)] for s in range(P)])  # rows s = 0..P-1
    val, sol = l1_min_on_plane(cols)
    res["span_K1_K2_K3_K4"] = {"rank": rank, "min_off_comb_mass_with_f(0)=1": val, "argmin_beta": sol,
                               "dominance_margin_best": 1 - val}
    # (2) full divisor span of 12: contains D_12 with off-comb mass 0
    cols6 = [[arith.kernel(d, s) for s in range(1, P)] for d in arith.divisors(P)]
    val6, sol6 = l1_min_on_plane(cols6)
    res["span_all_divisors_of_12"] = {"rank": rank_q([[arith.kernel(d, s) for d in arith.divisors(P)]
                                                      for s in range(P)]),
                                       "min_off_comb_mass": val6, "argmin_beta": sol6}
    # (3) mixtures a K_3 + (1-a) K_4: piecewise linear in a, check breakpoints and endpoints exactly
    k3 = [arith.kernel(3, s) for s in range(P)]
    k4 = [arith.kernel(4, s) for s in range(P)]
    cands = {Fraction(0), Fraction(1)}
    for s in range(1, P):
        if k3[s] != k4[s]:
            a = -k4[s] / (k3[s] - k4[s])
            if 0 <= a <= 1:
                cands.add(a)
    mass = {a: sum((abs(a * k3[s] + (1 - a) * k4[s]) for s in range(1, P)), Fraction(0)) for a in cands}
    amin = min(mass, key=lambda a: mass[a])
    res["mixtures_K3_K4"] = {"best_a": amin, "best_off_comb_mass": mass[amin]}
    # (4) cosine_fund span folds (character L9.2'): span{cos(2 pi s/d): d | P}; floating point
    cos_out = {}
    for PP in (12, 13):
        ds = arith.divisors(PP)
        cc = [[math.cos(2 * math.pi * s / d) for s in range(1, PP)] for d in ds]
        v, sl = l1_min_on_plane(cc, tol=1e-9)
        fold_vec = [sum(sl[j] * math.cos(2 * math.pi * s / d) for j, d in enumerate(ds)) for s in range(PP)]
        cos_out[PP] = {"orders": list(ds), "min_off_comb_mass_with_f(0)=1": v, "dominance_margin_best": 1 - v,
                       "argmin_coefficients": sl, "fold": fold_vec}
        tau, bl = linf_min_on_plane(cc, tol=1e-9)
        cos_out[PP]["Linf_optimal"] = {"min_max_abs_off_comb_with_f(0)=1": tau, "coefficients": bl,
                                       "fold": cosine_fold_from_coefficients(PP, ds, bl)}
    res["cosine_fund_span"] = cos_out
    # (5) exact rational sub-span (d in {1,2,3,4,6} at P = 12, all values rational)
    cr = [[Fraction(round(2 * math.cos(2 * math.pi * s / d)), 2) if d in (1, 2, 3, 4, 6) else None
           for s in range(1, P)] for d in (1, 2, 3, 4, 6)]
    # cos(2 pi s/d) for d in {1,2,3,4,6} takes values in {+-1, +-1/2, 0}: rounding 2cos to an integer is exact
    vr, slr = l1_min_on_plane(cr)
    res["cosine_span_rational_part_d_in_1_2_3_4_6"] = {"min_off_comb_mass": vr, "argmin": slr}
    res["ok"] = (rank == 4 and val == 2 and val6 == 0 and mass[amin] == 4 and amin == Fraction(1, 2)
                 and abs(cos_out[12]["min_off_comb_mass_with_f(0)=1"] - 1.0) < 1e-9
                 and abs(cos_out[13]["min_off_comb_mass_with_f(0)=1"] - 5.0739) < 1e-3
                 and vr == 1)
    return res


# --- T9c.6: random motifs ---------------------------------------------------------------------------
# Folds computed by the audit (audit/consistency_out.txt, "fold values kappa_P(s)", arith backbone, development
# pilot, one training seed); quoted to three decimals.  They are data here, nothing is re-trained.
AUDIT_FOLDS = {
    12: {
        "arith": [8.868, 0.007, 0.061, -0.056, -0.010, 0.007, 0.010, 0.007, -0.010, -0.056, 0.061, 0.007],
        "cosine_fund": [5.806, 0.802, -1.132, 0.033, 0.630, -0.680, 1.477, -0.680, 0.630, 0.033, -1.132, 0.802],
        "cosine_full": [9.759, -0.134, -0.023, -0.059, 0.078, 0.149, 0.186, 0.149, 0.078, -0.059, -0.023, -0.134],
        "cosine_learned (window fold N=8192)": [-0.195, -0.776, -0.510, -0.366, -0.459, 0.003, 0.490, 0.586,
                                                 0.529, 0.462, 0.280, -0.033],
        "uniform": [1.0] * 12,
    },
    13: {
        "arith": [6.437] + [0.010] * 12,
        "cosine_fund": [0.964, 0.910, 0.759, 0.546, 0.320, 0.132, 0.027, 0.027, 0.132, 0.320, 0.546, 0.759, 0.910],
        "cosine_full": [10.301, -0.265, -0.208, -0.061, 0.116, 0.232, 0.357, 0.357, 0.232, 0.116, -0.061, -0.208,
                        -0.265],
        "cosine_learned (window fold N=8192)": [0.036, 0.097, 0.088, 0.001, -0.101, -0.120, -0.042, 0.058, 0.085,
                                                 0.034, -0.037, -0.065, -0.034],
        "uniform": [1.0] * 13,
    },
}


def symbol_mass(kP, motif, V, t):
    """M_a(t) = sum_{r: mu_r = a} kP(t - r): the kP-weighted histogram of the motif seen from residue t."""
    P = len(kP)
    M = [0] * V
    for r in range(P):
        M[motif[r]] = M[motif[r]] + kP[(t - r) % P]
    return M


def residue_ok(kP, motif, V, t, tol=0):
    """True iff the true symbol mu_t is the strict argmax of M(t) (margin > tol against every other symbol)."""
    M = symbol_mass(kP, motif, V, t)
    c = motif[t]
    return all(M[c] - M[b] > tol for b in range(V) if b != c)


def prob_all_positive_enum(kP, V):
    """Exact Pr over uniform motifs that every residue has a positive margin (rational kP): V^P enumeration."""
    P = len(kP)
    good = 0
    for motif in product(range(V), repeat=P):
        if all(residue_ok(kP, motif, V, t) for t in range(P)):
            good += 1
    return Fraction(good, V ** P)


def pair_tail(kP, V, tol=0):
    """T = Pr(sum_{s != 0} |kP(s)| B_s >= kP(0) - tol), B_s iid Bernoulli(1/V): exact subset enumeration.

    Failure of residue t against competitor b requires this event (T9c.6(d)), so Pr(some residue fails) <= P (V-1) T."""
    P = len(kP)
    k0 = kP[0]
    if k0 <= 0:
        return 1
    w = [abs(x) for x in kP[1:]]
    n = len(w)
    q = Fraction(1, V) if all(isinstance(x, Fraction) for x in kP) else 1.0 / V
    wsum = [0] * (1 << n)
    cnt = [0] * (1 << n)
    total = 0
    for mask in range(1 << n):
        if mask:
            low = mask & -mask
            j = low.bit_length() - 1
            wsum[mask] = wsum[mask ^ low] + w[j]
            cnt[mask] = cnt[mask ^ low] + 1
        if wsum[mask] >= k0 - tol:
            total = total + q ** cnt[mask] * (1 - q) ** (n - cnt[mask])
    return total


def n_min(kP, tol=0):
    """Smallest number j of off-comb residues whose |kP| sum reaches kP(0) (None if even all of them fall short)."""
    w = sorted((abs(x) for x in kP[1:]), reverse=True)
    acc = 0
    for j, x in enumerate(w, start=1):
        acc = acc + x
        if acc >= kP[0] - tol:
            return j
    return None


def leakage_fold(P, delta):
    return [Fraction(1)] + [delta] * (P - 1)


def _partitions(n, maxpart=None, maxlen=None):
    if maxpart is None:
        maxpart = n
    if n == 0:
        yield ()
        return
    if maxlen == 0:
        return
    for first in range(min(n, maxpart), 0, -1):
        for rest in _partitions(n - first, first, None if maxlen is None else maxlen - 1):
            yield (first,) + rest


def prob_all_positive_leakage(P, V, delta):
    """Exact Pr(all residues positive) for kP = [1, delta, ..., delta] (= (1-delta) D_P + delta K_1), W = lambda I.

    M_a(t) = (1-delta) 1[a = mu_t] + delta h_a with h the symbol counts of the motif, so only the multiset of counts
    matters; sum over partitions of P into at most V parts."""
    from math import factorial
    good = Fraction(0)
    for part in _partitions(P, None, V):
        counts = list(part) + [0] * (V - len(part))
        ok = True
        for c in range(V):
            if counts[c] == 0:
                continue
            other_max = max(counts[b] for b in range(V) if b != c)
            if not (1 - delta) + delta * (counts[c] - other_max) > 0:
                ok = False
                break
        if not ok:
            continue
        mult = factorial(V)
        for v in set(counts):
            mult //= factorial(counts.count(v))
        motifs_per_comp = factorial(P)
        for x in part:
            motifs_per_comp //= factorial(x)
        good += mult * motifs_per_comp
    return good / Fraction(V ** P)


def wilson(k, n, z=1.959964):
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def mc_random_motifs(kP, V, N, seed, tol=1e-9):
    """NUMERICAL: fraction of uniform random motifs with some non-positive margin; fraction of positive residues."""
    rng = random.Random(seed)
    P = len(kP)
    fails = 0
    pos_res = 0
    for _ in range(N):
        motif = [rng.randrange(V) for _ in range(P)]
        bad = False
        for t in range(P):
            if residue_ok(kP, motif, V, t, tol):
                pos_res += 1
            else:
                bad = True
        if bad:
            fails += 1
    lo, hi = wilson(fails, N)
    return {"N": N, "motifs_with_a_nonpositive_margin": fails, "fail_fraction": fails / N,
            "wilson95": [lo, hi], "positive_residue_fraction": pos_res / (N * P)}


def check_c_random_motif(seed=SEED, n_mc=20000, n_mc_rare=100000):
    res = {"label": "EXACT for (a)-(d) and the enumerations; Monte Carlo rows NUMERICAL (fixed seed %d)" % seed}
    rng = random.Random(seed)
    # (a) characterization, exactly, against the slope formula: random rational kP, random integer W
    bad_a = []
    cases_a = 0
    for V, P in ((3, 4), (3, 5), (2, 6)):
        p, nu, _ = noise_params(V, Fraction(1, 2))
        for _ in range(3):
            kP = [Fraction(rng.randint(-6, 6), rng.randint(1, 4)) for _ in range(P)]
            sigma = sum(kP, Fraction(0))
            Wg = [[Fraction(rng.randint(-3, 3)) for _ in range(V)] for _ in range(V)]
            d_, o_ = Fraction(3), Fraction(1)
            Wd = [[d_ if i == j else o_ for j in range(V)] for i in range(V)]  # (d-o) I + o J, d > o
            for motif in product(range(V), repeat=P):
                pit = pi_table(motif, V, p, nu)
                for t in range(P):
                    M = symbol_mass(kP, motif, V, t)
                    c = motif[t]
                    for b in range(V):
                        if b == c:
                            continue
                        cases_a += 1
                        sg = slope(kP, Wg, pit, t, c, b)
                        WM = [sum(Wg[a][j] * M[j] for j in range(V)) for a in range(V)]
                        W1 = [sum(Wg[a][j] for j in range(V)) for a in range(V)]
                        pred = ((p - nu) * (WM[c] - WM[b]) + nu * sigma * (W1[c] - W1[b])) / P
                        if sg != pred:
                            bad_a.append(("general W formula", V, P))
                        sd = slope(kP, Wd, pit, t, c, b)
                        if sd != (p - nu) * (d_ - o_) * (M[c] - M[b]) / P:
                            bad_a.append(("(d-o)I+oJ", V, P))
    res["a_characterization"] = {"slope_formula_checks": cases_a, "failures": bad_a[:5]}
    # (b) exact probabilities
    exact_b = []
    ok_b = True
    for V, P in ((2, 5), (3, 5), (3, 6), (4, 5)):
        uni = prob_all_positive_enum([Fraction(1)] * P, V)
        dP = prob_all_positive_enum([Fraction(1)] + [Fraction(0)] * (P - 1), V)
        row = {"V": V, "P": P, "uniform_fold": uni, "uniform_closed_form_V^(1-P)": Fraction(1, V ** (P - 1)),
               "D_P_fold": dP}
        ok_b = ok_b and uni == Fraction(1, V ** (P - 1)) and dP == 1
        for delta in (Fraction(1, 5), Fraction(1, 3), Fraction(1, 2)):
            e = prob_all_positive_enum(leakage_fold(P, delta), V)
            q = prob_all_positive_leakage(P, V, delta)
            row["leak_delta=%s" % delta] = {"enumeration": e, "partition_method": q}
            ok_b = ok_b and e == q
        exact_b.append(row)
    leak = []
    for P in (12, 13):
        for delta in (Fraction(1, 20), Fraction(1, 10), Fraction(1, 5), Fraction(1, 3)):
            q = prob_all_positive_leakage(P, 16, delta)
            leak.append({"V": 16, "P": P, "delta": delta, "Pr_all_positive": q, "Pr_fail": 1 - q,
                         "Pr_fail_float": float(1 - q)})
    res["b_exact_small"] = exact_b
    res["b_exact_leakage_V16"] = leak
    # (c)/(d): n_min bound against exact failure probabilities (random rational kP with kP(0) > 0)
    bad_d = []
    rows_d = []
    for V, P in ((3, 5), (4, 5), (2, 7)):
        for _ in range(4):
            kP = [Fraction(rng.randint(2, 9), rng.randint(1, 3))] + \
                 [Fraction(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(P - 1)]
            fail = 1 - prob_all_positive_enum(kP, V)
            T = pair_tail(kP, V)
            bound = min(Fraction(1), P * (V - 1) * T)
            nm = n_min(kP)
            dominant = kP[0] > sum((abs(x) for x in kP[1:]), Fraction(0))
            rows_d.append({"V": V, "P": P, "n_min": nm, "exact_fail": float(fail), "bound": float(bound)})
            if fail > bound or (dominant and (fail != 0 or nm is not None)):
                bad_d.append((V, P, [str(x) for x in kP]))
    for P in (5, 6):
        for delta in (Fraction(1, 5), Fraction(1, 3)):
            kP = leakage_fold(P, delta)
            fail = 1 - prob_all_positive_enum(kP, 3)
            bound = min(Fraction(1), P * 2 * pair_tail(kP, 3))
            if fail > bound:
                bad_d.append(("leak", P, str(delta)))
    res["d_bound_vs_exact"] = {"rows": rows_d, "failures": bad_d}
    # MC (NUMERICAL): V = 16, P = 12, 13
    V = 16
    mc_rows = []
    ok_mc = True
    best_fold = {}
    for P in (12, 13):
        ds = arith.divisors(P)
        cc = [[math.cos(2 * math.pi * s / d) for s in range(1, P)] for d in ds]
        v, sl = l1_min_on_plane(cc, tol=1e-9)
        tau, bl = linf_min_on_plane(cc, tol=1e-9)
        best_fold[P] = (cosine_fold_from_coefficients(P, ds, sl), cosine_fold_from_coefficients(P, ds, bl))
    for P in (12, 13):
        arms = [("D_P (comb)", [1.0] + [0.0] * (P - 1)),
                ("cosine_fund span, L1-optimal fold", best_fold[P][0]),
                ("cosine_fund span, Linf-optimal fold", best_fold[P][1])]
        arms += [("pilot " + k, v) for k, v in AUDIT_FOLDS[P].items()]
        for idx, (name, kP) in enumerate(arms):
            rare = (name.startswith("pilot cosine_fund") and P == 12) or "Linf" in name
            N = n_mc_rare if rare else n_mc
            r = mc_random_motifs(kP, V, N, seed + 100 * P + idx)
            nm = n_min(kP, 1e-9)
            T = pair_tail(kP, V, 1e-9)
            ub = min(1.0, P * (V - 1) * float(T))
            dom = kP[0] - sum(abs(x) for x in kP[1:])
            r.update({"P": P, "arm": name, "kappa_P(0)": kP[0], "sum_abs_off_comb": sum(abs(x) for x in kP[1:]),
                      "dominance_margin": dom, "n_min": nm, "pair_tail_T": float(T), "union_bound_P(V-1)T": ub})
            mc_rows.append(r)
            ok_mc = ok_mc and r["wilson95"][0] <= ub + 1e-12
            if dom > 1e-9:
                ok_mc = ok_mc and r["motifs_with_a_nonpositive_margin"] == 0
    res["mc_V16"] = mc_rows
    res["ok"] = (not bad_a) and ok_b and (not bad_d) and ok_mc
    return res


# --- T9a.5: exponent of the expectation ---------------------------------------------------------------
def vote_walk_distribution(V, eps, n):
    """Exact law (Fractions) of S_n = sum of n iid steps +1 (prob nu), -1 (prob p), 0 (prob (V-2) nu)."""
    p, nu, _ = noise_params(V, eps)
    dist = {0: Fraction(1)}
    for _ in range(n):
        new = {}
        for s, pr in dist.items():
            new[s + 1] = new.get(s + 1, Fraction(0)) + pr * nu
            new[s - 1] = new.get(s - 1, Fraction(0)) + pr * p
            if V > 2:
                new[s] = new.get(s, Fraction(0)) + pr * (V - 2) * nu
        dist = new
    return dist


def check_a_tightness():
    """T9a.5: lim (1/n) log E NLL = log chi.  Lower bound Pr(S_n >= 0) >= (1/4) chi^n rho^{-sqrt(2 n s2)/2} with
    s2 = 2 sqrt(p nu)/chi (tilted measure is symmetric with variance s2); checked against exact distributions."""
    out = []
    ok = True
    for V, eps in ((16, Fraction(1, 2)), (3, Fraction(1, 2))):
        p, nu, rho = noise_params(V, eps)
        chi = chi_value(V, p, nu)
        s2 = 2.0 * math.sqrt(float(p * nu)) / chi
        theta = 0.5 * math.log(float(rho))
        rows = []
        # float DP for large n
        probs = {0: 1.0}
        pf, nf = float(p), float(nu)
        ns = {10, 25, 50, 100, 200, 400, 800}
        for n in range(1, 801):
            new = {}
            for s, pr in probs.items():
                new[s + 1] = new.get(s + 1, 0.0) + pr * nf
                new[s - 1] = new.get(s - 1, 0.0) + pr * pf
                if V > 2:
                    new[s] = new.get(s, 0.0) + pr * (V - 2) * nf
            probs = new
            tail = sum(pr for s, pr in probs.items() if s >= 0)
            lower = 0.25 * chi ** n * math.exp(-theta * math.sqrt(2.0 * n * s2))
            upper = chi ** n
            if not (lower <= tail <= upper * (1 + 1e-9)):
                ok = False
            if n in ns:
                rows.append({"n": n, "Pr(S_n>=0)": tail, "chi^n": upper, "lower_bound": lower,
                             "(1/n)log Pr - log chi": math.log(tail) / n - math.log(chi),
                             "sqrt(n) Pr / chi^n": math.sqrt(n) * tail / upper})
        # exact tilted identity for small n: Pr(S_n>=0) = chi^n E_Q[exp(-theta S_n); S_n >= 0]
        dist = vote_walk_distribution(V, eps, 30)
        qs = {s: float(pr) * math.exp(theta * s) / chi ** 30 for s, pr in dist.items()}
        ident = sum(pr * math.exp(-theta * s) for s, pr in qs.items() if s >= 0) * chi ** 30
        exact_tail = float(sum((pr for s, pr in dist.items() if s >= 0), Fraction(0)))
        tilt_total = sum(qs.values())
        tilt_var = sum(pr * s * s for s, pr in qs.items()) / 30
        ok = ok and abs(ident - exact_tail) < 1e-12 * max(1.0, exact_tail) and abs(tilt_total - 1) < 1e-12 \
            and abs(tilt_var - s2) < 1e-9
        # E NLL (b = 0) >= log 2 Pr(S_m >= 0), m = 1..6, from the exact count-state enumeration of T9a.3
        lam = math.log(float(rho))
        lb_ok = True
        for m in range(1, 7):
            states = count_states(V, p, nu, m)
            e = expected_nll(states, lam, 1)
            tl = float(sum((pr for s, pr in vote_walk_distribution(V, eps, m).items() if s >= 0), Fraction(0)))
            lb_ok = lb_ok and e >= math.log(2) * tl - 1e-12
        ok = ok and lb_ok
        out.append({"V": V, "eps": eps, "chi": chi, "tilted_variance_s2": s2, "theta_star=(1/2)log rho": theta,
                    "rows": rows, "tilt_identity_n=30": {"lhs": exact_tail, "rhs": ident},
                    "E_NLL_ge_log2_Pr(S_m>=0)_m=1..6": lb_ok})
    return {"label": "NUMERICAL (floating point checks of a PROVED chain; the n = 30 distribution is exact)",
            "cases": out, "ok": ok}


# --- T9d finite-window fold --------------------------------------------------------------------------
def window_fold_bruteforce(theta, P, N):
    sums = [0.0] * P
    for n in range(N):
        sums[n % P] += math.cos(n * theta)
    return [x * P / N for x in sums]


def window_fold_closed(a, eta, P, N):
    """kappa-hat_{P,N}(s) for kappa(n) = cos(n theta), theta = 2 pi a/P + eta, closed form (Dirichlet kernel)."""
    out = []
    x = P * eta
    for s in range(P):
        U = (N - s + P - 1) // P  # number of n < N with n = s mod P
        if U <= 0:
            out.append(0.0)
            continue
        if abs(math.sin(x / 2)) < 1e-14:
            ratio = float(U)
        else:
            ratio = math.sin(x * U / 2) / math.sin(x / 2)
        out.append(P / N * ratio * math.cos(2 * math.pi * a * s / P + eta * (s + P * (U - 1) / 2)))
    return out


def check_d_window_fold(seed=SEED):
    rng = random.Random(seed)
    worst = 0.0
    cases = 0
    for P in (5, 12, 13):
        for _ in range(30):
            a = rng.randrange(0, P)
            eta = rng.choice([1, -1]) * 10 ** rng.uniform(-5, -0.7)
            N = rng.randint(P, 3000)
            th = 2 * math.pi * a / P + eta
            bf = window_fold_bruteforce(th, P, N)
            cf = window_fold_closed(a, eta, P, N)
            worst = max(worst, max(abs(x - y) for x, y in zip(bf, cf)))
            cases += 1
    # envelope |D_U(P eta)| <= pi/(N |eta|) for N = P U, |P eta| <= pi
    env_ok = True
    for P in (5, 12, 13):
        for _ in range(200):
            eta = rng.choice([1, -1]) * 10 ** rng.uniform(-5, -1) / P * 3
            if abs(P * eta) > math.pi:
                continue
            U = rng.randint(1, 5000)
            N = P * U
            x = P * eta
            D = 1.0 if abs(math.sin(x / 2)) < 1e-14 else abs(math.sin(U * x / 2) / (U * math.sin(x / 2)))
            if D > math.pi / (N * abs(eta)) + 1e-12:
                env_ok = False
    # the table kernel of T9d-num: theta = 2 pi/13 (1 + sqrt2 1e-3), a = 1
    P = 13
    eta = 2 * math.pi / 13 * math.sqrt(2.0) * 1e-3
    envelope = []
    for N in (247, 1014, 4095, 16380, 65533):
        U = N // P
        x = P * eta
        envelope.append({"N": N, "|D_U(P eta)|": abs(math.sin(U * x / 2) / (U * math.sin(x / 2))),
                         "bound pi/(N eta)": math.pi / (N * eta)})
    return {"label": "NUMERICAL (floating point comparison of a PROVED closed form with brute force)",
            "cases": cases, "max_abs_dev_closed_vs_bruteforce": worst, "envelope_bound_holds": env_ok,
            "table_kernel_envelope_P13_a1": envelope, "ok": worst < 1e-9 and env_ok}


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def run_all():
    res = {}
    res["a_identity"] = check_a_identity()
    res["a_bayes_equality"] = check_a_bayes_equality()
    res["a_head_comb_counts"] = check_a_head_comb_counts()
    res["a_rate_bound"] = check_a_rate_bound()
    res["b_ramanujan_span"] = check_b_ramanujan()
    res["b_cosine_span"] = check_b_cosine()
    res["b_residual_counterexample"] = check_b_residual_counterexample()
    res["b_bank_design"] = check_b_bank_design()
    res["b_harmonic_criterion"] = check_b_harmonic_criterion()
    res["c_fold_projection"] = check_c_fold_projection()
    res["c_decomposition"] = check_c_decomposition()
    res["c_dominance"] = check_c_dominance()
    res["c_zero_mean_and_comb"] = check_c_zero_mean_and_comb()
    res["c_two_order_family"] = check_c_two_order_family()
    res["d_algebra"] = check_d_algebra()
    res["d_numeric"] = check_d_numeric()
    res["e_dependence"] = check_e_dependence()
    res["l91_component_bound"] = check_l91_component_bound()
    res["l92_character_fold"] = check_fold_character()
    res["c3_hoeffding_constant"] = check_hoeffding_constant()
    res["b_span_dominance"] = check_span_dominance()
    res["c_random_motif"] = check_c_random_motif()
    res["a_tightness"] = check_a_tightness()
    res["d_window_fold"] = check_d_window_fold()
    res["all_ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict) and "ok" in v)
    return res


def _print_summary(res):
    print("T9 checks (seed %d)" % SEED)
    for key, val in res.items():
        if isinstance(val, dict) and "ok" in val:
            print("  %-28s %-6s %s" % (key, "OK" if val["ok"] else "FAIL", val.get("label", "")))
    print()
    b = res["b_ramanujan_span"]["specific_cases"]
    print("(b) D_P in span{K_m}?  (exact)")
    for r in b:
        print("   %-52s P=%-3d in_span=%-5s divisors_in_bank=%s" % (
            r["bank_name"], r["P"], r["D_P_in_span"], r["all_divisors_of_P_in_bank"]))
    c = res["b_cosine_span"]
    print("(b) cosines: P=12 harmonics a=0..6 in span of matched cosines:",
          [(h["a"], h["in_span_of_matched_cosines_d|12"]) for h in c["P=12"]["harmonics"]])
    print("    D_12 in span:", c["P=12"]["D_12_in_span_matched_cosines"],
          "| prime P>=5 in span:", [(r["P"], r["in_span"]) for r in c["prime_P_ge_5_not_in_span_{1,cos(2 pi n/P)}"]])
    print("    P with D_P in span of matched cosines:", c["criterion_sweep"]["P_values_in_span"])
    r = res["b_residual_counterexample"]
    for cs in r["cases"]:
        print("(b) residual: P=%d delta=%s slope(W=I, /log rho)=%s negative=%s" % (
            cs["P"], cs["delta"], cs["slope_without_log_rho (W=I)"], cs["negative"]))
    rb = res["b_ramanujan_span"]
    print("(b) P in 2..40 with D_P in span, B16+{1}:", rb["periods_with_D_P_in_span_(P in 2..pmax)"][
        "B16 with order 1 {1,2,3,4,5,7,8,9,11,13,16}"])
    print("    selectable (T4) but not countable:", rb["B16+1_selectable_but_not_countable"])
    bd = res["b_bank_design"]
    print("(b) bank design: B16+{1}, P=12, missing divisors %s; b=(K_3+K_4)/2 on 0..11 = %s" % (
        bd["divisors_of_P_missing_from_bank"], [str(x) for x in bd["b=(K_3+K_4)/2_on_residues_0..11"]]))
    print("    sum|offcomb|=%s, worst-motif slope (V=3, W=I) = %s  (= (p-nu)/P*(b(0)-sum|b|))" % (
        bd["sum_abs_offcomb"], bd["worst_motif_slope_V3 (query residue 0, W=I)"]))
    print()
    print("(a) rate: a.s. slope per vote (p-nu)log(rho) vs expectation exponent -log(chi):")
    for case in res["a_rate_bound"]["cases"]:
        print("    V=%d: %.6f vs %.6f (chi=%.6f)" % (case["V"], case["as_slope_per_vote_(p-nu)log(rho)"],
                                                     case["expectation_exponent_-log(chi)"], case["chi"]))
    print()
    print("(d) NUMERICAL table (P=13, V=16, eps=1/2; query residue 0, true=0, other=1):")
    print("    %-52s %6s %10s %10s %10s | %9s %9s" % ("kernel", "i", "E_true/i", "E_other/i", "gap/i",
                                                     "lim_true", "lim_other"))
    for row in res["d_numeric"]["rows"]:
        if row["i"] in (res["d_numeric"]["checkpoints"][0], res["d_numeric"]["checkpoints"][2],
                        res["d_numeric"]["checkpoints"][4], res["d_numeric"]["checkpoints"][6],
                        res["d_numeric"]["checkpoints"][-1]):
            print("    %-52s %6d %10.5f %10.5f %10.5f | %9.5f %9.5f" % (
                row["kernel"][:52], row["i"], row["E_true/i"], row["E_other/i"], row["gap/i"],
                row["limit_true"], row["limit_other"]))
    print()
    print("(e) dependence, expected NLL (head = D_P count head, W = log(rho) I; copies k; m independent votes):")
    print("    %3s %3s %3s | %12s | %s" % ("V", "m", "k", "E NLL Bayes", "E NLL head (excess)"))
    for case in res["e_dependence"]["cases"]:
        for k, v in case["per_k"].items():
            if k == 1:
                continue
            print("    %3d %3d %3d | %12.6f | %12.6f (%+.6f)   E[p_true]: head=%s bayes=%s" % (
                case["V"], case["m"], k, v["E_NLL_bayes"], v["E_NLL_head"], v["excess_NLL"],
                v["E_prob_true_head"], v["E_prob_true_bayes"]))
    h = res["e_dependence"]["headline_V3_eps1/2_k2_m1"]
    print("    headline V=3, eps=1/2, k=2, m=1: E p_true head=%s, bayes=%s; E NLL head=%.9f, bayes=%.9f" % (
        h["E_prob_true_head"], h["E_prob_true_bayes"], h["E_NLL_head"], h["E_NLL_bayes"]))
    print()
    print("    excess strictly decreasing in m?", res["e_dependence"]["excess_strictly_decreasing_in_m"])
    print()
    l = res["l91_component_bound"]
    print("(L9.1') per-component bound versus 2*Lambda*G (V=16, eps=1/2; informative = 2B <= (i+1) g/2):")
    for name, q in l["quoted"].items():
        print("    %-28s B=%10.2f  2LG=%12.1f  kP(0)=%7.3f  g=%.5f  informative from i>=%d" % (
            name, q["bound_per_component"], q["bound_2LambdaG"], q["kappa_P(0)"], q["gap_slope_(p-nu)kP(0)/P"],
            q["i_informative (2B <= (i+1) g/2)"]))
    print("    random K_m cases: max |D_i-(i+1)e| / bound = %s" % ", ".join(
        "%.2f/%.0f" % (c["max_dev"], c["bound_per_component"]) for c in l["K_m_cases"][:6]))
    sd = res["b_span_dominance"]
    print("(T9b.2) span{K1..K4}, P=12: min off-comb mass %s (margin %s); K3/K4 mixtures: a=%s mass %s; "
          "cosine_fund spans: P=12 %.4f, P=13 %.4f" % (
              sd["span_K1_K2_K3_K4"]["min_off_comb_mass_with_f(0)=1"], sd["span_K1_K2_K3_K4"]["dominance_margin_best"],
              sd["mixtures_K3_K4"]["best_a"], sd["mixtures_K3_K4"]["best_off_comb_mass"],
              sd["cosine_fund_span"][12]["min_off_comb_mass_with_f(0)=1"],
              sd["cosine_fund_span"][13]["min_off_comb_mass_with_f(0)=1"]))
    print("        cosine_fund span, min over coefficients of max_{s!=0}|f(s)| with f(0)=1: P=12 %.5f, P=13 %.5f" % (
        sd["cosine_fund_span"][12]["Linf_optimal"]["min_max_abs_off_comb_with_f(0)=1"],
        sd["cosine_fund_span"][13]["Linf_optimal"]["min_max_abs_off_comb_with_f(0)=1"]))
    rm = res["c_random_motif"]
    print("(T9c.6) exact leakage family, V=16: Pr(some residue non-positive)")
    for r in rm["b_exact_leakage_V16"]:
        print("    P=%d delta=%-5s Pr_fail=%.3e" % (r["P"], r["delta"], r["Pr_fail_float"]))
    print("(T9c.6) Monte Carlo, V=16, uniform random motifs (NUMERICAL):")
    print("    %2s %-44s %7s %8s %6s %9s  %-21s %9s %9s" % ("P", "fold", "N", "fails", "n_min", "dom.marg",
                                                           "Wilson 95%", "T", "union"))
    for r in rm["mc_V16"]:
        print("    %2d %-44s %7d %8d %6s %9.3f  [%.2e, %.2e] %9.2e %9.2e" % (
            r["P"], r["arm"][:44], r["N"], r["motifs_with_a_nonpositive_margin"], r["n_min"], r["dominance_margin"],
            r["wilson95"][0], r["wilson95"][1], r["pair_tail_T"], r["union_bound_P(V-1)T"]))
    for case in res["a_tightness"]["cases"]:
        print("(T9a.5) V=%d chi=%.6f: " % (case["V"], case["chi"]) + "; ".join(
            "n=%d: (1/n)log Pr-log chi=%.4f, sqrt(n)Pr/chi^n=%.3f" % (
                r["n"], r["(1/n)log Pr - log chi"], r["sqrt(n) Pr / chi^n"]) for r in case["rows"]))
    wf = res["d_window_fold"]
    print("(T9d window) closed form vs brute force: max dev %.2e over %d cases; envelope P=13,a=1 (eta=%.2e): %s" % (
        wf["max_abs_dev_closed_vs_bruteforce"], wf["cases"], 2 * math.pi / 13 * math.sqrt(2.0) * 1e-3,
        ", ".join("N=%d: %.4f" % (e["N"], e["|D_U(P eta)|"]) for e in wf["table_kernel_envelope_P13_a1"])))
    print()
    print("ALL CHECKS OK:", res["all_ok"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None, help="write all results to this JSON file")
    args = ap.parse_args(argv)
    res = run_all()
    _print_summary(res)
    if args.json:
        Path(args.json).write_text(json.dumps(jsonable(res), indent=2, sort_keys=True))
        print("wrote", args.json)
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
