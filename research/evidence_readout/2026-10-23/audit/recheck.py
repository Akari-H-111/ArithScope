"""Independent re-derivation of selected T9 claims (referee script; stdlib only; own implementations).

Run:  /usr/bin/python3 -I audit/recheck.py        (about 1-2 minutes)
Nothing here imports check_t9.py. K_m is implemented from the closed form with its own Moebius/phi.
"""
import itertools
import math
import random
import sys
from fractions import Fraction
from math import gcd

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


# ---------------------------------------------------------------- arithmetic helpers (own)
def factor(n):
    f, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def mobius(n):
    f = factor(n)
    if any(e > 1 for e in f.values()):
        return 0
    return -1 if len(f) % 2 else 1


def phi(n):
    r = n
    for p in factor(n):
        r = r // p * (p - 1)
    return r


def K(m, n):
    q = m // gcd(m, n)          # gcd(m,0)=m -> q=1 -> K=1
    return Fraction(mobius(q), phi(q))


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def lcm(a, b):
    return a // gcd(a, b) * b


# ---------------------------------------------------------------- 1. D_P identity and fold lemma
def check_identity(pmax=40):
    for P in range(1, pmax + 1):
        for n in range(-3 * P, 3 * P + 1):
            v = sum(Fraction(phi(m), P) * K(m, n) for m in divisors(P))
            assert v == (1 if n % P == 0 else 0), (P, n)
    say("[A] D_P = sum_{m|P} phi(m)/P K_m = 1[P|n] exactly, P=1..%d, |n|<=3P: OK" % pmax)


def check_fold_lemma(pmax=30, mmax=24):
    # K_m folded to P: average of K_m(n) over n = s mod P (one full period of n mod lcm(m,P))
    bad = 0
    for P in range(1, pmax + 1):
        for m in range(1, mmax + 1):
            Lm = lcm(m, P)
            for s in range(P):
                vals = [K(m, n) for n in range(s, Lm + s, P)]
                f = sum(vals) / len(vals)
                want = K(m, s) if P % m == 0 else 0
                if f != want:
                    bad += 1
    say("[B] L9.2 (Ramanujan kernels): fold(K_m)=K_m if m|P else 0, P<=%d, m<=%d: mismatches=%d" % (pmax, mmax, bad))
    # character version (needed for cosine banks; NOT stated in the draft): cos(2 pi a n / m), gcd(a,m)=1
    bad = 0
    for P in range(1, 21):
        for m in range(1, 25):
            for a in range(1, m + 1):
                if gcd(a, m) != 1:
                    continue
                Lm = lcm(m, P)
                for s in range(P):
                    vals = [math.cos(2 * math.pi * ((a * n) % m) / m) for n in range(s, Lm + s, P)]
                    f = sum(vals) / len(vals)
                    want = math.cos(2 * math.pi * ((a * s) % m) / m) if P % m == 0 else 0.0
                    if abs(f - want) > 1e-12:
                        bad += 1
    say("[B'] character version: fold(cos(2 pi a n/m))=cos(2 pi a s/m) if m|P else 0, P<=20, m<=24: mismatches=%d" % bad)


# ---------------------------------------------------------------- 2. T9b.1 span criterion, own rank test
def rank_fractions(rows):
    M = [list(r) for r in rows]
    rk, ncol = 0, len(M[0]) if M else 0
    for c in range(ncol):
        piv = None
        for r in range(rk, len(M)):
            if M[r][c] != 0:
                piv = r
                break
        if piv is None:
            continue
        M[rk], M[piv] = M[piv], M[rk]
        for r in range(len(M)):
            if r != rk and M[r][c] != 0:
                f = M[r][c] / M[rk][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[rk])]
        rk += 1
    return rk


def check_span(trials=300, seed=1):
    rng = random.Random(seed)
    mism = 0
    both = [0, 0]
    for _ in range(trials):
        P = rng.randint(1, 14)
        pool = list(range(1, 19))
        if rng.random() < 0.6:   # bias toward banks that contain (or nearly contain) all divisors of P
            M = set(divisors(P)) | set(rng.sample(pool, rng.randint(0, 5)))
            if rng.random() < 0.5 and len(M) > 1:
                M.discard(rng.choice(sorted(M)))
            M = sorted(M)
        else:
            M = sorted(rng.sample(pool, rng.randint(1, 12)))
        L = 1
        for m in set(M) | {P}:
            L = lcm(L, m)
        if L > 4000:
            continue
        cols = lambda m: [K(m, n) for n in range(L)]
        Dp = [Fraction(1 if n % P == 0 else 0) for n in range(L)]
        gens = [cols(m) for m in M]
        r0 = rank_fractions(gens)
        r1 = rank_fractions(gens + [Dp])
        in_span = (r0 == r1)
        crit = all(d in M for d in divisors(P))
        assert r0 == len(M), "K_m dependent?"
        if in_span != crit:
            mism += 1
        both[int(crit)] += 1
    say("[C] T9b.1: random banks (incl. P=1, orders>P): iff-mismatches=%d (criterion true/false counts %s); K_m independent in all" % (mism, both))


# ---------------------------------------------------------------- 3. T9b.3 : phi(P)<=2 criterion via spectra
def cos_span_criterion(P):
    for h in range(1, P):
        g = gcd(h, P)
        if h not in (g, P - g):
            return False
    return True


def check_cos(pmax=300):
    S = [P for P in range(1, pmax + 1) if cos_span_criterion(P)]
    S2 = [P for P in range(1, pmax + 1) if phi(P) <= 2]
    say("[D] T9b.3: spectral criterion set == {phi(P)<=2} for P<=%d: %s ; set=%s" % (pmax, S == S2, S))
    # harmonic criterion
    bad = 0
    for P in range(2, 121):
        for a in range(1, P // 2 + 1):
            g = gcd(a, P)
            fundamental = a in (g, P - g)
            if fundamental != (P % a == 0):
                bad += 1
    say("    harmonic criterion (1<=a<=P/2: fundamental iff a|P), P<=120: mismatches=%d" % bad)


# ---------------------------------------------------------------- 4. T9c.5 dominance, brute force (also V=2)
def bracket_all_positive(kP, V, P):
    # W=I; margins s_t(c,b) proportional to kP(0)+sum_s kP(s) eps_s, eps_s = 1[mu_{t-s}=c]-1[mu_{t-s}=b]
    for mot in itertools.product(range(V), repeat=P):
        for t in range(P):
            c = mot[t]
            for b in range(V):
                if b == c:
                    continue
                val = kP[0] + sum(kP[s] * ((mot[(t - s) % P] == c) - (mot[(t - s) % P] == b)) for s in range(1, P))
                if val <= 0:
                    return False
    return True


def check_dominance(seed=3):
    rng = random.Random(seed)
    mism = cases = dom = 0
    for (P, V) in [(3, 2), (4, 2), (5, 2), (6, 2), (4, 3), (5, 3)]:
        for trial in range(40):
            kP = [Fraction(rng.randint(-6, 6), 6) for _ in range(P)]
            if trial % 3 == 1:
                kP[0] = sum(abs(x) for x in kP[1:])        # equality case
            if trial % 3 == 2:
                kP[0] = sum(abs(x) for x in kP[1:]) + Fraction(1, 6)
            d = kP[0] > sum(abs(x) for x in kP[1:])
            ok = bracket_all_positive(kP, V, P)
            cases += 1
            dom += d
            if ok != d:
                mism += 1
    say("[E] T9c.5: brute force incl. V=2 and equality cases: cases=%d dominant=%d mismatches=%d (V=2 suffices: worst motif uses 2 symbols)" % (cases, dom, mism))


# ---------------------------------------------------------------- 5. best dominance achievable inside non-divisor-closed spans
def solve(A, b):
    n = len(A)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def best_dominance(basis_vals, P, ncomb_max=None):
    """basis_vals[j][s] = value of generator j at residue s (floats), s=0..P-1.
    Maximise 1 - sum_{s>0}|f(s)| subject to f(0)=1 (so margin>0 iff dominance possible).
    Vertex enumeration of the L1 problem over the affine space {beta: sum beta_j g_j(0)=1}."""
    J = len(basis_vals)
    # distinct residue classes for even functions: s and P-s identical for even generators
    reps = {}
    for s in range(1, P):
        key = tuple(round(basis_vals[j][s], 12) for j in range(J))
        reps.setdefault(key, []).append(s)
    classes = list(reps.items())
    best = None

    def obj(beta):
        return sum(len(ss) * abs(sum(beta[j] * k[j] for j in range(J))) for k, ss in classes)
    row0 = [basis_vals[j][0] for j in range(J)]
    for sub in itertools.combinations(range(len(classes)), J - 1):
        A = [row0] + [list(classes[i][0]) for i in sub]
        b = [1.0] + [0.0] * (J - 1)
        beta = solve(A, b)
        if beta is None:
            continue
        v = obj(beta)
        if best is None or v < best[0]:
            best = (v, beta)
    return best


def check_best_dominance():
    # (i) P=12, bank B16+{1}: orders dividing 12 present: 1,2,3,4 -> generators K_1..K_4 folded
    gens = [[float(K(m, s)) for s in range(12)] for m in (1, 2, 3, 4)]
    v, beta = best_dominance(gens, 12)
    say("[F] P=12, span{K_1,K_2,K_3,K_4} (fold of B16+{1}): min sum_{s>0}|f(s)| with f(0)=1 is %.4f -> dominance margin %.4f (positive means a dominant kernel EXISTS)" % (v, 1 - v))
    say("    argmin beta = %s" % [round(x, 4) for x in beta])
    # (ii) P=12 divisor-closed: K_1,K_2,K_3,K_4,K_6,K_12
    gens = [[float(K(m, s)) for s in range(12)] for m in (1, 2, 3, 4, 6, 12)]
    v, beta = best_dominance(gens, 12)
    say("    P=12, span{K_d : d|12} = full: min = %.4f -> margin %.4f (D_12 itself gives 0)" % (v, 1 - v))
    # (iii) fundamental cosines orders | P (pilot cosine_fund fold)
    for P in (12, 13):
        ds = divisors(P)
        gens = [[math.cos(2 * math.pi * s / d) for s in range(P)] for d in ds]
        v, beta = best_dominance(gens, P)
        say("[G] P=%d, span{cos(2 pi s/d): d|%d} (fold of cosine_fund): min sum|f(s)| = %.4f -> best margin %.4f  (argmin coef over d=%s: %s)" % (P, P, v, 1 - v, ds, [round(x, 4) for x in beta]))
    # (iv) cosine_full fold = all harmonics a=0..floor(P/2): must reach D_P
    for P in (12, 13):
        gens = [[math.cos(2 * math.pi * a * s / P) for s in range(P)] for a in range(0, P // 2 + 1)]
        v, beta = best_dominance(gens, P)
        say("    P=%d, span of all harmonics (fold of cosine_full): min = %.6f -> margin %.6f" % (P, v, 1 - v))
    # (v) CRT S={3,4} family: alpha K_3 + (1-alpha) K_4
    best = None
    for k in range(0, 101):
        a = k / 100.0
        f = [a * float(K(3, s)) + (1 - a) * float(K(4, s)) for s in range(12)]
        v = sum(abs(x) for x in f[1:])
        if best is None or v < best[0]:
            best = (v, a)
    say("[H] convex mixtures aK_3+(1-a)K_4 at P=12: best sum|off-comb| = %.4f at a=%.2f (draft quotes a=1/2: 4)" % best)


# ---------------------------------------------------------------- 6. T9e and T9a numbers
def e_state(V, p, nu, m):
    """Enumerate (M_c, rest-counts multiset) -> probability, true symbol c has count Mc, others counts tuple (sorted)."""
    from math import factorial
    out = {}
    # compositions of m into V parts: first part is the true symbol
    def comps(n, k):
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in comps(n - i, k - 1):
                yield (i,) + rest
    for cmp in comps(m, V):
        coef = factorial(m)
        for x in cmp:
            coef //= factorial(x)
        pr = Fraction(coef) * p ** cmp[0] * nu ** (m - cmp[0])
        out[cmp] = pr
    return out


def e_table():
    for (V, eps, ms, ks) in [(3, Fraction(1, 2), (1, 2, 3, 4, 5, 6), (1, 2, 4)), (16, Fraction(1, 2), (1, 2, 3, 4), (1, 2, 4))]:
        nu = eps / V
        p = 1 - eps + nu
        rho = p / nu
        lam = math.log(rho)
        say("[I] T9e table, V=%d (own enumeration)" % V)
        prev = None
        for m in ms:
            st = e_state(V, p, nu, m)
            res = {}
            for k in ks:
                e = 0.0
                for cmp, pr in st.items():
                    Mc, others = cmp[0], cmp[1:]
                    s = sum(math.exp(k * lam * (o - Mc)) for o in others)
                    e += float(pr) * math.log1p(s)
                res[k] = e
            bayes = res[1]
            row = "    m=%d Bayes=%.6f  " % (m, bayes) + "  ".join("k=%d head=%.6f (+%.6f)" % (k, res[k], res[k] - bayes) for k in ks if k > 1)
            say(row)
    # monotonic check of excess at k=2, V=3
    V, eps = 3, Fraction(1, 2)
    nu = eps / V
    p = 1 - eps + nu
    lam = math.log(p / nu)
    ex = []
    for m in range(1, 7):
        st = e_state(V, p, nu, m)
        b = h = 0.0
        for cmp, pr in st.items():
            Mc, others = cmp[0], cmp[1:]
            b += float(pr) * math.log1p(sum(math.exp(lam * (o - Mc)) for o in others))
            h += float(pr) * math.log1p(sum(math.exp(2 * lam * (o - Mc)) for o in others))
        ex.append(h - b)
    say("    V=3,k=2 excess by m=1..6: %s  monotone decreasing? %s" % ([round(x, 6) for x in ex], all(ex[i] > ex[i + 1] for i in range(5))))


def check_hoeffding_const():
    # draft T9c.3: Pr(A <= EA - s i/4) <= exp(-s^2 i/(8 R^2)); Hoeffding actually gives exp(-s^2 i^2 / (8 (i+1) R^2))
    s, R = 1.0, 1.0
    bad = [i for i in range(1, 2000) if math.exp(-s * s * i * i / (8 * (i + 1) * R * R)) > math.exp(-s * s * i / (8 * R * R)) + 1e-15]
    say("[J] T9c.3 Hoeffding constant: exp(-s^2 i^2/(8(i+1)R^2)) exceeds the draft's exp(-s^2 i/(8R^2)) for every i>=1 (count %d of 1999); a valid uniform replacement is exp(-s^2 i/(16R^2))" % len(bad))


# ---------------------------------------------------------------- 7. L9.1 tightness: error vs 2*Lambda*G for the pilot bank
def l91_tightness():
    P = 12
    rng = random.Random(5)
    orders = list(range(1, 17))
    beta = {m: rng.uniform(-1, 1) for m in orders}
    kt = {m: [float(K(m, r)) for r in range(m)] for m in orders}
    L = 1
    for m in orders:
        L = lcm(L, m)
    Lam = lcm(L, P)
    # fold exactly
    foldP = [sum(beta[m] * kt[m][s % m] for m in orders if P % m == 0) for s in range(P)]
    V, eps = 16, 0.5
    nu = eps / V
    p = 1 - eps + nu
    motif = [rng.randrange(V) for _ in range(P)]
    pi = lambda r, a: p if motif[r % P] == a else nu
    G = sum(abs(b) for b in beta.values())
    a = motif[0]
    worst = 0.0
    D = 0.0
    N = 8192
    for i in range(N + 1):
        # D_i(a) incremental needs recompute (kernel indexed by i-j): O(N^2); use class sums instead
        pass
    # class sums S_i[s] = sum_{n<=i, n=s mod P} kappa(n); D_i(a) = sum_s S_i[s] pi_{i-s}(a)
    S = [0.0] * P
    for i in range(N + 1):
        kn = sum(beta[m] * kt[m][i % m] for m in orders)
        S[i % P] += kn
        t = i % P
        D = sum(S[s] * pi(t - s, a) for s in range(P))
        e = sum(foldP[s] * pi(t - s, a) for s in range(P)) / P
        worst = max(worst, abs(D - (i + 1) * e))
    say("[K] L9.1 on pilot bank {1..16}, P=12: Lambda=%d, G=sum|beta|=%.2f, draft bound 2*Lambda*G=%.3g; actual max_i<=%d |D_i-(i+1)e| = %.3f" % (Lam, G, 2 * Lam * G, N, worst))
    # sharper per-component bound: sum_c 2 |beta_c| lcm(m_c,P)
    sharp = sum(2 * abs(beta[m]) * lcm(m, P) for m in orders)
    say("    per-component bound sum 2|beta_m| lcm(m,P) = %.1f (still valid, ~%dx smaller)" % (sharp, int(2 * Lam * G / sharp)))


def main():
    say("T9 referee recheck (own code)")
    check_identity()
    check_fold_lemma()
    check_span()
    check_cos()
    check_dominance()
    check_best_dominance()
    e_table()
    check_hoeffding_const()
    l91_tightness()
    open("/Users/akari_hayami_64/ArithScope/research/evidence_readout/2026-10-23/audit/recheck.log", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
