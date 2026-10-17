"""Where the Hopf boundary is subcritical, sweeping alpha_1 (Section 3.3.2).

Figure 3.10 reports the sign of the first Lyapunov coefficient along the Hopf boundary for
alpha_1 = 0.69, and the text states that the subcritical stretch falls OUTSIDE the range of
kappa_3 that Section 3.4 recovers from the data. This script supports that statement by
sweeping alpha_1 far beyond the value of the figure.

WHAT IS FOUND. There are two subcritical regions, not one:

  (i)  an ever-present tongue, from kappa_3 = alpha_3 kappa_2/(kappa_2 + alpha_2 K3)
       ~= 0.1308, where E* is born (and which does NOT depend on alpha_1), up to
       kappa_3^B(alpha_1), which grows monotonically with alpha_1;
  (ii) a second, much wider band at intermediate kappa_3, which exists ONLY for alpha_1
       below ~0.1004 and widens as alpha_1 decreases.

Neither reaches the kappa_3 recovered from the data: 0.09 lies below 0.1308, where E* does
not even exist, and 0.21 is supercritical for every alpha_1 tested from 0.02 to 1.20.

NUMERICAL CARE. The second and third derivatives of the field are SYMBOLIC (sympy), not
finite differences, because inside the second band the three terms of g21 cancel by a factor
of up to ~6700 and finite-difference truncation would contaminate the sign. Even with that
cancellation about 12 of the 16 digits of double precision survive. At kappa_3 = 0.21, the
case that supports the statement in the text, there is no cancellation at all.

Run: python bautin_alpha1_sweep.py
Output: results/bautin_alpha1_sweep.txt
"""
import os
import time

import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

os.makedirs("results", exist_ok=True)

K1 = K2 = 1e9
mu1, alpha2, kappa2, K3, alpha3 = 1e-7, 1e-7, 0.34, 2e7, 0.9
K3LO = alpha3*kappa2/(kappa2 + alpha2*K3)      # E* exists above this; independent of alpha_1

KAPPA_DATA = (0.09, 0.21)                      # Section 3.4: vector copy number and tumor volume
ALPHA_TAB = (0.69, 0.97)                       # range of Table 3.1
ALPHA_DATA = (0.104, 0.2648)                   # long hinge and r_B16

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# --------------------------------------------------------------------------- field
T, M, C, b1, k3, a1s = sp.symbols('T M C b1 k3 a1s')
X = (T, M, C)
S = K1                                          # state rescaled by K1 (only changes the scale of l1)
F = sp.Matrix([a1s*S*T*(1 - S*T/K1)*(1 + b1*S*M) - mu1*S*C*S*T,
               alpha2*S*M*S*T*(1 - S*M/K2) - kappa2*S*M,
               alpha3*S*C*S*T/(K3 + S*T) - k3*S*C])/S
_args = (T, M, C, b1, k3, a1s)
Jf = sp.lambdify(_args, F.jacobian(X), 'numpy')
D2f = sp.lambdify(_args, [[[sp.diff(F[i], X[j], X[k]) for k in range(3)]
                           for j in range(3)] for i in range(3)], 'numpy')
D3f = sp.lambdify(_args, [[[[sp.diff(F[i], X[j], X[k], X[m]) for m in range(3)]
                            for k in range(3)] for j in range(3)] for i in range(3)], 'numpy')


def eq(a1, kk, b):
    t = kk*K3/(alpha3 - kk)
    m = K2*(1 - kappa2/(alpha2*t))
    return np.array([t, m, a1/mu1*(1 - t/K1)*(1 + b*m)])/S


def Q(a1, kk, b):
    A = np.asarray(Jf(*eq(a1, kk, b), b, kk, a1), dtype=float)
    a11, a12, a13 = A[0]
    a21, a22, a31 = A[1, 0], A[1, 1], A[2, 0]
    return (-(a11 + a22))*(a11*a22 - a12*a21 - a13*a31) - a13*a22*a31


def beta1_star(a1, kk, h=1e-10):
    """Smallest positive root of Q, which is exactly quadratic in beta_1."""
    q0, q1, q2 = Q(a1, kk, 0.0), Q(a1, kk, h), Q(a1, kk, 2*h)
    A2 = (q2 - 2*q1 + q0)/(2*h*h)
    A1 = (q1 - q0 - A2*h*h)/h
    disc = A1*A1 - 4*A2*q0
    if disc < 0:
        return None
    r = sorted(x for x in ((-A1 - np.sqrt(disc))/(2*A2), (-A1 + np.sqrt(disc))/(2*A2)) if x > 0)
    return r[0] if r else None


def l1_terms(a1, kk, b):
    """l1 (in units of K1) and the cancellation |smallest term| / |sum| in g21."""
    pt = eq(a1, kk, b)
    A = np.asarray(Jf(*pt, b, kk, a1), dtype=float)
    D2 = np.asarray(D2f(*pt, b, kk, a1), dtype=complex)
    D3 = np.asarray(D3f(*pt, b, kk, a1), dtype=complex)
    vals, vecs = np.linalg.eig(A)
    w = float(np.max(np.abs(vals.imag)))
    if w < 1e-12:
        return np.nan, np.nan
    q = vecs[:, int(np.argmin(np.abs(vals - 1j*w)))]
    q = q/np.linalg.norm(q)
    valsT, vecsT = np.linalg.eig(A.T)
    p = vecsT[:, int(np.argmin(np.abs(valsT + 1j*w)))]
    p = p/np.conj(np.vdot(p, q))
    B = lambda x, y: np.einsum('ijk,j,k->i', D2, x, y)
    Cf = lambda x, y, z: np.einsum('ijkl,j,k,l->i', D3, x, y, z)
    qb = np.conj(q)
    h11 = -np.linalg.solve(A, B(q, qb))
    h20 = np.linalg.solve(2j*w*np.eye(3) - A, B(q, q))
    t = (np.vdot(p, Cf(q, q, qb)), 2*np.vdot(p, B(q, h11)), np.vdot(p, B(qb, h20)))
    g = sum(t)
    return g.real/(2*w), min(abs(x.real) for x in t)/abs(g.real)


def L(a1, kk):
    b = beta1_star(a1, kk)
    return np.nan if b is None else l1_terms(a1, kk, b)[0]


def kappa3_c(a1):
    """Value of kappa_3 above which no beta_1 >= 0 destabilizes E*."""
    f = lambda kk: minimize_scalar(lambda b: Q(a1, kk, b), bounds=(0, 1e-4),
                                   method='bounded', options={'xatol': 1e-16}).fun
    try:
        return brentq(f, 0.45, 0.87, xtol=1e-9)
    except Exception:
        return np.nan


def subcritical_intervals(a1, n=260):
    """Intervals of kappa_3 with l1 > 0 along the lower branch."""
    kc = kappa3_c(a1)
    hi = (kc if np.isfinite(kc) else 0.66) - 1e-4
    ks = np.linspace(K3LO + 2e-4, hi, n)
    v = np.array([L(a1, k) for k in ks])
    ok = np.isfinite(v)
    cuts = [brentq(lambda k: L(a1, k), ks[i], ks[i+1], xtol=1e-9)
            for i in range(len(ks)-1)
            if ok[i] and ok[i+1] and np.sign(v[i]) != np.sign(v[i+1])]
    edges = [ks[0]] + cuts + [ks[-1]]
    ivs = [(edges[j], edges[j+1]) for j in range(len(edges)-1)
           if L(a1, 0.5*(edges[j] + edges[j+1])) > 0]
    return ivs, kc


# =========================================================================== run
t0 = time.time()
say("=" * 78)
say("WHERE THE HOPF BOUNDARY IS SUBCRITICAL, SWEEPING alpha_1   (mu_3 = 0)")
say("=" * 78)
say(f"E* exists for kappa_3 > {K3LO:.4f}, a value that does not depend on alpha_1.")
say(f"Table 3.1: alpha_1 in {ALPHA_TAB}.  Data (Section 3.4): alpha_1 ~ {ALPHA_DATA}.")
say(f"kappa_3 recovered from the data: {KAPPA_DATA[0]} (vector copy number) and "
    f"{KAPPA_DATA[1]} (tumor volume).")

say("")
say("-" * 78)
say("(1) subcritical intervals of kappa_3, by alpha_1")
say("")
say("  alpha_1   kappa_3^c   intervals with l1 > 0                    width   % of window")
for a1 in [0.02, 0.05, 0.08, 0.095, 0.104, 0.15, 0.2522, 0.40, 0.69, 0.80, 0.97, 1.5, 3.0]:
    ivs, kc = subcritical_intervals(a1)
    w = sum(b - a for a, b in ivs)
    txt = "  ".join(f"({a:.4f},{b:.4f})" for a, b in ivs) if ivs else "none"
    say(f"  {a1:6.3f}   {kc:.4f}    {txt:40s} {w:.4f}    {100*w/(kc - K3LO):5.1f}%")
say("")
say("  The second band closes between alpha_1 = 0.095 and 0.104; bisection gives ~0.1004.")

say("")
say("-" * 78)
say("(2) dense confirmation: l1 at kappa_3 = 0.21, over 10000 values of alpha_1")
N = 10000
A = np.linspace(0.02, 1.20, N)
V = np.array([L(a, 0.21) for a in A])
fin = np.isfinite(V)
say(f"    alpha_1 from {A[0]:.3f} to {A[-1]:.3f},  finite evaluations {fin.sum()}/{N}")
say(f"    subcritical points (l1 > 0): {(V[fin] > 0).sum()}")
say(f"    largest value of K1^2 l1 (closest approach to zero): {V[fin].max():+.4e} "
    f"at alpha_1 = {A[fin][V[fin].argmax()]:.4f}")
say(f"    smallest value: {V[fin].min():+.4e} at alpha_1 = {A[fin][V[fin].argmin()]:.4f}")
say(f"    => kappa_3 = 0.21 is supercritical over the whole range: {bool((V[fin] < 0).all())}")
say("")
say("    margin of the sign. The cancellation is the ratio between the smallest term of")
say("    g21 and the sum; it costs log10(cancellation) of the 16 digits of double precision:")
for a1 in [0.02, 0.104, 0.2522, 0.69, 0.97, 1.20]:
    v, c = l1_terms(a1, 0.21, beta1_star(a1, 0.21))
    say(f"      alpha_1={a1:6.3f}:  K1^2 l1 = {v:+.4e},  cancellation {c:8.1f}x, "
        f"~{16 - np.log10(c):.1f} of the 16 digits left")
kstar = brentq(lambda a: L(a, 0.21), 1.2, 3.0, xtol=1e-6)
say(f"    kappa_3 = 0.21 only becomes subcritical above alpha_1 ~= {kstar:.3f}, "
    f"{100*(kstar/ALPHA_TAB[1] - 1):.0f}% above the ceiling of Table 3.1.")

say("")
say("-" * 78)
say("(3) dense confirmation: the whole lower branch at alpha_1 = 0.69 (that of Figure 3.10)")
a1 = 0.69
kc = kappa3_c(a1)
KK = np.linspace(K3LO + 1e-5, kc - 1e-5, N)
W = np.array([L(a1, k) for k in KK])
fin2 = np.isfinite(W)
sg = np.sign(W[fin2])
chg = np.where(sg[:-1] != sg[1:])[0]
kf = KK[fin2]
say(f"    kappa_3 from {KK[0]:.5f} to {KK[-1]:.5f} (kappa_3^c = {kc:.4f}), {N} points")
say(f"    sign changes of l1: {len(chg)}")
for i in chg:
    say(f"      between kappa_3 = {kf[i]:.6f} and {kf[i+1]:.6f}")
say(f"    subcritical fraction of the branch: {100*(W[fin2] > 0).sum()/fin2.sum():.2f}%")
say(f"    => exactly one Bautin point at this alpha_1: {len(chg) == 1}")

say("")
say("=" * 78)
say(f"total time: {time.time() - t0:.0f} s")
with open(os.path.join("results", "bautin_alpha1_sweep.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES) + "\n")
print("\nresults in results/bautin_alpha1_sweep.txt")
