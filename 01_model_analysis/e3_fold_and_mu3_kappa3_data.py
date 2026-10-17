"""Data of the E3 saddle-node and (kappa3, mu3) stability figures of the thesis (drawn by e3_fold_and_mu3_kappa3_plot.py).

A: saddle-node diagram of the branches T̄1 and T̄2 of the macrophage-free equilibrium E3 as functions of mu3
   (results/figA_data.npz).
B: classification of the (kappa3, mu3) plane by the locally asymptotically stable equilibrium E2 or E3, with
   alpha2 and alpha3 fixed at the values of Table 3.1 (results/figB_data.npz).
"""
import os

import numpy as np
from scipy.optimize import brentq

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)

K1, K2, K3 = 1e9, 1e9, 2e7
alpha2, kappa2 = 1e-7, 0.34
alpha1, mu1 = 0.69, 1e-7

def f3(T, alpha3): return alpha3*T/(K3+T)
def f3p(T, alpha3): return alpha3*K3/(K3+T)**2
def g3(T, alpha3, kappa3): return (f3(T,alpha3)-kappa3)/T
def Tbar0(alpha3, kappa3): return np.sqrt(kappa3)/(np.sqrt(alpha3)-np.sqrt(kappa3))*K3
def mu3max(alpha3, kappa3): return g3(Tbar0(alpha3,kappa3), alpha3, kappa3)

def find_upper_root(func, lo, hi0, growth=2.0, max_hi=1e16):
    hi = hi0
    flo = func(lo)
    while func(hi) * flo > 0 and hi < max_hi:
        hi *= growth
    return brentq(func, lo, hi)

# ---------- A: saddle-node diagram of the branches of E3 versus mu3 ----------
alpha3_A = 0.9
kappa3_A = 0.10   # chosen so that the branch T1 crosses kappa2/alpha2 inside (0, mu3max)
T0_A = Tbar0(alpha3_A, kappa3_A)
m3max_A = mu3max(alpha3_A, kappa3_A)
Tzero_A = kappa3_A*K3/(alpha3_A-kappa3_A)
print(f"[A] kappa3={kappa3_A}  T0={T0_A:.4e}  mu3max={m3max_A:.4e}  Tzero={Tzero_A:.4e}  kappa2/alpha2={kappa2/alpha2:.4e}")

mu3_grid = np.logspace(np.log10(m3max_A*1e-3), np.log10(m3max_A*0.999), 600)
T1_branch, T2_branch = [], []
for m3 in mu3_grid:
    r1 = brentq(lambda T: g3(T,alpha3_A,kappa3_A)-m3, Tzero_A*0.999, T0_A)
    r2 = find_upper_root(lambda T: g3(T,alpha3_A,kappa3_A)-m3, T0_A, T0_A*10)
    T1_branch.append(r1); T2_branch.append(r2)
T1_branch = np.array(T1_branch); T2_branch = np.array(T2_branch)

T1_stable = T1_branch < (kappa2/alpha2)
T2_stable = np.zeros_like(T2_branch, dtype=bool)
mu3_cross = brentq(lambda m3: brentq(lambda T: g3(T,alpha3_A,kappa3_A)-m3, Tzero_A*0.999, T0_A) - kappa2/alpha2,
                    mu3_grid[0], mu3_grid[-1])
print(f"[A] mu3_cross (T1=kappa2/alpha2) = {mu3_cross:.4e}")

np.savez(os.path.join(RESULTS, 'figA_data.npz'), mu3_grid=mu3_grid, T1=T1_branch, T2=T2_branch,
         T1_stable=T1_stable, mu3max=m3max_A, mu3_cross=mu3_cross, T0=T0_A, kappa3=kappa3_A, alpha3=alpha3_A)

# ---------- B: (kappa3, mu3) map of E2 and E3 (alpha2, alpha3 fixed at Table 3.1) ----------
# The domain is the admissible range of Table 3.1 (kappa3 in [0, 0.9] day^-1, mu3 in (0, 5e-9]); the
# earlier version swept mu3 up to 10^-6.5, most of which Table 3.1 excludes. The kappa3 cap is 0.9 =
# alpha3: above its own antigen-driven proliferation rate the CAR T-cell population can never grow,
# so that is the natural ceiling of the model. It also puts kappa3_c(E2) = f3(K1) = 0.882 inside the
# domain, where the E2 boundary reaches mu3 = 0 and E2 becomes unavoidable for every mu3 >= 0.
#
# E2 and E3 are classified INDEPENDENTLY, as their propositions are independent: E2 is LAS iff
# alpha2 > alpha2min and mu3 > g3(K1); E3 is LAS iff T1 exists and T1 < kappa2/alpha2. Nothing forbids
# both from holding at once, and over much of this domain they do -- a bistable region that the earlier
# "first match wins" classification silently painted as pure E3 control.
alpha3_B, alpha2_B = 0.9, 1e-7
alpha2min = kappa2/K1
Tm = kappa2/alpha2_B                       # TAM viability threshold, upper bound for a stable T1
print(f"[B] alpha2={alpha2_B:.2e}  alpha2min={alpha2min:.2e}  alpha2>alpha2min: {alpha2_B>alpha2min}")

def E2_is_LAS(mu3, kappa3):
    """Proposition on E2: LAS iff alpha2 > alpha2min and mu3 > g3(K1)."""
    if alpha2_B <= alpha2min:
        return False
    g3K1 = g3(K1, alpha3_B, kappa3) if kappa3 < alpha3_B else -np.inf
    return mu3 > g3K1

def E3_is_LAS(mu3, kappa3):
    """Proposition on E3: the lower root T1 of g3(T)=mu3 must exist and satisfy T1 < kappa2/alpha2.

    f3'(T1) > mu3 holds automatically on the lower branch (f3 concave), so T1 < kappa2/alpha2 decides.
    """
    if kappa3 >= alpha3_B or mu3 > mu3max(alpha3_B, kappa3):
        return False
    Tzero = kappa3*K3/(alpha3_B-kappa3)
    T0v = Tbar0(alpha3_B, kappa3)
    try:
        T1 = brentq(lambda T: g3(T, alpha3_B, kappa3)-mu3, max(Tzero*(1+1e-9), 1e-3), T0v)
    except Exception:
        return False
    return T1 < K1 and T1 < Tm

def classify(mu3, kappa3):
    """Bit 0 = E2 locally asymptotically stable, bit 1 = E3 locally asymptotically stable.

    0 neither (E* is the candidate, but Proposition P6 covers only mu3 = 0)
    1 E2 only (treatment failure)
    2 E3 only (TAM-free control)
    3 both    (bistable: the outcome depends on the initial condition)
    """
    return (1 if E2_is_LAS(mu3, kappa3) else 0) + (2 if E3_is_LAS(mu3, kappa3) else 0)

# kappa3 starts just above 0: at kappa3 = 0 exactly the lower branch T1 degenerates (Tzero and the fold
# T0 both collapse to 0 and g3 is monotone), so that single column is a measure-zero artefact, not the
# limit of its neighbours. mu3 starts at 1e-11, one decade below the E2 boundary; everything below is
# uniformly unclassified and would only add dead space.
kappa3_grid = np.linspace(0.005, 0.9, 900)        # Table 3.1 range [0, 0.9]
mu3_grid_B = np.logspace(-11, np.log10(5e-9), 600)  # Table 3.1 upper bound 5e-9
KG, MG = np.meshgrid(kappa3_grid, mu3_grid_B)
Z = np.zeros(KG.shape)
for i in range(KG.shape[0]):
    for j in range(KG.shape[1]):
        Z[i, j] = classify(MG[i, j], KG[i, j])

# Closed-form boundaries. Both are the SAME function mu3 = g3(T, kappa3) = (f3(T)-kappa3)/T, evaluated at
# the two tumour sizes that matter: the TAM viability threshold kappa2/alpha2 (above which E3 loses
# stability) and the carrying capacity K1 (above which E2 gains it). Each vanishes at kappa3 = f3(T).
bnd_E3 = (f3(Tm, alpha3_B) - kappa3_grid) / Tm          # upper edge of the E3 region
bnd_E2 = (f3(K1, alpha3_B) - kappa3_grid) / K1          # lower edge of the E2 region
kappa3_c_E3 = f3(Tm, alpha3_B)      # E3 impossible for any mu3 >= 0 above this
kappa3_c_E2 = f3(K1, alpha3_B)      # E2 unavoidable for every mu3 >= 0 above this

np.savez(os.path.join(RESULTS, 'figB_data.npz'), kappa3_grid=kappa3_grid, mu3_grid=mu3_grid_B, Z=Z,
         alpha2=alpha2_B, alpha3=alpha3_B, bnd_E3=bnd_E3, bnd_E2=bnd_E2,
         kappa3_c_E3=kappa3_c_E3, kappa3_c_E2=kappa3_c_E2, Tm=Tm)
# Size of the bistable window. The pixel fraction of Z is NOT a usable number: the mu3 axis is
# logarithmic and truncated at an arbitrary floor, so it changes with that floor (9.2% at 1e-10, 5.8% at
# 1e-11, 3.4% at 1e-13). The invariant quantity is the LINEAR measure over the Table 3.1 rectangle
# kappa3 in [0, 0.9] x mu3 in [0, 5e-9], i.e. the probability under a uniform prior on those ranges.
MU3_MAX_TAB, KAPPA3_MAX_TAB = 5e-9, 0.9
_lo = lambda k3: g3(K1, alpha3_B, k3)
_hi = lambda k3: min(MU3_MAX_TAB, g3(Tm, alpha3_B, k3))
_grid = np.linspace(0.0, kappa3_c_E3, 200001)
_w = np.maximum(0.0, np.array([_hi(k) for k in _grid]) - np.array([_lo(k) for k in _grid]))
bistable_measure = float(np.trapezoid(_w, _grid) / (KAPPA3_MAX_TAB * MU3_MAX_TAB))
kappa3_cap_binds = kappa3_c_E3 - MU3_MAX_TAB * Tm   # below this, the Table 3.1 cap is the upper limit

frac = {v: float((Z == v).mean()) for v in [0, 1, 2, 3]}
print(f"[B] kappa3_c(E3) = f3(kappa2/alpha2) = {kappa3_c_E3:.5f} /day")
print(f"[B] kappa3_c(E2) = f3(K1)            = {kappa3_c_E2:.5f} /day  (inside the Table 3.1 range)")
print(f"[B] mu3 boundary of E2 over the domain: {bnd_E2.min():.3e} .. {bnd_E2.max():.3e}")
print(f"[B] pixel fractions (log axis, floor-dependent -- do NOT quote): neither={frac[0]:.3f}  "
      f"E2 only={frac[1]:.3f}  E3 only={frac[2]:.3f}  bistable={frac[3]:.3f}")
print(f"[B] bistable window, LINEAR measure over the Table 3.1 rectangle = {bistable_measure:.4f}")
print(f"[B] the Table 3.1 cap mu3<=5e-9 binds below kappa3 = {kappa3_cap_binds:.5f} /day")
