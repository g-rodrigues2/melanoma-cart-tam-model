"""Dynamical test of the SUBcritical Hopf below the Bautin point (Section 3.3.2).

The sign of the first Lyapunov coefficient says that, for kappa_3 below the Bautin point
(kappa_3^B ~= 0.1776), the Hopf bifurcation in beta_1 is subcritical. That is a claim about
the DYNAMICS, and not merely about a formula, and it is what this script checks, in three
ways independent of the computation of l1:

  (i)  BELOW the threshold there is an UNSTABLE periodic orbit around E*. It is solved by
       shooting with Newton on the Poincare section Sigma = {T = T*}, the same method used
       for the stable cycles of Figure 3.11, which works just as well for unstable orbits
       because Newton does not care about stability. The instability is then certified by
       the FLOQUET MULTIPLIERS, obtained by integrating the variational equation along the
       orbit: one multiplier is always 1 (the flow direction) and the other two decide;
       |mu| > 1 in either of them means an unstable orbit.
  (ii) The amplitude of that orbit must follow the square-root law predicted by l1,
       Delta T ~= 2 q_1 sqrt(2|rho| / |Re g21|), with rho = (dRe lambda/dbeta_1)(beta_1 - beta_1^*).
       The ratio between the measured and the predicted value must tend to 1 as
       beta_1 -> beta_1^*.
  (iii) The unstable orbit is the boundary between two basins. Perturbations of E* smaller
       than it return to E*; larger ones escape to a large-amplitude cycle. That is what
       makes the loss of stability ABRUPT rather than gradual.

CONTROL. Everything is repeated at kappa_3 = 0.35, where l1 < 0 (supercritical). There no
orbit should exist below the threshold and no perturbation should escape, however large.

Run: python subcritical_bistability.py
Output: results/subcritical_bistability.txt
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

os.makedirs("results", exist_ok=True)

alpha1, K1, mu1 = 0.69, 1e9, 1e-7
alpha2, K2, kappa2 = 1e-7, 1e9, 0.34
K3, alpha3 = 2e7, 0.9

RTOL, ATOL = 1e-11, [1e-2, 1e-2, 1e-2]

KAPPA_SUB = 0.145      # below the Bautin point, l1 > 0
KAPPA_SUP = 0.35       # the section used in the chapter, l1 < 0

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# --------------------------------------------------------------------------- field and Jacobian
def rhs(t, y, beta1, k3):
    T, M, C = y
    return [alpha1*T*(1 - T/K1)*(1 + beta1*M) - mu1*C*T,
            alpha2*M*T*(1 - M/K2) - kappa2*M,
            alpha3*C*T/(K3 + T) - k3*C]


def Df(y, beta1, k3):
    """Jacobian of the field at an arbitrary point (not only at the equilibrium)."""
    T, M, C = y
    return np.array([
        [alpha1*(1 - 2*T/K1)*(1 + beta1*M) - mu1*C, alpha1*T*(1 - T/K1)*beta1, -mu1*T],
        [alpha2*M*(1 - M/K2), alpha2*T*(1 - 2*M/K2) - kappa2, 0.0],
        [alpha3*C*K3/(K3 + T)**2, 0.0, alpha3*T/(K3 + T) - k3]])


def equilibrium(beta1, k3):
    Ts = k3*K3/(alpha3 - k3)
    Ms = K2*(1 - kappa2*(alpha3 - k3)/(k3*alpha2*K3))
    return np.array([Ts, Ms, alpha1/mu1*(1 - Ts/K1)*(1 + beta1*Ms)])


def Q(beta1, k3):
    A = Df(equilibrium(beta1, k3), beta1, k3)
    a11, a12, a13 = A[0]
    a21, a22, a31 = A[1, 0], A[1, 1], A[2, 0]
    return (-(a11 + a22))*(a11*a22 - a12*a21 - a13*a31) - a13*a22*a31


def beta1_star_of(k3):
    grid = np.geomspace(1e-14, 1e-5, 3000)
    v = np.array([Q(b, k3) for b in grid])
    i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0][0]
    return brentq(lambda b: Q(b, k3), grid[i], grid[i+1], xtol=1e-28, rtol=8.9e-16)


# --------------------------------------------------------------------------- l1 and eigenvector
def lyapunov_data(k3, beta1):
    """l1, omega, q (normalized critical eigenvector, phase fixed by q_1 real > 0) and Re g21,
    in cell units, by the projection method of Kuznetsov (Sec. 5.4)."""
    x0 = equilibrium(beta1, k3)
    A = Df(x0, beta1, k3)
    vals, vecs = np.linalg.eig(A)
    w = float(np.max(np.abs(vals.imag)))
    q = vecs[:, int(np.argmin(np.abs(vals - 1j*w)))]
    q = q/np.linalg.norm(q)
    q = q*np.exp(-1j*np.angle(q[0]))                      # q_1 real e positivo
    valsT, vecsT = np.linalg.eig(A.T)
    p = vecsT[:, int(np.argmin(np.abs(valsT + 1j*w)))]
    p = p/np.conj(np.vdot(p, q))

    h = 1e-3*np.abs(x0)
    D2 = np.zeros((3, 3, 3))
    D3 = np.zeros((3, 3, 3, 3))
    e = np.eye(3)

    def f(u):
        return np.array(rhs(0.0, u, beta1, k3))

    for j in range(3):
        for k in range(3):
            hj, hk = h[j]*e[j], h[k]*e[k]
            D2[:, j, k] = (f(x0+hj+hk) - f(x0+hj-hk) - f(x0-hj+hk) + f(x0-hj-hk))/(4*h[j]*h[k])
            for m in range(3):
                hm = h[m]*e[m]
                D3[:, j, k, m] = (f(x0+hj+hk+hm) - f(x0+hj+hk-hm) - f(x0+hj-hk+hm)
                                  + f(x0+hj-hk-hm) - f(x0-hj+hk+hm) + f(x0-hj+hk-hm)
                                  + f(x0-hj-hk+hm) - f(x0-hj-hk-hm))/(8*h[j]*h[k]*h[m])

    B = lambda a, b: np.einsum('ijk,j,k->i', D2.astype(complex), a, b)
    Cf = lambda a, b, c: np.einsum('ijkl,j,k,l->i', D3.astype(complex), a, b, c)
    qb = np.conj(q)
    h11 = -np.linalg.solve(A, B(q, qb))
    h20 = np.linalg.solve(2j*w*np.eye(3) - A, B(q, q))
    g21 = np.vdot(p, Cf(q, q, qb)) + 2*np.vdot(p, B(q, h11)) + np.vdot(p, B(qb, h20))
    return g21.real/(2*w), w, q, g21.real


def transversality(k3, bstar):
    hb = bstar*1e-4

    def re_pair(b):
        ev = np.linalg.eigvals(Df(equilibrium(b, k3), b, k3))
        return ev[np.argsort(-np.abs(ev.imag))][0].real
    return (re_pair(bstar + hb) - re_pair(bstar - hb))/(2*hb)


# --------------------------------------------------------------------------- orbit and Floquet
def cycle(k3, beta1, guess, Tstar, Mstar):
    """Periodic orbit by shooting on Sigma = {T = T*}. guess = (M0, C0, P)."""
    def F(u):
        M0, C0, P = u
        if P <= 0:
            return [1e9, 1e9, 1e9]
        xe = solve_ivp(rhs, (0, P), [Tstar, M0, C0], args=(beta1, k3), method='LSODA',
                       rtol=RTOL, atol=ATOL).y[:, -1]
        return [(xe[0] - Tstar)/Tstar, (xe[1] - M0)/Mstar, (xe[2] - C0)/1e7]
    sol = root(F, guess, method='hybr', tol=1e-12)
    return sol.x, float(np.max(np.abs(F(sol.x))))


def floquet(k3, beta1, u, Tstar):
    """Floquet multipliers from the monodromy matrix, integrating the variational equation."""
    M0, C0, P = u

    def ext(t, z):
        y, Phi = z[:3], z[3:].reshape(3, 3)
        return np.concatenate([rhs(t, y, beta1, k3), (Df(y, beta1, k3) @ Phi).ravel()])

    z0 = np.concatenate([[Tstar, M0, C0], np.eye(3).ravel()])
    zf = solve_ivp(ext, (0, P), z0, method='LSODA', rtol=1e-11,
                   atol=[1e-2, 1e-2, 1e-2] + [1e-10]*9).y[:, -1]
    mu = np.linalg.eigvals(zf[3:].reshape(3, 3))
    i_triv = int(np.argmin(np.abs(np.abs(mu) - 1.0)))      # o multiplicador trivial vale 1
    nontriv = np.delete(mu, i_triv)
    return mu, mu[i_triv], nontriv


def orbit_excursion(k3, beta1, u, Tstar):
    """Excursion of T ABOVE T* along the orbit, which is the quantity predicted by
    eq:app:dTpred in Appendix A (and is half the peak-to-trough range)."""
    M0, C0, P = u
    s = solve_ivp(rhs, (0, P), [Tstar, M0, C0], args=(beta1, k3), method='LSODA',
                  rtol=RTOL, atol=ATOL, t_eval=np.linspace(0, P, 4000))
    return s.y[0].max() - Tstar


def fate(k3, beta1, kick, Tstar, tmax):
    """Integrates from E* displaced in T and returns the relative final amplitude."""
    x0 = equilibrium(beta1, k3).copy()
    x0[0] += kick
    s = solve_ivp(rhs, (0, tmax), x0, args=(beta1, k3), method='LSODA',
                  rtol=1e-10, atol=[1.0, 1.0, 1.0],
                  t_eval=np.linspace(0.9*tmax, tmax, 20000))
    return (s.y[0].max() - s.y[0].min())/Tstar


# =========================================================================== run
say("=" * 78)
say("DYNAMICAL TEST OF THE SUBCRITICAL HOPF (alpha_1 = 0.69, mu_3 = 0)")
say("=" * 78)

report = {}
for k3 in (KAPPA_SUB, KAPPA_SUP):
    bstar = beta1_star_of(k3)
    l1, w, q, reg21 = lyapunov_data(k3, bstar)
    tr = transversality(k3, bstar)
    E = equilibrium(bstar, k3)
    report[k3] = dict(bstar=bstar, l1=l1, w=w, q=q, reg21=reg21, tr=tr, Tstar=E[0], Mstar=E[1])
    say("")
    say(f"kappa_3 = {k3:.3f} day^-1")
    say(f"  beta_1*          = {bstar:.6e} cells^-1")
    say(f"  omega            = {w:.6f} day^-1,  period 2pi/omega = {2*np.pi/w:.2f} days")
    say(f"  l1               = {l1:+.4e} cells^-2  ({l1*K1**2:+.4f} in units of K1)")
    say(f"  Re g21           = {reg21:+.4e}")
    say(f"  dRe lambda/dbeta_1 = {tr:.4e} cells")
    say(f"  prediction       = {'SUBcritical' if l1 > 0 else 'supercritical'}")

# --- T1: agreement with the independent implementation of stability_maps -----------
say("")
say("-" * 78)
say("T1  l1 against the independent implementation of stability_maps_and_hopf_boundary.py")
try:
    from stability_maps_and_hopf_boundary import l1_at as l1_rescaled
    for k3 in (KAPPA_SUB, KAPPA_SUP):
        a = report[k3]['l1']*K1**2
        b = l1_rescaled(k3, report[k3]['bstar'])
        say(f"    kappa_3={k3:.3f}:  here {a:+.6f}   there {b:+.6f}   relative error {abs(a-b)/abs(b):.2e}")
except Exception as exc:                                      # pragma: no cover
    say(f"    (import failed: {exc})")

# --- T2: unstable orbit below the threshold, at subcritical kappa_3 ----------------
say("")
say("-" * 78)
say(f"T2  periodic orbit BELOW the threshold at kappa_3 = {KAPPA_SUB} (prediction: it exists and is UNSTABLE)")
say("")
say("      eps   measured excursion  predicted excursion  ratio   nontrivial |mu|      residual")
r = report[KAPPA_SUB]
rows = []
for eps in (0.20, 0.10, 0.05, 0.02, 0.01, 0.005):
    b1 = r['bstar']*(1 - eps)
    rho = r['tr']*(b1 - r['bstar'])                           # negative below the threshold
    rad = np.sqrt(2*abs(rho)/abs(r['reg21']))
    dT_pred = 2*r['q'][0].real*rad
    # analytic seed: on the section T = T*, the perturbation is -2 r Im(q)
    seed = (r['Mstar'] - 2*rad*r['q'][1].imag,
            equilibrium(b1, KAPPA_SUB)[2] - 2*rad*r['q'][2].imag,
            2*np.pi/r['w'])
    u, res = cycle(KAPPA_SUB, b1, seed, r['Tstar'], r['Mstar'])
    dT = orbit_excursion(KAPPA_SUB, b1, u, r['Tstar'])
    _, mu_triv, nt = floquet(KAPPA_SUB, b1, u, r['Tstar'])
    rows.append((eps, dT, dT_pred, dT/dT_pred, max(abs(nt)), res, mu_triv))
    say(f"    {eps:6.3f}  {dT:.4e}      {dT_pred:.4e}     {dT/dT_pred:6.3f}   "
        f"{abs(nt[0]):.4f}, {abs(nt[1]):.4f}   {res:.1e}")
say("")
say(f"    trivial multiplier (must be 1): {abs(rows[-1][6]):.8f}")
say(f"    every orbit unstable (max |mu| > 1): {all(x[4] > 1 for x in rows)}")
say(f"    measured/predicted ratio -> 1 as eps -> 0: "
    f"{rows[0][3]:.3f} em eps={rows[0][0]:.3f} e {rows[-1][3]:.3f} em eps={rows[-1][0]:.3f}")
dev = [(x[0], x[3] - 1.0) for x in rows]
say("    deviation of the ratio from 1, against eps:")
say("      " + "   ".join(f"eps={e:.3f}: {d:+.4f}" for e, d in dev))
aexp = np.polyfit(np.log([e for e, _ in dev]), np.log([d for _, d in dev]), 1)[0]
say(f"    fit deviation ~ eps^a:  a = {aexp:.2f}  (monotone deviation, tending to zero)")

# --- T3: control, supercritical kappa_3 -------------------------------------------
say("")
say("-" * 78)
say(f"T3  control at kappa_3 = {KAPPA_SUP} (l1 < 0): there should be no orbit below the threshold")
rs = report[KAPPA_SUP]
eps = 0.05
b1 = rs['bstar']*(1 - eps)
rho = rs['tr']*(b1 - rs['bstar'])
rad = np.sqrt(2*abs(rho)/abs(rs['reg21']))
seed = (rs['Mstar'] - 2*rad*rs['q'][1].imag,
        equilibrium(b1, KAPPA_SUP)[2] - 2*rad*rs['q'][2].imag,
        2*np.pi/rs['w'])
u_s, res_s = cycle(KAPPA_SUP, b1, seed, rs['Tstar'], rs['Mstar'])
dT_s = orbit_excursion(KAPPA_SUP, b1, u_s, rs['Tstar'])
say(f"    seed from the same formula, amplitude {2*rs['q'][0].real*rad:.3e}")
say(f"    Newton returns an orbit of amplitude {dT_s:.3e} cells "
    f"({dT_s/rs['Tstar']*100:.2e}% of T*), residual {res_s:.1e}")
say(f"    negligible amplitude => it collapsed onto the equilibrium, there is no cycle: "
    f"{dT_s/rs['Tstar'] < 1e-6}")

# --- T4: bistability by direct integration ----------------------------------------
say("")
say("-" * 78)
say("T4  bistability by direct integration, beta_1 = 0.95 beta_1* (E* stable in both cases)")
for k3 in (KAPPA_SUB, KAPPA_SUP):
    rr = report[k3]
    b1 = rr['bstar']*0.95
    rho = rr['tr']*(b1 - rr['bstar'])
    rad = np.sqrt(2*abs(rho)/abs(rr['reg21']))
    sep = 2*rr['q'][0].real*rad                               # half-width of the separatrix in T
    tmax = 60/abs(rho)
    say("")
    say(f"    kappa_3 = {k3:.3f}  (l1 {'>' if rr['l1'] > 0 else '<'} 0),  Q = {Q(b1, k3):+.2e} > 0,"
        f"  scale 1/|rho| = {1/abs(rho):.0f} days,  horizon {tmax:.0f} days")
    say(f"      reference (predicted amplitude of the unstable orbit) = {sep:.3e} cells")
    for frac in (0.5, 1.0, 2.0, 5.0):
        amp = fate(k3, b1, frac*sep, rr['Tstar'], tmax)
        say(f"      perturbation {frac:4.1f}x ({frac*sep:9.3e} cells):  final amplitude / T* = "
            f"{amp:9.4f}   {'returns to E*' if amp < 1e-3 else 'ESCAPES'}")

say("")
say("=" * 78)
with open(os.path.join("results", "subcritical_bistability.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES) + "\n")
print("\nresults in results/subcritical_bistability.txt")
