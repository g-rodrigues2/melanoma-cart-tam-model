"""Figure 3.11: the Hopf bifurcation diagram in beta1 and the family of limit cycles.

The earlier version showed two (T,M) phase portraits, at beta1 = 0 and 1.05 beta1*, both
started far from E*. The transient dominated both panels and the only visual clue to the
cycle was a white hole in the middle of the tangle; besides, the beta1 = 0 panel repeated,
in another projection, what Figure 3.8(c) already shows, and the caption still said
"numerically consistent with a Hopf-type bifurcation", an unnecessary hedge once the first
Lyapunov coefficient had been computed (l1 < 0, supercritical).

This version shows what that computation predicts:

  (a) bifurcation diagram: minimum and maximum of T on the attractor against beta1. Below
      beta1* the attractor is E*, and T* = k3 K3/(a3 - k3) does NOT depend on beta1, so that
      branch is a horizontal line; above it, the envelope of the cycle opens as the square
      root of the distance to the threshold, the signature of a supercritical Hopf.
  (b) the family of limit cycles in (T,M), growing with beta1.

HOW THE CYCLE IS COMPUTED. Integrating until the transient dies out is impractical: the rate
of approach to the attractor is (dRe lam/dbeta1)(beta1 - beta1*), which vanishes at the
threshold, so the horizon needed grows like 1/eps and reaches millions of days. Instead the
cycle is solved for DIRECTLY as a periodic orbit, by shooting with Newton:

  Poincare section Sigma = {T = T*} (the same for every beta1, since T* does not depend on it);
  unknowns u = (M0, C0, P); equations  phi_P(T*,M0,C0) - (T*,M0,C0) = 0.

Three equations and three unknowns -- the phase is fixed by starting on the section. Each
evaluation integrates ONE period (tens of days), not millions, and Newton converges
quadratically. The continuation reuses the solution of the previous beta1 as its guess, so
only the first point needs a long integration (and a short one at that: it is done at the
largest eps, where convergence is fast). The cost drops from hours to seconds; the function
`check_against_integration` checks the result against direct integration at one point.

With beta1 > 0 the (T,M) plane is a projection of the three-dimensional system, not a closed
subsystem like the (T,C) of Figure 3.8 (where beta1 = 0).
"""
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

alpha1, K1, mu1 = 0.69, 1e9, 1e-7
alpha2, K2, kappa2 = 1e-7, 1e9, 0.34
K3, alpha3, kappa3 = 2e7, 0.9, 0.35

RTOL, ATOL = 1e-11, [1e-2, 1e-2, 1e-2]


def rhs(t, y, beta1):
    T, M, C = y
    return [alpha1*T*(1 - T/K1)*(1 + beta1*M) - mu1*C*T,
            alpha2*M*T*(1 - M/K2) - kappa2*M,
            alpha3*C*T/(K3 + T) - kappa3*C]


def equilibrium(beta1):
    Ts = kappa3*K3/(alpha3 - kappa3)
    Ms = K2*(1 - kappa2/(alpha2*Ts))
    return Ts, Ms, alpha1/mu1*(1 - Ts/K1)*(1 + beta1*Ms)


def jac(beta1):
    Ts, Ms, Cs = equilibrium(beta1)
    return np.array([
        [-(alpha1/K1)*Ts*(1 + beta1*Ms), alpha1*Ts*(1 - Ts/K1)*beta1, -mu1*Ts],
        [alpha2*Ms*(1 - Ms/K2), -kappa2 + alpha2*Ts*(1 - 2*Ms/K2), 0.0],
        [alpha3*Cs*K3/(K3 + Ts)**2, 0.0, 0.0]])


def Q(beta1):
    A = jac(beta1)
    a11, a12, a13 = A[0]
    a21, a22 = A[1, 0], A[1, 1]
    a31 = A[2, 0]
    return (-(a11 + a22))*(a11*a22 - a12*a21 - a13*a31) - a13*a22*a31


beta1_star = brentq(Q, 1e-13, 2e-10, xtol=1e-24, rtol=8.9e-16)
Tstar, Mstar, _ = equilibrium(beta1_star)
_h = beta1_star*1e-4


def _re_pair(beta1):
    ev = np.linalg.eigvals(jac(beta1))
    return ev[np.argsort(-np.abs(ev.imag))][0].real


TRANSV = (_re_pair(beta1_star + _h) - _re_pair(beta1_star - _h))/(2*_h)
OMEGA = float(np.max(np.abs(np.linalg.eigvals(jac(beta1_star)).imag)))


# --------------------------------------------------------------- cycle by shooting
def _flow(beta1, x0, P, dense=False, n=4000):
    te = np.linspace(0, P, n) if dense else None
    return solve_ivp(rhs, (0, P), x0, args=(beta1,), method='LSODA',
                     rtol=RTOL, atol=ATOL, t_eval=te)


def cycle(beta1, guess):
    """Solves the periodic orbit by Newton. guess = (M0, C0, P)."""
    def F(u):
        M0, C0, P = u
        if P <= 0:
            return [1e9, 1e9, 1e9]
        xe = _flow(beta1, [Tstar, M0, C0], P).y[:, -1]
        return [(xe[0] - Tstar)/Tstar, (xe[1] - M0)/Mstar, (xe[2] - C0)/1e7]
    sol = root(F, guess, method='hybr', tol=1e-11)
    return sol.x, sol.success, float(np.max(np.abs(F(sol.x))))


def seed_from_integration(eps):
    """Single initial guess: integrates up to the cycle at the largest eps (fast convergence)."""
    beta1 = beta1_star*(1 + eps)
    E = np.array(equilibrium(beta1))
    g = TRANSV*beta1_star*eps
    Tend = 45/g
    s = solve_ivp(rhs, (0, Tend), [E[0]*1.02, E[1], E[2]], args=(beta1,), method='LSODA',
                  rtol=1e-10, atol=[1.0, 1.0, 1.0],
                  t_eval=np.linspace(Tend*0.98, Tend, 200000))
    T, M, C, t = s.y[0], s.y[1], s.y[2], s.t
    up = np.where((T[:-1] < Tstar) & (T[1:] >= Tstar))[0]       # upward crossings
    i, j = up[-3], up[-2]
    return beta1, Tend, (M[j], C[j], float(t[up[-1]] - t[j]))


def check_against_integration(beta1, u):
    """Checks the cycle obtained by Newton against a long direct integration."""
    E = np.array(equilibrium(beta1))
    g = TRANSV*abs(beta1 - beta1_star)
    Tend = 45/g
    s = solve_ivp(rhs, (0, Tend), [E[0]*1.02, E[1], E[2]], args=(beta1,), method='LSODA',
                  rtol=1e-10, atol=[1.0, 1.0, 1.0],
                  t_eval=np.linspace(Tend*0.97, Tend, 120000))
    orb = _flow(beta1, [Tstar, u[0], u[1]], u[2], dense=True).y
    return (abs(s.y[0].max() - orb[0].max())/orb[0].max(),
            abs(s.y[0].min() - orb[0].min())/orb[0].min())


EPS_BELOW = [-0.30, -0.20, -0.12, -0.06]
EPS_ABOVE = [1.00, 0.70, 0.50, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.012]
EPS_CYCLES = [0.05, 0.18, 0.50, 1.00]   # must be in EPS_ABOVE

t0 = time.time()
b1_seed, seed_horizon, guess = seed_from_integration(EPS_ABOVE[0])
print(f"seed at eps={EPS_ABOVE[0]}: horizon {seed_horizon:.0f} d, "
      f"P~{guess[2]:.2f} d  ({time.time()-t0:.1f} s)")

orbits, rows = {}, []
for eps in EPS_ABOVE:                      # downward continuation, guess from the previous eps
    beta1 = beta1_star*(1 + eps)
    u, ok, res = cycle(beta1, guess)
    assert ok and res < 1e-8, (eps, ok, res)
    guess = u
    orb = _flow(beta1, [Tstar, u[0], u[1]], u[2], dense=True, n=6000).y
    orbits[eps] = (orb, beta1, u)
    rows.append((eps, beta1, float(orb[0].min()), float(orb[0].max()), u[2], res))
    print(f"  eps={eps:+.3f}  P={u[2]:7.2f} d  T in [{orb[0].min():.4e}, {orb[0].max():.4e}]"
          f"  residuo {res:.1e}")
for eps in EPS_BELOW:                      # the attractor is E*: nothing to integrate
    rows.append((eps, beta1_star*(1 + eps), Tstar, Tstar, np.nan, 0.0))
rows.sort()
print(f"\ntotal time of the diagram: {time.time()-t0:.1f} s")

e_chk = 0.35
err_max, err_min = check_against_integration(orbits[e_chk][1], orbits[e_chk][2])
print(f"comparison with direct integration at eps={e_chk}: "
      f"erro rel. em Tmax {err_max:.2e}, em Tmin {err_min:.2e}")

# -------------------------------------------------- spiral below the threshold (short)
eps_sp = -0.20
b_sp = beta1_star*(1 + eps_sp)
E_sp = np.array(equilibrium(b_sp))
sp = solve_ivp(rhs, (0, 900), [E_sp[0]*1.9, E_sp[1]*0.75, E_sp[2]], args=(b_sp,),
               method='LSODA', rtol=1e-10, atol=[1.0, 1.0, 1.0],
               t_eval=np.linspace(0, 900, 40000)).y

np.savez(os.path.join('results', 'hopf_bifurcation_diagram.npz'),
         eps=np.array([r[0] for r in rows]), beta1=np.array([r[1] for r in rows]),
         Tmin=np.array([r[2] for r in rows]), Tmax=np.array([r[3] for r in rows]),
         period=np.array([r[4] for r in rows]),
         beta1_star=beta1_star, Tstar=Tstar, transversality=TRANSV, omega=OMEGA)

# ------------------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 13,
})
RED, BLUE, GREY = "#C1272D", "#1F4E9C", "0.35"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.68))

b = np.array([r[1] for r in rows])/1e-11
tmin = np.array([r[2] for r in rows])/1e7
tmax = np.array([r[3] for r in rows])/1e7
above = np.array([r[0] for r in rows]) > 0
bs = beta1_star/1e-11

# The branches meet AT beta1*, where the cycle is born with zero amplitude: the equilibrium
# ones are drawn as full segments (the line T = T* holds for every beta1) and the cycle one
# gets the point (beta1*, T*). Without that the eps grid leaves a gap around the threshold.
Tst = Tstar/1e7
bmin, bmax = float(b.min()), float(b.max())
ba = np.concatenate(([bs], b[above]))
ta_max = np.concatenate(([Tst], tmax[above]))
ta_min = np.concatenate(([Tst], tmin[above]))

ax1.axvline(bs, color=GREY, lw=1.4, ls=(0, (1, 2.2)), zorder=2)
ax1.plot([bmin, bs], [Tst, Tst], color=BLUE, lw=2.6, zorder=5)
ax1.plot([bs, bmax], [Tst, Tst], color=RED, lw=1.4, ls=(0, (5, 3)), zorder=4)
ax1.plot(ba, ta_max, color=RED, lw=2.4, zorder=5)
ax1.plot(ba, ta_min, color=RED, lw=2.4, zorder=5)
ax1.fill_between(ba, ta_min, ta_max, color=RED, alpha=0.11, zorder=1)
ax1.plot(b[above], tmax[above], 'o', ms=4.2, mfc='white', mec=RED, mew=1.1, zorder=6)
ax1.plot(b[above], tmin[above], 'o', ms=4.2, mfc='white', mec=RED, mew=1.1, zorder=6)
ax1.plot([bs], [Tst], 'o', ms=7, mfc=GREY, mec='white', mew=1.3, zorder=8)

sel = np.where(above)[0][:3]
A_fit = float(np.mean((tmax[sel] - Tstar/1e7)/np.sqrt(b[sel] - bs)))
bb = np.linspace(bs, bs + 0.10*(b[above].max() - bs), 400)   # the law holds only near the threshold
# zorder above the red branch: near the threshold the two curves almost coincide, and it is
# that coincidence the line has to show, so it goes on top.
ax1.plot(bb, Tstar/1e7 + A_fit*np.sqrt(bb - bs), color='0.15', lw=2.0, ls=(0, (5, 2.5)),
         zorder=7)

ax1.set_xlim(b.min(), b.max())
ax1.set_ylim(0, tmax.max()*1.42)   # headroom for the legend in the upper right corner
ax1.set_xlabel(r"$\beta_1$ ($\times10^{-11}$ cells$^{-1}$)", fontsize=14)
ax1.set_ylabel(r"$T$ on the attractor ($\times10^{7}$ cells)", fontsize=14)
ax1.tick_params(labelsize=12)
ax1.annotate(r"$\beta_1^*$", (bs - 0.10, tmax.max()*1.05), fontsize=13, ha="right",
             va="center", color=GREY)
ax1.legend(handles=[
    Line2D([], [], color=BLUE, lw=2.6, label=r"stable $E^*$"),
    Line2D([], [], color=RED, lw=2.4, label="limit cycle, min and max of $T$"),
    Line2D([], [], color=RED, lw=1.4, ls=(0, (5, 3)), label=r"$E^*$, now unstable"),
    Line2D([], [], color='0.15', lw=2.0, ls=(0, (5, 2.5)),
           label=r"$T^*+A\sqrt{\beta_1-\beta_1^*}$ near $\beta_1^*$"),
    Line2D([], [], color=RED, marker='o', ls='None', ms=5.5, mfc='white', mew=1.1,
           label='computed periodic orbits')],
    fontsize=10.5, loc="upper right", framealpha=0.95)

cmap = plt.get_cmap("YlOrRd")
ax2.plot(sp[0]/1e7, sp[1]/1e8, color=BLUE, lw=1.2, alpha=0.95, zorder=4)
for i, eps in enumerate(EPS_CYCLES):
    orb = orbits[eps][0]
    ax2.plot(orb[0]/1e7, orb[1]/1e8, lw=2.0, zorder=5,
             color=cmap(0.35 + 0.55*i/(len(EPS_CYCLES) - 1)),
             label=r"$\beta_1=%.2f\,\beta_1^*$" % (1 + eps))
ax2.plot([sp[0][0]/1e7], [sp[1][0]/1e8], "o", ms=6, mfc="white", mec=BLUE, mew=1.4,
         zorder=8)
ax2.plot([Tstar/1e7], [Mstar/1e8], "*", ms=16, mfc=RED, mec="white", mew=1.2, zorder=9)
ax2.set_xlabel(r"$T$ ($\times10^{7}$ cells)", fontsize=14)
ax2.set_ylabel(r"$M$ ($\times10^{8}$ cells)", fontsize=14)
ax2.tick_params(labelsize=12)
hs, _ = ax2.get_legend_handles_labels()
hs.append(Line2D([], [], color=BLUE, lw=1.2, marker="o", ms=6, mfc="white",
                 mec=BLUE, mew=1.4, markevery=[0],
                 label=r"$\beta_1=0.80\,\beta_1^*$ (into $E^*$)"))
hs.append(Line2D([], [], color=RED, marker="*", ls="None", ms=11, mec="white", label=r"$E^*$"))
ax2.legend(handles=hs, fontsize=10, loc="lower right", framealpha=0.95)

for ax, lab in ((ax1, "a)"), (ax2, "b)")):
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=14, va="top",
            ha="left", fontweight="bold")

fig.tight_layout(w_pad=2.6)
fig.savefig("figures/fig8.pdf", bbox_inches="tight", dpi=300)

ea = np.array([r[0] for r in rows])[above]
order = np.argsort(ea)
amp = (tmax[above] - Tstar/1e7)
eexp = np.polyfit(np.log(ea[order][:4]), np.log(amp[order][:4]), 1)[0]
eexp_all = np.polyfit(np.log(ea), np.log(amp), 1)[0]
print(f"\nbeta1*        = {beta1_star:.6e} cells^-1")
print(f"T*            = {Tstar:.6e} cells (independent of beta1)")
print(f"period at eps->0   = {rows[np.where(above)[0][0]][4]:.2f} d  "
      f"(2 pi / omega = {2*np.pi/OMEGA:.2f} d)")
print(f"log-log exponent of (Tmax - T*) against eps: {eexp:.4f} over the 4 smallest "
      f"(up to eps={ea[order][3]:.3f}), {eexp_all:.4f} over the whole range  (supercritical => 0.5)")
