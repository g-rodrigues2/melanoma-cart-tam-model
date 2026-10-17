"""Figure 3.8: the two local regimes of E* of Proposition P7 (beta1 = 0, mu3 = 0).

Figure 3.7 classifies the whole branch by the LINEARIZATION. This figure shows what that
one cannot: the actual nonlinear dynamics, and the geometry that gives the two regimes
their names.

With beta1 = 0, M appears in neither dT/dt nor dC/dt: (T,C) is a CLOSED subsystem of the
full system (checked below against the 3D integration). The phase portrait of the bottom row
is therefore the true dynamics, not a projection, and the spiral and the sink are literal.

  top row   : T(t) from the initial condition of the chapter, log axis. The excursion reaches
              ~10x T* at the first peak, far outside the linear regime: the local rate is only
              attained asymptotically.
  bottom row: (T,C) phase plane near E*, with nullclines and a direction field; in the sink
              case, the two eigendirections as well.
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)

alpha1, K1, mu1 = 0.69, 1e9, 1e-7
alpha2, K2, kappa2 = 1e-7, 1e9, 0.34
K3, alpha3 = 2e7, 0.9
beta1 = 0.0

TMC0 = (2e6, 1e6, 1e7)
KAPPA3 = (0.5, 0.86)


def f3d(t, y, k3):
    T, M, C = y
    return [alpha1*T*(1-T/K1)*(1+beta1*M) - mu1*T*C,
            alpha2*T*M*(1-M/K2) - kappa2*M,
            alpha3*T*C/(K3+T) - k3*C]


def f2d(t, y, k3):
    T, C = y
    return [alpha1*T*(1-T/K1) - mu1*T*C, alpha3*T*C/(K3+T) - k3*C]


def equilibrium(k3):
    T = k3*K3/(alpha3-k3)
    return T, alpha1/mu1*(1-T/K1)


def block(k3):
    T, C = equilibrium(k3)
    J = np.array([[-alpha1/K1*T, -mu1*T], [alpha3*C*K3/(K3+T)**2, 0.0]])
    w, v = np.linalg.eig(J)
    return w, v


for _k3 in KAPPA3:                       # (T,C) really is closed
    _s2 = solve_ivp(f2d, (0, 400), [TMC0[0], TMC0[2]], args=(_k3,), rtol=1e-11, atol=1e-6, dense_output=True)
    _s3 = solve_ivp(f3d, (0, 400), list(TMC0), args=(_k3,), rtol=1e-11, atol=1e-6, dense_output=True)
    _t = np.linspace(0, 400, 20000)
    assert np.max(np.abs(_s3.sol(_t)[0] - _s2.sol(_t)[0]))/np.max(_s2.sol(_t)[0]) < 1e-8

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 13,
})
BLACK, RED, BLUE, GREY = "#111111", "#C1272D", "#1F4E9C", "0.45"
NULL = "#1B7F79"   # nullclines: their own color, so as not to be confused with the eigendirections

fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
TMAX = 400.0

for col, k3 in enumerate(KAPPA3):
    Ts, Cs = equilibrium(k3)
    w, v = block(k3)
    spiral = bool(np.any(np.abs(w.imag) > 1e-12))
    ck = RED if spiral else BLUE

    # ----- (a),(b) global trajectory, log axis -----------------------------
    ax = axes[0, col]
    sol = solve_ivp(f2d, (0, TMAX), [TMC0[0], TMC0[2]], args=(k3,), rtol=1e-11, atol=1e-6, dense_output=True)
    t = np.linspace(0, TMAX, 60000)
    T = sol.sol(t)[0]
    ax.plot(t, T, color=BLACK, lw=1.8, zorder=5)
    ax.axhline(Ts, color=ck, ls=(0, (6, 3)), lw=1.6, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(0, TMAX)
    ax.set_ylim(min(T.min(), Ts)/2.2, max(T.max(), Ts)*2.6)
    ax.set_xlabel("time (days)", fontsize=14)
    ax.set_ylabel(r"$T(t)$ (cells)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.annotate(r"$T^*$", (TMAX*0.985, Ts*1.22), color=ck, fontsize=12.5, ha="right", va="bottom")
    if spiral:
        note = (r"period $%.0f$ d,  envelope half-life $%.0f$ d" % (2*np.pi/abs(w[0].imag), np.log(2)/abs(w[0].real)))
    else:
        note = (r"slow mode half-life $%.0f$ d" % (np.log(2)/min(np.abs(w.real))))
    ax.annotate(note, (0.5, 0.025), xycoords="axes fraction", fontsize=11, ha="center",
                va="bottom", color=ck)

    # ----- (c),(d) local phase plane ---------------------------------------
    ax = axes[1, col]
    Tmax, Cmax = (2.5*Ts, 1.95*Cs) if spiral else (1.85*Ts, 1.95*Cs)
    Tg, Cg = np.meshgrid(np.linspace(Tmax*2e-2, Tmax, 34), np.linspace(Cmax*2e-2, Cmax, 34))
    ax.streamplot(Tg, Cg, alpha1*Tg*(1-Tg/K1) - mu1*Tg*Cg, alpha3*Tg*Cg/(K3+Tg) - k3*Cg,
                  color="0.82", linewidth=0.7, density=0.8, arrowsize=0.8, zorder=1)

    Tl = np.linspace(0, Tmax, 400)
    ax.plot(Tl, alpha1/mu1*(1-Tl/K1), color=NULL, lw=1.6, ls=(0, (7, 3)), zorder=4)
    ax.axvline(Ts, color=NULL, lw=1.6, ls=(0, (1, 2.2)), zorder=4)

    if spiral:
        starts, span = [(0.45, 0.55), (1.70, 1.45)], 88.0
    else:
        starts, span = [(0.32, 1.70), (1.65, 0.40)], 160.0
    for fT, fC in starts:
        s = solve_ivp(f2d, (0, span), [fT*Ts, fC*Cs], args=(k3,), rtol=1e-11, atol=1e-8, dense_output=True)
        tt = np.linspace(0, span, 40000)
        ax.plot(s.sol(tt)[0], s.sol(tt)[1], color=ck, lw=1.8, zorder=6)
        ax.plot([fT*Ts], [fC*Cs], "o", ms=4.5, mfc="white", mec=ck, mew=1.3, zorder=7)

    if not spiral:
        for idx, sty in zip(np.argsort(np.abs(w.real)), [(0, (5, 2)), (0, (1, 2))]):
            d = np.real(v[:, idx])
            n = np.hypot(d[0]/Ts, d[1]/Cs)
            ax.plot([Ts - 0.75*d[0]/n, Ts + 0.75*d[0]/n],
                    [Cs - 0.75*d[1]/n, Cs + 0.75*d[1]/n],
                    color=BLUE, lw=1.4, ls=sty, alpha=0.95, zorder=5)

    ax.plot([Ts], [Cs], "o", ms=9.5, mfc=ck, mec="white", mew=1.7, zorder=10)
    ax.annotate(r"$E^*$", (Ts, Cs), textcoords="offset points", xytext=(10, 8),
                fontsize=13, color=ck, fontweight="bold")
    ax.set_xlim(0, Tmax)
    ax.set_ylim(0, Cmax)
    ax.set_xlabel(r"$T$ (cells)", fontsize=14)
    ax.set_ylabel(r"$C$ (cells)", fontsize=14)
    ax.tick_params(labelsize=12)

# panel letter + the value of kappa3, inside the axes (the standard of the chapter)
for ax, lab, k3 in zip(axes.ravel(), ["a)", "b)", "c)", "d)"], [0.5, 0.86, 0.5, 0.86]):
    ax.text(0.025, 0.965, lab, transform=ax.transAxes, fontsize=14, va="top",
            ha="left", fontweight="bold")
    ax.text(0.082, 0.963, r"$\kappa_3=%.2f$ day$^{-1}$" % k3, transform=ax.transAxes,
            fontsize=12, va="top", ha="left")

axes[0, 0].legend(handles=[Line2D([], [], color=BLACK, lw=1.8, label=r"$T(t)$, nonlinear system")],
                  fontsize=10.5, loc="upper right", framealpha=0.95)
axes[1, 0].legend(handles=[
    Line2D([], [], color=NULL, lw=1.6, ls=(0, (7, 3)), label=r"$T$-nullcline"),
    Line2D([], [], color=NULL, lw=1.6, ls=(0, (1, 2.2)), label=r"$C$-nullcline"),
    Line2D([], [], color="0.82", lw=0.9, label="direction field")],
    fontsize=10, loc="lower left", framealpha=0.95)
axes[1, 1].legend(handles=[
    Line2D([], [], color=BLUE, lw=1.4, ls=(0, (5, 2)), label="slow eigendirection"),
    Line2D([], [], color=BLUE, lw=1.4, ls=(0, (1, 2)), label="fast eigendirection")],
    fontsize=10, loc="lower left", framealpha=0.95)

fig.tight_layout(h_pad=2.2, w_pad=2.4)
fig.savefig("figures/fig6.pdf", bbox_inches="tight", dpi=300)

for k3 in KAPPA3:
    Ts, Cs = equilibrium(k3)
    w, _ = block(k3)
    s = solve_ivp(f2d, (0, TMAX), [TMC0[0], TMC0[2]], args=(k3,), rtol=1e-11, atol=1e-6, dense_output=True)
    T = s.sol(np.linspace(0, TMAX, 60000))[0]
    print(f"kappa3={k3}: E*=({Ts:.3e},{Cs:.3e})  {'spiral' if np.any(abs(w.imag)>1e-12) else 'sink'}  "
          f"autovalores={np.round(w,5)}  pico/T*={T.max()/Ts:.1f}")
