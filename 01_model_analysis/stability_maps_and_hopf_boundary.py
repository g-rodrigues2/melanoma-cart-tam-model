"""Stability maps of Chapter 3 (figures/stability_a2_a3.png and figures/hopf_k3_b1.png).

(1) (alpha2, alpha3) plane with mu3 = 0 and beta1 = 0: regions where E1, E2, E* or E3 is the attracting
    equilibrium, with the reference point of Table 3.1.
(2) (kappa3, beta1) plane: E* as a stable limit cycle, a damped-oscillation spiral or a monotone sink, with
    both branches of the Hopf boundary Q(beta1) = 0 and the value kappa3^c above which the instability
    window closes. The spiral/sink boundary at beta1 -> 0 is checked against Proposition 3.7. The lower
    panel reports the first Lyapunov coefficient along that boundary, whose sign splits it into a
    supercritical and a subcritical stretch, separated by a Bautin point.
"""
import os

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import brentq, minimize_scalar

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
    'axes.linewidth': 1.0,
})

# Table 3.1 (cell units)
K1 = 1e9; K2 = 1e9; K3 = 2e7; k2 = 0.34; u1 = 1e-7; a1 = 0.69

# Pastel palette shared by the stability figures of the chapter
C_E1 = '#fbb4ae'   # pastel red    -- treatment failure (T-only)
C_E2 = '#b3cde3'   # pastel blue   -- treatment failure (T+M)
C_ES = '#ccebc5'   # pastel green  -- coexistence / LAS
C_E3 = '#decbe4'   # pastel purple -- TAM-free controlled equilibrium
C_LC = '#fbb4ae'   # pastel red    -- limit cycle (unstable/oscillatory regime)
C_SPIRAL = '#ccebc5'   # pastel green (light)  -- E* stable, damped-oscillation spiral
C_SINK = '#66b26c'     # pastel green (deeper) -- E* stable, monotone sink
C_SUB = '#8c2d04'      # dark brown-orange     -- subcritical stretch of the Hopf boundary


# ---------------------------------------------------------------------------
# (1) (alpha2, alpha3) plane, mu3 = 0, beta1 = 0
# ---------------------------------------------------------------------------
def diagram_a2_a3():
    k3 = 0.35
    a2min = k2 / K1
    a3min = k3 * (1 + K3 / K1)
    a3max = lambda a2: k3 * (1 + a2 * K3 / k2)

    a2 = np.logspace(-11, -6, 900)
    a3 = np.linspace(0.0, 3.0, 900)
    A2, A3 = np.meshgrid(a2, a3)

    Z = np.zeros(A2.shape)
    Z[(A2 < a2min) & (A3 < a3min)] = 1
    Z[(A2 > a2min) & (A3 < a3min)] = 2
    Z[(A3 > a3min) & (A3 < a3max(A2))] = 3
    Z[A3 > np.maximum(a3min, a3max(A2))] = 4

    cmap = ListedColormap([C_E1, C_E1, C_E2, C_ES, C_E3])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.pcolormesh(A2, A3, Z, cmap=cmap, vmin=0, vmax=4, shading='auto')

    ax.axvline(a2min, color='k', ls=':', lw=1.5)
    ax.axhline(a3min, color='k', ls='--', lw=1.5)
    a2c = a2[a2 >= a2min]
    ax.plot(a2c, a3max(a2c), 'k-.', lw=1.7)

    ax.plot(1e-7, 0.9, marker='*', color='crimson', ms=17, mec='k', mew=0.8, zorder=5)

    ax.set_xscale('log')
    ax.set_xlim(a2.min(), a2.max())
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(r'$\alpha_2$  (cell$\cdot$day)$^{-1}$, log scale')
    ax.set_ylabel(r'$\alpha_3$  (day$^{-1}$)')

    leg = [
        Patch(fc=C_E1, ec='k', lw=0.6, label=r'$E_1=(K_1,0,0)$ (failure)'),
        Patch(fc=C_E2, ec='k', lw=0.6, label=r'$E_2=(K_1,M^*,0)$ (failure)'),
        Patch(fc=C_ES, ec='k', lw=0.6, label=r'$E^*$ (coexistence)'),
        Patch(fc=C_E3, ec='k', lw=0.6, label=r'$E_3=(\bar T,0,\bar C)$ (TAM-free control)'),
        Line2D([0], [0], marker='*', color='none', markerfacecolor='crimson',
               markeredgecolor='k', markersize=13, label='reference point'),
    ]
    ax.legend(handles=leg, loc='upper left', framealpha=0.95, borderpad=0.7,
              handletextpad=0.6, labelspacing=0.55)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'stability_a2_a3.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

    mask_E3 = (Z == 4)
    Tbar = np.where(A3 > k3, k3 * K3 / (A3 - k3), np.nan)
    las_E3 = (Tbar > 0) & (Tbar < np.minimum(K1, k2 / A2))
    n = mask_E3.sum(); ok = (mask_E3 & las_E3).sum()
    print(f"(1) stability_a2_a3.png | a2min={a2min:.2e} a3min={a3min:.4f} | "
          f"E3 locally asymptotically stable in the E3 region: {ok}/{n} ({100*ok/max(n,1):.2f}%)")


# ---------------------------------------------------------------------------
# (2) (kappa3, beta1) plane: Hopf boundary and kappa3^c
# ---------------------------------------------------------------------------
def _hopf_helpers():
    a2, a3 = 1e-7, 0.9
    Tstar = lambda k3: k3 * K3 / (a3 - k3)
    Mstar = lambda k3: K2 * (1 - k2 * (a3 - k3) / (k3 * a2 * K3))

    def Q_and_D(b1, k3):
        """Q(beta1) = q1 q2 - q3 (Routh-Hurwitz stability criterion) and the discriminant D of the cubic
        characteristic polynomial lambda^3 + q1 lambda^2 + q2 lambda + q3 (D < 0: a complex-conjugate pair
        and a real eigenvalue, i.e. a spiral; D >= 0: three real eigenvalues, i.e. a sink). Vectorized in
        k3 and b1."""
        Ts, Ms = Tstar(k3), Mstar(k3)
        Cs = (a1 / u1) * (1 - Ts / K1) * (1 + b1 * Ms)
        a11 = -(a1 / K1) * Ts * (1 + b1 * Ms); a12 = a1 * Ts * (1 - Ts / K1) * b1
        a13 = -u1 * Ts; a21 = a2 * Ms * (1 - Ms / K2)
        a22 = -k2 + a2 * Ts * (1 - 2 * Ms / K2)
        a31 = a3 * Cs * K3 / (K3 + Ts) ** 2
        q1 = -(a11 + a22); q2 = a11 * a22 - a12 * a21 - a13 * a31
        q3 = a13 * a22 * a31
        Q = q1 * q2 - q3
        p, q, r = q1, q2, q3
        D = 18 * p * q * r - 4 * p ** 3 * r + p ** 2 * q ** 2 - 4 * q ** 3 - 27 * r ** 2
        return Q, D

    def Q(b1, k3):
        return Q_and_D(b1, k3)[0]

    def coef_b2(k3):
        bs = np.array([0.0, 1e-9, 2e-9])
        return np.polyfit(bs, [Q(b, k3) for b in bs], 2)[0]

    return a2, a3, Tstar, Mstar, Q, Q_and_D, coef_b2


# ---------------------------------------------------------------------------
# First Lyapunov coefficient along the Hopf boundary (criticality of the bifurcation)
# ---------------------------------------------------------------------------
_L1 = {}


def _l1_machinery():
    """Lambdified Jacobian and second and third derivative tensors of the field, in state variables
    rescaled by K1. The rescaling is exact and harmless, since x -> s x gives g21 -> s^2 g21: the sign
    of l1 is untouched and its value is brought to order unity. Only the sign is intrinsic, the
    magnitude depending on the normalization of the critical eigenvector (Appendix A)."""
    if not _L1:
        a2, a3 = 1e-7, 0.9
        T, M, C, b1, k3 = sp.symbols('T M C b1 k3')
        X = (T, M, C)
        Ts, Ms, Cs = K1 * T, K1 * M, K1 * C
        F = sp.Matrix([
            a1 * Ts * (1 - Ts / K1) * (1 + b1 * Ms) - u1 * Cs * Ts,
            a2 * Ms * Ts * (1 - Ms / K2) - k2 * Ms,
            a3 * Cs * Ts / (K3 + Ts) - k3 * Cs,
        ]) / K1
        args = (T, M, C, b1, k3)
        _L1['J'] = sp.lambdify(args, F.jacobian(X), 'numpy')
        _L1['D2'] = sp.lambdify(args, [[[sp.diff(F[i], X[j], X[k]) for k in range(3)]
                                        for j in range(3)] for i in range(3)], 'numpy')
        _L1['D3'] = sp.lambdify(args, [[[[sp.diff(F[i], X[j], X[k], X[m]) for m in range(3)]
                                         for k in range(3)] for j in range(3)]
                                       for i in range(3)], 'numpy')
    return _L1


def l1_at(k3, b1):
    """First Lyapunov coefficient at the Hopf point (kappa3, beta1), by the projection method of
    Kuznetsov (Elements of Applied Bifurcation Theory, 3rd ed., Sec. 5.4),

        l1 = Re g21 / (2 omega),
        g21 = <p,C(q,q,qb)> + 2 <p,B(q,h11)> + <p,B(qb,h20)>,
        A h11 = -B(q,qb),   (2 i omega I - A) h20 = B(q,q),

    with A q = i omega q, A^T p = -i omega p and <q,q> = <p,q> = 1. A negative value is the
    supercritical case, in which the cycle born beyond the boundary is stable and opens continuously
    from zero amplitude; a positive value is the subcritical case, in which an unstable cycle exists
    before the boundary and the loss of stability is abrupt. In the K1 state units set above."""
    m = _l1_machinery()
    a2, a3 = 1e-7, 0.9
    Ts = k3 * K3 / (a3 - k3)
    Ms = K2 * (1 - k2 * (a3 - k3) / (k3 * a2 * K3))
    Cs = (a1 / u1) * (1 - Ts / K1) * (1 + b1 * Ms)
    pt = (Ts / K1, Ms / K1, Cs / K1)
    A = np.asarray(m['J'](*pt, b1, k3), dtype=float)
    D2 = np.asarray(m['D2'](*pt, b1, k3), dtype=complex)
    D3 = np.asarray(m['D3'](*pt, b1, k3), dtype=complex)
    vals, vecs = np.linalg.eig(A)
    w = float(np.max(np.abs(vals.imag)))
    if w < 1e-12:
        return np.nan
    q = vecs[:, int(np.argmin(np.abs(vals - 1j * w)))]
    q = q / np.linalg.norm(q)
    valsT, vecsT = np.linalg.eig(A.T)
    p = vecsT[:, int(np.argmin(np.abs(valsT + 1j * w)))]
    p = p / np.conj(np.vdot(p, q))
    B = lambda x, y: np.einsum('ijk,j,k->i', D2, x, y)
    Cf = lambda x, y, z: np.einsum('ijkl,j,k,l->i', D3, x, y, z)
    qb = np.conj(q)
    h11 = -np.linalg.solve(A, B(q, qb))
    h20 = np.linalg.solve(2j * w * np.eye(3) - A, B(q, q))
    g21 = np.vdot(p, Cf(q, q, qb)) + 2 * np.vdot(p, B(q, h11)) + np.vdot(p, B(qb, h20))
    return g21.real / (2 * w)


def _find_Q_roots(Q, k3, bmax_search=1.0):
    """Positive roots of Q(., k3) = 0 (0, 1 or 2), from a log-scale scan followed by Brent's method on each
    sign change. Q(0, k3) > 0, so the number of roots sets the regime: none, E* stable for every beta1 >= 0;
    one, a Hopf threshold (unstable for every larger beta1); two, a closed instability window (unstable only
    between the roots, stable again above the larger one)."""
    f = lambda b: Q(b, k3)
    bs = np.logspace(-13, np.log10(bmax_search), 4000)
    vals = np.array([f(b) for b in bs])
    idx_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    # ABSOLUTE xtol: the scipy default (2e-12) is larger than beta_1* ~ 6e-11, which would
    # leave the root with up to 3% relative error. It must be far smaller than the root.
    return [brentq(f, bs[i], bs[i + 1], xtol=1e-28, rtol=8.9e-16) for i in idx_changes]


def diagram_k3_b1():
    a2, a3, Tstar, Mstar, Q, Q_and_D, coef_b2 = _hopf_helpers()

    k3_lo = brentq(lambda k: Tstar(k) - k2 / a2, 0.11, 0.6)
    k3_hi = brentq(lambda k: Tstar(k) - K1, 0.6, 0.899)

    B1_TOP = 1e-5          # the upper branch rises above 1e-6 near k3_c1
    B1_BOT = 1e-12
    B1_MAX_TAB = 3e-6      # teto de beta1 na Tabela 3.1

    # kappa3^(1): sign change of the leading coefficient of Q, a quadratic polynomial in beta1 (second
    # difference with a wide step, to avoid cancellation). It separates a simple threshold (unstable for
    # every larger beta1) from a closed instability window; it is not the end of the instability.
    def coef_a_direct(k3, h=1e-7):
        return (Q(2 * h, k3) - 2 * Q(h, k3) + Q(0.0, k3)) / (2 * h ** 2)
    k3_c1 = brentq(coef_a_direct, k3_lo + 1e-3, k3_hi - 1e-3)

    # kappa3^c: value above which no beta1 >= 0 is unstable, where the closed window of (k3_c1, k3_c) vanishes.
    # Q is an exact parabola in beta1, convex for k3 > k3_c1, whose minimum is negative while the window is
    # open; kappa3^c is where that minimum crosses zero.
    def min_Q(k3):
        res = minimize_scalar(lambda b: Q(b, k3), bounds=(0, 1e-4), method='bounded',
                               options={'xatol': 1e-16})
        return res.fun
    k3_c = brentq(min_Q, k3_c1, k3_hi - 1e-3)

    # Classification of the (kappa3, beta1) plane on a fine grid: Q >= 0 decides stability (Routh-Hurwitz)
    # and, inside the stable region, the cubic discriminant D decides spiral (D < 0) or sink (D >= 0). This
    # extends Proposition 3.7 (beta1 = 0) to the whole plane; for beta1 > 0 the boundary has no closed form.
    k3v = np.linspace(k3_lo + 1e-4, k3_hi - 1e-4, 1100)
    b1v = np.logspace(np.log10(B1_BOT), np.log10(B1_TOP), 1100)
    K3G, B1G = np.meshgrid(k3v, b1v)
    Qg, Dg = Q_and_D(B1G, K3G)
    stable = Qg >= 0
    Z = np.zeros(K3G.shape)
    Z[~stable] = 1                       # limit cycle
    Z[stable & (Dg < 0)] = 2             # E* spiral
    Z[stable & (Dg >= 0)] = 3            # E* sink

    cmap = ListedColormap([C_LC, C_LC, C_SPIRAL, C_SINK])

    # Hopf boundary drawn explicitly, including the upper branch that closes the instability window for
    # k3 in (k3_c1, k3_c).
    k3_lower = np.linspace(k3_lo + 1e-3, k3_c - 1e-4, 500)
    b1_lower, b1_upper = [], []
    k3_upper = []
    for k in k3_lower:
        rts = _find_Q_roots(Q, k, bmax_search=1e-3)
        if len(rts) >= 1:
            b1_lower.append(rts[0])
        else:
            b1_lower.append(np.nan)
        if len(rts) == 2:
            k3_upper.append(k)
            b1_upper.append(rts[1])
    b1_lower = np.array(b1_lower)
    k3_upper = np.array(k3_upper)
    b1_upper = np.array(b1_upper)

    # Criticality of the boundary: l1 < 0 (supercritical, a stable cycle opens continuously from the
    # threshold) or l1 > 0 (subcritical, an unstable cycle exists before it and the transition is abrupt).
    # The two are separated by a Bautin point, where l1 = 0 and the Hopf bifurcation is degenerate.
    # Layout: the map (a) and its magnification (b) on the left, the first Lyapunov coefficient (c) on
    # the right, and a single legend below the three panels, so that nothing is annotated inside them.
    l1_lower = np.array([l1_at(k, b) if np.isfinite(b) else np.nan
                         for k, b in zip(k3_lower, b1_lower)])
    l1_upper = np.array([l1_at(k, b) for k, b in zip(k3_upper, b1_upper)])

    def l1_of_k3(k):
        return l1_at(k, _find_Q_roots(Q, k, bmax_search=1e-3)[0])

    sgn = np.sign(l1_lower)
    jump = np.where((sgn[:-1] > 0) & (sgn[1:] < 0))[0]
    if len(jump):
        j = jump[0]
        k3_bautin = brentq(l1_of_k3, k3_lower[j], k3_lower[j + 1], xtol=1e-10)
        b1_bautin = _find_Q_roots(Q, k3_bautin, bmax_search=1e-3)[0]
    else:
        k3_bautin = b1_bautin = np.nan

    sup = l1_lower < 0
    SUB = dict(color=C_SUB, ls=(0, (3.5, 1.4)), lw=2.6)
    DASH_C = (0, (6, 3))            # kappa_3^c
    DOT_B = (0, (1, 2))             # kappa_3^B
    KZ0, KZ1 = 0.148, 0.218         # janela ampliada no painel (b)
    BZ0, BZ1 = 2.5e-11, 1.7e-10

    # Times New Roman (the body font of the thesis) with STIX math, which matches it.
    # Applied to this figure only: the rcParams are restored after saving.
    _rc_old = {k: plt.rcParams[k] for k in ('font.family', 'mathtext.fontset')}
    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})
    fig = plt.figure(figsize=(11.2, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.85, 1.0], height_ratios=[1.30, 1.0],
                          hspace=0.34, wspace=0.26, bottom=0.11, top=0.975,
                          left=0.07, right=0.985)
    ax = fig.add_subplot(gs[0, 0])
    axz = fig.add_subplot(gs[1, 0])
    axl = fig.add_subplot(gs[:, 1])

    def letra(a, txt, nota=None):
        # label inside the panel, in the upper left corner and in bold, as in the other
        # figures of the thesis. No white box behind it; the note goes to the opposite
        # corner, where the map is of a single color, so as not to fall on the curves.
        a.text(0.022, 0.955, txt, transform=a.transAxes, fontsize=12.5, va='top', ha='left',
               fontweight='bold', zorder=10)
        if nota:
            a.text(0.982, 0.955, nota, transform=a.transAxes, fontsize=9.5, va='top',
                   ha='right', color='0.15', zorder=10)

    def boundary(a, lw_sup=2.0):
        a.plot(np.where(sup, k3_lower, np.nan), np.where(sup, b1_lower, np.nan),
               'k-', lw=lw_sup, zorder=4)
        a.plot(np.where(~sup, k3_lower, np.nan), np.where(~sup, b1_lower, np.nan),
               zorder=5, **SUB)

    # ---------------- (a) the whole plane ----------------
    ax.set_yscale('log'); ax.set_ylim(B1_BOT, B1_TOP); ax.set_xlim(k3_lo, k3_hi)
    ax.pcolormesh(K3G, B1G, Z, cmap=cmap, vmin=0, vmax=3, shading='auto')
    # the spiral/sink boundary is not a bifurcation, only a switch from complex to real
    # eigenvalues, so it is drawn as a thin stroke and not like the thick Hopf curves.
    ax.contour(K3G, B1G, Z, levels=[2.5], colors=['0.30'], linewidths=1.1, zorder=4)
    boundary(ax)
    ax.plot(k3_upper, b1_upper, 'k-', lw=2.0, zorder=4)
    ax.axvline(k3_c, color='k', ls=DASH_C, lw=1.5)
    ax.plot(0.35, 5.9056e-11, marker='o', color='crimson', ms=8.5, mec='k', mew=0.8, zorder=6)
    ax.add_patch(plt.Rectangle((KZ0, BZ0), KZ1 - KZ0, BZ1 - BZ0, fill=False,
                               ec='0.25', lw=1.1, zorder=7))
    ax.set_xlabel(r'$\kappa_3$  (day$^{-1}$)', fontsize=11)
    ax.set_ylabel(r'$\beta_1$  (cell$^{-1}$)', fontsize=11)
    letra(ax, 'a)')

    # ---------------- (b) the subcritical corner ----------------
    axz.set_yscale('log'); axz.set_xlim(KZ0, KZ1); axz.set_ylim(BZ0, BZ1)
    axz.pcolormesh(K3G, B1G, Z, cmap=cmap, vmin=0, vmax=3, shading='auto')
    boundary(axz, lw_sup=2.2)
    if np.isfinite(k3_bautin):
        axz.axvline(k3_bautin, color=C_SUB, ls=DOT_B, lw=1.2)
        axz.plot(k3_bautin, b1_bautin, marker='s', color='w', ms=9.5, mec=C_SUB, mew=2.0,
                 zorder=8)
    axz.set_yticks([3e-11, 5e-11, 1e-10])
    axz.set_yticklabels([r'$3\times10^{-11}$', r'$5\times10^{-11}$', r'$10^{-10}$'])
    axz.yaxis.set_minor_formatter(plt.NullFormatter())
    axz.set_xlabel(r'$\kappa_3$  (day$^{-1}$)', fontsize=11)
    axz.set_ylabel(r'$\beta_1$  (cell$^{-1}$)', fontsize=11)
    letra(axz, 'b)', nota='magnification of the rectangle in a)')

    # ---------------- (c) the Lyapunov coefficient ----------------
    axl.set_xlim(k3_lo, 0.672)
    if np.isfinite(k3_bautin):
        axl.axvspan(k3_lo, k3_bautin, color=C_SUB, alpha=0.12, lw=0)
        axl.axvline(k3_bautin, color=C_SUB, ls=DOT_B, lw=1.2)
    axl.axhline(0.0, color='0.45', lw=0.9)
    axl.plot(np.where(sup, k3_lower, np.nan), np.where(sup, l1_lower, np.nan), 'k-', lw=2.0)
    axl.plot(np.where(~sup, k3_lower, np.nan), np.where(~sup, l1_lower, np.nan), **SUB)
    axl.plot(0.35, l1_at(0.35, 5.9056050e-11), marker='o', color='crimson', ms=8.5, mec='k',
             mew=0.8, zorder=6)
    if np.isfinite(k3_bautin):
        axl.plot(k3_bautin, 0.0, marker='s', color='w', ms=9.5, mec=C_SUB, mew=2.0, zorder=8)
    axl.set_yscale('symlog', linthresh=1e-2, linscale=0.55)
    axl.set_ylim(-6e2, 6e3)
    axl.set_yticks([-100, -1, -0.01, 0.01, 1, 100])
    axl.set_yticklabels([r'$-10^{2}$', r'$-1$', r'$-10^{-2}$', r'$10^{-2}$', r'$1$', r'$10^{2}$'])
    axl.set_xlabel(r'$\kappa_3$  (day$^{-1}$)', fontsize=11)
    axl.set_ylabel('first Lyapunov coefficient  $K_1^{2}\\,\\ell_1$', fontsize=11)
    letra(axl, 'c)')

    # ---------------- a single legend, below the three panels ----------------
    import matplotlib.colors as mcolors
    _rgb = mcolors.to_rgb(C_SUB)
    C_BAND = tuple(0.12*c + 0.88 for c in _rgb)      # C_SUB at alpha 0.12 over white
    leg = [
        Patch(fc=C_LC, ec='0.4', lw=0.6, label='sustained oscillations ($E^*$ unstable)'),
        Patch(fc=C_SPIRAL, ec='0.4', lw=0.6, label='$E^*$ stable, damped-oscillation spiral'),
        Patch(fc=C_SINK, ec='0.4', lw=0.6, label='$E^*$ stable, monotone sink'),
        Line2D([0], [0], color='k', lw=2.0,
               label=r'Hopf boundary, supercritical ($\ell_1<0$)'),
        Line2D([0], [0], color=C_SUB, lw=2.6, ls=(0, (3.5, 1.4)),
               label=r'Hopf boundary, subcritical ($\ell_1>0$)'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='w', markeredgecolor=C_SUB,
               markeredgewidth=1.8, markersize=9,
               label=r'Bautin point $\kappa_3^{\,B}\approx%.3f$, where $\ell_1=0$' % k3_bautin),
        Line2D([0], [0], color='k', lw=1.5, ls=DASH_C,
               label=r'$\kappa_3^{\,c}\approx%.3f$, instability window closes' % k3_c),
        Line2D([0], [0], color=C_SUB, lw=1.2, ls=DOT_B,
               label=r'$\kappa_3^{\,B}$, marked in panels b) and c)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='crimson',
               markeredgecolor='k', markersize=8.5,
               label=r'$\beta_1^*\approx5.91\times10^{-11}$ at $\kappa_3=0.35$'),
        Patch(fc=C_BAND, ec='0.6', lw=0.5,
              label=r'subcritical stretch, $\kappa_3<\kappa_3^{\,B}$'),
        Line2D([0], [0], color='0.30', lw=1.1, label='spiral/sink boundary'),
        Patch(fc='none', ec='0.25', lw=1.1, label='region magnified in panel b)'),
    ]
    fig.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -0.015), ncol=3,
               frameon=True, framealpha=1.0, edgecolor='0.75', borderpad=0.8,
               handletextpad=0.7, labelspacing=0.65, columnspacing=1.8, fontsize=9.8)

    fig.savefig(os.path.join(OUT, 'hopf_k3_b1.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(_rc_old)

    # check: as beta1 -> 0 the numerical spiral/sink boundary must match T* = T~ of Proposition 3.7
    Dt = lambda T: T**3 + 2*K3*T**2 + (K3 + 4*a3/a1*K1)*K3*T - 4*a3/a1*K1**2*K3
    Ttil = brentq(Dt, 1e3, K1)
    k3_spiral_analytic = brentq(lambda k: Tstar(k) - Ttil, k3_lo, k3_hi)
    row0 = Z[0, :]
    idx_transition = np.where(np.diff(row0) != 0)[0]
    k3_transition_numeric = k3v[idx_transition[-1]] if len(idx_transition) else np.nan
    print(f"(2) hopf_k3_b1.png | E* exists for k3 in ({k3_lo:.3f},{k3_hi:.3f})")
    print(f"    k3_c1 (the instability window becomes closed) = {k3_c1:.4f}")
    print(f"    k3_c  (the window closes; no instability above) = {k3_c:.4f}")
    print(f"    spiral/sink check at beta1->0: Proposition 3.7 gives k3={k3_spiral_analytic:.4f}; "
          f"numerical grid gives k3~={k3_transition_numeric:.4f} (must agree)")
    print(f"    Bautin point (l1 = 0) at k3={k3_bautin:.5f}, beta1*={b1_bautin:.6e}")
    print(f"    K1^2 l1 = {l1_at(0.35, 5.9056050e-11):+.5f} at k3=0.35 (supercritical) and "
          f"{l1_lower[0]:+.2f} at k3={k3_lower[0]:.4f} (subcritical)")
    print(f"    l1 < 0 on the whole upper branch: {bool(np.all(l1_upper < 0))}")


if __name__ == '__main__':
    diagram_a2_a3()
    diagram_k3_b1()
