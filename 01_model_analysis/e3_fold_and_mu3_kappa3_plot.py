"""Thesis figures of Section 3.3.1 (steady states and their local asymptotic stability), drawn from
the data written by e3_fold_and_mu3_kappa3_data.py:

    figures/E3_fold_bifurcation.png   saddle-node diagram of the branches of E3 versus mu3
    figures/mu3_kappa3_map.png        (kappa3, mu3) regions where E2 or E3 is locally asymptotically stable
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, 'figures')
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

C_E2 = '#b3cde3'
C_E3 = '#decbe4'
C_OPEN = '#e5e5e5'
C_STABLE = '#c0392b'
C_UNSTABLE = '#7f8c8d'
C_INADM = '#b9b9b9'  # branch continued outside the positive orthant
C_MARK = '#1f5fa8'   # blue: crimson is already the marker colour of Figure 3.2

# ---------------------------------------------------------------------------
# Saddle-node diagram of the branches of E3 versus mu3
# ---------------------------------------------------------------------------
dA = np.load(os.path.join(RESULTS, 'figA_data.npz'))
mu3 = dA['mu3_grid']; T1 = dA['T1']; T2 = dA['T2']
T1_stable = dA['T1_stable']; mu3max = float(dA['mu3max'])
mu3_cross = float(dA['mu3_cross']); T0 = float(dA['T0']); kappa3_A = float(dA['kappa3'])

# Two panels: the two bifurcations that organize the equilibria of the model. (a) the transcritical
# exchange between E1 and E2 at alpha2min, (b) the saddle-node of E3 at mu3max. Both panels share one
# encoding: solid red = locally asymptotically stable, gray = unstable, blue marker = bifurcation point.
fig, (axA, ax) = plt.subplots(1, 2, figsize=(13.0, 4.3))

# ---- panel (a): transcritical bifurcation E1 <-> E2 in alpha2 -------------------------------------
# Proposition on E1: LAS iff alpha2 < alpha2min (and mu3 > g3(K1)); proposition on E2: exists and is LAS
# iff alpha2 > alpha2min (and mu3 > g3(K1)). At alpha2min the macrophage coordinate of E2 vanishes and
# E2 coincides with E1: the two branches cross and exchange stability.
K1_A, K2_A, kappa2_A = 1e9, 1e9, 0.34
alpha2min = kappa2_A / K1_A
a2 = np.logspace(np.log10(alpha2min) - 1.2, np.log10(alpha2min) + 3.2, 600)
Mbar = K2_A * (1.0 - kappa2_A / (a2 * K1_A))      # macrophage coordinate of E2
feasible = a2 >= alpha2min                        # below alpha2min it leaves the positive orthant

# E1 branch (M = 0): stable before the crossing, unstable after
axA.plot(a2[a2 <= alpha2min], np.zeros((a2 <= alpha2min).sum()) / 1e9,
         color=C_STABLE, lw=2.4, solid_capstyle='butt')
axA.plot(a2[a2 >= alpha2min], np.zeros((a2 >= alpha2min).sum()) / 1e9,
         color=C_UNSTABLE, lw=2.0, ls='--')
# E2 branch, stable wherever it is biologically admissible
axA.plot(a2[feasible], Mbar[feasible] / 1e9, color=C_STABLE, lw=2.4)
# short continuation below alpha2min, where M < 0. It is not a biological state, but without it only one
# arm of the crossing is visible and the transcritical reads as a single curve bending upwards
Y_FLOOR = -0.28
arm = (~feasible) & (Mbar / 1e9 >= Y_FLOOR)
axA.plot(a2[arm], Mbar[arm] / 1e9, color=C_INADM, lw=1.8, ls=':')
axA.plot(alpha2min, 0.0, marker='s', color=C_MARK, ms=9, mec='k', mew=0.6, zorder=6)

axA.set_xscale('log')
axA.set_xlabel(r'$\alpha_2$  (cells$\times$day)$^{-1}$, log scale')
axA.set_ylabel(r'$\bar M$  ($\times10^{9}$ cells)')
axA.set_xlim(a2[0], a2[-1])
axA.set_ylim(Y_FLOOR - 0.04, 1.08)
legA = [
    Line2D([0], [0], color=C_STABLE, lw=2.4, label=r'locally asymptotically stable'),
    Line2D([0], [0], color=C_UNSTABLE, lw=2.0, ls='--', label=r'$E_1$ unstable'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor=C_MARK, markeredgecolor='k',
           markersize=9, label=r'transcritical point $\alpha_{2\min}=\kappa_2/K_1$'),
    Line2D([0], [0], color=C_INADM, lw=1.8, ls=':', label=r'$E_2$ with $\bar M<0$ (not admissible)'),
]
# centre right is the only region free of both branches: the rising curve occupies the upper left
axA.legend(handles=legA, loc='center right', framealpha=0.95, borderpad=0.7,
           handletextpad=0.6, labelspacing=0.45, fontsize=9.0)
axA.text(0.02, 0.97, 'a)', transform=axA.transAxes, fontsize=13, va='top', ha='left',
         fontweight='bold')
axA.text(0.03, 0.27, r'$E_1=(K_1,0,0)$', transform=axA.transAxes, fontsize=10)
# placed where the branch has already flattened: further left the curve is still rising and runs through
# the label
axA.text(0.62, 0.79, r'$E_2=(K_1,\bar M,0)$', transform=axA.transAxes, fontsize=10)

# ---- panel (b): saddle-node of E3 in mu3 -----------------------------------------------------------
ax.set_yscale('log')
ax.set_xscale('log')

# branch T1: stable up to mu3_cross, unstable afterwards
m1 = mu3 <= mu3_cross
ax.plot(mu3[m1], T1[m1], color=C_STABLE, lw=2.4,
        label=r'$\bar T_1$ (locally asymptotically stable)')
m2 = mu3 >= mu3_cross
ax.plot(mu3[m2], T1[m2], color=C_UNSTABLE, lw=2.0, ls='--',
        label=r'$\bar T_1$ (unstable)')
ax.plot(mu3, T2, color=C_UNSTABLE, lw=2.0, ls=':',
        label=r'$\bar T_2$ (unstable)')

ax.plot(mu3max, T0, marker='D', color=C_MARK, ms=9, mec='k', mew=0.6, zorder=6)
ax.plot(mu3_cross, np.interp(mu3_cross, mu3, T1), marker='o', color=C_MARK,
        ms=8, mec='k', mew=0.6, zorder=6)

ax.set_xlabel(r'$\mu_3$  (cells$\times$day)$^{-1}$, log scale')
ax.set_ylabel(r'$\bar T$  (cells), log scale')
# the T2 branch reaches the top-left corner, where the panel letter goes: a little headroom frees the
# corner without moving any curve
ax.set_ylim(top=ax.get_ylim()[1] * 4)

leg = [
    Line2D([0], [0], color=C_STABLE, lw=2.4, label=r'$\bar T_1$ (locally asymptotically stable)'),
    Line2D([0], [0], color=C_UNSTABLE, lw=2.0, ls='--', label=r'$\bar T_1$ (unstable)'),
    Line2D([0], [0], color=C_UNSTABLE, lw=2.0, ls=':', label=r'$\bar T_2$ (always unstable)'),
    Line2D([0], [0], marker='D', color='none', markerfacecolor=C_MARK, markeredgecolor='k',
           markersize=9, label=r'fold point $(\mu_{3\max},\bar T_0)$'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=C_MARK, markeredgecolor='k',
           markersize=8, label=r'stability change $\bar T_1=\kappa_2/\alpha_2$'),
]
# upper right: with the reduced height, the upper left is where the T2 branch runs
ax.legend(handles=leg, loc='upper right', framealpha=0.95, borderpad=0.7,
          handletextpad=0.6, labelspacing=0.45, fontsize=9.0)
ax.text(0.02, 0.97, 'b)', transform=ax.transAxes, fontsize=13, va='top', ha='left',
        fontweight='bold')
# branch labels mirroring panel (a). Both branches belong to E3, so they carry the chapter's own
# notation for the two roots rather than a single ambiguous "E3"
# the dotted branch falls steeply, so the label sits as high as its right end allows without touching it
ax.text(0.04, 0.62, r'$E_{32}=(\bar T_2,0,\bar C_2)$', transform=ax.transAxes, fontsize=10)
ax.text(0.05, 0.11, r'$E_{31}=(\bar T_1,0,\bar C_1)$', transform=ax.transAxes, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'E3_fold_bifurcation.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print("figures/E3_fold_bifurcation.png written.")

# ---------------------------------------------------------------------------
# (kappa3, mu3) map of E2 and E3 (alpha2, alpha3 fixed at Table 3.1)
# ---------------------------------------------------------------------------
dB = np.load(os.path.join(RESULTS, 'figB_data.npz'))
k3g = dB['kappa3_grid']; m3g = dB['mu3_grid']; Z = dB['Z']
bnd_E3 = dB['bnd_E3']; bnd_E2 = dB['bnd_E2']
k3c_E3 = float(dB['kappa3_c_E3']); k3c_E2 = float(dB['kappa3_c_E2'])

# Z in {0,1,2,3}: 0 unclassified, 1 E2 only, 2 E3 only, 3 both (bistable).
# E2 and E3 are tested independently, so the overlap is shown as its own class instead of being
# absorbed into E3 by whichever test happened to run first.
C_BOTH = '#c9a9c4'
cmap = ListedColormap([C_OPEN, C_E2, C_E3, C_BOTH])

fig, ax = plt.subplots(figsize=(9.6, 5.8))
K3G, M3G = np.meshgrid(k3g, m3g)
ax.pcolormesh(K3G, M3G, Z, cmap=cmap, vmin=-0.5, vmax=3.5, shading='auto')
ax.set_yscale('log')
ax.set_xlim(k3g[0], k3g[-1])
ax.set_ylim(m3g[0], m3g[-1])
ax.set_xlabel(r'$\kappa_3$  (day$^{-1}$)')
ax.set_ylabel(r'$\mu_3$  (cells$\times$day)$^{-1}$, log scale')

# Both region boundaries are the SAME closed form mu3 = g3(T, kappa3) = (f3(T) - kappa3)/T, read at the
# two tumour sizes that matter: T = kappa2/alpha2, above which E3 loses stability, and T = K1, above
# which E2 gains it. Each one vanishes at kappa3 = f3(T), which is where its region closes.
ax.plot(k3g, bnd_E3, color='#5b2d6e', lw=2.0, zorder=4)
ax.plot(k3g, bnd_E2, color='#1f5fa8', lw=2.0, ls='--', zorder=4)
ax.axvline(k3c_E3, color='#5b2d6e', lw=1.2, ls=':', zorder=4)
ax.axvline(k3c_E2, color='#1f5fa8', lw=1.2, ls=':', zorder=4)

ax.annotate(r'$\kappa_3=f_3(\kappa_2/\alpha_2)\approx' + ('%.3f$' % k3c_E3),
            xy=(k3c_E3 + 0.014, 1.5e-11), fontsize=9.5, color='#5b2d6e',
            va='bottom', ha='left')
ax.annotate(r'$\kappa_3=f_3(K_1)\approx' + ('%.3f$' % k3c_E2),
            xy=(k3c_E2 - 0.022, 1.5e-11), fontsize=9.5, color='#1f5fa8',
            va='bottom', ha='right')

ax.text(0.062, 2.3e-9, 'bistable', fontsize=10.5, ha='center', va='center', style='italic')
ax.text(0.062, 6e-11, r'$E_3$ only', fontsize=10.5, ha='center', va='center')
ax.text(0.50, 2.3e-9, r'$E_2$ only', fontsize=10.5, ha='center', va='center')
ax.text(0.50, 4.5e-11, r'neither: $E^*$ is the candidate', fontsize=10.5, ha='center', va='center')

leg = [
    Patch(fc=C_E3, ec='k', lw=0.6, label=r'$E_3=(\bar T_1,0,\bar C_1)$ only (TAM-free control)'),
    Patch(fc=C_BOTH, ec='k', lw=0.6, label=r'$E_2$ and $E_3$ both LAS (bistable)'),
    Patch(fc=C_E2, ec='k', lw=0.6, label=r'$E_2=(K_1,M^*,0)$ only (treatment failure)'),
    # no proposition numbers inside the image: they are rasterized here and would silently go stale if
    # a proposition were inserted earlier in the chapter. The precise citation lives in the LaTeX
    # caption, where \ref keeps it in step with the numbering.
    Patch(fc=C_OPEN, ec='k', lw=0.6, label=r'neither classified ($E^*$ the candidate)'),
    Line2D([0], [0], color='#5b2d6e', lw=2.0,
           label=r'$\mu_3=g_3(\kappa_2/\alpha_2)$: upper edge of $E_3$'),
    Line2D([0], [0], color='#1f5fa8', lw=2.0, ls='--',
           label=r'$\mu_3=g_3(K_1)$: lower edge of $E_2$'),
]
ax.legend(handles=leg, loc='upper right', framealpha=0.96, borderpad=0.7,
          handletextpad=0.6, labelspacing=0.45, fontsize=9.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'mu3_kappa3_map.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print("figures/mu3_kappa3_map.png written.")
print("  kappa3_c(E3) = %.5f   kappa3_c(E2) = %.5f" % (k3c_E3, k3c_E2))
print("  E2 boundary over the domain: %.3e .. %.3e" % (bnd_E2.min(), bnd_E2.max()))
