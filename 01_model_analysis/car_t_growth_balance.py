"""Growth and decay terms of the CAR T-cell equation (thesis Figure 3.2: figures/fig2.pdf).

Compares f3(T) = alpha3 T / (K3 + T) with the lines mu3 T + kappa3 for mu3 equal to, below and above
mu3max, and marks the tangency point T̄0 and the two roots T̄1, T̄2 obtained for mu3 < mu3max.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.linewidth': 1.0,
})

K3 = 2e7
alpha3 = 0.9
kappa3 = 0.35

def f3(T): return alpha3*T/(K3+T)
def g3(T): return (f3(T)-kappa3)/T
def Tbar0(): return np.sqrt(kappa3)/(np.sqrt(alpha3)-np.sqrt(kappa3))*K3
def mu3max(): return g3(Tbar0())

m3max = mu3max()
T0 = Tbar0()
m3_lower = m3max*0.5
m3_higher = m3max*1.5
print(f"mu3max={m3max:.4e}  T0={T0:.4e}")

Tzero = kappa3*K3/(alpha3-kappa3)
T1 = brentq(lambda T: g3(T)-m3_lower, Tzero*0.999, T0)
T2 = brentq(lambda T: g3(T)-m3_lower, T0, T0*30)
print(f"T1={T1:.4e}  T2={T2:.4e}")

Tv = np.linspace(1e5, 2.5e8, 2000)
fv = f3(Tv)
line_max = m3max*Tv + kappa3
line_lower = m3_lower*Tv + kappa3
line_higher = m3_higher*Tv + kappa3

fig, ax = plt.subplots(figsize=(8.6, 3.73))   # height reduced by a third, for a wider aspect ratio
ax.plot(Tv/1e8, fv, color='#1f5fa8', lw=2.2, label=r'$f_3(T)$')
ax.plot(Tv/1e8, line_max, color='k', ls='--', lw=1.8, label=r'$\mu_{3\max}T+\kappa_3$ (tangent)')
ax.plot(Tv/1e8, line_lower, color='k', ls='-.', lw=1.8, label=r'$\mu_3<\mu_{3\max}$ (two roots)')
ax.plot(Tv/1e8, line_higher, color='k', ls=':', lw=1.8, label=r'$\mu_3>\mu_{3\max}$ (no root)')

ax.plot(T0/1e8, f3(T0), marker='D', color='crimson', ms=10, mec='k', mew=0.6, zorder=6)
# T̄1 and T̄2 are the same kind of object -- the two intersections of the mu3 < mu3max case -- so they
# share one marker, which is also what the legend and the thesis caption state
ax.plot(T1/1e8, f3(T1), marker='o', color='crimson', ms=8, mec='k', mew=0.6, zorder=6)
ax.plot(T2/1e8, f3(T2), marker='o', color='crimson', ms=8, mec='k', mew=0.6, zorder=6)

ax.set_xlabel(r'Tumor cells  ($\times10^{8}$ cells)')
ax.set_ylabel(r'Growth/decay terms for CAR T-cells')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 1)

leg = [
    Line2D([0],[0], color='#1f5fa8', lw=2.2, label=r'$f_3(T)$'),
    Line2D([0],[0], color='k', ls='--', lw=1.8, label=r'$\mu_{3\max}T+\kappa_3$'),
    Line2D([0],[0], color='k', ls='-.', lw=1.8, label=r'$\mu_3<\mu_{3\max}$'),
    Line2D([0],[0], color='k', ls=':', lw=1.8, label=r'$\mu_3>\mu_{3\max}$'),
    Line2D([0],[0], marker='D', color='none', markerfacecolor='crimson', markeredgecolor='k',
           markersize=9, label=r'$\bar T_0$ (tangency)'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor='crimson', markeredgecolor='k',
           markersize=8, label=r'$\bar T_1,\bar T_2$ (two roots)'),
]
ax.legend(handles=leg, loc='lower right', framealpha=0.95, borderpad=0.7,
          handletextpad=0.6, labelspacing=0.5, fontsize=9.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2.pdf'), bbox_inches='tight')
plt.close(fig)
print("figures/fig2.pdf written.")
