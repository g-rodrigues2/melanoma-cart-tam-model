"""Scenario 4: re-runs the Section-3.4.1 mu3-threshold bisection (same numerical
procedure as Figure fig_numeric) but replacing two Table-3.1 literature rates by
the values directly calibrated against the real Jilani et al. B16 data in
Section 3.5 (long-hinge group): alpha1 (bare logistic growth, 0.69 -> 0.2648,
the PBS-control exponential rate r_B16, maximum-likelihood value fixed in the treatment models) and kappa3 (CAR T-cell apoptosis rate,
0.35 -> 0.09, from the persistence-anchored reduced model, eq:data:reduced).
beta1 and mu1 are left at their Table 1 (literature) values: beta1 is not
identifiable from these data at any value (Section sec:data:practical), and
mu1's calibrated value lives in different (persistence-normalized) units,
not directly transferable to the cells-based full system (C in absolute
cells here, not the normalized C(0)=1 of the reduced model) -- only its
order of magnitude is corroborated (Section sec:data:practical).
"""
import os

os.makedirs("figures", exist_ok=True)

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    # sizes chosen so that, scaled to the 6.30 in text width, the labels land between 8 and 10 pt
    'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 12, 'axes.linewidth': 1.0,
})

# fixed (Table 1, not touched by this comparison)
K1 = 1e9
alpha2 = 1e-7
K2 = 1e9
kappa2 = 0.34
alpha3 = 0.9
K3 = 2e7
mu1 = 1e-7
beta1 = 2.3e-10
T0, M0, C0 = 2e6, 1e6, 5e5

def rhs(y, t, alpha1, kappa3, mu3):
    T, M, C = y
    dT = alpha1*T*(1-T/K1)*(1+beta1*M) - mu1*C*T
    dM = alpha2*M*T*(1-M/K2) - kappa2*M
    dC = alpha3*C*T/(K3+T) - mu3*C*T - kappa3*C
    return [dT, dM, dC]

def T60(alpha1, kappa3, mu3):
    t = np.linspace(0, 60, 601)
    sol = odeint(rhs, [T0, M0, C0], t, args=(alpha1, kappa3, mu3), rtol=1e-9, atol=1e-6)
    return sol[-1, 0], t, sol

def find_threshold(alpha1, kappa3, mu3_lo, mu3_hi, n=140):
    grid = np.linspace(mu3_lo, mu3_hi, n)
    Tvals = np.array([T60(alpha1, kappa3, m)[0] for m in grid])
    Tmid = 0.5*(Tvals.min() + Tvals.max())
    sign = np.sign(Tvals - Tmid)
    idx = np.where(np.diff(sign) != 0)[0]
    a, b = grid[idx[0]], grid[idx[0]+1]
    f = lambda m: T60(alpha1, kappa3, m)[0] - Tmid
    return brentq(f, a, b, xtol=1e-14), grid, Tvals

# literature baseline (reproduces the 6.75e-10 already in the thesis text, sanity check)
mu3_lit, grid_lit, T_lit = find_threshold(0.69, 0.35, 6e-10, 7.5e-10)
# data-informed (alpha1 = r_B16 from PBS controls, kappa3 from persistence data)
mu3_data, grid_data, T_data = find_threshold(0.2648, 0.09, 1e-11, 2e-9, n=200)

print('mu3_lit threshold  :', mu3_lit)
print('mu3_data threshold :', mu3_data)
print('ratio data/lit      :', mu3_data/mu3_lit)

# time series at each threshold, to compare tumor trajectories directly
_, t_lit, sol_lit = T60(0.69, 0.35, mu3_lit)
_, t_data, sol_data = T60(0.2648, 0.09, mu3_data)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
ax.plot(grid_lit*1e10, T_lit, color='#c0392b', lw=2, marker='o', ms=3,
        label=r'literature rates ($\alpha_1=0.69$, $\kappa_3=0.35$)')
ax.axvline(mu3_lit*1e10, color='#c0392b', ls='--', lw=1.3)
ax.plot(grid_data*1e10, T_data, color='#2a9d5c', lw=2, marker='o', ms=3,
        label=r'data-informed rates ($\alpha_1=0.2648$, $\kappa_3=0.09$)')
ax.axvline(mu3_data*1e10, color='#2a9d5c', ls='--', lw=1.3)
ax.set_yscale('log')
ax.set_xlabel(r'$\mu_3\ (\times10^{-10}$ (cells$\cdot$day)$^{-1})$')
ax.set_ylabel(r'$T(60)$ (cells), log scale')
ax.legend(loc='center left', fontsize=8.6, framealpha=0.95)
# panel label inside the axes, in bold, as in the published figures of this chapter
# (lowercase letter with a closing parenthesis); the description is offset in points
_anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
               fontsize=16)
ax.annotate('a)', fontweight='bold', **_anchor)
ax.annotate('immunosuppression threshold', xytext=(20, 0), textcoords='offset points', **_anchor)

ax2 = axes[1]
ax2.plot(t_lit, sol_lit[:, 0], color='#c0392b', lw=2, label='literature rates, at $\\mu_3^\\dagger$')
ax2.plot(t_data, sol_data[:, 0], color='#2a9d5c', lw=2, label='data-informed rates, at $\\mu_3^\\dagger$')
ax2.set_yscale('log')
ax2.set_xlabel('time (days)')
ax2.set_ylabel(r'$T(t)$ (cells), log scale')
ax2.legend(loc='lower right', fontsize=8.6, framealpha=0.95)
_anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
               fontsize=16)
ax2.annotate('b)', fontweight='bold', **_anchor)
ax2.annotate('tumor trajectory at threshold', xytext=(20, 0), textcoords='offset points', **_anchor)

fig.tight_layout()
fig.savefig(os.path.join('figures', 'scenario4_data_informed.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print('figure saved.')
