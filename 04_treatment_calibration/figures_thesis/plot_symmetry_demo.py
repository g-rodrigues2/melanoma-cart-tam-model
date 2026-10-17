"""Numerical demonstration of the scaling symmetry of the T-M-C system (thesis equation eq:data:symM): the full
system is integrated with (beta1, M(0), K2) and with (beta1/lambda, lambda*M(0), lambda*K2) for several lambda.
T(t) and C(t) coincide while M(t) scales with lambda, so beta1 and M cannot be distinguished from tumor volumes
alone.
"""
import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from _paths import FIGURES

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'axes.linewidth': 1.0,
})

# reference parameters (Table 3.1), mu3 = 0 (does not affect the symmetry)
alpha1, K1 = 0.69, 1e9
mu1 = 1e-7
alpha3, K3, kappa3, mu3 = 0.9, 2e7, 0.35, 0.0
alpha2, kappa2 = 1e-7, 0.34

beta1_0 = 2e-9
K2_0 = 1e9
T0 = 2e6
M0_0 = 1e6
C0 = 5e5

def rhs(y, t, beta1, K2):
    T, M, C = y
    dT = alpha1 * T * (1 - T / K1) * (1 + beta1 * M) - mu1 * T * C
    dM = alpha2 * T * M * (1 - M / K2) - kappa2 * M
    dC = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    return [dT, dM, dC]

tt = np.linspace(0, 40, 400)
lambdas = [0.25, 1.0, 4.0]
colors = ['#4393c3', '#2a2a2a', '#c0392b']
styles = ['--', '-', ':']

fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

Tref = None
max_abs_diff = 0.0
for lam, color, ls in zip(lambdas, colors, styles):
    beta1_lam = beta1_0 / lam
    K2_lam = K2_0 * lam
    M0_lam = M0_0 * lam
    sol = odeint(rhs, [T0, M0_lam, C0], tt, args=(beta1_lam, K2_lam))
    T, M, C = sol[:, 0], sol[:, 1], sol[:, 2]
    if Tref is None:
        Tref = T.copy()
    else:
        max_abs_diff = max(max_abs_diff, np.max(np.abs(T - Tref) / np.maximum(Tref, 1)))
    axes[0].plot(tt, T, color=color, ls=ls, lw=2.2,
                 label=rf'$\lambda={lam:g}$: $\beta_1={beta1_lam:.2e}$, $M(0)={M0_lam:.1e}$')
    axes[1].plot(tt, M, color=color, ls=ls, lw=2.2, label=rf'$\lambda={lam:g}$')

print(f'largest relative difference in T(t) between the three integrations: {max_abs_diff:.2e}')

axes[0].set_xlabel('time (days)')
axes[0].set_ylabel(r'$T(t)$ (cells)')
axes[0].set_title(r'Tumor cells $T(t)$: identical for every $\lambda$', fontsize=11.5)
axes[0].legend(loc='upper left', fontsize=8.6, framealpha=0.95)

axes[1].set_xlabel('time (days)')
axes[1].set_ylabel(r'$M(t)$ (cells)')
axes[1].set_yscale('log')
axes[1].set_title(r'TAMs $M(t)$: scales by $\lambda$', fontsize=11.5)
axes[1].legend(loc='upper left', fontsize=9, framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'structural_symmetry_demo.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print('figure saved.')
