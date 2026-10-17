"""Scenario 1: tumor, TAM and CAR T-cell trajectories for four values of mu3.

Extracted from notebooks/melanoma_CART_TAM.ipynb.
"""
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def sis(t, TMC, alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3):
    T, M, C = TMC
    dT = alpha1 * T * (1 - (T / K1)) * (1 + (beta1 * M)) - mu1 * T * C
    dM = alpha2 * T * M * (1 - (M / K2)) - kappa2 * M
    dC = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    return [dT, dM, dC]


alpha1 = 0.69
K1 = 10**9
beta1 = 2.3 * 10**-10
mu1 = 10**-7

alpha2 = 10**-7
K2 = 10**9
kappa2 = 0.34

K3 = 2 * 10**7
alpha3 = 0.9
mu3 = 1e-9
kappa3 = 0.35

TMC0 = [2 * 10**6, 10**6, 5 * 10**5]

t_span = (0, 60)
t_eval = np.linspace(0, 30, 1000000)

mu3_values = [1e-9, 5e-9, 5e-10, 1e-11]
solutions = []

for mu3 in mu3_values:
    sol = solve_ivp(
        sis,
        t_span,
        TMC0,
        args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3),
        t_eval=t_eval,
        method='LSODA'
    )
    solutions.append(sol)

plt.rcParams.update({
    'font.size': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax in axes:
    ax.yaxis.offsetText.set_visible(False)

axes[0].plot(solutions[0].t, np.log10(solutions[0].y[0, :]), '-b', label=r'$\mu_3 = 6.751 \cdot 10^{-10}$')
axes[0].plot(solutions[1].t, np.log10(solutions[1].y[0, :]), '--b', label=r'$\mu_3 = 6.752 \cdot 10^{-10}$')
axes[0].plot(solutions[2].t, np.log10(solutions[2].y[0, :]), '-.b', label=r'$\mu_3 = 6.753 \cdot 10^{-10}$')
axes[0].plot(solutions[3].t, np.log10(solutions[3].y[0, :]), ':b', label=r'$\mu_3 = 6.754 \cdot 10^{-10}$')
axes[0].set_xlabel('Time (days)', fontsize=20)
axes[0].set_ylabel(r'Tumor cells, $\log_{10}(\text{cells})$', fontsize=20)
axes[0].text(0.02, 0.95, 'a)', transform=axes[0].transAxes, fontsize=20, va='top', fontweight='bold')

axes[1].plot(solutions[0].t, np.log10(solutions[0].y[1, :]), '-k', label=r'$\mu_3 = 6.751 \cdot 10^{-10}$')
axes[1].plot(solutions[1].t, np.log10(solutions[1].y[1, :]), '--k', label=r'$\mu_3 = 6.752 \cdot 10^{-10}$')
axes[1].plot(solutions[2].t, np.log10(solutions[2].y[1, :]), '-.k', label=r'$\mu_3 = 6.753 \cdot 10^{-10}$')
axes[1].plot(solutions[3].t, np.log10(solutions[3].y[1, :]), ':k', label=r'$\mu_3 = 6.754 \cdot 10^{-10}$')
axes[1].set_xlabel('Time (days)', fontsize=20)
axes[1].set_ylabel(r'TAMs, $\log_{10}(\text{cells})$', fontsize=20)
axes[1].text(0.02, 0.95, 'b)', transform=axes[1].transAxes, fontsize=20, va='top', fontweight='bold')

axes[2].plot(solutions[0].t, solutions[0].y[2, :], '-r', label=r'$\mu_3 = 6.751 \cdot 10^{-10}$')
axes[2].plot(solutions[1].t, solutions[1].y[2, :], '--r', label=r'$\mu_3 = 6.752 \cdot 10^{-10}$')
axes[2].plot(solutions[2].t, solutions[2].y[2, :], '-.r', label=r'$\mu_3 = 6.753 \cdot 10^{-10}$')
axes[2].plot(solutions[3].t, solutions[3].y[2, :], ':r', label=r'$\mu_3 = 6.754 \cdot 10^{-10}$')
axes[2].set_xlabel('Time (days)', fontsize=20)
axes[2].set_ylabel(r'CAR T-cells ($\times 10^{7} \text{cells}$)', fontsize=20)
axes[2].text(0.02, 0.95, 'c)', transform=axes[2].transAxes, fontsize=20, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig11.pdf', bbox_inches='tight')
plt.close('all')
