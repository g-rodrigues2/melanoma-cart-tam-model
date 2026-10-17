"""Scenario 3: control duration for two values of beta1 and four CAR T-cell doses.

Extracted from notebooks/melanoma_CART_TAM.ipynb.
"""
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def sis(TMC, t, alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3):
    T, M, C = TMC
    dT = alpha1 * T * (1 - (T / K1)) * (1 + (beta1 * M)) - mu1 * T * C
    dM = alpha2 * T * M * (1 - (M / K2)) - kappa2 * M
    dC = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    return [dT, dM, dC]


alpha1 = 0.69
K1 = 10**9
mu1 = 1e-7

alpha2 = 1e-7
K2 = 10**9
kappa2 = 0.34

K3 = 2e7
alpha3 = 0.9
mu3 = 6.752e-10
kappa3 = 0.35

t = np.linspace(0, 10000, 100000)
threshold = 1e2
C0_values = [1e6, 1e7, 7e7, 9e7]
beta_values = [1e-7, 2e-6]

sols_T = []
sols_M = []
sols_C = []

for beta1 in beta_values:
    row_T = []
    row_M = []
    row_C = []
    print(f"\n===== Results for beta1 = {beta1:.1e} =====")

    for C0 in C0_values:
        TMC0 = [2 * 10**6, 10**6, C0]
        sol = odeint(
            sis,
            TMC0,
            t,
            args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
        )

        below_threshold_time = 0.0
        for i in range(1, len(t)):
            if sol[i, 0] < threshold:
                below_threshold_time += (t[i] - t[i - 1])

        print(
            f"C(0) = {C0:.1e}: total time with T(t) < {threshold} cells = "
            f"{below_threshold_time:.2f} days"
        )

        row_T.append(sol[:, 0])
        row_M.append(sol[:, 1])
        row_C.append(sol[:, 2])

    sols_T.append(row_T)
    sols_M.append(row_M)
    sols_C.append(row_C)

aux = np.full_like(t, threshold)

plt.rcParams.update({
    'font.size': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

fig, axes = plt.subplots(2, 3, figsize=(20, 8))

for ax in axes.flatten():
    ax.yaxis.offsetText.set_visible(False)

line_styles = ['-', '--', '-.', ':']
labels_C0 = [
    r'$C(0)=10^{6}$',
    r'$C(0)=10^{7}$',
    r'$C(0)=7\cdot10^{7}$',
    r'$C(0)=9\cdot10^{7}$'
]

row = 0

for j in range(4):
    axes[0, 0].plot(t, np.log10(sols_T[row][j]), line_styles[j], color='b', label=labels_C0[j])
axes[0, 0].fill_between(t, np.log10(10**-2), np.log10(aux), color='grey', alpha=0.3)
axes[0, 0].set_xlim(0, 30)
axes[0, 0].set_ylim(-2, 10)
axes[0, 0].set_xlabel('Time (days)', fontsize=20)
axes[0, 0].set_ylabel(r'Tumor cells, $\log_{10}(\text{cells})$', fontsize=20)
axes[0, 0].text(0.01, 0.98, 'a)', transform=axes[0, 0].transAxes, fontsize=20, va='top', fontweight='bold')

for j in range(4):
    axes[0, 1].plot(t, np.log10(sols_M[row][j]), line_styles[j], color='k')
axes[0, 1].set_xlim(0, 30)
axes[0, 1].set_ylim(0, 10)
axes[0, 1].set_xlabel('Time (days)', fontsize=20)
axes[0, 1].set_ylabel(r'TAMs, $\log_{10}(\text{cells})$', fontsize=20)
axes[0, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
axes[0, 1].text(0.01, 0.98, 'b)', transform=axes[0, 1].transAxes, fontsize=20, va='top', fontweight='bold')

for j in range(4):
    axes[0, 2].plot(t, sols_C[row][j], line_styles[j], color='r')
axes[0, 2].set_xlim(0, 30)
axes[0, 2].set_xlabel('Time (days)', fontsize=20)
axes[0, 2].set_ylabel(r'CAR T-cells ($\times 10^{7} \text{cells}$)', fontsize=20)
axes[0, 2].text(0.01, 0.98, 'c)', transform=axes[0, 2].transAxes, fontsize=20, va='top', fontweight='bold')

row = 1

for j in range(4):
    axes[1, 0].plot(t, np.log10(sols_T[row][j]), line_styles[j], color='b')
axes[1, 0].fill_between(t, np.log10(10**-2), np.log10(aux), color='grey', alpha=0.3)
axes[1, 0].set_xlim(0, 30)
axes[1, 0].set_ylim(-2, 10)
axes[1, 0].set_xlabel('Time (days)', fontsize=20)
axes[1, 0].set_ylabel(r'Tumor cells, $\log_{10}(\text{cells})$', fontsize=20)
axes[1, 0].text(0.01, 0.98, 'd)', transform=axes[1, 0].transAxes, fontsize=20, va='top', fontweight='bold')

for j in range(4):
    axes[1, 1].plot(t, np.log10(sols_M[row][j]), line_styles[j], color='k')
axes[1, 1].set_xlim(0, 30)
axes[1, 1].set_ylim(0, 10)
axes[1, 1].set_xlabel('Time (days)', fontsize=20)
axes[1, 1].set_ylabel(r'TAMs, $\log_{10}(\text{cells})$', fontsize=20)
axes[1, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
axes[1, 1].text(0.01, 0.98, 'e)', transform=axes[1, 1].transAxes, fontsize=20, va='top', fontweight='bold')

for j in range(4):
    axes[1, 2].plot(t, sols_C[row][j], line_styles[j], color='r')
axes[1, 2].set_xlim(0, 30)
axes[1, 2].set_xlabel('Time (days)', fontsize=20)
axes[1, 2].set_ylabel(r'CAR T-cells ($\times 10^{7} \text{cells}$)', fontsize=20)
axes[1, 2].text(0.01, 0.98, 'f)', transform=axes[1, 2].transAxes, fontsize=20, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig13.pdf', bbox_inches='tight')
plt.close('all')
