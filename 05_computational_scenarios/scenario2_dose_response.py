"""Scenario 2: CAR T-cell dose response at the immunosuppression threshold.

Extracted from notebooks/melanoma_CART_TAM.ipynb.
"""
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sis(TMC, t, alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3):
    T, M, C = TMC
    dT = alpha1 * T * (1 - (T / K1)) * (1 + (beta1 * M)) - mu1 * T * C
    dM = alpha2 * T * M * (1 - (M / K2)) - kappa2 * M
    dC = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    return [dT, dM, dC]


alpha1 = 0.69
K1 = 10**9
beta1 = 2.3e-10
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

TMC0 = [2 * 10**6, 10**6, 1e6]
sol1 = odeint(
    sis, TMC0, t,
    args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
)
below_threshold_time = 0.0
for i in range(1, len(t)):
    if sol1[i, 0] < threshold:
        below_threshold_time += (t[i] - t[i - 1])
print(f"Total time with T(t) < {threshold} cells (C0 = 1e6): {below_threshold_time:.2f} days")

TMC0 = [2 * 10**6, 10**6, 1e7]
sol2 = odeint(
    sis, TMC0, t,
    args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
)
below_threshold_time = 0.0
for i in range(1, len(t)):
    if sol2[i, 0] < threshold:
        below_threshold_time += (t[i] - t[i - 1])
print(f"Total time with T(t) < {threshold} cells (C0 = 1e7): {below_threshold_time:.2f} days")

TMC0 = [2 * 10**6, 10**6, 7e7]
sol3 = odeint(
    sis, TMC0, t,
    args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
)
below_threshold_time = 0.0
for i in range(1, len(t)):
    if sol3[i, 0] < threshold:
        below_threshold_time += (t[i] - t[i - 1])
print(f"Total time with T(t) < {threshold} cells (C0 = 7e7): {below_threshold_time:.2f} days")

TMC0 = [2 * 10**6, 10**6, 9e7]
sol4 = odeint(
    sis, TMC0, t,
    args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
)
below_threshold_time = 0.0
for i in range(1, len(t)):
    if sol4[i, 0] < threshold:
        below_threshold_time += (t[i] - t[i - 1])
print(f"Total time with T(t) < {threshold} cells (C0 = 9e7): {below_threshold_time:.2f} days")

aux = np.full_like(t, 1e2)

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

axes[0].plot(t, np.log10(sol1[:, 0]), '-b', label=r'$C(0)=10^{6}$')
axes[0].plot(t, np.log10(sol2[:, 0]), '--b', label=r'$C(0)=10^{7}$')
axes[0].plot(t, np.log10(sol3[:, 0]), '-.b', label=r'$C(0)=7\cdot10^{7}$')
axes[0].plot(t, np.log10(sol4[:, 0]), ':b', label=r'$C(0)=9\cdot10^{7}$')
axes[0].fill_between(t, np.log10(10**-2), np.log10(aux), color='grey', alpha=0.3)
axes[0].set_xlim(0, 30)
axes[0].set_xlabel('Time (days)', fontsize=20)
axes[0].set_ylabel(r'Tumor cells, $\log_{10}(\text{cells})$', fontsize=20)
axes[0].text(0.02, 0.95, 'a)', transform=axes[0].transAxes, fontsize=20, va='top', fontweight='bold')

axes[1].plot(t, np.log10(sol1[:, 1]), '-k')
axes[1].plot(t, np.log10(sol2[:, 1]), '--k')
axes[1].plot(t, np.log10(sol3[:, 1]), '-.k')
axes[1].plot(t, np.log10(sol4[:, 1]), ':k')
axes[1].set_xlim(0, 30)
axes[1].set_xlabel('Time (days)', fontsize=20)
axes[1].set_ylabel(r'TAMs, $\log_{10}(\text{cells})$', fontsize=20)
axes[1].text(0.02, 0.95, 'b)', transform=axes[1].transAxes, fontsize=20, va='top', fontweight='bold')

axes[2].plot(t, sol1[:, 2], '-r', label=r'$C(0)=10^{6}$')
axes[2].plot(t, sol2[:, 2], '--r', label=r'$C(0)=10^{7}$')
axes[2].plot(t, sol3[:, 2], '-.r', label=r'$C(0)=7\cdot10^{7}$')
axes[2].plot(t, sol4[:, 2], ':r', label=r'$C(0)=9\cdot10^{7}$')
axes[2].set_xlim(0, 30)
axes[2].set_xlabel('Time (days)', fontsize=20)
axes[2].set_ylabel(r'CAR T-cells ($\times 10^{7} \text{cells}$)', fontsize=20)
axes[2].text(0.02, 0.95, 'c)', transform=axes[2].transAxes, fontsize=20, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig12.pdf', bbox_inches='tight')
plt.close('all')
