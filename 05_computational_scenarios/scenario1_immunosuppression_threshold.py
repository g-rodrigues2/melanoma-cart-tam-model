"""Scenario 1: final tumor size against mu3 and bisection of the threshold mu3_dagger.

Extracted from notebooks/melanoma_CART_TAM.ipynb.
"""
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

alpha1 = 0.69
K1 = 10**9
beta1 = 2.3e-10
mu1 = 1e-7

alpha2 = 1e-7
K2 = 10**9
kappa2 = 0.34

K3 = 2e7
alpha3 = 0.9
mu3_est = 1e-9
kappa3 = 0.35


def sis(TMC, t, alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3):
    T, M, C = TMC
    dT = alpha1 * T * (1 - (T / K1)) * (1 + (beta1 * M)) - mu1 * T * C
    dM = alpha2 * T * M * (1 - (M / K2)) - kappa2 * M
    dC = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    return [dT, dM, dC]


TMC0 = [2e6, 1e6, 5e5]

t_time = np.linspace(0, 60, 100000)
t_final = np.linspace(0, 60, 100000)

mu3_values_time = np.linspace(5e-10, 5e-9, 1000)

solutions_time = []
for mu3 in mu3_values_time:
    sol = odeint(
        sis, TMC0, t_time,
        args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
    )
    solutions_time.append(sol)

mu3_values_final = np.linspace(6e-10, 7.5e-10, 20000)

final_T_values = []
for mu3 in mu3_values_final:
    sol = odeint(
        sis, TMC0, t_final,
        args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
    )
    final_T_values.append(sol[-1, 0])

final_T_values = np.array(final_T_values)
mu3_values_final = np.array(mu3_values_final)

T_min = final_T_values.min()
T_max = final_T_values.max()
T_mid = 0.5 * (T_min + T_max)

a = b = None
for i in range(len(mu3_values_final) - 1):
    f_i = final_T_values[i] - T_mid
    f_ip1 = final_T_values[i + 1] - T_mid
    if f_i * f_ip1 < 0:
        a = mu3_values_final[i]
        b = mu3_values_final[i + 1]
        break


def final_T(mu3):
    sol = odeint(
        sis, TMC0, t_final,
        args=(alpha1, alpha2, alpha3, beta1, mu1, kappa2, mu3, kappa3, K1, K2, K3)
    )
    return sol[-1, 0]


mu3_star = None
if a is not None:
    for _ in range(100):
        m = 0.5 * (a + b)
        f_a = final_T(a) - T_mid
        f_m = final_T(m) - T_mid
        if f_a * f_m <= 0:
            b = m
        else:
            a = m
    mu3_star = 0.5 * (a + b)
    print(f"mu3* ~ {mu3_star:.3e}")
else:
    print("No clear numerical crossing was found. Adjust the mu3 interval or the final time horizon.")

plt.rcParams.update({
    'font.size': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax in axes:
    ax.yaxis.offsetText.set_visible(False)

for sol in solutions_time:
    axes[0].plot(t_time, sol[:, 0], '-', color='blue')

axes[0].set_xlabel('Time (days)', fontsize=20)
axes[0].set_ylabel(r'Tumor cells ($\times10^9$ cells)', fontsize=20)
axes[0].grid(False)
axes[0].text(0.02, 0.95, 'a)', transform=axes[0].transAxes, fontsize=20, va='top', fontweight='bold')

axes[1].plot(mu3_values_final, final_T_values, '-o', markersize=3, label='Final $T(t)$')
if mu3_star is not None:
    axes[1].axvline(x=mu3_star, color='r', linestyle='--')
    axes[1].text(
        mu3_star * 1.01,
        0.5 * (T_min + T_max),
        rf'$\mu_3^{{\dagger}}={mu3_star:.3e}$',
        rotation=90,
        verticalalignment='center',
        color='r',
        fontsize=18
    )

axes[1].set_xlabel(r'$\mu_3^{\dagger}$ (cells$\times$day)$^{-1}$', fontsize=20)
axes[1].set_ylabel(r'Stability of Tumor Population ($\times10^9$ cells)', fontsize=16)
axes[1].grid(False)
axes[1].text(0.02, 0.95, 'b)', transform=axes[1].transAxes, fontsize=20, va='top', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig10.pdf', bbox_inches='tight')
plt.close('all')
