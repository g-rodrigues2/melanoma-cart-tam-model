"""Scenarios 5 and 6, revised.

Scenario 5 adds a quantitative description of the late-window oscillation
(amplitude over days 200-300) for each beta1, so that damped and sustained
oscillations are distinguished explicitly.
Scenario 6 replaces the single day-300 z statistic by a Monte Carlo estimate of
the probability that the likelihood ratio between the two simple hypotheses
(beta1 = 0 and beta1 = beta1*) selects the hypothesis that generated the data,
for three sampling designs of M(t) with 15% relative Gaussian noise.
"""
import csv

import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

import numpy as np
from scipy.integrate import odeint

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    # sizes chosen so that, scaled to the 6.30 in text width, the labels land between 8 and 10 pt
    'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.fontsize': 12, 'axes.linewidth': 1.0,
})

A1_C, K1_C, MU1_C = 0.69, 1000.0, 0.1
A3_C, K3_C, K3RATE_C, MU3_C = 0.9, 20.0, 0.35, 1e-9
T0_C, M0_C, C0_C = 2.0, 1.0, 10.0
BETA1_STAR = 5.9e-5
SIGMA_REL = 0.15
N_MC = 20000


def rhs_full(y, t, beta1):
    T, M, C = y
    dT = A1_C * T * (1 - T / K1_C) * (1 + beta1 * M) - MU1_C * C * T
    dM = 0.1 * T * M * (1 - M / 1000.0) - 0.34 * M
    dC = A3_C * C * T / (K3_C + T) - MU3_C * C * T - K3RATE_C * C
    return [dT, dM, dC]


t_grid = np.linspace(0, 300, 30001)
betas = [0.0, BETA1_STAR, 10 * BETA1_STAR]
sol = {b: odeint(rhs_full, [T0_C, M0_C, C0_C], t_grid, args=(b,), rtol=1e-10, atol=1e-10)
       for b in betas}

# ---------------------------------------------------------------- Scenario 5
late = t_grid >= 200
early = (t_grid >= 50) & (t_grid < 150)
rows5 = []
for b in betas:
    T = sol[b][:, 0]
    amp_early = np.log10(T[early].max() / T[early].min())
    amp_late = np.log10(T[late].max() / T[late].min())
    rows5.append([b, T[late].min(), T[late].max(), amp_early, amp_late])
    print(f'S5 beta1={b:.2e}: T in days 200-300 [{T[late].min():.3f}, {T[late].max():.1f}] mm3, '
          f'log10 amplitude days 50-150 = {amp_early:.3f}, days 200-300 = {amp_late:.3f}')
with open('scenario5_amplitudes.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['beta1_mm-3', 'T_min_200_300', 'T_max_200_300', 'log10_amp_50_150', 'log10_amp_200_300'])
    w.writerows(rows5)

fig5, axes5 = plt.subplots(1, 3, figsize=(11.5, 4.0))
labels5 = [r'$\beta_1=0$', r'$\beta_1=\beta_1^{*}=5.9\times10^{-5}$ mm$^{-3}$', r'$\beta_1=10\,\beta_1^{*}$']
colors5 = ['#377eb8', '#4daf4a', '#e41a1c']
names = [r'$T(t)$ (mm$^3$)', r'$M(t)$ (mm$^3$)', r'$C(t)$ (normalized)']
for j in range(3):
    ax = axes5[j]
    for b, lab, col in zip(betas, labels5, colors5):
        ax.plot(t_grid, sol[b][:, j], color=col, lw=1.6, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('time (days)')
    ax.set_ylabel(names[j] + ', log scale')
    # panel label inside the axes, in bold, as in the published figures of this chapter
    # headroom on the log scale so that the panel label does not sit on the curve
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    ax.annotate('abc'[j] + ')', xy=(0.028, 0.96), xycoords='axes fraction', ha='left',
                va='top', fontsize=16, fontweight='bold')
handles, labs = axes5[0].get_legend_handles_labels()
fig5.legend(handles, labs, loc='upper center', ncol=3, frameon=False, fontsize=13,
            bbox_to_anchor=(0.5, 1.04))
fig5.tight_layout(rect=(0, 0, 1, 0.95))
fig5.savefig(os.path.join('figures', 'scenario5_coexistence_virtual.png'), dpi=220, bbox_inches='tight')
plt.close(fig5)

# ---------------------------------------------------------------- Scenario 6
M0h = sol[0.0][:, 1]
M1h = sol[BETA1_STAR][:, 1]
designs = {
    'six time points (0, 30, 60, 90, 150, 300 days)': np.array([0, 30, 60, 90, 150, 300.0]),
    'weekly, days 0-300 (43 points)': np.arange(0, 301, 7.0),
    'every 3 days, days 200-300 (34 points)': np.arange(201, 301, 3.0),
}
rng = np.random.default_rng(2026)


def loglik(y, mu):
    s = SIGMA_REL * mu
    return -0.5 * np.sum(((y - mu) / s) ** 2 + 2 * np.log(s), axis=-1)


rows6 = []
for name, days in designs.items():
    mu0 = np.interp(days, t_grid, M0h)
    mu1 = np.interp(days, t_grid, M1h)
    acc = []
    for truth, mu in ((0, mu0), (1, mu1)):
        y = mu + rng.normal(size=(N_MC, len(days))) * SIGMA_REL * mu
        llr = loglik(y, mu1) - loglik(y, mu0)
        acc.append(np.mean(llr > 0) if truth == 1 else np.mean(llr < 0))
    p_correct = 0.5 * (acc[0] + acc[1])
    rows6.append([name, len(days), acc[0], acc[1], p_correct])
    print(f'S6 {name}: P(correct | beta1=0)={acc[0]:.3f}, P(correct | beta1*)={acc[1]:.3f}, mean={p_correct:.3f}')
z300 = abs(np.interp(300, t_grid, M1h) - np.interp(300, t_grid, M0h)) / np.sqrt(
    (SIGMA_REL * np.interp(300, t_grid, M1h)) ** 2 + (SIGMA_REL * np.interp(300, t_grid, M0h)) ** 2)
print(f'S6 separation at day 300 alone: {z300:.2f} sigma')
with open('scenario6_design_power.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['design', 'n_points', 'P_correct_beta1_0', 'P_correct_beta1_star', 'P_correct_mean'])
    w.writerows(rows6)

fig6, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.6, 1]})
axa.plot(t_grid, M0h, color='#377eb8', lw=1.5, label=r'$M(t)$, $\beta_1=0$')
axa.plot(t_grid, M1h, color='#e41a1c', lw=1.5, label=r'$M(t)$, $\beta_1=\beta_1^{*}$')
days6 = designs['six time points (0, 30, 60, 90, 150, 300 days)']
rng_plot = np.random.default_rng(7)
for mu_curve, col, mk, lab in ((M0h, '#377eb8', 'o', r'noisy sample, $\beta_1=0$'),
                               (M1h, '#e41a1c', 's', r'noisy sample, $\beta_1=\beta_1^{*}$')):
    mu = np.interp(days6, t_grid, mu_curve)
    y = mu + rng_plot.normal(size=len(days6)) * SIGMA_REL * mu
    axa.errorbar(days6, y, yerr=SIGMA_REL * mu, fmt=mk, color=col, ms=6, capsize=3, label=lab)
axa.set_yscale('log')
axa.set_ylim(0.3, 3e3)
axa.set_xlabel('time (days)')
axa.set_ylabel(r'$M(t)$ (mm$^3$), log scale')
# panel label inside the axes, in bold, as in the published figures of this chapter
# (lowercase letter with a closing parenthesis); the description is offset in points
_anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
               fontsize=16)
axa.annotate('a)', fontweight='bold', **_anchor)
axa.annotate('six-time-point design, 15% relative noise', xytext=(20, 0), textcoords='offset points', **_anchor)
axa.legend(loc='lower right', fontsize=11.5, framealpha=0.95)
z_t = np.abs(M1h - M0h) / np.sqrt((SIGMA_REL * M0h) ** 2 + (SIGMA_REL * M1h) ** 2)
axb.plot(t_grid, z_t, color='0.25', lw=1.2)
z6 = np.interp(days6, t_grid, z_t)
axb.plot(days6, z6, 'D', color='#e41a1c', ms=6, label='six sampling days')
for d, z in zip(days6, z6):
    axb.annotate(f'{z:.2f}', (d, z), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=11)
axb.axhline(2.0, color='0.4', ls=':', lw=1.1, label=r'$2\sigma$')
# headroom so that the panel label and the legend do not sit on the curve
axb.set_ylim(top=5.2)
axb.set_xlabel('time (days)')
axb.set_ylabel(r'single-sample separation $z(t)$ ($\sigma$ units)')
_anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
               fontsize=16)
axb.annotate('b)', fontweight='bold', **_anchor)
# upper right: the only region of this panel free of curve peaks and of the point annotations
axb.legend(loc='upper right', fontsize=11.5, framealpha=0.95)
fig6.tight_layout()
fig6.savefig(os.path.join('figures', 'scenario6_virtual_experiment_noise.png'), dpi=220, bbox_inches='tight')
plt.close(fig6)
print('figures saved')
