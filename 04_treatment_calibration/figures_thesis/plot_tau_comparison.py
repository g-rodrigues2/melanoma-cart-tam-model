"""Integrated autocorrelation time (tau, in MCMC steps) of the full model (_mac_treat, saved backend, long-hinge
group) and of the minimal identifiable submodel (same group). tau does not depend on the number of walkers or steps,
unlike the effective sample size, which convergence_check.py computes per walker (kept/tau) and reduced_fit.py
pooled over walkers (kept*nwalkers/tau).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import emcee

from _paths import FIGURES

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.labelsize': 13, 'xtick.labelsize': 10.5, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'axes.linewidth': 1.0,
})

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reinitiate_files',
                       'reinitiate_cart_sl_b16_mac_treat.h5')
if not os.path.exists(BACKEND):
    raise SystemExit(
        f"{BACKEND} not found. This figure reads the _mac_treat backend of the long-hinge group, "
        "which is distributed in the Zenodo deposit and not in this repository (see the repository README).")

b = emcee.backends.HDFBackend(BACKEND, read_only=True)
tau_full = b.get_autocorr_time(tol=0)[:8]  # the 8 model parameters only

labels = [r'$\log_{10}\alpha_1$', r'$\log_{10}K_1$', r'$\log_{10}\beta_1$',
          r'$\mu_1$', r'$\alpha_3$', r'$K_3$', r'$\kappa_3$', r'$\mu_3$']

tau_reduced_worst = 409.52  # largest integrated autocorrelation time of reduced_fit.py (long-hinge group)

fig, ax = plt.subplots(figsize=(8.8, 5.4))
x = np.arange(8)
ax.bar(x, tau_full, color='#c0392b', edgecolor='k', lw=0.6, width=0.6,
       label='full model ($T$-$M$-$C$, 8 free parameters)')
ax.axhline(tau_reduced_worst, color='#2a9d5c', lw=2.4, ls='-',
           label=r'minimal identifiable submodel (3 free parameters), $\tau_{\max}=410$')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel(r'autocorrelation time $\tau$ (MCMC steps), log scale')
ax.legend(loc='upper left', fontsize=9.3, framealpha=0.95)

ratio = tau_full.max() / tau_reduced_worst
ax.text(0.98, 0.06, rf'worst-case improvement: $\tau$ falls by a factor of ${ratio:.0f}\times$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10.5,
        bbox=dict(boxstyle='round', fc='white', ec='0.4', alpha=0.95))

fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'tau_full_vs_reduced.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print('tau_full:', tau_full)
print('ratio worst:', ratio)
print('figure saved.')
