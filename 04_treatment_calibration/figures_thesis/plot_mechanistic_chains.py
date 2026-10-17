"""Walker traces of the full T-M-C model in the long-hinge group (Section 3.4 and appendix).

The sampling failure of the mechanistic model is a statement about where the walkers go, so it is
shown as traces rather than described. Two figures are written.

  jilani_tmc_chain.png       three panels for the body of the section: the log-posterior, mu1 and
                             log10(beta1), each with the bounded maximum-likelihood estimate marked.
                             The walkers cover the whole prior interval of the two parameters that
                             carry the biological hypothesis while the log-posterior stays about
                             nineteen units below the optimum that the optimizer reaches.

  jilani_tmc_chain_full.png  the same run with one panel per model parameter, for the appendix.

Input: the emcee backend of the run, reinitiate_files/reinitiate_cart_sl_b16_mac_treat.h5, which is
not published with the repository (the chains are distributed separately), and the bounded estimate
Output_Calibration/TMC/mle_bounded_cart_sl_b16.npz.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emcee

from _paths import CALIBRATION_DIR, FIGURES, STYLE

plt.rcParams.update(STYLE)
plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 13.0})

GROUP = 'cart_sl_b16'
BACKEND = os.path.join(CALIBRATION_DIR, 'reinitiate_files', f'reinitiate_{GROUP}_mac_treat.h5')
MLE = os.path.join(CALIBRATION_DIR, 'Output_Calibration', 'TMC', f'mle_bounded_{GROUP}.npz')
COL_WALK, COL_MLE = '#6f6f6f', '#c0392b'
THIN, N_WALKERS = 400, 45

# panel, index in theta (None for the log-posterior), axis label, prior interval
PARAMS = [
    (None, r'$\log$ posterior', None),
    (3, r'$\mu_1$ (day$^{-1}$ per dose)', (0.0, 5.0)),
    (2, r'$\log_{10}\beta_1$ (mm$^{-3}$)', (-6.0, -2.0)),
    (0, r'$\log_{10}\alpha_1$ (day$^{-1}$)', (-3.0, -0.30103)),
    (1, r'$\log_{10}K_1$ (mm$^3$)', (2.0, 4.69897)),
    (4, r'$\alpha_3$ (day$^{-1}$)', (0.0, 5.0)),
    (5, r'$K_3$ (mm$^3$)', (1.0, 10000.0)),
    (6, r'$\kappa_3$ (day$^{-1}$)', (0.0, 2.0)),
    (7, r'$\mu_3$ (mm$^3\times$day)$^{-1}$', (0.0, 0.05)),
]


def load():
    if not os.path.exists(BACKEND):
        sys.exit(f'{BACKEND} not found. This figure reads the _mac_treat backend of the long-hinge '
                 'group, which is distributed separately from the repository.')
    b = emcee.backends.HDFBackend(BACKEND, read_only=True)
    chain, logp = b.get_chain(thin=THIN), b.get_log_prob(thin=THIN)
    n_steps = b.get_chain().shape[0]
    accept = float(np.mean(b.accepted)) / n_steps
    z = np.load(MLE, allow_pickle=True)
    steps = np.arange(chain.shape[0]) * THIN / 1.0e4          # in units of 10^4 steps
    return chain[:, :N_WALKERS, :], logp[:, :N_WALKERS], steps, np.asarray(z['theta'], float), float(z['ll']), accept


def draw(ax, steps, series, mle_value, label, prior, panel):
    ax.plot(steps, series, color=COL_WALK, lw=0.45, alpha=0.22, zorder=2)
    ax.axhline(mle_value, color=COL_MLE, lw=1.8, ls='--', zorder=4,
               label='bounded maximum likelihood')
    if prior is not None:
        ax.set_ylim(prior)
    ax.set_ylabel(label)
    ax.annotate(f'{panel})', xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
                fontsize=16, fontweight='bold')


def main():
    chain, logp, steps, theta, ll, accept = load()

    # ---- body figure: log-posterior, mu1, log10(beta1) ----
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    for ax, (idx, label, prior), panel in zip(axes, PARAMS[:3], 'abc'):
        series = logp if idx is None else chain[:, :, idx]
        draw(ax, steps, series, ll if idx is None else theta[idx], label, prior, panel)
    # the shared x-label sits on the middle panel, as in the other figures of this chapter
    axes[1].set_xlabel(r'MCMC step ($\times10^{4}$)')
    axes[0].set_ylim(-720, -655)
    axes[0].legend(loc='lower right', fontsize=12.0, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'jilani_tmc_chain.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

    # ---- appendix figure: every model parameter ----
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.2))
    for ax, (idx, label, prior), panel in zip(axes.ravel(), PARAMS, 'abcdefghi'):
        series = logp if idx is None else chain[:, :, idx]
        draw(ax, steps, series, ll if idx is None else theta[idx], label, prior, panel)
    axes[0, 0].set_ylim(-780, -650)
    for ax in axes[-1]:
        ax.set_xlabel(r'MCMC step ($\times10^{4}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'jilani_tmc_chain_full.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

    print('walkers drawn: %d of %d, thinned every %d steps' % (chain.shape[1], 102, THIN))
    print('acceptance %.3f | chain max log-posterior %.2f | bounded MLE %.2f | gap %.2f'
          % (accept, logp.max(), ll, ll - logp.max()))
    for idx, label, prior in PARAMS[1:3]:
        col = chain[:, :, idx]
        print('  %-24s sampled [%.3f, %.3f] of prior [%.3f, %.3f], MLE %.3f'
              % (label, col.min(), col.max(), prior[0], prior[1], theta[idx]))
    print('figures saved in', FIGURES)


if __name__ == '__main__':
    main()
