"""MCMC chains of the models selected in Chapter 3 (thesis appendix).

Reads the chains of the Zenodo deposit (they are not in the code repository), whose location is given by
the environment variable MELANOMA_CHAINS or by DEPOSIT below.

  chains_selected_models.png  walker traces of the free parameters of the model selected by BIC in each
                              group: the residual standard deviation of the three (top row) and the three
                              parameters of the killing term of the long-hinge group (bottom row). The
                              initial tumor volumes, one per animal, are not shown.

Also prints the posterior summaries quoted in the appendix.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _paths import FIGURES, GROUP_LABELS, STYLE

plt.rcParams.update(STYLE)
plt.rcParams.update({'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'legend.fontsize': 12})

DEPOSIT = os.environ.get(
    'MELANOMA_CHAINS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                 'zenodo_deposit', 'melanoma_chains', '04_treatment_calibration'))

SELECTED = {'cart_ss_b16': '_exp', 'cart_sm_b16': '_exp', 'cart_sl_b16': '_exp_EDwithSin'}
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
MODEL_LABELS = {'_exp': 'no treatment term', '_exp_EDwithSin': 'sinusoidal killing term'}
SL_PARAMS = [(0, r'$a$ (day$^{-1}$)'), (1, r'$b$ (day$^{-1}$)'), (2, r'$\omega$ (day$^{-1}$)')]
WALKER_COLOR = '#2b2b2b'


def load(group, model):
    path = os.path.join(DEPOSIT, f'reinitiate_{group}{model}.npz')
    if not os.path.exists(path):
        raise SystemExit(f'chain file not found: {path}\n'
                         'set MELANOMA_CHAINS to the folder of the deposit.')
    z = np.load(path, allow_pickle=True)
    return z['chain'], z['log_prob'], int(z['thin']), int(z['burn_in'])


def trace(ax, steps, values, burn_in, every=2):
    ax.plot(steps, values[:, ::every], color=WALKER_COLOR, lw=0.2, alpha=0.2)
    ax.axvline(burn_in, color='#c0392b', ls='--', lw=1.2)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_xlabel('MCMC step')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))


def summary(name, post):
    print(f'{name}: median {np.median(post):.4g}, '
          f'95% CrI [{np.percentile(post, 2.5):.4g}, {np.percentile(post, 97.5):.4g}]')


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.0))

    for ax, group in zip(axes[0], GROUPS):
        model = SELECTED[group]
        chain, _, thin, burn_in = load(group, model)
        steps = np.arange(chain.shape[0]) * thin
        trace(ax, steps, chain[:, :, -1], burn_in)
        ax.set_ylabel(r'$\sigma$ (mm$^3$)')
        ax.set_title(f'{GROUP_LABELS[group]}, {MODEL_LABELS[model]}', fontsize=13, loc='left')
        post = chain[chain.shape[0] // 4:, :, -1].ravel()
        ax.set_ylim(np.percentile(post, 0.05) * 0.8, np.percentile(post, 99.95) * 1.2)
        summary(f'sigma, {group}', post)

    chain, _, thin, burn_in = load('cart_sl_b16', SELECTED['cart_sl_b16'])
    steps = np.arange(chain.shape[0]) * thin
    for ax, (idx, label) in zip(axes[1], SL_PARAMS):
        trace(ax, steps, chain[:, :, idx], burn_in)
        post = chain[chain.shape[0] // 4:, :, idx].ravel()
        if idx != 2:
            ax.set_ylim(np.percentile(post, 0.05) * 0.8, np.percentile(post, 99.95) * 1.2)
        ax.set_ylabel(label)
        ax.set_title('long hinge, killing term', fontsize=13, loc='left')
        summary(f'{label}, cart_sl_b16', post)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'chains_selected_models.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('chains_selected_models.png written')


if __name__ == '__main__':
    main()
