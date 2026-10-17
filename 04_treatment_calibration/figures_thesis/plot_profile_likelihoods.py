"""Profile-likelihood figure of the mechanistic models (practical identifiability, Section 3.4).

Reads profile_bounded_grid.csv and profile_bounded_summary.csv written by profiles_bounded.py.
  profile_likelihoods.png  top row: profile log-likelihood of log10(beta1), T-M and T-M-C models;
                           bottom row: profile log-likelihood of mu1, T-M-C model (symmetric log axis), with the 95% interval and the
                           likelihood-ratio statistic of the T-M model against the T-M-C model;
                           one column per group, values relative to the maximum of each profile, truncated at -12
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from _paths import FIGURES, GROUP_LABELS, STYLE, TMC

plt.rcParams.update(STYLE)
LR = 1.9207
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
YMIN = -12.0
PASTEL = {'grey': '#7f7f7f', 'blue': '#377eb8', 'red': '#fbb4ae', 'lightblue': '#b3cde3', 'thr': '#c0392b'}


def read_grid():
    data = {}
    with open(os.path.join(TMC, 'profile_bounded_grid.csv'), newline='') as f:
        for r in csv.DictReader(f):
            data.setdefault((r['parameter'], r['model'], r['group']), []).append(
                (float(r['value']), float(r['profile_ll'])))
    return {k: np.array(sorted(v)) for k, v in data.items()}


def read_summary():
    with open(os.path.join(TMC, 'profile_bounded_summary.csv'), newline='') as f:
        return {(r['parameter'], r['model'], r['group']): r for r in csv.DictReader(f)}


def interval_patch(ax, s, label_fmt):
    lo, hi = float(s['CI95_lo']), float(s['CI95_hi'])
    contiguous = s['CI_contiguous'] == '1'
    label = label_fmt.format(lo=lo, hi=hi) + ('' if contiguous else ' (not contiguous)')
    return lo, hi, label


def main():
    grid, summary = read_grid(), read_summary()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.5))

    for ax, g in zip(axes[0], GROUPS):
        for model, col, mk, lab in [('_mac', PASTEL['grey'], 's', r'$T$-$M$ model'),
                                    ('_mac_treat', PASTEL['blue'], 'o', r'$T$-$M$-$C$ model')]:
            v = grid[('log10_beta1', model, g)]
            rel = v[:, 1] - v[:, 1].max()
            ax.plot(v[:, 0], np.maximum(rel, YMIN), marker=mk, ms=3.2, color=col, lw=1.5, label=lab)
        ax.axhline(-LR, color=PASTEL['thr'], ls='--', lw=1.2, label=r'95% threshold ($-1.92$)')
        ax.set_xlim(-6.05, -1.95)
        ax.set_ylim(YMIN - 0.3, 0.6)
        ax.set_xlabel(r'$\log_{10}\beta_1$ (mm$^{-3}$)')
        ax.set_title(GROUP_LABELS[g])
    axes[0, 0].set_ylabel('profile log-likelihood\n(relative to maximum)')
    axes[0, 0].legend(loc='lower left', fontsize=10.5)

    for ax, g in zip(axes[1], GROUPS):
        v = grid[('mu1', '_mac_treat', g)]
        s = summary[('mu1', '_mac_treat', g)]
        rel = v[:, 1] - v[:, 1].max()
        lo, hi, label = interval_patch(ax, s, '95% interval [{lo:.3f}, {hi:.3f}]')
        ax.axvspan(lo, hi, color=PASTEL['lightblue'], alpha=0.65, label=label)
        ax.plot(v[:, 0], np.maximum(rel, YMIN), marker='o', ms=3.2, color=PASTEL['blue'], lw=1.5)
        ax.axhline(-LR, color=PASTEL['thr'], ls='--', lw=1.2)
        lr_tm = 2 * (float(s['ll_max']) - float(summary[('log10_beta1', '_mac', g)]['ll_max']))
        ax.text(0.97, 0.90, rf'$T$-$M$ vs $T$-$M$-$C$: $2\Delta\ell={lr_tm:.1f}$', transform=ax.transAxes,
                ha='right', fontsize=11.0,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
        ax.set_xscale('symlog', linthresh=0.05, linscale=0.6)
        ax.set_xlim(0, 5.2)
        ax.set_xticks([0, 0.05, 0.1, 0.5, 1, 5])
        ax.set_xticklabels(['0', '0.05', '0.1', '0.5', '1', '5'])
        ax.set_ylim(YMIN - 0.3, 0.6)
        ax.set_xlabel(r'$\mu_1$ (day$^{-1}$ per administered dose)')
        ax.legend(loc='lower right', fontsize=10.5)
    axes[1, 0].set_ylabel('profile log-likelihood\n(relative to maximum)')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'profile_likelihoods.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('saved profile_likelihoods.png in', FIGURES)


if __name__ == '__main__':
    main()
