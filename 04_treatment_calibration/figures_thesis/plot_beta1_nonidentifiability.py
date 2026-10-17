"""Practical non-identifiability of beta_1 shown directly on the data (Section 3.4.1).

(a) Tumor-volume data of one long-hinge mouse with three fits of the full T-M-C model whose values of
    beta_1 span the whole likelihood-based 95% interval; the curves are indistinguishable.
(b) The compensation along that interval: the intrinsic growth rate alpha_1 falls as beta_1 rises, so
    that the apparent growth alpha_1(1 + beta_1 M) seen by the data stays the same.

Reads profile_bounded_theta.npz / profile_bounded_grid.csv (profiles_bounded.py).
  beta1_nonidentifiability.png
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_treatment_data
from models import _split_theta, solve_model

from _paths import FIGURES, STYLE, TMC

plt.rcParams.update(STYLE)
# the figure is scaled to the text width of the thesis, so the fonts are set larger here
plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 12.5})

GROUP = 'cart_sl_b16'
MODEL = '_mac_treat'
KEY = f'log10_beta1__{MODEL}__{GROUP}__'
MOUSE = 7                      # 3b_m8, the longest record of the group
SHOW = (-3.1667, -2.5833, -2.0)
COLORS = ('#1f78b4', '#33a02c', '#c0392b')
LR = 1.9207


def profile():
    npz = np.load(os.path.join(TMC, 'profile_bounded_theta.npz'), allow_pickle=True)
    grid, theta = npz[KEY + 'grid'], npz[KEY + 'theta']
    with open(os.path.join(TMC, 'profile_bounded_grid.csv'), newline='') as f:
        ll = np.array([float(r['profile_ll']) for r in csv.DictReader(f)
                       if r['parameter'] == 'log10_beta1' and r['model'] == MODEL
                       and r['group'] == GROUP])
    return grid, theta, ll


def panel_label(ax, letter, text, fontsize=15):
    """Panel marker inside the axes, bold, as in the published figures of Chapter 3."""
    anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
                  fontsize=fontsize)
    ax.annotate(f'{letter})', fontweight='bold', **anchor)
    # the gap is given in points, so it does not depend on how wide the panel is
    ax.annotate(text, xytext=(19, 0), textcoords='offset points', **anchor)


def main():
    mice = load_treatment_data()[GROUP]
    grid, theta, ll = profile()
    ll_max = ll.max()
    inside = ll >= ll_max - LR

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    d = mice[MOUSE]['data']
    shown = []
    t = np.linspace(0.0, d[-1, 0], 240)
    ax.plot(d[:, 0], d[:, 1], 'o', ms=6, mfc='white', mec='#2b2b2b', mew=1.3,
            label='data (one mouse)', zorder=5)
    for value, color in zip(SHOW, COLORS):
        i = int(np.argmin(np.abs(grid - value)))
        ode_args, ic_fn = _split_theta(theta[i], MODEL, len(mice))
        sol = solve_model(MODEL, t, ode_args, ic_fn(MOUSE))
        alpha1, beta1 = ode_args[0], ode_args[2]
        shown.append(sol[:, 0])
        ax.plot(t, sol[:, 0], lw=2.0, color=color,
                label=(r'$\beta_1=%.1f\times10^{-3}$, $\alpha_1=%.3f$'
                       % (beta1 * 1e3, alpha1)))
        print(f'log10_beta1={grid[i]:7.4f} beta1={beta1:.3e} alpha1={alpha1:.4f} '
              f'mu1={ode_args[3]:.4f} dll={ll_max - ll[i]:.3f}')
    spread = np.array(shown)
    rel = (spread.max(axis=0) - spread.min(axis=0)) / spread.mean(axis=0)
    print(f'max relative spread between the three trajectories: {rel.max():.3f}')
    ax.set_yscale('log')
    ax.set_ylim(top=3.0e4)
    ax.set_xlabel('time since adoptive transfer (days)')
    ax.set_ylabel(r'tumor volume (mm$^3$), log scale')
    panel_label(ax, 'a', 'three fits of the same data')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=13.0, framealpha=0.95)

    alpha1 = np.array([10.0 ** th[0] for th in theta])
    bx.plot(grid[inside], alpha1[inside], 'o-', ms=4, lw=1.4, color='#2b2b2b', zorder=3)
    for value, color in zip(SHOW, COLORS):
        i = int(np.argmin(np.abs(grid - value)))
        bx.plot(grid[i], alpha1[i], 'o', ms=10, mfc=color, mec='white', mew=1.2, zorder=4)
    bx.set_yscale('log')
    # headroom so that the panel label does not sit on the curve
    bx.set_ylim(top=float(alpha1[inside].max()) * 2.2)
    bx.set_xlabel(r'$\log_{10}\beta_1$ (mm$^{-3}$)')
    bx.set_ylabel(r'intrinsic growth rate $\alpha_1$ (day$^{-1}$)')
    panel_label(bx, 'b', r'compensation along the $95\%$ set')
    bx.grid(alpha=0.25, lw=0.6)
    lo, hi = grid[inside].min(), grid[inside].max()
    print(f'95% set: log10_beta1 in [{lo:.3f},{hi:.3f}], alpha1 in '
          f'[{alpha1[inside].min():.4f},{alpha1[inside].max():.4f}], n={inside.sum()}')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'beta1_nonidentifiability.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('figure saved.')


if __name__ == '__main__':
    main()
