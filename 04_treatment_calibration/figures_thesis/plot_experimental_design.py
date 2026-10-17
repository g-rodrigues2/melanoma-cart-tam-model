"""In silico experimental design: marginal uncertainty of log10(beta1) by observation scheme and window.

Reads experimental_design.csv written by experimental_design.py.
"""
import csv
import os

import matplotlib.pyplot as plt

from _paths import FIGURES, STYLE, TMC

plt.rcParams.update(STYLE)
COLORS = {'T': '#1f5fa8', 'T+M': '#e08214', 'T+M+C': '#2a9d5c'}
MARKERS = {'T': 'o', 'T+M': 's', 'T+M+C': 'D'}
LABELS = {'T': r'observe $T$', 'T+M': r'observe $T, M$', 'T+M+C': r'observe $T, M, C$'}
TITLES = {'escape_SL': 'escape regime (long-hinge estimate)', 'coexistence_Hopf': 'coexistence regime (near the Hopf boundary)'}


def main():
    rows = list(csv.DictReader(open(os.path.join(TMC, 'experimental_design.csv'), newline='', encoding='utf-8')))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, regime in zip(axes, ['escape_SL', 'coexistence_Hopf']):
        for scheme in ['T', 'T+M', 'T+M+C']:
            sel = sorted((int(r['window_days']), float(r['std_log10_beta1'])) for r in rows
                         if r['regime'] == regime and r['design'] == scheme)
            ax.plot([s[0] for s in sel], [s[1] for s in sel], marker=MARKERS[scheme], color=COLORS[scheme],
                    lw=2.0, ms=7, mec='k', mew=0.5, label=LABELS[scheme])
        ax.axhline(0.3, color='#2a9d5c', ls='--', lw=1.4, label=r'target: std $<0.3$')
        ax.set_yscale('log')
        ax.set_xlabel('observation window (days)')
        ax.set_title(TITLES[regime], fontsize=11.5)
    axes[0].set_ylabel(r'marginal std of $\log_{10}\beta_1$, log scale')
    axes[1].legend(loc='upper left', fontsize=9.6, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'experimental_design_beta1.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('saved experimental_design_beta1.png in', FIGURES)


if __name__ == '__main__':
    main()
