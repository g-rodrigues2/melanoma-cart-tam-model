"""Eigenvalue spectrum of the Fisher information matrix at the long-hinge maximum-likelihood estimate.

Reads fim_eigenvalues_cart_sl_b16.csv written by fim_identifiability_map.py.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from _paths import FIGURES, STYLE, TMC

plt.rcParams.update(STYLE)


def main():
    w = np.sort(np.loadtxt(os.path.join(TMC, 'fim_eigenvalues_cart_sl_b16.csv'), skiprows=1))[::-1]
    tol = w.max() * 1e-12
    above = w > tol
    x = np.arange(1, len(w) + 1)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(x, np.clip(w, tol * 1e-3, None), color=['#2a9d5c' if a else '#c0392b' for a in above],
           edgecolor='k', lw=0.6, width=0.65)
    ax.axhline(tol, color='k', ls='--', lw=1.3)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xlabel('eigenvalue rank (largest to smallest)')
    ax.set_ylabel('eigenvalue of the Fisher information, log scale')
    cond = w.max() / w[above][-1]
    ax.legend(handles=[Patch(fc='#2a9d5c', ec='k', label=r'above the numerical threshold $10^{-12}\lambda_{\max}$'),
                       Patch(fc='#c0392b', ec='k', label='below the threshold (near-null direction)')],
              loc='upper right', fontsize=9.3, framealpha=0.95,
              title=rf'$\lambda_{{\max}}/\lambda_{{\min}}$ above threshold $\approx {cond:.1e}$', title_fontsize=9.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'fim_eigenvalue_spectrum.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'eigenvalues {w.min():.2e}..{w.max():.2e}; below threshold: {int((~above).sum())}; cond above threshold {cond:.2e}')


if __name__ == '__main__':
    main()
