"""Figure of the time-resolved dummy-parameter test (Section 3.3.3).

Reads results/dummy_test_time.csv and draws, for each output, which total-effect indices are
distinguishable from the dummy at each analyzed time. What the figure is for is the contrast between a
persistent core, significant at every time, and parameters whose significance flickers in and out: an
index that sits at the level of the estimator noise crosses the threshold only intermittently, which a
single test at the final time cannot show.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# two steps of one blue hue: monotone in lightness, single hue, the light end still visible against the
# page (validated as an ordinal ramp). The distinction is carried by lightness, so it survives grayscale
# printing and color-vision deficiency.
C_SIG = '#1c5cab'
C_NOT = '#86b6ef'

OUTPUTS = (('T', r'$T(t)$', 'a)'), ('M', r'$M(t)$', 'b)'), ('C', r'$C(t)$', 'c)'))

PARAM_LABEL = {
    'alpha1': r'$\alpha_1$', 'alpha2': r'$\alpha_2$', 'alpha3': r'$\alpha_3$',
    'beta1': r'$\beta_1$', 'K1': r'$K_1$', 'K2': r'$K_2$', 'K3': r'$K_3$',
    'mu1': r'$\mu_1$', 'kappa2': r'$\kappa_2$', 'kappa3': r'$\kappa_3$',
}


def _edges(centers):
    """Cell edges of a pcolormesh from the analyzed times."""
    c = np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2
    return np.concatenate(([c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]))


def main():
    os.makedirs('figures', exist_ok=True)
    df = pd.read_csv('results/dummy_test_time.csv')

    times = np.sort(df['time'].unique())
    params = [p for p in PARAM_LABEL if p in set(df['parameter'])]
    x_edges = _edges(times)
    y_edges = np.arange(len(params) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    cmap = ListedColormap([C_NOT, C_SIG])

    for ax, (out, var_label, letter) in zip(axes, OUTPUTS):
        sub = df[df['output'] == out]
        # rows are drawn bottom-up, so the parameter order is reversed to read top-down in the figure
        grid = np.array([
            sub[sub['parameter'] == p].sort_values('time')['ST_significant'].to_numpy(dtype=float)
            for p in reversed(params)
        ])
        ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, vmin=0, vmax=1,
                      edgecolors='white', linewidth=0.6)

        ax.set_yticks(np.arange(len(params)) + 0.5)
        ax.set_yticklabels([PARAM_LABEL[p] for p in reversed(params)], fontsize=14)
        ax.set_ylabel(var_label, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.text(-0.055, 1.04, letter, transform=ax.transAxes, fontsize=16, va='bottom', ha='left', fontweight='bold')

        # direct label of the persistence of each parameter, the quantity the text reports, placed
        # outside the axes so the frame stays tight around the data
        for row, p in enumerate(reversed(params)):
            frac = sub[sub['parameter'] == p]['ST_significant'].mean()
            ax.text(1.008, (row + 0.5) / len(params), f'{100 * frac:.0f}%', transform=ax.transAxes,
                    va='center', ha='left', fontsize=12, color='#3f3f46', clip_on=False)
        ax.set_xlim(x_edges[0], x_edges[-1])
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel('Time (days)', fontsize=16)
    axes[0].legend(
        handles=[Patch(facecolor=C_SIG, edgecolor='white', label='distinguishable from the dummy'),
                 Patch(facecolor=C_NOT, edgecolor='white', label='not distinguishable')],
        loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=14)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig('figures/dummy_test_significance_time.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Figure 'figures/dummy_test_significance_time.png' was saved.")


if __name__ == '__main__':
    main()
