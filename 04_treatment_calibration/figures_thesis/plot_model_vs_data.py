"""Confrontation of the calibrated models with the tumor-volume data (Section 3.4.1).

One panel per CAR T-cell group, with the observations of every mouse, the group mean, and the
maximum-likelihood trajectory of two models: the no-treatment baseline (exponential growth at the
rate calibrated on the PBS controls) and the killing term selected by BIC in that group. The
annotation gives Delta BIC of the selected term against the baseline and the concordance
correlation coefficient of each fit.

  jilani_model_vs_data.png
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_treatment_data
from metrics import ccc
from models import _split_theta, solve_model

from _paths import CALIBRATION_DIR, FIGURES, GROUP_LABELS, STYLE

plt.rcParams.update(STYLE)
# the figure is scaled to the text width of the thesis, so the fonts are set larger here
plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 13.0})

OUT = os.path.join(CALIBRATION_DIR, 'Output_Calibration')
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
BASELINE = '_exp'
TREATMENT_TERMS = {
    '_exp_ED', '_exp_CD', '_exp_AccelED', '_exp_AccelCD', '_exp_MM', '_exp_LS', '_exp_PF',
    '_exp_dePR', '_exp_dePRDecay', '_exp_CDwithLog', '_exp_CDwithExp', '_exp_CDwithSin',
    '_exp_EDwithLog', '_exp_EDwithExp', '_exp_EDwithSin',
}
TERM_LABELS = {
    '_exp_LS': 'log-skew term', '_exp_CD': 'constant-decay term',
    '_exp_EDwithSin': 'sinusoidal killing term',
}
COL_DATA, COL_BASE, COL_FIT = '#2b2b2b', '#7f7f7f', '#c0392b'
label_bic = r'$\Delta$BIC $= %+.1f$'


def bic_table():
    with open(os.path.join(OUT, 'model_selection_table.csv'), newline='') as f:
        return {(r['group'], r['model']): float(r['BIC']) for r in csv.DictReader(f)}


def best_term(bic, group):
    cand = [(bic[(group, m)], m) for m in TREATMENT_TERMS if (group, m) in bic]
    return min(cand)[1]


def fit(group, model, mice, t_grid):
    """Mean trajectory over the mice and CCC at the observation times, at the maximum likelihood."""
    theta = np.load(os.path.join(OUT, f'multi_ll_pars_{group}{model}.npz'), allow_pickle=True)['pars']
    ode_args, ic_fn = _split_theta(theta, model, len(mice))
    curves, obs, pred = [], [], []
    for idx, mouse in enumerate(mice):
        ic = ic_fn(idx)
        sol = solve_model(model, t_grid, ode_args, ic)
        curves.append(sol[:, 0] if sol.ndim == 2 else sol)
        d = mouse['data']
        sol_obs = solve_model(model, d[:, 0], ode_args, ic)
        obs.append(d[:, 1])
        pred.append(sol_obs[:, 0] if sol_obs.ndim == 2 else sol_obs)
    return np.mean(curves, axis=0), ccc(np.concatenate(pred), np.concatenate(obs))


def main():
    full = load_treatment_data()
    bic = bic_table()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3), sharey=True)

    for ax, group, panel in zip(axes, GROUPS, 'abc'):
        mice = full[group]
        t_max = max(m['data'][-1, 0] for m in mice)
        t_grid = np.linspace(0.0, t_max, 200)

        for mouse in mice:
            d = mouse['data']
            ax.plot(d[:, 0], d[:, 1], 'o', ms=2.6, color=COL_DATA, alpha=0.35, zorder=2)
        by_day = {}
        for mouse in mice:
            for t, v in mouse['data']:
                by_day.setdefault(round(float(t)), []).append(float(v))
        days = np.array(sorted(by_day))
        means = np.array([np.mean(by_day[d]) for d in days])
        ax.plot(days, means, 'o-', ms=5.0, lw=1.0, color=COL_DATA, zorder=5,
                label='data (group mean)')

        term = best_term(bic, group)
        base_curve, base_ccc = fit(group, BASELINE, mice, t_grid)
        term_curve, term_ccc = fit(group, term, mice, t_grid)
        d_bic = bic[(group, term)] - bic[(group, BASELINE)]

        ax.plot(t_grid, base_curve, ls='--', lw=2.2, color=COL_BASE, zorder=4,
                label='no treatment')
        ax.plot(t_grid, term_curve, ls='-', lw=2.2, color=COL_FIT, zorder=3,
                label='best killing term')

        ax.set_yscale('log')
        ax.set_xlim(-0.5, t_max + 0.5)
        # panel label inside the axes, in bold, as in the published figures of this chapter
        # (fig5, fig8, fig10, fig12: lowercase letter with a closing parenthesis, Times bold)
        anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top', fontsize=16)
        ax.annotate(f'{panel})', fontweight='bold', **anchor)
        # the gap is given in points, so it does not depend on how wide the panel is
        ax.annotate(GROUP_LABELS[group], xytext=(20, 0), textcoords='offset points', **anchor)
        ax.text(0.97, 0.04,
                TERM_LABELS.get(term, term) + '\n' + label_bic % d_bic + '\n'
                + 'CCC $= %.3f$ / $%.3f$' % (term_ccc, base_ccc),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=12.5,
                bbox=dict(fc='white', ec='0.7', lw=0.6, alpha=0.92))
        print(f'{group}: term={term} dBIC={d_bic:+.2f} CCC_term={term_ccc:.4f} CCC_base={base_ccc:.4f}')

    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=13.0,
                   framealpha=0.95)
    axes[0].set_ylabel(r'tumor volume (mm$^3$), log scale')
    # the shared x-label sits on the middle panel: a figure-level supxlabel leaves an
    # empty strip under the axes, which wastes page height once scaled to the text width
    axes[1].set_xlabel('time since adoptive transfer (days)', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'jilani_model_vs_data.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('figure saved.')


if __name__ == '__main__':
    main()
