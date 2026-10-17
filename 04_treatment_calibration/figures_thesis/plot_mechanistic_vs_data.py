"""Confrontation of the mechanistic models with the tumor-volume data (Section 3.4).

Companion of plot_model_vs_data.py, which shows the empirical killing terms. Here the two
mechanistic models are drawn against the data at their bounded maximum-likelihood estimates
(Output_Calibration/TMC/mle_bounded_*.npz, the estimates reported in the thesis table):

  _mac_treat  the full T-M-C system, eight parameters
  _mac        the T-M model, which is the same system with mu1 = 0 and no CAR T-cell compartment

One panel per CAR T-cell group, with the observations of every mouse, the group mean, and the
mean trajectory of the two models over the mice of the group, each mouse keeping its own
estimated initial volume. The annotation gives the two log-likelihoods, the likelihood-ratio
statistic of the T-M model against the full model, and the concordance correlation coefficient
of the full model. The conservative reference for the statistic is the 95% quantile of the
chi-square distribution with five degrees of freedom, 11.07, one degree for each parameter that
vanishes when mu1 = 0.

Palette and typography are those of the published figures of this chapter (see _paths.STYLE):
the data in near-black, the full model in the red used for the selected model in the companion
figure, the comparison model in gray. Identity is carried by line style as well as by color, so
the panels survive grayscale printing.

  jilani_mechanistic_vs_data.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_treatment_data
from metrics import ccc, pcc
from models import _split_theta, solve_model

from _paths import CALIBRATION_DIR, FIGURES, GROUP_LABELS, STYLE

plt.rcParams.update(STYLE)
# the figure is scaled to the text width of the thesis, so the fonts are set larger here
plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 13.0})

OUT_DIR = os.path.join(CALIBRATION_DIR, 'Output_Calibration')
TMC_DIR = os.path.join(OUT_DIR, 'TMC')
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
COL_DATA, COL_TM, COL_TMC = '#2b2b2b', '#7f7f7f', '#c0392b'
CHI2_5_95 = 11.07
N_DRAWS = 500            # posterior draws of the agreement metrics, as in Chapter 4
SEED = 20261016


def mle(group, model):
    """Bounded maximum-likelihood estimate of one model and group: (theta, log-likelihood)."""
    name = f'mle_bounded_{group}.npz' if model == '_mac_treat' else f'mle_bounded_{group}{model}.npz'
    z = np.load(os.path.join(TMC_DIR, name), allow_pickle=True)
    return np.asarray(z['theta'], float), float(z['ll'])


def curves(group, model, theta, mice, t_grid):
    """Mean trajectory over the mice and the agreement metrics at the observation times."""
    ode_args, ic_fn = _split_theta(theta, model, len(mice))
    traj, obs, pred = [], [], []
    for idx, mouse in enumerate(mice):
        ic = ic_fn(idx)
        sol = solve_model(model, t_grid, ode_args, ic)
        traj.append(sol[:, 0] if sol.ndim == 2 else sol)
        d = mouse['data']
        sol_obs = solve_model(model, d[:, 0], ode_args, ic)
        obs.append(d[:, 1])
        pred.append(sol_obs[:, 0] if sol_obs.ndim == 2 else sol_obs)
    p, o = np.concatenate(pred), np.concatenate(obs)
    return np.mean(traj, axis=0), ccc(p, o), pcc(p, o)


def metrics_at(theta, model, mice):
    """CCC and PCC of one parameter vector, or None when the solution is not finite."""
    ode_args, ic_fn = _split_theta(np.asarray(theta, float), model, len(mice))
    obs, pred = [], []
    for i, mouse in enumerate(mice):
        d = mouse['data']
        sol = solve_model(model, d[:, 0], ode_args, ic_fn(i))
        y = sol[:, 0] if sol.ndim == 2 else sol
        if not np.all(np.isfinite(y)):
            return None
        obs.append(d[:, 1])
        pred.append(y)
    p, o = np.concatenate(pred), np.concatenate(obs)
    return ccc(p, o), pcc(p, o)


def posterior_metrics(group, model, mice):
    """Posterior median and 95% credible interval of CCC and PCC, as reported in Chapter 4.

    The draws are taken from the stored posterior sample of the run (burn-in already discarded).
    """
    chain = np.loadtxt(os.path.join(OUT_DIR, f'multi_chain_{group}{model}.gz'))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(chain), size=min(N_DRAWS, len(chain)), replace=False)
    vals = [metrics_at(chain[i], model, mice) for i in idx]
    vals = np.array([v for v in vals if v is not None])
    out = []
    for col in (0, 1):
        out.append((np.median(vals[:, col]), np.percentile(vals[:, col], 2.5),
                    np.percentile(vals[:, col], 97.5)))
    return out, len(vals)


def main():
    full = load_treatment_data()
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

        th_tm, ll_tm = mle(group, '_mac')
        th_tmc, ll_tmc = mle(group, '_mac_treat')
        curve_tm, ccc_tm, pcc_tm = curves(group, '_mac', th_tm, mice, t_grid)
        curve_tmc, ccc_tmc, pcc_tmc = curves(group, '_mac_treat', th_tmc, mice, t_grid)
        lrt = 2.0 * (ll_tmc - ll_tm)

        ax.plot(t_grid, curve_tm, ls='--', lw=2.2, color=COL_TM, zorder=4,
                label=r'$T$-$M$ model')
        ax.plot(t_grid, curve_tmc, ls='-', lw=2.2, color=COL_TMC, zorder=3,
                label=r'$T$-$M$-$C$ model')

        ax.set_yscale('log')
        ax.set_xlim(-0.5, t_max + 0.5)
        anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top', fontsize=16)
        ax.annotate(f'{panel})', fontweight='bold', **anchor)
        ax.annotate(GROUP_LABELS[group], xytext=(20, 0), textcoords='offset points', **anchor)
        # the agreement metrics of the rest of the thesis, posterior median with its 95% credible
        # interval, in the format of the Chapter 4 figures; the likelihood numbers stay in the text
        (ccc_post, pcc_post), n_draws = posterior_metrics(group, '_mac_treat', mice)
        (ccc_tm_post, pcc_tm_post), _ = posterior_metrics(group, '_mac', mice)
        # both readings are given, because they differ in the long-hinge group and the difference is
        # itself the result: the drawn curve is the maximum-likelihood fit, which the sampler misses
        ax.text(0.97, 0.04,
                'CCC $= %.3f$  $[%.3f;\\,%.3f]$,  ML $%.3f$' % (ccc_post + (ccc_tmc,)) + '\n'
                + 'PCC $= %.3f$  $[%.3f;\\,%.3f]$,  ML $%.3f$' % (pcc_post + (pcc_tmc,)),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10.5,
                bbox=dict(fc='white', ec='0.7', lw=0.6, alpha=0.92))
        print(f'{group}: ll={ll_tmc:.2f}/{ll_tm:.2f} 2dl={lrt:.2f} | '
              f'T-M-C posterior CCC {ccc_post[0]:.3f} [{ccc_post[1]:.3f}; {ccc_post[2]:.3f}] '
              f'PCC {pcc_post[0]:.3f} [{pcc_post[1]:.3f}; {pcc_post[2]:.3f}] ({n_draws} draws) | '
              f'at the MLE CCC {ccc_tmc:.3f} PCC {pcc_tmc:.3f} | '
              f'T-M posterior CCC {ccc_tm_post[0]:.3f} [{ccc_tm_post[1]:.3f}; {ccc_tm_post[2]:.3f}], '
              f'at the MLE CCC {ccc_tm:.3f}')

    # the curves of the short-hinge panel run through the upper left corner, where the legend
    # sits, so the shared axis is given headroom instead of letting the box cover them
    axes[0].set_ylim(top=2.0e4)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=13.0, framealpha=0.95)
    axes[0].set_ylabel(r'tumor volume (mm$^3$), log scale')
    axes[1].set_xlabel('time since adoptive transfer (days)', fontsize=16)
    fig.tight_layout()
    out = os.path.join(FIGURES, 'jilani_mechanistic_vs_data.png')
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('figure saved:', out)


if __name__ == '__main__':
    main()
