"""Prior predictive check of the CAR T-cell treatment models.

Figure of each (group, model):
  - one band pooling all mice of the group, with stacked quantile ranges
        1-99%   (lightest, alpha=0.10)
        5-95%   (alpha=0.18)
        25-75%  (darkest, alpha=0.28)
  - thin black lines: mean over mice of simulated trajectories of individual prior draws
  - observed volumes of all mice in red (number of observations in the title)
  - box with the pooled coverage, the interquartile range and mean/min/max of the coverage per mouse, and the
    numbers of mice, times and draws

Pooled coverage: percentage of all observations of the group inside the 5-95% band (target 100%).

Usage:
    python prior_predictive.py [--n 1000] [--model _exp_ED] [--group cart_sl_b16]
                               [--seed 42] [--outdir Output_Calibration]
                               [--noise / --no-noise]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from models import solve_model, _split_theta


DEFAULT_MODELS = [
    '_exp',
    '_exp_ED', '_exp_CD', '_exp_AccelED', '_exp_AccelCD',
    '_exp_MM', '_exp_LS', '_exp_PF',
    '_exp_dePR', '_exp_dePRDecay',
    '_exp_CDwithLog', '_exp_CDwithExp', '_exp_CDwithSin',
    '_exp_EDwithLog', '_exp_EDwithExp', '_exp_EDwithSin',
]
DEFAULT_GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']


def _first_ic_index(labels):
    for i, lbl in enumerate(labels):
        if lbl.startswith('ic'):
            return i
    raise ValueError("labels without ic*")


def prior_predictive_for_group(full_data, group, model_extension, max_ic,
                               num_samples=1000, use_measurement_noise=True,
                               seed=42):
    """Samples the prior, simulates the trajectory of every mouse and pools them.

    Returns a dict with
      times              union of the observation times of the group
      q01/q05/q25/q50/q75/q95/q99  quantiles over all simulations
      mean_per_draw      num_samples x len(times) mean over mice of each draw (thin lines of the figure)
      stats              pooled coverage ('global': % of observations inside the 5-95% band) and mean, median,
                         min, max, q25, q75 of the coverage per mouse
      n_obs_total        number of observations of the group
      n_mice             number of mice
      n_draws            number of prior draws
    """
    rng = np.random.default_rng(seed)
    group_size = len(full_data[group])

    l_bound, u_bound, labels = define_bounds_labels(
        model_extension, group_size=group_size, max_ic=max_ic,
        bool_multiple_mice=True,
    )
    l_bound = np.array(l_bound); u_bound = np.array(u_bound)
    i_std = len(labels) - 1

    # observation times of the group
    unique_times = sorted({float(t) for m in full_data[group] for t in m['data'][:, 0]})
    times = np.array(unique_times, dtype=float)

    # indices of the times of each mouse in `times`
    idx_mouse_in_times = []
    for m in full_data[group]:
        mt = m['data'][:, 0]
        idx = np.nonzero(np.isin(times, mt))[0]
        idx_mouse_in_times.append(idx)

    # (num_samples * num_mice, len(times)): all simulations
    sim_all = np.zeros((num_samples * group_size, len(times)), dtype=float)
    # thin lines: mean over mice of each draw (num_samples x len(times))
    mean_per_draw = np.zeros((num_samples, len(times)), dtype=float)

    valid_mask = np.ones(num_samples * group_size, dtype=bool)
    for j in range(num_samples):
        theta = rng.uniform(l_bound, u_bound)
        std = theta[i_std]
        # _split_theta builds the initial condition (1D, 2D or 3D)
        ode_args, ic_fn = _split_theta(theta, model_extension, group_size)

        per_mouse_this_draw = np.zeros((group_size, len(times)), dtype=float)
        for idx_mouse, mouse_data in enumerate(full_data[group]):
            ic = ic_fn(idx_mouse)
            try:
                sol = solve_model(
                    model_extension=model_extension,
                    time_array=times,
                    parameters=ode_args,
                    initial_condition=ic,
                    type_sol='data',
                )
                pred = sol[:, 0] if sol.ndim == 2 else np.asarray(sol).ravel()
            except Exception:
                pred = np.full(len(times), np.nan)

            if use_measurement_noise and std > 0:
                pred = pred + rng.normal(0.0, std, size=pred.shape)
            pred = np.clip(pred, 0.0, None)

            row = j * group_size + idx_mouse
            sim_all[row, :] = pred
            per_mouse_this_draw[idx_mouse, :] = pred
            if not np.all(np.isfinite(pred)):
                valid_mask[row] = False

        mean_per_draw[j, :] = per_mouse_this_draw.mean(axis=0)

    sim_valid = sim_all[valid_mask]
    if len(sim_valid) == 0:
        sim_valid = sim_all  # fallback

    # pooled band
    q01 = np.percentile(sim_valid,  1, axis=0)
    q05 = np.percentile(sim_valid,  5, axis=0)
    q25 = np.percentile(sim_valid, 25, axis=0)
    q50 = np.percentile(sim_valid, 50, axis=0)
    q75 = np.percentile(sim_valid, 75, axis=0)
    q95 = np.percentile(sim_valid, 95, axis=0)
    q99 = np.percentile(sim_valid, 99, axis=0)

    # pooled coverage (all observations against the 5-95% band)
    n_in, n_total = 0, 0
    coverage_per_mouse = []
    for idx_mouse, mouse_data in enumerate(full_data[group]):
        obs = mouse_data['data'][:, 1]
        inds = idx_mouse_in_times[idx_mouse]
        in_band = (obs >= q05[inds]) & (obs <= q95[inds])
        n_in += int(in_band.sum())
        n_total += len(obs)
        coverage_per_mouse.append(in_band.mean() * 100.0)

    coverage_global = (n_in / max(n_total, 1)) * 100.0
    coverage_per_mouse = np.asarray(coverage_per_mouse)

    stats = {
        'global':  float(coverage_global),
        'mean':    float(np.mean(coverage_per_mouse)),
        'median':  float(np.median(coverage_per_mouse)),
        'min':     float(np.min(coverage_per_mouse)),
        'max':     float(np.max(coverage_per_mouse)),
        'q25':     float(np.percentile(coverage_per_mouse, 25)),
        'q75':     float(np.percentile(coverage_per_mouse, 75)),
    }

    return {
        'times': times,
        'q01': q01, 'q05': q05, 'q25': q25, 'q50': q50,
        'q75': q75, 'q95': q95, 'q99': q99,
        'mean_per_draw': mean_per_draw,
        'stats': stats,
        'n_obs_total': n_total,
        'n_mice': group_size,
        'n_draws': num_samples,
    }


def _ppc_annotation_text(stats, n_mice, n_times, n_draws):
    iqr_low, iqr_high = stats['q25'], stats['q75']
    return (
        f"Coverage: {stats['global']:.1f}% (IQR: {iqr_low:.1f}-{iqr_high:.1f}%)\n"
        f"mean={stats['mean']:.1f}%  min={stats['min']:.1f}%  max={stats['max']:.1f}%\n"
        f"mice={n_mice}  times={n_times}  draws={n_draws}"
    )


def plot_prior_prediction_band(result, full_data, group, save=True, show=False,
                                figure_name='ppc_band', loc_xy=(0.02, 0.98),
                                model_extension=''):
    """Pooled band (three stacked levels), mean lines, data and annotation."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    t = result['times']
    # three stacked band levels (light -> dark)
    ax.fill_between(t, result['q01'], result['q99'], alpha=0.10,
                    color='cornflowerblue', label='1-99%')
    ax.fill_between(t, result['q05'], result['q95'], alpha=0.18,
                    color='cornflowerblue', label='5-95%')
    ax.fill_between(t, result['q25'], result['q75'], alpha=0.28,
                    color='cornflowerblue', label='25-75%')

    # thin mean lines (a subset)
    mpd = result['mean_per_draw']
    rng = np.random.default_rng(0)
    n_lines = min(60, mpd.shape[0])
    idx = rng.choice(mpd.shape[0], size=n_lines, replace=False)
    for i in idx:
        ax.plot(t, mpd[i, :], linewidth=0.8, alpha=0.30, color='black')

    # observed volumes (all mice)
    max_obs = 0.0
    for m in full_data[group]:
        d = m['data']
        ax.scatter(d[:, 0], d[:, 1], s=42, color='red', alpha=0.85,
                   edgecolors='black', linewidths=0.5, zorder=5)
        max_obs = max(max_obs, float(d[:, 1].max()))

    ax.set_xlabel("Time (days after ACT)", fontsize=14)
    ax.set_ylabel("Tumor Volume (mm³)", fontsize=14)
    ax.set_title(
        f"PPC -- group: {group} / model: {model_extension} "
        f"(n_mice={result['n_mice']}, n_obs={result['n_obs_total']})",
        fontsize=12,
    )

    n_times = len(t)
    ax.text(loc_xy[0], loc_xy[1],
            _ppc_annotation_text(result['stats'], result['n_mice'],
                                 n_times, result['n_draws']),
            transform=ax.transAxes, fontsize=9, fontfamily='monospace',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.9, edgecolor='gray'))

    if max_obs > 0:
        ax.set_ylim(bottom=0, top=max_obs * 1.6)
    else:
        ax.set_ylim(bottom=0)

    if save:
        os.makedirs(os.path.dirname(figure_name) or '.', exist_ok=True)
        plt.savefig(f"{figure_name}.pdf", bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f"{figure_name}.png", bbox_inches='tight', pad_inches=0.02, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--model', default=None)
    parser.add_argument('--group', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='Output_Calibration')
    parser.add_argument('--noise', action='store_true', default=True)
    parser.add_argument('--no-noise', dest='noise', action='store_false')
    args = parser.parse_args()

    full_data = load_treatment_data()
    max_ic = find_max_initial_condition(full_data)[0]

    models = [args.model] if args.model else DEFAULT_MODELS
    groups = [args.group] if args.group else DEFAULT_GROUPS

    print(f"{'group':<14} {'model':<22} {'glob':>6} {'mean':>6} {'min':>6} {'max':>6}")
    print("-" * 72)

    overall = []
    for m in models:
        for g in groups:
            if len(full_data.get(g, [])) == 0:
                print(f'[skip] {g}: empty'); continue
            result = prior_predictive_for_group(
                full_data, g, m, max_ic, num_samples=args.n,
                use_measurement_noise=args.noise, seed=args.seed,
            )
            figname = os.path.join(args.outdir, f'prior_predictive_{g}{m}')
            plot_prior_prediction_band(
                result, full_data, g,
                save=True, show=False, figure_name=figname, model_extension=m,
            )
            s = result['stats']
            print(f"{g:<14} {m:<22} "
                  f"{s['global']:6.1f} {s['mean']:6.1f} "
                  f"{s['min']:6.1f} {s['max']:6.1f}")
            overall.append((g, m, s['global']))

    print()
    print("=" * 72)
    print("POOLED COVERAGE BELOW 100% (observations outside the 5-95% band):")
    n_warn = 0
    for g, m, cov in overall:
        if cov < 100.0:
            print(f"  [{g} / {m}] global={cov:.1f}%  -> widen the bounds")
            n_warn += 1
    if n_warn == 0:
        print("  None -- 100% coverage for every pair.")


if __name__ == '__main__':
    main()
