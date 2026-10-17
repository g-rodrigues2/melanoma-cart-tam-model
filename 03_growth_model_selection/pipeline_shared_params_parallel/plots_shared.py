"""Scatter plots (CCC/PCC) and posterior summaries of the multi-group calibration.

Unlike `postprocess.py` of pipeline_control_parallel, the partition of theta depends on the sharing scenario
(`scenarios.split_model_args_for_group` and `scenarios.ic_slice_for_group`).
"""
import _path_setup  # noqa: F401

import math

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from metrics import ccc, pcc
from models import solve_model
from postprocess import summarize_posterior

from scenarios import (
    MODEL_PARAM_NAMES, count_model_params,
    ic_slice_for_group, split_model_args_for_group,
)


class _ScalarFmt(ScalarFormatter):
    def _set_format(self):
        self.format = '%.1f'


def posterior_corr_shared(chain_path, model_extension, scenario, full_data,
                          groups, n_samp=500, seed=42):
    """Median and 95% interval of CCC and PCC for each group over n_samp samples of the joint posterior.
    Returns {group: {pcc_summary, ccc_summary}}.
    """
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)

    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(chain.shape[0], size=n_samp, replace=False)
    thetas = chain[idx, :]

    param_names = MODEL_PARAM_NAMES[model_extension]
    n_model = count_model_params(scenario, groups)

    results = {}
    for group in groups:
        pcc_samples, ccc_samples = [], []
        for theta in thetas:
            try:
                ode_args, _ = split_model_args_for_group(
                    theta, scenario, param_names, groups, group,
                )
                T0s = ic_slice_for_group(theta, n_model, full_data, groups, group)
                full_model, full_exp = [], []
                for i, mouse in enumerate(full_data[group]):
                    t = mouse['data'][:, 0]
                    y = mouse['data'][:, 1]
                    sol = solve_model(model_extension, t, ode_args, float(T0s[i]),
                                      type_sol='data')
                    sol = sol[:, 0] if sol.ndim == 2 else sol
                    full_model.append(sol)
                    full_exp.append(y)
                m = np.concatenate(full_model)
                e = np.concatenate(full_exp)
                pcc_samples.append(float(np.clip(pcc(m, e), -1, 1)))
                ccc_samples.append(float(np.clip(ccc(m, e), -1, 1)))
            except Exception:
                continue
        pcc_arr = np.asarray(pcc_samples)
        ccc_arr = np.asarray(ccc_samples)
        results[group] = {
            'pcc_summary': summarize_posterior(pcc_arr, alpha=0.05),
            'ccc_summary': summarize_posterior(ccc_arr, alpha=0.05),
        }
    return results


def _plot_scatter_panel(ax, full_exp, full_model, label_prefix,
                        ccc_val, ccc_ic, pcc_val, pcc_ic):
    """Same as plot_scatter_ax of pipeline_control_parallel, without assuming its two fixed groups."""
    for i in range(len(full_exp)):
        ax.plot(full_exp[i], full_model[i], 'o', color='blue')

    ax.set_xlabel(r'Data - Tumor Volume ($\times 10^3$ mm$^3$)', fontsize=20)
    ax.set_ylabel(r'Model - Tumor Volume ($\times 10^3$ mm$^3$)', fontsize=20)

    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)

    flat_exp = np.concatenate(full_exp)
    flat_model = np.concatenate(full_model)
    max_value = math.ceil(max(flat_exp.max(), flat_model.max()) / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value / 1000) + 1)
    ax.set_xlim((0, max_value)); ax.set_ylim((0, max_value))
    ax.set_xticks(ticks); ax.set_yticks(ticks)

    fmt = _ScalarFmt(useMathText=True)
    fmt.set_scientific(True); fmt.set_powerlimits((-3, 2))
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 2),
                        useOffset=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', length=6, width=1,
                   direction='out', color='black', bottom=True, left=True)
    ax.minorticks_off()

    ax.text(0.06, 0.90,
            f'{label_prefix} CCC = {ccc_val:.3f}  [{ccc_ic[0]:.3f}; {ccc_ic[1]:.3f}]',
            transform=ax.transAxes, fontsize=18, verticalalignment='top')
    ax.text(0.058, 0.835,
            f'        PCC = {pcc_val:.3f}  [{pcc_ic[0]:.3f}; {pcc_ic[1]:.3f}]',
            transform=ax.transAxes, fontsize=18, verticalalignment='top')


def make_scatter_both(chain_path, npz_path, model_extension, scenario,
                      full_data, groups, figure_name, n_samp=500, seed=42):
    """Scatter plot (1x2): A) first group, B) second group. Points from the best-fit sample of the .npz,
    95% intervals from posterior_corr_shared. Writes PNG and PDF to `figure_name` (no extension).
    """
    plt.rcParams['font.size'] = '14'

    npz = np.load(npz_path)
    theta_best = npz['pars']

    param_names = MODEL_PARAM_NAMES[model_extension]
    n_model = count_model_params(scenario, groups)

    # 95% intervals of CCC/PCC over posterior samples
    corr = posterior_corr_shared(
        chain_path, model_extension, scenario, full_data, groups,
        n_samp=n_samp, seed=seed,
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / 3.4)))
    prefixes = ['A)', 'B)']

    for gi, group in enumerate(groups):
        ode_args, _ = split_model_args_for_group(
            theta_best, scenario, param_names, groups, group,
        )
        T0s = ic_slice_for_group(theta_best, n_model, full_data, groups, group)

        full_exp, full_model = [], []
        for i, mouse in enumerate(full_data[group]):
            t = mouse['data'][:, 0]
            y = mouse['data'][:, 1]
            sol = solve_model(model_extension, t, ode_args, float(T0s[i]),
                              type_sol='data')
            sol = sol[:, 0] if sol.ndim == 2 else sol
            full_exp.append(y)
            full_model.append(sol)

        cs = corr[group]
        ccc_med, ccc_lo, ccc_hi, *_ = cs['ccc_summary']
        pcc_med, pcc_lo, pcc_hi, *_ = cs['pcc_summary']

        _plot_scatter_panel(
            axes[gi], full_exp, full_model, prefixes[gi],
            ccc_val=ccc_med, ccc_ic=(ccc_lo, ccc_hi),
            pcc_val=pcc_med, pcc_ic=(pcc_lo, pcc_hi),
        )

    fig.tight_layout()
    fig.savefig(figure_name + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(figure_name + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    return corr


def parameter_summary(chain_path, labels, params_of_interest):
    """{label: (median, lo, hi)} of the requested parameters present in `labels`."""
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    out = {}
    for name in params_of_interest:
        if name not in labels:
            continue
        idx = labels.index(name)
        med, lo, hi, _hdi_lo, _hdi_hi = summarize_posterior(chain[:, idx], alpha=0.05)
        out[name] = (med, lo, hi)
    return out
