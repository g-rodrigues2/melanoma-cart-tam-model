"""Post-processing: figures, posterior summaries and compression of the HDF5 backends.

Main functions:
  - compress_h5_inplace : gzip compression of the datasets of an HDF5 backend
  - summarize_posterior : median, 95% interval and 95% HDI (arviz optional)
  - posterior_corr_from_chain : CCC/PCC with 95% intervals from 500 posterior samples
  - plot_scatter_ax     : observed versus predicted volumes with CCC/PCC
  - plot_maxll_solution_global : maximum-likelihood fit of each mouse
"""
import glob
import math
import os
import string

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter

from metrics import ccc, pcc
from models import solve_model, _split_theta

# The dimensional <-> dimensionless conversions of the _macAD model are in
# ../03_growth_model_selection/pipeline_control_parallel/postprocess.py.


class ScalarFormatterWithDecimals(ScalarFormatter):
    def _set_format(self):
        self.format = '%.1f'


# === HDF5 utilities ===

def compress_h5_inplace(path, complevel=4, min_size=1024):
    """Rewrites the HDF5 file with gzip compression of the datasets larger than min_size.

    gzip is a native HDF5 filter, transparent to emcee. maxshape is preserved, so the backend remains
    resumable. Typical reduction: 3-8x.
    """
    import h5py
    tmp = path + ".tmp"
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for k in src.attrs:
            dst.attrs[k] = src.attrs[k]

        def copy_group(srcg, dstg):
            for k in srcg.attrs:
                dstg.attrs[k] = srcg.attrs[k]
            for name, item in srcg.items():
                if isinstance(item, h5py.Group):
                    copy_group(item, dstg.create_group(name))
                else:
                    data = item[...]
                    kwargs = dict(data=data, maxshape=item.maxshape)
                    if data.nbytes > min_size:
                        kwargs.update(compression='gzip',
                                      compression_opts=complevel,
                                      chunks=True)
                    ds = dstg.create_dataset(name, **kwargs)
                    for k in item.attrs:
                        ds.attrs[k] = item.attrs[k]

        copy_group(src, dst)
    os.replace(tmp, path)


# === Posterior summaries ===

def summarize_posterior(arr, alpha=0.05):
    """Returns (median, lo, hi, hdi_lo, hdi_hi). The HDI comes from arviz when available; otherwise the
    percentile interval is returned."""
    arr = np.asarray(arr, dtype=float)
    med = np.median(arr)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    try:
        import arviz as az
        hdi_lo, hdi_hi = az.hdi(arr, hdi_prob=1 - alpha)
    except ImportError:
        hdi_lo, hdi_hi = lo, hi
    return med, lo, hi, hdi_lo, hdi_hi


def posterior_corr_from_chain(chain_path, model_extension, full_data, group,
                              n_samp=500, seed=42):
    """Median and 95% interval of CCC and PCC over n_samp posterior samples."""
    if not os.path.exists(chain_path):
        print(f"[ERROR] Chain not found: {chain_path}")
        return None
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    if group not in full_data:
        print(f"[ERROR] Group '{group}' not found in full_data.")
        return None
    ndim = len(full_data[group])

    n_samp = min(n_samp, chain.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(chain.shape[0], size=n_samp, replace=False)
    thetas = chain[idx, :]

    pcc_samples, ccc_samples = [], []
    for k, theta in enumerate(thetas):
        try:
            ode_args, ic_fn = _split_theta(theta, model_extension, ndim)
            full_model, full_exp = [], []
            for i, mouse in enumerate(full_data[group]):
                t = mouse['data'][:, 0]
                y = mouse['data'][:, 1]
                sol = solve_model(model_extension, t, ode_args, ic_fn(i),
                                  type_sol='data')
                sol = sol[:, 0] if sol.ndim == 2 else sol
                full_model.append(sol)
                full_exp.append(y)
            model_concat = np.concatenate(full_model)
            exp_concat = np.concatenate(full_exp)
            pcc_val = float(np.clip(pcc(model_concat, exp_concat), -1.0, 1.0))
            ccc_val = float(np.clip(ccc(model_concat, exp_concat), -1.0, 1.0))
            pcc_samples.append(pcc_val)
            ccc_samples.append(ccc_val)
        except Exception as e:
            print(f"[WARN] sample failure {k}: {e}")

    pcc_samples = np.asarray(pcc_samples, dtype=float)
    ccc_samples = np.asarray(ccc_samples, dtype=float)
    pcc_summary = summarize_posterior(pcc_samples, alpha=0.05)
    ccc_summary = summarize_posterior(ccc_samples, alpha=0.05)

    print(f"\n[Posterior {group} | {model_extension}]  n={len(pcc_samples)}")
    print(f"PCC: median={pcc_summary[0]:.3f}, CI95%=[{pcc_summary[1]:.3f}, {pcc_summary[2]:.3f}], HDI95%=[{pcc_summary[3]:.3f}, {pcc_summary[4]:.3f}]")
    print(f"CCC: median={ccc_summary[0]:.3f}, CI95%=[{ccc_summary[1]:.3f}, {ccc_summary[2]:.3f}], HDI95%=[{ccc_summary[3]:.3f}, {ccc_summary[4]:.3f}]")

    return {
        "pcc_samples": pcc_samples, "ccc_samples": ccc_samples,
        "pcc_summary": pcc_summary, "ccc_summary": ccc_summary,
    }


# === Plots ===

def configure_plot_settings(fontsize):
    plt.rcParams['font.size'] = fontsize
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 2))
    return formatter


def find_max_time_per_group(full_data):
    return {grp: max(max(m['data'][:, 0]) for m in mice)
            for grp, mice in full_data.items()}


def finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model,
                  save, show, figure_name, formatter):
    final_exp = np.concatenate(full_exp, axis=0) if full_exp else np.array([])
    final_model = np.concatenate(full_model, axis=0) if full_model else np.array([])
    if final_exp.size == 0 or final_model.size == 0:
        print("Error: One of the final arrays is empty.")
        return
    rounded_max = math.ceil(np.max(np.concatenate([final_exp, final_model])) / 1000) * 1000
    for ax in axes[:nScenarios]:
        ax.set_ylim((0, rounded_max))
        ax.yaxis.set_major_formatter(formatter)
    for i in range(nScenarios, nRows * nCols):
        fig.delaxes(axes[i])
    if save:
        plt.savefig(figure_name + '.pdf', bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close()


def plot_scatter_ax(ax, final_exp, final_model, formatter,
                    ccc_val, ccc_ic, pcc_val, pcc_ic):
    for i in range(len(final_exp)):
        ax.plot(final_exp[i], final_model[i], 'o', color='blue')

    ax.set_xlabel(r'Data - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)
    ax.set_ylabel(r'Model - Tumor Volume ($\times 10^3$ mm³)', fontsize=20)

    line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='dashed')
    line.set_transform(ax.transAxes)
    ax.add_line(line)

    final_exp = np.concatenate(final_exp)
    final_model = np.concatenate(final_model)
    max_value = math.ceil(max(max(final_exp), max(final_model)) / 1000) * 1000
    ticks = np.linspace(0, max_value, num=int(max_value / 1000) + 1)
    ax.set_xlim((0, max_value)); ax.set_ylim((0, max_value))
    ax.set_xticks(ticks); ax.set_yticks(ticks)

    fmt = ScalarFormatterWithDecimals(useMathText=True)
    fmt.set_scientific(True); fmt.set_powerlimits((-3, 2))
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 2), useOffset=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', length=6, width=1,
                   direction='out', color='black', bottom=True, left=True)
    ax.minorticks_off()

    pccT = pcc(final_model, final_exp)
    cccT = ccc(final_model, final_exp)
    subplot_index = getattr(ax, 'get_subplotspec', lambda: None)()
    label_prefix = ''
    if subplot_index is not None:
        label_prefix = 'A) ' if subplot_index.colspan.start == 0 else 'B) '

    if pcc_ic is not None and ccc_ic is not None:
        ax.text(0.06, 0.90,
                f'{label_prefix}CCC = {ccc_val:.3f}  [{ccc_ic[0]:.3f}; {ccc_ic[1]:.3f}]',
                transform=ax.transAxes, fontsize=18, verticalalignment='top')
        ax.text(0.058, 0.835,
                f'     PCC = {pcc_val:.3f}  [{pcc_ic[0]:.3f}; {pcc_ic[1]:.3f}]',
                transform=ax.transAxes, fontsize=18, verticalalignment='top')
    else:
        ax.text(0.06, 0.9,
                f'{label_prefix}CCC/PCC = {cccT:.2f}/{pccT:.2f}',
                transform=ax.transAxes, fontsize=18, verticalalignment='top')


def plot_maxll_solution_global(files_location, full_data, nCols=3, show=True,
                               save=True, fontsize='14', figure_name='maxll_figure',
                               group_name=None, model=None):
    """Maximum-likelihood fit of each mouse and combined scatter plot of the two control groups.

    `files_location` may be a glob over several (group, model) pairs, in which case the group and the
    model extension are read from each file name. That reading splits at the last underscore and so only
    works for single-token extensions such as `_exp`; pass `group_name` and `model` explicitly for the
    treatment models, whose extensions have more than one token (`_exp_ED`, `_exp_dePRDecay`).
    """
    if not show and not save:
        return

    all_chains = sorted(glob.glob(files_location))
    formatter = configure_plot_settings(fontsize)

    best_chain_per_group = {}
    max_ll_per_group = {}
    for chain in all_chains:
        npzfile = np.load(chain)
        max_ll = npzfile['max_ll']
        stem = os.path.basename(chain).replace('multi_ll_pars_', '').rsplit('.', 1)[0]
        group = group_name if group_name is not None else stem.rsplit('_', 1)[0]
        if group not in max_ll_per_group or max_ll > max_ll_per_group[group]:
            max_ll_per_group[group] = max_ll
            best_chain_per_group[group] = chain

    max_times = find_max_time_per_group(full_data)
    scatter_pairs = {}

    for group, best_chain in best_chain_per_group.items():
        print(f"Processing: Group = {group}, max_ll = {max_ll_per_group[group]}")
        npzfile = np.load(best_chain)
        theta = npzfile['pars']

        nScenarios = len(full_data[group])
        nRows = int(np.ceil(nScenarios / nCols))
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols,
                                 figsize=(nCols * 11, nRows * 5))
        axes = axes.ravel()

        full_exp, full_model = [], []
        model_extension = model if model is not None else \
            '_' + os.path.basename(best_chain).rsplit('.', 1)[0].rsplit('_', 1)[1]
        ndim = len(full_data[group])
        print(f'Size pars = {len(theta)}, animals = {ndim}')
        max_time_group = max_times[group]
        ode_args, ic_fn = _split_theta(theta, model_extension, ndim)

        for idx, mouse_data in enumerate(full_data[group]):
            ax = axes[idx]
            data = mouse_data['data']
            ax.set_ylabel('Tumor Volume (mm³)')
            ax.text(0.05, .9,
                    f'{string.ascii_uppercase[idx]}) {mouse_data["name"]}',
                    horizontalalignment='left', transform=ax.transAxes, fontsize=18)
            ax.plot(data[:, 0], data[:, 1], 'o', color='black', markersize=10,
                    label='Tumor Volume')
            ax.set_xlabel('Time (days)')
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim((-1, max_time_group + 1))

            solution = solve_model(model_extension, data[:, 0], ode_args,
                                   ic_fn(idx), "smooth")
            mask = np.isin(solution[:, 0], data[:, 0])
            matched_t = solution[mask, 0]
            matched_v = solution[mask, 1]
            ax.plot(matched_t, matched_v, color='black', linestyle='dashed',
                    linewidth=2, label='Model')

            full_model.append(matched_v); full_exp.append(data[:, 1])
            pccT = pcc(matched_v, data[:, 1])
            cccT = ccc(matched_v, data[:, 1])
            ax.text(0.5, .05, f'CCC/PCC = {cccT:.2f}/{pccT:.2f}',
                    horizontalalignment='left', transform=ax.transAxes)

        finalize_plot(fig, axes, nScenarios, nCols, nRows, full_exp, full_model,
                      save, show, figure_name, formatter)
        scatter_pairs[group] = (full_exp.copy(), full_model.copy())

    # Combined scatter plot of the two control groups
    if 'control_b16' in scatter_pairs and 'control_pdx' in scatter_pairs:
        (exp_s, mod_s) = scatter_pairs['control_b16']
        (exp_r, mod_r) = scatter_pairs['control_pdx']
        fig_sc, axes_sc = plt.subplots(1, 2, figsize=(22, 15 * (1.4 / 3.4)))

        res_cs = posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_b16' + model_extension + '.gz',
            model_extension=model_extension, full_data=full_data,
            group='control_b16', n_samp=500, seed=42)
        res_cr = posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_pdx' + model_extension + '.gz',
            model_extension=model_extension, full_data=full_data,
            group='control_pdx', n_samp=500, seed=42)

        if res_cs and res_cr:
            ccc_cs = res_cs["ccc_summary"]; pcc_cs = res_cs["pcc_summary"]
            ccc_cr = res_cr["ccc_summary"]; pcc_cr = res_cr["pcc_summary"]
            plot_scatter_ax(axes_sc[0], exp_s, mod_s, formatter,
                            ccc_val=ccc_cs[0], ccc_ic=[ccc_cs[1], ccc_cs[2]],
                            pcc_val=pcc_cs[0], pcc_ic=[pcc_cs[1], pcc_cs[2]])
            plot_scatter_ax(axes_sc[1], exp_r, mod_r, formatter,
                            ccc_val=ccc_cr[0], ccc_ic=[ccc_cr[1], ccc_cr[2]],
                            pcc_val=pcc_cr[0], pcc_ic=[pcc_cr[1], pcc_cr[2]])

            fig_sc.tight_layout()
            if save:
                fig_sc.savefig('./Output_Calibration/multi_scatter_both' + model_extension + '.png',
                               dpi=300, bbox_inches='tight', pad_inches=0.02)
                fig_sc.savefig('./Output_Calibration/multi_scatter_both' + model_extension + '.pdf',
                               bbox_inches='tight', pad_inches=0.02)
            if show:
                plt.show()
            else:
                plt.close(fig_sc)
