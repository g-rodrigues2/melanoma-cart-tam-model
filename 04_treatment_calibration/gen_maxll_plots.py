"""Best-fit and scatter plots of the treatment models, in the style of the growth-model selection:
  - dashed black curve (linewidth 2), black points (size 10)
  - panel labels "A) name", "B) name" in the upper-left corner
  - axes "Time (days)" / "Tumor Volume (mm³)" (y axis in units of 10^3)
  - "CCC/PCC = X.XX/X.XX" inside each panel
  - common ylim: ceil(max(data + model)/1000)*1000
  - no title

Also writes
  - multi_scatter_<group><model>.png/pdf : one panel per group (CCC/PCC with 95% intervals)
  - multi_scatter_all<model>.png/pdf     : the three groups (A/B/C)
"""
import gc
import math
import os
import string

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_treatment_data
from metrics import ccc, pcc
from models import solve_model
from postprocess import configure_plot_settings, plot_scatter_ax, posterior_corr_from_chain

OUT_DIR = './Output_Calibration'

full_data = load_treatment_data()
GROUPS = sorted(full_data.keys())

MODELS = [
    '_exp',
    '_exp_ED', '_exp_CD', '_exp_AccelED', '_exp_AccelCD',
    '_exp_MM', '_exp_LS', '_exp_PF',
    '_exp_dePR', '_exp_dePRDecay',
    '_exp_CDwithLog', '_exp_CDwithExp', '_exp_CDwithSin',
    '_exp_EDwithLog', '_exp_EDwithExp', '_exp_EDwithSin',
    '_mac', '_mac_treat',
]


def _split_theta_compat(theta, model, group_size):
    """Usa _split_theta do models.py (suporta 1D, 2D e 3D)."""
    from models import _split_theta as _st
    ode_args, ic_fn = _st(theta, model, group_size)
    return ode_args, ic_fn


def plot_maxll(group, model):
    """Grid de max_ll por mouse, estilo Fase 1."""
    npz_path = os.path.join(OUT_DIR, f'multi_ll_pars_{group}{model}.npz')
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path, allow_pickle=True)
    theta = npz['pars']

    mice = full_data[group]
    n_mice = len(mice)
    n_cols = min(4, n_mice)
    n_rows = math.ceil(n_mice / n_cols)

    formatter = configure_plot_settings('14')
    max_time = max(m['data'][-1, 0] for m in mice)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(n_cols * 11, n_rows * 5), squeeze=False)
    axes = axes.ravel()

    ode_args, ic_fn = _split_theta_compat(theta, model, n_mice)

    full_exp, full_model = [], []
    for idx, mouse in enumerate(mice):
        ax = axes[idx]
        d = mouse['data']
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.text(0.05, 0.9,
                f'{string.ascii_uppercase[idx]}) {mouse["name"]}',
                horizontalalignment='left', transform=ax.transAxes, fontsize=18)
        ax.plot(d[:, 0], d[:, 1], 'o', color='black', markersize=10)
        ax.set_xlabel('Time (days)')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlim((-1, max_time + 1))

        ic = ic_fn(idx)

        # smooth curve
        t_smooth = np.linspace(d[0, 0], d[-1, 0], 200)
        sol = solve_model(model, t_smooth, ode_args, ic)
        sol_T = sol[:, 0] if sol.ndim == 2 else sol
        ax.plot(t_smooth, sol_T, color='black', linestyle='dashed', linewidth=2)

        # model at the observation times (for CCC/PCC)
        sol_obs = solve_model(model, d[:, 0], ode_args, ic)
        matched_v = sol_obs[:, 0] if sol_obs.ndim == 2 else sol_obs
        cccT = ccc(matched_v, d[:, 1])
        pccT = pcc(matched_v, d[:, 1])
        ax.text(0.5, 0.05, f'CCC/PCC = {cccT:.2f}/{pccT:.2f}',
                horizontalalignment='left', transform=ax.transAxes)
        full_exp.append(d[:, 1])
        full_model.append(matched_v)

    # ylim padronizado
    final_exp = np.concatenate(full_exp)
    final_model = np.concatenate(full_model)
    rounded_max = math.ceil(max(final_exp.max(), final_model.max()) / 1000) * 1000
    for ax in axes[:n_mice]:
        ax.set_ylim((0, rounded_max))
        ax.yaxis.set_major_formatter(formatter)
    for i in range(n_mice, len(axes)):
        fig.delaxes(axes[i])

    fname = os.path.join(OUT_DIR, f'multi_max_ll_{group}{model}')
    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fname + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return full_exp, full_model


def plot_scatter_all_groups(model):
    """Scatter combinado: 3 paineis (cart_ss, cart_sm, cart_sl), padrao Fase 1."""
    formatter = configure_plot_settings('14')
    fig, axes = plt.subplots(1, 3, figsize=(33, 9))

    summaries = {}
    for i, group in enumerate(GROUPS):
        # full_exp/full_model from the maximum-likelihood sample
        npz_path = os.path.join(OUT_DIR, f'multi_ll_pars_{group}{model}.npz')
        if not os.path.exists(npz_path):
            continue
        npz = np.load(npz_path, allow_pickle=True)
        theta = npz['pars']
        n_mice_g = len(full_data[group])
        ode_args, ic_fn = _split_theta_compat(theta, model, n_mice_g)

        full_exp, full_model = [], []
        for idx, mouse in enumerate(full_data[group]):
            d = mouse['data']
            sol = solve_model(model, d[:, 0], ode_args, ic_fn(idx))
            full_model.append(sol[:, 0] if sol.ndim == 2 else sol)
            full_exp.append(d[:, 1])

        # posterior CCC/PCC with 95% intervals
        chain_path = os.path.join(OUT_DIR, f'multi_chain_{group}{model}.gz')
        try:
            res = posterior_corr_from_chain(
                chain_path=chain_path, model_extension=model,
                full_data=full_data, group=group, n_samp=300, seed=42,
            )
        except Exception:
            res = None

        ax = axes[i]
        if res is not None:
            ccc_s = res['ccc_summary']; pcc_s = res['pcc_summary']
            plot_scatter_ax(ax, full_exp, full_model, formatter,
                            ccc_val=ccc_s[0], ccc_ic=[ccc_s[1], ccc_s[2]],
                            pcc_val=pcc_s[0], pcc_ic=[pcc_s[1], pcc_s[2]])
            summaries[group] = res
        else:
            plot_scatter_ax(ax, full_exp, full_model, formatter,
                            ccc_val=0, ccc_ic=None, pcc_val=0, pcc_ic=None)

        # replace the panel label with A/B/C (plot_scatter_ax assigns a single letter to a 1xN row of panels)
        # Limpa textos antigos e adiciona label customizado
        for t in list(ax.texts):
            t.remove()
        if res is not None:
            ccc_s = res['ccc_summary']; pcc_s = res['pcc_summary']
            label = f'{string.ascii_uppercase[i]}) {group}'
            ax.text(0.06, 0.95, label, transform=ax.transAxes, fontsize=18,
                    verticalalignment='top', weight='bold')
            ax.text(0.06, 0.88,
                    f'CCC = {ccc_s[0]:.3f}  [{ccc_s[1]:.3f}; {ccc_s[2]:.3f}]',
                    transform=ax.transAxes, fontsize=16,
                    verticalalignment='top')
            ax.text(0.06, 0.82,
                    f'PCC = {pcc_s[0]:.3f}  [{pcc_s[1]:.3f}; {pcc_s[2]:.3f}]',
                    transform=ax.transAxes, fontsize=16,
                    verticalalignment='top')

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f'multi_scatter_all{model}')
    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fname + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return summaries


def main():
    print(f"Best fits and scatter plots for {len(GROUPS)} groups x {len(MODELS)} models...")
    for m in MODELS:
        for g in GROUPS:
            plot_maxll(g, m)
        plot_scatter_all_groups(m)
        print(f"  {m}: best fit per group and combined scatter plot done")
        gc.collect()


if __name__ == '__main__':
    main()
