"""Figures of the eFAST sensitivity indices: final-time S1 and ST (figures/fig9a_b.pdf) and normalized
time-dependent ST (figures/fast_TMC_ST_TMC_ST_stacked_time.png). Reads the outputs of efast_sensitivity.py.

Extracted from notebooks/melanoma_CART_TAM.ipynb.
"""
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model_data = np.load('results/fast_model_outputs_TMC.npz', allow_pickle=True)
param_names = model_data['param_names'].tolist()

sens_data = np.load('results/fast_time_dependent_sensitivity_TMC.npz', allow_pickle=True)
t_sel = sens_data['t_sel']
S1_T = sens_data['S1_T']
ST_T = sens_data['ST_T']
S1_M = sens_data['S1_M']
ST_M = sens_data['ST_M']
S1_C = sens_data['S1_C']
ST_C = sens_data['ST_C']

df_T_end = pd.read_csv('results/fast_Si_T_end.csv')
df_M_end = pd.read_csv('results/fast_Si_M_end.csv')
df_C_end = pd.read_csv('results/fast_Si_C_end.csv')

print("Data loaded successfully.")

from matplotlib.ticker import FormatStrFormatter

param_label_map = {
    'alpha1': r'$\alpha_1$',
    'alpha2': r'$\alpha_2$',
    'alpha3': r'$\alpha_3$',
    'beta1': r'$\beta_1$',
    'K1': r'$K_1$',
    'K2': r'$K_2$',
    'K3': r'$K_3$',
    'mu1': r'$\mu_1$',
    'kappa2': r'$\kappa_2$',
    'kappa3': r'$\kappa_3$'
}


def plot_end_bars(df_T_end, df_M_end, df_C_end, filename_prefix):
    params_T = df_T_end['parameter'].values
    params_M = df_M_end['parameter'].values
    params_C = df_C_end['parameter'].values

    if not (np.array_equal(params_T, params_M) and np.array_equal(params_T, params_C)):
        print("Warning: parameter order differs across dataframes. Using the order from T.")

    params = params_T
    x = np.arange(len(params))
    width = 0.25

    S1_T = df_T_end['S1'].values
    S1_M = df_M_end['S1'].values
    S1_C = df_C_end['S1'].values

    ST_T = df_T_end['ST'].values
    ST_M = df_M_end['ST'].values
    ST_C = df_C_end['ST'].values

    xtick_labels = [param_label_map.get(p, p) for p in params]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_top = axes[0]
    ax_top.bar(x - width, S1_T, width, label=r'$T(t)$', alpha=0.9)
    ax_top.bar(x, S1_M, width, label=r'$M(t)$', alpha=0.9)
    ax_top.bar(x + width, S1_C, width, label=r'$C(t)$', alpha=0.9)

    ax_top.set_ylabel(r'First-order sensitivity indices $S_1$', fontsize=16)
    ax_top.tick_params(axis='both', which='major', labelsize=14)
    ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_top.text(0.01, 0.98, 'a)', transform=ax_top.transAxes, fontsize=16, va='top', ha='left', fontweight='bold')
    ax_top.legend(fontsize=14)

    ax_bottom = axes[1]
    ax_bottom.bar(x - width, ST_T, width, label='T', alpha=0.9)
    ax_bottom.bar(x, ST_M, width, label='M', alpha=0.9)
    ax_bottom.bar(x + width, ST_C, width, label='C', alpha=0.9)

    ax_bottom.set_ylabel(r'Total sensitivity indices $S_T$', fontsize=16)
    ax_bottom.tick_params(axis='both', which='major', labelsize=14)
    ax_bottom.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_bottom.text(0.01, 0.98, 'b)', transform=ax_bottom.transAxes, fontsize=16, va='top', ha='left', fontweight='bold')

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=16)

    fig.tight_layout()
    fig.savefig('figures/fig9a_b.pdf', format='pdf')
    plt.close('all')
    plt.close(fig)


plot_end_bars(df_T_end, df_M_end, df_C_end, 'fast_TMC')

print("Final-time bar plot for stacked S1 and ST panels was generated.")

def plot_time_curves_stacked_TMC(
    S_T,
    S_M,
    S_C,
    t_sel,
    param_names,
    filename_prefix,
    normalize=True
):
    n_times, n_par = S_T.shape
    assert S_M.shape == (n_times, n_par)
    assert S_C.shape == (n_times, n_par)

    x = np.arange(n_times)

    def _normalize(S_mat):
        if not normalize:
            return S_mat.copy()
        totals = S_mat.sum(axis=1)
        totals[totals == 0.0] = 1.0
        return S_mat / totals[:, None]

    S_T_plot = _normalize(S_T)
    S_M_plot = _normalize(S_M)
    S_C_plot = _normalize(S_C)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    def _plot_panel(ax, S_plot, var_label, panel_letter, show_legend=False):
        bottom = np.zeros(n_times)

        for j, pname in enumerate(param_names):
            values = S_plot[:, j]
            label = param_label_map.get(pname, pname)
            ax.bar(
                x,
                values,
                bottom=bottom,
                label=label if show_legend else None,
                width=0.9
            )
            bottom += values

        ax.set_ylabel(var_label, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.text(
            0.01, 0.98, panel_letter,
            transform=ax.transAxes,
            fontsize=16,
            va='top',
            ha='left',
            fontweight='bold'
        )

    _plot_panel(axes[0], S_T_plot, r'$T(t)$', 'a)', show_legend=True)
    _plot_panel(axes[1], S_M_plot, r'$M(t)$', 'b)', show_legend=False)
    _plot_panel(axes[2], S_C_plot, r'$C(t)$', 'c)', show_legend=False)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        [f'{ti:.0f}' for ti in t_sel],
        rotation=90,
        ha='center',
        fontsize=12
    )
    axes[2].set_xlabel('Time (days)', fontsize=16)

    ncol_legend = min(len(param_names), 5)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=ncol_legend,
        frameon=False,
        fontsize=14
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.82, left=0.12)

    fig.savefig(f'figures/{filename_prefix}_TMC_ST_stacked_time.png', dpi=300, format='png', bbox_inches='tight')
    plt.close(fig)


plot_time_curves_stacked_TMC(
    S_T=ST_T,
    S_M=ST_M,
    S_C=ST_C,
    t_sel=t_sel,
    param_names=param_names,
    filename_prefix='fast_TMC_ST',
    normalize=True
)
