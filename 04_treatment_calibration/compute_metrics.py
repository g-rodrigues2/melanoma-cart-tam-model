"""BIC, BIC weights, DIC, p_D and WBIC of every (group, treatment model) pair.

BIC needs only the maximum-likelihood file (``multi_ll_pars_<group><model>.npz``), which is in the
repository. DIC, p_D and WBIC need the MCMC chain (``multi_chain_<group><model>.gz``), which is in the
Zenodo deposit (see the repository README); where the chain is absent they are reported as ``nan`` and
the BIC columns are computed all the same.

Writes Output_Calibration/model_selection_table.csv and prints Markdown tables per group.
"""
import os
import sys

import numpy as np

# the tables below carry Greek letters; the Windows console defaults to cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from inference import log_likelihood

OUT_DIR = './Output_Calibration'

MODELS = [
    '_exp',
    '_exp_ED', '_exp_CD', '_exp_AccelED', '_exp_AccelCD',
    '_exp_MM', '_exp_LS', '_exp_PF',
    '_exp_dePR', '_exp_dePRDecay',
    '_exp_CDwithLog', '_exp_CDwithExp', '_exp_CDwithSin',
    '_exp_EDwithLog', '_exp_EDwithExp', '_exp_EDwithSin',
    '_mac', '_mac_treat',
]
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']


def bic_value(ll_max, k, n):
    return -2.0 * ll_max + k * np.log(n)


def bic_weights(bic_arr):
    bic_min = np.min(bic_arr)
    w = np.exp(-0.5 * (bic_arr - bic_min))
    return w / w.sum()


def wbic_value(ll_chain, n):
    """Watanabe BIC: -2 * <log L>_beta com beta = 1/log(n)."""
    beta = 1.0 / np.log(n)
    w = np.exp(beta * ll_chain - (beta * ll_chain).max())
    return -2.0 * np.average(ll_chain, weights=w)


def dic_value(ll_at_mean, ll_chain):
    """DIC = D_bar + p_D. A negative p_D indicates a posterior far from Gaussian or a chain that has not converged."""
    D_bar = -2.0 * np.mean(ll_chain)
    D_at_mean = -2.0 * ll_at_mean
    p_D = D_bar - D_at_mean
    return D_bar + p_D, p_D


def compute_one(group, model, full_data, max_ic):
    """Metrics of one pair."""
    chain_path = os.path.join(OUT_DIR, f'multi_chain_{group}{model}.gz')
    npz_path   = os.path.join(OUT_DIR, f'multi_ll_pars_{group}{model}.npz')

    if not os.path.exists(npz_path):
        return None

    npz = np.load(npz_path, allow_pickle=True)
    max_ll = float(npz['max_ll'])

    has_chain = os.path.exists(chain_path)
    group_size = len(full_data[group])
    l_bound, u_bound, labels = define_bounds_labels(
        model, group_size=group_size, max_ic=max_ic, bool_multiple_mice=True,
    )
    k = len(l_bound)
    n_obs = sum(len(m['data']) for m in full_data[group])

    if has_chain:
        chain = np.loadtxt(chain_path)
        if chain.ndim == 1:
            chain = chain.reshape(1, -1)
        # log-likelihood of the chain samples (without the prior term)
        ll_chain = np.array([log_likelihood(th, model, full_data, group)
                             for th in chain])
        # Filtra invalidos (raros)
        valid = np.isfinite(ll_chain)
        ll_chain = ll_chain[valid]
        chain_valid = chain[valid]

        theta_mean = chain_valid.mean(axis=0)
        ll_at_mean = log_likelihood(theta_mean, model, full_data, group)
        wbic = wbic_value(ll_chain, n_obs)
        dic, p_D = dic_value(ll_at_mean, ll_chain)
    else:
        wbic = dic = p_D = float('nan')

    bic = bic_value(max_ll, k, n_obs)

    return {
        'group': group,
        'model': model,
        'k': k,
        'n_obs': n_obs,
        'max_ll': max_ll,
        'BIC': bic,
        'DIC': dic,
        'p_D': p_D,
        'WBIC': wbic,
    }


def fmt_table_for_group(results_group):
    """Prints a Markdown table highlighting the winners."""
    bics = np.array([r['BIC'] for r in results_group])
    wbics = np.array([r['WBIC'] for r in results_group])
    w = bic_weights(bics)
    dics = np.array([r['DIC'] for r in results_group])
    p_Ds = np.array([r['p_D'] for r in results_group])

    bic_min = bics.argmin()
    wbic_min = wbics.argmin() if np.isfinite(wbics).any() else None
    # DIC so confiavel se p_D > 0
    valid_dic = np.where(p_Ds > 0)[0]
    dic_min = valid_dic[dics[valid_dic].argmin()] if len(valid_dic) > 0 else None

    delta_bic = bics - bics[bic_min]
    delta_wbic = wbics - wbics[wbic_min] if wbic_min is not None else np.full_like(wbics, np.nan)

    print("| Model | k | max_ll | BIC | ΔBIC | w_BIC | DIC | p_D | WBIC | ΔWBIC |")
    print("|--------|---|--------|-----|------|-------|-----|-----|------|-------|")
    for i, r in enumerate(results_group):
        bic_cell = f"<mark>**{r['BIC']:.2f}**</mark>" if i == bic_min else f"{r['BIC']:.2f}"
        wbic_cell = f"<mark>**{r['WBIC']:.2f}**</mark>" if i == wbic_min else f"{r['WBIC']:.2f}"
        dic_cell = f"<mark>**{r['DIC']:.2f}**</mark>" if dic_min is not None and i == dic_min else f"{r['DIC']:.2f}"
        delta_bic_cell = f"<mark>**{delta_bic[i]:.2f}**</mark>" if i == bic_min else f"{delta_bic[i]:.2f}"
        delta_wbic_cell = f"<mark>**{delta_wbic[i]:.2f}**</mark>" if i == wbic_min else f"{delta_wbic[i]:.2f}"
        w_cell = f"<mark>**{w[i]:.3f}**</mark>" if i == bic_min else f"{w[i]:.3f}"
        print(f"| `{r['model']}` | {r['k']} | {r['max_ll']:.2f} | "
              f"{bic_cell} | {delta_bic_cell} | {w_cell} | "
              f"{dic_cell} | {r['p_D']:.1f} | "
              f"{wbic_cell} | {delta_wbic_cell} |")


def main():
    full_data = load_treatment_data()
    max_ic = find_max_initial_condition(full_data)[0]

    all_results = []
    print(f"Metrics of the {len(GROUPS) * len(MODELS)} (group, model) pairs...")
    print()
    for group in GROUPS:
        for model in MODELS:
            r = compute_one(group, model, full_data, max_ic)
            if r is not None:
                all_results.append(r)

    if not all_results:
        raise SystemExit(
            "No maximum-likelihood file (Output_Calibration/multi_ll_pars_<group><model>.npz) was found. "
            "Run global.py first, or see the repository README for the Zenodo deposit.")

    n_no_chain = sum(1 for r in all_results if not np.isfinite(r['WBIC']))
    if n_no_chain:
        print(f"[note] {n_no_chain} of {len(all_results)} pairs have no MCMC chain in Output_Calibration/: "
              "BIC is computed from the maximum-likelihood file, DIC/p_D/WBIC are reported as nan. "
              "The chains are in the Zenodo deposit (see the repository README).")

    # CSV
    import csv
    csv_path = os.path.join(OUT_DIR, 'model_selection_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f"CSV salvo em {csv_path}\n")

    # Markdown tables per group
    for group in GROUPS:
        rg = [r for r in all_results if r['group'] == group]
        if not rg:
            continue
        n_obs = rg[0]['n_obs']
        print(f"\n### `{group}` (n = {n_obs} observacoes)\n")
        fmt_table_for_group(rg)
        bics = np.array([r['BIC'] for r in rg])
        best_bic = rg[np.argmin(bics)]
        print(f"\n=> Vencedor por BIC: **`{best_bic['model']}`** "
              f"(max_ll={best_bic['max_ll']:.2f}, k={best_bic['k']})\n")


if __name__ == '__main__':
    main()
