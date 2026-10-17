"""Posterior summaries of _mac and _mac_treat in physical units.

The chains sample alpha1, K1 and beta1 in log10 (see bounds.py); this script converts them back (10**x) and
reports the median, the maximum-likelihood sample and the 95% interval.
"""
import os
import numpy as np

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data

OUT = './Output_Calibration'
LOG_PARAMS = {'log10_alpha1', 'log10_K1', 'log10_beta1'}

full = load_treatment_data()
max_ic = find_max_initial_condition(full)[0]
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']


def stats_for(group, model):
    chain_p = os.path.join(OUT, f'multi_chain_{group}{model}.gz')
    npz_p = os.path.join(OUT, f'multi_ll_pars_{group}{model}.npz')
    if not (os.path.exists(chain_p) and os.path.exists(npz_p)):
        return
    chain = np.loadtxt(chain_p)
    npz = np.load(npz_p, allow_pickle=True)
    map_pars = npz['pars']
    gs = len(full[group])
    _, _, labels = define_bounds_labels(model, group_size=gs, max_ic=max_ic,
                                        bool_multiple_mice=True)
    print(f"\n=== {group} {model} (n_samples={chain.shape[0]}) ===")
    for i, lbl in enumerate(labels):
        if lbl.startswith('ic') or lbl == 'std':
            continue
        col = chain[:, i]
        mp = float(map_pars[i])
        if lbl in LOG_PARAMS:
            col = 10.0 ** col
            mp = 10.0 ** mp
            name = lbl.replace('log10_', '')
        else:
            name = lbl
        med = np.median(col)
        lo, hi = np.percentile(col, [2.5, 97.5])
        print(f"  {name:8s} med={med:.5g}  MAP={mp:.5g}  "
              f"CI95=[{lo:.5g}, {hi:.5g}]  includes0={'yes' if lo <= 0 else 'no'}")


if __name__ == '__main__':
    for m in ['_mac', '_mac_treat']:
        for g in GROUPS:
            stats_for(g, m)
