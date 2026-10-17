"""Model-selection metrics (BIC, DIC) of the multi-group calibration.

Computed from the files written by `global.py`:
  - thinned chain       : multi_chain_{model}{_sN}.gz
  - max_ll and best pars: multi_ll_pars_{model}{_sN}.npz
  - HDF5 backend        : reinitiate_files/reinitiate_{model}{_sN}.h5 (number of steps)

Number of observations n = observations of all groups; number of parameters k = len(theta) (including the
initial conditions and std).
"""
import _path_setup  # adds pipeline_control_parallel/ to sys.path  # noqa: F401

import os

import numpy as np

from inference_shared import log_likelihood_shared


def n_observations(full_data, groups):
    """Total number of observations (t, T) over the mice of `groups`."""
    n = 0
    for g in groups:
        for m in full_data[g]:
            n += m['data'].shape[0]
    return n


def bic(max_ll, k, n):
    """Schwarz: BIC = k*ln(n) - 2*max_ll."""
    return k * np.log(n) - 2.0 * max_ll


def dic(chain, model_extension, full_data, groups, scenario, n_sub=2000):
    """DIC = D_bar + p_D, with p_D = D_bar - D_thetabar and D = -2 log-likelihood, computed on evenly
    spaced samples of the chain (at most n_sub). Multimodal posteriors often give p_D < 0 (the implicit
    Gaussian approximation fails)."""
    if chain.shape[0] > n_sub:
        idx = np.linspace(0, chain.shape[0] - 1, n_sub).astype(int)
        sub = chain[idx]
    else:
        sub = chain
    ll_samples = np.array([
        log_likelihood_shared(theta, model_extension, full_data, groups, scenario)
        for theta in sub
    ])
    ll_samples = ll_samples[np.isfinite(ll_samples)]
    if len(ll_samples) < 10:
        return float('nan'), float('nan')
    D_bar = -2.0 * np.mean(ll_samples)
    theta_bar = np.mean(sub, axis=0)
    ll_thetabar = log_likelihood_shared(
        theta_bar, model_extension, full_data, groups, scenario,
    )
    D_thetabar = -2.0 * ll_thetabar
    pD = D_bar - D_thetabar
    return D_bar + pD, pD


def wbic(chain_ll_at_temperature):
    """Watanabe BIC: WBIC = -E_{theta ~ p_beta}[log L(theta)] at beta = 1/log(n). It requires sampling the
    tempered posterior, which these calibrations do not do; only BIC and DIC are reported for the
    scenarios.
    """
    raise NotImplementedError(
        'WBIC requires sampling at temperature beta = 1/log(n); '
        'only BIC and DIC are computed for the scenario comparison.'
    )


def compare_scenarios(
    out_dir, full_data, groups, model_extension, scenarios_to_compare,
    compute_dic=True,
):
    """BIC (and optionally DIC) of every calibrated scenario.

    Reads
        {out_dir}/multi_chain_{model}_s{N}.gz
        {out_dir}/multi_ll_pars_{model}_s{N}.npz

    Returns a list of dicts with keys scenario, k, n, max_ll, BIC, dBIC, DIC, pD.
    """
    n = n_observations(full_data, groups)
    rows = []
    for snum, sdef in scenarios_to_compare.items():
        ll_path = os.path.join(
            out_dir, f'multi_ll_pars_{model_extension}_s{snum}.npz',
        )
        chain_path = os.path.join(
            out_dir, f'multi_chain_{model_extension}_s{snum}.gz',
        )
        if not (os.path.exists(ll_path) and os.path.exists(chain_path)):
            continue

        npz = np.load(ll_path)
        max_ll = float(npz['max_ll'])
        best = npz['pars']
        k = int(len(best))

        row = dict(
            scenario=snum, k=k, n=n, max_ll=max_ll,
            BIC=bic(max_ll, k, n),
            DIC=float('nan'), pD=float('nan'),
        )

        if compute_dic:
            chain = np.loadtxt(chain_path)
            if chain.ndim == 1:
                chain = chain.reshape(1, -1)
            d, pD = dic(chain, model_extension, full_data, groups, sdef)
            row['DIC'] = d
            row['pD'] = pD

        rows.append(row)

    if not rows:
        return rows

    bic_min = min(r['BIC'] for r in rows)
    for r in rows:
        r['dBIC'] = r['BIC'] - bic_min

    return rows


def print_comparison_table(rows, model_extension):
    """Prints the scenario comparison table."""
    if not rows:
        print(f'[compare] {model_extension}: no calibrated scenario yet.')
        return
    print(f'\n=== Scenario comparison: {model_extension} (n={rows[0]["n"]}) ===')
    hdr = f'{"scen":>4} {"k":>4} {"max_ll":>12} {"BIC":>12} {"dBIC":>8} {"DIC":>12} {"pD":>10}'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(
            f'{r["scenario"]:>4d} {r["k"]:>4d} {r["max_ll"]:>12.2f} '
            f'{r["BIC"]:>12.2f} {r["dBIC"]:>8.2f} '
            f'{r["DIC"]:>12.2f} {r["pD"]:>10.2f}'
        )
    best = min(rows, key=lambda r: r['BIC'])
    print(f'\n  -> lowest BIC: scenario s{best["scenario"]}')
