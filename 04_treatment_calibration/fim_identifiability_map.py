"""Practical identifiability of _mac_treat from the Fisher information matrix (FIM) at the bounded
maximum-likelihood estimate (the MCMC marginals have ESS ~ 15 and are not used).

FIM_{jk} = (1/sigma^2) * sum_obs (dT_i/dtheta_j)(dT_i/dtheta_k)   (observing y = T)
Unknowns: 8 model parameters and the initial condition of each mouse (std does not affect T).

Practical identifiability of each parameter = marginal standard deviation sqrt((FIM^-1)_jj), from the
eigendecomposition (directions with eigenvalue ~0 are non-identifiable). alpha1, K1 and beta1 are in log10, so
their std is in orders of magnitude.

Output: Output_Calibration/TMC/fim_identifiability_map.csv, fim_eigenvalues_<group>.csv and the spectrum printed.
"""
import csv
import os
import numpy as np

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from inference import log_likelihood
from models import solve_model, _split_theta

OUT = './Output_Calibration'
TMC = './Output_Calibration/TMC'
os.makedirs(TMC, exist_ok=True)
MODEL = '_mac_treat'
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
# labels of the 8 model parameters (indices 0..7 of theta); 0, 1, 2 in log10
PARN = ['log10_alpha1', 'log10_K1', 'log10_beta1', 'mu1',
        'alpha3', 'K3', 'kappa3', 'mu3']
LOGPAR = {0, 1, 2}

full = load_treatment_data()
max_ic = find_max_initial_condition(full)[0]


def predict_T(theta, group):
    """Predicted T(t) at all observation times of the group, concatenated."""
    gs = len(full[group])
    ode_args, ic_fn = _split_theta(theta, MODEL, gs)
    out = []
    for idx, mouse in enumerate(full[group]):
        t = mouse['data'][:, 0]
        sol = solve_model(MODEL, t, ode_args, ic_fn(idx))
        out.append(sol[:, 0] if sol.ndim == 2 else sol)
    return np.concatenate(out)


def reoptimize(group):
    """Bounded maximum-likelihood estimate produced by mle_bounded.py.

    An earlier version used an unbounded Nelder-Mead step followed by clipping, which for cart_sl_b16 returned
    kappa3 = -0.0125 (outside the prior) and a clipped point with log-likelihood -972.6.
    """
    gs = len(full[group])
    lb, ub, labels = define_bounds_labels(MODEL, group_size=gs, max_ic=max_ic,
                                          bool_multiple_mice=True)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    z = np.load(os.path.join(TMC, f'mle_bounded_{group}.npz'))
    theta = np.asarray(z['theta'], float)
    ll = log_likelihood(theta, MODEL, full, group)
    return theta, lb, ub, labels, float(ll)


def sensitivity(theta, group, n_unknown):
    """Central differences dT_i/dtheta_j for j in 0..n_unknown-1 (model parameters and initial conditions)."""
    base = predict_T(theta, group)
    S = np.zeros((len(base), n_unknown))
    for j in range(n_unknown):
        h = 1e-5 * (1.0 + abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        S[:, j] = (predict_T(tp, group) - predict_T(tm, group)) / (2*h)
    return S


def main():
    rows = []
    for group in GROUPS:
        gs = len(full[group])
        theta, lb, ub, labels, mll = reoptimize(group)
        sigma = theta[-1]
        # unknowns affecting T: 8 model parameters and gs initial conditions (std excluded)
        n_unknown = 8 + gs
        S = sensitivity(theta, group, n_unknown)
        FIM = (S.T @ S) / sigma**2
        # symmetric eigendecomposition
        w, V = np.linalg.eigh(FIM)
        w_clip = np.clip(w, 0, None)
        wmax = w_clip.max()
        # marginal std of each unknown from the spectral pseudo-inverse
        tol = wmax * 1e-12
        inv_diag = np.zeros(n_unknown)
        for i in range(n_unknown):
            s = 0.0
            for k in range(n_unknown):
                if w[k] > tol:
                    s += V[i, k]**2 / w[k]
                else:
                    s += V[i, k]**2 / tol  # flat direction -> very large variance
            inv_diag[i] = s
        std_marg = np.sqrt(inv_diag)
        print(f"\n=== {group} (MLE ll={mll:.2f}, sigma={sigma:.1f}) ===")
        print(f"  cond(FIM) = {wmax/max(w_clip[w_clip>tol].min(),tol):.2e}  "
              f"eigenvalues min..max = {w_clip[w_clip>0].min():.2e} .. {wmax:.2e}")
        for j in range(8):
            val = 10**theta[j] if j in LOGPAR else theta[j]
            ident = f'{std_marg[j]:.3g} decades' if j in LOGPAR else f'{std_marg[j]/max(abs(val),1e-300):.3g} relative'
            print(f"  {PARN[j]:14s} MLE={val:11.4g}  std_marg={std_marg[j]:9.3g}"
                  f"  {'(log10)' if j in LOGPAR else '       '}  -> {ident}")
            rows.append({'group': group, 'param': PARN[j],
                         'MLE': f'{val:.5g}', 'std_marginal': f'{std_marg[j]:.4g}',
                         'scale': 'log10' if j in LOGPAR else 'linear',
                         'marginal_uncertainty': ident})
        with open(os.path.join(TMC, f'fim_eigenvalues_{group}.csv'), 'w', newline='') as fe:
            fe.write('eigenvalue\n' + '\n'.join(f'{v:.6e}' for v in sorted(w, reverse=True)) + '\n')
    with open(os.path.join(TMC, 'fim_identifiability_map.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w_ = csv.DictWriter(f, fieldnames=rows[0].keys())
        w_.writeheader(); w_.writerows(rows)
    print(f"\nCSV: {TMC}/fim_identifiability_map.csv")


if __name__ == '__main__':
    main()
