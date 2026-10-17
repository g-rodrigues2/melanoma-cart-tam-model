"""Maximum-likelihood estimates of the mechanistic models with the prior bounds enforced.

Both mechanistic models are fitted to each CAR T-cell group: _mac_treat (T-M-C, 8 model parameters) and _mac
(T-M, 3 model parameters), each with one initial volume per mouse and a common residual standard deviation.
The fit is a bounded nonlinear least-squares problem with sigma concentrated out (lsq_fit.py).

Starts, per model and group:
  - the MCMC maximum-likelihood sample (Output_Calibration/multi_ll_pars_<group><model>.npz)
  - N_CHAIN samples spread evenly over the stored posterior sample (multi_chain_<group><model>.gz)
  - for _mac_treat, the earlier quasi-Newton estimate (TMC/archive_lbfgsb/mle_bounded_<group>.npz)

Outputs (Output_Calibration/TMC):
  mle_bounded_<group>.npz        _mac_treat estimate (theta, ll, labels, map_ll, lls of all starts)
  mle_bounded_<group>_mac.npz    _mac estimate
  mle_bounded_summary.csv        one row per model and group
"""
import csv
import os
from multiprocessing import Pool

import numpy as np

from lsq_fit import Problem

OUT = './Output_Calibration'
TMC = './Output_Calibration/TMC'
MODELS = ['_mac_treat', '_mac']
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
N_CHAIN = 40
WORKERS = 15


def npz_name(model, group):
    return f'mle_bounded_{group}.npz' if model == '_mac_treat' else f'mle_bounded_{group}{model}.npz'


def starts(model, group):
    prob = Problem(model, group)
    z = np.load(os.path.join(OUT, f'multi_ll_pars_{group}{model}.npz'), allow_pickle=True)
    out = [('mcmc_max_ll', np.asarray(z['pars'], float))]
    chain = np.loadtxt(os.path.join(OUT, f'multi_chain_{group}{model}.gz'))
    for k, i in enumerate(np.linspace(0, len(chain) - 1, N_CHAIN).astype(int)):
        out.append((f'chain_{k}', chain[i]))
    legacy = os.path.join(TMC, 'archive_lbfgsb', f'mle_bounded_{group}.npz')
    if model == '_mac_treat' and os.path.exists(legacy):
        out.append(('lbfgsb', np.load(legacy)['theta']))
    return [(name, np.clip(s, prob.lb, prob.ub)) for name, s in out]


def run_start(args):
    model, group, name, start = args
    ll, th = Problem(model, group).fit(start)
    return model, group, name, ll, th


def main():
    jobs = [(m, g, name, s) for m in MODELS for g in GROUPS for name, s in starts(m, g)]
    res = {}
    with Pool(WORKERS) as pool:
        for m, g, name, ll, th in pool.imap_unordered(run_start, jobs):
            res.setdefault((m, g), []).append((ll, name, th))
    with open(os.path.join(TMC, 'mle_bounded_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'group', 'll', 'best_start', 'mcmc_map_ll', 'n_starts', 'n_within_0.1', 'n_within_2',
                    'sigma'] + [f'p{i}' for i in range(8)])
        for m in MODELS:
            for g in GROUPS:
                runs = sorted(res[(m, g)], key=lambda r: -r[0])
                ll, name, th = runs[0]
                lls = np.array([r[0] for r in runs])
                prob = Problem(m, g)
                map_ll = float(np.load(os.path.join(OUT, f'multi_ll_pars_{g}{m}.npz'), allow_pickle=True)['max_ll'])
                np.savez(os.path.join(TMC, npz_name(m, g)), theta=th, ll=ll, labels=np.array(prob.labels),
                         map_ll=map_ll, start_lls=lls)
                n_par = 8 if m == '_mac_treat' else 3
                w.writerow([m, g, f'{ll:.4f}', name, f'{map_ll:.3f}', len(lls), int(np.sum(lls > ll - 0.1)),
                            int(np.sum(lls > ll - 2)), f'{th[-1]:.3f}'] + [f'{v:.6g}' for v in th[:n_par]])
                phys = {prob.labels[i].replace('log10_', ''): (10 ** th[i] if prob.labels[i].startswith('log10')
                                                               else th[i]) for i in range(n_par)}
                print(f'== {g}{m}: ll={ll:.3f} (start {name}; MCMC max ll {map_ll:.3f}); '
                      f'{int(np.sum(lls > ll - 0.1))}/{len(lls)} starts within 0.1, '
                      f'{int(np.sum(lls > ll - 2))} within 2', flush=True)
                print('   ' + ', '.join(f'{k}={v:.4g}' for k, v in phys.items()), flush=True)


if __name__ == '__main__':
    main()
