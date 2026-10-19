"""Joint fit of T(t) and C(t) for the long-hinge group (_mac_treat) and profiles of mu1 and beta1.

C(t) is the CAR T-cell persistence (vector copy number, VCN, in the tumor) of Jilani et al., Fig. 6b, 20D7SL-28z,
days 5, 13 and 27 (population mean +- SD; a cohort different from that of the volumes, so C enters as a population
curve informing the CAR T-cell dynamics). The joint likelihood is the persistence likelihood also used by
fim_joint_TC.py and fim_joint_TM.py:

  ll = ll_T (per mouse, as in the calibration) + ll_C,
  ll_C = -0.5 * sum_k [(s*Cbar(t_k) - VCN_k)^2 / SD_k^2 + log(2 pi SD_k^2)],
  Cbar(t_k) = mean over the mice of C_model(t_k) (C0 = 1); s = free VCN/C scale.

Compares the 95% intervals of mu1 and beta1 from T alone and from T and C.

Output: Output_Calibration/TMC/joint_TC_profile.csv and a figure.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from inference import log_likelihood
from models import solve_model, _split_theta

OUT = './Output_Calibration'
TMC = './Output_Calibration/TMC'
os.makedirs(TMC, exist_ok=True)
MODEL, GROUP = '_mac_treat', 'cart_sl_b16'
LR = 1.9207

# VCN in the tumor, Jilani et al. Fig. 6b, 20D7SL-28z
C_T = np.array([5.0, 13.0, 27.0])
C_VCN = np.array([0.1326, 0.05268, 0.01866])
C_DP = np.array([0.04918, 0.02302, 0.01191])

# scale s (VCN per unit of C, C0 = 1): wide bounds
S_LO, S_HI = 1e-3, 100.0

full = load_treatment_data()
max_ic = find_max_initial_condition(full)[0]
gs = len(full[GROUP])
LB, UB, LABELS = define_bounds_labels(MODEL, group_size=gs, max_ic=max_ic,
                                      bool_multiple_mice=True)
LB, UB = np.asarray(LB, float), np.asarray(UB, float)
NDIM = len(LB)


def C_mean_at(theta, tqs):
    """Mean over the mice of the group of C_model(t) at the times tqs."""
    ode_args, ic_fn = _split_theta(theta, MODEL, gs)
    acc = np.zeros(len(tqs))
    for idx in range(gs):
        t0 = full[GROUP][idx]['data'][0, 0]
        tg = np.concatenate(([t0], tqs))
        sol = solve_model(MODEL, tg, ode_args, ic_fn(idx))
        acc += sol[1:, 2]  # column C
    return acc / gs


def negll_joint(theta_full):
    theta, s = theta_full[:-1], theta_full[-1]
    llT = log_likelihood(theta, MODEL, full, GROUP)
    if not np.isfinite(llT):
        return 1e12
    try:
        Cbar = C_mean_at(theta, C_T)
    except Exception:
        return 1e12
    if not np.all(np.isfinite(Cbar)):
        return 1e12
    llC = -0.5*np.sum((s*Cbar - C_VCN)**2/C_DP**2 + np.log(2*np.pi*C_DP**2))
    return -(llT + llC)


def negll_Tonly(theta):
    ll = log_likelihood(theta, MODEL, full, GROUP)
    return 1e12 if not np.isfinite(ll) else -ll


def profile(param_idx, joint):
    """Profile of parameter param_idx (0..7); joint=True uses T and C, with the scale s."""
    npz = np.load(os.path.join(OUT, f'multi_ll_pars_{GROUP}{MODEL}.npz'),
                  allow_pickle=True)
    map_t = np.clip(np.asarray(npz['pars'], float), LB+1e-9, UB-1e-9)
    eps = 1e-9
    if joint:
        lb = np.append(LB, S_LO); ub = np.append(UB, S_HI)
        x_map = np.append(map_t, 0.3)
        neg = negll_joint
    else:
        lb, ub, x_map, neg = LB, UB, map_t, negll_Tonly
    n = len(lb)
    free = np.array([i for i in range(n) if i != param_idx])
    bnds = [(lb[i]+eps, ub[i]-eps) for i in free]
    is_log = param_idx in (0, 1, 2)
    grid = np.linspace(lb[param_idx], ub[param_idx], 21)
    i0 = int(np.argmin(np.abs(grid - x_map[param_idx])))
    order = sorted(range(21), key=lambda i: abs(i-i0))
    prof = np.full(21, -np.inf); warm = {}

    def nf(fv, pv):
        th = np.empty(n); th[param_idx] = pv; th[free] = fv
        return neg(th)
    for i in order:
        pv = grid[i]
        nb = [j for j in (i-1, i+1) if j in warm]
        x0 = warm[nb[0]] if nb else x_map[free].copy()
        x0 = np.clip(x0, lb[free]+eps, ub[free]-eps)
        best = None
        for m in ('L-BFGS-B', 'Powell'):
            try:
                r = minimize(nf, x0, method=m,
                             bounds=bnds,
                             args=(pv,), options={'maxiter': 400})
            except Exception:
                continue
            if r is not None and np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        if best is not None:
            prof[i] = -best.fun
            warm[i] = np.clip(best.x, lb[free]+eps, ub[free]-eps)
    for _ in range(2):
        imp = False
        for i in range(21):
            for j in (i-1, i+1):
                if j not in warm:
                    continue
                x0 = np.clip(warm[j], lb[free]+eps, ub[free]-eps)
                try:
                    r = minimize(nf, x0, method='L-BFGS-B', bounds=bnds,
                                 args=(grid[i],), options={'maxiter': 400})
                except Exception:
                    continue
                if np.isfinite(r.fun) and -r.fun > prof[i]+1e-6:
                    prof[i] = -r.fun; warm[i] = np.clip(r.x, lb[free]+eps, ub[free]-eps); imp = True
        if not imp:
            break
    llmax = np.nanmax(prof)
    within = np.isfinite(prof) & (prof >= llmax-LR)
    gg = 10**grid if is_log else grid
    lo, hi = gg[within].min(), gg[within].max()
    frac = within.sum()/np.isfinite(prof).sum()
    return grid, prof, gg, lo, hi, frac, is_log


def main():
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for col, (pidx, pname) in enumerate([(3, 'mu1'), (2, 'log10_beta1')]):
        for row, joint in enumerate([False, True]):
            grid, prof, gg, lo, hi, frac, is_log = profile(pidx, joint)
            tag = 'T+C' if joint else 'T-only'
            rows.append({'param': pname, 'data': tag,
                         'CI95_lo': f'{lo:.4g}', 'CI95_hi': f'{hi:.4g}',
                         'frac_prior': f'{frac:.2f}',
                         'excludes_0': 'yes' if lo > (1e-6 if is_log else 1e-3) and frac < 0.9 else 'no/partial'})
            print(f'{pname:12s} {tag:7s} CI95=[{lo:.4g},{hi:.4g}] frac={frac:.2f}')
            ax = axes[row, col]
            x = grid
            ax.plot(x, prof, 'o-', color='steelblue' if not joint else 'darkgreen', ms=5)
            ax.axhline(np.nanmax(prof)-LR, color='crimson', ls='--', lw=1.3)
            ax.set_title(f'{pname}  [{tag}]  CI95=[{lo:.3g}, {hi:.3g}]')
            ax.set_xlabel(pname); ax.set_ylabel('profile ll')
    fig.suptitle('long hinge _mac_treat: profile from T only (blue) and from T and C (green)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(TMC, 'joint_TC_profile.png'), dpi=200,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    with open(os.path.join(TMC, 'joint_TC_profile.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'\nCSV and figure in {TMC}/joint_TC_profile*')


if __name__ == '__main__':
    main()
