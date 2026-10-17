"""Scenario 7, revised: dose response under the minimal identifiable submodel.

The first version combined posterior medians (alpha1, mu1) with a separately
optimized K1 and the tabulated day-0 volume T(0)=13.5 mm^3, whereas the
calibration estimates one initial condition per mouse; the administered-dose
trajectory therefore did not reproduce the data. Here every quantity comes from
a single maximum-likelihood fit of the reduced model (same fixed values and free
set as reduced_fit.py), the dose-response is simulated from each mouse's fitted
initial condition, and the administered dose (D=1) is checked against the data.
Outputs: Output_Calibration/TMC/scenario7_v2.csv and figures/scenario7_dose_response_calibrated.png
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, '..', '04_treatment_calibration')):
    if os.path.exists(os.path.join(_p, 'inference.py')) and _p not in sys.path:
        sys.path.insert(0, _p)

from bounds import define_bounds_labels, find_max_initial_condition  # noqa: E402
from data_loader import load_treatment_data  # noqa: E402
from inference import log_likelihood  # noqa: E402

TMC = os.path.join(HERE, 'results')
os.makedirs(TMC, exist_ok=True)
MODEL = '_mac_treat'
GROUP = 'cart_sl_b16'
FIX = {2: -4.0, 4: 1e-9, 5: 20.0, 6: 0.09, 7: 0.0}
FREE_MODEL = [0, 1, 3]
KAPPA3 = 0.09
BETA1 = 1e-4
ALPHA2, K2, KAPPA2 = 0.1, 1000.0, 0.34
DOSES = [0.5, 1.0, 2.0, 4.0]
HORIZON = 60.0

full = load_treatment_data()
max_ic = find_max_initial_condition(full)[0]
gs = len(full[GROUP])
lb, ub, labels = define_bounds_labels(MODEL, group_size=gs, max_ic=max_ic, bool_multiple_mice=True)
lb, ub = np.asarray(lb, float), np.asarray(ub, float)
ndim = len(lb)
free = np.array(FREE_MODEL + list(range(8, ndim)))
lbf, ubf = lb[free], ub[free]


def build(x):
    th = np.empty(ndim)
    for i, v in FIX.items():
        th[i] = v
    th[free] = x
    return th


def neg_ll(x):
    if np.any(x < lbf) or np.any(x > ubf):
        return 1e10
    ll = log_likelihood(build(x), MODEL, full, GROUP)
    return -ll if np.isfinite(ll) else 1e10


def fit():
    rng = np.random.default_rng(0)
    x0 = 0.5 * (lbf + ubf)
    x0[2] = 0.15
    best = None
    for trial in range(24):
        xs = x0 + rng.normal(scale=0.15 * (ubf - lbf), size=x0.shape) if trial else x0.copy()
        xs = np.clip(xs, lbf + 1e-6, ubf - 1e-6)
        for method in ('L-BFGS-B', 'Powell', 'L-BFGS-B'):
            res = minimize(neg_ll, xs, method=method, bounds=list(zip(lbf, ubf)))
            xr = np.clip(res.x, lbf, ubf)
            fr = neg_ll(xr)
            if best is None or fr < best[0]:
                best = (fr, xr)
            xs = xr
    return build(best[1]), -best[0]


def rhs(y, t, a1, k1, mu1, D):
    T, M = y
    C = D * np.exp(-KAPPA3 * t)
    return [a1 * T * (1 - T / k1) * (1 + BETA1 * M) - mu1 * C * T,
            ALPHA2 * T * M * (1 - M / K2) - KAPPA2 * M]


def stored_fit():
    """Parameters of the published fit, so that the figure can be redrawn without refitting.

    The optimization of fit() takes tens of minutes; main() repeats it only when the stored result is
    missing or does not match the current group, or when the script is called with --refit.
    """
    path = os.path.join(TMC, 'scenario7_v2.csv')
    if not os.path.exists(path):
        return None
    vals = {}
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] in ('alpha1', 'K1_mm3', 'mu1', 'loglik'):
                vals[row[0]] = float(row[1])
            elif row[0] == 'fitted_T0_per_mouse':
                vals['ics'] = np.array([float(x) for x in row[1:]])
    if not {'alpha1', 'K1_mm3', 'mu1', 'loglik', 'ics'} <= set(vals) or len(vals['ics']) != gs:
        return None
    return vals['alpha1'], vals['K1_mm3'], vals['mu1'], vals['ics'], vals['loglik']


def main():
    stored = None if '--refit' in sys.argv else stored_fit()
    if stored is None:
        th, ll = fit()
        a1, k1, mu1 = 10 ** th[0], 10 ** th[1], th[3]
        ics = th[8:8 + gs]
    else:
        a1, k1, mu1, ics, ll = stored
        print('using the stored fit of scenario7_v2.csv (run with --refit to repeat the optimization)')
    print(f'MLE ll={ll:.3f} alpha1={a1:.4f} K1={k1:.1f} mu1={mu1:.4f}')
    print('fitted T(0) per mouse:', np.round(ics, 2))

    days = np.unique(np.concatenate([m['data'][:, 0] for m in full[GROUP]]))
    data_mean = np.array([np.mean(np.concatenate([m['data'][m['data'][:, 0] == d, 1] for m in full[GROUP]]))
                          for d in days])
    t = np.linspace(0, HORIZON, 1201)
    traj = {}
    for D in DOSES:
        per_mouse = np.array([odeint(rhs, [ic, ic / 2.0], t, args=(a1, k1, mu1, D), rtol=1e-9, atol=1e-9)[:, 0]
                              for ic in ics])
        traj[D] = per_mouse
    pred_D1 = np.interp(days, t, traj[1.0].mean(axis=0))
    rel_err = np.abs(pred_D1 - data_mean) / data_mean
    print('day, data mean, model mean (D=1):')
    for d, y, p in zip(days, data_mean, pred_D1):
        print(f'  {d:5.1f} {y:8.1f} {p:8.1f}')

    rows = []
    for D in DOSES:
        esc = []
        for ic, T in zip(ics, traj[D]):
            idx = np.where(T >= 10 * ic)[0]
            esc.append(t[idx[0]] if len(idx) else np.nan)
        esc = np.array(esc)
        n_esc = int(np.sum(np.isfinite(esc)))
        med = float(np.nanmedian(esc)) if n_esc else np.nan
        rows.append([D, med, n_esc, gs])
        print(f'D={D}: median time to 10x fitted baseline = {med:.1f} d ({n_esc}/{gs} mice escape within {HORIZON:.0f} d)')

    with open(os.path.join(TMC, 'scenario7_v2.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['alpha1', a1]); w.writerow(['K1_mm3', k1]); w.writerow(['mu1', mu1]); w.writerow(['loglik', ll])
        w.writerow(['fitted_T0_per_mouse'] + list(ics))
        w.writerow(['median_abs_rel_error_D1_vs_data_mean', float(np.median(rel_err))])
        w.writerow([]); w.writerow(['dose_multiple', 'median_days_to_10x_baseline', 'n_escaped', 'n_mice'])
        w.writerows(rows)

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    cols = ['#fdb863', '#e66101', '#5e3c99', '#1b7837']
    for D, c in zip(DOSES, cols):
        ax.plot(t, traj[D].mean(axis=0), color=c, lw=2.0, label=rf'$D={D}\times$ administered dose')
    ax.plot(days, data_mean, 'ko', ms=5, label='long-hinge data (group mean)')
    ax.set_yscale('log'); ax.set_xlabel('time (days)'); ax.set_ylabel(r'mean $T(t)$ (mm$^3$), log scale')
    # panel label inside the axes, in bold, as in the published figures of this chapter
    # (lowercase letter with a closing parenthesis); the description is offset in points
    _anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
                   fontsize=16)
    ax.annotate('a)', fontweight='bold', **_anchor)
    ax.annotate('dose response, calibrated minimal model', xytext=(20, 0), textcoords='offset points', **_anchor)
    ax.legend(loc='lower right', framealpha=0.95)
    labels_b = [rf'{D}$\times$' for D in DOSES]
    vals = [r[1] if np.isfinite(r[1]) else 0 for r in rows]
    bx.bar(labels_b, vals, color=cols, edgecolor='k', lw=0.6)
    for i, r in enumerate(rows):
        txt = f'{r[1]:.1f} d' if np.isfinite(r[1]) else f'no escape\nwithin {HORIZON:.0f} d'
        bx.text(i, (vals[i] if vals[i] else 0) + 0.8, txt, ha='center', fontsize=9.5)
    bx.set_xlabel('CAR T-cell dose (multiples of the administered dose)')
    bx.set_ylabel(r'median time to $10\times$ baseline volume (days)')
    _anchor = dict(xy=(0.028, 0.96), xycoords='axes fraction', ha='left', va='top',
                   fontsize=16)
    bx.annotate('b)', fontweight='bold', **_anchor)
    bx.annotate('control duration by dose', xytext=(20, 0), textcoords='offset points', **_anchor)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig(os.path.join('figures', 'scenario7_dose_response_calibrated.png'), dpi=220, bbox_inches='tight')
    print('saved')


if __name__ == '__main__':
    main()
