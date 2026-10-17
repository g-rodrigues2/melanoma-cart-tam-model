"""Global sensitivity analysis (eFAST) over the uncertainty declared in Table 3.1.

The companion of `efast_sensitivity.py`, which perturbs every parameter by +-20% around a reference
point. That design answers how the outputs respond to a small uncertainty around one operating point.
This one answers a different question, namely how they respond over the uncertainty the thesis itself
declares in Table 3.1, which for three parameters is far wider than +-20%:

    alpha1  [0.69, 0.97]           as declared
    beta1   [1e-11, 3e-6]          declared as [0, 3e-6]; sampled log-uniformly, see below
    mu3     [0, 5e-9]              as declared, and included here, so p = 11
    kappa3  [0, 0.9]               as declared

and for the remaining eight parameters, which Table 3.1 gives as point values, +-20% around them
(alpha2, alpha3, K1, K2, K3, mu1, kappa2).

WHY beta1 IS SAMPLED IN log10. Its declared range covers more than five decades, and a uniform
distribution over [0, 3e-6] puts over 99.99% of its mass above 1e-9, so it would never visit the
region where the Hopf threshold of Section 3.3.2 lies (beta1* = 5.9e-11) nor the value used in
Scenario 1. A log-uniform distribution is the usual choice for a rate known only to within orders of
magnitude, and it is the one adopted here. A log scale cannot include zero, so the lower end is set
at 1e-11, which is four decades below the smallest value of beta1 used anywhere in the chapter. The
reported index is therefore the index of log10(beta1), which is what the design varies uniformly.

The immunosuppression parameter mu3, absent from the +-20% design, enters the CAR T-cell equation as
-mu3 C T, so the model integrated here is system (3.1) in full.

Outputs (results/): table31_Si_{T,M,C}_end.csv, table31_S1_{T,M,C}_time.csv,
table31_ST_{T,M,C}_time.csv, table31_model_outputs.npz and table31_diagnostics.txt.

Run: python efast_sensitivity_table31.py
"""
import os
import time

import numpy as np
from numba import njit, prange
from SALib.sample import fast_sampler
from SALib.analyze import fast

os.makedirs("results", exist_ok=True)

# --------------------------------------------------------------------------- model
# Same Dormand-Prince 5(4) scheme with adaptive steps as efast_sensitivity.py, with mu3 restored,
# so the parameter vector has eleven entries.
RTOL, ATOL = 1e-10, 1e-6


def sis(TMC, t, alpha1, alpha2, alpha3, beta1, K1, K2, K3, mu1, kappa2, kappa3, mu3):
    """Right-hand side in scipy order, used only by the verification against odeint."""
    T, M, C = TMC
    dT = alpha1 * T * (1 - T / K1) * (1 + beta1 * M) - mu1 * T * C
    dM = alpha2 * T * M * (1 - M / K2) - kappa2 * M
    dC = alpha3 * T * C / (K3 + T) - mu3 * C * T - kappa3 * C
    return dT, dM, dC


@njit
def _rhs(y, p):
    T, M, C = y[0], y[1], y[2]
    out = np.empty(3)
    out[0] = p[0] * T * (1 - T / p[4]) * (1 + p[3] * M) - p[7] * T * C
    out[1] = p[1] * T * M * (1 - M / p[5]) - p[8] * M
    out[2] = p[2] * T * C / (p[6] + T) - p[10] * C * T - p[9] * C
    return out


@njit
def _solve(y0, p, t_out, rtol, atol):
    n = t_out.shape[0]
    out = np.empty((n, 3))
    y = y0.copy()
    t = 0.0
    h = 1e-3
    k = 0
    while k < n and t_out[k] <= t:
        out[k] = y
        k += 1
    k1 = _rhs(y, p)
    while k < n:
        t_end = t_out[k]
        last = t + h >= t_end
        h_try = t_end - t if last else h
        k2 = _rhs(y + h_try * (1 / 5 * k1), p)
        k3 = _rhs(y + h_try * (3 / 40 * k1 + 9 / 40 * k2), p)
        k4 = _rhs(y + h_try * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3), p)
        k5 = _rhs(y + h_try * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3
                               - 212 / 729 * k4), p)
        k6 = _rhs(y + h_try * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3
                               + 49 / 176 * k4 - 5103 / 18656 * k5), p)
        y5 = y + h_try * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4
                          - 2187 / 6784 * k5 + 11 / 84 * k6)
        k7 = _rhs(y5, p)
        y4 = y + h_try * (5179 / 57600 * k1 + 7571 / 16695 * k3 + 393 / 640 * k4
                          - 92097 / 339200 * k5 + 187 / 2100 * k6 + 1 / 40 * k7)
        err = 0.0
        for j in range(3):
            sc = atol + rtol * max(abs(y[j]), abs(y5[j]))
            e = abs(y5[j] - y4[j]) / sc
            if e > err:
                err = e
        ratio = err if err > 1e-16 else 1e-16
        if err <= 1.0:
            t = t + h_try
            y = y5
            k1 = k7
            if not last:
                h = h_try * min(5.0, 0.9 * ratio ** -0.2)
            while k < n and t_out[k] <= t + 1e-12:
                out[k] = y
                k += 1
        else:
            h = h_try * max(0.2, 0.9 * ratio ** -0.2)
    return out


@njit(parallel=True)
def solve_all(y0, X, t_out, rtol, atol):
    Y = np.empty((X.shape[0], t_out.shape[0], 3))
    for i in prange(X.shape[0]):
        Y[i] = _solve(y0, X[i], t_out, rtol, atol)
    return Y


# --------------------------------------------------------------------------- design
PARAM_NAMES = ['alpha1', 'alpha2', 'alpha3', 'log10_beta1', 'K1', 'K2', 'K3',
               'mu1', 'kappa2', 'kappa3', 'mu3']
# Table 3.1: declared interval where it gives one, +-20% around the point value otherwise.
BOUNDS = [
    [0.69, 0.97],              # alpha1   declared
    [0.8e-7, 1.2e-7],          # alpha2   point value 1e-7
    [0.8 * 0.9, 1.2 * 0.9],    # alpha3   point value 0.9
    [-11.0, np.log10(3e-6)],   # log10(beta1)   declared [0, 3e-6], log-uniform from 1e-11
    [0.8e9, 1.2e9],            # K1       point value 1e9
    [0.8e9, 1.2e9],            # K2       point value 1e9
    [0.8 * 2e7, 1.2 * 2e7],    # K3       point value 2e7
    [0.8e-7, 1.2e-7],          # mu1      point value 1e-7
    [0.8 * 0.34, 1.2 * 0.34],  # kappa2   point value 0.34
    [0.0, 0.9],                # kappa3   declared
    [0.0, 5e-9],               # mu3      declared
]
TMC0 = [2e6, 1e6, 1e7]
N, M_FAST, SEED, N_RESAMPLES = 10000, 4, 123, 100
t_grid = np.linspace(0, 60, 1000)
T_IDX = np.arange(0, 1000, 20)          # analyzed times, as in the +-20% design
t_out = np.ascontiguousarray(t_grid[T_IDX])

problem = {'num_vars': len(PARAM_NAMES), 'names': PARAM_NAMES, 'bounds': BOUNDS}


def to_model(X_design):
    """Design matrix (log10 beta1) -> parameter matrix in model units."""
    X = X_design.copy()
    X[:, 3] = 10.0 ** X[:, 3]
    return np.ascontiguousarray(X)


def check_against_odeint(X_model, n=60):
    """Largest relative difference to odeint (rtol 1e-12), over n samples."""
    from scipy.integrate import odeint
    Y = solve_all(np.array(TMC0), X_model[:n], t_out, RTOL, ATOL)
    worst = 0.0
    for i in range(n):
        ref = odeint(sis, TMC0, np.concatenate(([0.0], t_out)), args=tuple(X_model[i]),
                     rtol=1e-12, atol=1e-6, mxstep=1000000)[1:]
        worst = max(worst, float(np.max(np.abs(Y[i] - ref) / np.maximum(np.abs(ref), 1.0))))
    return worst


def analyze(Y_scalar):
    si = fast.analyze(problem, Y_scalar, M=M_FAST, num_resamples=N_RESAMPLES,
                      print_to_console=False, seed=SEED)
    return {k: np.asarray(v) for k, v in si.items()}


if __name__ == '__main__':
    LINES = []

    def say(s=''):
        print(s)
        LINES.append(s)

    t0 = time.time()
    X_design = fast_sampler.sample(problem, N, M=M_FAST, seed=SEED)
    X_model = to_model(X_design)
    say('=' * 78)
    say('eFAST OVER THE UNCERTAINTY DECLARED IN TABLE 3.1')
    say('=' * 78)
    say(f'  samples {X_design.shape[0]} = N {N} x p {len(PARAM_NAMES)}, '
        f'harmonic order M = {M_FAST}, seed {SEED}')
    say(f'  beta1 sampled in [{X_model[:, 3].min():.3e}, {X_model[:, 3].max():.3e}] cells^-1 '
        f'({np.log10(X_model[:, 3].max() / X_model[:, 3].min()):.1f} decades)')

    Y = solve_all(np.array(TMC0), X_model, t_out, RTOL, ATOL)
    say(f'  integration of {X_model.shape[0]} trajectories: {time.time() - t0:.0f} s')

    worst = check_against_odeint(X_model)
    say(f'  verification against odeint (rtol 1e-12), 60 samples: worst relative difference '
        f'{worst:.2e}')

    say('')
    say('  output spread at t = 60')
    say(f'  {"output":8s} {"minimum":>12s} {"maximum":>12s} {"decades":>8s} {"CV":>7s} '
        f'{"% within 1% of K1":>18s}')
    for j, lab in enumerate('TMC'):
        y = Y[:, -1, j]
        dec = np.log10(y.max() / max(y.min(), 1e-300))
        sat = 100.0 * np.mean(y > 0.99 * X_model[:, 4]) if lab == 'T' else np.nan
        say(f'  {lab + "(60)":8s} {y.min():12.4e} {y.max():12.4e} {dec:8.2f} '
            f'{y.std() / y.mean():7.3f} {sat:18.1f}')

    say('')
    say('  first-order and total-effect indices at t = 60')
    for j, lab in enumerate('TMC'):
        y = Y[:, -1, j]
        si = analyze(y)
        si_log = analyze(np.log10(np.maximum(y, 1e-300)))
        order = np.argsort(-si['ST'])
        say(f'\n  --- {lab}(60) ---')
        say(f'  {"parameter":12s} {"S1":>9s} {"S1_conf":>9s} {"ST":>9s} {"ST_conf":>9s} '
            f'{"ST (log Y)":>11s}')
        for i in order:
            say(f'  {PARAM_NAMES[i]:12s} {si["S1"][i]:9.4f} {si["S1_conf"][i]:9.4f} '
                f'{si["ST"][i]:9.4f} {si["ST_conf"][i]:9.4f} {si_log["ST"][i]:11.4f}')
        say(f'  sum S1 = {si["S1"].sum():.3f}   sum ST = {si["ST"].sum():.3f}')
        rank_lin = [PARAM_NAMES[i] for i in np.argsort(-si['ST'])][:5]
        rank_log = [PARAM_NAMES[i] for i in np.argsort(-si_log['ST'])][:5]
        say(f'  top 5 by ST, Y      : {", ".join(rank_lin)}')
        say(f'  top 5 by ST, log Y  : {", ".join(rank_log)}')
        say(f'  the two agree: {rank_lin == rank_log}')
        import pandas as pd
        pd.DataFrame({'parameter': PARAM_NAMES, 'S1': si['S1'], 'ST': si['ST'],
                      'S1_conf': si['S1_conf'], 'ST_conf': si['ST_conf'],
                      'ST_logY': si_log['ST']}).to_csv(
            os.path.join('results', f'table31_Si_{lab}_end.csv'), index=False)

    # time-dependent indices, on the same analyzed times as the +-20% design
    say('')
    say('  time-dependent indices')
    import pandas as pd
    for j, lab in enumerate('TMC'):
        S1_t, ST_t = [], []
        for k in range(len(t_out)):
            y = Y[:, k, j]
            if y.std() == 0:
                S1_t.append(np.zeros(len(PARAM_NAMES)))
                ST_t.append(np.zeros(len(PARAM_NAMES)))
                continue
            si = fast.analyze(problem, y, M=M_FAST, num_resamples=1,
                              print_to_console=False, seed=SEED)
            S1_t.append(np.asarray(si['S1']))
            ST_t.append(np.asarray(si['ST']))
        pd.DataFrame(np.array(S1_t), columns=PARAM_NAMES, index=t_out).to_csv(
            os.path.join('results', f'table31_S1_{lab}_time.csv'), index_label='t')
        pd.DataFrame(np.array(ST_t), columns=PARAM_NAMES, index=t_out).to_csv(
            os.path.join('results', f'table31_ST_{lab}_time.csv'), index_label='t')
    say(f'  written for T, M and C at {len(t_out)} times')

    np.savez_compressed(os.path.join('results', 'table31_model_outputs.npz'),
                        X_design=X_design, X_model=X_model, Y=Y, t=t_out,
                        param_names=np.array(PARAM_NAMES))
    say('')
    say(f'  total time {time.time() - t0:.0f} s')
    with open(os.path.join('results', 'table31_diagnostics.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LINES) + '\n')
    print('\nresults in results/table31_*')
