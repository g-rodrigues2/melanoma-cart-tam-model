"""Global sensitivity analysis (eFAST, SALib): final-time and time-dependent indices.

Extracted from notebooks/melanoma_CART_TAM.ipynb.
"""
import os

import numpy as np
from numba import njit, prange
from SALib.sample import fast_sampler
from SALib.analyze import fast
import pandas as pd
import matplotlib.pyplot as plt


def sis(TMC, t, alpha1, alpha2, alpha3, beta1, K1, K2, K3, mu1, kappa2, kappa3):
    T, M, C = TMC
    dT = alpha1 * T * (1 - (T / K1)) * (1 + (beta1 * M)) - mu1 * T * C
    dM = alpha2 * T * M * (1 - (M / K2)) - kappa2 * M
    dC = (alpha3 * T * C) / (K3 + T) - kappa3 * C
    return dT, dM, dC


# === Compiled integration of all samples ===
# The model is integrated in nopython mode with the Dormand-Prince 5(4) pair and adaptive steps, stopping
# exactly at the analyzed times, and the samples run in parallel on all cores (numba prange). With
# RTOL = 1e-10 the trajectories agree with scipy's odeint (rtol 1e-12) to a relative error below 1e-7
# (check_against_odeint), and the whole sample takes minutes instead of hours.
RTOL, ATOL = 1e-10, 1e-6


@njit
def _rhs(y, p):
    T, M, C = y[0], y[1], y[2]
    out = np.empty(3)
    out[0] = p[0] * T * (1 - T / p[4]) * (1 + p[3] * M) - p[7] * T * C
    out[1] = p[1] * T * M * (1 - M / p[5]) - p[8] * M
    out[2] = p[2] * T * C / (p[6] + T) - p[9] * C
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
        k5 = _rhs(y + h_try * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4), p)
        k6 = _rhs(y + h_try * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4
                               - 5103 / 18656 * k5), p)
        y_new = y + h_try * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
        k7 = _rhs(y_new, p)
        err = h_try * (71 / 57600 * k1 - 71 / 16695 * k3 + 71 / 1920 * k4 - 17253 / 339200 * k5
                       + 22 / 525 * k6 - 1 / 40 * k7)
        ratio = 0.0
        for i in range(3):
            ratio = max(ratio, abs(err[i]) / (atol + rtol * max(abs(y[i]), abs(y_new[i]))))
        if ratio <= 1.0:
            t = t_end if last else t + h_try
            y = y_new
            k1 = k7
            factor = 10.0 if ratio == 0.0 else min(10.0, 0.9 * ratio ** -0.2)
            if not last or h_try >= h:
                h = h_try * factor
            while k < n and t_out[k] <= t:
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


def check_against_odeint(X, t_out, n=50):
    """Largest relative difference to odeint (rtol 1e-12) over the first n samples."""
    from scipy.integrate import odeint
    Y = solve_all(np.array(TMC0), X[:n], t_out, RTOL, ATOL)
    worst = 0.0
    for i in range(n):
        ref = odeint(sis, TMC0, np.concatenate(([0.0], t_out)) if t_out[0] > 0 else t_out, args=tuple(X[i]),
                     rtol=1e-12, atol=1e-6, mxstep=100000)
        ref = ref[1:] if t_out[0] > 0 else ref
        worst = max(worst, float(np.max(np.abs(Y[i] - ref) / np.maximum(np.abs(ref), 1.0))))
    return worst


alpha1 = 0.69
K1 = 10**9
beta1 = 5.9e-11
mu1 = 1e-7

alpha2 = 1e-7
K2 = 10**9
kappa2 = 0.34

alpha3 = 0.9
K3 = 2e7
kappa3 = 0.35

TMC0 = [2e6, 1e6, 1e7]

t = np.linspace(0, 60, 1000)
n_t = len(t)

param_names = [
    'alpha1', 'alpha2', 'alpha3',
    'beta1', 'K1', 'K2', 'K3',
    'mu1', 'kappa2', 'kappa3'
]

base_values = np.array([
    alpha1, alpha2, alpha3,
    beta1, K1, K2, K3,
    mu1, kappa2, kappa3
])

bounds = np.column_stack((0.8 * base_values, 1.2 * base_values))

problem = {
    'num_vars': len(param_names),
    'names': param_names,
    'bounds': bounds
}

N = 10000
M_fast = 4
seed = 123

def main():
    """Runs the eFAST design, saves the trajectories and the indices, and draws the figures."""
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X = fast_sampler.sample(problem, N, M=M_fast, seed=seed)
    n_samples = X.shape[0]

    print(f"Number of generated samples: {n_samples}")

    # analyzed times: every 20th point of the grid (time-dependent indices) and the final time
    time_indices = np.arange(0, n_t, 20)
    t_out = np.unique(np.concatenate((t[time_indices], t[-1:])))
    column_of = {int(i): int(np.searchsorted(t_out, t[i])) for i in time_indices}

    print(f"Largest relative difference to odeint (first 50 samples): {check_against_odeint(X, t_out):.1e}")
    Y_sel = solve_all(np.array(TMC0, dtype=float), X, t_out, RTOL, ATOL)

    np.savez_compressed(
        'results/fast_model_outputs_TMC.npz',
        X=X,
        t=t_out,
        Y=Y_sel,
        param_names=np.array(param_names)
    )

    print("File 'fast_model_outputs_TMC.npz' saved.")

    Y_end = Y_sel[:, -1, :]

    Si_T_end = fast.analyze(
        problem,
        Y_end[:, 0],
        M=M_fast,
        num_resamples=100,
        conf_level=0.95,
        print_to_console=True,
        seed=seed
    )

    Si_M_end = fast.analyze(
        problem,
        Y_end[:, 1],
        M=M_fast,
        num_resamples=100,
        conf_level=0.95,
        print_to_console=True,
        seed=seed
    )

    Si_C_end = fast.analyze(
        problem,
        Y_end[:, 2],
        M=M_fast,
        num_resamples=100,
        conf_level=0.95,
        print_to_console=True,
        seed=seed
    )

    df_T_end = pd.DataFrame({
        'parameter': param_names,
        'S1': Si_T_end['S1'],
        'ST': Si_T_end['ST'],
        'S1_conf': Si_T_end['S1_conf'],
        'ST_conf': Si_T_end['ST_conf']
    })
    df_T_end.to_csv('results/fast_Si_T_end.csv', index=False)

    df_M_end = pd.DataFrame({
        'parameter': param_names,
        'S1': Si_M_end['S1'],
        'ST': Si_M_end['ST'],
        'S1_conf': Si_M_end['S1_conf'],
        'ST_conf': Si_M_end['ST_conf']
    })
    df_M_end.to_csv('results/fast_Si_M_end.csv', index=False)

    df_C_end = pd.DataFrame({
        'parameter': param_names,
        'S1': Si_C_end['S1'],
        'ST': Si_C_end['ST'],
        'S1_conf': Si_C_end['S1_conf'],
        'ST_conf': Si_C_end['ST_conf']
    })
    df_C_end.to_csv('results/fast_Si_C_end.csv', index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(param_names, Si_T_end['S1'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('S1 (T, t=60)')
    plt.tight_layout()
    plt.savefig('results/fast_S1_T_end_bar.pdf', format='pdf')
    plt.close()

    print("CSV files and bar plot at the final time were saved.")

    t_sel = t[time_indices]
    n_times_sel = len(t_sel)
    n_par = problem['num_vars']

    S1_T = np.zeros((n_times_sel, n_par))
    ST_T = np.zeros((n_times_sel, n_par))
    S1_T_conf = np.zeros((n_times_sel, n_par))
    ST_T_conf = np.zeros((n_times_sel, n_par))

    S1_M = np.zeros((n_times_sel, n_par))
    ST_M = np.zeros((n_times_sel, n_par))
    S1_M_conf = np.zeros((n_times_sel, n_par))
    ST_M_conf = np.zeros((n_times_sel, n_par))

    S1_C = np.zeros((n_times_sel, n_par))
    ST_C = np.zeros((n_times_sel, n_par))
    S1_C_conf = np.zeros((n_times_sel, n_par))
    ST_C_conf = np.zeros((n_times_sel, n_par))

    print("Starting time-dependent FAST analysis...")

    for k, idx in enumerate(time_indices):
        print(f"  Time t = {t[idx]:.2f} (index {idx})")

        Y_T = Y_sel[:, column_of[int(idx)], 0]
        Y_M = Y_sel[:, column_of[int(idx)], 1]
        Y_C = Y_sel[:, column_of[int(idx)], 2]

        Si_T_t = fast.analyze(
            problem,
            Y_T,
            M=M_fast,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
            seed=seed
        )

        Si_M_t = fast.analyze(
            problem,
            Y_M,
            M=M_fast,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
            seed=seed
        )

        Si_C_t = fast.analyze(
            problem,
            Y_C,
            M=M_fast,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
            seed=seed
        )

        S1_T[k, :] = Si_T_t['S1']
        ST_T[k, :] = Si_T_t['ST']
        S1_T_conf[k, :] = Si_T_t['S1_conf']
        ST_T_conf[k, :] = Si_T_t['ST_conf']

        S1_M[k, :] = Si_M_t['S1']
        ST_M[k, :] = Si_M_t['ST']
        S1_M_conf[k, :] = Si_M_t['S1_conf']
        ST_M_conf[k, :] = Si_M_t['ST_conf']

        S1_C[k, :] = Si_C_t['S1']
        ST_C[k, :] = Si_C_t['ST']
        S1_C_conf[k, :] = Si_C_t['S1_conf']
        ST_C_conf[k, :] = Si_C_t['ST_conf']

    np.savez_compressed(
        'results/fast_time_dependent_sensitivity_TMC.npz',
        t_sel=t_sel,
        param_names=np.array(param_names),
        S1_T=S1_T, ST_T=ST_T,
        S1_T_conf=S1_T_conf, ST_T_conf=ST_T_conf,
        S1_M=S1_M, ST_M=ST_M,
        S1_M_conf=S1_M_conf, ST_M_conf=ST_M_conf,
        S1_C=S1_C, ST_C=ST_C,
        S1_C_conf=S1_C_conf, ST_C_conf=ST_C_conf
    )

    df_S1_T_time = pd.DataFrame(
        np.column_stack([t_sel.reshape(-1, 1), S1_T]),
        columns=['time'] + param_names
    )
    df_S1_T_time.to_csv('results/fast_S1_T_time.csv', index=False)

    df_S1_M_time = pd.DataFrame(
        np.column_stack([t_sel.reshape(-1, 1), S1_M]),
        columns=['time'] + param_names
    )
    df_S1_M_time.to_csv('results/fast_S1_M_time.csv', index=False)

    df_S1_C_time = pd.DataFrame(
        np.column_stack([t_sel.reshape(-1, 1), S1_C]),
        columns=['time'] + param_names
    )
    df_S1_C_time.to_csv('results/fast_S1_C_time.csv', index=False)

    print("Time-dependent FAST analysis completed.")
    print("Files '.npz' and '.csv' were saved. You can now use the plotting script.")


if __name__ == '__main__':
    main()
