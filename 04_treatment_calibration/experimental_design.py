"""In silico experimental design with the Fisher information: which data would make beta1 and mu1 recoverable?

Compares the marginal uncertainty (from the FIM) of beta1 and mu1 across
  - operating regime: escape (bounded maximum-likelihood estimate of the long-hinge group, current data) and
    coexistence (theoretical values of the model, near the Hopf threshold beta1*);
  - observed states: {T}, {T,M}, {T,M,C};
  - observation window: 30, 60, 120, 300 days;
  - number of observation times (uniform sampling).

Relative noise sigma_rel per observed state (default 15%): sigma_obs(t) = sigma_rel * value.
FIM = sum_obs (1/sigma_obs^2) s s^T with s = d[state]/dtheta (8 model parameters); the marginal std of theta_j
is sqrt((FIM^-1)_jj) (spectral pseudo-inverse).

alpha1, K1 and beta1 (indices 0, 1, 2) are in log10, so their std is in orders of magnitude.
Output: Output_Calibration/TMC/experimental_design.csv and a diagnostic figure (the thesis figure is drawn by
figures_thesis/plot_experimental_design.py).
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from models import solve_model

TMC = './Output_Calibration/TMC'
os.makedirs(TMC, exist_ok=True)

PARN = ['log10_a1', 'log10_K1', 'log10_b1', 'mu1', 'a3', 'K3', 'k3', 'mu3']
LOGPAR = {0, 1, 2}
SIGMA_REL = 0.15
IB1, IMU1 = 2, 3  # indices of beta1 and mu1

# --- Operating points (log10 in indices 0, 1, 2) ---
# ode_args of _mac_treat = (10**la1, 10**lK1, 10**lb1, mu1, a3, K3, k3, mu3); theta is stored as
# (la1, lK1, lb1, mu1, a3, K3, k3, mu3) with the initial condition (T0, M0, C0).
def theta_phys(a1, K1, b1, mu1, a3, K3, k3, mu3):
    return np.array([np.log10(a1), np.log10(K1), np.log10(b1),
                     mu1, a3, K3, k3, mu3])

# (1) ESCAPE: bounded maximum-likelihood estimate of the long-hinge group (mle_bounded.py),
#     initial condition at the median fitted T(0) of the group. An earlier version used a
#     point with kappa3 outside the prior.
_Z = np.load(os.path.join('./Output_Calibration/TMC', 'mle_bounded_cart_sl_b16.npz'))
_T0 = float(np.median(_Z['theta'][8:16]))
OP_ESCAPE = dict(theta=np.asarray(_Z['theta'][:8], float), ic=[_T0, _T0 / 2.0, 1.0], name='escape_SL')
# (2) COEXISTENCE: theoretical values of the model (mm^3), near the Hopf threshold:
#     alpha1=0.69, K1=1000, beta1=5.9e-5 (~beta1*), mu1=0.1, a3=0.9, K3=20, k3=0.35, mu3=0.
#     Initial condition T0=2, M0=1, C0=10 (2e6, 1e6, 1e7 cells / 1e6).
OP_COEXIST = dict(
    theta=theta_phys(0.69, 1000, 5.9e-5, 0.1, 0.9, 20, 0.35, 1e-9),
    ic=[2.0, 1.0, 10.0], name='coexistence_Hopf')


def solve3(theta, ic, tgrid):
    a = list(theta.copy())
    a[0] = 10**a[0]; a[1] = 10**a[1]; a[2] = 10**a[2]
    sol = solve_model('_mac_treat', tgrid, tuple(a), ic)
    return sol  # (len,3) col T,M,C


def sens(theta, ic, tgrid):
    """d[T,M,C]/dtheta_j for j = 0..7. Returns (3, len, 8)."""
    base = solve3(theta, ic, tgrid)
    S = np.zeros((3, len(tgrid), 8))
    for j in range(8):
        h = 1e-5*(1.0 + abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h; tm[j] -= h
        sp, sm = solve3(tp, ic, tgrid), solve3(tm, ic, tgrid)
        S[:, :, j] = ((sp - sm)/(2*h)).T
    return base, S


def marg_std(FIM, idx):
    w, V = np.linalg.eigh(FIM)
    wmax = max(w.max(), 1e-300); tol = wmax*1e-12
    s = sum(V[idx, k]**2/(w[k] if w[k] > tol else tol) for k in range(len(w)))
    return np.sqrt(s)


def fim_for_design(base, S, tgrid, states, n_pts, window):
    """FIM of observing 'states' (subset of [0,1,2] = T,M,C) at n_pts uniform times up to 'window' days, with
    relative noise."""
    mask = tgrid <= window
    ti = np.where(mask)[0]
    if len(ti) < n_pts:
        sel = ti
    else:
        sel = ti[np.linspace(0, len(ti)-1, n_pts).astype(int)]
    FIM = np.zeros((8, 8))
    for s_state in states:
        for i in sel:
            val = abs(base[i, s_state])
            sigma = max(SIGMA_REL*val, 1e-6)
            svec = S[s_state, i, :]
            FIM += np.outer(svec, svec)/sigma**2
    return FIM


def main():
    rows = []
    designs = [
        ('T', [0]), ('T+M', [0, 1]), ('T+M+C', [0, 1, 2]),
    ]
    windows = [30, 60, 120, 300]
    n_pts = 8
    for OP in (OP_ESCAPE, OP_COEXIST):
        tgrid = np.linspace(0, max(windows), 601)
        base, S = sens(OP['theta'], OP['ic'], tgrid)
        print(f"\n=== Regime: {OP['name']} ===")
        print(f"  {'design':8s} {'window':>6s}  {'std(log10 b1)':>13s}  {'std(mu1)':>9s}")
        for wname in windows:
            for dname, states in designs:
                FIM = fim_for_design(base, S, tgrid, states, n_pts, wname)
                sb1 = marg_std(FIM, IB1)
                smu1 = marg_std(FIM, IMU1)
                rows.append({'regime': OP['name'], 'design': dname,
                             'window_days': wname, 'n_pts': n_pts,
                             'std_log10_beta1': f'{sb1:.3g}',
                             'std_mu1': f'{smu1:.3g}'})
                print(f"  {dname:8s} {wname:6d}  {sb1:13.3g}  {smu1:9.3g}")

    with open(os.path.join(TMC, 'experimental_design.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        w.writerows(rows)

    # diagnostic figure: std(beta1) against the window, one panel per regime
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, OP in zip(axes, (OP_ESCAPE, OP_COEXIST)):
        for dname, _ in designs:
            xs = windows
            ys = [float(r['std_log10_beta1']) for r in rows
                  if r['regime'] == OP['name'] and r['design'] == dname]
            ax.plot(xs, ys, 'o-', label=dname)
        ax.axhline(0.3, color='green', linestyle='--', alpha=0.6,
                   label='target: std<0.3 (factor ~2)')
        ax.set_yscale('log'); ax.set_xlabel('observation window (days)')
        ax.set_ylabel('marginal std of log10(beta1) [orders of magnitude]')
        ax.set_title(f"Regime: {OP['name']}")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(TMC, 'experimental_design_diagnostic.png'),
                dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"\nCSV and figure in {TMC}/experimental_design*")


if __name__ == '__main__':
    main()
