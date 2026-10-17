"""MCMC calibration of the minimal identifiable submodel.

The non-identifiable parameters are fixed and the CAR T-cell dynamics is anchored to the observed decay of the
persistence (Jilani et al. Fig. 6b: the VCN decreases about 7-fold in 27 days):
  - beta1 fixed (nuisance)                    -> log10_beta1 = -4  (= 1e-4 mm^-3)
  - alpha3 ~ 0 (no CAR T-cell proliferation)  -> C(t) decays exponentially
  - kappa3 = 0.09 /day  (decay rate of the VCN, Fig. 6b)
  - mu3 = 0
Free: log10_alpha1, log10_K1, mu1, the initial condition of each mouse and std. mu1 then multiplies a known
C(t) = exp(-kappa3 t) and is identifiable.

Runs a short emcee chain per group, reports tau, ESS and acceptance, and the posterior of mu1.
Output: Output_Calibration/TMC/reduced_mu1.csv and a figure.
"""
import os, csv, numpy as np
import emcee
import matplotlib.pyplot as plt

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from inference import log_likelihood

OUT = './Output_Calibration'; TMC = './Output_Calibration/TMC'
os.makedirs(TMC, exist_ok=True)
MODEL = '_mac_treat'
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']

# fixed values (indices in the theta of _mac_treat: 2=log10_beta1, 4=alpha3, 5=K3, 6=kappa3, 7=mu3)
FIX = {2: -4.0, 4: 1e-9, 5: 20.0, 6: 0.09, 7: 0.0}
FREE_MODEL = [0, 1, 3]   # log10_alpha1, log10_K1, mu1
NSTEPS = 4000

full = load_treatment_data()
max_ic = find_max_initial_condition(full)[0]


def run_group(group):
    gs = len(full[group])
    lb, ub, labels = define_bounds_labels(MODEL, group_size=gs, max_ic=max_ic,
                                          bool_multiple_mice=True)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    ndim_full = len(lb)
    free = FREE_MODEL + list(range(8, ndim_full))  # + initial conditions + std
    free = np.array(free)
    lbf, ubf = lb[free], ub[free]
    nd = len(free)

    def build(thf):
        th = np.empty(ndim_full)
        for i, v in FIX.items():
            th[i] = v
        th[free] = thf
        return th

    def logprob(thf):
        if np.any(thf <= lbf) or np.any(thf >= ubf):
            return -np.inf
        ll = log_likelihood(build(thf), MODEL, full, group)
        return ll if np.isfinite(ll) else -np.inf

    nw = 4 * nd
    p0 = lbf + (ubf - lbf) * np.random.uniform(0.3, 0.7, size=(nw, nd))
    # mu1 (local index 2) starts small
    p0[:, 2] = np.random.uniform(0.0, 1.0, size=nw)
    sampler = emcee.EnsembleSampler(nw, nd, logprob,
                                    moves=[(emcee.moves.StretchMove(), 0.7),
                                           (emcee.moves.DEMove(), 0.3)])
    sampler.run_mcmc(p0, NSTEPS, progress=False)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.nanmax(tau))
    except Exception:
        tau_max = float('nan')
    acc = float(np.mean(sampler.acceptance_fraction))
    burn = NSTEPS // 3
    chain = sampler.get_chain(discard=burn, flat=True)
    ess = (NSTEPS - burn) * nw / tau_max if tau_max > 0 else float('nan')
    mu1 = chain[:, 2]  # local index 2 = mu1
    a1 = 10 ** chain[:, 0]
    med = np.median(mu1); lo, hi = np.percentile(mu1, [2.5, 97.5])
    return dict(group=group, mu1_med=med, mu1_lo=lo, mu1_hi=hi,
                a1_med=np.median(a1), tau=tau_max, ess=ess, acc=acc,
                mu1_samples=mu1)


def main():
    rows = []; samples = {}
    for g in GROUPS:
        r = run_group(g)
        samples[g] = r.pop('mu1_samples')
        rows.append(r)
        print(f"{g}: mu1={r['mu1_med']:.3f} [{r['mu1_lo']:.3f},{r['mu1_hi']:.3f}] "
              f"excludes0={'yes' if r['mu1_lo']>1e-2 else 'no'}  a1={r['a1_med']:.3f}  "
              f"tau={r['tau']:.0f} ESS={r['ess']:.0f} acc={r['acc']:.2f}")
    # figure: posteriors of mu1
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = {'cart_ss_b16': 'tab:red', 'cart_sm_b16': 'tab:orange', 'cart_sl_b16': 'tab:green'}
    for g in GROUPS:
        ax.hist(samples[g], bins=60, density=True, alpha=0.5, color=cols[g],
                label=f"{g} (median {np.median(samples[g]):.2f})")
    ax.set_xlabel('mu1 (killing rate)'); ax.set_ylabel('posterior density')
    ax.set_title('Minimal identifiable submodel: posterior of mu1 by group')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(TMC, 'reduced_mu1.png'), dpi=200,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    with open(os.path.join(TMC, 'reduced_mu1.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys()])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v)
                        for k, v in r.items()})
    print('CSV and figure in', TMC)


if __name__ == '__main__':
    main()
