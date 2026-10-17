"""Gain in the identifiability of beta1 and mu1 from adding observations of M(t) (TAMs) and/or C(t) (CAR T-cells),
from the FIM at the bounded maximum-likelihood estimate of the long-hinge group.

The Fisher information depends on the model sensitivities and on the noise level, not on measured values: the
literature (TAMs measurable in subcutaneous C57BL/6 models, ~10-40% of the infiltrate, CV ~30%, days 7-21; VCN of
Jilani et al. Fig. 6b, days 5, 13, 27) sets the sampling design and the noise.

Compares the marginal std of log10(beta1) and mu1 for T | T+M | T+C | T+M+C.
"""
import os, csv, numpy as np
import joint_fit_TC as J
from fim_identifiability_map import reoptimize, sensitivity

TMC = J.TMC
SIGMA_REL = 0.30                      # relative noise (M and C)
M_DAYS = np.array([7.0, 14.0, 21.0])  # typical TAM sampling days
C_DAYS = J.C_T                        # 5, 13, 27 (Jilani et al. Fig. 6b)
IB1, IMU1 = 2, 3

theta, lb, ub, labels, mll = reoptimize(J.GROUP)
sigma = theta[-1]
gs = len(J.full[J.GROUP])
nT = 8 + gs

# --- sensitivity of T(t) at the observations ---
S_T = sensitivity(theta, J.GROUP, nT)

# --- M(t) and C(t) averaged over the mice ---
from models import solve_model, _split_theta
def state_mean_at(th, days, comp):
    ode_args, ic_fn = _split_theta(th, J.MODEL, gs)
    acc = np.zeros(len(days))
    for idx in range(gs):
        t0 = J.full[J.GROUP][idx]['data'][0, 0]
        sol = solve_model(J.MODEL, np.concatenate(([t0], days)), ode_args, ic_fn(idx))
        acc += sol[1:, comp]
    return acc / gs

def sens_state(days, comp):
    base = state_mean_at(theta, days, comp)
    S = np.zeros((len(days), nT))
    for j in range(nT):
        h = 1e-5*(1+abs(theta[j])); tp = theta.copy(); tm = theta.copy()
        tp[j] += h; tm[j] -= h
        S[:, j] = (state_mean_at(tp, days, comp) - state_mean_at(tm, days, comp))/(2*h)
    return base, S

Mbase, S_M = sens_state(M_DAYS, 1)   # comp 1 = M
Cbase, S_C = sens_state(C_DAYS, 2)   # comp 2 = C

FIM_T = (S_T.T @ S_T)/sigma**2
def fim_state(base, S):
    F = np.zeros((nT, nT))
    for k in range(len(base)):
        sig = max(SIGMA_REL*abs(base[k]), 1e-9)
        F += np.outer(S[k], S[k])/sig**2
    return F
FIM_M = fim_state(Mbase, S_M)
FIM_C = fim_state(Cbase, S_C)

def mstd(F, i):
    w, V = np.linalg.eigh(F); tol = max(w.max(), 1e-300)*1e-12
    return float(np.sqrt(sum(V[i, k]**2/(w[k] if w[k] > tol else tol) for k in range(len(w)))))

designs = {'T': FIM_T, 'T+M': FIM_T+FIM_M, 'T+C': FIM_T+FIM_C, 'T+M+C': FIM_T+FIM_M+FIM_C}
print(f"SL MLE ll={mll:.2f}  M(7,14,21)={np.round(Mbase,1)}  C(5,13,27)={np.round(Cbase,3)}")
rows = []
for name, F in designs.items():
    sb1, smu1 = mstd(F, IB1), mstd(F, IMU1)
    print(f"  {name:6s}  std(log10 b1)={sb1:8.3g}   std(mu1)={smu1:8.3g}")
    rows.append({'design': name, 'std_log10_beta1': f'{sb1:.4g}', 'std_mu1': f'{smu1:.4g}'})
# reduction factors relative to T
b0 = mstd(FIM_T, IB1); m0 = mstd(FIM_T, IMU1)
print("  reduction relative to T:  beta1:",
      {n: round(b0/mstd(F, IB1), 1) for n, F in designs.items()},
      " mu1:", {n: round(m0/mstd(F, IMU1), 1) for n, F in designs.items()})
with open(os.path.join(TMC, 'fim_joint_TM.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('CSV:', os.path.join(TMC, 'fim_joint_TM.csv'))
