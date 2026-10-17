"""Gain in the identifiability of mu1 and beta1 from adding C(t) (Jilani et al. Fig. 6b), from the FIM at the
bounded maximum-likelihood estimate of the long-hinge group.

FIM_T   = tumor volume T(t) only.
FIM_T+C = T(t) and the CAR T-cell persistence C(t_k), k = 5, 13, 27 days (VCN, mean +- SD).
Marginal std of theta_j = sqrt((FIM^-1)_jj) from the eigendecomposition.
"""
import os, csv, numpy as np
import joint_fit_TC as J
from fim_identifiability_map import reoptimize, sensitivity

TMC = J.TMC
theta, lb, ub, labels, mll = reoptimize(J.GROUP)
sigma = theta[-1]
gs = len(J.full[J.GROUP])
nT = 8 + gs  # 8 model parameters and initial conditions (std excluded)

# sensitivity of T(t) at all observations (from fim_identifiability_map)
S_T = sensitivity(theta, J.GROUP, nT)

# scale s (VCN per unit of C) fitted by least squares at the estimate
Cbar = J.C_mean_at(theta, J.C_T)
s = float(np.sum(Cbar*J.C_VCN/J.C_DP**2)/np.sum(Cbar**2/J.C_DP**2))

# sensitivity of Cbar(t_k) to each unknown (8 parameters and initial conditions)
S_C = np.zeros((len(J.C_T), nT))
for j in range(nT):
    h = 1e-5*(1.0+abs(theta[j])); tp = theta.copy(); tm = theta.copy()
    tp[j] += h; tm[j] -= h
    S_C[:, j] = (J.C_mean_at(tp, J.C_T) - J.C_mean_at(tm, J.C_T))/(2*h)

FIM_T = (S_T.T @ S_T)/sigma**2
FIM_C = np.zeros((nT, nT))
for k in range(len(J.C_T)):
    sc = s*S_C[k]                      # d(s*Cbar_k)/dtheta
    FIM_C += np.outer(sc, sc)/J.C_DP[k]**2
FIM_TC = FIM_T + FIM_C


def mstd(FIM, i):
    w, V = np.linalg.eigh(FIM); tol = max(w.max(), 1e-300)*1e-12
    return float(np.sqrt(sum(V[i, k]**2/(w[k] if w[k] > tol else tol)
                             for k in range(len(w)))))


rows = []
print(f"SL MLE ll={mll:.2f}  sigma={sigma:.1f}  VCN scale s={s:.4f}")
print(f"C_model(5,13,27)={np.round(Cbar,3)}  s*C={np.round(s*Cbar,4)}  VCN={J.C_VCN}")
for name, i, islog in [('mu1', 3, False), ('log10_beta1', 2, True)]:
    sT, sTC = mstd(FIM_T, i), mstd(FIM_TC, i)
    fator = sT/sTC if sTC > 0 else np.inf
    print(f"  {name:12s} std(T)={sT:.4g}  std(T+C)={sTC:.4g}  "
          f"reduction={fator:.1f}x  {'(log10)' if islog else ''}")
    rows.append({'param': name, 'std_T': f'{sT:.4g}', 'std_T_plus_C': f'{sTC:.4g}',
                 'reduction_factor': f'{fator:.1f}', 'scale': 'log10' if islog else 'linear'})

with open(os.path.join(TMC, 'fim_joint_TC.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('CSV:', os.path.join(TMC, 'fim_joint_TC.csv'))
