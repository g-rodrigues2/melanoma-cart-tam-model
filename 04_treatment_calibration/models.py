"""Treatment models of the CAR T-cell groups (18 models).

  16 models based on _exp (r = R_B16_FIXED, fixed from the growth-model selection):
      dT/dt = r·T - C(T,t)·T  -- no treatment term and 15 empirical killing terms

  _mac (as in the growth-model selection):
      dT = alpha1·T·(1-T/K1)·(1+beta1·M)
      dM = alpha2·T·M·(1-M/K2) - kappa2·M
      Calibrated: alpha1, K1, beta1. Fixed: alpha2, K2, kappa2.

  _mac_treat (full T, M, C system of the model):
      dT = alpha1·T·(1-T/K1)·(1+beta1·M) - mu1·T·C
      dM = alpha2·T·M·(1-M/K2) - kappa2·M               (fixed)
      dC = (alpha3·T·C)/(K3+T) - mu3·C·T - kappa3·C
      Calibrated: alpha1, K1, beta1, mu1, alpha3, K3, kappa3, mu3 (8 parameters).

Layout of theta:
  _exp:                  [T0_1..T0_n, std]
  _exp_<termo>:          [pars_treat, T0_1..T0_n, std]
  _mac:                  [alpha1, K1, beta1, T0_1..T0_n, std]
  _mac_treat:            [alpha1, K1, beta1, mu1, alpha3, K3, kappa3, mu3,
                          T0_1..T0_n, std]
"""
import os

import numpy as np
from numba import jit
from scipy.integrate import odeint, solve_ivp

import compiled_ode


# 'compiled' (default): compiled_ode.py; 'odeint' or 'solve_ivp': scipy
SOLVER        = os.environ.get('CALIB_SOLVER', 'compiled').lower()
SOLVER_METHOD = os.environ.get('CALIB_SOLVER_METHOD', 'LSODA')
SOLVER_RTOL   = float(os.environ.get('CALIB_SOLVER_RTOL', '1e-6'))
SOLVER_ATOL   = float(os.environ.get('CALIB_SOLVER_ATOL', '1e-8'))


R_B16_FIXED = 0.2648

# Fixed macrophage constants (rescaled with rho = 10^6 cells/mm^3)
MAC_ALPHA2 = 0.1
MAC_K2     = 1000.0
MAC_KAPPA2 = 0.34
MAC_M0_FRACTION = 0.5
CART_C0 = 1.0


# === _exp without treatment term and 15 killing terms =============
@jit(nopython=True)
def model_exp(y, t):
    return R_B16_FIXED * y

@jit(nopython=True)
def model_exp_ED(y, t, a, b):
    return R_B16_FIXED * y - (a * np.exp(-b * t) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_CD(y, t, a, b):
    return R_B16_FIXED * y - (a * 1.0 * np.exp(-b * t) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_AccelED(y, t, a, b):
    return R_B16_FIXED * y - (a * np.exp(-b * t * t) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_AccelCD(y, t, a, b):
    return R_B16_FIXED * y - (a * 1.0 * np.exp(-b * t * t) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_MM(y, t, a, b):
    return R_B16_FIXED * y - (a * 1.0 / (1.0 + b * 1.0) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_LS(y, t, a, b):
    return R_B16_FIXED * y - (a * 1.0 / (1.0 + b * y) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_PF(y, t, c, lam):
    return R_B16_FIXED * y - (c * (1.0 ** lam) if t >= 0.0 else 0.0) * y

@jit(nopython=True)
def model_exp_dePR(y, t, d, s, lam):
    if t >= 0.0:
        tumor = y if y > 1e-12 else 1e-12
        Tlam = tumor ** lam
        treat = d * (1.0 / (s * Tlam + 1.0))
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_dePRDecay(y, t, d, s, lam, k_elim):
    if t >= 0.0:
        tumor = y if y > 1e-12 else 1e-12
        Tlam = tumor ** lam
        effect_step = d * (1.0 / (s * Tlam + 1.0))
        treat = effect_step * np.exp(-k_elim * t) if t > 0.0 else effect_step
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_CDwithLog(y, t, a, b, steepness):
    if t >= 0.0:
        factor = 1.0 / (1.0 + np.exp(-steepness * t))
        treat = a * factor * np.exp(-b * t)
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_CDwithExp(y, t, a, b, rate):
    if t >= 0.0:
        factor = 1.0 - np.exp(-rate * t)
        treat = a * factor * np.exp(-b * t)
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_CDwithSin(y, t, a, b, frequency):
    if t >= 0.0:
        factor = 0.5 * (1.0 + np.sin(frequency * t))
        treat = a * factor * np.exp(-b * t)
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_EDwithLog(y, t, a, b, steepness):
    if t >= 0.0:
        factor = 1.0 / (1.0 + np.exp(-steepness * t))
        treat = a * factor * np.exp(-b * t)
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_EDwithExp(y, t, a, b, rate):
    if t >= 0.0:
        factor = 1.0 - np.exp(-rate * t)
        treat = a * factor * np.exp(-b * t)
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y

@jit(nopython=True)
def model_exp_EDwithSin(y, t, a, b, frequency):
    if t >= 0.0:
        factor = 0.5 * (1.0 + np.sin(frequency * t))
        treat = a * factor * np.exp(-b * t)
    else:
        treat = 0.0
    return R_B16_FIXED * y - treat * y


# === _mac (2D, no CAR T-cells) ====================================
def model_mac(y, t, alpha1, K1, beta1):
    """alpha1, K1, beta1 calibrated; alpha2, K2, kappa2 fixed."""
    T, M = y
    dT = alpha1 * T * (1.0 - T / K1) * (1.0 + beta1 * M)
    dM = MAC_ALPHA2 * T * M * (1.0 - M / MAC_K2) - MAC_KAPPA2 * M
    return [dT, dM]


# === _mac_treat (full 3D system) =================================
def model_mac_treat(y, t, alpha1, K1, beta1, mu1, alpha3, K3, kappa3, mu3):
    """Full (T, M, C) system:
        dT = alpha1·T·(1-T/K1)·(1+beta1·M) - mu1·T·C
        dM = alpha2·T·M·(1-M/K2) - kappa2·M            (alpha2, K2, kappa2 fixed)
        dC = (alpha3·T·C)/(K3+T) - mu3·C·T - kappa3·C
    C(0) = CART_C0 = 1.0 (normalized dose, absorbed by mu1 and mu3).
    """
    T, M, C = y
    dT = alpha1 * T * (1.0 - T / K1) * (1.0 + beta1 * M) - mu1 * T * C
    dM = MAC_ALPHA2 * T * M * (1.0 - M / MAC_K2) - MAC_KAPPA2 * M
    if T + K3 > 0:
        dC = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    else:
        dC = -mu3 * C * T - kappa3 * C
    return [dT, dM, dC]


# Compiled versions of the multistate systems (same equations, array state), used by compiled_ode
@jit(nopython=True)
def _mac_compiled(y, t, alpha1, K1, beta1):
    out = np.empty(2)
    out[0] = alpha1 * y[0] * (1.0 - y[0] / K1) * (1.0 + beta1 * y[1])
    out[1] = MAC_ALPHA2 * y[0] * y[1] * (1.0 - y[1] / MAC_K2) - MAC_KAPPA2 * y[1]
    return out


@jit(nopython=True)
def _mac_treat_compiled(y, t, alpha1, K1, beta1, mu1, alpha3, K3, kappa3, mu3):
    T, M, C = y[0], y[1], y[2]
    out = np.empty(3)
    out[0] = alpha1 * T * (1.0 - T / K1) * (1.0 + beta1 * M) - mu1 * T * C
    out[1] = MAC_ALPHA2 * T * M * (1.0 - M / MAC_K2) - MAC_KAPPA2 * M
    if T + K3 > 0:
        out[2] = (alpha3 * T * C) / (K3 + T) - mu3 * C * T - kappa3 * C
    else:
        out[2] = -mu3 * C * T - kappa3 * C
    return out


_COMPILED_RHS = {'_mac': _mac_compiled, '_mac_treat': _mac_treat_compiled}


# === Registry ===
EXP_BASELINE_MODELS  = {'_exp'}
EXP_TREATMENT_MODELS = {
    '_exp_ED', '_exp_CD', '_exp_AccelED', '_exp_AccelCD',
    '_exp_MM', '_exp_LS', '_exp_PF',
    '_exp_dePR', '_exp_dePRDecay',
    '_exp_CDwithLog', '_exp_CDwithExp', '_exp_CDwithSin',
    '_exp_EDwithLog', '_exp_EDwithExp', '_exp_EDwithSin',
}
MAC_2D_MODELS = {'_mac'}
MAC_3D_MODELS = {'_mac_treat'}
MULTISTATE_MODELS = MAC_2D_MODELS | MAC_3D_MODELS
ALL_MODELS = (EXP_BASELINE_MODELS | EXP_TREATMENT_MODELS
              | MAC_2D_MODELS | MAC_3D_MODELS)


def _split_theta(theta, model_extension, ndim):
    """Splits theta into (ode_args, ic_fn).

    For _mac and _mac_treat the first three entries (alpha1, K1, beta1) are sampled in log10 (see bounds.py)
    and are converted here to physical values; this is the only place that knows the reparameterization
    (inference.py, prior_predictive.py, gen_maxll_plots.py, ... call this function).
    """
    tail = ndim + 1
    ode_args = tuple(theta[:-tail])
    T0s = theta[-tail:-1]

    if model_extension in MAC_2D_MODELS or model_extension in MAC_3D_MODELS:
        physical = list(ode_args)
        physical[0] = 10.0 ** physical[0]  # alpha1
        physical[1] = 10.0 ** physical[1]  # K1
        physical[2] = 10.0 ** physical[2]  # beta1
        ode_args = tuple(physical)

    if model_extension in MAC_3D_MODELS:
        def ic_fn(idx):
            t0 = float(T0s[idx])
            return [t0, MAC_M0_FRACTION * t0, CART_C0]
    elif model_extension in MAC_2D_MODELS:
        def ic_fn(idx):
            t0 = float(T0s[idx])
            return [t0, MAC_M0_FRACTION * t0]
    else:
        def ic_fn(idx):
            return float(T0s[idx])
    return ode_args, ic_fn


def _integrate(rhs, t_array, y0, parameters):
    if SOLVER == 'solve_ivp':
        def f_adapter(t, y, *args):
            return rhs(y, t, *args)
        t_span = (float(t_array[0]), float(t_array[-1]))
        sol = solve_ivp(f_adapter, t_span, y0, t_eval=t_array,
                        args=parameters, method=SOLVER_METHOD,
                        rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
                        dense_output=False)
        if sol.status < 0:
            return np.full((len(t_array), len(y0)), np.nan)
        return sol.y.T
    return odeint(rhs, t=t_array, y0=y0, args=parameters, mxstep=2000)


def solve_model(model_extension, time_array, parameters, initial_condition,
                type_sol='data'):
    model_func = globals()['model' + model_extension]
    rhs = getattr(model_func, 'py_func', model_func)

    if np.isscalar(initial_condition):
        y0 = [float(initial_condition)]
    else:
        y0 = list(initial_condition)

    if type_sol == 'smooth':
        bgn_p = round(time_array[0], 1)
        end_p = round(time_array[-1], 1)
        time_array = np.linspace(bgn_p, end_p, int((end_p - bgn_p) / 0.1) + 1)

    if SOLVER == 'compiled':
        fn = _COMPILED_RHS.get(model_extension, model_func)
        sol = compiled_ode.solve(fn, time_array, parameters, y0, len(y0))
        if not np.all(np.isfinite(sol)):
            sol = odeint(rhs, t=time_array, y0=y0, args=parameters, mxstep=2000)   # fallback
    else:
        sol = _integrate(rhs, time_array, y0, parameters)

    if type_sol == 'smooth':
        return np.column_stack((time_array, sol))
    return sol
