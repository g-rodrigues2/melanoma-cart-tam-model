"""Tumor growth models and ODE solver dispatch.

Available models (suffix passed as `model_extension`):
  _exp, _mendel, _bert, _lin, _gomp, _surf, _log  -- classical one-dimensional models
  _mac                                            -- (T, M) with macrophages

Each model_*(y, t, ...) is the right-hand side of the ODE; `solve_model()` integrates it with the compiled
solver of compiled_ode.py (default), ``odeint`` or ``solve_ivp``, selected by ``CALIB_SOLVER``.

Layout of theta expected by _split_theta:
  - one-dimensional models:
      [par1, par2, ..., T0_1, ..., T0_ndim, std]
  - _mac:
      [alpha1, K1, beta1, T0_1, ..., T0_ndim, std]
"""
import os

import numpy as np
from numba import jit
from scipy.integrate import odeint, solve_ivp

import compiled_ode


# === ODE solver settings (read at import, inherited by spawned workers) ===
SOLVER        = os.environ.get('CALIB_SOLVER', 'compiled').lower()
SOLVER_METHOD = os.environ.get('CALIB_SOLVER_METHOD', 'LSODA')
SOLVER_RTOL   = float(os.environ.get('CALIB_SOLVER_RTOL', '1e-6'))
SOLVER_ATOL   = float(os.environ.get('CALIB_SOLVER_ATOL', '1e-8'))


# === One-dimensional models ===

@jit(nopython=True)
def model_exp(y, t, growth_rate):
    tumor = y
    return growth_rate * tumor


@jit(nopython=True)
def model_bert(y, t, growth_rate, b):
    tumor = y
    return growth_rate * (tumor ** (2./3.)) - b * tumor


@jit(nopython=True)
def model_mendel(y, t, growth_rate, power):
    tumor = y
    return growth_rate * (tumor ** power)


@jit(nopython=True)
def model_log(y, t, growth_rate, carrying_capacity):
    tumor = y
    return growth_rate * tumor * (1 - tumor / carrying_capacity)


@jit(nopython=True)
def model_lin(y, t, growth_rate, carrying_capacity):
    tumor = y
    return growth_rate * tumor / (tumor + carrying_capacity)


@jit(nopython=True)
def model_gomp(y, t, growth_rate, b, c):
    tumor = y
    return growth_rate * tumor * np.log(b / (tumor + c))


@jit(nopython=True)
def model_surf(y, t, growth_rate, carrying_capacity):
    tumor = y
    return growth_rate * tumor / ((tumor + carrying_capacity) ** (1./3.))


# === Macrophage model (T, M) ===
# Full (T, M, C) system with TAM-mediated immunosuppression and CAR T-cells:
#   dT = alpha1 * T * (1 - T/K1) * (1 + beta1 * M) - mu1 * T * C
#   dM = alpha2 * T * M * (1 - M/K2) - kappa2 * M
#   dC = (alpha3 * T * C) / (K3 + T) - kappa3 * C
# For the PBS controls C(0) = 0, so C(t) = 0 and the C equation is not integrated.
#
# ============================================================================
# Rescaling of the constants of the M compartment (cells -> mm^3)
# ============================================================================
# The original values (alpha2 = 1e-7 /cell/day, K2 = 1e9 cells, kappa2 = 0.34 /day) assume T and M in
# cell counts, whereas the data are tumor volumes in mm^3. The conversion uses the tumor cell density,
# typically in mathematical oncology
#
#     rho ~ 10^9 cells/cm^3 = 10^6 cells/mm^3
#
# Sources (classical values in T-M-C models):
#   - Folkman, J. (1971, 1989): tumor cell density ~ 10^9 cells/cm^3
#   - Anderson, A. & Chaplain, M. (Bull Math Biol 1998): ~10^6 cells/mm^3
#   - Mantzaris, N. et al. (2004): rho = 10^9 cells/cm^3
#   - de Pillis, L. et al. (Cancer Res 2005): a tumor of 10^9 cells has a volume of ~1 cm^3 = 1000 mm^3
#
# Since the equations (e.g. alpha2*T*M) give [alpha2] = 1/([T]*time):
#   - T in cells: alpha2 in 1/cell/day
#   - T in mm^3:  alpha2 in 1/mm^3/day = (1/cell/day) * (cells/mm^3)
#
# Applying rho = 10^6 cells/mm^3 to each constant:
#   alpha2: 1e-7  /cell/day  *  10^6 cells/mm^3 =  0.1   /mm^3/day
#   K2:     1e9   cells       /  10^6 cells/mm^3 =  1000  mm^3
#   kappa2: 0.34  /day        (intensive rate, invariant under rescaling)
#
# With these values alpha2*T > kappa2 for T > 3.4 mm^3 (M grows). The tumors of the data lie between about
# 50 and 3000 mm^3, so M has non-trivial dynamics: it responds to the tumor and modulates its growth through
# beta1.
#
# beta1 is rescaled as well: it multiplies M in mm^3 instead of cells. Literature value (Jilani 2024 and
# similar): beta1 ~ 2.3e-10 /cell -> 2.3e-4 /mm^3. The bounds in bounds.py follow this scale: beta1 in
# [1e-6, 1e-2] /mm^3.
# ============================================================================
MAC_ALPHA2 = 0.1     # /mm^3/day  (= 1e-7 /cell/day x 10^6 cells/mm^3)
MAC_K2     = 1000.0  # mm^3       (= 1e9 cells / 10^6 cells/mm^3)
MAC_KAPPA2 = 0.34    # /day       (intensive rate, invariant)
# M(0) = MAC_M0_FRACTION * T(0) for each mouse (TAMs ~50% of the microenvironment)
MAC_M0_FRACTION = 0.5

# Constants of the CAR T-cell compartment (used by the treatment stage)
CART_MU1    = 1e-7
CART_ALPHA3 = 0.9
CART_K3     = 2e7
CART_KAPPA3 = 0.35


def model_mac(y, t, alpha1, K1, beta1):
    """Reduced (T, M) system of the controls (C = 0), with the constants of M rescaled to the scale of the
    data (mm^3). The calibrated parameters (alpha1, K1, beta1) keep their dimensional meaning.
    """
    T, M = y
    dT = alpha1 * T * (1.0 - T / K1) * (1.0 + beta1 * M)
    dM = MAC_ALPHA2 * T * M * (1.0 - M / MAC_K2) - MAC_KAPPA2 * M
    return [dT, dM]


# === AD version: macrophage constants rescaled to mm^3 ===
# Same structure as _mac, with constants of the M compartment consistent with the scale of the data
# (mm^3 instead of cells), assuming ~10^6 cells/mm^3:
#   alpha2 = 1e-7 /cell/day -> 0.1 /mm^3/day
#   K2     = 1e9 cells      -> 1000 mm^3
#   kappa2 = 0.34 /day      -> 0.34 /day (intensive rate)
#
# Without the rescaling alpha2*T (in mm^3) is always far below kappa2, M decays to zero in about 3 days and
# the model reduces to the logistic model.
#
# The calibrated parameters are those of _mac (alpha1, K1, beta1); only the unit of beta1 changes
# (1/mm^3 instead of 1/cell), and the bounds follow (see bounds.py).
#
# Dimensionless form (u = T/K1, v = M/K2, tau = alpha1 t):
#   alpha_tilde = MAC_ALPHA2_AD * K1 / alpha1
#   beta_tilde  = beta1 * MAC_K2_AD
#   kappa_tilde = MAC_KAPPA2_AD / alpha1
# (macAD_to_dimensionless in postprocess.py computes them from a chain.)
MAC_ALPHA2_AD       = 0.1     # /mm^3/day
MAC_K2_AD           = 1000.0  # mm^3
MAC_KAPPA2_AD       = 0.34    # /day
MAC_M0_FRACTION_AD  = 0.5


def model_macAD(y, t, alpha1, K1, beta1):
    """(T, M) system with the constants of M rescaled to mm^3 (AD version)."""
    T, M = y
    dT = alpha1 * T * (1.0 - T / K1) * (1.0 + beta1 * M)
    dM = MAC_ALPHA2_AD * T * M * (1.0 - M / MAC_K2_AD) - MAC_KAPPA2_AD * M
    return [dT, dM]


@jit(nopython=True)
def _mac_compiled(y, t, alpha1, K1, beta1):
    """model_mac with an array state (compiled solver)."""
    out = np.empty(2)
    out[0] = alpha1 * y[0] * (1.0 - y[0] / K1) * (1.0 + beta1 * y[1])
    out[1] = MAC_ALPHA2 * y[0] * y[1] * (1.0 - y[1] / MAC_K2) - MAC_KAPPA2 * y[1]
    return out


@jit(nopython=True)
def _macAD_compiled(y, t, alpha1, K1, beta1):
    """model_macAD with an array state (compiled solver)."""
    out = np.empty(2)
    out[0] = alpha1 * y[0] * (1.0 - y[0] / K1) * (1.0 + beta1 * y[1])
    out[1] = MAC_ALPHA2_AD * y[0] * y[1] * (1.0 - y[1] / MAC_K2_AD) - MAC_KAPPA2_AD * y[1]
    return out


_COMPILED_RHS = {'_mac': _mac_compiled, '_macAD': _macAD_compiled}


# Models with a multidimensional state
MULTISTATE_MODELS = {'_mac', '_macAD'}


def _split_theta(theta, model_extension, ndim):
    """Splits theta into (ode_args, ic_for_mouse(idx)).

    _mac/_macAD: theta = [alpha1, K1, beta1, T0_1..T0_ndim, std];
                 ic_for_mouse(idx) = [T0[idx], M0_FRAC * T0[idx]].
    1D models:   theta = [par1..parN, T0_1..T0_ndim, std];
                 ic_for_mouse(idx) = T0[idx] (scalar).
    """
    tail = ndim + 1  # initial conditions + std
    if model_extension in ('_mac', '_macAD'):
        alpha1, K1, beta1 = theta[0], theta[1], theta[2]
        ode_args = (alpha1, K1, beta1)
        T0s = theta[-tail:-1]
        # _mac uses MAC_M0_FRACTION and _macAD MAC_M0_FRACTION_AD (same value, kept separate)
        m0_frac = MAC_M0_FRACTION_AD if model_extension == '_macAD' else MAC_M0_FRACTION
        def ic_fn(idx):
            t0 = T0s[idx]
            return [t0, m0_frac * t0]
        return ode_args, ic_fn
    else:
        ode_args = tuple(theta[:-tail])
        T0s = theta[-tail:-1]
        def ic_fn(idx):
            return T0s[idx]
        return ode_args, ic_fn


# === Integration ===

def _integrate(rhs, t_array, y0, parameters):
    """Dispatches to odeint (default, faster) or solve_ivp."""
    if SOLVER == 'solve_ivp':
        def f_adapter(t, y, *args):
            return rhs(y, t, *args)
        t_span = (float(t_array[0]), float(t_array[-1]))
        sol = solve_ivp(
            f_adapter, t_span, y0,
            t_eval=t_array, args=parameters,
            method=SOLVER_METHOD, rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
            dense_output=False,
        )
        if sol.status < 0:
            return np.full((len(t_array), len(y0)), np.nan)
        return sol.y.T
    return odeint(rhs, t=t_array, y0=y0, args=parameters, mxstep=2000)


def solve_model(model_extension, time_array, parameters, initial_condition,
                type_sol='data'):
    """Integrates the model identified by model_extension from time_array[0] to time_array[-1].
    ``initial_condition`` is a scalar (1D models) or a list (multistate models). type_sol='smooth' returns
    the solution on a 0.1-day grid (for plots)."""
    model_func = globals()['model' + model_extension]
    rhs = getattr(model_func, 'py_func', model_func)  # pure Python function of the @jit model

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
