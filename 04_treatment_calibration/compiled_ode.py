"""Compiled ODE solver of the tumor models (default solver of the calibrations; CALIB_SOLVER=odeint restores
scipy's odeint).

`solve(fn, times, parameters, y0, n_states)` integrates dy/dt = fn(y, t, *parameters) in nopython mode with
the explicit Dormand-Prince 5(4) pair and adaptive steps, stopping exactly at every requested time (no
interpolation), and returns an array (len(times), n_states) as odeint. `fn` is a numba function: for one
state it takes and returns a float (the growth and killing laws of models.py); for several states it takes
and returns an array (the compiled T-M and T-M-C systems of models.py).

The step error is controlled componentwise (rtol 1e-9, atol 1e-8). Against odeint with rtol 1e-12, the
volumes agree to a relative error below 1e-7 on posterior samples of the calibrations
(tests/test_compiled_ode.py). When the step budget is exhausted or the solution is not finite, `solve`
returns NaN and the caller falls back to odeint, so the compiled solver never rejects a parameter vector
that odeint can integrate.
"""
import inspect

import numpy as np
from numba import njit

RTOL = 1e-9
ATOL = 1e-8
MAX_STEPS = 200_000

A21 = 1.0 / 5.0
A31, A32 = 3.0 / 40.0, 9.0 / 40.0
A41, A42, A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
A51, A52, A53, A54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
A61, A62, A63, A64, A65 = 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0
B1, B3, B4, B5, B6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0
E1, E3, E4, E5, E6, E7 = (71.0 / 57600.0, -71.0 / 16695.0, 71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0,
                          -1.0 / 40.0)
C2, C3, C4, C5 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0


def _wrap(fn, n_states):
    """Compiled rhs(t, y, p) -> array of a numba model fn(y, t, *scalars)."""
    py = getattr(fn, 'py_func', fn)
    n_par = len(inspect.signature(py).parameters) - 2
    pars = ''.join(f', p[{i}]' for i in range(n_par))
    if n_states == 1:
        body = f'    out = np.empty(1)\n    out[0] = fn(y[0], t{pars})\n    return out'
    else:
        body = f'    return fn(y, t{pars})'
    namespace = {'fn': fn, 'np': np}
    exec('def rhs(t, y, p):\n' + body, namespace)
    return njit(namespace['rhs'])


def _make_solver(rhs):
    @njit
    def solve_compiled(y0, times, p, rtol, atol):
        n, d = times.shape[0], y0.shape[0]
        out = np.empty((n, d))
        y = y0.copy()
        t = times[0]
        out[0] = y
        k = 1
        while k < n and times[k] <= t:
            out[k] = y
            k += 1
        h = 1e-3                       # small first step: an effect starting at t0 may be a narrow pulse
        steps = 0
        k1 = rhs(t, y, p)
        while k < n:
            if steps >= MAX_STEPS:
                out[k:] = np.nan
                return out
            t_end = times[k]
            last = t + h >= t_end
            h_try = t_end - t if last else h
            k2 = rhs(t + C2 * h_try, y + h_try * A21 * k1, p)
            k3 = rhs(t + C3 * h_try, y + h_try * (A31 * k1 + A32 * k2), p)
            k4 = rhs(t + C4 * h_try, y + h_try * (A41 * k1 + A42 * k2 + A43 * k3), p)
            k5 = rhs(t + C5 * h_try, y + h_try * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4), p)
            k6 = rhs(t + h_try, y + h_try * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5), p)
            y_new = y + h_try * (B1 * k1 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6)
            t_new = t_end if last else t + h_try
            k7 = rhs(t_new, y_new, p)
            err = h_try * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)
            ratio = 0.0
            for i in range(d):
                ratio = max(ratio, abs(err[i]) / (atol + rtol * max(abs(y[i]), abs(y_new[i]))))
            steps += 1
            if not np.isfinite(ratio):
                h = 0.2 * h_try
                if h < 1e-14:
                    out[k:] = np.nan
                    return out
                continue
            if ratio <= 1.0:
                t = t_new
                y = y_new
                k1 = k7
                factor = 10.0 if ratio == 0.0 else min(10.0, 0.9 * ratio ** -0.2)
                if not last or h_try >= h:
                    h = h_try * factor
                while k < n and times[k] <= t:
                    out[k] = y
                    k += 1
            else:
                h = h_try * max(0.2, 0.9 * ratio ** -0.2)
                if h < 1e-14:
                    out[k:] = np.nan
                    return out
        return out

    return solve_compiled


_SOLVERS = {}


def solve(fn, times, parameters, y0, n_states=1, rtol=RTOL, atol=ATOL):
    """Solution at `times` (times[0] is the initial time), shape (len(times), n_states)."""
    key = (fn, n_states)
    if key not in _SOLVERS:
        _SOLVERS[key] = _make_solver(_wrap(fn, n_states))
    return _SOLVERS[key](np.asarray(y0, dtype=float).reshape(n_states), np.ascontiguousarray(times, dtype=float),
                         np.asarray(parameters, dtype=float), rtol, atol)
