"""Bounded maximum-likelihood fitting of the mechanistic models by nonlinear least squares.

The Gaussian log-likelihood of inference.py has a single residual standard deviation sigma shared by all
observations of a group. For fixed ODE parameters and initial volumes it is maximized by sigma^2 = RSS/N, so the
concentrated (profile-in-sigma) log-likelihood is

    ll_c(theta) = -N/2 * (log(2*pi*RSS(theta)/N) + 1),

and maximizing it is equivalent to minimizing the residual sum of squares. The residual vector is minimized with
the bounded trust-region reflective algorithm (scipy.optimize.least_squares, method 'trf'), which uses a
Gauss-Newton model of the curvature. This matters for these models: in the calibrated escape regime the growth
and killing terms nearly cancel, so the likelihood is concentrated on a narrow, curved ridge: moving mu1 alone by 0.01 from the
long-hinge estimate, with the remaining parameters held fixed, costs more than ten log-likelihood units,
whereas the profile interval of mu1, which re-estimates the other parameters at each value, spans
0.212 to 0.589. (Checked on 2026-10-16; the figure of 100 units quoted here before did not reproduce.) Quasi-Newton methods
applied to the scalar log-likelihood stall on this ridge; the Gauss-Newton model follows it.

The free parameters are rescaled to the unit interval, so finite-difference steps are comparable across
parameters whose natural scales differ by several orders of magnitude.

The concentrated log-likelihood equals log_likelihood(theta) of inference.py with theta[-1] = sqrt(RSS/N).
"""
import numpy as np
from scipy.optimize import least_squares

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from models import _split_theta, solve_model

FULL = load_treatment_data()
MAX_IC = find_max_initial_condition(FULL)[0]
FAIL_RESIDUAL = 1e4          # residual assigned to every observation when the ODE solution is not finite
LSQ_OPTIONS = dict(method='trf', x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-10, diff_step=1e-6)


def bounds(model, group):
    lb, ub, labels = define_bounds_labels(model, group_size=len(FULL[group]), max_ic=MAX_IC,
                                          bool_multiple_mice=True)
    return np.asarray(lb, float), np.asarray(ub, float), labels


def n_obs(group):
    return sum(len(m['data']) for m in FULL[group])


def residuals(theta, model, group):
    ode_args, ic_fn = _split_theta(theta, model, len(FULL[group]))
    out = []
    for i, mouse in enumerate(FULL[group]):
        obs = mouse['data']
        try:
            sol = solve_model(model, obs[:, 0], ode_args, ic_fn(i))[:, 0]
        except Exception:
            sol = None
        if sol is None or not np.all(np.isfinite(sol)):
            out.append(np.full(len(obs), FAIL_RESIDUAL))
        else:
            out.append(sol - obs[:, 1])
    return np.concatenate(out)


def concentrated_ll(res):
    n = len(res)
    return -0.5 * n * (np.log(2 * np.pi * np.sum(res ** 2) / n) + 1.0)


class Problem:
    """Least-squares problem of one model and group, with an optional set of fixed parameters {index: value}.

    Parameter vectors passed in and returned are full theta vectors (model parameters, initial volumes, sigma).
    """

    def __init__(self, model, group, fixed=None):
        self.model, self.group = model, group
        self.lb, self.ub, self.labels = bounds(model, group)
        self.fixed = dict(fixed or {})
        n_ode_ic = len(self.lb) - 1                                   # sigma (last entry) is concentrated out
        self.free = np.array([k for k in range(n_ode_ic) if k not in self.fixed])
        self.span = self.ub[self.free] - self.lb[self.free]

    def theta(self, u):
        th = np.empty(len(self.lb))
        for k, v in self.fixed.items():
            th[k] = v
        th[self.free] = self.lb[self.free] + self.span * np.clip(u, 0.0, 1.0)
        th[-1] = 1.0
        th[-1] = np.sqrt(np.mean(residuals(th, self.model, self.group) ** 2))
        return th

    def to_unit(self, theta):
        return np.clip((np.asarray(theta, float)[self.free] - self.lb[self.free]) / self.span, 0.0, 1.0)

    def _fun(self, u):
        th = np.empty(len(self.lb))
        for k, v in self.fixed.items():
            th[k] = v
        th[self.free] = self.lb[self.free] + self.span * u
        th[-1] = 1.0
        return residuals(th, self.model, self.group)

    def loglik(self, theta):
        th = np.asarray(theta, float).copy()
        for k, v in self.fixed.items():
            th[k] = v
        return concentrated_ll(residuals(th, self.model, self.group))

    def fit(self, start, max_nfev=300):
        """Bounded least squares from one full theta vector; returns (ll, theta)."""
        u0 = self.to_unit(start)
        try:
            r = least_squares(self._fun, u0, bounds=(0.0, 1.0), max_nfev=max_nfev, **LSQ_OPTIONS)
            u = np.clip(r.x, 0.0, 1.0)
        except Exception:
            u = u0
        th = self.theta(u)
        return self.loglik(th), th

    def fit_many(self, starts, max_nfev=300):
        best = (-np.inf, None)
        for s in starts:
            ll, th = self.fit(s, max_nfev)
            if ll > best[0]:
                best = (ll, th)
        return best
