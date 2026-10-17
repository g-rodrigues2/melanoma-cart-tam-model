"""Log-likelihood, log-prior and log-posterior of the multi-group calibration.

The Gaussian log-likelihood is summed over all groups, with theta partitioned as in `scenarios.py`
and a single shared `std`.
"""
import _path_setup  # adds pipeline_control_parallel/ to sys.path  # noqa: F401

import numpy as np

from models import solve_model
from scenarios import (
    MODEL_PARAM_NAMES,
    count_model_params,
    ic_slice_for_group,
    split_model_args_for_group,
)


def log_likelihood_shared(theta, model_extension, full_data, groups, scenario):
    """Sum of Gaussian log-likelihoods over all groups.

    `std` (theta[-1]) is shared. NaN or an exception in the ODE solution returns -inf, so that emcee
    rejects the proposal.
    """
    std = theta[-1]
    variance = std ** 2
    if not np.isfinite(variance) or variance <= 0:
        return -np.inf

    param_names = MODEL_PARAM_NAMES[model_extension]
    n_model = count_model_params(scenario, groups)

    ll = 0.0
    for group in groups:
        ode_args, _ = split_model_args_for_group(
            theta, scenario, param_names, groups, group,
        )
        T0s = ic_slice_for_group(theta, n_model, full_data, groups, group)

        for idx, mouse_data in enumerate(full_data[group]):
            time_array = mouse_data['data'][:, 0]
            T0 = float(T0s[idx])
            try:
                solution = solve_model(model_extension, time_array, ode_args, T0)
            except Exception:
                return -np.inf

            sol_T = solution[:, 0] if solution.ndim == 2 else solution
            if not np.all(np.isfinite(sol_T)):
                return -np.inf

            observed = mouse_data['data'][:, 1]
            ll += -0.5 * np.sum(
                (sol_T - observed) ** 2 / variance
                + np.log(2 * np.pi) + np.log(variance)
            )

    if not np.isfinite(ll):
        return -np.inf
    return ll


def log_prior_uniform(theta, l_bound, u_bound):
    """Uniform prior on the box [l_bound, u_bound]: 0 inside, -inf outside."""
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def log_probability_shared(theta, l_bound, u_bound, model_extension,
                           full_data, groups, scenario):
    """Unnormalized log-posterior = log_prior + log_likelihood_shared."""
    lp = log_prior_uniform(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_shared(
        theta, model_extension, full_data, groups, scenario,
    )
