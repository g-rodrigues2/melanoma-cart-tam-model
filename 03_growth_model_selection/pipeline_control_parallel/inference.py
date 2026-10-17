"""Log-prior, log-likelihood and log-probability for the MCMC."""
import numpy as np

from models import _split_theta, solve_model


def log_likelihood(theta, model_extension, full_data, group):
    """Sum of Gaussian log-likelihoods over all mice of the group.

    If the ODE returns NaN or raises (pathological parameters, e.g. _bert with a dominant b), -inf is
    returned, so that emcee rejects the proposal without stopping the sampler.
    """
    ll = 0.0
    variance = theta[-1] ** 2
    ndim = len(full_data[group])
    ode_args, ic_fn = _split_theta(theta, model_extension, ndim)

    for idx, mouse_data in enumerate(full_data[group]):
        time_array = mouse_data['data'][:, 0]
        initial_condition = ic_fn(idx)
        try:
            solution = solve_model(model_extension, time_array, ode_args,
                                   initial_condition)
        except Exception:
            return -np.inf

        sol_T = solution[:, 0]
        if not np.all(np.isfinite(sol_T)):
            return -np.inf

        observed_volume = mouse_data['data'][:, 1]
        ll += -0.5 * np.sum(
            (sol_T - observed_volume) ** 2 / variance
            + np.log(2 * np.pi) + np.log(variance)
        )

    if not np.isfinite(ll):
        return -np.inf
    return ll


def log_prior(theta, l_bound, u_bound):
    """Uniform prior on the box [l_bound, u_bound]: 0 inside, -inf outside."""
    for l, p, u in zip(l_bound, theta, u_bound):
        if not (l < p < u):
            return -np.inf
    return 0.0


def log_probability(theta, l_bound, u_bound, model_extension, full_data, group):
    """Unnormalized log-posterior = log_prior + log_likelihood."""
    lp = log_prior(theta, l_bound, u_bound)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model_extension, full_data, group)
