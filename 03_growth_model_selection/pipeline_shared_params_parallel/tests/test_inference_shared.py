"""Tests of the multi-group log-likelihood:
  - with a large std the log-likelihood is finite for parameters inside the bounds;
  - with a small std the true parameters of the synthetic data give a larger log-likelihood than wrong ones;
  - log_prior returns -inf outside the bounds.
"""
import numpy as np

from inference_shared import (
    log_likelihood_shared, log_prior_uniform, log_probability_shared,
)
from scenarios import SCENARIOS


GROUPS = ['control_b16', 'control_pdx']


def _theta_log_s1(r_b16, cc_b16, r_pdx, cc_pdx, T0s, std):
    return np.array([r_b16, r_pdx, cc_b16, cc_pdx] + list(T0s) + [std])


def _theta_exp_s4(r, T0s, std):
    return np.array([r] + list(T0s) + [std])


def test_log_lik_finite_inside_bounds(synthetic_two_groups):
    """Reasonable parameters and a large std give a finite log-likelihood."""
    theta = _theta_exp_s4(0.10, [100.0, 50.0, 80.0, 120.0], 50.0)
    ll = log_likelihood_shared(
        theta, '_exp', synthetic_two_groups, GROUPS, SCENARIOS['_exp'][4],
    )
    assert np.isfinite(ll)


def test_log_lik_best_at_true_params_exp_s1(synthetic_two_groups):
    """Scenario 1 of _exp with the rates of the fixture (r_b16 = 0.1, r_pdx = 0.2) gives a larger
    log-likelihood than a common wrong rate."""
    sdef1 = SCENARIOS['_exp'][1]

    # scenario 1: rate per group, true values
    theta_true = np.array([0.10, 0.20,         # r_b16, r_pdx
                           100., 50.,           # T0 b16
                           80., 120.,           # T0 pdx
                           1.0])                # small std
    ll_true = log_likelihood_shared(
        theta_true, '_exp', synthetic_two_groups, GROUPS, sdef1,
    )

    # scenario 1: same layout, wrong common rate
    theta_wrong = np.array([0.15, 0.15,
                            100., 50.,
                            80., 120.,
                            1.0])
    ll_wrong = log_likelihood_shared(
        theta_wrong, '_exp', synthetic_two_groups, GROUPS, sdef1,
    )

    assert ll_true > ll_wrong


def test_log_prior_returns_minus_inf_outside_box():
    """log_prior_uniform is -inf when a parameter is outside its bounds."""
    l = [0.0, 0.0, 0.1]
    u = [1.0, 100.0, 10.0]
    theta_in  = [0.5, 50.0, 1.0]
    theta_out = [0.5, 150.0, 1.0]   # 150 > 100
    assert log_prior_uniform(theta_in,  l, u) == 0.0
    assert log_prior_uniform(theta_out, l, u) == -np.inf


def test_log_probability_returns_minus_inf_for_bad_theta(synthetic_two_groups):
    """log_probability_shared is -inf outside the prior box, without evaluating the likelihood."""
    sdef = SCENARIOS['_exp'][4]
    # std = 0 (below the lower bound 0.1) -> rejected by the prior
    theta_bad = np.array([0.5, 100., 100., 100., 100., 0.0])
    l = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    u = [1.0, 500., 500., 500., 500., 100.]
    val = log_probability_shared(
        theta_bad, l, u, '_exp', synthetic_two_groups, GROUPS, sdef,
    )
    assert val == -np.inf
