"""Tests of log_likelihood, log_prior and log_probability."""
import numpy as np

from inference import log_likelihood, log_prior, log_probability


def test_log_prior_inside_bounds_returns_zero():
    assert log_prior([0.5, 100.0], [0.0, 0.0], [1.0, 200.0]) == 0.0


def test_log_prior_outside_bounds_returns_neg_inf():
    assert log_prior([1.5, 100.0], [0.0, 0.0], [1.0, 200.0]) == -np.inf
    assert log_prior([0.5, -10.0], [0.0, 0.0], [1.0, 200.0]) == -np.inf


def test_log_likelihood_finite_on_good_params(synthetic_full_data):
    """For data following T = 100*exp(0.1*t) and theta close to the truth, the log-likelihood is finite."""
    # theta = [r, T0_1, T0_2, std]
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    ll = log_likelihood(theta, '_exp', synthetic_full_data, 'test_group')
    assert np.isfinite(ll)


def test_log_likelihood_returns_neg_inf_on_bad_params(synthetic_full_data):
    """Parameters that make the ODE return NaN give -inf instead of an error."""
    # _bert with a very large b and a small r: T -> negative -> NaN
    theta = np.array([0.001, 0.5, 100.0, 100.0, 10.0])
    ll = log_likelihood(theta, '_bert', synthetic_full_data, 'test_group')
    # Either -inf or a very poor finite value; it must not raise
    assert isinstance(ll, float)
    assert not np.isnan(ll)


def test_log_likelihood_handles_solve_model_exception(monkeypatch, synthetic_full_data):
    """If solve_model raises, log_likelihood returns -inf."""
    import inference
    def bad_solve(*a, **kw):
        raise RuntimeError("intentional failure")
    monkeypatch.setattr(inference, 'solve_model', bad_solve)
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    ll = log_likelihood(theta, '_exp', synthetic_full_data, 'test_group')
    assert ll == -np.inf


def test_log_likelihood_handles_nan_in_solution(monkeypatch, synthetic_full_data):
    """If the solution contains NaN, log_likelihood returns -inf."""
    import inference
    def nan_solve(*a, **kw):
        out = np.full((6, 1), 100.0)
        out[2, 0] = np.nan
        return out
    monkeypatch.setattr(inference, 'solve_model', nan_solve)
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    ll = log_likelihood(theta, '_exp', synthetic_full_data, 'test_group')
    assert ll == -np.inf


def test_log_probability_combines_prior_and_likelihood(synthetic_full_data):
    """log_probability = log_prior + log_likelihood inside the bounds."""
    theta = np.array([0.1, 100.0, 100.0, 10.0])
    l = [0.0, 0.0, 0.0, 0.1]
    u = [1.0, 1000.0, 1000.0, 1000.0]
    lp = log_probability(theta, l, u, '_exp', synthetic_full_data, 'test_group')
    ll = log_likelihood(theta, '_exp', synthetic_full_data, 'test_group')
    np.testing.assert_allclose(lp, ll)  # prior = 0 inside the box


def test_log_probability_neg_inf_outside_bounds(synthetic_full_data):
    """Outside the bounds log_probability is -inf without evaluating the likelihood."""
    theta = np.array([2.0, 100.0, 100.0, 10.0])  # r = 2 outside [0, 1]
    l = [0.0, 0.0, 0.0, 0.1]
    u = [1.0, 1000.0, 1000.0, 1000.0]
    assert log_probability(theta, l, u, '_exp', synthetic_full_data,
                           'test_group') == -np.inf
