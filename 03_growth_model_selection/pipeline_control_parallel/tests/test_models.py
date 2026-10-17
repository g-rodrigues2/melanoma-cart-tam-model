"""Tests of the ODE models and solver dispatch."""
import numpy as np
import pytest

from models import _split_theta, solve_model, MAC_M0_FRACTION


def test_model_exp_growth():
    """T(t) = T0 * exp(r*t). For r = 0.1, T0 = 100, t = 10: T(10) ~ 271.83."""
    t_arr = np.array([0.0, 5.0, 10.0])
    sol = solve_model('_exp', t_arr, (0.1,), 100.0)
    assert sol.shape == (3, 1)
    np.testing.assert_allclose(sol[0, 0], 100.0, rtol=1e-3)
    np.testing.assert_allclose(sol[-1, 0], 100.0 * np.exp(1.0), rtol=1e-3)


def test_model_log_saturates_at_cc():
    """The logistic model tends to cc for large t."""
    sol = solve_model('_log', np.linspace(0, 100, 50), (1.0, 1000.0), 10.0)
    assert sol[-1, 0] == pytest.approx(1000.0, rel=1e-3)


def test_solve_model_smooth_returns_time_column():
    """type_sol='smooth' returns (t, *states) on a 0.1-day grid."""
    sol = solve_model('_exp', np.array([0.0, 5.0]), (0.1,), 100.0,
                      type_sol='smooth')
    # 0.1-day grid between 0 and 5 -> 51 points
    assert sol.shape == (51, 2)
    np.testing.assert_allclose(sol[0, 0], 0.0)
    np.testing.assert_allclose(sol[-1, 0], 5.0)


def test_model_mac_multistate():
    """_mac is (T, M) with macrophages on the mm^3 scale. After rescaling alpha2 (1e-7 /cell -> 0.1 /mm^3),
    M grows when T > kappa2/alpha2 = 3.4 mm^3."""
    # T0 = 100 mm^3 (above the 3.4 mm^3 threshold): T and M grow
    sol = solve_model('_mac', np.array([0.0, 5.0]),
                      (0.3, 5000.0, 1e-4), [100.0, 50.0])
    assert sol.shape == (2, 2)
    assert sol[-1, 0] > sol[0, 0]       # T grows
    assert sol[-1, 1] > sol[0, 1]       # M grows too (alpha2*T >> kappa2)


def test_model_mac_low_tumor_M_decays():
    """For a very small T (< kappa2/alpha2 = 3.4 mm^3), M decays."""
    sol = solve_model('_mac', np.array([0.0, 5.0]),
                      (0.3, 5000.0, 1e-4), [1.0, 50.0])   # T0 = 1 mm^3 (small)
    assert sol.shape == (2, 2)
    # alpha2 * T = 0.1 * 1 = 0.1 << kappa2 = 0.34 -> M decays
    assert sol[-1, 1] < sol[0, 1]


def test_split_theta_1d_model():
    """1D models: ode_args = leading parameters; ic_fn(idx) = scalar T0."""
    # _exp: theta = [r, T0_1, T0_2, std]
    theta = np.array([0.3, 50.0, 80.0, 10.0])
    ode_args, ic_fn = _split_theta(theta, '_exp', ndim=2)
    assert ode_args == (0.3,)
    assert ic_fn(0) == 50.0
    assert ic_fn(1) == 80.0


def test_split_theta_mac_multistate():
    """_mac: ode_args = (alpha1, K1, beta1); ic_fn(idx) = [T0, M0]."""
    theta = np.array([0.3, 5000.0, 1e-6, 100.0, 200.0, 10.0])
    ode_args, ic_fn = _split_theta(theta, '_mac', ndim=2)
    assert ode_args == (0.3, 5000.0, 1e-6)
    np.testing.assert_array_equal(ic_fn(0), [100.0, MAC_M0_FRACTION * 100.0])
    np.testing.assert_array_equal(ic_fn(1), [200.0, MAC_M0_FRACTION * 200.0])


def test_odeint_and_solve_ivp_agree(monkeypatch):
    """Both solvers give the same result for a simple model."""
    import models
    t = np.linspace(0, 5, 10)

    monkeypatch.setattr(models, 'SOLVER', 'odeint')
    sol_odeint = models.solve_model('_exp', t, (0.3,), 50.0)

    monkeypatch.setattr(models, 'SOLVER', 'solve_ivp')
    sol_ivp = models.solve_model('_exp', t, (0.3,), 50.0)

    np.testing.assert_allclose(sol_odeint, sol_ivp, rtol=1e-4)
