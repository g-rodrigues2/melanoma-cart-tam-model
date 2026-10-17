"""Tests of the 18 treatment models."""
import numpy as np
import pytest

from models import (ALL_MODELS, EXP_TREATMENT_MODELS,
                    MAC_2D_MODELS, MAC_3D_MODELS,
                    MAC_M0_FRACTION, CART_C0,
                    R_B16_FIXED, solve_model, _split_theta)


T0    = 100.0
TGRID = np.linspace(0, 20, 21)


def test_all_models_count():
    assert len(ALL_MODELS) == 18
    assert len(EXP_TREATMENT_MODELS) == 15
    assert MAC_2D_MODELS == {'_mac'}
    assert MAC_3D_MODELS == {'_mac_treat'}


def test_r_b16_fixed_constant():
    assert abs(R_B16_FIXED - 0.2648) < 1e-3


def test_exp_baseline():
    sol = solve_model('_exp', TGRID, (), T0)
    sol_T = sol[:, 0] if sol.ndim == 2 else sol
    expected = T0 * np.exp(R_B16_FIXED * TGRID)
    np.testing.assert_allclose(sol_T, expected, rtol=1e-4)


@pytest.mark.parametrize("model,extra", [
    ('_exp_ED',      (0.0, 1.0)),
    ('_exp_CD',      (0.0, 1.0)),
    ('_exp_AccelED', (0.0, 1.0)),
    ('_exp_AccelCD', (0.0, 1.0)),
    ('_exp_MM',      (0.0, 1.0)),
    ('_exp_LS',      (0.0, 0.001)),
    ('_exp_PF',      (0.0, 0.5)),
    ('_exp_CDwithLog', (0.0, 1.0, 0.5)),
    ('_exp_CDwithExp', (0.0, 1.0, 0.5)),
    ('_exp_CDwithSin', (0.0, 1.0, 0.5)),
    ('_exp_EDwithLog', (0.0, 1.0, 0.5)),
    ('_exp_EDwithExp', (0.0, 1.0, 0.5)),
    ('_exp_EDwithSin', (0.0, 1.0, 0.5)),
])
def test_zero_amplitude_collapses_to_exp(model, extra):
    sol = solve_model(model, TGRID, extra, T0)
    T_t = sol[:, 0] if sol.ndim == 2 else sol
    T_e = T0 * np.exp(R_B16_FIXED * TGRID)
    np.testing.assert_allclose(T_t, T_e, rtol=1e-5)


def test_dePR_zero_d_collapses_to_exp():
    sol = solve_model('_exp_dePR', TGRID, (0.0, 0.5, 1.0), T0)
    T_t = sol[:, 0] if sol.ndim == 2 else sol
    np.testing.assert_allclose(T_t, T0 * np.exp(R_B16_FIXED * TGRID), rtol=1e-5)


def test_dePRDecay_zero_d_collapses_to_exp():
    sol = solve_model('_exp_dePRDecay', TGRID, (0.0, 0.5, 1.0, 1.0), T0)
    T_t = sol[:, 0] if sol.ndim == 2 else sol
    np.testing.assert_allclose(T_t, T0 * np.exp(R_B16_FIXED * TGRID), rtol=1e-5)


def test_mac_solves():
    ic = [T0, MAC_M0_FRACTION * T0]
    sol = solve_model('_mac', TGRID, (0.2, 5000.0, 1e-4), ic)
    assert sol.shape == (len(TGRID), 2)
    assert np.all(sol >= 0)


def test_mac_treat_solves():
    """8 params calibrados, 3D."""
    ic = [T0, MAC_M0_FRACTION * T0, CART_C0]
    args = (0.2, 5000.0, 1e-4, 0.5, 1.0, 100.0, 0.3, 1e-4)
    sol = solve_model('_mac_treat', TGRID, args, ic)
    assert sol.shape == (len(TGRID), 3)
    assert np.all(sol[:, 0] >= 0)
    assert np.all(sol[:, 1] >= 0)
    assert sol[0, 2] == pytest.approx(CART_C0)


def test_mac_treat_with_mu1_zero_equals_mac():
    args_mac   = (0.2, 5000.0, 1e-4)
    args_treat = (0.2, 5000.0, 1e-4, 0.0, 1.0, 100.0, 0.3, 1e-4)
    ic_mac   = [T0, MAC_M0_FRACTION * T0]
    ic_treat = [T0, MAC_M0_FRACTION * T0, CART_C0]
    sol_mac   = solve_model('_mac', TGRID, args_mac, ic_mac)
    sol_treat = solve_model('_mac_treat', TGRID, args_treat, ic_treat)
    np.testing.assert_allclose(sol_mac[:, 0], sol_treat[:, 0], rtol=1e-4)
    np.testing.assert_allclose(sol_mac[:, 1], sol_treat[:, 1], rtol=1e-4)


def test_mac_treat_positive_mu1_reduces_tumor():
    base  = (0.2, 5000.0, 1e-4, 0.0, 1.0, 100.0, 0.3, 1e-4)
    treat = (0.2, 5000.0, 1e-4, 1.0, 1.0, 100.0, 0.3, 1e-4)
    ic = [T0, MAC_M0_FRACTION * T0, CART_C0]
    sol_base  = solve_model('_mac_treat', TGRID, base, ic)
    sol_treat = solve_model('_mac_treat', TGRID, treat, ic)
    assert sol_treat[-1, 0] < sol_base[-1, 0]


def test_mac_treat_mu3_exhausts_cart():
    no_exh  = (0.2, 5000.0, 1e-4, 0.5, 1.0, 100.0, 0.3, 0.0)
    yes_exh = (0.2, 5000.0, 1e-4, 0.5, 1.0, 100.0, 0.3, 1e-2)
    ic = [T0, MAC_M0_FRACTION * T0, CART_C0]
    sol_no  = solve_model('_mac_treat', TGRID, no_exh,  ic)
    sol_yes = solve_model('_mac_treat', TGRID, yes_exh, ic)
    assert sol_yes[-1, 2] < sol_no[-1, 2]


# === _split_theta: reparametrizacao log10 de alpha1/K1/beta1 ===

def test_split_theta_mac_exponentiates_first_three():
    """theta[0:3] are log10 values; _split_theta returns 10**value."""
    alpha1_phys, K1_phys, beta1_phys = 0.2, 5000.0, 1e-4
    log_theta = np.array([
        np.log10(alpha1_phys), np.log10(K1_phys), np.log10(beta1_phys),
        T0,      # ic1
        50.0,    # std
    ])
    ode_args, ic_fn = _split_theta(log_theta, '_mac', ndim=1)
    assert ode_args[0] == pytest.approx(alpha1_phys)
    assert ode_args[1] == pytest.approx(K1_phys)
    assert ode_args[2] == pytest.approx(beta1_phys)
    assert ic_fn(0) == [T0, MAC_M0_FRACTION * T0]


def test_split_theta_mac_treat_exponentiates_only_first_three():
    """In _mac_treat only alpha1, K1 and beta1 are log10; mu1..mu3 stay linear."""
    alpha1_phys, K1_phys, beta1_phys = 0.1, 8000.0, 2e-3
    mu1, alpha3, K3, kappa3, mu3 = 0.5, 1.2, 200.0, 0.3, 0.01
    log_theta = np.array([
        np.log10(alpha1_phys), np.log10(K1_phys), np.log10(beta1_phys),
        mu1, alpha3, K3, kappa3, mu3,
        T0,   # ic1
        50.0, # std
    ])
    ode_args, ic_fn = _split_theta(log_theta, '_mac_treat', ndim=1)
    assert ode_args[0] == pytest.approx(alpha1_phys)
    assert ode_args[1] == pytest.approx(K1_phys)
    assert ode_args[2] == pytest.approx(beta1_phys)
    assert ode_args[3:] == pytest.approx((mu1, alpha3, K3, kappa3, mu3))
    assert ic_fn(0) == [T0, MAC_M0_FRACTION * T0, CART_C0]


def test_split_theta_exp_models_unaffected_by_log_transform():
    """The _exp_* models are not exponentiated (only _mac and _mac_treat are)."""
    theta = np.array([0.5, 1.5, T0, 50.0])  # a_imuno, b_imuno, ic1, std
    ode_args, ic_fn = _split_theta(theta, '_exp_ED', ndim=1)
    assert ode_args == pytest.approx((0.5, 1.5))
    assert ic_fn(0) == pytest.approx(T0)


def test_split_theta_mac_roundtrip_matches_direct_solve():
    """solve_model with log10 theta through _split_theta equals solve_model with the same physical parameters."""
    alpha1_phys, K1_phys, beta1_phys = 0.15, 6000.0, 3e-4
    log_theta = np.array([np.log10(alpha1_phys), np.log10(K1_phys),
                          np.log10(beta1_phys), T0, 50.0])
    ode_args, ic_fn = _split_theta(log_theta, '_mac', ndim=1)
    sol_via_split = solve_model('_mac', TGRID, ode_args, ic_fn(0))
    sol_direct = solve_model('_mac', TGRID, (alpha1_phys, K1_phys, beta1_phys),
                             [T0, MAC_M0_FRACTION * T0])
    np.testing.assert_allclose(sol_via_split, sol_direct)
