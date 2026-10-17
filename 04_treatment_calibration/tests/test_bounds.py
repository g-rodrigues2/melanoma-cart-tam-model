"""Tests of bounds.py (r is not calibrated: it is fixed in models.py)."""
import numpy as np
import pytest

from bounds import define_bounds_labels, find_max_initial_condition


# (model, number of treatment parameters in theta)
ALL_MODEL_PARS = [
    ('_exp', 0),           # no treatment term
    ('_exp_ED', 2),
    ('_exp_CD', 2),
    ('_exp_AccelED', 2),
    ('_exp_AccelCD', 2),
    ('_exp_MM', 2),
    ('_exp_LS', 2),
    ('_exp_PF', 2),
    ('_exp_dePR', 3),
    ('_exp_dePRDecay', 4),
    ('_exp_CDwithLog', 3),
    ('_exp_CDwithExp', 3),
    ('_exp_CDwithSin', 3),
    ('_exp_EDwithLog', 3),
    ('_exp_EDwithExp', 3),
    ('_exp_EDwithSin', 3),
]


def test_find_max_initial_condition_basic():
    full = {
        'g1': [{'name': 'm1', 'data': np.array([[0., 50.], [5., 100.]])}],
        'g2': [{'name': 'm2', 'data': np.array([[0., 80.], [5., 200.]])}],
    }
    ic, group, name = find_max_initial_condition(full)
    assert ic == 80.0
    assert group == 'g2'
    assert name == 'm2'


@pytest.mark.parametrize("model_extension,expected_n_pars", ALL_MODEL_PARS)
def test_bounds_have_correct_dim(model_extension, expected_n_pars):
    group_size = 5
    max_ic = 200.0
    l, u, labels = define_bounds_labels(model_extension, group_size, max_ic,
                                        bool_multiple_mice=True)
    assert len(l) == expected_n_pars + group_size + 1
    assert len(u) == len(l) == len(labels)
    for li, ui in zip(l, u):
        assert li < ui, f'{model_extension}: l={li} >= u={ui}'


def test_bounds_unknown_model_raises():
    with pytest.raises(ValueError):
        define_bounds_labels('_unknown', 3, 100.0)


def test_bounds_no_r_in_labels():
    """r is not among the calibrated parameters (it is a constant of models.py)."""
    for m, _ in ALL_MODEL_PARS:
        _, _, labels = define_bounds_labels(m, 3, 100.0, bool_multiple_mice=True)
        assert 'r' not in labels, f'{m}: tem r em {labels}'


def test_bounds_std_last_label():
    for m, _ in ALL_MODEL_PARS:
        _, _, labels = define_bounds_labels(m, 3, 100.0, bool_multiple_mice=True)
        assert labels[-1] == 'std'


def test_bounds_ic_range_multi_mice():
    """Initial conditions in [0, 5*max_ic] (wide enough for outlying mice)."""
    max_ic = 100.0
    l, u, labels = define_bounds_labels('_exp_ED', 5, max_ic, bool_multiple_mice=True)
    ic_indices = [i for i, lbl in enumerate(labels) if lbl.startswith('ic')]
    for i in ic_indices:
        assert l[i] == 0.0
        assert u[i] >= 2 * max_ic, f'ic upper {u[i]} < 2*max_ic'


def test_bounds_std_range():
    """std covers a wide variability between mice."""
    from bounds import STD_LOW, STD_HIGH
    l, u, labels = define_bounds_labels('_exp_ED', 5, 100.0, bool_multiple_mice=True)
    assert l[-1] == STD_LOW
    assert u[-1] == STD_HIGH
    assert STD_HIGH >= 200.0


# === _mac / _mac_treat: alpha1/K1/beta1 em log10 (reparametrizacao) ===

def test_mac_bounds_dim():
    l, u, labels = define_bounds_labels('_mac', 5, 200.0, bool_multiple_mice=True)
    assert len(l) == 3 + 5 + 1  # log10_alpha1, log10_K1, log10_beta1 + ICs + std
    assert labels[:3] == ['log10_alpha1', 'log10_K1', 'log10_beta1']


def test_mac_treat_bounds_dim():
    l, u, labels = define_bounds_labels('_mac_treat', 5, 200.0, bool_multiple_mice=True)
    assert len(l) == 8 + 5 + 1
    assert labels[:8] == ['log10_alpha1', 'log10_K1', 'log10_beta1', 'mu1',
                          'alpha3', 'K3', 'kappa3', 'mu3']


def test_mac_bounds_are_log10_of_physical_range():
    """The bounds of log10_alpha1/K1/beta1 are the log10 of the physical bounds (MAC_ALPHA1_LO/HI, ...)."""
    from bounds import (MAC_ALPHA1_LO, MAC_ALPHA1_HI, MAC_K1_LO, MAC_K1_HI,
                        MAC_BETA1_LO, MAC_BETA1_HI)
    l, u, labels = define_bounds_labels('_mac', 3, 100.0, bool_multiple_mice=True)
    assert l[0] == pytest.approx(np.log10(MAC_ALPHA1_LO))
    assert u[0] == pytest.approx(np.log10(MAC_ALPHA1_HI))
    assert l[1] == pytest.approx(np.log10(MAC_K1_LO))
    assert u[1] == pytest.approx(np.log10(MAC_K1_HI))
    assert l[2] == pytest.approx(np.log10(MAC_BETA1_LO))
    assert u[2] == pytest.approx(np.log10(MAC_BETA1_HI))


def test_mac_treat_cart_params_stay_linear():
    """mu1, alpha3, K3, kappa3 and mu3 are not log-transformed (their bounds include 0)."""
    from bounds import (CART_MU1_LO, CART_MU1_HI, CART_ALPHA3_LO, CART_ALPHA3_HI,
                        CART_K3_LO, CART_K3_HI, CART_KAPPA3_LO, CART_KAPPA3_HI,
                        CART_MU3_LO, CART_MU3_HI)
    l, u, labels = define_bounds_labels('_mac_treat', 3, 100.0, bool_multiple_mice=True)
    assert (l[3], u[3]) == (CART_MU1_LO, CART_MU1_HI)
    assert (l[4], u[4]) == (CART_ALPHA3_LO, CART_ALPHA3_HI)
    assert (l[5], u[5]) == (CART_K3_LO, CART_K3_HI)
    assert (l[6], u[6]) == (CART_KAPPA3_LO, CART_KAPPA3_HI)
    assert (l[7], u[7]) == (CART_MU3_LO, CART_MU3_HI)
