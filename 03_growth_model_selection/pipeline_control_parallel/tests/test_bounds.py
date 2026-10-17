"""Tests of bounds.py."""
import numpy as np
import pytest

from bounds import define_bounds_labels, find_max_initial_condition


def test_define_bounds_exp_has_one_growth_param():
    lb, ub, lab = define_bounds_labels('_exp', group_size=5, max_ic=100.0,
                                       bool_multiple_mice=True)
    # 1 growth (r) + 5 ICs + 1 std
    assert len(lb) == len(ub) == len(lab) == 7
    assert lab[0] == 'r'
    assert lab[-1] == 'std'


def test_define_bounds_mac_has_three_growth_params():
    lb, ub, lab = define_bounds_labels('_mac', group_size=10, max_ic=100.0,
                                       bool_multiple_mice=True)
    # 3 growth (alpha1, K1, beta1) + 10 ICs + 1 std
    assert len(lb) == 14
    assert lab[:3] == ['alpha1', 'K1', 'beta1']


def test_define_bounds_lower_strictly_less_than_upper():
    for model in ['_exp', '_mendel', '_bert', '_lin', '_gomp', '_surf', '_log', '_mac']:
        lb, ub, _ = define_bounds_labels(model, group_size=5, max_ic=100.0,
                                         bool_multiple_mice=True)
        for l, u in zip(lb, ub):
            assert l < u, f"{model}: lower {l} >= upper {u}"


def test_define_bounds_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        define_bounds_labels('_unknown', group_size=5, max_ic=100.0)


def test_define_bounds_single_mouse_mode():
    lb, ub, lab = define_bounds_labels('_exp', group_size=5, max_ic=100.0,
                                       bool_multiple_mice=False)
    # 1 growth + 1 ic + 1 std (single-mouse mode)
    assert len(lb) == 3
    assert lab == ['r', 'ic', 'std']


def test_find_max_initial_condition_picks_largest_T0():
    full = {
        'A': [
            {'name': 'a1', 'data': np.array([[0, 50], [5, 100]])},
            {'name': 'a2', 'data': np.array([[0, 200], [5, 400]])},  # largest
        ],
        'B': [
            {'name': 'b1', 'data': np.array([[0, 150], [5, 300]])},
        ],
    }
    max_ic, grp, name = find_max_initial_condition(full)
    assert max_ic == 200
    assert grp == 'A'
    assert name == 'a2'
