"""Tests of the bounds of the multi-group calibration."""

from bounds_shared import define_shared_bounds_labels
from scenarios import SCENARIOS


GROUPS = ['control_b16', 'control_pdx']


def test_bounds_exp_s1_group(synthetic_two_groups):
    """_exp s1: theta = [r_b16, r_pdx, T0_b16_1, T0_b16_2, T0_pdx_1, T0_pdx_2, std].
    ndim = 2 + 2 + 2 + 1 = 7."""
    fd = synthetic_two_groups
    l, u, labels = define_shared_bounds_labels(
        '_exp', SCENARIOS['_exp'][1], GROUPS, fd, max_ic=200.0,
    )
    assert len(l) == len(u) == len(labels) == 7
    assert labels[0] == 'r_b16'
    assert labels[1] == 'r_pdx'
    assert labels[-1] == 'std'
    # bounds of r (from _exp): [0, 1]
    assert l[0] == 0.0 and u[0] == 1.0
    assert l[1] == 0.0 and u[1] == 1.0


def test_bounds_exp_s4_global(synthetic_two_groups):
    """_exp s4: theta = [r_global, T0_b16_1, T0_b16_2, T0_pdx_1, T0_pdx_2, std].
    ndim = 1 + 2 + 2 + 1 = 6."""
    fd = synthetic_two_groups
    l, u, labels = define_shared_bounds_labels(
        '_exp', SCENARIOS['_exp'][4], GROUPS, fd, max_ic=200.0,
    )
    assert len(l) == 6
    assert labels[0] == 'r_global'
    assert labels[-1] == 'std'


def test_bounds_log_s1_full_group_spec(synthetic_two_groups):
    """_log s1: theta = [r_b16, cc_b16, r_pdx, cc_pdx, T0s, std]. ndim = 4+4+1 = 9."""
    fd = synthetic_two_groups
    l, u, labels = define_shared_bounds_labels(
        '_log', SCENARIOS['_log'][1], GROUPS, fd, max_ic=200.0,
    )
    assert len(l) == 9
    # label order: r in group order (b16, pdx), then cc (b16, pdx)
    assert labels[:4] == ['r_b16', 'r_pdx', 'cc_b16', 'cc_pdx']


def test_bounds_log_s4_all_global(synthetic_two_groups):
    """_log s4: theta = [r, cc, T0s, std]. ndim = 1+1+4+1 = 7."""
    fd = synthetic_two_groups
    l, u, labels = define_shared_bounds_labels(
        '_log', SCENARIOS['_log'][4], GROUPS, fd, max_ic=200.0,
    )
    assert len(l) == 7
    assert labels[:2] == ['r_global', 'cc_global']


def test_bounds_ic_bounds_correct(synthetic_two_groups):
    """Initial conditions have bounds [0, 2*max_ic] and std [0.1, 1000]."""
    fd = synthetic_two_groups
    l, u, labels = define_shared_bounds_labels(
        '_exp', SCENARIOS['_exp'][4], GROUPS, fd, max_ic=200.0,
    )
    # initial-condition bounds in positions 1..4 (after r_global)
    for i in range(1, 5):
        assert l[i] == 0.0
        assert u[i] == 400.0  # max_ic * 2
        assert labels[i].startswith('ic_')
    # std
    assert l[-1] == 0.1 and u[-1] == 1000.0
