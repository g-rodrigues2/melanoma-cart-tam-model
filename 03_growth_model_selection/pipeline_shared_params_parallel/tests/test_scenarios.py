"""Tests of the partition of theta by scenario."""
import numpy as np

from scenarios import (
    MODEL_PARAM_NAMES, SCENARIOS,
    count_model_params, ic_slice_for_group, split_model_args_for_group,
)


GROUPS = ['control_b16', 'control_pdx']


def test_count_model_params_exp_scenarios():
    """_exp s1: r group-spec => 2 entries.  _exp s4: r global => 1 entry."""
    assert count_model_params(SCENARIOS['_exp'][1], GROUPS) == 2
    assert count_model_params(SCENARIOS['_exp'][4], GROUPS) == 1


def test_count_model_params_log_scenarios():
    """_log: s1 => 4 (r, cc per group); s2 and s3 => 3; s4 => 2."""
    assert count_model_params(SCENARIOS['_log'][1], GROUPS) == 4
    assert count_model_params(SCENARIOS['_log'][2], GROUPS) == 3
    assert count_model_params(SCENARIOS['_log'][3], GROUPS) == 3
    assert count_model_params(SCENARIOS['_log'][4], GROUPS) == 2


def test_split_log_s1_group_spec(synthetic_two_groups):
    """Scenario 1 (_log): theta = [r_b16, r_pdx, cc_b16, cc_pdx, T0s..., std] (by model parameter, then by
    group). Returns (r_b16, cc_b16) for b16 and (r_pdx, cc_pdx) for pdx.
    """
    theta = np.array([0.10, 0.25,                # r per group (b16, pdx)
                      1000., 2000.,              # cc per group (b16, pdx)
                      80., 90.,                  # T0 b16 (2 mice)
                      100., 120.,                # T0 pdx (2 mice)
                      50.])                      # std
    sdef = SCENARIOS['_log'][1]
    pn = MODEL_PARAM_NAMES['_log']

    args_b16, off = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_b16')
    assert args_b16 == (0.10, 1000.0)
    assert off == 4

    args_pdx, off2 = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_pdx')
    assert args_pdx == (0.25, 2000.0)
    assert off2 == 4


def test_split_log_s4_all_global(synthetic_two_groups):
    """Scenario 4 (_log): theta = [r, cc, T0s..., std]; b16 and pdx get the same (r, cc)."""
    theta = np.array([0.15, 1500.,              # global r and cc
                      80., 90., 100., 120.,     # T0s
                      40.])                     # std
    sdef = SCENARIOS['_log'][4]
    pn = MODEL_PARAM_NAMES['_log']

    args_b16, _ = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_b16')
    args_pdx, _ = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_pdx')
    assert args_b16 == args_pdx == (0.15, 1500.0)


def test_split_log_s2_cc_global(synthetic_two_groups):
    """Scenario 2 (_log): r per group, global cc. theta = [r_b16, r_pdx, cc, T0s, std]."""
    theta = np.array([0.10, 0.25, 1800.,
                      80., 90., 100., 120.,
                      50.])
    sdef = SCENARIOS['_log'][2]
    pn = MODEL_PARAM_NAMES['_log']

    args_b16, _ = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_b16')
    args_pdx, _ = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_pdx')
    assert args_b16 == (0.10, 1800.0)
    assert args_pdx == (0.25, 1800.0)


def test_split_log_s3_r_global(synthetic_two_groups):
    """Scenario 3 (_log): global r, cc per group. theta = [r, cc_b16, cc_pdx, T0s, std]."""
    theta = np.array([0.15, 800., 2500.,
                      80., 90., 100., 120.,
                      50.])
    sdef = SCENARIOS['_log'][3]
    pn = MODEL_PARAM_NAMES['_log']

    args_b16, _ = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_b16')
    args_pdx, _ = split_model_args_for_group(theta, sdef, pn, GROUPS, 'control_pdx')
    assert args_b16 == (0.15, 800.0)
    assert args_pdx == (0.15, 2500.0)


def test_ic_slice_extracts_correct_block(synthetic_two_groups):
    """ic_slice_for_group extracts the initial-condition block of each group."""
    fd = synthetic_two_groups
    # scenario 1 -> 4 model parameters; b16 has 2 mice (80, 90), pdx 2 mice (100, 120)
    theta = np.array([0.10, 1000., 0.25, 2000.,
                      80., 90., 100., 120., 50.])

    np.testing.assert_array_equal(
        ic_slice_for_group(theta, 4, fd, GROUPS, 'control_b16'),
        [80., 90.],
    )
    np.testing.assert_array_equal(
        ic_slice_for_group(theta, 4, fd, GROUPS, 'control_pdx'),
        [100., 120.],
    )
