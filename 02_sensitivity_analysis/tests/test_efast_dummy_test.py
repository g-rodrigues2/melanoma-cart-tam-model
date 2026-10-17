"""Dummy-parameter significance test of Section 3.3.3.

What is checked here is the construction of the test: that the eleventh parameter is genuinely inert (the
model never reads it), that the search curves are independent replicates rather than repetitions of the
same design, and that the t-test recovers the expected answer on synthetic inputs whose truth is known.
"""
import os
import sys

import numpy as np
from SALib.sample import fast_sampler

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE)

import efast_sensitivity as efast          # noqa: E402
import efast_dummy_test as dt              # noqa: E402

T_OUT = np.array([5.0, 20.0, 60.0])
N_TEST = 70          # eFAST requires N > 4 M^2; the chapter uses N = 10,000


def _design(seed, n=N_TEST):
    return fast_sampler.sample(dt.problem, n, M=efast.M_fast, seed=seed)


def test_problem_adds_exactly_one_parameter_to_the_chapter_design():
    assert dt.problem['num_vars'] == efast.problem['num_vars'] + 1
    assert dt.problem['names'][:-1] == efast.param_names
    assert dt.problem['names'][-1] == 'dummy'
    assert np.allclose(dt.problem['bounds'][:-1], efast.bounds)


def test_the_dummy_never_reaches_the_model():
    """Two designs differing only in the dummy column must give identical trajectories."""
    X = _design(seed=1000)
    X_moved = X.copy()
    X_moved[:, dt.i_dummy] = dt.problem['bounds'][-1, 1]     # push the dummy to its upper bound
    y0 = np.array(efast.TMC0, dtype=float)
    Y = efast.solve_all(y0, X, T_OUT, efast.RTOL, efast.ATOL)
    Y_moved = efast.solve_all(y0, X_moved, T_OUT, efast.RTOL, efast.ATOL)
    assert np.array_equal(Y, Y_moved)


def test_search_curves_are_distinct_designs_over_the_same_ranges():
    X0, X1 = _design(seed=dt.SEEDS[0]), _design(seed=dt.SEEDS[1])
    assert X0.shape == X1.shape
    assert not np.allclose(X0, X1)                            # different phase shifts
    lower, upper = dt.problem['bounds'][:, 0], dt.problem['bounds'][:, 1]
    for X in (X0, X1):
        assert np.all(X >= lower) and np.all(X <= upper)


def test_seeds_are_distinct_and_match_the_declared_number_of_curves():
    assert len(dt.SEEDS) == dt.NR
    assert len(set(dt.SEEDS)) == dt.NR


def test_bonferroni_level_divides_by_the_number_of_real_parameters():
    assert dt.n_par == efast.problem['num_vars']
    assert np.isclose(dt.alpha_bonferroni, dt.ALPHA / dt.n_par)


def test_ttest_separates_an_influential_column_from_a_null_one():
    """A column clearly above the dummy is significant; one drawn from the same law is not."""
    rng = np.random.default_rng(0)
    idx = rng.normal(0.001, 0.0002, size=(dt.NR, dt.problem['num_vars']))
    idx[:, 0] = rng.normal(0.5, 0.0002, size=dt.NR)           # strongly influential parameter
    p = dt.dummy_test(idx)
    assert p[0] < dt.alpha_bonferroni
    assert np.all(p[1:dt.n_par] > dt.alpha_bonferroni)


def test_ttest_is_one_sided():
    """A parameter well below the dummy is not called influential."""
    rng = np.random.default_rng(1)
    idx = rng.normal(0.5, 0.001, size=(dt.NR, dt.problem['num_vars']))
    idx[:, 0] = rng.normal(0.001, 0.001, size=dt.NR)
    p = dt.dummy_test(idx)
    assert p[0] > 0.5


def test_ttest_returns_one_p_value_per_real_parameter():
    rng = np.random.default_rng(2)
    p = dt.dummy_test(rng.normal(0.01, 0.001, size=(dt.NR, dt.problem['num_vars'])))
    assert p.shape == (dt.n_par,)
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_a_column_identical_to_the_dummy_is_not_significant():
    """Degenerate case: Welch's statistic is undefined with zero variance in both samples."""
    idx = np.full((dt.NR, dt.problem['num_vars']), 0.01)
    p = dt.dummy_test(idx)
    assert np.all(p == 1.0)
