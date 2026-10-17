"""Sampling design and compiled integrator of the eFAST analysis of Section 3.3.3.

The indices themselves are produced by SALib; what is checked here is the input of that analysis: that the
design has the size and the ranges declared in the chapter, and that the compiled solver that integrates
the 100,000 trajectories agrees with `scipy.integrate.odeint`.
"""
import os
import sys

import numpy as np
from SALib.sample import fast_sampler

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE)

import efast_sensitivity as efast  # noqa: E402

T_OUT = np.array([5.0, 20.0, 40.0, 60.0])
N_TEST = 70          # eFAST requires N > 4 M^2; the chapter uses N = 10,000


def _design(n=N_TEST):
    return fast_sampler.sample(efast.problem, n, M=efast.M_fast, seed=efast.seed)


def test_design_has_one_block_of_samples_per_parameter():
    X = _design()
    assert X.shape == (N_TEST * efast.problem['num_vars'], efast.problem['num_vars'])


def test_parameter_ranges_are_twenty_percent_around_the_reference_values():
    lower, upper = efast.bounds[:, 0], efast.bounds[:, 1]
    assert np.allclose(lower, 0.8 * efast.base_values)
    assert np.allclose(upper, 1.2 * efast.base_values)
    X = _design()
    assert np.all(X >= lower) and np.all(X <= upper)


def test_compiled_solver_agrees_with_odeint():
    X = _design()
    worst = efast.check_against_odeint(X, T_OUT, n=5)
    assert worst < 1e-7, worst


def test_solution_does_not_depend_on_the_output_grid():
    """The solver stops exactly at the requested times, so adding times must not move the others."""
    X = _design()
    coarse = efast.solve_all(np.array(efast.TMC0, dtype=float), X[:3], T_OUT, efast.RTOL, efast.ATOL)
    fine_times = np.unique(np.concatenate((T_OUT, np.linspace(0.5, 59.5, 40))))
    fine = efast.solve_all(np.array(efast.TMC0, dtype=float), X[:3], fine_times, efast.RTOL, efast.ATOL)
    keep = np.searchsorted(fine_times, T_OUT)
    assert np.allclose(coarse, fine[:, keep, :], rtol=1e-8, atol=1e-6)


def test_trajectories_stay_finite_and_non_negative():
    X = _design()
    Y = efast.solve_all(np.array(efast.TMC0, dtype=float), X, T_OUT, efast.RTOL, efast.ATOL)
    assert np.all(np.isfinite(Y))
    assert np.all(Y >= 0.0)
    assert np.all(Y[:, :, 0] <= efast.bounds[4, 1])   # T stays below the largest sampled K1
