import os
import sys

import numpy as np
from scipy.integrate import odeint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compiled_ode  # noqa: E402
import models  # noqa: E402

TIMES = np.array([0.0, 2, 5, 7, 9, 12, 14, 16, 19, 21, 23, 27])


def _reference(fn, y0, pars):
    return odeint(getattr(fn, 'py_func', fn), y0, TIMES, args=pars, rtol=1e-12, atol=1e-10, mxstep=100000)


def test_killing_terms_match_odeint():
    for model, pars in [('_exp', ()), ('_exp_ED', (0.3, 0.1)), ('_exp_EDwithSin', (0.4, 0.05, 0.8)),
                        ('_exp_dePR', (0.5, 0.01, 1.2))]:
        fn = getattr(models, 'model' + model)
        new = compiled_ode.solve(fn, TIMES, pars, [60.0])
        assert np.allclose(new, _reference(fn, [60.0], pars), rtol=1e-7, atol=1e-6), model


def test_tmc_system_matches_odeint():
    pars = (0.3, 3000.0, 1e-3, 0.8, 0.9, 50.0, 0.1, 1e-4)
    y0 = [60.0, 30.0, 1.0]
    new = compiled_ode.solve(models._mac_treat_compiled, TIMES, pars, y0, 3)
    ref = _reference(models.model_mac_treat, y0, pars)
    assert np.allclose(new, ref, rtol=1e-7, atol=1e-6)


def test_solve_model_default_is_compiled_and_agrees_with_odeint():
    theta = (0.3, 3000.0, 1e-3, 0.8, 0.9, 50.0, 0.1, 1e-4)
    models.SOLVER = 'compiled'
    fast = models.solve_model('_mac_treat', TIMES, theta, [60.0, 30.0, 1.0])
    models.SOLVER = 'odeint'
    slow = models.solve_model('_mac_treat', TIMES, theta, [60.0, 30.0, 1.0])
    models.SOLVER = 'compiled'
    assert np.allclose(fast, slow, rtol=1e-5)
