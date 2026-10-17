import os
import sys

import numpy as np
from scipy.integrate import odeint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compiled_ode  # noqa: E402
import models  # noqa: E402

TIMES = np.array([0.0, 2, 5, 7, 9, 12, 14, 16, 19, 21, 23, 27])


def test_growth_laws_and_tam_system_match_odeint():
    cases = [('_exp', (0.25,), [60.0]), ('_gomp', (0.1, 5000.0, 1.0), [60.0]), ('_bert', (0.9, 0.05), [60.0]),
             ('_mac', (0.3, 3000.0, 1e-3), [60.0, 30.0])]
    for model, pars, y0 in cases:
        fn = getattr(models, 'model' + model)
        ref = odeint(getattr(fn, 'py_func', fn), y0, TIMES, args=pars, rtol=1e-12, atol=1e-10, mxstep=100000)
        new = compiled_ode.solve(models._COMPILED_RHS.get(model, fn), TIMES, pars, y0, len(y0))
        assert np.allclose(new, ref, rtol=1e-7, atol=1e-6), model
