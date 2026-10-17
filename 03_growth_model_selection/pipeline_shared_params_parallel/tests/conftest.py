"""Pytest fixtures. Puts pipeline_shared_params_parallel/ and pipeline_control_parallel/ on sys.path
(through _path_setup)."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.abspath(os.path.dirname(__file__))
_PIPELINE = os.path.abspath(os.path.join(_HERE, '..'))
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)

# adds pipeline_control_parallel/ to sys.path
import _path_setup  # noqa: F401


@pytest.fixture
def synthetic_two_groups():
    """Synthetic full_data with 2 groups x 2 mice, growing as exp(0.1*t) in the first group and exp(0.2*t)
    in the second."""
    t = np.array([0, 2, 4, 6, 8, 10], dtype=float)

    def make_mouse(name, r, T0):
        T = T0 * np.exp(r * t)
        return {
            'name': name,
            'data': np.column_stack((t, T)),
            'treatment': 'pbs', 'treatment_days': np.array([], dtype=int),
            'panel': 'X', 'cohort': 'X',
        }

    return {
        'control_b16': [make_mouse('m1', 0.1, 100.0),
                        make_mouse('m2', 0.1,  50.0)],
        'control_pdx': [make_mouse('m1', 0.2,  80.0),
                        make_mouse('m2', 0.2, 120.0)],
    }
