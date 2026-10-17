"""Shared pytest fixtures. Puts the pipeline folder on sys.path so that its modules (models, bounds, ...)
can be imported."""
import os
import sys

import numpy as np
import pytest

# tests/.. is the pipeline folder
_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)


@pytest.fixture
def synthetic_mouse_data():
    """One synthetic mouse with known exponential growth:
    T(t) = 100 * exp(0.1 * t), 6 points in [0, 10]."""
    t = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    T = 100.0 * np.exp(0.1 * t)
    return np.column_stack((t, T))


@pytest.fixture
def synthetic_full_data(synthetic_mouse_data):
    """Minimal full_data with one group of two mice."""
    return {
        'test_group': [
            {'name': 'm1', 'data': synthetic_mouse_data,
             'treatment': 'synthetic', 'treatment_days': np.array([], dtype=int),
             'panel': 'X', 'cohort': 'X'},
            {'name': 'm2', 'data': synthetic_mouse_data * np.array([1.0, 1.5]),
             'treatment': 'synthetic', 'treatment_days': np.array([], dtype=int),
             'panel': 'X', 'cohort': 'X'},
        ]
    }
