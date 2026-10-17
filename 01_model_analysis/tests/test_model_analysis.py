"""Analytical results of Section 3.3: equilibria, CAR T-cell expansion condition and Hopf bifurcation.

The reference values are those of Table 3.1 and of the text; `verify_chapter3_computations.py` is the
script that recomputes them, and is exercised here as a whole.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE)

import verify_chapter3_computations as chapter  # noqa: E402

BETA1_STAR = 5.9056e-11      # Hopf threshold reported in Section 3.3.2 [1/cell]


def _coexistence_equilibrium(beta1):
    """E* = (T*, M*, C*) of Section 3.3.2, for mu3 = 0."""
    T = chapter.k3 * chapter.K3 / (chapter.a3 - chapter.k3)
    M = chapter.K2 * (1 - chapter.k2 * (chapter.a3 - chapter.k3) / (chapter.k3 * chapter.a2 * chapter.K3))
    C = (chapter.a1 / chapter.u1) * (1 - T / chapter.K1) * (1 + beta1 * M)
    return T, M, C


def _relative_amplitude(beta1, t_end=3000.0, window=1000.0):
    """Peak-to-peak amplitude of T(t) over the last `window` days, relative to its mean."""
    T0, M0, C0 = _coexistence_equilibrium(beta1)
    sol = solve_ivp(chapter.rhs, (0.0, t_end), [1.05 * T0, M0, C0], args=(beta1, 0.0),
                    method='LSODA', rtol=1e-9, atol=1e-2,
                    t_eval=np.linspace(t_end - window, t_end, 2000))
    T = sol.y[0]
    return (T.max() - T.min()) / T.mean()


def test_verification_script_reproduces_every_value_of_the_chapter():
    out = subprocess.run([sys.executable, 'verify_chapter3_computations.py'], cwd=STAGE,
                         capture_output=True, text=True, check=True).stdout
    checked = [line for line in out.splitlines() if '(reported' in line]
    assert len(checked) == 7, out
    assert all(line.rstrip().endswith('OK') for line in checked), out


def test_g3_is_the_per_cell_net_expansion_rate():
    T = np.array([1e5, 1e6, 1e7, 1e8])
    assert np.allclose(chapter.g3(T), (chapter.f3(T) - chapter.k3) / T)


def test_car_t_cells_expand_only_above_a_minimum_tumor_size():
    """dC/dt > 0 requires f3(T) > kappa3, which happens above T = K3 kappa3 / (alpha3 - kappa3)."""
    T_min = chapter.k3 * chapter.K3 / (chapter.a3 - chapter.k3)
    assert chapter.f3(T_min) == pytest.approx(chapter.k3, rel=1e-12)
    assert chapter.f3(0.9 * T_min) < chapter.k3
    assert chapter.f3(1.1 * T_min) > chapter.k3


def test_coexistence_equilibrium_annihilates_the_vector_field():
    beta1 = BETA1_STAR
    state = _coexistence_equilibrium(beta1)
    derivative = chapter.rhs(0.0, list(state), beta1, 0.0)
    scale = (chapter.a1 * state[0], chapter.k2 * state[1], chapter.k3 * state[2])
    assert np.allclose(np.asarray(derivative) / np.asarray(scale), 0.0, atol=1e-10)


def test_coexistence_equilibrium_exists_for_the_parameters_of_table_3_1():
    """alpha3 must lie between kappa3 (1 + K3/K1) and kappa3 (1 + alpha2 K3 / kappa2)."""
    lower = chapter.k3 * (1 + chapter.K3 / chapter.K1)
    upper = chapter.k3 * (1 + chapter.a2 / chapter.k2 * chapter.K3)
    assert lower < chapter.a3 < upper


def test_oscillations_decay_below_the_hopf_threshold_and_persist_above_it():
    assert _relative_amplitude(0.2 * BETA1_STAR) < 1e-3
    assert _relative_amplitude(5.0 * BETA1_STAR) > 1.0
