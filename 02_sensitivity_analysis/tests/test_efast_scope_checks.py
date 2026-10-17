"""The two scope checks of Section 3.3.3: the horizon sweep and the design over the Table 3.1 ranges.

What is checked here is what the thesis asserts about them, namely that the designs are the ones
described (the Table 3.1 intervals where the table declares one, +-20% around the point value
otherwise, beta1 in log10 and mu3 present), that the integrator of the eleven-parameter model agrees
with `scipy.integrate.odeint`, and that the two qualitative claims hold, that is, that the index of
beta1 grows with the horizon and that over the declared ranges the tumor burden saturates.
"""
import os
import sys

import numpy as np
from SALib.analyze import fast
from SALib.sample import fast_sampler

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE)

import efast_horizon as horizon  # noqa: E402
import efast_sensitivity as efast  # noqa: E402
import efast_sensitivity_table31 as wide  # noqa: E402

N_TEST = 70          # eFAST requires N > 4 M^2; the chapter uses N = 10,000


# --------------------------------------------------------------- Table 3.1 design
def test_bounds_follow_table_3_1():
    """Declared intervals where Table 3.1 gives one, +-20% around the point value otherwise."""
    b = dict(zip(wide.PARAM_NAMES, wide.BOUNDS))
    assert b['alpha1'] == [0.69, 0.97]                       # declared
    assert b['kappa3'] == [0.0, 0.9]                         # declared
    assert b['mu3'] == [0.0, 5e-9]                           # declared
    assert b['log10_beta1'][1] == np.log10(3e-6)             # declared ceiling, in log10
    assert b['log10_beta1'][0] == -11.0                      # floor of the log scale
    for name, point in [('alpha2', 1e-7), ('alpha3', 0.9), ('K1', 1e9), ('K2', 1e9),
                        ('K3', 2e7), ('mu1', 1e-7), ('kappa2', 0.34)]:
        lo, hi = b[name]
        assert np.isclose(lo, 0.8 * point) and np.isclose(hi, 1.2 * point), name


def test_beta1_is_sampled_log_uniformly_over_five_decades():
    X = wide.to_model(fast_sampler.sample(wide.problem, N_TEST, M=wide.M_FAST, seed=wide.SEED))
    b1 = X[:, wide.PARAM_NAMES.index('log10_beta1')]
    assert b1.min() >= 1e-11 and b1.max() <= 3e-6
    assert np.log10(b1.max() / b1.min()) > 5
    # the Hopf threshold of Section 3.3.2 lies inside the sampled range, which the +-20% design
    # brackets but the uniform alternative would effectively never visit
    assert b1.min() < 5.9056e-11 < b1.max()


def test_eleven_parameter_model_includes_the_immunosuppression_term():
    """mu3 enters as -mu3 C T, so raising it must lower the CAR T-cell burden."""
    p = np.array([0.69, 1e-7, 0.9, 5.9e-11, 1e9, 1e9, 2e7, 1e-7, 0.34, 0.35, 0.0])
    t_out = np.array([30.0, 60.0])
    y0 = np.array(wide.TMC0)
    c_without = wide.solve_all(y0, np.ascontiguousarray(p[None, :]), t_out,
                               wide.RTOL, wide.ATOL)[0, -1, 2]
    p_with = p.copy()
    p_with[10] = 5e-9
    c_with = wide.solve_all(y0, np.ascontiguousarray(p_with[None, :]), t_out,
                            wide.RTOL, wide.ATOL)[0, -1, 2]
    assert c_with < c_without


def test_compiled_solver_matches_odeint():
    X = wide.to_model(fast_sampler.sample(wide.problem, N_TEST, M=wide.M_FAST, seed=wide.SEED))
    assert wide.check_against_odeint(X, n=8) < 1e-4


def test_declared_ranges_saturate_the_tumor_burden():
    """The reason the chapter reports this design only qualitatively."""
    X = wide.to_model(fast_sampler.sample(wide.problem, N_TEST, M=wide.M_FAST, seed=wide.SEED))
    Y = wide.solve_all(np.array(wide.TMC0), X, np.array([60.0]), wide.RTOL, wide.ATOL)
    T60 = Y[:, -1, 0]
    saturated = np.mean(T60 > 0.99 * X[:, wide.PARAM_NAMES.index('K1')])
    assert saturated > 0.5, saturated


# --------------------------------------------------------------- horizon sweep
def test_horizon_sweep_keeps_the_chapter_design():
    assert horizon.HORIZONS[0] == 60.0 and horizon.HORIZONS[-1] == 400.0
    assert horizon.design.problem is efast.problem


def test_oscillatory_mask_agrees_with_the_eigenvalues():
    """Q < 0 must mark exactly the samples whose Jacobian at E* has a positive real part."""
    X = fast_sampler.sample(efast.problem, N_TEST, M=efast.M_fast, seed=efast.seed)
    mask = horizon.oscillatory_mask(X)
    names = efast.param_names
    for i in range(0, X.shape[0], 37):
        a1, a2, a3, b1, K1, K2, K3, u1, k2, k3 = (X[i, names.index(n)] for n in
                                                  ['alpha1', 'alpha2', 'alpha3', 'beta1', 'K1',
                                                   'K2', 'K3', 'mu1', 'kappa2', 'kappa3'])
        Ts = k3 * K3 / (a3 - k3)
        Ms = K2 * (1 - k2 / (a2 * Ts))
        Cs = a1 / u1 * (1 - Ts / K1) * (1 + b1 * Ms)
        J = np.array([
            [a1 * (1 - 2 * Ts / K1) * (1 + b1 * Ms) - u1 * Cs, a1 * Ts * (1 - Ts / K1) * b1, -u1 * Ts],
            [a2 * Ms * (1 - Ms / K2), a2 * Ts * (1 - 2 * Ms / K2) - k2, 0.0],
            [Cs * a3 * K3 / (K3 + Ts) ** 2, 0.0, 0.0],
        ])
        assert bool(mask[i]) == bool(np.max(np.linalg.eigvals(J).real) > 0)


def test_index_of_beta1_grows_with_the_horizon():
    """The claim of Section 3.3.3: sixty days is too short for beta1 to act."""
    X = np.ascontiguousarray(fast_sampler.sample(efast.problem, 200, M=efast.M_fast,
                                                 seed=efast.seed))
    t_out = np.ascontiguousarray(np.array([60.0, 400.0]))
    Y = efast.solve_all(np.array(efast.TMC0), X, t_out, efast.RTOL, efast.ATOL)
    i_b1 = efast.param_names.index('beta1')
    st = []
    for k in range(2):
        si = fast.analyze(efast.problem, Y[:, k, 0], M=efast.M_fast, num_resamples=2,
                          print_to_console=False, seed=efast.seed)
        st.append(float(np.asarray(si['ST'])[i_b1]))
    assert st[1] > 10 * st[0], st
