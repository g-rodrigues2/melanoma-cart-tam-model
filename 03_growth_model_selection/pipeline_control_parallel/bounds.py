"""Bounds (uniform priors) and labels of the parameters of each model.

The upper bound of the initial conditions (``ic_max``) is twice the largest observed T(0) over all mice
(``find_max_initial_condition()``), so that the free initial condition of each mouse covers the scale of the
data without being excessively wide.
"""


def find_max_initial_condition(full_data):
    """Returns (largest observed T0, its group, its mouse name)."""
    largest_ic = -float('inf')
    group_largest_ic = None
    mouse_largest_ic = None
    for group, mice in full_data.items():
        for mouse in mice:
            ic = mouse['data'][0, 1]
            if ic > largest_ic:
                largest_ic = ic
                group_largest_ic = group
                mouse_largest_ic = mouse['name']
    return largest_ic, group_largest_ic, mouse_largest_ic


def define_bounds_labels(model_extension, group_size, max_ic,
                         bool_multiple_mice=False):
    """Returns (l_bound, u_bound, labels) of the model.

    Parameters:
      model_extension     - model suffix (e.g. '_exp', '_mac')
      group_size          - number of mice in the group (initial conditions)
      max_ic              - largest T(0) observed in any group
      bool_multiple_mice  - True: appends ic1..ic{group_size} and std;
                            False: appends 'ic' and 'std'.
    """
    ic_max = max_ic * 2

    if model_extension == '_exp':
        labels  = ["r"]
        l_bound = [0.0]
        u_bound = [1.0]
    elif model_extension == '_mendel':
        labels  = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [1.0, 1.0]
    elif model_extension == '_bert':
        # dT/dt = r*T^(2/3) - b*T, equilibrium T_eq = (r/b)^3.
        # b <= 0.1 avoids the pathology b -> 1 (T -> 0 -> NaN in T^(2/3)).
        # With r = 2 and b = 0.1, T_eq = 8000 mm^3 (covers the scale of the data).
        labels  = ["r", "b"]
        l_bound = [1e-3, 1e-5]
        u_bound = [2.0,  0.1]
    elif model_extension == '_lin':
        # dT/dt = r*T/(T+b); b is the half-saturation volume, on the scale of the data (50-3000 mm^3).
        labels  = ["r", "b"]
        l_bound = [1e-3, 1.0]
        u_bound = [1.0,  5000.0]
    elif model_extension == '_gomp':
        # dT/dt = r*T*ln(b/(T+c)); b > max(T+c) for sustained growth.
        labels  = ["r", "b",     "c"]
        l_bound = [1e-3, 100.0,  0.1]
        u_bound = [1.0,  50000., 500.0]
    elif model_extension == '_surf':
        labels  = ["r", "b"]
        l_bound = [1e-6, 1e-6]
        u_bound = [1.0,  ic_max]
    elif model_extension == '_log':
        labels  = ["r", "cc"]
        l_bound = [0.0,  100.]
        u_bound = [1.0,  20000]
    elif model_extension == '_mac':
        # (T, M) with macrophages; alpha2, K2 and kappa2 fixed on the mm^3 scale (see models.py: rescaling
        # with rho = 10^6 cells/mm^3). beta1 multiplies M in mm^3 ~ [0, K2 = 1000]. Literature value:
        # beta1 ~ 2.3e-4 /mm^3 (= 2.3e-10 /cell x rho). The bounds [1e-6, 1e-2] range from a negligible
        # effect (1 + beta1*M ~ 1) to a tenfold growth rate through TAMs (1 + beta1*M ~ 10).
        labels  = ["alpha1", "K1",    "beta1"]
        l_bound = [1e-3,     100.0,   1e-6]
        u_bound = [1.0,      50000.,  1e-2]
    elif model_extension == '_macAD':
        # Same as the rescaled _mac, kept for a calibration in dimensionless coordinates
        # (alpha_tilde, beta_tilde, kappa_tilde); the physics is currently that of _mac (mm^3).
        labels  = ["alpha1", "K1",    "beta1"]
        l_bound = [1e-3,     100.0,   1e-6]
        u_bound = [1.0,      50000.,  1e-2]
    else:
        raise ValueError(f"Unknown model: {model_extension}")

    if bool_multiple_mice:
        for i in range(group_size):
            labels.append("ic" + str(i + 1))
            l_bound.append(0.0)
            u_bound.append(ic_max)
    else:
        labels.append("ic")
        l_bound.append(0.0)
        u_bound.append(ic_max)

    labels.append("std")
    l_bound.append(0.1)
    u_bound.append(1000.0)

    return l_bound, u_bound, labels
