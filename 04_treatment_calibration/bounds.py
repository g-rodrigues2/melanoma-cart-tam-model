"""Bounds of the calibrated parameters of the 18 treatment models.

  - The lower bound of `b_imuno` is 0 in ED/CD/AccelED/AccelCD/EDwith*/CDwith* (the posterior of the selected
    term accumulated at an earlier bound of 0.01).
  - `_mac` calibrates alpha1, K1 and beta1 as in the growth-model selection.
  - `_mac_treat` (full T, M, C system) calibrates alpha1, K1, beta1, mu1, alpha3, K3, kappa3 and mu3; alpha2,
    K2 and kappa2 are fixed in models.py.

`alpha1`, `K1` and `beta1` are sampled in log10. They enter as the product `alpha1*(1-T/K1)*(1+beta1*M)` and are
strongly correlated (many combinations give the same T(t)); with linear coordinates over 2.7-4 orders of
magnitude a uniform prior makes the sampler spend its time in regions of low posterior density (acceptance below
0.10 in a first run of 100k steps). Log-uniform sampling reduces tau and raises the acceptance rate. The
conversion to physical values (10**value) is done in `models.py::_split_theta` only. The other CAR T-cell
parameters (mu1, alpha3, K3, kappa3, mu3) stay linear: their bounds include 0.

Conversion to mm^3 (rho ~ 10^6 cells/mm^3), theoretical values covered by the bounds:
    alpha1: 0.69 /day    (intensivo, invariante)         em [1e-3, 0.5]
    K1:     10^9 cells   -> 1000 mm^3                    em [100, 50000]
    beta1:  2.3e-10/cell -> 2.3e-4 /mm^3                 em [1e-6, 1e-2]
    mu1:    1e-7/cell/day * dose -> ~0.5 with a dose of 5e6,
            or ~0.1/mm^3/day without the dose                        em [0.0, 5.0]
    alpha3: 0.9 /day                                     em [0.0, 5.0]
    K3:     2e7 cells -> 20 mm^3                         em [1, 1e4]
    kappa3: 0.35 /day                                    em [0.0, 2.0]
    mu3:    1e-9/cell/day * dose -> ~5e-3                em [0.0, 5e-2]
"""
import numpy as np

# === General bounds ===
STD_LOW  = 0.0
STD_HIGH = 800.0

# === Tumor parameters (shared by _mac and _mac_treat) ===
# Upper bound of alpha1 reduced from 1.0 to 0.5 to avoid divergent prior predictive trajectories: with
# (1+beta1*M) >= 2, alpha1 = 0.5 already gives an apparent rate well above 1/day.
# Physical values for reference; the sampled bounds are the *_LOG_LO/*_LOG_HI below.
MAC_ALPHA1_LO = 1e-3
MAC_ALPHA1_HI = 0.5
MAC_K1_LO     = 100.0
MAC_K1_HI     = 50000.0
MAC_BETA1_LO  = 1e-6
MAC_BETA1_HI  = 1e-2

# log10 bounds: these define the uniform prior. models.py::_split_theta returns 10**value to the solver.
MAC_ALPHA1_LOG_LO = np.log10(MAC_ALPHA1_LO)
MAC_ALPHA1_LOG_HI = np.log10(MAC_ALPHA1_HI)
MAC_K1_LOG_LO     = np.log10(MAC_K1_LO)
MAC_K1_LOG_HI     = np.log10(MAC_K1_HI)
MAC_BETA1_LOG_LO  = np.log10(MAC_BETA1_LO)
MAC_BETA1_LOG_HI  = np.log10(MAC_BETA1_HI)

# === CAR T-cell parameters (_mac_treat only) ===
CART_MU1_LO    = 0.0
CART_MU1_HI    = 5.0
CART_ALPHA3_LO = 0.0
CART_ALPHA3_HI = 5.0
CART_K3_LO     = 1.0
CART_K3_HI     = 1e4
CART_KAPPA3_LO = 0.0
CART_KAPPA3_HI = 2.0
CART_MU3_LO    = 0.0
CART_MU3_HI    = 5e-2


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


def _exp_treatment_bounds(token):
    """Bounds of the treatment terms added to _exp."""
    if token == 'ED':
        return ["a_imuno", "b_imuno"], [0.0, 0.0], [1.0, 1.5]
    if token == 'CD':
        return ["a_imuno", "b_imuno"], [0.0, 0.0], [1.0, 1.5]
    if token == 'AccelED':
        return ["a_imuno", "b_imuno"], [0.0, 0.0], [1.0, 0.5]
    if token == 'AccelCD':
        return ["a_imuno", "b_imuno"], [0.0, 0.0], [1.0, 0.5]
    if token == 'MM':
        return ["a_imuno", "b_imuno"], [0.0, 0.0], [1.0, 60.0]
    if token == 'LS':
        return ["a_imuno", "b_imuno"], [0.0, 0.0], [1.0, 0.1]
    if token == 'PF':
        return ["c", "lam"],           [0.0, 0.0], [1.0, 1.0]
    if token == 'dePR':
        return ["d", "s", "lam"],      [0.0, 0.0, 0.01], [2.0, 2.0, 2.0]
    if token == 'dePRDecay':
        return ["d", "s", "lam", "k_elim"], [0.0, 0.0, 0.01, 0.0], [2.0, 2.0, 2.0, 0.5]
    if token in ('CDwithLog', 'EDwithLog'):
        return ["a_imuno", "b_imuno", "steepness"], [0.0, 0.0, 0.0], [1.0, 1.5, 1.0]
    if token in ('CDwithExp', 'EDwithExp'):
        return ["a_imuno", "b_imuno", "rate"],      [0.0, 0.0, 0.0], [1.0, 1.5, 1.0]
    if token in ('CDwithSin', 'EDwithSin'):
        return ["a_imuno", "b_imuno", "frequency"], [0.0, 0.0, 0.0], [1.0, 1.5, 1.0]
    raise ValueError(f"Unknown treatment term: {token}")


def define_bounds_labels(model_extension, group_size, max_ic,
                         bool_multiple_mice=False):
    """Returns (l_bound, u_bound, labels)."""
    ic_max = max_ic * 5

    if model_extension == '_exp':
        labels, l_bound, u_bound = [], [], []

    elif model_extension.startswith('_exp_'):
        token = model_extension[len('_exp_'):]
        labels, l_bound, u_bound = _exp_treatment_bounds(token)
        labels, l_bound, u_bound = list(labels), list(l_bound), list(u_bound)

    elif model_extension == '_mac':
        # alpha1, K1 and beta1 sampled in log10 (see the module docstring).
        labels  = ["log10_alpha1",     "log10_K1",     "log10_beta1"]
        l_bound = [MAC_ALPHA1_LOG_LO,  MAC_K1_LOG_LO,  MAC_BETA1_LOG_LO]
        u_bound = [MAC_ALPHA1_LOG_HI,  MAC_K1_LOG_HI,  MAC_BETA1_LOG_HI]

    elif model_extension == '_mac_treat':
        # alpha1, K1, beta1 in log10; mu1, alpha3, K3, kappa3, mu3 linear.
        labels  = ["log10_alpha1",     "log10_K1",     "log10_beta1",     "mu1",
                   "alpha3",           "K3",           "kappa3",          "mu3"]
        l_bound = [MAC_ALPHA1_LOG_LO,  MAC_K1_LOG_LO,  MAC_BETA1_LOG_LO,  CART_MU1_LO,
                   CART_ALPHA3_LO,     CART_K3_LO,     CART_KAPPA3_LO,    CART_MU3_LO]
        u_bound = [MAC_ALPHA1_LOG_HI,  MAC_K1_LOG_HI,  MAC_BETA1_LOG_HI,  CART_MU1_HI,
                   CART_ALPHA3_HI,     CART_K3_HI,     CART_KAPPA3_HI,    CART_MU3_HI]

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
    l_bound.append(STD_LOW)
    u_bound.append(STD_HIGH)

    return l_bound, u_bound, labels
