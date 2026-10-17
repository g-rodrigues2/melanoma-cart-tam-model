"""Independent verification of the computational results of Chapter 3 (T-M-C model).

Recomputes from scratch, in cell units (Table 3.1), the analytical results (eigenvalues, g3, discriminant) and
the numerical values reported in the chapter (mu3max, alpha1_tilde, T_tilde*, the Hopf threshold beta1*,
mu3_dagger of Scenario 1 and the control durations of Scenario 2), and checks that the phase
portraits of Section 3.1.2 do not depend on the integrator.

    python verify_chapter3_computations.py

Requires numpy, scipy and sympy. Every check must print 'OK'.
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Parameters of Table 3.1 (cell units). Regime of the Hopf analysis:
# alpha1 = 0.69 /day, kappa3 = 0.35 /day (as in the Hopf figures and Scenario 1).
# ---------------------------------------------------------------------------
a1 = 0.69       # alpha1  [1/day]
a2 = 1e-7       # alpha2  [1/(cell*day)]
a3 = 0.9        # alpha3  [1/day]
K1 = 1e9        # K1      [cells]
K2 = 1e9        # K2      [cells]
K3 = 2e7        # K3      [cells]
u1 = 1e-7       # mu1     [1/(cell*day)]
k2 = 0.34       # kappa2  [1/day]
k3 = 0.35       # kappa3  [1/day]

# Coexistence equilibrium E* (mu3 = 0) and the last Routh-Hurwitz function on it.
Ts = k3*K3/(a3 - k3)
Ms = K2*(1 - k2*(a3 - k3)/(k3*a2*K3))


def Q(b1):
    Cs = (a1/u1)*(1 - Ts/K1)*(1 + b1*Ms)
    a11 = -(a1/K1)*Ts*(1 + b1*Ms)
    a12 = a1*Ts*(1 - Ts/K1)*b1
    a13 = -u1*Ts
    a21 = a2*Ms*(1 - Ms/K2)
    a22 = -k2 + a2*Ts*(1 - 2*Ms/K2)
    a31 = a3*Cs*K3/(K3 + Ts)**2
    q1 = -(a11 + a22)
    q2 = a11*a22 - a12*a21 - a13*a31
    q3 = a13*a22*a31
    return q1*q2 - q3


def _ok(got, ref, tol=0.02):
    return 'OK' if abs(got - ref) / abs(ref) < tol else '*** DIFFERS ***'


def check(name, got, ref, tol=0.02, fmt='.4e'):
    print(f"  {name:16s} = {format(got, fmt):>12s}   (reported ~{format(ref, fmt)})   {_ok(got, ref, tol)}")


# ===========================================================================
# PART A -- symbolic verification (eigenvalues, g3', discriminant)
# ===========================================================================
def symbolic_part():
    import sympy as sp
    T, M, C = sp.symbols('T M C', positive=True)
    A1, A2, A3, B1, KK1, KK2, KK3, U1, U3, KA2, KA3 = sp.symbols(
        'a1 a2 a3 b1 K1 K2 K3 u1 u3 k2 k3', positive=True)
    F1 = A1*T*(1 - T/KK1)*(1 + B1*M) - U1*C*T
    F2 = A2*M*T*(1 - M/KK2) - KA2*M
    F3 = A3*C*T/(KK3 + T) - U3*C*T - KA3*C
    J = sp.Matrix([[sp.diff(f, x) for x in (T, M, C)] for f in (F1, F2, F3)])

    print("\n[A] Symbolic verification of the eigenvalues and auxiliary functions")
    ev0 = set(map(str, J.subs({T: 0, M: 0, C: 0}).eigenvals()))
    print(f"  E0 eigenvalues: {sorted(ev0)}   (expected a1, -k2, -k3)")

    Ms = KK2*(1 - KA2/(A2*KK1))
    J2 = J.subs({T: KK1, M: Ms, C: 0})
    ev2 = [sp.simplify(e) for e in J2.eigenvals()]
    has_l2 = any(sp.simplify(e - (KA2 - A2*KK1)) == 0 for e in ev2)
    print(f"  E2 has the eigenvalue (k2 - a2*K1), independent of b1?  {has_l2}")

    f3 = A3*T/(KK3 + T)
    num = sp.expand(sp.simplify((sp.diff(f3, T)*T - (f3 - KA3))*(KK3 + T)**2))
    expected = sp.expand((KA3 - A3)*T**2 + 2*KA3*KK3*T + KA3*KK3**2)
    print(f"  numerator of g3'(T) matches?  {sp.simplify(num - expected) == 0}")
    disc = sp.simplify((2*KA3*KK3)**2 - 4*(KA3 - A3)*(KA3*KK3**2))
    print(f"  discriminant = {disc}   (expected 4*a3*k3*K3^2)   "
          f"{sp.simplify(disc - 4*A3*KA3*KK3**2) == 0}")


# ===========================================================================
# PART B -- numerical verification
# ===========================================================================
def f3(T):
    return a3*T/(K3 + T)


def g3(T):
    return (f3(T) - k3)/T


def numerical_part():
    print("\n[B] Numerical verification (values of Table 3.1)")

    # (1) mu3max = g3(T0bar)
    T0bar = np.sqrt(k3)*K3/(np.sqrt(a3) - np.sqrt(k3))
    check('mu3max', g3(T0bar), 6.37e-9)

    # (2) alpha1_tilde
    a1t = 4*a2**2*a3*K1*K3*(a2*K1 - k2)/(k2*(a2*K3 + k2)**2)
    check('alpha1_tilde', a1t, 3.85e4)

    # (3) T_tilde*: root of the cubic Delta_tilde(T) = 0 in (0, K1)
    def Dt(T):
        return T**3 + 2*K3*T**2 + (K3 + 4*a3/a1*K1)*K3*T - 4*a3/a1*K1**2*K3
    Tt = brentq(Dt, 1e3, K1)
    check('T_tilde*', Tt, 3.87e8, tol=0.03)

    # (4) Hopf threshold beta1*: Q(b1) = q1 q2 - q3 = 0, mu3 = 0, at the equilibrium E*
    exists = (a3 > k3*(1 + K3/K1)) and (a3 < k3*(1 + a2/k2*K3))
    print(f"    E*: T*={Ts:.4e}  M*={Ms:.4e}  exists(a3min<a3<a3max)={exists}")
    print(f"    Q(0)={Q(0):.3e} (>0, stable)   Q(1e-9)={Q(1e-9):.3e} (<0, unstable)")
    # ABSOLUTE xtol: the scipy default (2e-12) would allow up to ~3% error on a root of
    # order 6e-11, more than the 2% tolerance of the check itself.
    check('beta1*', brentq(Q, 1e-13, 1e-8, xtol=1e-28, rtol=8.9e-16), 5.9056e-11)


# ===========================================================================
# PART C -- verification by simulation (Scenarios 1 and 2)
# ===========================================================================
def rhs(t, y, b1, u3):
    T, M, C = y
    dT = a1*T*(1 - T/K1)*(1 + b1*M) - u1*C*T
    dM = a2*M*T*(1 - M/K2) - k2*M
    dC = a3*C*T/(K3 + T) - u3*C*T - k3*C
    return [dT, dM, dC]


def T_at(t_end, C0, b1, u3, comp=0, t_eval=None):
    s = solve_ivp(rhs, (0, t_end), [2e6, 1e6, C0], args=(b1, u3),
                  method='LSODA', rtol=1e-9, atol=1e-2, t_eval=t_eval)
    return s


def simulation_part():
    print("\n[C] Verification by simulation of the ODE")

    # (5) mu3_dagger (Scenario 1): transition of T(60) between plateaus, beta1 = 2.3e-10
    b1 = 2.3e-10
    grid = np.linspace(6e-10, 7.5e-10, 60)
    T60 = np.array([T_at(60, 5e5, b1, u).y[0, -1] for u in grid])
    Tmid = (T60.min() + T60.max())/2
    idx = np.where(np.diff(np.sign(T60 - Tmid)) != 0)[0]
    u3d = brentq(lambda u: T_at(60, 5e5, b1, u).y[0, -1] - Tmid,
                 grid[idx[0]], grid[idx[0]+1])
    check('mu3_dagger', u3d, 6.75e-10, tol=0.03)

    # (6) Scenario 2: duration of control (T < 1e2) for C0 = 7e7 and 9e7, mu3 = 6.752e-10
    def control_days(C0, thr=1e2):
        tg = np.linspace(0, 60, 6001)
        T = T_at(60, C0, b1, 6.752e-10, t_eval=tg).y[0]
        below = T < thr
        return (tg[below].max() - tg[below].min()) if below.any() else 0.0
    for C0, ref in [(7e7, 12.4), (9e7, 21.7)]:
        d = control_days(C0)
        print(f"  control C0={C0:.0e}: {d:5.1f} days   (reported {ref})   "
              f"{'OK' if abs(d - ref) < 1.5 else '*** check ***'}")


# ===========================================================================
# PART D -- the phase portraits do not depend on the integrator (Section 3.1.2)
# ===========================================================================
def solver_part():
    """The portraits of Section 3.1.2 are integrated with odeint (LSODA of ODEPACK).
    Here the same two trajectories are recomputed with an implicit (Radau IIA) and an
    explicit (DOP853) scheme at tighter tolerances, to show the limit cycle is not an
    artefact of the solver."""
    print("\n[D] Solver independence of the phase portraits (Section 3.1.2)")
    bstar = brentq(Q, 1e-13, 1e-8, xtol=1e-28, rtol=8.9e-16)
    t = np.linspace(0, 8000.0, 20001)

    def rhs_t(y, t_, b1):            # odeint signature
        return rhs(t_, y, b1, 0.0)

    for b1, lab in [(0.0, 'beta1 = 0        '), (1.05*bstar, 'beta1 = 1.05 beta1*')]:
        ref = odeint(rhs_t, [2e6, 1e6, 1e7], t, args=(b1,),
                     rtol=1e-10, atol=1e-12, mxstep=5000)          # as in the figure
        half = slice(len(t)//2, None)                              # the attractor
        print(f"  {lab}: T in [{ref[half, 0].min():.4e}, {ref[half, 0].max():.4e}], "
              f"M in [{ref[half, 1].min():.4e}, {ref[half, 1].max():.4e}]")
        for meth in ('Radau', 'DOP853'):
            s = solve_ivp(rhs, (0, t[-1]), [2e6, 1e6, 1e7], args=(b1, 0.0),
                          method=meth, rtol=1e-12, atol=1e-14, t_eval=t).y.T
            d = (np.abs(s - ref) / np.maximum(np.abs(ref), 1.0)).max()
            print(f"    vs solve_ivp {meth:7s} (rtol 1e-12): max rel. diff. = {d:.2e}   "
                  f"{'OK' if d < 1e-5 else '*** DIFFERS ***'}")


if __name__ == '__main__':
    print("=" * 70)
    print("VERIFICATION OF THE COMPUTATIONAL RESULTS -- Chapter 3 (T-M-C model)")
    print("=" * 70)
    symbolic_part()
    numerical_part()
    simulation_part()
    solver_part()
    print("\nDone. Every item must print OK.")
