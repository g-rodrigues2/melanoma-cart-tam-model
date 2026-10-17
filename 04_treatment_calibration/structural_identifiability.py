"""Local structural identifiability of the T-M-C model.

Method: rank test of the observability-identifiability matrix (Lie derivatives of the output y = T) at a random
rational point (Sedoglavic 2002; the basis of SIAN and StructuralIdentifiability.jl).

A parameter or initial condition is locally identifiable if and only if the corresponding column of the Jacobian
J[k,j] = d(L_f^k h)/d(unknown_j) is not in its kernel. If rank(J) < number of unknowns, the kernel gives the
non-identifiable directions (parameter combinations that leave y = T invariant).

Model (system (3.1) of the thesis):
  T' = a1*T*(1-T/K1)*(1+b1*M) - u1*C*T
  M' = a2*M*T*(1-M/K2) - k2*M
  C' = a3*C*T/(K3+T) - u3*C*T - k3*C
  y  = T   (only observed variable)

Scenarios:
  (A) GENERAL: unknowns a1, K1, b1, u1, a3, K3, k3, u3, T0, M0, C0 (a2, K2, k2 known); the theoretical model.
  (B) CALIBRATED CONFIGURATION: a2, K2, k2 known and M0 = T0/2, C0 = 1 fixed as in _mac_treat;
      unknowns a1, K1, b1, u1, a3, K3, k3, u3, T0.
"""
import random
import sympy as sp

T, M, C = sp.symbols('T M C', positive=True)
a1, K1, b1, u1 = sp.symbols('a1 K1 b1 u1', positive=True)
a2, K2, k2 = sp.symbols('a2 K2 k2', positive=True)
a3, K3, k3, u3 = sp.symbols('a3 K3 k3 u3', positive=True)

fT = a1*T*(1 - T/K1)*(1 + b1*M) - u1*C*T
fM = a2*M*T*(1 - M/K2) - k2*M
fC = a3*C*T/(K3 + T) - u3*C*T - k3*C
F = [fT, fM, fC]
X = [T, M, C]
H = T

ALL_SYMS = [T, M, C, a1, K1, b1, u1, a2, K2, k2, a3, K3, k3, u3]


def lie(g):
    return sum(sp.diff(g, xi)*fi for xi, fi in zip(X, F))


def rand_point(seed):
    random.seed(seed)
    return {s: sp.Rational(random.randint(2, 30), random.randint(1, 7))
            for s in ALL_SYMS}


def run(unknowns, state_subs, label, seed=7):
    pt = rand_point(seed)
    n = len(unknowns)
    max_order = n + 4
    rows = []
    yk = H
    for k in range(max_order + 1):
        expr = yk.subs(state_subs, simultaneous=True) if state_subs else yk
        row = [sp.diff(expr, uj).subs(pt) for uj in unknowns]
        rows.append(row)
        yk = lie(yk)
    J = sp.Matrix(rows)
    r = J.rank()
    print(f"\n=== {label} ===")
    print(f"incognitas ({n}): {[str(u) for u in unknowns]}")
    print(f"derivadas de Lie usadas: 0..{max_order}")
    print(f"rank(J) = {r}   (posto cheio = {n})")
    if r == n:
        print("=> TODAS localmente identificaveis (posto cheio).")
    else:
        print(f"=> NOT identifiable: nullity = {n - r}. Kernel directions:")
        ns = J.nullspace()
        for vec in ns:
            terms = [(unknowns[i], vec[i]) for i in range(n) if vec[i] != 0]
            s = "  ".join(f"{c}*d({u})" for u, c in terms)
            involved = [str(u) for u, _ in terms]
            print(f"   [{s}] = 0   -> envolve: {involved}")
    return r, n


if __name__ == '__main__':
    print("T-M-C model | output: y = T")

    # (A) GERAL: a2,K2,k2 conhecidos; T0,M0,C0 livres
    unk_A = [a1, K1, b1, u1, a3, K3, k3, u3, T, M, C]
    run(unk_A, state_subs=None, label="(A) GERAL (paper): IC livres")

    # (B) calibrated configuration: M0 = T0/2, C0 = 1 fixed
    unk_B = [a1, K1, b1, u1, a3, K3, k3, u3, T]
    run(unk_B, state_subs={M: T/2, C: sp.Integer(1)},
        label="(B) CALIBRATED CONFIGURATION: M0=T0/2, C0=1 fixed")

    # (C) T-M subsystem only (no CAR T-cells, _mac): C = 0
    print("\n--- Subsistema T-M (_mac, C=0) ---")
    fT2 = a1*T*(1 - T/K1)*(1 + b1*M)
    fM2 = a2*M*T*(1 - M/K2) - k2*M
    F2, X2 = [fT2, fM2], [T, M]
    def lie2(g):
        return sum(sp.diff(g, xi)*fi for xi, fi in zip(X2, F2))
    for label, unk, ss in [
        ("(C1) _mac GERAL: T0,M0 livres", [a1, K1, b1, T, M], None),
        ("(C2) _mac PIPELINE: M0=T0/2", [a1, K1, b1, T], {M: T/2}),
    ]:
        pt = rand_point(7)
        n = len(unk); rows = []; yk = H
        for k in range(n + 5):
            expr = yk.subs(ss, simultaneous=True) if ss else yk
            rows.append([sp.diff(expr, uj).subs(pt) for uj in unk])
            yk = lie2(yk)
        J = sp.Matrix(rows); r = J.rank()
        print(f"\n=== {label} ===")
        print(f"incognitas ({n}): {[str(u) for u in unk]}  rank={r} (cheio={n})")
        if r < n:
            for vec in J.nullspace():
                inv = [str(unk[i]) for i in range(n) if vec[i] != 0]
                print(f"   nucleo envolve: {inv}")
        else:
            print("   => all identifiable")
