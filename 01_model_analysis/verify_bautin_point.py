"""Audit of the Bautin point at 50 digits, with exact symbolic derivatives.

Independent of the figure script: beta_1* comes from the EXACT root of the quadratic Q (not
from brentq on a grid), the eigenvectors come from linear systems solved in mpmath (not from
numpy.linalg.eig), and B and C come from exact sp.diff (not from finite differences).
Exits with status 1 if any check fails.
"""
import sys

import sympy as sp
from mpmath import mp

mp.dps = 50
R = sp.Rational

a1 = R(69, 100); K1 = K2 = R(10)**9; mu1 = R(10)**-7
a2 = R(10)**-7; k2 = R(34, 100); K3 = 2*R(10)**7; a3 = R(9, 10)
T, M, C, b1, k3 = sp.symbols('T M C b1 k3', positive=True)
F = sp.Matrix([a1*T*(1 - T/K1)*(1 + b1*M) - mu1*C*T,
               a2*M*T*(1 - M/K2) - k2*M,
               a3*C*T/(K3 + T) - k3*C])
X = (T, M, C)
Ts = k3*K3/(a3 - k3); Ms = K2*(1 - k2/(a2*Ts)); Cs = a1/mu1*(1 - Ts/K1)*(1 + b1*Ms)
SUB = {T: Ts, M: Ms, C: Cs}
J = F.jacobian(X).subs(SUB)
q1e = sp.simplify(-(J[0, 0] + J[1, 1]))
q2e = sp.simplify(J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0] - J[0, 2]*J[2, 0])
q3e = sp.simplify(J[0, 2]*J[1, 1]*J[2, 0])
Qe = sp.expand(q1e*q2e - q3e)

# exact derivatives of the field, BEFORE substituting the equilibrium
D2e = [[[sp.diff(F[i], X[j], X[k]) for k in range(3)] for j in range(3)] for i in range(3)]
D3e = [[[[sp.diff(F[i], X[j], X[k], X[m]) for m in range(3)] for k in range(3)]
        for j in range(3)] for i in range(3)]


def mpc_of(e):
    v = sp.N(e, mp.dps)
    return mp.mpc(str(sp.re(v)), str(sp.im(v)))


def l1_exact(kv):
    """l1 (in cell units) at the Hopf point of the section kappa_3 = kv, at 50 digits."""
    Qp = sp.Poly(Qe.subs(k3, kv), b1)
    roots = sorted([r for r in sp.solve(sp.Eq(Qp.as_expr(), 0), b1) if r.is_real and r > 0])
    bs = roots[0]
    s = {k3: kv, b1: bs}
    pt = {T: Ts.subs(s), M: Ms.subs(s), C: Cs.subs(s)}
    s2 = dict(s); s2.update(pt)

    A = mp.matrix(3, 3)
    Jn = F.jacobian(X).subs(s2)
    for i in range(3):
        for j in range(3):
            A[i, j] = mpc_of(Jn[i, j])
    w = mp.sqrt(mp.mpf(str(sp.N(q2e.subs(s), mp.dps))))        # omega^2 = q2 at the threshold

    def kernel(Mat):
        """eigenvector of a singular 3x3 matrix, fixing the third component at 1."""
        S = mp.matrix([[Mat[0, 0], Mat[0, 1]], [Mat[1, 0], Mat[1, 1]]])
        r = mp.lu_solve(S, mp.matrix([-Mat[0, 2], -Mat[1, 2]]))
        return mp.matrix([r[0], r[1], mp.mpc(1)])

    I3 = mp.eye(3)
    q = kernel(A - 1j*w*I3)
    q = q/mp.sqrt(sum(abs(x)**2 for x in q))
    p = kernel(A.T + 1j*w*I3)
    p = p/mp.conj(sum(mp.conj(p[i])*q[i] for i in range(3)))

    D2 = [[[mpc_of(D2e[i][j][k].subs(s2)) for k in range(3)] for j in range(3)] for i in range(3)]
    D3 = [[[[mpc_of(D3e[i][j][k][m].subs(s2)) for m in range(3)] for k in range(3)]
           for j in range(3)] for i in range(3)]
    B = lambda x, y: mp.matrix([sum(D2[i][j][k]*x[j]*y[k] for j in range(3) for k in range(3))
                                for i in range(3)])
    Cf = lambda x, y, z: mp.matrix([sum(D3[i][j][k][m]*x[j]*y[k]*z[m]
                                        for j in range(3) for k in range(3) for m in range(3))
                                    for i in range(3)])
    herm = lambda x, y: sum(mp.conj(x[i])*y[i] for i in range(3))
    qb = mp.matrix([mp.conj(x) for x in q])
    h11 = mp.lu_solve(A, -B(q, qb))
    h20 = mp.lu_solve(2j*w*I3 - A, B(q, q))
    g21 = herm(p, Cf(q, q, qb)) + 2*herm(p, B(q, h11)) + herm(p, B(qb, h20))
    res = max(abs(x) for x in (A*q - 1j*w*q))
    return g21.real/(2*w), float(mp.nstr(res, 3).replace("e", "E") or 0), w


print("=" * 78)
print("BLOCK 5  -  the Bautin point, at 50 digits and by an independent route")
print("=" * 78)
for kv, rot in [(R(17, 100), "0.17"), (R(1776, 10000), "0.1776"), (R(19, 100), "0.19"),
                (R(35, 100), "0.35")]:
    v, res, w = l1_exact(kv)
    print(f"  kappa_3 = {rot:7s}  l1 = {mp.nstr(v, 12):>22s} cells^-2   "
          f"K1^2 l1 = {mp.nstr(v*mp.mpf(10)**18, 10):>14s}   eigenvector residual {res:.1e}")

print()
f = lambda x: l1_exact(sp.nsimplify(sp.Rational(str(round(float(x), 12)))))[0]
kB = mp.findroot(lambda x: l1_exact(sp.Rational(str(round(float(x), 12))))[0], mp.mpf("0.1776"),
                 tol=1e-20)
print(f"  Bautin point (l1 = 0) at 50 digits: kappa_3^B = {mp.nstr(kB, 15)}")
print()
ok1 = abs(kB - mp.mpf("0.178")) < mp.mpf("0.0005")
ok2 = l1_exact(R(17, 100))[0] > 0
ok3 = l1_exact(R(19, 100))[0] < 0
ok4 = l1_exact(R(35, 100))[0] < 0
CHECKS = [("P2/caption: kappa_3^B ~= 0.178 day^-1", ok1, mp.nstr(kB, 8)),
          ("P2: l1 > 0 (subcritical) below the Bautin point, at kappa_3 = 0.17", ok2, ""),
          ("P2: l1 < 0 (supercritical) above it, at kappa_3 = 0.19", ok3, ""),
          ("P1: l1 < 0 at kappa_3 = 0.35, hence supercritical", ok4, "")]
for name, c, det in CHECKS:
    print(("  [OK]   " if c else "  [FAIL] ") + name + ("   (" + det + ")" if det else ""))

print()
print("  comparison with the value the figure script prints (0.17758):")
print(f"    relative difference = {float(abs(kB - mp.mpf('0.17758'))/kB):.2e}")

BAD = [name for name, c, _ in CHECKS if not c]
print()
print("=" * 78)
print("SUMMARY sweep 5: " + str(len(CHECKS)) + " checks, " + str(len(BAD)) + " failures")
for b in BAD:
    print("  FAIL: " + b)
print("=" * 78)
sys.exit(1 if BAD else 0)
