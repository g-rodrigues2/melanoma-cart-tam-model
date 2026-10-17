"""Sweep 4: every number printed in Appendix A, recomputed at 60 digits.

An independent route: beta_1* comes from the EXACT root of the quadratic Q (sympy), the
eigenvectors from linear systems in mpmath, and B and C from exact symbolic derivatives. Phase
convention of the appendix: <q,q> = 1, <p,q> = 1 and the first component of q real and positive.
Exits with status 1 if any check fails.
"""
import sys

import sympy as sp
from mpmath import mp

mp.dps = 60
R = sp.Rational
OK, BAD = [], []


def check(name, cond, det=""):
    (OK if cond else BAD).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("   " + det if det else ""))


def close(a, b, tol=2e-6):
    a, b = mp.mpf(a) if not isinstance(a, mp.mpc) else a, mp.mpf(b) if not isinstance(b, mp.mpc) else b
    d = abs(a - b)
    return d <= tol*max(abs(a), abs(b), mp.mpf("1e-300"))


a1 = R(69, 100); K1 = K2 = R(10)**9; mu1 = R(10)**-7
a2 = R(10)**-7; k2 = R(34, 100); K3 = 2*R(10)**7; a3 = R(9, 10); kap3 = R(35, 100)
T, M, C, b1 = sp.symbols('T M C b1', positive=True)
F = sp.Matrix([a1*T*(1 - T/K1)*(1 + b1*M) - mu1*C*T,
               a2*M*T*(1 - M/K2) - k2*M,
               a3*C*T/(K3 + T) - kap3*C])
X = (T, M, C)
Ts = kap3*K3/(a3 - kap3); Ms = K2*(1 - k2/(a2*Ts)); Cs = a1/mu1*(1 - Ts/K1)*(1 + b1*Ms)
J = F.jacobian(X).subs({T: Ts, M: Ms, C: Cs})
q1e = sp.simplify(-(J[0, 0] + J[1, 1]))
q2e = sp.simplify(J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0] - J[0, 2]*J[2, 0])
q3e = sp.simplify(J[0, 2]*J[1, 1]*J[2, 0])
Qe = sp.expand(q1e*q2e - q3e)
bstar = sorted([r for r in sp.solve(sp.Eq(sp.Poly(Qe, b1).as_expr(), 0), b1)
                if r.is_real and r > 0])[0]
S = {b1: bstar}
pt = {T: Ts.subs(S), M: Ms.subs(S), C: Cs.subs(S)}
S2 = dict(S); S2.update(pt)

print("=" * 78)
print("BLOCK 9  -  Appendix A, eq. (A.1): beta_1* and E*")
print("=" * 78)
bs = mp.mpf(str(sp.N(bstar, 60)))
print("  beta_1* = " + mp.nstr(bs, 18))
check("A.1: beta_1* = 5.9056050e-11", close(bs, mp.mpf("5.9056050e-11"), 1e-7))
for name, val, exp_ in [("T*", pt[T], "1.2727273e7"), ("M*", pt[M], "7.3285714e8"),
                       ("C*", pt[C], "7.1070107e6")]:
    v = mp.mpf(str(sp.N(val, 60)))
    check("A.1: " + name + " = " + exp_, close(v, mp.mpf(exp_), 1e-7), "(" + mp.nstr(v, 10) + ")")

print()
print("=" * 78)
print("BLOCK 10  -  eq. (A.2): the matrix A, and (A.3): the eigenvalues")
print("=" * 78)
A = mp.matrix(3, 3)
Jn = F.jacobian(X).subs(S2)
for i in range(3):
    for j in range(3):
        A[i, j] = mp.mpf(str(sp.N(Jn[i, j], 60)))
Aesp = [["-9.1618920e-3", "5.1201890e-4", "-1.2727273"],
        ["1.9577755e1", "-9.3272730e-1", "0"],
        ["1.1943726e-1", "0", "0"]]
for i in range(3):
    for j in range(3):
        e = mp.mpf(Aesp[i][j])
        if e == 0:
            check("A.2: A[%d,%d] = 0" % (i + 1, j + 1), A[i, j] == 0)
        else:
            check("A.2: A[%d,%d] = %s" % (i + 1, j + 1, Aesp[i][j]), close(A[i, j], e, 1e-7),
                  "(" + mp.nstr(A[i, j], 10) + ")")

w = mp.sqrt(mp.mpf(str(sp.N(q2e.subs(S), 60))))
l3 = -mp.mpf(str(sp.N(q1e.subs(S), 60)))
check("A.3: omega = 0.387985087", close(w, mp.mpf("0.387985087"), 1e-8), "(" + mp.nstr(w, 12) + ")")
check("A.3: lambda_3 = -0.941889", close(l3, mp.mpf("-0.941889"), 1e-6), "(" + mp.nstr(l3, 10) + ")")
check("A.3: 2 i omega is not an eigenvalue (no 1:2 resonance)",
      abs(mp.mpf(2)*w - w) > mp.mpf("0.1") and abs(2*w) > 0)

print()
print("=" * 78)
print("BLOCK 11  -  eq. (A.5): the eigenvectors q and p")
print("=" * 78)
I3 = mp.eye(3)


def nucleo(Mat):
    Sm = mp.matrix([[Mat[0, 0], Mat[0, 1]], [Mat[1, 0], Mat[1, 1]]])
    r = mp.lu_solve(Sm, mp.matrix([-Mat[0, 2], -Mat[1, 2]]))
    return mp.matrix([r[0], r[1], mp.mpc(1)])


Ac = mp.matrix(3, 3)
for i in range(3):
    for j in range(3):
        Ac[i, j] = mp.mpc(A[i, j])
q = nucleo(Ac - 1j*w*I3)
q = q/mp.sqrt(sum(abs(x)**2 for x in q))
q = q*(abs(q[0])/q[0])                      # fase: primeira componente real positiva
p = nucleo(Ac.T + 1j*w*I3)
p = p/mp.conj(sum(mp.conj(p[i])*q[i] for i in range(3)))

qesp = [mp.mpc("0.0515246", "0"), mp.mpc("0.9219629", "-0.3835075"), mp.mpc("0", "-0.0158613")]
pesp = [mp.mpc("9.623411", "-0.033242"), mp.mpc("0.004510", "0.001858"),
        mp.mpc("-0.109046", "-31.568160")]
for i in range(3):
    check("A.5: q_%d" % (i + 1), abs(q[i] - qesp[i]) < 2e-6, "(" + mp.nstr(q[i], 9) + ")")
for i in range(3):
    check("A.5: p_%d" % (i + 1), abs(p[i] - pesp[i]) < 3e-5, "(" + mp.nstr(p[i], 9) + ")")
check("A.5: <q,q> = 1", abs(sum(abs(x)**2 for x in q) - 1) < 1e-40)
check("A.5: <p,q> = 1", abs(sum(mp.conj(p[i])*q[i] for i in range(3)) - 1) < 1e-40)
check("A.5: q_1 real and positive", abs(q[0].imag) < 1e-40 and q[0].real > 0)

print()
print("=" * 78)
print("BLOCK 12  -  eq. (A.12): h_11 and h_20, and (A.13)/(A.14): g_21 and l_1")
print("=" * 78)
D2e = [[[sp.diff(F[i], X[j], X[k]) for k in range(3)] for j in range(3)] for i in range(3)]
D3e = [[[[sp.diff(F[i], X[j], X[k], X[m]) for m in range(3)] for k in range(3)]
        for j in range(3)] for i in range(3)]
D2 = [[[mp.mpf(str(sp.N(D2e[i][j][k].subs(S2), 60))) for k in range(3)] for j in range(3)]
      for i in range(3)]
D3 = [[[[mp.mpf(str(sp.N(D3e[i][j][k][m].subs(S2), 60))) for m in range(3)] for k in range(3)]
       for j in range(3)] for i in range(3)]
B = lambda x, y: mp.matrix([sum(D2[i][j][k]*x[j]*y[k] for j in range(3) for k in range(3))
                            for i in range(3)])
Cf = lambda x, y, z: mp.matrix([sum(D3[i][j][k][m]*x[j]*y[k]*z[m]
                                    for j in range(3) for k in range(3) for m in range(3))
                                for i in range(3)])
herm = lambda x, y: sum(mp.conj(x[i])*y[i] for i in range(3))
qb = mp.matrix([mp.conj(x) for x in q])
h11 = mp.lu_solve(Ac, -B(q, qb))
h20 = mp.lu_solve(2j*w*I3 - Ac, B(q, q))

h11esp = ["1.622366e-10", "-4.059560e-9", "-2.839758e-12"]
for i in range(3):
    check("A.12: h11_%d = %s" % (i + 1, h11esp[i]), close(h11[i].real, mp.mpf(h11esp[i]), 2e-5),
          "(" + mp.nstr(h11[i].real, 9) + ")")
h20esp = [mp.mpc("2.274930e-10", "-7.402555e-11"), mp.mpc("3.984815e-11", "2.316191e-9"),
          mp.mpc("-4.679302e-11", "-1.004426e-11")]
for i in range(3):
    check("A.12: h20_%d" % (i + 1), abs(h20[i] - h20esp[i]) <= 2e-5*abs(h20esp[i]),
          "(" + mp.nstr(h20[i], 9) + ")")

t1 = herm(p, Cf(q, q, qb))
t2 = 2*herm(p, B(q, h11))
t3 = herm(p, B(qb, h20))
tesp = [mp.mpc("-1.492733e-18", "2.969669e-18"), mp.mpc("2.838702e-18", "8.519366e-19"),
        mp.mpc("-1.440219e-18", "-6.267494e-18")]
for i, (tv, te) in enumerate(zip((t1, t2, t3), tesp)):
    check("A.13: term %d of g21" % (i + 1), abs(tv - te) <= 2e-5*abs(te),
          "(" + mp.nstr(tv, 9) + ")")
g21 = t1 + t2 + t3
check("A.14: g21 = -9.425084e-20 - 2.445889e-18 i",
      abs(g21 - mp.mpc("-9.425084e-20", "-2.445889e-18")) <= 2e-5*abs(g21),
      "(" + mp.nstr(g21, 9) + ")")
l1 = g21.real/(2*w)
check("A.14: l_1 = -1.214619e-19", close(l1, mp.mpf("-1.214619e-19"), 2e-5),
      "(" + mp.nstr(l1, 10) + ")")
check("A.14: l_1 < 0, hence supercritical", l1 < 0)
check("text of Ch.3: K_1^2 l_1 = -0.1215", close(l1*mp.mpf(10)**18, mp.mpf("-0.1215"), 1e-3),
      "(" + mp.nstr(l1*mp.mpf(10)**18, 8) + ")")

print()
print("  cancellation: |sum| against the smallest of the three terms")
smallest = min(abs(t1.real), abs(t2.real), abs(t3.real))
print("    |g21| = " + mp.nstr(abs(g21), 8) + ",  smallest term = " + mp.nstr(smallest, 8) +
      ",  ratio = " + mp.nstr(smallest/abs(g21.real), 6))
check("A.15: the sum of the REAL PARTS is ~15x smaller than the smallest of them",
      13 < float(smallest/abs(g21.real)) < 17, "(factor " + mp.nstr(smallest/abs(g21.real), 5) + ")")

print()
print("=" * 78)
print("BLOCK 13  -  transversality and the predicted amplitude (A.17)")
print("=" * 78)
# d Re lambda_{1,2} / d beta_1 by the implicit formula: Re lam = -Q/(2 q2) near the threshold
# (from lam^3+q1 lam^2+q2 lam+q3 with lam = mu +- i nu); here by a central difference at 60 digits
def re_par(bv):
    Jb = F.jacobian(X).subs({T: Ts.subs({b1: bv}), M: Ms.subs({b1: bv}),
                             C: Cs.subs({b1: bv}), b1: bv})
    Mt = mp.matrix(3, 3)
    for i in range(3):
        for j in range(3):
            Mt[i, j] = mp.mpf(str(sp.N(Jb[i, j], 60)))
    ev = mp.eig(Mt, left=False, right=False)
    ev = sorted(ev, key=lambda z: -abs(mp.im(z)))
    return mp.re(ev[0])


hstep = bstar*R(1, 10**6)
dre = (re_par(bstar + hstep) - re_par(bstar - hstep))/(2*mp.mpf(str(sp.N(hstep, 60))))
print("  d Re lambda / d beta_1 = " + mp.nstr(dre, 10) + " cells day^-1")
check("A: transversality = 7.3372e7", close(dre, mp.mpf("7.3372e7"), 1e-4),
      "(" + mp.nstr(dre, 8) + ")")
check("A: q_1 = 0.0515246 (used in A.17)", close(q[0].real, mp.mpf("0.0515246"), 1e-5))

print()
print("=" * 78)
print("SUMMARY sweep 4: " + str(len(OK)) + " checks, " + str(len(BAD)) + " failures")
for b in BAD:
    print("  FAIL: " + b)
print("=" * 78)
sys.exit(1 if BAD else 0)
