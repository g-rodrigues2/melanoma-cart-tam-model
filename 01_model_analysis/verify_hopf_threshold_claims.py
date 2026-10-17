"""Independent audit of every numerical claim of this subsection that entered the thesis.

Independence: everything here is done with EXACT sympy (rationals) and mpmath at 50 digits,
without reusing the floating-point path that produced the original numbers.
Exits with status 1 if any check fails.
"""
import sys

import sympy as sp
from mpmath import mp

mp.dps = 50
R = sp.Rational
OK, BAD = [], []


def check(name, cond, det=""):
    (OK if cond else BAD).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("   " + det if det else ""))


# EXACT parameters of Table 3.1
a1 = R(69, 100)
K1 = K2 = R(10)**9
mu1 = R(10)**-7
a2 = R(10)**-7
k2 = R(34, 100)
K3 = 2*R(10)**7
a3 = R(9, 10)

T, M, C, b1, k3 = sp.symbols('T M C b1 k3', positive=True)
F = sp.Matrix([a1*T*(1 - T/K1)*(1 + b1*M) - mu1*C*T,
               a2*M*T*(1 - M/K2) - k2*M,
               a3*C*T/(K3 + T) - k3*C])
X = (T, M, C)

# coexistence equilibrium E*, exact
Ts = k3*K3/(a3 - k3)
Ms = K2*(1 - k2/(a2*Ts))
Cs = a1/mu1*(1 - Ts/K1)*(1 + b1*Ms)
sub = {T: Ts, M: Ms, C: Cs}

J = sp.simplify(F.jacobian(X).subs(sub))
a11, a12, a13 = J[0, 0], J[0, 1], J[0, 2]
a21, a22, a31 = J[1, 0], J[1, 1], J[2, 0]
q1 = sp.simplify(-(a11 + a22))
q2 = sp.simplify(a11*a22 - a12*a21 - a13*a31)
q3 = sp.simplify(a13*a22*a31)
Qe = sp.expand(q1*q2 - q3)

print("=" * 78)
print("BLOCK 1  -  claims about the section kappa_3 = 0.35")
print("=" * 78)

K35 = R(35, 100)
Q35 = sp.Poly(Qe.subs(k3, K35), b1)
roots = sorted([r for r in sp.solve(sp.Eq(Q35.as_expr(), 0), b1) if r.is_real and r > 0])
bs = roots[0]
bs50 = mp.mpf(str(sp.N(bs, 50)))
print(f"  beta_1* (50 digits) = {mp.nstr(bs50, 20)}")
check("P1/caption: beta_1* ~= 5.9056e-11 cells^-1",
      abs(bs50 - mp.mpf("5.9056e-11"))/bs50 < 1e-4,
      f"(valor exato {mp.nstr(bs50, 8)})")

# at the Hopf point, omega^2 = q2 and the third eigenvalue is -q1
w2 = sp.N(q2.subs({k3: K35, b1: bs}), 50)
w = mp.sqrt(mp.mpf(str(w2)))
l3 = -mp.mpf(str(sp.N(q1.subs({k3: K35, b1: bs}), 50)))
print(f"  omega = {mp.nstr(w, 20)}   lambda_3 = {mp.nstr(l3, 20)}")
check("P1: lambda_{1,2} = +-0.388 i day^-1", abs(w - mp.mpf("0.388")) < mp.mpf("0.0005"),
      f"(omega = {mp.nstr(w, 8)}, rounds to 0.388)")
check("P1: lambda_3 = -0.942 day^-1", abs(l3 + mp.mpf("0.942")) < mp.mpf("0.0005"),
      f"(lambda_3 = {mp.nstr(l3, 8)}, rounds to -0.942)")

# independent check: does the characteristic polynomial factor as (lam+q1)(lam^2+q2)?
lam = sp.symbols('lam')
charp = sp.expand(lam**3 + q1*lam**2 + q2*lam + q3).subs({k3: K35, b1: bs})
fator = sp.expand((lam + q1.subs({k3: K35, b1: bs}))*(lam**2 + q2.subs({k3: K35, b1: bs})))
check("P1: at the threshold the cubic factors as (lam + q1)(lam^2 + q2), hence a purely imaginary pair",
      sp.simplify(charp - fator) == 0)

per = 2*mp.pi/w
check("P1: period 2pi/0.388 ~= 16.2 days", abs(per - mp.mpf("16.2")) < mp.mpf("0.05"),
      f"(2pi/omega = {mp.nstr(per, 6)} days)")
per_texto = 2*mp.pi/mp.mpf("0.388")
check("P1: the value 2pi/0.388 written in the text also gives 16.2",
      abs(per_texto - mp.mpf("16.2")) < mp.mpf("0.05"), f"({mp.nstr(per_texto, 6)})")

# transversality: d(Re lambda)/d beta_1 != 0 at the threshold
dQ = sp.diff(Qe.subs(k3, K35), b1)
dQ_num = mp.mpf(str(sp.N(dQ.subs(b1, bs), 50)))
check("P1: the pair crosses with nonzero speed (dQ/dbeta_1 != 0 at the threshold)",
      abs(dQ_num) > 0, f"(dQ/dbeta_1 = {mp.nstr(dQ_num, 6)})")

print()
print("=" * 78)
print("BLOCK 2  -  structural claims about Q(beta_1)")
print("=" * 78)
Qpoly = sp.Poly(Qe, b1)
check("P2/caption: Q(beta_1) is an EXACTLY quadratic polynomial in beta_1",
      Qpoly.degree() == 2, f"(grau {Qpoly.degree()})")
c2, c1, c0 = Qpoly.all_coeffs()
Q0 = sp.simplify(c0)
# Q(0) > 0 over the whole existence interval (Proposition P6)
amostras = [R(14, 100), R(2, 10), R(35, 100), R(5, 10), R(65, 100), R(8, 10), R(87, 100)]
signs = [sp.N(Q0.subs(k3, v), 30) > 0 for v in amostras]
check("P2: Q(0) > 0 over the existence interval (basis of the root count)", all(signs))

# kappa_3^(1): sign change of the leading coefficient
f_c2 = sp.lambdify(k3, c2, 'mpmath')
k3c1 = mp.findroot(lambda x: f_c2(x), mp.mpf("0.59"))
print(f"  kappa_3^(1) (root of c2) = {mp.nstr(k3c1, 12)}")
check("P2: the leading coefficient changes sign at kappa_3^(1) ~= 0.590",
      abs(k3c1 - mp.mpf("0.590")) < mp.mpf("0.0005"), f"({mp.nstr(k3c1, 7)})")
check("P2: below kappa_3^(1) the parabola is CONCAVE (c2 < 0)", f_c2(mp.mpf("0.35")) < 0)
check("P2: above kappa_3^(1) the parabola is CONVEX (c2 > 0)", f_c2(mp.mpf("0.63")) > 0)

# count of positive roots in the two regimes
def raizes_pos(kv):
    p = sp.Poly(Qe.subs(k3, kv), b1)
    rr = sp.solve(sp.Eq(p.as_expr(), 0), b1)
    return sorted([mp.mpf(str(sp.N(r, 30))) for r in rr if r.is_real and r > 0])
r_below = raizes_pos(R(35, 100))
r_above = raizes_pos(R(63, 100))
check("P2: below kappa_3^(1) there is ONE positive root (a simple threshold)", len(r_below) == 1,
      f"({len(r_below)} raiz)")
check("P2: above kappa_3^(1) there are TWO positive roots (a closed window)", len(r_above) == 2,
      f"({len(r_above)} roots: {mp.nstr(r_above[0],5)}, {mp.nstr(r_above[1],5)})")

# kappa_3^c: where the minimum of Q over beta_1 >= 0 crosses zero, that is, discriminant = 0
disc = sp.simplify(c1**2 - 4*c2*c0)
f_disc = sp.lambdify(k3, disc, 'mpmath')
k3c = mp.findroot(lambda x: f_disc(x), mp.mpf("0.66"))
print(f"  kappa_3^c (discriminant of Q = 0) = {mp.nstr(k3c, 12)}")
check("P2/caption: the window closes at kappa_3^c ~= 0.660",
      abs(k3c - mp.mpf("0.660")) < mp.mpf("0.0005"), f"({mp.nstr(k3c, 7)})")
check("P2: above kappa_3^c there is no positive root (E* stable for every beta_1 >= 0)",
      len(raizes_pos(R(68, 100))) == 0)

print()
print("=" * 78)
print("BLOCK 3  -  spiral/sink boundary as beta_1 -> 0")
print("=" * 78)
# T~* resolve T^3 + 2 K3 T^2 + (K3 + 4 a3 K1/a1) K3 T - 4 a3 K1^2 K3 / a1 = 0
Tt = sp.symbols('Tt', positive=True)
polyT = Tt**3 + 2*K3*Tt**2 + (K3 + 4*a3/a1*K1)*K3*Tt - 4*a3/a1*K1**2*K3
Ttil = mp.findroot(sp.lambdify(Tt, polyT, 'mpmath'), mp.mpf("3.9e8"))
k3_spiral = a3*Ttil/(K3 + Ttil)         # inverte T* = k3 K3/(a3 - k3)
print(f"  T~* = {mp.nstr(Ttil, 10)}   corresponding kappa_3 = {mp.nstr(k3_spiral, 10)}")
check("P2: the spiral/sink boundary converges to kappa_3 ~= 0.86 as beta_1 -> 0",
      abs(k3_spiral - mp.mpf("0.86")) < mp.mpf("0.005"), f"({mp.nstr(k3_spiral, 7)})")
check("caption Fig 3.7: T~* ~= 3.87e8 cells (value already quoted in the thesis)",
      abs(Ttil - mp.mpf("3.87e8"))/Ttil < 2e-3, f"({mp.nstr(Ttil, 6)})")

print()
print("=" * 78)
print("BLOCK 4  -  the relevance sentence")
print("=" * 78)
k3_min = a3*k2/(k2 + a2*K3)
k3_min50 = mp.mpf(str(sp.N(k3_min, 50)))
print(f"  lower existence limit of E* = {mp.nstr(k3_min50, 12)}")
check("P2: the lower end of the existence interval is kappa_3 ~= 0.131",
      abs(k3_min50 - mp.mpf("0.131")) < mp.mpf("0.0005"), f"({mp.nstr(k3_min50, 7)})")
check("P2: 0.09 lies BELOW that limit (E* does not even exist)", mp.mpf("0.09") < k3_min50)
check("P2: 0.21 lies ABOVE kappa_3^B ~= 0.178", mp.mpf("0.21") > mp.mpf("0.178"))
check("P2: 0.21 also lies above the existence limit", mp.mpf("0.21") > k3_min50)

print()
print("=" * 78)
print(f"SUMMARY: {len(OK)} checks, {len(BAD)} failures")
if BAD:
    for b in BAD:
        print("  FAIL: " + b)
print("=" * 78)
sys.exit(1 if BAD else 0)
