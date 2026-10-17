"""Sweep 3: numbers of the Hopf block that had not yet been checked.

Everything in exact rationals / mpmath at 50 digits, without reusing the floating-point path.
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


a1 = R(69, 100); K1 = K2 = R(10)**9; mu1 = R(10)**-7
a2 = R(10)**-7; k2 = R(34, 100); K3 = 2*R(10)**7; a3 = R(9, 10)
T, M, C, b1, k3 = sp.symbols('T M C b1 k3', positive=True)
F = sp.Matrix([a1*T*(1 - T/K1)*(1 + b1*M) - mu1*C*T,
               a2*M*T*(1 - M/K2) - k2*M,
               a3*C*T/(K3 + T) - k3*C])
X = (T, M, C)
Ts = k3*K3/(a3 - k3); Ms = K2*(1 - k2/(a2*Ts)); Cs = a1/mu1*(1 - Ts/K1)*(1 + b1*Ms)
J = F.jacobian(X).subs({T: Ts, M: Ms, C: Cs})
q1 = sp.simplify(-(J[0, 0] + J[1, 1]))
q2 = sp.simplify(J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0] - J[0, 2]*J[2, 0])
q3 = sp.simplify(J[0, 2]*J[1, 1]*J[2, 0])
Qe = sp.expand(q1*q2 - q3)


def positive_roots(kv):
    p = sp.Poly(Qe.subs(k3, kv), b1)
    rr = sp.solve(sp.Eq(p.as_expr(), 0), b1)
    return sorted([mp.mpf(str(sp.N(r, 40))) for r in rr if r.is_real and r > 0])


print("=" * 78)
print("BLOCK 6  -  caption of Figure 3.9 (fig7.pdf), the section kappa_3 = 0.64")
print("=" * 78)
r64 = positive_roots(R(64, 100))
print("  positive roots of Q at kappa_3 = 0.64: " +
      ", ".join(mp.nstr(x, 8) for x in r64))
check("caption 3.9: two roots at kappa_3 = 0.64", len(r64) == 2)
if len(r64) == 2:
    check("caption 3.9: smaller root ~= 2.65e-9",
          abs(r64[0] - mp.mpf("2.65e-9"))/r64[0] < 5e-3, "(" + mp.nstr(r64[0], 6) + ")")
    check("caption 3.9: larger root ~= 1.72e-7",
          abs(r64[1] - mp.mpf("1.72e-7"))/r64[1] < 5e-3, "(" + mp.nstr(r64[1], 6) + ")")

for kv, name, expected in [(R(35, 100), "0.35", 1), (R(55, 100), "0.55", 1),
                           (R(70, 100), "0.70", 0)]:
    rr = positive_roots(kv)
    check("caption 3.9: kappa_3 = " + name + " has " + str(expected) + " positive root(s)",
          len(rr) == expected, "(" + str(len(rr)) + ")")

print()
print("=" * 78)
print("BLOCK 7  -  omega and the period, in the section kappa_3 = 0.35")
print("=" * 78)
K35 = R(35, 100)
bs = sorted([r for r in sp.solve(sp.Eq(sp.Poly(Qe.subs(k3, K35), b1).as_expr(), 0), b1)
             if r.is_real and r > 0])[0]
w = mp.sqrt(mp.mpf(str(sp.N(q2.subs({k3: K35, b1: bs}), 50))))
print("  omega = sqrt(q2) at the threshold = " + mp.nstr(w, 16) + " day^-1")
print("  2 pi / omega = " + mp.nstr(2*mp.pi/w, 10) + " days")
check("text: omega rounds to 0.388 day^-1", abs(w - mp.mpf("0.388")) < mp.mpf("5e-4"))
check("text: 2 pi / omega = 16.19 days (caption / closing paragraph)",
      abs(2*mp.pi/w - mp.mpf("16.19")) < mp.mpf("0.005"), "(" + mp.nstr(2*mp.pi/w, 6) + ")")
check("text: 2 pi / 0.388 ~= 16.2 days (paragraph 'The second concerns')",
      abs(2*mp.pi/mp.mpf("0.388") - mp.mpf("16.2")) < mp.mpf("0.05"))

print()
print("=" * 78)
print("BLOCK 8  -  Table 3.1: the ceiling of beta_1 quoted in the caption of Figure 3.9")
print("=" * 78)
print("  (to be checked by hand in the .tex; here only the order of magnitude of the roots above)")

print()
print("=" * 78)
print("SUMMARY sweep 3: " + str(len(OK)) + " checks, " + str(len(BAD)) + " failures")
for b in BAD:
    print("  FAIL: " + b)
print("=" * 78)
sys.exit(1 if BAD else 0)
