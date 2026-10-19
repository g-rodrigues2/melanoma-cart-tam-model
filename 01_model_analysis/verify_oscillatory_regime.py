"""Sweep 5: first half of subsection 3.3.2 (oscillatory regime, Figures 3.7 and 3.8).

A route independent of the scripts that draw the figures: the formulas are rewritten from
scratch out of Proposition P7, the roots come from mpmath, and the integration uses DOP853
with tight tolerances instead of the LSODA of the original scripts.
Exits with status 1 if any check fails.
"""
import sys

import numpy as np
from mpmath import mp
from scipy.integrate import solve_ivp

mp.dps = 40
OK, BAD = [], []


def check(name, cond, det=""):
    (OK if cond else BAD).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("   " + det if det else ""))


a1, K1, mu1 = 0.69, 1e9, 1e-7
a2, K2, k2 = 1e-7, 1e9, 0.34
K3, a3 = 2e7, 0.9
T0, M0, C0 = 2e6, 1e6, 1e7          # initial condition of Table 3.1 (C(0) = 1e7, from the interval)

print("=" * 78)
print("BLOCK 14  -  Proposition P7: alpha1_til, T_til*, Omega and the kappa_3 interval")
print("=" * 78)
a1til = 4*a2**2*a3*K1*K3*(a2*K1 - k2)/(k2*(a2*K3 + k2)**2)
print("  alpha1_til = %.6e" % a1til)
check("P7: alpha_1 = 0.69 < alpha1_til, hence the spiral case exists", a1 < a1til)
check("text: alpha1_til ~= 3.85e4", abs(a1til - 3.85e4)/a1til < 2e-3, "(%.5e)" % a1til)

Dt = lambda x: x**3 + 2*K3*x**2 + (K3 + 4*a3/a1*K1)*K3*x - 4*a3/a1*K1**2*K3
Ttil = float(mp.findroot(Dt, mp.mpf("3.9e8")))
print("  T_til* = %.6e" % Ttil)
check("text/caption 3.7: T_til* ~= 3.87e8 cells", abs(Ttil - 3.87e8)/Ttil < 2e-3, "(%.5e)" % Ttil)

Tlo, Thi = k2/a2, K1
check("Omega = (kappa_2/alpha_2, K_1) = (3.40e6, 1e9)", abs(Tlo - 3.4e6) < 1 and Thi == 1e9)
check("P7: T_til* falls inside Omega", Tlo < Ttil < Thi)

k3lo = a3*k2/(k2 + a2*K3)
k3hi = a3*K1/(K1 + K3)
print("  kappa_3 in (%.5f, %.5f)" % (k3lo, k3hi))
check("text: existence interval ~= (0.131, 0.882)",
      abs(k3lo - 0.131) < 5e-4 and abs(k3hi - 0.882) < 5e-4, "(%.5f, %.5f)" % (k3lo, k3hi))

Tstar = lambda k: k*K3/(a3 - k)
from scipy.optimize import brentq as _bq
k3til = _bq(lambda k: Tstar(k) - Ttil, 0.80, 0.88, xtol=1e-14)
print("  kappa_3_til (spiral -> sink) = %.6f" % k3til)
check("caption 3.7: kappa_3_til ~= 0.856", abs(k3til - 0.856) < 5e-4, "(%.5f)" % k3til)
check("caption 3.7: kappa_3 = 0.86 lies 0.5%% above the threshold",
      abs((0.86 - k3til)/k3til - 0.005) < 7e-4, "(%.3f%%)" % (100*(0.86 - k3til)/k3til))
check("text: kappa_3 = 0.5 and 0.86 lie in the existence interval",
      k3lo < 0.5 < k3hi and k3lo < 0.86 < k3hi)
check("text: 0.5 gives a spiral (T* < T_til*) and 0.86 a sink (T* > T_til*)",
      Tstar(0.5) < Ttil < Tstar(0.86),
      "(T*(0.5)=%.3e, T*(0.86)=%.3e)" % (Tstar(0.5), Tstar(0.86)))

print()
print("=" * 78)
print("BLOCK 15  -  caption 3.7(b): cycles until the amplitude halves")
print("=" * 78)


def autovalores(k):
    """(T,C) block of the closed system at beta_1 = 0, as in the proof of P7."""
    Ts = Tstar(k)
    Cs = a1/mu1*(1 - Ts/K1)
    b = (a1/K1)*Ts
    c = mu1*a3*Cs*K3*Ts/(K3 + Ts)**2
    return np.roots([1.0, b, c])


def cycles(k):
    w = autovalores(k)
    if abs(w[0].imag) < 1e-14:
        return np.nan
    periodo = 2*np.pi/abs(w[0].imag)
    meia_vida = np.log(2)/abs(w[0].real)
    return meia_vida/periodo


edge = k3lo + 1e-6
print("  at the lower edge kappa_3 = %.6f: %.2f cycles" % (edge, cycles(edge)))
check("caption 3.7(b): about 25 cycles near the lower edge",
      22 < cycles(edge) < 28, "(%.1f)" % cycles(edge))
from scipy.optimize import brentq
k3prac = brentq(lambda k: cycles(k) - 1.0, 0.2, k3til - 1e-6, xtol=1e-12)
print("  cycles = 1 at kappa_3 = %.6f" % k3prac)
check("caption 3.7(b): the practical threshold lies at kappa_3 ~= 0.74",
      abs(k3prac - 0.74) < 5e-3, "(%.5f)" % k3prac)
check("caption 3.7(b): that threshold comes well before kappa_3_til", k3prac < k3til)

print()
print("=" * 78)
print("BLOCK 16  -  eigenvalues in the two sections and the half-lives quoted")
print("=" * 78)
w5 = autovalores(0.5)
print("  kappa_3 = 0.50: %s" % np.array2string(w5, precision=6))
check("text: |Re lambda| = 0.0086 day^-1 at kappa_3 = 0.5",
      abs(abs(w5[0].real) - 0.0086) < 5e-5, "(%.6f)" % abs(w5[0].real))
check("caption/text: complex pair at kappa_3 = 0.5 (spiral)", abs(w5[0].imag) > 1e-6)
per5 = 2*np.pi/abs(w5[0].imag)
print("  period at kappa_3 = 0.5: %.3f days" % per5)

w86 = autovalores(0.86)
w86 = np.sort(w86.real)
print("  kappa_3 = 0.86: %.6f and %.6f (real)" % (w86[0], w86[1]))
check("text: eigenvalues -0.232 and -0.0648 at kappa_3 = 0.86",
      abs(w86[0] + 0.232) < 5e-4 and abs(w86[1] + 0.0648) < 5e-5,
      "(%.5f, %.5f)" % (w86[0], w86[1]))
t12_slow = np.log(2)/abs(w86[1])
print("  half-life of the slow mode = %.3f days" % t12_slow)
check("text/caption: half-life of the slow mode ~= 11 days", abs(t12_slow - 11) < 0.5,
      "(%.2f)" % t12_slow)

print()
print("=" * 78)
print("BLOCK 17  -  the nonlinear trajectory against the local spectrum")
print("=" * 78)


def f2d(t, y, k):
    T, C = y
    return [a1*T*(1 - T/K1) - mu1*C*T, a3*C*T/(K3 + T) - k*C]


def trajetoria(k, tmax):
    return solve_ivp(f2d, (0, tmax), [T0, C0], args=(k,), method='DOP853',
                     rtol=1e-12, atol=[1e-3, 1e-3], dense_output=True, max_step=0.5)


# ---- kappa_3 = 0.5: peaks, rate between peaks and convergence to the local rate
s5 = trajetoria(0.5, 600)
tt = np.linspace(0, 600, 600001)
Tt = s5.sol(tt)[0]
Ts5 = Tstar(0.5)
i = np.where((Tt[1:-1] > Tt[:-2]) & (Tt[1:-1] > Tt[2:]))[0] + 1
tp, Ap = tt[i], Tt[i] - Ts5
print("  kappa_3 = 0.5: %d peaks; the first at t=%.2f d with T/T* = %.2f"
      % (len(tp), tp[0], Tt[i[0]]/Ts5))
check("text/caption: the first peak reaches about 10 times T*",
      9.0 < Tt[i[0]]/Ts5 < 11.5, "(%.2f)" % (Tt[i[0]]/Ts5))

rate = np.log(Ap[:-1]/Ap[1:])/np.diff(tp)
tmid = 0.5*(tp[:-1] + tp[1:])
local = abs(w5[0].real)
print("  rate between the first two peaks = %.5f day^-1 (local = %.5f)" % (rate[0], local))
check("text: rate of 0.036 day^-1 between the first peaks",
      abs(rate[0] - 0.036) < 1.5e-3, "(%.5f)" % rate[0])
check("text/caption: that is about four times the local rate",
      3.5 < rate[0]/local < 4.6, "(%.2f vezes)" % (rate[0]/local))
j200 = np.argmin(abs(tmid - 200))
print("  rate near t = 200 d = %.5f day^-1  (%.1f%% above the local one)"
      % (rate[j200], 100*(rate[j200]/local - 1)))
check("text: rate ~= 0.0105 day^-1 after 200 days",
      abs(rate[j200] - 0.0105) < 6e-4, "(%.5f)" % rate[j200])
check("text: still 22%% above the local rate",
      abs(100*(rate[j200]/local - 1) - 22) < 4, "(%.1f%%)" % (100*(rate[j200]/local - 1)))
inside = np.where(abs(rate/local - 1) < 0.05)[0]
t5 = tmid[inside[0]] if len(inside) else np.nan
print("  enters the 5%% band of the local rate at t = %.0f days" % t5)
check("text: it only comes within 5%% of the local rate after some 350 days",
      abs(t5 - 350) < 20, "(%.0f days)" % t5)

# ---- kappa_3 = 0.86: saturation near K1 and settling time
s86 = trajetoria(0.86, 1200)
tt2 = np.linspace(0, 1200, 1200001)
Tt2 = s86.sol(tt2)[0]
Ts86 = Tstar(0.86)
print("  kappa_3 = 0.86: maximum T = %.4e  (%.1f%% of K_1);  T* = %.4e"
      % (Tt2.max(), 100*Tt2.max()/K1, Ts86))
check("text: the tumor rises to about 97%% of K_1", abs(100*Tt2.max()/K1 - 97) < 2,
      "(%.1f%%)" % (100*Tt2.max()/K1))
# settling = the LAST exit from the band, not the first entry: on the way up the trajectory
# crosses T* around day 11 en route to K_1, which is no settling at all.
fora2 = np.where(np.abs(Tt2/Ts86 - 1) >= 0.01)[0]
t_ac = tt2[fora2[-1] + 1] if len(fora2) and fora2[-1] + 1 < len(tt2) else np.nan
print("  stays for good within 1%% of T* from t = %.0f days on" % t_ac)
check("text/caption: settling at T* takes some 250 days", abs(t_ac - 250) < 45,
      "(%.0f days)" % t_ac)
check("text: 250 days are far more than the 11-day half-life of the slow mode",
      t_ac > 15*t12_slow)

print()
print("=" * 78)
print("SUMMARY sweep 5: %d checks, %d failures" % (len(OK), len(BAD)))
for b in BAD:
    print("  FAIL: " + b)
print("=" * 78)


# ---------------------------------------------------------------------------------------
print()
print("=" * 78)
print("BLOCK 18  -  the algebra of Proposition P7, symbolically")
print("=" * 78)
import sympy as sp

_a1, _a2, _a3, _K1, _K3, _k2, _T = sp.symbols('a1 a2 a3 K1 K3 k2 T', positive=True)
_Dt = _T**3 + 2*_K3*_T**2 + (_K3 + 4*_a3/_a1*_K1)*_K3*_T - 4*_a3/_a1*_K1**2*_K3
_alvo = (_k2**3/_a2**3 + 2*_K3*_k2**2/_a2**2 + _K3**2*_k2/_a2
         - 4*_K1*_K3*_a3/_a1*(_K1 - _k2/_a2))
_a1til = 4*_a2**2*_a3*_K1*_K3*(_a2*_K1 - _k2)/(_k2*(_a2*_K3 + _k2)**2)

check("P7: Delta_til(kappa_2/alpha_2) matches the expression printed in the proof",
      sp.simplify(sp.expand(_Dt.subs(_T, _k2/_a2) - _alvo)) == 0)
_raiz = sp.solve(sp.Eq(sp.simplify(_Dt.subs(_T, _k2/_a2)), 0), _a1)[0]
check("P7: the root in alpha_1 of Delta_til(kappa_2/alpha_2)=0 is exactly alpha_1_til",
      sp.simplify(_raiz - _a1til) == 0)
_d = sp.simplify(sp.diff(_Dt.subs(_T, _k2/_a2), _a1))
check("P7: Delta_til(kappa_2/alpha_2) grows with alpha_1 when alpha_2 K_1 > kappa_2, "
      "logo o 'se e somente se' da prova vale",
      sp.simplify(_d*_a1**2*_a2/(4*_K1*_K3*_a3) - (_K1*_a2 - _k2)) == 0)
check("P7: Delta_til is increasing in T >= 0 (derivative with all terms positive)",
      sp.simplify(sp.diff(_Dt, _T) - (_K3*(4*_K1*_a3 + _K3*_a1) + _T*_a1*(4*_K3 + 3*_T))/_a1) == 0)
check("P7: Delta_til(0) = -4 K_1^2 K_3 alpha_3/alpha_1 < 0",
      sp.simplify(_Dt.subs(_T, 0) + 4*_K1**2*_K3*_a3/_a1) == 0)
check("P7: Delta_til(K_1) = K_1^3 + 2 K_3 K_1^2 + K_3^2 K_1 > 0",
      sp.simplify(sp.expand(_Dt.subs(_T, _K1) - (_K1**3 + 2*_K3*_K1**2 + _K3**2*_K1))) == 0)

print()
print("=" * 78)
print("FINAL SUMMARY: %d checks, %d failures" % (len(OK), len(BAD)))
for b in BAD:
    print("  FAIL: " + b)
print("=" * 78)
sys.exit(1 if BAD else 0)
