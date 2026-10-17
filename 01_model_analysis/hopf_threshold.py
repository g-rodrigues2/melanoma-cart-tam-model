"""Figure 3.9: the structure of Q(beta1) and the Hopf conditions at E* (mu3 = 0).

The earlier version plotted Q(beta1) at a single kappa3 = 0.35. Over that interval Q is
practically a straight line, so the whole figure conveyed a single number, beta1* -- which
Figure 3.10 already marks as one point of its boundary. It was therefore redundant.

This version shows the two things Figure 3.10 cannot show:

  (a) Q is a polynomial of degree TWO in beta1, and the coefficient of beta1^2 changes sign
      at kappa3 = 0.5897. Below that the parabola opens downwards and there is only one
      root, so the instability persists for every beta1 above the threshold; above it the
      parabola opens upwards and the two roots delimit a window, which closes when they
      merge at kappa3_c = 0.660. That is where the shape of the boundary of Figure 3.10
      comes from.

The Hopf conditions (a purely imaginary pair, a third real and negative eigenvalue, a
transversal crossing) are three numbers, printed below and quoted in the text: a plot of
them would be a straight line crossing zero, with no visual content, and would be read as a
vertical section of Figure 3.10.
"""
import os

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

alpha1, K1, mu1 = 0.69, 1e9, 1e-7
alpha2, K2, kappa2 = 1e-7, 1e9, 0.34
K3, alpha3 = 2e7, 0.9
KAPPA3_REF = 0.35


def jacobian(beta1, kappa3):
    T = kappa3*K3/(alpha3-kappa3)
    M = K2*(1-kappa2/(alpha2*T))
    C = alpha1/mu1*(1-T/K1)*(1+beta1*M)
    return np.array([[-(alpha1/K1)*T*(1+beta1*M), alpha1*T*(1-T/K1)*beta1, -mu1*T],
                     [alpha2*M*(1-M/K2), -kappa2+alpha2*T*(1-2*M/K2), 0.0],
                     [alpha3*C*K3/(K3+T)**2, 0.0, 0.0]])


def Q(beta1, kappa3):
    A = jacobian(beta1, kappa3)
    a11, a12, a13 = A[0]
    a21, a22 = A[1, 0], A[1, 1]
    a31 = A[2, 0]
    return (-(a11+a22))*(a11*a22-a12*a21-a13*a31) - a13*a22*a31


def q_coeffs(kappa3):
    """Q is quadratic in beta1: returns (c0, c1, c2) by exact interpolation."""
    xs = np.array([0.0, 1e-9, 2e-9])
    ys = np.array([Q(x, kappa3) for x in xs])
    c2, c1, c0 = np.polyfit(xs, ys, 2)
    return c0, c1, c2


beta1_star = brentq(lambda b: Q(b, KAPPA3_REF), 1e-13, 2e-10, xtol=1e-24, rtol=8.9e-16)
kappa3_flip = brentq(lambda k: q_coeffs(k)[2], 0.5, 0.7, xtol=1e-12)
kappa3_c = brentq(lambda k: (lambda c: c[1]**2 - 4*c[2]*c[0])(q_coeffs(k)), 0.62, 0.665, xtol=1e-12)   # first zero: the two roots merge

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 13,
})
RED, BLUE, GREY = "#C1272D", "#1F4E9C", "0.30"

fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
BETA1_MAX_TAB = 3e-6                      # ceiling of beta1 in Table 3.1

# ---------------- (a) Q is quadratic in beta1 ------------------------------
# log axis: the two roots of one kappa3 can be four orders of magnitude apart
CASES = [(0.35, "#8C1C13"), (0.55, "#C1272D"), (0.64, "#2E7D8F"), (0.70, "#1F4E9C")]
b = np.logspace(-11.4, -5.0, 3000)
for k3, c in CASES:
    c0, c1, c2 = q_coeffs(k3)
    ax1.plot(b, (c0 + c1*b + c2*b**2)/c0, color=c, lw=2.1, zorder=4,
             label=r"$\kappa_3=%.2f$" % k3)
    r = np.sort([x.real for x in np.roots([c2, c1, c0]) if abs(x.imag) < 1e-30 and x.real > 0])
    ax1.plot(r, np.zeros(len(r)), "o", ms=7, mfc=c, mec="white", mew=1.3, zorder=7)
ax1.axhline(0.0, color=GREY, lw=1.2, ls=(0, (5, 3)), zorder=2)
ax1.axvline(BETA1_MAX_TAB, color="0.55", lw=1.3, ls=(0, (1, 2.2)), zorder=3)
ax1.set_xscale("log")
ax1.set_yscale("symlog", linthresh=1.0, linscale=0.9)
ax1.set_xlim(b[0], b[-1])
ax1.set_ylim(-3e5, 60)
ax1.set_xlabel(r"$\beta_1$ (cells$^{-1}$), log scale", fontsize=14)
ax1.set_ylabel(r"$Q(\beta_1)\,/\,Q(0)$", fontsize=14)
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=11, loc="lower left", framealpha=0.95, ncol=2)
ax1.annotate(r"$Q>0$: $E^*$ stable", (1.3e-11, 6.0), fontsize=11, color=GREY, va="center")
ax1.annotate(r"$Q<0$: $E^*$ unstable", (1.3e-11, -8.0), fontsize=11, color=GREY, va="center")

fig.tight_layout()
fig.savefig("figures/fig7.pdf", bbox_inches="tight", dpi=300)

ev = np.linalg.eigvals(jacobian(beta1_star, KAPPA3_REF))
h = beta1_star*1e-4
dre = ((lambda b: np.linalg.eigvals(jacobian(b, KAPPA3_REF))[np.argsort(-np.abs(np.linalg.eigvals(jacobian(b, KAPPA3_REF)).imag))][0].real)(beta1_star+h)
       - (lambda b: np.linalg.eigvals(jacobian(b, KAPPA3_REF))[np.argsort(-np.abs(np.linalg.eigvals(jacobian(b, KAPPA3_REF)).imag))][0].real)(beta1_star-h))/(2*h)
print(f"beta1*                      = {beta1_star:.6e}  (|Q| = {abs(Q(beta1_star, KAPPA3_REF)):.2e})")
print(f"eigenvalues at beta1*       = {np.round(ev, 6)}")
print(f"transversality dRe/dbeta1   = {dre:+.4e}")
print(f"period of the nascent orbit = {2*np.pi/max(abs(ev.imag)):.2f} days")
print(f"kappa3 of the b1^2 coef flip = {kappa3_flip:.5f}")
print(f"kappa3_c (roots merge)      = {kappa3_c:.5f}")
for k3, _ in CASES:
    c0, c1, c2 = q_coeffs(k3)
    r = np.sort([x.real for x in np.roots([c2, c1, c0]) if abs(x.imag) < 1e-30 and x.real > 0])
    print(f"  kappa3={k3}: c2={c2:+.3e}  positive roots={['%.4e' % x for x in r]}")
