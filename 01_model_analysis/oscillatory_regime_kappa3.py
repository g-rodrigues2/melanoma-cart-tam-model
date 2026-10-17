"""Figure 3.7: the oscillatory regime around the coexistence equilibrium E*, beta1 = 0.

(a) the coexistence branch T*(kappa3) on Omega, classified by Proposition P7, on a log axis: T*
    spans 2.5 decades over Omega, so the linear axis of the earlier version collapsed everything
    below kappa3 ~ 0.7 onto the frame and made the lower edge kappa2/alpha2 indistinguishable
    from zero. The two values used in Figure 3.8 are marked.
(b) period of the oscillation against the half-life of its envelope. The spiral/sink label of
    Proposition P7 is binary, but the oscillation it certifies is not: over Omega the number of
    cycles completed before the envelope halves falls from about 25 to well under one, so there
    is a second, practical threshold -- where the two curves cross -- above which E* is a spiral
    that no measurement could distinguish from a sink.

Classification uses the quadratic block lambda^2 - a11*lambda - a13*a31 of the proof of
Proposition P7; the third eigenvalue, a22, is always real and transversal, and at kappa3 near its
lower limit it is the slowest mode of all, so selecting the eigenvalue of largest real part from
the full 3x3 Jacobian would misclassify that end of the branch.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

alpha1, K1, mu1 = 0.69, 1e9, 1e-7
alpha2, K2, kappa2 = 1e-7, 1e9, 0.34
K3, alpha3 = 2e7, 0.9

KAPPA3_FIG38 = (0.5, 0.86)        # the two values of Figure 3.8

coef = [1, 2 * K3, (K3 + 4 * K1 * alpha3 / alpha1) * K3, -4 * K3 * K1**2 * alpha3 / alpha1]
Ttilde = float([r.real for r in np.roots(coef) if abs(r.imag) < 1e-9 and r.real > 0][0])
alpha1tilde = 4 * alpha2**2 * alpha3 * K1 * K3 * (alpha2 * K1 - kappa2) / (kappa2 * (alpha2 * K3 + kappa2) ** 2)
liminf = alpha3 * kappa2 / (kappa2 + alpha2 * K3)
limsup = alpha3 * K1 / (K1 + K3)
kappa3_tilde = alpha3 * Ttilde / (K3 + Ttilde)
T_lower = kappa2 / alpha2

k3 = np.linspace(liminf + 1e-9, limsup - 1e-9, 2000)
Tc = k3 * K3 / (alpha3 - k3)
Cs = alpha1 / mu1 * (1 - Tc / K1)
a11 = -alpha1 / K1 * Tc
a13 = -mu1 * Tc
a31 = alpha3 * Cs * K3 / (K3 + Tc) ** 2

disc = a11**2 + 4 * a13 * a31          # discriminant of the quadratic block
osc = disc < 0                         # complex pair -> spiral
period = np.where(osc, 2 * np.pi / (np.sqrt(np.abs(disc)) / 2.0), np.nan)
halflife = np.log(2) / np.abs(a11 / 2.0)

# independent check against the eigenvalues of the full 3x3 Jacobian
def _full_is_complex(i):
    Ms = K2 * (1 - kappa2 / (alpha2 * Tc[i]))
    J = np.array([[a11[i], 0.0, a13[i]],
                  [alpha2 * Ms * (1 - Ms / K2), -kappa2 + alpha2 * Tc[i] * (1 - 2 * Ms / K2), 0.0],
                  [a31[i], 0.0, 0.0]])
    return bool(np.any(np.abs(np.linalg.eigvals(J).imag) > 1e-12))
assert all(_full_is_complex(i) == osc[i] for i in np.linspace(0, len(k3) - 1, 400).astype(int))
assert abs(k3[osc].max() - kappa3_tilde) < 1e-3

# practical threshold: one oscillation per envelope half-life
ko, d = k3[osc], (period - halflife)[osc]
j = int(np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0][0])
k3_prac = float(np.interp(0.0, [d[j], d[j + 1]], [ko[j], ko[j + 1]]))

RED, BLUE, GREY = "#C1272D", "#1F4E9C", "0.25"
# same typeface family as Figure 3.8 (spiral_and_sink.py), which illustrates the same proposition
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 13,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.9))

for ax in (ax1, ax2):
    ax.axvline(kappa3_tilde, color=GREY, lw=1.5, ls=(0, (6, 3)), zorder=3)
    ax.set_xlim(liminf - 0.022, limsup)
    ax.set_xlabel(r"$\kappa_3$ (day$^{-1}$)", fontsize=15)
    ax.tick_params(labelsize=13)

# ---------------- (a) ----------------
ax1.plot(k3[osc], Tc[osc], color=RED, lw=3.2, solid_capstyle="round", zorder=5)
ax1.plot(k3[~osc], Tc[~osc], color=BLUE, lw=3.2, solid_capstyle="round", zorder=5)
ax1.axhline(Ttilde, color=GREY, lw=1.5, ls=(0, (6, 3)), zorder=2)
ax1.axhline(T_lower, color="0.5", lw=1.2, ls=(0, (1, 2.5)), zorder=2)
ax1.axhline(K1, color="0.5", lw=1.2, ls=(0, (1, 2.5)), zorder=2)
ax1.set_yscale("log")
ax1.set_ylim(1.5e6, 2.6e9)
ax1.set_ylabel(r"$T^*$ (cells)", fontsize=15)
ax1.plot([kappa3_tilde], [Ttilde], "o", ms=9, mfc="white", mec="0.15", mew=1.8, zorder=9)

# labels of the reference lines: x in data coordinates, shifted vertically so as to cross neither
# the line nor the panel letter. The Ttilde one goes to the middle, where its line is clear of the curve.
for x, y, txt, col, fy, va, ha in [(0.45, Ttilde, r"$\tilde{T}^*\approx3.87\times10^{8}$", GREY, 1.16, "bottom", "left"),
                                   (liminf - 0.016, T_lower, r"$\kappa_2/\alpha_2$", "0.5", 1 / 1.16, "top", "left"),
                                   (0.80, K1, r"$K_1$", "0.5", 1 / 1.16, "top", "right")]:
    ax1.annotate(txt, (x, y * fy), fontsize=12.5, va=va, ha=ha, color=col)
ax1.annotate(r"$\tilde{\kappa}_3\approx0.856$", (kappa3_tilde - 0.012, 1.9e9),
             fontsize=12.5, va="center", ha="right", color=GREY)

# markers of the two values of kappa3 used in the next figure; which ones they are is stated in
# the LaTeX caption, not written inside the image
for kv in KAPPA3_FIG38:
    Tv = kv * K3 / (alpha3 - kv)
    ax1.plot([kv], [Tv], "s", ms=8, mfc=RED if kv < kappa3_tilde else BLUE,
             mec="white", mew=1.4, zorder=10)

# Zoom near the threshold. The value kappa3 = 0.86 of Figure 3.8(b) is only 0.5% above
# kappa3_til, so the threshold marker and the square of that panel sit about 2 pt apart in the
# main panel: separating them requires the inset (the same device as in Figure 3.4).
axz = inset_axes(ax1, width="35%", height="30%", loc="lower right", borderpad=1.1)
xlo, xhi = kappa3_tilde - 0.012, 0.8665
mz = (k3 >= xlo) & (k3 <= xhi)
axz.plot(k3[mz & osc], Tc[mz & osc], color=RED, lw=2.2, zorder=4)
axz.plot(k3[mz & ~osc], Tc[mz & ~osc], color=BLUE, lw=2.2, zorder=4)
axz.axhline(Ttilde, color=GREY, lw=1.0, ls=(0, (5, 2.5)), zorder=2)
axz.axvline(kappa3_tilde, color=GREY, lw=1.0, ls=(0, (5, 2.5)), zorder=2)
axz.plot([kappa3_tilde], [Ttilde], "o", ms=7, mfc="white", mec="0.15", mew=1.5, zorder=9)
Tsq = 0.86 * K3 / (alpha3 - 0.86)
axz.plot([0.86], [Tsq], "s", ms=7, mfc=BLUE, mec="white", mew=1.2, zorder=10)
axz.set_xlim(xlo, xhi)
axz.set_ylim(Tc[mz].min(), Tc[mz].max())
axz.set_yscale("log")
axz.set_xticks([0.85, 0.86])
axz.tick_params(labelsize=8, length=3, pad=1.5)
axz.tick_params(axis="y", which="both", left=False, labelleft=False)
axz.set_facecolor("white")
axz.set_title(r"zoom near $\tilde{\kappa}_3$", fontsize=9, pad=3)
for sp in axz.spines.values():
    sp.set_linewidth(0.8)
mark_inset(ax1, axz, loc1=2, loc2=1, fc="none", ec="0.55", lw=0.8, zorder=1)

ax1.legend(handles=[Line2D([], [], color=RED, lw=3.2, label=r"$E^*$ stable spiral"),
                    Line2D([], [], color=BLUE, lw=3.2, label=r"$E^*$ stable sink")],
           fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.93),
           bbox_transform=ax1.transAxes, framealpha=0.96)

# ---------------- (b) ----------------
ax2.fill_between(k3, 1.5, 4e3, where=(k3 > k3_prac), color="#EAF0FA", zorder=0)
ax2.plot(k3[osc], period[osc], color=RED, lw=2.8)
ax2.plot(k3, halflife, color=GREY, lw=2.4, ls=(0, (5, 2)))
ax2.axvline(k3_prac, color=BLUE, lw=1.6, ls=(0, (1, 2)), zorder=3)
ax2.plot([k3_prac], [np.interp(k3_prac, ko, period[osc])], "o", ms=8,
         mfc="white", mec=BLUE, mew=1.8, zorder=9)
ax2.set_yscale("log")
ax2.set_ylim(1.5, 4e3)
ax2.set_ylabel("time (days)", fontsize=15)
ax2.annotate(r"$\kappa_3\approx%.2f$" % k3_prac, (k3_prac - 0.009, 2.4e3),
             fontsize=12.5, va="center", ha="right", color=BLUE)
ax2.annotate(r"$\tilde{\kappa}_3$", (kappa3_tilde - 0.010, 2.4e3),
             fontsize=12.5, va="center", ha="right", color=GREY)
for kv in KAPPA3_FIG38:
    if kv < kappa3_tilde:
        ax2.plot([kv], [np.interp(kv, ko, period[osc])], "s", ms=8,
                 mfc=RED, mec="white", mew=1.4, zorder=10)
ax2.legend(handles=[Line2D([], [], color=RED, lw=2.8, label=r"oscillation period $2\pi/|\mathrm{Im}\,\lambda|$"),
                    Line2D([], [], color=GREY, lw=2.4, ls=(0, (5, 2)), label=r"envelope half-life $\ln 2/|\mathrm{Re}\,\lambda|$")],
           fontsize=11.5, loc="lower left", framealpha=0.96)

for ax, lab in ((ax1, 'a)'), (ax2, 'b)')):
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=14, va='top',
            ha='left', fontweight='bold')

fig.tight_layout()
fig.savefig("figures/fig5.pdf", bbox_inches="tight", dpi=300)

print(f"alpha1 = {alpha1}  <  alpha1tilde = {alpha1tilde:.6e}")
print(f"Ttilde                        = {Ttilde:.6e}")
print(f"Omega = ({T_lower:.3e}, {K1:.3e});  kappa3 in ({liminf:.5f}, {limsup:.5f})")
print(f"kappa3_tilde (spiral -> sink) = {kappa3_tilde:.5f}")
print(f"practical threshold (P = t12) = {k3_prac:.5f}")
for kv in KAPPA3_FIG38:
    i = int(np.argmin(np.abs(k3 - kv)))
    n = halflife[i] / period[i] if osc[i] else float("nan")
    print(f"  kappa3={kv}: T*={Tc[i]:.3e}  {'spiral' if osc[i] else 'sink'}  cycles per half-life={n:.2f}")
