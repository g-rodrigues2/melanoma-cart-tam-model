"""M*(T*) and C*(T*) along the coexistence branch (thesis Figure 3.4: figures/fig3.pdf).

The two regimes of Proposition 3.5 are shown in a single 2x2 panel: the top row is case 1 (K1 > K1bar),
where the critical point Tbar* lies inside Omega = (kappa2/alpha2, K1) and C*(T*) has an interior maximum;
the bottom row is case 2 (K1 <= K1bar), where Tbar* falls below kappa2/alpha2 and C*(T*) is decreasing on
all of Omega. (Before the merge these were Figures 3.4 and 3.5, fig3.pdf and fig4.pdf.)

Volumes are divided by fixed powers of ten before plotting, so every panel and the inset share the same
axis units; the automatic offsets of matplotlib can differ between axes and are deliberately not used.

The inset window is chosen around Tbar* so that the increasing branch of C*(T*) is actually visible: the
rise between T* = kappa2/alpha2 and T* = Tbar* is only ~0.06% of C*, so the window is closed just past the
point where C*(T*) has fallen back to its value at the left endpoint of Omega. This makes the maximum
occupy most of the inset height instead of being flattened by the steep decreasing branch.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.optimize import brentq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
    'axes.linewidth': 1.0,
})

alpha1 = 0.69
mu1 = 1e-7
alpha2 = 1e-7
kappa2 = 0.34
K2 = 1e9
beta1 = 2.3e-10

C_M = '#252525'       # M*(T*) curve
C_C = '#c0392b'       # C*(T*) curve
C_MARK = 'crimson'    # critical point (Tbar*, Cbar*), same color as the other reference points of Chapter 3
C_FILL = '#f7dcd7'    # increasing sub-interval of Omega, highlighted in the inset


def M_star(T_star):
    return K2 * (1.0 - kappa2 / (alpha2 * T_star))


def C_star(T_star, K1):
    return (alpha1 / mu1) * (1.0 - T_star / K1) * (1.0 + beta1 * M_star(T_star))


def bar_K1():
    return kappa2 * (1.0 + beta1 * K2) / (alpha2 * beta1 * K2)


def bar_T_star(K1):
    return np.sqrt(beta1 * K1 * K2 * kappa2 / (alpha2 * (1.0 + beta1 * K2)))


T_min = kappa2 / alpha2
barK1_val = bar_K1()


def draw_row(ax_M, ax_C, K1, with_inset):
    """Draw the M* and C* panels for one value of K1.

    Returns a callable that adds the zoom inset, to be invoked after tight_layout (inset axes are not
    compatible with it), or None when the case has no critical point to zoom into.
    """
    T_values = np.linspace(T_min, K1, 4000)
    Tx = T_values / 1e7
    Mx = M_star(T_values) / 1e8
    Cx = C_star(T_values, K1) / 1e6

    ax_M.plot(Tx, Mx, color=C_M, lw=2.0, label=r'$M^*(T^*)$')
    ax_M.set_xlabel(r'$T^*$  ($\times10^{7}$ cells)')
    ax_M.set_ylabel(r'$M^*(T^*)$  ($\times10^{8}$ cells)')
    ax_M.set_xlim(Tx[0], Tx[-1])
    # headroom at the top so the panel letter never sits on the curve: in the C* panels the maximum is at
    # the left edge, exactly where the letter goes
    ax_M.set_ylim(-0.04 * Mx.max(), 1.18 * Mx.max())
    ax_M.legend(loc='lower right', framealpha=0.95, handletextpad=0.6, borderpad=0.5)

    ax_C.plot(Tx, Cx, color=C_C, lw=2.0, label=r'$C^*(T^*)$', zorder=3)
    ax_C.set_xlabel(r'$T^*$  ($\times10^{7}$ cells)')
    ax_C.set_ylabel(r'$C^*(T^*)$  ($\times10^{6}$ cells)')
    ax_C.set_xlim(Tx[0], Tx[-1])
    ax_C.set_ylim(-0.04 * Cx.max(), 1.18 * Cx.max())

    leg_handles = [Line2D([0], [0], color=C_C, lw=2.0, label=r'$C^*(T^*)$')]
    T_bar = bar_T_star(K1)

    if with_inset:
        C_bar = C_star(T_bar, K1)
        Tbar_x, Cbar_x = T_bar / 1e7, C_bar / 1e6
        ax_C.plot(Tbar_x, Cbar_x, marker='*', color=C_MARK, ms=14, mec='k', mew=0.6, zorder=6)
        leg_handles.append(Line2D([0], [0], marker='*', color='none', markerfacecolor=C_MARK,
                                  markeredgecolor='k', markersize=12, label=r'$\bar T^*$'))
        ax_C.legend(handles=leg_handles, loc='upper right', framealpha=0.95,
                    handletextpad=0.6, borderpad=0.5)

        # right endpoint of the zoom: just past where C* falls back to its value at T* = kappa2/alpha2, so
        # the window is nearly symmetric in C* around the maximum and the hump fills the inset height
        C_left = C_star(T_min, K1)
        T_right = brentq(lambda T: C_star(T, K1) - C_left, T_bar, K1)
        Tz_min, Tz_max = T_min, T_right + 0.18 * (T_right - T_min)
        print(f"  inset window: T* in [{Tz_min:.4e}, {Tz_max:.4e}] cells; Tbar* = {T_bar:.4e}")

        mask = (T_values >= Tz_min) & (T_values <= Tz_max)
        Cz = Cx[mask]
        xlo, xhi = Tz_min / 1e7, Tz_max / 1e7
        pad = 0.17 * (Cz.max() - Cz.min())
        ylo, yhi = Cz.min() - pad, Cz.max() + pad

        def add_inset():
            axins = inset_axes(ax_C, width="41%", height="41%", loc="lower left",
                               bbox_to_anchor=(0.09, 0.07, 1, 1),
                               bbox_transform=ax_C.transAxes, borderpad=0.4)
            axins.axvspan(xlo, Tbar_x, color=C_FILL, zorder=0)       # increasing sub-interval
            axins.plot(Tx[mask], Cz, color=C_C, lw=1.8, zorder=3)
            axins.axvline(Tbar_x, color=C_MARK, ls='--', lw=0.9, zorder=2)
            axins.plot(Tbar_x, Cbar_x, marker='*', color=C_MARK, ms=10, mec='k', mew=0.5, zorder=5)
            axins.set_xlim(xlo, xhi)
            axins.set_ylim(ylo, yhi)
            axins.set_xticks(np.linspace(xlo, xhi, 4))
            axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axins.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            axins.locator_params(axis='y', nbins=4)
            axins.tick_params(labelsize=8, length=3, pad=1.5)
            axins.set_facecolor('white')
            axins.set_title(r'zoom near $\bar T^*$', fontsize=9, pad=3)
            for spine in axins.spines.values():
                spine.set_linewidth(0.8)
            # the zoom window is far too small to be drawn as a visible box on the main panel, so the two
            # leaders simply converge on the marked point (Tbar*, Cbar*)
            mark_inset(ax_C, axins, loc1=2, loc2=1, fc="none", ec="0.55", lw=0.8, zorder=1)
            return axins

        return add_inset
    else:
        # Tbar* exists as a formula but falls outside Omega, so C* is decreasing on all of Omega
        ax_C.annotate(r'$\bar T^*\approx%.2f\times10^{6} < \kappa_2/\alpha_2$' % (T_bar / 1e6) + '\n'
                      + r'(outside $\Omega$: no interior maximum)',
                      xy=(0.72, 0.70), xycoords='axes fraction', ha='center', va='center',
                      fontsize=10.5,
                      bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.7', lw=0.8))
        ax_C.legend(handles=leg_handles, loc='upper right', framealpha=0.95,
                    handletextpad=0.6, borderpad=0.5)
    return None


K1_case1 = 2.0e7   # case 1: K1 > K1bar
K1_case2 = 1.0e7   # case 2: K1 <= K1bar
print(f"bar_K1 = {barK1_val:.4e} cells")
print(f"case 1: K1={K1_case1:.3e} > bar_K1: {K1_case1 > barK1_val}")
print(f"case 2: K1={K1_case2:.3e} <= bar_K1: {K1_case2 <= barK1_val}")

fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
pending_insets = [
    draw_row(axes[0, 0], axes[0, 1], K1_case1, with_inset=True),
    draw_row(axes[1, 0], axes[1, 1], K1_case2, with_inset=False),
]

fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), h_pad=4.5)

# panel letters inside the axes, upper left: same placement and style as every other panel label in the
# thesis (see e3_fold_and_mu3_kappa3_plot.py, plot_efast_sensitivity.py, scenario1-3)
for ax, lab in zip(axes.ravel(), ['a)', 'b)', 'c)', 'd)']):
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=13, va='top', ha='left',
            fontweight='bold')

row_titles = [
    r'Case 1:  $K_1 = 2\times10^{7} > \bar K_1 \approx 1.82\times10^{7}$ cells',
    r'Case 2:  $K_1 = 1\times10^{7} \leq \bar K_1 \approx 1.82\times10^{7}$ cells',
]
for row, title in enumerate(row_titles):
    top = max(axes[row, c].get_position().y1 for c in (0, 1))
    fig.text(0.5, top + 0.040, title, fontsize=13.5, ha='center', va='center')

for add_inset in pending_insets:
    if add_inset is not None:
        add_inset()

fig.savefig(os.path.join(OUT, 'fig3.pdf'), bbox_inches='tight')
plt.close(fig)
print("figures/fig3.pdf written (2x2 panel, replaces the former fig3.pdf + fig4.pdf pair).")
