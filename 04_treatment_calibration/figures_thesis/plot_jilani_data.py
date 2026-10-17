"""Data of Jilani et al. (Fig. 3b): tumor volumes of the five arms (PBS, UTD, short/medium/long hinge), showing
the anti-tumor hierarchy without any model fit. The exponential growth of the PBS controls (maximum-likelihood
r_B16 = 0.2648/day, the value fixed in the treatment models) is drawn as a reference.
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _paths import FIGURES

from _paths import STYLE

plt.rcParams.update(STYLE)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')

groups = {
    'pbs': ('PBS (no treatment)', '#7f7f7f'),
    'utd': ('UTD (mock T cells)', '#b3b3b3'),
    '20d7ss-28z': ('short hinge', '#4393c3'),
    '20d7sm-28z': ('medium hinge', '#e08214'),
    '20d7sl-28z': ('long hinge', '#c0392b'),
}

fig, ax = plt.subplots(figsize=(9.0, 4.5))

for key, (label, color) in groups.items():
    files = sorted(glob.glob(os.path.join(DATA, f'fig3b_{key}_c3b_m*.txt')))
    by_day = {}
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t, v = line.split()
                by_day.setdefault(round(float(t)), []).append(float(v))
    days = sorted(by_day)
    means = np.array([np.mean(by_day[d]) for d in days])
    sems = np.array([np.std(by_day[d], ddof=1) / max(len(by_day[d]), 1) ** 0.5
                      if len(by_day[d]) > 1 else 0.0 for d in days])
    days = np.array(days)
    ax.plot(days, means, color=color, lw=2.2, marker='o', ms=4.5, label=label,
            zorder=4)
    ax.fill_between(days, np.clip(means - sems, 1, None), means + sems,
                     color=color, alpha=0.18, lw=0)

# reference: exponential growth calibrated on the PBS controls
r = 0.2648
t0, V0 = 0, 13.5
tt = np.linspace(0, 27, 100)
ax.plot(tt, V0 * np.exp(r * (tt - t0)), color='k', ls='--', lw=1.8, zorder=5)

ax.set_yscale('log')
ax.set_xlabel('time since adoptive transfer (days)')
ax.set_ylabel(r'tumor volume (mm$^3$), log scale')
ax.set_xlim(0, 27)

leg = [Line2D([0], [0], color=c, marker='o', ms=4, lw=1.4, label=l)
       for l, c in groups.values()]
leg.append(Line2D([0], [0], color='k', ls='--', lw=1.8,
                   label=r'exponential fit, PBS ($r_{B16}=0.2648$ day$^{-1}$)'))
ax.legend(handles=leg, loc='upper left', framealpha=0.95, fontsize=11.0,
          handletextpad=0.6, borderpad=0.6)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'jilani_b16_hierarchy.png'), dpi=220, bbox_inches='tight')
plt.close(fig)
print('figure saved.')
