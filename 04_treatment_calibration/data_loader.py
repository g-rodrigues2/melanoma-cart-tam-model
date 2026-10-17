"""Treatment data of Figure 3 of Jilani et al. (2024).

``load_data()`` is the general loader (as in the growth-model selection) and ``load_treatment_data()`` builds
the CAR T-cell groups of panel 3b:
  - cart_ss_b16, cart_sm_b16, cart_sl_b16

PBS and untransduced T cells are not included (the PBS controls were calibrated in the growth-model selection as
`control_b16`). Panel 3b is used because the three CAR T-cell products (SS, SM, SL) received the same regimen
(5x10^6 CAR+ cells and three doses of IL-2), so IL-2 does not confound the comparison.

Group names: `cart_{hinge}_b16` with hinge in {ss, sm, sl}.
"""
import glob
import os
import re

import numpy as np


# File-name pattern
_FNAME_RE = re.compile(r'fig(3[a-z])_(.+)_c(3[a-z])_m(\d+)$')


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')


def load_data(data_dir=DATA_DIR, treatments=None, panel_to_group=None,
              default_group='other'):
    """Loads the mice, filtered by treatment and grouped by panel.

    Parameters:
      data_dir         - folder with the extracted .txt files.
      treatments       - treatment slugs to include; None includes all.
      panel_to_group   - dict {panel: group name} or {(panel, treatment): group name}; both kinds of keys
                         may be mixed and (panel, treatment) keys take precedence. None groups by panel
                         (e.g. 'fig3b').
      default_group    - group of the files without a match; ``None`` discards them.

    Returns ``{group name: [mouse record, ...]}``.
    """
    if treatments is not None:
        treatments = set(treatments)

    full_data = {}
    for path in sorted(glob.glob(os.path.join(data_dir, '*.txt'))):
        base = os.path.basename(path)[:-4]
        m = _FNAME_RE.match(base)
        if m is None:
            continue
        panel, treatment, cohort, mouse = m.group(1), m.group(2), m.group(3), m.group(4)

        if treatments is not None and treatment not in treatments:
            continue

        if panel_to_group is None:
            group = f'fig{panel}'
        else:
            # (panel, treatment) keys first, then panel keys; otherwise default_group (None discards)
            key = (panel, treatment)
            if key in panel_to_group:
                group = panel_to_group[key]
            elif panel in panel_to_group:
                group = panel_to_group[panel]
            elif default_group is None:
                continue
            else:
                group = default_group

        data = np.loadtxt(path, ndmin=2)
        full_data.setdefault(group, []).append({
            'name': f'{panel}_m{mouse}',
            'data': data,
            'treatment': treatment,
            'treatment_days': np.array([0], dtype=int),  # adoptive cell transfer on day 0 (Fig. 3)
            'panel': panel,
            'cohort': cohort,
        })

    return full_data


def load_treatment_data(data_dir=DATA_DIR):
    """CAR T-cell groups of panel 3b against B16.

    Returns full_data with the keys
        'cart_ss_b16'  (20D7SS-28z, short hinge)
        'cart_sm_b16'  (20D7SM-28z, medium hinge)
        'cart_sl_b16'  (20D7SL-28z, long hinge)

    All treatments of panel 3b receive the same IL-2 regimen, which is not modeled separately.
    """
    panel_treatment_map = {
        ('3b', '20d7ss-28z'): 'cart_ss_b16',
        ('3b', '20d7sm-28z'): 'cart_sm_b16',
        ('3b', '20d7sl-28z'): 'cart_sl_b16',
    }
    full = load_data(
        data_dir,
        treatments=['20d7ss-28z', '20d7sm-28z', '20d7sl-28z'],
        panel_to_group=panel_treatment_map,
        default_group=None,  # discards panel 3h (same treatments, another experiment)
    )
    # all groups are present (possibly empty)
    for g in ('cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16'):
        full.setdefault(g, [])
    return full
