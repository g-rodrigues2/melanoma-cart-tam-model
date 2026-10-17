"""Data extracted from Figure 3 of Jilani et al. (2024).

Files in ``data/processed/`` follow the pattern
    fig{panel}_{treatment}_c{cohort}_m{mouse}.txt
with two columns: day after adoptive cell transfer and tumor volume (mm^3).

``load_data()`` filters by treatment and groups the mice by panel with a configurable map.
``load_control_data()`` returns the PBS controls grouped by tumor model (growth-model selection); the
treatment calibration uses ``load_data()`` with its own filters.
"""
import glob
import os
import re

import numpy as np


# Default map panel -> tumor model
B16_PANELS = {'3b', '3e', '3h'}      # syngeneic B16 melanoma (TYRP1 high)
PDX_PANELS = {'3k', '3m'}            # patient-derived xenograft

# File-name pattern
_FNAME_RE = re.compile(r'fig(3[a-z])_(.+)_c(3[a-z])_m(\d+)$')


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')


def load_data(data_dir=DATA_DIR, treatments=None, panel_to_group=None,
              default_group='other'):
    """Loads the mice, filtered by treatment and grouped by panel.

    Parameters:
      data_dir         - folder with the extracted .txt files.
      treatments       - treatment slugs to include (e.g. ['pbs'], ['20d7sl-28z', '20d7sl-28z+il2']);
                         None includes all.
      panel_to_group   - dict {panel: group name} (e.g. {'3b': 'control_b16', '3k': 'control_pdx'}); panels
                         missing from the dict go to ``default_group``. None groups by panel ('fig3b', ...).
      default_group    - group of the panels not listed in panel_to_group (used only when it is given).

    Returns ``{group name: [mouse record, ...]}``; each record has the keys 'name', 'data' (N x 2 array),
    'treatment', 'treatment_days', 'panel' and 'cohort'.
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
            group = panel_to_group.get(panel, default_group)

        data = np.loadtxt(path, ndmin=2)
        full_data.setdefault(group, []).append({
            'name': f'{panel}_m{mouse}',
            'data': data,
            'treatment': treatment,
            'treatment_days': np.array([], dtype=int),  # set by the treatment calibration
            'panel': panel,
            'cohort': cohort,
        })

    return full_data


def load_control_data(data_dir=DATA_DIR):
    """PBS controls grouped by tumor model.

    Returns full_data with the keys ``'control_b16'`` and ``'control_pdx'``. Equivalent to

        load_data(data_dir,
                  treatments=['pbs'],
                  panel_to_group={p: 'control_b16' for p in B16_PANELS}
                               | {p: 'control_pdx' for p in PDX_PANELS})
    """
    panel_map = {}
    for p in B16_PANELS:
        panel_map[p] = 'control_b16'
    for p in PDX_PANELS:
        panel_map[p] = 'control_pdx'
    full = load_data(data_dir, treatments=['pbs'], panel_to_group=panel_map)
    # both groups are always present (possibly empty)
    full.setdefault('control_b16', [])
    full.setdefault('control_pdx', [])
    return full
