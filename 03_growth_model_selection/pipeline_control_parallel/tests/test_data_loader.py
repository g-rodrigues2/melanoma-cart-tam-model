"""Tests of data_loader."""
import os

import pytest

from data_loader import B16_PANELS, DATA_DIR, PDX_PANELS, load_control_data, load_data


# Skipped when the extracted data are not present
_HAS_DATA = os.path.isdir(DATA_DIR) and any(f.startswith('fig3') for f in os.listdir(DATA_DIR))
pytestmark = pytest.mark.skipif(
    not _HAS_DATA, reason='data/processed is empty or missing; loader tests skipped.'
)


def test_load_control_data_has_both_groups():
    full = load_control_data()
    assert 'control_b16' in full
    assert 'control_pdx' in full


def test_load_control_data_only_pbs():
    full = load_control_data()
    for grp_mice in full.values():
        for m in grp_mice:
            assert m['treatment'] == 'pbs'


def test_load_control_data_panel_assignment():
    full = load_control_data()
    for m in full['control_b16']:
        assert m['panel'] in B16_PANELS
    for m in full['control_pdx']:
        assert m['panel'] in PDX_PANELS


def test_load_data_treatment_filter():
    """load_data with treatments=['cd19car'] returns only those mice."""
    full = load_data(treatments=['cd19car'],
                     panel_to_group={'3k': 'pdx', '3m': 'pdx'})
    assert 'pdx' in full
    for m in full['pdx']:
        assert m['treatment'] == 'cd19car'


def test_load_data_panel_group_grouping():
    """The panel -> group map is applied."""
    full = load_data(treatments=['pbs'],
                     panel_to_group={'3b': 'X', '3e': 'X', '3h': 'X'})
    assert 'X' in full
    panels_found = {m['panel'] for m in full['X']}
    assert panels_found == {'3b', '3e', '3h'}


def test_load_data_no_filter_returns_all_treatments():
    full = load_data()
    all_treatments = set()
    for grp_mice in full.values():
        for m in grp_mice:
            all_treatments.add(m['treatment'])
    # Must include pbs and at least one active treatment
    assert 'pbs' in all_treatments
    assert any('20d7' in t for t in all_treatments)


def test_load_data_default_group_for_unmapped_panel():
    """Panels missing from panel_to_group go to default_group."""
    full = load_data(panel_to_group={'3b': 'b16'}, default_group='rest')
    assert 'b16' in full
    assert 'rest' in full
    for m in full['rest']:
        assert m['panel'] != '3b'
