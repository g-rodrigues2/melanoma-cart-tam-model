"""Tests of data_loader.py."""

from data_loader import DATA_DIR, load_treatment_data


def test_load_treatment_returns_three_cart_groups():
    full = load_treatment_data(data_dir=DATA_DIR)
    for group in ('cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16'):
        assert len(full[group]) > 0, f'{group} has no mice (data folder {DATA_DIR})'


def test_load_treatment_only_panel_3b():
    full = load_treatment_data(data_dir=DATA_DIR)
    for g, mice in full.items():
        for m in mice:
            assert m['panel'] == '3b', \
                f"{g}/{m['name']}: unexpected panel {m['panel']}"


def test_load_treatment_correct_treatments_per_group():
    full = load_treatment_data(data_dir=DATA_DIR)
    expected = {
        'cart_ss_b16': '20d7ss-28z',
        'cart_sm_b16': '20d7sm-28z',
        'cart_sl_b16': '20d7sl-28z',
    }
    for g, tx in expected.items():
        for m in full[g]:
            assert m['treatment'] == tx


def test_load_treatment_data_arrays_shape():
    full = load_treatment_data(data_dir=DATA_DIR)
    for g, mice in full.items():
        for m in mice:
            assert m['data'].ndim == 2
            assert m['data'].shape[1] == 2  # (time, volume)
            assert m['data'].shape[0] > 1


def test_load_treatment_excludes_pbs_utd():
    full = load_treatment_data(data_dir=DATA_DIR)
    # PBS and UTD are not groups
    assert 'pbs_b16' not in full
    assert 'utd_b16' not in full
    # nor treatments inside the three CAR T-cell groups
    for g, mice in full.items():
        for m in mice:
            assert m['treatment'] not in ('pbs', 'utd')
