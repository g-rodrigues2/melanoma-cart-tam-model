"""Computational scenarios of Section 3.5.

Scenarios 2 and 3 run end to end here, because they are cheap and Scenario 2 prints the control durations
quoted in the text. The scenarios that take minutes (1, 4 to 7) are checked through the result tables they
committed to `results/`.
"""
import io
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(STAGE, 'results')

CONTROL_DURATIONS = {'1e6': 0.0, '1e7': 0.0, '7e7': 12.40, '9e7': 21.70}   # days, Section 3.5


def _run(script, workdir):
    return subprocess.run([sys.executable, os.path.join(STAGE, script)], cwd=workdir,
                          capture_output=True, text=True, check=True).stdout


def test_scenario2_reproduces_the_control_durations_of_the_chapter(tmp_path):
    out = _run('scenario2_dose_response.py', tmp_path)
    found = dict(re.findall(r'\(C0 = (\S+)\): ([\d.]+) days', out))
    assert set(found) == set(CONTROL_DURATIONS), out
    for dose, expected in CONTROL_DURATIONS.items():
        assert float(found[dose]) == pytest.approx(expected, abs=0.05)
    assert (tmp_path / 'figures' / 'fig12.pdf').exists()


def test_scenario3_runs_and_draws_its_figure(tmp_path):
    _run('scenario3_tam_factors.py', tmp_path)
    assert (tmp_path / 'figures' / 'fig13.pdf').exists()


def test_scenario5_oscillation_amplitude_grows_with_the_tam_feedback():
    table = pd.read_csv(os.path.join(RESULTS, 'scenario5_amplitudes.csv'))
    table = table.sort_values('beta1_mm-3')
    assert table['log10_amp_200_300'].is_monotonic_increasing
    assert (table['T_max_200_300'] > table['T_min_200_300']).all()


def test_scenario6_design_power_is_a_probability():
    table = pd.read_csv(os.path.join(RESULTS, 'scenario6_design_power.csv'))
    probabilities = table[['P_correct_beta1_0', 'P_correct_beta1_star', 'P_correct_mean']]
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all()
    mean_of_the_two = table[['P_correct_beta1_0', 'P_correct_beta1_star']].mean(axis=1)
    assert (table['P_correct_mean'] - mean_of_the_two).abs().max() < 1e-12
    assert (table['n_points'] > 0).all()


def _scenario7():
    """scenario7_v2.csv holds a key/value block, a blank line and then the dose-response table."""
    with open(os.path.join(RESULTS, 'scenario7_v2.csv')) as f:
        header, table = f.read().split('\n\n', 1)
    values = {}
    for line in header.strip().splitlines():
        key, *rest = line.split(',')
        values[key] = [float(v) for v in rest]
    return values, pd.read_csv(io.StringIO(table))


def test_scenario7_calibration_is_a_plausible_fit():
    values, _table = _scenario7()
    for name in ('alpha1', 'K1_mm3', 'mu1'):
        assert values[name][0] > 0.0
    assert values['loglik'][0] < 0.0
    assert len(values['fitted_T0_per_mouse']) == 8
    assert all(T0 > 0.0 for T0 in values['fitted_T0_per_mouse'])


def test_scenario7_larger_doses_delay_tumor_escape():
    _values, table = _scenario7()
    table = table.sort_values('dose_multiple')
    assert table['median_days_to_10x_baseline'].is_monotonic_increasing
    assert (table['n_escaped'] <= table['n_mice']).all()
