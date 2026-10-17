"""The independent verification sweeps of Chapter 3, run as tests.

Each `verify_*.py` recomputes, by a route independent of the scripts that produced the figures
(exact rationals in sympy, mpmath at 50-60 digits, tighter integrators), the numbers of
Section 3.3 and of Appendix A that entered the thesis. Each one prints `[OK]` or `[FAIL]` per
check, a summary line, and exits with status 1 if anything failed.

These tests assert the exit status, that no check failed, and that the expected number of checks
actually ran, so that a sweep that silently stops early also fails. They are slow by nature
(symbolic derivatives and high-precision arithmetic); `-m "not slow"` skips them.

`verify_chapter3_computations.py` is exercised by test_model_analysis.py instead, which checks
its seven reported values one by one.
"""
import os
import re
import subprocess
import sys

import pytest

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# script, number of checks it must run (the count reported in the thesis sweeps)
SWEEPS = [
    ("verify_hopf_threshold_claims.py", 22),
    ("verify_appendix_lyapunov.py", 41),
    ("verify_hopf_figure_numbers.py", 9),
    ("verify_bautin_point.py", 4),
    ("verify_oscillatory_regime.py", 32),
]


@pytest.mark.slow
@pytest.mark.parametrize("script,n_checks", SWEEPS, ids=[s[0] for s in SWEEPS])
def test_verification_sweep_passes_every_check(script, n_checks):
    run = subprocess.run([sys.executable, script], cwd=STAGE, capture_output=True, text=True)
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "[FAIL]" not in out, out
    assert out.count("[OK]") == n_checks, (
        f"{script}: {out.count('[OK]')} checks ran, {n_checks} expected\n{out}")
    assert re.search(r"SUMMARY.*0 failures", out), out


@pytest.mark.slow
def test_lyapunov_coefficient_script_passes_its_own_checks():
    """lyapunov_coefficient.py carries the checks T1-T6 of Appendix A and asserts them itself."""
    run = subprocess.run([sys.executable, "lyapunov_coefficient.py"], cwd=STAGE,
                         capture_output=True, text=True)
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "every check passed" in out, out
    for label in ("T1", "T2", "T3", "T4", "T5", "T6"):
        assert label in out, f"{label} missing from the report\n{out}"
    # the amplitude table of Appendix A must be persisted, not only printed
    csv = os.path.join(STAGE, "results", "lyapunov_amplitude.csv")
    assert os.path.exists(csv), "results/lyapunov_amplitude.csv was not written"
    rows = [ln for ln in open(csv, encoding="utf-8").read().splitlines() if ln.strip()]
    assert len(rows) == 9, rows               # header + the eight values of eps
    last = float(rows[-1].split(",")[3])
    assert abs(last - 1) < 0.05, f"the ratio at the smallest eps must approach 1, got {last}"
