# 05 — Computational scenarios (Section 3.5)

| Script | Scenario | Parameters |
|---|---|---|
| `scenario1_immunosuppression_threshold.py` | 1: final tumor size against `μ3` and bisection of the transition threshold `μ3† ≈ 6.75×10⁻¹⁰`. | Table 3.1 (cells) |
| `scenario1_populations.py` | 1: trajectories of `T`, `M`, `C` for four values of `μ3`. | Table 3.1 |
| `scenario2_dose_response.py` | 2: CAR T-cell doses from `10⁶` to `9×10⁷` cells at `μ3 = μ3†`. | Table 3.1 |
| `scenario3_tam_factors.py` | 3: control duration for `β1 = 10⁻⁷` and `2×10⁻⁶` and four doses. | Table 3.1 |
| `scenario4_data_calibrated_threshold.py` | 4: threshold `μ3†` with `α1` and `κ3` estimated from the data. | Table 3.1 with `α1 = 0.2648` (maximum-likelihood `r_B16`), `κ3 = 0.09` |
| `scenarios5_6_virtual_coexistence_experiment.py` | 5 and 6: trajectories for three values of `β1` in the coexistence regime, and discrimination between `β1 = 0` and `β1 = β1*` from noisy samples of `M(t)` (likelihood ratio, Monte Carlo). | coexistence operating point (mm³) |
| `scenario7_calibrated_dose_response.py` | 7: dose response under the minimal identifiable submodel, from a single maximum-likelihood fit with one initial volume per mouse; checks the administered dose against the data. Uses the modules of `../04_treatment_calibration/`. | calibrated submodel (mm³) |

Scenarios 1–6 are self-contained. Scenario 7 requires the data in `../data/processed/`.

Figures are written to `figures/` with the file names of the thesis; the tables that accompany them (`scenario5_amplitudes.csv`, `scenario6_design_power.csv`, `scenario7_v2.csv`) go to `results/`.
