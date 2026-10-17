# Reproducing the figures of Chapter 3

The table of [`README.md`](README.md#index-by-thesis-section) maps each numbered figure and table of the
chapter to its section, folder and script; this file gives the commands, in the order they must run.

Every command runs from the folder of its step. Figures are written with the file names used in the thesis
(`Figuras/cap3_melanoma/`). Steps 1, 2 and Scenarios 1–6 need no calibration output; the other analyses read the
maximum-likelihood files and profile solutions included in the repository (see the step READMEs). The MCMC chains of
the selected models are in the data deposit (see the repository README); the only step that needs them is
`figures_thesis/plot_tau_comparison.py`.

## Step 1 — mathematical analysis (`01_model_analysis`, outputs in `figures/`)

| Thesis figure | Command | Output |
|---|---|---|
| Model diagram (`melanoma_model_V2.png`) | drawn by hand | — |
| Growth and decay terms of the CAR T-cell equation | `python car_t_growth_balance.py` | `fig2.pdf` |
| Coexistence branches, `K1 > K̄1` and `K1 ≤ K̄1` (2×2 panel) | `python coexistence_branches.py` | `fig3.pdf` |
| Saddle-node diagram of `E3` and `(κ3, μ3)` map | `python e3_fold_and_mu3_kappa3_data.py` then `python e3_fold_and_mu3_kappa3_plot.py` | `E3_fold_bifurcation.png`, `mu3_kappa3_map.png` |
| Stability map `(α2, α3)` and Hopf boundary `(κ3, β1)` | `python stability_maps_and_hopf_boundary.py` | `stability_a2_a3.png`, `hopf_k3_b1.png` |
| Stable-spiral range of `κ3` | `python oscillatory_regime_kappa3.py` | `fig5.pdf` |
| Spiral and sink | `python spiral_and_sink.py` | `fig6.pdf` |
| Quadratic structure of `Q(β1)` and the Hopf conditions | `python hopf_threshold.py` | `fig7.pdf` |
| Hopf bifurcation diagram and the family of limit cycles | `python hopf_phase_portraits.py` | `fig8.pdf`, `results/hopf_bifurcation_diagram.npz` |
| First Lyapunov coefficient and the checks T1–T6 of Appendix A | `python lyapunov_coefficient.py` | `results/lyapunov_coefficient.txt`, `results/lyapunov_amplitude.csv` |
| Where the Hopf boundary is subcritical, sweeping `α1` | `python bautin_alpha1_sweep.py` | `results/bautin_alpha1_sweep.txt` |
| Dynamical test of the subcritical Hopf (Floquet, bistability) | `python subcritical_bistability.py` | `results/subcritical_bistability.txt` |
| Numerical checks of the propositions | `python verify_chapter3_computations.py` | printed report |
| Verification sweeps of Section 3.3.2 and Appendix A (108 checks) | `python verify_hopf_threshold_claims.py`, `verify_oscillatory_regime.py`, `verify_hopf_figure_numbers.py`, `verify_bautin_point.py`, `verify_appendix_lyapunov.py` | printed reports; status 1 on failure |

## Step 2 — global sensitivity analysis (`02_sensitivity_analysis`)

| Thesis figure | Command | Output |
|---|---|---|
| Final-time eFAST indices | `python efast_sensitivity.py` (a few minutes) then `python plot_efast_sensitivity.py` | `figures/fig9a_b.pdf` |
| Time-dependent total-effect indices | same | `figures/fast_TMC_ST_TMC_ST_stacked_time.png` |
| Dummy-parameter significance test (Appendix B) | `python efast_dummy_test.py` (about 3 min on 16 cores) | `results/dummy_test_end.csv`, `results/dummy_test_time.csv`, `results/dummy_test_replicates.npz` |
| Table of Appendix B | `python make_dummy_test_table.py` | `results/dummy_test_table.tex` |
| Time-resolved significance against the dummy | `python plot_dummy_test.py` | `figures/dummy_test_significance_time.png` |
| Ranking against the horizon, 60 to 400 days (Section 3.3.3) | `python efast_horizon.py` (about 90 s) | `results/horizon_ST.csv`, `results/horizon_diagnostics.txt` |
| Scope check over the ranges of Table 3.1 (Section 3.3.3, last paragraph) | `python efast_sensitivity_table31.py` (about 90 s on 16 cores) | `results/table31_Si_*_end.csv`, `results/table31_diagnostics.txt` |

## Step 3 — growth-model selection (`03_growth_model_selection`)

| Thesis item | Command | Output |
|---|---|---|
| Growth laws of the PBS controls (BIC) | `cd pipeline_control_parallel && python global.py` (Lonestar6: `sbatch control_parallel.slurm`) | `Output_Calibration/` |
| Parameter sharing and the B16 growth rate | `cd pipeline_shared_params_parallel && python global.py` (`sbatch shared_parallel.slurm`), then `python generate_best_figures.py` | `Output_Calibration/`, see the README for the BIC table |

## Step 4 — treatment calibration and identifiability (`04_treatment_calibration`, figures in `Output_Calibration/TMC/figures/`)

| Thesis figure or table | Command | Output |
|---|---|---|
| Treatment models (BIC) | `python compute_metrics.py` rebuilds the table from the maximum-likelihood files in the repository; the full calibration is `python global.py` (`sbatch treatment_parallel.slurm`) then `python extract_chains.py` | `Output_Calibration/model_selection_table.csv` |
| Data of the five arms of the B16 experiment | `python figures_thesis/plot_jilani_data.py` | `jilani_b16_hierarchy.png` |
| Calibrated models against the data | `python figures_thesis/plot_model_vs_data.py` | `jilani_model_vs_data.png` |
| Mechanistic models against the data | `python figures_thesis/plot_mechanistic_vs_data.py` | `jilani_mechanistic_vs_data.png` |
| Table of the candidate treatment models (appendix) | `python figures_thesis/make_model_table.py` | `model_table_cap3.tex` |
| Chains of the selected models (appendix) | `python figures_thesis/plot_appendix_chains.py` (set `MELANOMA_CHAINS` to the folder of the deposit) | `chains_selected_models.png` |
| Scaling symmetry of `β1`, `M(0)`, `K2` | `python figures_thesis/plot_symmetry_demo.py` | `structural_symmetry_demo.png` |
| Structural identifiability | `julia structural_id.jl`, `julia structural_id_pipeline.jl` (both activate the pinned environment of that directory), `python structural_identifiability.py` | `Output_Calibration/TMC/structural_identifiability_julia.txt` (the four runs reported in the thesis), printed report |
| Bounded maximum-likelihood estimates and profile likelihoods | `python mle_bounded.py`, `python profiles_bounded.py` (about 7 h), then `python figures_thesis/plot_profile_likelihoods.py` | `profile_likelihoods.png` |
| Non-identifiability of `β1` on the data | `python figures_thesis/plot_beta1_nonidentifiability.py` (after `profiles_bounded.py`) | `beta1_nonidentifiability.png` |
| Re-optimization of the profile values outside the 95% intervals | `python profiles_verify.py` | printed report |
| Fisher information spectrum | `python fim_identifiability_map.py` then `python figures_thesis/plot_fim_spectrum.py` | `fim_eigenvalue_spectrum.png` |
| Persistence data and joint Fisher information | `python fim_joint_TC.py`, `python fim_joint_TM.py` | `fim_joint_TC.csv`, `fim_joint_TM.csv` |
| In silico experimental design | `python experimental_design.py` (writes `experimental_design.csv` and the diagnostic `experimental_design_diagnostic.png`) then `python figures_thesis/plot_experimental_design.py` | `experimental_design_beta1.png` |
| Minimal identifiable submodel and autocorrelation times | `python reduced_fit.py` then `python figures_thesis/plot_tau_comparison.py` (the only step that needs a file from the Zenodo deposit: the `_mac_treat` backend of the long-hinge group) | `tau_full_vs_reduced.png` |

## Step 5 — computational scenarios (`05_computational_scenarios`, outputs in `figures/`)

| Thesis figure | Command | Output |
|---|---|---|
| Scenario 1: immunosuppression threshold | `python scenario1_immunosuppression_threshold.py` | `fig10.pdf` |
| Scenario 1: populations | `python scenario1_populations.py` | `fig11.pdf` |
| Scenario 2: dose response | `python scenario2_dose_response.py` | `fig12.pdf` |
| Scenario 3: TAM factors | `python scenario3_tam_factors.py` | `fig13.pdf` |
| Scenario 4: data-calibrated threshold | `python scenario4_data_calibrated_threshold.py` | `scenario4_data_informed.png` |
| Scenarios 5 and 6: virtual coexistence experiment | `python scenarios5_6_virtual_coexistence_experiment.py` | `scenario5_coexistence_virtual.png`, `scenario6_virtual_experiment_noise.png` |
| Scenario 7: calibrated dose response | `python scenario7_calibrated_dose_response.py` | `scenario7_dose_response_calibrated.png` |

## Tests

`python -m pytest tests -q` in each step folder: `01_model_analysis` (12),
`02_sensitivity_analysis` (14), `03_growth_model_selection/pipeline_control_parallel` (30),
`03_growth_model_selection/pipeline_shared_params_parallel` (16), `04_treatment_calibration` (61),
and `05_computational_scenarios` (6), 139 tests in all.

The suite of `01_model_analysis` runs the verification sweeps of Section 3.3.2 and Appendix A, which take
about four minutes; `-m "not slow"` skips them.
