# Mathematical modeling of melanoma dynamics with TAM-mediated immunosuppression and CAR T-cell therapy

All the code and data in this repository are part of the doctoral thesis

> G. Rodrigues, *Mathematical models to describe tumoral behaviors under treatment regimens*,
> Graduate Program in Biometrics, Institute of Biosciences, São Paulo State University (UNESP),
> Botucatu, 2026. Advisor: P. F. A. Mancera. Co-advisor: E. A. B. F. Lima.

and correspond to its **Chapter 3**, *Mathematical modeling of melanoma dynamics with TAM-mediated
immunosuppression and CAR T-cell therapy*, whose code, data and results are deposited here in full.
The chapter studies the model

```
dT/dt = α1 T (1 − T/K1)(1 + β1 M) − μ1 C T
dM/dt = α2 M T (1 − M/K2) − κ2 M
dC/dt = α3 C T/(K3 + T) − μ3 C T − κ3 C
```

for melanoma cells `T`, tumor-associated macrophages `M`, and CAR T-cells `C`, from its mathematical analysis to its
confrontation with the B16 data of Jilani et al. (2024).

## Workflow

The folders follow the order of the chapter. Each one has a README listing its scripts and the results they produce.

| Step | Folder | Thesis section | What it does |
|---|---|---|---|
| 1 | [`01_model_analysis/`](01_model_analysis/) | 3.3.1–3.3.2 | Equilibria, local stability, stability maps, spiral/sink classification, Hopf threshold and Hopf boundary, phase portraits, symbolic and numerical verification of the propositions. |
| 2 | [`02_sensitivity_analysis/`](02_sensitivity_analysis/) | 3.3.3 | Global sensitivity analysis with eFAST (final-time and time-dependent indices). |
| 3 | [`03_growth_model_selection/`](03_growth_model_selection/) | 3.4 | Bayesian calibration of classical growth laws to the PBS controls and estimation of the B16 growth rate. |
| 4 | [`04_treatment_calibration/`](04_treatment_calibration/) | 3.4 | Bayesian calibration of empirical treatment terms and of the mechanistic model to the CAR T-cell groups; bounded maximum-likelihood estimation; structural identifiability (Julia); profile likelihoods; Fisher information; experimental design; persistence data; minimal identifiable submodel. |
| 5 | [`05_computational_scenarios/`](05_computational_scenarios/) | 3.5 | Scenarios 1–3 (literature parameters) and Scenarios 4–7 (data-calibrated rates and virtual experiments). |

`notebooks/melanoma_CART_TAM.ipynb` is the exploratory notebook from which the scripts of steps 1, 2 and Scenarios 1–3
were extracted; the scripts are the reference implementation. `export_chains.py` packs the MCMC chains of the
selected models for the data deposit. The chains themselves are deposited separately because of their size; the DOI of that deposit is given in the description of this record.

Every number of Sections 3.3.1–3.3.2 and of Appendix A that entered the thesis is recomputed by the
`verify_*.py` sweeps of step 1, by a route independent of the scripts that drew the figures; they run in the test
suite of that folder. Code is in English, with LF line endings; each step writes its figures with the file names
used in the thesis, to `figures/` (steps 1, 2, 5) or to `Output_Calibration/` and `Output_Optimization/`
(steps 3, 4, 6).

Steps 1, 2 and 3 are independent. Step 4 uses the growth rate from step 3. Scenarios 4–7 (step 5) and step 6 use the
outputs of step 4.

## Index by thesis section

Every numbered figure and table of Chapter 3 and of its appendices, with the section it appears in, the folder it
comes from and the script that writes it. Figure numbers are those of the compiled thesis; the commands that produce
each one, with their order and runtime, are in [`REPRODUCE.md`](REPRODUCE.md).

| Thesis item | Section | Folder | Script | Output |
|---|---|---|---|---|
| Fig. 3.1 | 3.2 | — | drawn by hand | `melanoma_model_V2.png` |
| Fig. 3.2 | 3.3 | `01_model_analysis` | `car_t_growth_balance.py` | `figures/fig2.pdf` |
| Fig. 3.3 | 3.3.1 | `01_model_analysis` | `e3_fold_and_mu3_kappa3_data.py` then `e3_fold_and_mu3_kappa3_plot.py` | `figures/E3_fold_bifurcation.png` |
| Fig. 3.4 | 3.3.1 | `01_model_analysis` | `coexistence_branches.py` | `figures/fig3.pdf` |
| Fig. 3.5(a) | 3.3.1 | — | reproduced from the CoAM paper | `coam_figures/stability_space1.png` |
| Fig. 3.5(b) | 3.3.1 | `01_model_analysis` | `stability_maps_and_hopf_boundary.py` | `figures/stability_a2_a3.png` |
| Fig. 3.6 | 3.3.1 | `01_model_analysis` | `e3_fold_and_mu3_kappa3_data.py` then `e3_fold_and_mu3_kappa3_plot.py` | `figures/mu3_kappa3_map.png` |
| Fig. 3.7 | 3.3.2 | `01_model_analysis` | `oscillatory_regime_kappa3.py` | `figures/fig5.pdf` |
| Fig. 3.8 | 3.3.2 | `01_model_analysis` | `spiral_and_sink.py` | `figures/fig6.pdf` |
| Fig. 3.9 | 3.3.2 | `01_model_analysis` | `hopf_threshold.py` | `figures/fig7.pdf` |
| Fig. 3.10 | 3.3.2 | `01_model_analysis` | `stability_maps_and_hopf_boundary.py` | `figures/hopf_k3_b1.png` |
| Fig. 3.11 | 3.3.2 | `01_model_analysis` | `hopf_phase_portraits.py` | `figures/fig8.pdf` |
| Fig. 3.12 | 3.3.3 | `02_sensitivity_analysis` | `efast_sensitivity.py` then `plot_efast_sensitivity.py` | `figures/fig9a_b.pdf` |
| Fig. 3.13 | 3.3.3 | `02_sensitivity_analysis` | `plot_efast_sensitivity.py` | `figures/fast_TMC_ST_TMC_ST_stacked_time.png` |
| Fig. 3.14 | 3.3.3 | `02_sensitivity_analysis` | `efast_dummy_test.py` then `plot_dummy_test.py` | `figures/dummy_test_significance_time.png` |
| Fig. 3.15 | 3.4 | `04_treatment_calibration` | `figures_thesis/plot_jilani_data.py` | `.../TMC/figures/jilani_b16_hierarchy.png` |
| Fig. 3.16 | 3.4 | `04_treatment_calibration` | `figures_thesis/plot_mechanistic_vs_data.py` | `.../TMC/figures/jilani_mechanistic_vs_data.png` |
| Fig. 3.17 | 3.4 | `04_treatment_calibration` | `figures_thesis/plot_mechanistic_chains.py` (reads the MCMC backend) | `.../TMC/figures/jilani_tmc_chain.png` |
| Fig. 3.18 | 3.4 | `04_treatment_calibration` | `mle_bounded.py`, `profiles_bounded.py` then `figures_thesis/plot_profile_likelihoods.py` | `.../TMC/figures/profile_likelihoods.png` |
| Fig. 3.19 | 3.4 | `04_treatment_calibration` | `profiles_bounded.py` then `figures_thesis/plot_beta1_nonidentifiability.py` | `.../TMC/figures/beta1_nonidentifiability.png` |
| Fig. 3.20 | 3.5.1 | `05_computational_scenarios` | `scenario1_immunosuppression_threshold.py` | `figures/fig10.pdf` |
| Fig. 3.21 | 3.5.1 | `05_computational_scenarios` | `scenario1_populations.py` | `figures/fig11.pdf` |
| Fig. 3.22 | 3.5.2 | `05_computational_scenarios` | `scenario2_dose_response.py` | `figures/fig12.pdf` |
| Fig. 3.23 | 3.5.3 | `05_computational_scenarios` | `scenario3_tam_factors.py` | `figures/fig13.pdf` |
| Fig. 3.24 | 3.5.4 | `05_computational_scenarios` | `scenario4_data_calibrated_threshold.py` | `figures/scenario4_data_informed.png` |
| Fig. 3.25 | 3.5.5 | `05_computational_scenarios` | `scenarios5_6_virtual_coexistence_experiment.py` | `figures/scenario5_coexistence_virtual.png` |
| Fig. 3.26 | 3.5.6 | `05_computational_scenarios` | `scenarios5_6_virtual_coexistence_experiment.py` | `figures/scenario6_virtual_experiment_noise.png` |
| Fig. 3.27 | 3.5.7 | `05_computational_scenarios` | `scenario7_calibrated_dose_response.py` | `figures/scenario7_dose_response_calibrated.png` |
| Fig. C.1 | Appendix C | `04_treatment_calibration` | `figures_thesis/plot_model_vs_data.py` | `.../TMC/figures/jilani_model_vs_data.png` |
| Fig. C.2 | Appendix C | `04_treatment_calibration` | `figures_thesis/plot_mechanistic_chains.py` | `.../TMC/figures/jilani_tmc_chain_full.png` |
| Fig. C.3 | Appendix C | `04_treatment_calibration` | `figures_thesis/plot_appendix_chains.py` (needs the chains of the data deposit) | `.../TMC/figures/chains_selected_models.png` |
| Table 3.1 | 3.2 | — | parameters from the literature | — |
| Table 3.2 | 3.4 | `04_treatment_calibration` | `structural_id.jl` and `structural_id_pipeline.jl` | `.../TMC/structural_identifiability_julia.txt` |
| Table 3.3 | 3.4 | `04_treatment_calibration` | `mle_bounded.py` | `.../TMC/mle_bounded_summary.csv` |
| Table A.1 | Appendix A | `01_model_analysis` | `lyapunov_coefficient.py` (checks T1–T5) | `results/lyapunov_coefficient.txt` |
| Table A.2 | Appendix A | `01_model_analysis` | `lyapunov_coefficient.py` (check T6) | `results/lyapunov_amplitude.csv` |
| Table B.1 | Appendix B | `02_sensitivity_analysis` | `efast_dummy_test.py` then `make_dummy_test_table.py` | `results/dummy_test_table.tex` |
| Tables C.1 and C.2 | Appendix C | `03_growth_model_selection` | `pipeline_control_parallel` and `pipeline_shared_params_parallel` | BIC rankings of the growth laws |
| Table C.3 | Appendix C | `04_treatment_calibration` | `compute_metrics.py` then `figures_thesis/make_model_table.py` | `.../TMC/model_table_cap3.tex` |

Results of the chapter that are not figures, all of them in Section 3.4 except the last: the BIC ranking of the
growth laws and of the sixteen treatment terms (`03_growth_model_selection` and
`04_treatment_calibration/compute_metrics.py`), the structural identifiability report (`structural_id.jl` and
`structural_identifiability.py`), the bounded maximum-likelihood estimates (`mle_bounded.py`), the experimental-design
scan and the joint Fisher information with the persistence data (`experimental_design.py`, `fim_joint_TC.py` and
`fim_joint_TM.py`), and the criticality of the Hopf boundary (Section 3.3.2, `lyapunov_coefficient.py`,
`bautin_alpha1_sweep.py` and `subcritical_bistability.py`).

Four figures left the thesis in the rewrite of Section 3.4 of 2026-10-14 and are still produced by this folder, since
the analysis they belong to is archived for a paper of its own (see
`TESE/NovoModeloBiometria/docs/baseline_cap3_identifiability_2026-10-14/`): `structural_symmetry_demo.png`
(`figures_thesis/plot_symmetry_demo.py`), `fim_eigenvalue_spectrum.png` (`figures_thesis/plot_fim_spectrum.py`),
`experimental_design_beta1.png` (`figures_thesis/plot_experimental_design.py`) and `tau_full_vs_reduced.png`
(`figures_thesis/plot_tau_comparison.py`, which needs the chains of the deposit).

## Reproducing the results

[`REPRODUCE.md`](REPRODUCE.md) lists the command that produces each figure of the chapter.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Each script is run from its own folder, for example:

```bash
cd 01_model_analysis
python stability_maps_and_hopf_boundary.py
```

The Bayesian calibrations of step 3 and step 4 are long runs (hundreds of thousands of MCMC iterations per model and
group) and were executed on the Lonestar6 supercomputer with the SLURM scripts provided. The analyses downstream of the
calibration (maximum-likelihood estimation, identifiability and scenarios) run on a workstation. The
maximum-likelihood parameter files (`multi_ll_pars_*.npz`, `mle_bounded_*.npz`) and the profile-likelihood solutions
(`profile_bounded_*.csv`, `profile_bounded_theta.npz`) are included, so the identifiability and scenario
analyses do not require the MCMC chains. The chains are needed to regenerate the posterior summaries, the
autocorrelation figure, and the starting points of `mle_bounded.py`. `profiles_bounded.py` is the longest local run
(about seven hours on 15 cores), because the likelihood of the mechanistic model is multimodal and every grid value is
re-optimized from several starting points.

## Performance

The calibrations (`03_growth_model_selection`, `04_treatment_calibration`) integrate the models with a compiled
Dormand–Prince solver (`compiled_ode.py`, default `CALIB_SOLVER=compiled`), with an automatic fallback to `odeint`
when it fails; on posterior samples of the thesis chains the log-likelihoods agree with those of `odeint` within
3e-5. `CALIB_SOLVER=odeint` restores scipy. The eFAST analysis (`02_sensitivity_analysis`) integrates its 100,000
trajectories with a compiled solver in parallel on all cores, in a few minutes.

## Units

The calibration steps use tumor volume in mm³ (1 mm³ ≈ 10⁶ cells); intensive rates are unchanged
and extensive quantities scale accordingly. The CAR T-cell dose is normalized so that `C(0) = 1` corresponds to the dose
administered in the experiment of Jilani et al. (2024). Steps 1, 2 and Scenarios 1–3 use the cell-based parameters of
Table 3.1 of the thesis.

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it, please cite this deposit,
the thesis and the article in which the model was first published; the three citations are in
`CITATION.cff`.

<!-- After the first Zenodo release, paste the DOI badge of the record on the line below. -->
