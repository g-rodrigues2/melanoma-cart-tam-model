# 04 — Treatment calibration and identifiability (Section 3.4)

Calibration of the CAR T-cell groups of Jilani et al. (2024) (short, medium and long hinge) and identifiability analysis
of the mechanistic model. Tumor volume in mm³; macrophage kinetics fixed (`α2 = 0.1`, `K2 = 1000`, `κ2 = 0.34`);
`C(0) = 1` is the administered dose; `α1`, `K1` and `β1` are sampled in log10 scale.

## Core modules

| Module | Role |
|---|---|
| `data_loader.py` | Reads `../data/processed/` and groups the mice (`cart_ss_b16`, `cart_sm_b16`, `cart_sl_b16`). |
| `models.py` | Empirical treatment models (`_exp_*`, exponential growth with 16 candidate killing terms) and mechanistic models `_mac` (`T`-`M`) and `_mac_treat` (`T`-`M`-`C`). |
| `bounds.py` | Prior bounds of every model. |
| `inference.py` | Log-prior, Gaussian log-likelihood and log-posterior. |
| `metrics.py`, `compute_metrics.py` | CCC/PCC, BIC, BIC weights, DIC, WBIC. |

## Bayesian calibration

| Script | Role |
|---|---|
| `global.py` | MCMC calibration (`emcee`, stretch and differential-evolution moves) of every model and group; resumable HDF5 backends. `treatment_parallel.slurm` and `treatment_test.slurm` reproduce the Lonestar6 runs. |
| `extract_chains.py`, `postprocess.py` | Maximum-likelihood parameters (`Output_Calibration/multi_ll_pars_*.npz`), posterior summaries, compressed chains. |
| `convergence_check.py`, `gen_chain_plots.py`, `gen_maxll_plots.py`, `posterior_histogram.py`, `posterior_physical_stats.py` | Autocorrelation times, trace and corner plots, best fits, posterior histograms. |
| `prior_predictive.py` | Prior predictive check. |

## Identifiability of the mechanistic model (`_mac_treat`)

Run in this order after the calibration:

| Script | Result | Output (`Output_Calibration/TMC/`) |
|---|---|---|
| `structural_id.jl`, `structural_id_pipeline.jl` | Global structural identifiability with `StructuralIdentifiability.jl`, with all quantities free and in the calibrated configuration. The environment is pinned in `Project.toml` and `Manifest.toml` of this folder, which the scripts activate and instantiate themselves. | `structural_identifiability_julia.txt` |
| `structural_identifiability.py` | Local structural identifiability (sensitivity rank). | printed report |
| `lsq_fit.py` | Bounded nonlinear least squares (trust-region reflective) with the residual standard deviation concentrated out; shared by the two scripts below. | — |
| `mle_bounded.py` | Maximum-likelihood estimates of `_mac_treat` and `_mac` with the prior bounds enforced, 41–42 starts per model and group (MCMC maximum-likelihood sample, posterior samples, earlier quasi-Newton estimate). | `mle_bounded_<group>.npz`, `mle_bounded_<group>_mac.npz`, `mle_bounded_summary.csv` |
| `profiles_bounded.py` | Profile likelihoods of `β1` (both mechanistic models) and `μ1` (`_mac_treat`): continuation from the estimate, neighbour sweeps until no grid value improves, release of all parameters when a profile exceeds the estimate. | `profile_bounded_grid.csv`, `profile_bounded_summary.csv`, `profile_bounded_theta.npz` |
| `profiles_verify.py` | Verification pass of the long-hinge profiles: every grid value whose drop exceeds the 1.92 threshold is re-optimized from the solutions of the other profile, which span the likelihood ridge. | printed report |
| `fim_identifiability_map.py` | Fisher information at the bounded estimate: marginal uncertainties and eigenvalues. | `fim_identifiability_map.csv`, `fim_eigenvalues_<group>.csv` |
| `fim_joint_TC.py`, `fim_joint_TM.py` | Fisher information after adding the CAR T-cell persistence data and/or hypothetical macrophage measurements. `joint_fit_TC.py` provides the persistence likelihood. | `fim_joint_TC.csv`, `fim_joint_TM.csv` |
| `experimental_design.py` | In silico design: uncertainty of `β1` and `μ1` by observed variables and observation window, in the escape and coexistence regimes. | `experimental_design.csv`, `experimental_design_diagnostic.png` |
| `reduced_fit.py` | MCMC calibration of the minimal identifiable submodel (`β1` fixed, `C(t) = exp(−0.09 t)` prescribed). | `reduced_mu1.csv` |

## Thesis figures (`figures_thesis/`)

`_paths.py` resolves the locations of the calibration outputs (`Output_Calibration/TMC/` and its `figures/`
subfolder, where the thesis figures are written) for every script in this folder.

| Script | Figure |
|---|---|
| `plot_jilani_data.py` | Group means of the five arms of the B16 experiment with the exponential growth law of the controls. |
| `plot_symmetry_demo.py` | Numerical demonstration of the scaling symmetry between `β1`, `M(0)` and `K2`. |
| `plot_model_vs_data.py` | Selected empirical killing terms against the data, one panel per group (`jilani_model_vs_data.png`). |
| `plot_mechanistic_vs_data.py` | The `_mac_treat` and `_mac` models against the data at their bounded maximum-likelihood estimates, with the log-likelihoods, the likelihood-ratio statistic and the CCC (`jilani_mechanistic_vs_data.png`). |
| `plot_profile_likelihoods.py` | Profile likelihoods of `β1` (both models) and `μ1`, one column per group (`profile_likelihoods.png`). |
| `plot_fim_spectrum.py` | Eigenvalue spectrum of the Fisher information (long hinge). |
| `plot_experimental_design.py` | Uncertainty of `β1` by observation scheme and window. |
| `plot_tau_comparison.py` | Integrated autocorrelation times of the full model and of the submodel (requires the MCMC backend). |
