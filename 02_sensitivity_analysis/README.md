# 02 — Global sensitivity analysis (Section 3.3.3)

| Script | Result |
|---|---|
| `efast_sensitivity.py` | eFAST sampling (`SALib`, base sample size 10 000, harmonic order 4, seed 123), integration of the model on `[0, 60]` days, first-order and total-effect indices at the final time and every 20 time points. Writes `results/`. |
| `plot_efast_sensitivity.py` | Bar plot of the final-time indices (`figures/fig9a_b.pdf`) and stacked plot of the normalized time-dependent total-effect indices (`figures/fast_TMC_ST_TMC_ST_stacked_time.png`). |
| `efast_dummy_test.py` | Dummy-parameter significance test of Marino et al. (2008): the same design plus an eleventh inert parameter, repeated over 5 independent search curves, and a one-sided Welch *t*-test of every index against the dummy at the Bonferroni level `0.05/10`. Writes `results/dummy_test_end.csv`, `results/dummy_test_time.csv` and `results/dummy_test_replicates.npz`. |
| `make_dummy_test_table.py` | LaTeX body of the appendix table of Chapter 3 from `results/dummy_test_end.csv` (`results/dummy_test_table.tex`). |
| `efast_horizon.py` | Same design as `efast_sensitivity.py`, with the output read at 60, 100, 150, 200, 300 and 400 days instead of only at 60. The total-effect index of `beta1` rises from 0.003 to 0.68 over that range, because near the Hopf threshold the cycle amplitude opens as the square root of the distance to it, so sixty days is too short for `beta1` to act. The output stays well behaved (no saturation at `K1` at any horizon, same five leading parameters on `T` and on `log T`). Sixty days is the only horizon of experimental relevance, so the chapter reports it; the rest bounds the scope of the ranking. Writes `results/horizon_*`. |
| `efast_sensitivity_table31.py` | Companion design over the uncertainty **Table 3.1 itself declares** (`alpha1` in [0.69, 0.97], `beta1` log-uniform over five decades, `kappa3` in [0, 0.9], `mu3` in [0, 5e-9] and therefore included, so `p = 11`; the remaining eight at +-20%). Writes `results/table31_*`. It is a scope check, not a result of the chapter: over those ranges `T(60)` becomes a switch rather than a graded output (72.9% of the samples saturate within 1% of `K1`, 2.3% fall below a single cell), so its indices are reported in the thesis only qualitatively. What it establishes is that the +-20% ranking does not extend to the declared ranges, with `kappa3`, `beta1` and `mu3` leading there and `alpha1`, `alpha3` falling out; this is stable across seeds and across the scale of the output. |
| `plot_dummy_test.py` | Time-resolved significance of the total-effect indices from `results/dummy_test_time.csv` (`figures/dummy_test_significance_time.png`): which parameters are distinguishable from the dummy at each analyzed time, and the fraction of times at which each one is. |

The ten parameters `α1, α2, α3, β1, K1, K2, K3, μ1, κ2, κ3` are sampled uniformly within ±20% of their reference values
(`α1 = 0.69`, `β1 = 5.9×10⁻¹¹`, `κ3 = 0.35` for the parameters given as intervals in Table 3.1). The immunosuppression
term is not included (`μ3 = 0`), consistently with the analysis of the coexistence equilibrium.

Run `efast_sensitivity.py` before `plot_efast_sensitivity.py`. The 100,000 trajectories are integrated by a compiled Dormand-Prince solver in parallel on all cores (a few minutes on a workstation; relative difference to `odeint` below 1e-9, printed by the script); `results/fast_model_outputs_TMC.npz` keeps the trajectories at the analyzed times only.

`efast_dummy_test.py` is independent of the two scripts above and integrates 550,000 trajectories (5 curves × 110,000),
which the compiled solver does in about 2.6 min on 16 cores. It is the statistical counterpart of the `0.05` screening
cutoff used in the chapter, and the two criteria are reported side by side: because the variance the *t*-test compares is
the variance across search curves at fixed `N`, at `N = 10,000` the test is close to fully powered and calls significant
indices as small as `S1 ~ 1e-5`. Its useful role is therefore factor fixing, not ranking. Run `make_dummy_test_table.py`
afterwards to regenerate the appendix table.
