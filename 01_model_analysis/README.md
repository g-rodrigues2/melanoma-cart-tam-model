# 01 — Mathematical analysis of the model (Sections 3.3.1–3.3.2 and Appendix A)

All scripts use the parameters of Table 3.1 (cells), write their figures to `figures/` with the file names of the thesis, and write intermediate data to `results/`.

## Figures and computations

| Script | Thesis item | Result |
|---|---|---|
| `car_t_growth_balance.py` | Fig. 3.2 (`fig2.pdf`) | Comparison between `f3(T)` and `μ3 T + κ3` (Proposition 1): tangency point and the two roots. |
| `e3_fold_and_mu3_kappa3_data.py`, `e3_fold_and_mu3_kappa3_plot.py` | `E3_fold_bifurcation.png`, `mu3_kappa3_map.png` | Saddle-node diagram of the macrophage-free equilibrium `E3` in `μ3`, and stability map of `E2`/`E3` in the `(κ3, μ3)` plane. The first script computes the data (`figA_data.npz`, `figB_data.npz`), the second draws the figures. |
| `coexistence_branches.py` | Fig. 3.4 (`fig3.pdf`) | `M*(T*)` and `C*(T*)` along the coexistence branch, in the two regimes of Proposition 3.5, as a single 2×2 panel. |
| `stability_maps_and_hopf_boundary.py` | `stability_a2_a3.png` (3.3.1), `hopf_k3_b1.png` (3.3.2) | Stability regions in the `(α2, α3)` plane; classification of `E*` (unstable, spiral, sink) in the `(κ3, β1)` plane, with both branches of the Hopf boundary, the rate `κ3^c` above which the instability disappears, the Bautin point, and the first Lyapunov coefficient along the boundary. |
| `oscillatory_regime_kappa3.py` | Fig. 3.7 (`fig5.pdf`) | Range of `κ3` for which `E*` is a stable spiral when `β1 = 0` (Proposition 7), with the practical threshold of one oscillation per envelope half-life. |
| `spiral_and_sink.py` | Fig. 3.8 (`fig6.pdf`) | Damped oscillations versus monotone convergence around `E*` for `κ3 = 0.5` and `κ3 = 0.86`, with the `(T, C)` phase plane, which is a closed subsystem at `β1 = 0`. |
| `hopf_threshold.py` | Fig. 3.9 (`fig7.pdf`) | Quadratic structure of `Q(β1)` across sections of `κ3`, the Hopf threshold `β1* ≈ 5.9056×10⁻¹¹`, and the three Hopf conditions. |
| `hopf_phase_portraits.py` | Fig. 3.11 (`fig8.pdf`) | Hopf bifurcation diagram in `β1` and the family of limit cycles, the cycles solved for directly as periodic orbits by shooting on `Σ = {T = T*}` (`results/hopf_bifurcation_diagram.npz`). |
| `lyapunov_coefficient.py` | Appendix A | First Lyapunov coefficient at the Hopf point by the projection method, `l1 < 0` (supercritical), with the checks T1–T6 of the appendix, including the quantitative amplitude prediction. Writes `results/lyapunov_coefficient.txt` (the scalars of the appendix) and `results/lyapunov_amplitude.csv` (its amplitude table). |
| `bautin_alpha1_sweep.py` | Section 3.3.2 | Where the Hopf boundary is subcritical, sweeping `α1` far beyond the value of the figure: supports the claim that the subcritical stretch falls outside the range of `κ3` recovered from the data in Section 3.4. Writes `results/bautin_alpha1_sweep.txt`. |
| `subcritical_bistability.py` | Section 3.3.2 | Dynamical test of the subcritical Hopf below the Bautin point: unstable periodic orbit certified by its Floquet multipliers, its square-root amplitude law, and bistability by direct integration, with `κ3 = 0.35` as the supercritical control. Writes `results/subcritical_bistability.txt`. |

## Verification sweeps

Independent recomputations of the numbers that entered the thesis, by a route that does not reuse the floating-point
path of the scripts above: exact rationals in `sympy`, `mpmath` at 50–60 digits, and tighter integrators. Each one
prints `[OK]` or `[FAIL]` per check, ends with a summary line, and **exits with status 1 if any check fails**.

| Script | What it audits | Checks |
|---|---|---|
| `verify_chapter3_computations.py` | Equilibria, Jacobians and stability propositions against direct eigenvalue computations; the values reported in the chapter (`μ3max`, `α1_tilde`, `T_tilde*`, `β1*`, `μ3†` of Scenario 1, control durations of Scenario 2); solver independence of the phase portraits. | 7 reported values |
| `verify_hopf_threshold_claims.py` | Every numerical claim of Section 3.3.2: `β1*`, the eigenvalues and period at the threshold, the quadratic structure and root count of `Q`, `κ3^(1)`, `κ3^c`, the spiral/sink boundary and the existence limits. | 22 |
| `verify_oscillatory_regime.py` | First half of Section 3.3.2 (Figures 3.7 and 3.8): `α1_tilde`, `T_tilde*`, `Ω`, the existence interval, `κ3_tilde`, the cycle counts of the caption, the eigenvalues and half-lives of the two sections, the nonlinear trajectory against the local spectrum, and the algebra of Proposition P7. | 32 |
| `verify_hopf_figure_numbers.py` | Caption of Figure 3.9 (the section `κ3 = 0.64`, root counts) and the period `2π/ω = 16.19` days. | 9 |
| `verify_bautin_point.py` | The Bautin point at 50 digits with exact symbolic derivatives: `κ3^B ≈ 0.17757`, and the sign of `l1` on each side of it. | 4 |
| `verify_appendix_lyapunov.py` | Every number printed in Appendix A, recomputed at 60 digits: `β1*`, `E*`, the matrix `A`, the eigenvalues, `q` and `p`, `h11` and `h20`, the three terms of `g21`, `l1`, the cancellation factor, transversality and `q1`. | 41 |

## Tests

```bash
python -m pytest tests -q                 # 12 tests, about 4 min
python -m pytest tests -q -m "not slow"   # skips the verification sweeps
```

`tests/test_model_analysis.py` checks the analytical results directly and runs
`verify_chapter3_computations.py`; `tests/test_verification_sweeps.py` runs the five sweeps above and
`lyapunov_coefficient.py`, asserting the exit status, the absence of `[FAIL]`, and that the expected number of checks
actually ran.
