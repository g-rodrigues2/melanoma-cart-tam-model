# pipeline_shared_params_parallel/ — Parameter sharing between the control groups

Companion to `pipeline_control_parallel/`. It tests whether the parameters of the exponential (`_exp`) and
logistic (`_log`) models can be shared between the control groups `control_b16` (26 mice) and `control_pdx`
(20 mice), and provides the posterior of the B16 growth rate used in the thesis.

## Sharing

A parameter marked `'global'` in a scenario has a single value calibrated with the data of both groups: theta
holds one entry for it (instead of one per group), and `log_likelihood_shared` sums the Gaussian log-likelihoods
of all mice of both groups. A `'group'` parameter has one value per group, informed only by that group. The
initial volumes are always per mouse and the residual standard deviation `std` is always global.

| Scenario | Model | `r` | `K` (`cc`) | Model parameters |
|---|---|---|---|---|
| 1 | `_exp` | per group | — | 2 |
| 4 | `_exp` | global | — | 1 |
| 1 | `_log` | per group | per group | 4 |
| 2 | `_log` | per group | global | 3 |
| 3 | `_log` | global | per group | 3 |
| 4 | `_log` | global | global | 2 |

`ndim = model parameters + 46 initial volumes + 1 std`.

## Results

BIC with n = 314 observations (maximum-likelihood samples in `Output_Calibration/`):

| Stem | k | max log L | BIC |
|---|---|---|---|
| `_exp_s1` | 49 | −1978.23 | **4238.19** |
| `_exp_s4` | 48 | −2110.44 | 4496.86 |
| `_log_s1` | 51 | −1982.96 | 4259.13 |
| `_log_s2` | 50 | −1983.72 | 4254.91 |
| `_log_s3` | 50 | −2053.75 | 4394.97 |
| `_log_s4` | 49 | −2112.83 | 4507.39 |

The exponential model with group-specific rates (`_exp_s1`) has the lowest BIC. Its B16 growth rate has posterior
median `r_b16 = 0.2522` day⁻¹ (95% interval `[0.2352, 0.2663]`) and maximum-likelihood value `0.2648` day⁻¹; the
PDX rate is `0.1009` `[0.0949, 0.1085]`. `generate_best_figures.py` writes the fits and parameter summaries of
`_exp_s1` and `_log_s2`.

## Modules

```
_path_setup.py          adds pipeline_control_parallel/ to sys.path; data folder
scenarios.py            SCENARIOS and the partition of theta
bounds_shared.py        define_shared_bounds_labels(model, scenario, groups, ...)
inference_shared.py     log_likelihood_shared / log_probability_shared
metrics_shared.py       BIC, DIC and compare_scenarios
plots_shared.py         scatter plots and posterior summaries
posterior_histogram.py  posterior histograms
generate_best_figures.py figures of the selected scenarios
global.py               calibration driver
shared_parallel.slurm   Lonestar6 job
tests/                  pytest (scenarios, bounds, inference)
```

`models.py`, `bounds.py`, `data_loader.py`, `postprocess.py` and `convergence_check.py` are reused from
`pipeline_control_parallel/`. The PBS data are read from `../../data/processed/`.

## Running

```bash
CALIB_TEST=1 python global.py                                        # 50-step test
CALIB_MODELS=_log CALIB_SCENARIOS=4 CALIB_CHAIN_SIZE=5000 python global.py
CALIB_CHAIN_SIZE=30000 python global.py                              # all scenarios
sbatch shared_parallel.slurm                                         # 6 processes, 20 cores each
python generate_best_figures.py
python -m pytest tests -q
```

Settings are those of `pipeline_control_parallel/` plus `CALIB_SCENARIOS` (comma-separated; default: all valid
scenarios of the model) and `CALIB_SCENARIOS_EXP` / `CALIB_SCENARIOS_LOG` (per-model overrides when both models
run in one process). The backends are resumable. After sampling, `global.py` prints the BIC and DIC of the
scenarios of each model.
