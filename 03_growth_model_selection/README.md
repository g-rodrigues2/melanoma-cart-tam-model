# 03 — Growth-model selection (Section 3.4.1)

Bayesian calibration of classical growth laws to the PBS control mice of the B16 experiments and selection by BIC.

| Folder | Content |
|---|---|
| `pipeline_control_parallel/` | Calibration of each growth law (exponential, logistic, Gompertz, von Bertalanffy, and others) for the control groups with `emcee`; one initial tumor volume per mouse; model selection by BIC/BIC weights. Entry point: `global.py`. |
| `pipeline_shared_params_parallel/` | Calibration scenarios in which the growth parameters are shared or group specific; posterior of the B16 growth rate (median `r_B16 = 0.2522` day⁻¹, 95% interval `[0.2352, 0.2663]`; the maximum-likelihood value `0.2648` day⁻¹ is fixed in the treatment models). Entry point: `global.py`. It reuses the modules of `pipeline_control_parallel/`. |

## Running

```bash
cd pipeline_control_parallel
CALIB_TEST=1 python global.py                       # short test run
CALIB_CHAIN_SIZE=300000 CALIB_N_CORES=16 python global.py
```

Besides the modules listed in the step 4 README (which this pipeline shares), `initialization.py` places the
`emcee` walkers in the region of posterior mass instead of spreading them over the whole prior box, and
`walkers.py` detects walkers stuck in spurious modes from the per-walker log-probability.

Relevant environment variables: `CALIB_MODELS`, `CALIB_GROUPS`, `CALIB_CHAIN_SIZE`, `CALIB_N_CORES`,
`CALIB_WALKERS_PER_DIM`, `CALIB_TEST`. The runs are resumable: resubmitting continues from the stored backends.
`control_parallel.slurm` and `shared_parallel.slurm` reproduce the Lonestar6 runs. Unit tests: `pytest tests/`.

Outputs are written to `Output_Calibration/` (figures, summaries, maximum-likelihood parameters). MCMC chains are
distributed separately (see the repository README).
