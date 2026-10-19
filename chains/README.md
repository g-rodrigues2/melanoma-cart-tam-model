# MCMC chains

Posterior samples of the models selected in Chapter 3, written by [`export_chains.py`](../export_chains.py)
from the resumable HDF5 backends, which are not published. Chains of the models that were not selected are
not here; their maximum-likelihood samples are in the output folders of each step.

| Folder | Chains |
|---|---|
| `03_growth_model_selection/` | exponential growth with group-specific rates (sharing scenario 1), the calibration behind the B16 growth rate |
| `04_treatment_calibration/` | the empirical treatment models selected by BIC, namely no treatment term (`_exp`) for the short- and medium-hinge groups and the sinusoidally modulated term (`_exp_EDwithSin`) for the long-hinge group, and the T-M-C model (`_mac_treat`) for the three CAR T-cell groups |

`manifest.json` lists every file with the number of steps, the thinning interval, the numbers of walkers
and parameters, and the file size.

## Format

Each file is a compressed NumPy archive, read by `read_exported_chain` in `export_chains.py`:

| Key | Content |
|---|---|
| `chain` | array (stored steps, walkers, parameters), float32, every `thin`-th step of the chain |
| `log_prob` | array (stored steps, walkers), float32, log-posterior of the stored samples |
| `thin` | thinning interval |
| `n_steps` | number of steps of the original chain |
| `burn_in` | burn-in used in the thesis (first quarter of the steps), in original steps |

The parameter order is that of the calibration code (`bounds*.py` of each step). The posterior samples used
in the thesis are `chain[burn_in // thin:]`. The thinning is the largest of ten, the value that keeps at
most about five thousand stored steps, and the value that keeps each file under the size a repository file
may have; it is recorded in every file and the reader applies it, so the diagnostics are unaffected. For
the widest chain, that of the growth rate, doubling the thinning moves the largest autocorrelation time
from 1407.6 to 1410.3 steps and the split R-hat from 1.0895 to 1.0892.
