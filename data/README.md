# Data

All data come from the source data file of Jilani et al. (2024), *Nature Communications* 15, 1244
(supplementary file `41467_2024_45221_MOESM4_ESM.xlsx`), stored unchanged in `raw/jilani2024_source_data.xlsx`.

## `processed/` — tumor volume per mouse

One text file per mouse, two whitespace-separated columns: days after CAR T-cell administration and tumor volume (mm³).
File names follow `fig<panel>_<arm>_c<panel>_m<mouse>.txt`, where `<panel>` is the figure panel of the source data
and `<arm>` the experimental arm.

| Panel | Experiment | Arms | Used in the thesis |
|---|---|---|---|
| 3b | Immunocompetent B16 model, three TYRP1 CAR constructs | `pbs`, `utd`, `20d7ss-28z` (short hinge), `20d7sm-28z` (medium hinge), `20d7sl-28z` (long hinge) | PBS arm: growth-model selection; CAR arms: treatment calibration, identifiability and scenarios |
| 3e | B16, two doses of the long-hinge construct | `pbs`, `utd(3.5m)`, `utd(7m)`, `20d7sl-28z(3.5m)`, `20d7sl-28z(7m)` | PBS arm: growth-model selection; CAR arms: calibration of the dose axis |
| 3h | B16, long-hinge construct with and without IL-2 | `pbs`, `mock`, `mock+il2`, `20d7sl-28z`, `20d7sl-28z+il2` | PBS arm: growth-model selection |
| 3k, 3m | Patient-derived xenograft models | `pbs`, `cd19car`, `20d7sl-28z` | PBS arms: second control group of the growth-model selection pipeline (not reported in the thesis) |

The day-0 value reported for every mouse in the source data (13.5 mm³) is also the value that appears in some later
measurements, which suggests a lower reporting limit; the calibrations therefore estimate one initial tumor volume per mouse.

## `processed/jilani2024_fig6_car_t_persistence_vcn.csv`

CAR T-cell persistence measured as vector copy number (VCN) by PCR (source data panels 6b and 6c): mean and standard
deviation per day after administration, long-hinge construct.
