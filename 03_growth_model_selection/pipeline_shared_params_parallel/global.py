"""Driver of the analysis of parameter sharing between the control groups.

For each model in CALIB_MODELS and each scenario in CALIB_SCENARIOS:
  - builds the multi-group theta of `scenarios.SCENARIOS`
  - runs emcee with log_probability_shared (resumable HDF5 backend)
  - writes the thinned chain and the maximum-likelihood parameters to Output_Calibration/

Settings (CALIB_* environment variables):
  CALIB_MODELS=_exp,_log              (default; models to compare)
  CALIB_SCENARIOS=1,4                 (for _exp; 1,2,3,4 for _log)
  CALIB_CHAIN_SIZE=30000              (total number of steps; resumable)
  CALIB_WALKERS_PER_DIM=4
  CALIB_USE_POOL=1
  CALIB_N_CORES=<auto>
  CALIB_BACKEND_ONLY=0                (1 = sampling only, no comparison)
  CALIB_COMPRESS_H5=0
  CALIB_TEST=0                        (test mode: 50 steps)
"""
import gc
import os
from multiprocessing import cpu_count, get_context

import emcee
import numpy as np

# This import adds pipeline_control_parallel/ to sys.path; it must precede
# "from bounds/data_loader/... import ...".
import _path_setup
from data_loader import load_control_data
from bounds import find_max_initial_condition
from postprocess import compress_h5_inplace

from bounds_shared import define_shared_bounds_labels
from inference_shared import log_probability_shared
from metrics_shared import compare_scenarios, print_comparison_table
from scenarios import SCENARIOS, scenario_id


# Output folders
os.makedirs('./Output_Calibration', exist_ok=True)
os.makedirs('./reinitiate_files', exist_ok=True)

# PBS data of the two groups (same files as the growth-model selection)
GROUPS = ['control_b16', 'control_pdx']
full_data = load_control_data(data_dir=_path_setup.DATA_DIR)
max_ic = find_max_initial_condition(full_data)[0]


def _parse_scenarios(env_value, model_extension):
    """Comma-separated string -> sorted scenario numbers, validated against SCENARIOS[model_extension]."""
    valid = set(SCENARIOS[model_extension].keys())
    if not env_value:
        return sorted(valid)
    requested = [int(x) for x in env_value.split(',') if x.strip()]
    invalid = [s for s in requested if s not in valid]
    if invalid:
        raise ValueError(
            f'Invalid scenarios for {model_extension}: {invalid}. '
            f'Valid: {sorted(valid)}'
        )
    return sorted(set(requested))


if __name__ == '__main__':
    for g, mice in full_data.items():
        print(f'{g}: {len(mice)} mice -> {[m["name"] for m in mice]}')

    # --- Settings from environment variables ---
    MODELS_TO_FIT = os.environ.get('CALIB_MODELS', '_exp,_log').split(',')
    CHAIN_SIZE     = int(os.environ.get('CALIB_CHAIN_SIZE', '1000'))
    USE_POOL       = os.environ.get('CALIB_USE_POOL', '1') == '1'
    N_CORES        = int(os.environ.get('CALIB_N_CORES', str(cpu_count())))
    BACKEND_ONLY   = os.environ.get('CALIB_BACKEND_ONLY', '0') == '1'
    COMPRESS_H5    = os.environ.get('CALIB_COMPRESS_H5', '0') == '1'
    COMPRESS_LEVEL = int(os.environ.get('CALIB_COMPRESS_LEVEL', '4'))
    WALKERS_PER_DIM = int(os.environ.get('CALIB_WALKERS_PER_DIM', '4'))

    TEST_MODE = os.environ.get('CALIB_TEST', '0') == '1'
    if TEST_MODE:
        MODELS_TO_FIT = ['_exp']
        CHAIN_SIZE = 50
        BACKEND_ONLY = True
        print("[TEST MODE] _exp + scenarios 1,4 + 50 steps + backend-only; "
              "files prefixed with TEST_.")
    PREFIX = 'TEST_' if TEST_MODE else ''

    # Solver settings (read by models.py at import)
    from models import SOLVER, SOLVER_METHOD
    print(f"[config] models={MODELS_TO_FIT} groups={GROUPS} "
          f"chain={CHAIN_SIZE} pool={USE_POOL} n_cores={N_CORES} "
          f"walkers_per_dim={WALKERS_PER_DIM} solver={SOLVER}"
          + (f"({SOLVER_METHOD})" if SOLVER == 'solve_ivp' else "")
          + f" backend_only={BACKEND_ONLY} compress_h5={COMPRESS_H5}"
          + (' [TEST]' if TEST_MODE else ''))

    for model_extension in MODELS_TO_FIT:
        scenarios_env = os.environ.get(f'CALIB_SCENARIOS_{model_extension[1:].upper()}',
                                       os.environ.get('CALIB_SCENARIOS', ''))
        scenarios_to_run = _parse_scenarios(scenarios_env, model_extension)
        print(f'\n>>> Model {model_extension}: scenarios {scenarios_to_run}')

        for snum in scenarios_to_run:
            sdef = SCENARIOS[model_extension][snum]
            tag = scenario_id(snum)  # ex.: '_s1', '_s4'
            stem = f'{model_extension}{tag}'
            print(f'\n--- {stem}  (scenario def = {sdef}) ---')

            l_bound, u_bound, labels = define_shared_bounds_labels(
                model_extension=model_extension,
                scenario=sdef,
                groups=GROUPS,
                full_data=full_data,
                max_ic=max_ic,
            )
            ndim = len(l_bound)
            nwalkers = WALKERS_PER_DIM * ndim
            print(f'    ndim={ndim} ({len(sdef)*1} model params + ICs + std), '
                  f'nwalkers={nwalkers}, labels[:8]={labels[:8]}')

            filename = f'./reinitiate_files/reinitiate_{PREFIX}{stem}.h5'

            if not os.path.exists(filename):
                print('    HDF5 file not found. Creating a new backend...')
                backend = emcee.backends.HDFBackend(filename)
                backend.reset(nwalkers, ndim)
                pos = np.zeros((nwalkers, ndim))
                for i in range(ndim):
                    pos[:, i] = np.random.uniform(
                        low=l_bound[i], high=u_bound[i], size=nwalkers,
                    )
            else:
                print(f'    HDF5 file found: {filename}')
                backend = emcee.backends.HDFBackend(filename)
                print(f'    Current progress of the backend: {backend.iteration} iterations.')
                pos = None

            additional_chain_size = max(0, CHAIN_SIZE - backend.iteration)
            SKIPPED = (additional_chain_size == 0)

            if SKIPPED:
                print(f'    [skip] {stem}: backend already at {backend.iteration} '
                      f'steps (target {CHAIN_SIZE})')
                if COMPRESS_H5:
                    sb = os.path.getsize(filename)
                    compress_h5_inplace(filename, complevel=COMPRESS_LEVEL)
                    sa = os.path.getsize(filename)
                    print(f'    [compress] {os.path.basename(filename)}: '
                          f'{sb/1e6:.1f}MB -> {sa/1e6:.1f}MB '
                          f'({sb/max(sa,1):.1f}x)')
                if BACKEND_ONLY:
                    gc.collect(); continue
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_shared,
                    args=(l_bound, u_bound, model_extension,
                          full_data, GROUPS, sdef),
                    backend=backend,
                )
            else:
                print(f'    Calibrating {stem}: {additional_chain_size} new '
                      f'steps (backend={backend.iteration} -> target {CHAIN_SIZE})')

                def _run_sampler(pool):
                    sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability_shared,
                        args=(l_bound, u_bound, model_extension,
                              full_data, GROUPS, sdef),
                        pool=pool, backend=backend,
                    )
                    if backend.iteration > 0:
                        sampler.run_mcmc(None, additional_chain_size,
                                         progress=True)
                    else:
                        sampler.run_mcmc(pos, additional_chain_size,
                                         progress=True)
                    return sampler

                if USE_POOL:
                    mp_ctx = get_context('spawn')
                    print(f'    Calibration using {N_CORES} cores (spawn)')
                    with mp_ctx.Pool(processes=N_CORES) as pool:
                        sampler = _run_sampler(pool)
                else:
                    print('    Calibration in serial mode')
                    sampler = _run_sampler(None)

                if COMPRESS_H5:
                    sb = os.path.getsize(filename)
                    compress_h5_inplace(filename, complevel=COMPRESS_LEVEL)
                    sa = os.path.getsize(filename)
                    print(f'    [compress] {os.path.basename(filename)}: '
                          f'{sb/1e6:.1f}MB -> {sa/1e6:.1f}MB '
                          f'({sb/max(sa,1):.1f}x)')

                if BACKEND_ONLY:
                    print(f'    [backend-only] {stem}: HDF5 in {filename}')
                    del sampler, backend
                    gc.collect(); continue

            # === Outputs for model selection ===
            flat_ll = sampler.get_log_prob(flat=True)
            flat_chain = sampler.get_chain(flat=True)
            best_pars = flat_chain[flat_ll.argmax(axis=0)]
            total_iters = int(backend.iteration)
            discard_burnin = total_iters // 4
            kept_samples = (total_iters - discard_burnin) * nwalkers
            thin = max(1, kept_samples // 10000)

            save_chain = sampler.get_chain(
                discard=discard_burnin, flat=True, thin=thin,
            )
            np.savetxt(
                f'./Output_Calibration/multi_chain_{stem}.gz', save_chain,
            )
            np.savez(
                f'./Output_Calibration/multi_ll_pars_{stem}.npz',
                max_ll=max(flat_ll), pars=best_pars, labels=np.array(labels),
            )

            print(f'    [ok] {stem}: max_ll={max(flat_ll):.2f}, '
                  f'chain saved ({save_chain.shape[0]} samples).')

            del sampler, backend, flat_ll, flat_chain, save_chain
            gc.collect()

    # === Comparison of the scenarios ===
    if not BACKEND_ONLY:
        for model_extension in MODELS_TO_FIT:
            rows = compare_scenarios(
                out_dir='./Output_Calibration',
                full_data=full_data, groups=GROUPS,
                model_extension=model_extension,
                scenarios_to_compare=SCENARIOS[model_extension],
                compute_dic=True,
            )
            print_comparison_table(rows, model_extension)
