"""Driver of the MCMC calibration of the tumor growth models.

Loads the data (data_loader), defines the bounds (bounds), builds the emcee sampler with the log-probability
(inference), writes the HDF5 backend, optionally compresses it, and produces the figures (postprocess) and the
convergence diagnostics (convergence_check).

Configured with CALIB_* environment variables (see the README of the step and control_parallel.slurm). Run as
``python global.py``; creates ``reinitiate_files/`` and ``Output_Calibration/``.
"""
import gc
import os
from multiprocessing import cpu_count, get_context

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

# Pipeline modules
from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_control_data
from inference import log_probability
from initialization import initial_positions
from walkers import drop_stuck
from postprocess import (
    compress_h5_inplace,
    plot_maxll_solution_global,
    posterior_corr_from_chain,
)


# Output folders
os.makedirs('./Output_Calibration', exist_ok=True)
os.makedirs('./reinitiate_files', exist_ok=True)

# Control (PBS) data; the module can be imported without running __main__
full_data = load_control_data()
max_ic = find_max_initial_condition(full_data)[0]


if __name__ == "__main__":
    for g, mice in full_data.items():
        print(f"{g}: {len(mice)} mice -> {[m['name'] for m in mice]}")

    # --- Settings from environment variables ---
    MODELS_TO_FIT = os.environ.get(
        'CALIB_MODELS', '_exp,_mendel,_bert,_lin,_gomp,_surf,_log').split(',')
    CONTROL_GROUPS = os.environ.get('CALIB_GROUPS', 'control_b16,control_pdx').split(',')
    CHAIN_SIZE     = int(os.environ.get('CALIB_CHAIN_SIZE', '1000'))
    USE_POOL       = os.environ.get('CALIB_USE_POOL', '1') == '1'
    N_CORES        = int(os.environ.get('CALIB_N_CORES', str(cpu_count())))
    BACKEND_ONLY   = os.environ.get('CALIB_BACKEND_ONLY', '0') == '1'
    COMPRESS_H5    = os.environ.get('CALIB_COMPRESS_H5', '0') == '1'
    COMPRESS_LEVEL = int(os.environ.get('CALIB_COMPRESS_LEVEL', '4'))
    MAKE_CORNER    = os.environ.get('CALIB_CORNER', '1') == '1'
    WALKERS_PER_DIM = int(os.environ.get('CALIB_WALKERS_PER_DIM', '2'))
    # Stuck walkers: initialization in a ball around a MAP estimate and removal.
    INIT_MODE      = os.environ.get('CALIB_INIT', 'ball').lower()
    DROP_STUCK     = os.environ.get('CALIB_DROP_STUCK', '1') == '1'

    # Test mode
    TEST_MODE = os.environ.get('CALIB_TEST', '0') == '1'
    if TEST_MODE:
        MODELS_TO_FIT  = ['_exp']
        CONTROL_GROUPS = ['control_pdx']
        CHAIN_SIZE     = 50
        BACKEND_ONLY   = True
        print("[TEST MODE] _exp + control_pdx + 50 steps + backend-only; "
              "files prefixed with 'TEST_'.")
    REINIT_PREFIX = "TEST_" if TEST_MODE else ""

    # Solver settings (read by models.py at import)
    from models import SOLVER, SOLVER_METHOD
    print(f"[config] models={MODELS_TO_FIT} groups={CONTROL_GROUPS} "
          f"chain={CHAIN_SIZE} pool={USE_POOL} n_cores={N_CORES} "
          f"walkers_per_dim={WALKERS_PER_DIM} solver={SOLVER}"
          + (f"({SOLVER_METHOD})" if SOLVER == 'solve_ivp' else "")
          + f" backend_only={BACKEND_ONLY} compress_h5={COMPRESS_H5}"
          + (f" (level={COMPRESS_LEVEL})" if COMPRESS_H5 else "")
          + f" corner={MAKE_CORNER}"
          + (" [TEST]" if TEST_MODE else ""))

    for model_extension in MODELS_TO_FIT:
        print(f"Solving model {model_extension}")
        for group in CONTROL_GROUPS:
            print(f"Solving group {group}")
            group_size = len(full_data[group])
            l_bound, u_bound, labels = define_bounds_labels(
                model_extension, group_size=group_size, max_ic=max_ic,
                bool_multiple_mice=True,
            )
            ndim = len(l_bound)
            nwalkers = WALKERS_PER_DIM * ndim

            filename = f"./reinitiate_files/reinitiate_{REINIT_PREFIX}{group}{model_extension}.h5"

            if not os.path.exists(filename):
                print("HDF5 file not found. Creating a new file...")
                backend = emcee.backends.HDFBackend(filename)
                backend.reset(nwalkers, ndim)
                # 'ball' (default): walkers around an approximate MAP estimate. Uniform initialization over
                # the prior box left walkers stuck in a high-std mode (~7% of the samples), which distorted
                # DIC and WBIC. CALIB_INIT=uniform reproduces the original behavior.
                pos, _info = initial_positions(
                    nwalkers, l_bound, u_bound, log_probability,
                    (l_bound, u_bound, model_extension, full_data, group),
                    mode=INIT_MODE)
                print(f"[init] {_info['mode']}")
            else:
                print(f"HDF5 file found: {filename}")
                backend = emcee.backends.HDFBackend(filename)
                print(f"Current progress on the backend: {backend.iteration} iterations.")
                pos = None

            # CHAIN_SIZE is the total number of steps; run only the missing ones
            additional_chain_size = max(0, CHAIN_SIZE - backend.iteration)
            SKIPPED_MCMC = (additional_chain_size == 0)

            if SKIPPED_MCMC:
                print(f"[skip] {group}{model_extension}: backend already at "
                      f"{backend.iteration} steps (target {CHAIN_SIZE})")
                if COMPRESS_H5:
                    sb = os.path.getsize(filename)
                    compress_h5_inplace(filename, complevel=COMPRESS_LEVEL)
                    sa = os.path.getsize(filename)
                    print(f"[compress] {os.path.basename(filename)}: "
                          f"{sb/1e6:.1f} MB -> {sa/1e6:.1f} MB ({sb/max(sa,1):.1f}x)")
                if BACKEND_ONLY:
                    gc.collect(); continue
                # Sampler bound to the backend, for post-processing only
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability,
                    args=(l_bound, u_bound, model_extension, full_data, group),
                    backend=backend,
                )
            else:
                print(f'Calibrating {group} with {model_extension.split("_")[-1]}: '
                      f'{additional_chain_size} new steps '
                      f'(backend={backend.iteration} -> target={CHAIN_SIZE})')

                def _run_sampler(pool):
                    sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability,
                        args=(l_bound, u_bound, model_extension, full_data, group),
                        pool=pool, backend=backend,
                    )
                    if backend.iteration > 0:
                        sampler.run_mcmc(None, additional_chain_size, progress=True)
                    else:
                        sampler.run_mcmc(pos, additional_chain_size, progress=True)
                    return sampler

                if USE_POOL:
                    # 'spawn' avoids OSError [Errno 12] on Linux when the parent process is large
                    mp_ctx = get_context('spawn')
                    print(f'Calibration using {N_CORES} cores (spawn)')
                    with mp_ctx.Pool(processes=N_CORES) as pool:
                        sampler = _run_sampler(pool)
                else:
                    print('Calibration in serial mode')
                    sampler = _run_sampler(None)

                if COMPRESS_H5:
                    sb = os.path.getsize(filename)
                    compress_h5_inplace(filename, complevel=COMPRESS_LEVEL)
                    sa = os.path.getsize(filename)
                    print(f"[compress] {os.path.basename(filename)}: "
                          f"{sb/1e6:.1f} MB -> {sa/1e6:.1f} MB ({sb/max(sa,1):.1f}x)")

                if BACKEND_ONLY:
                    print(f"[backend-only] {group}{model_extension}: HDF5 in {filename}")
                    del sampler, backend
                    gc.collect(); continue

            # === Post-processing (BACKEND_ONLY=0) ===
            flat_ll = sampler.get_log_prob(flat=True)
            flat_chain = sampler.get_chain(flat=True)
            best_pars = flat_chain[flat_ll.argmax(axis=0)]
            total_iters = int(backend.iteration)
            discard_burnin = total_iters // 4

            # Stuck walkers are removed before flattening. They do not change max_ll (nor the BIC) but distort
            # medians, 95% intervals, DIC (theta_bar -> p_D < 0) and WBIC. CALIB_DROP_STUCK=0 disables it.
            chain3d = sampler.get_chain(discard=discard_burnin)
            lp2d = sampler.get_log_prob(discard=discard_burnin)
            if DROP_STUCK:
                chain3d, lp2d, _presos = drop_stuck(chain3d, lp2d)
                if _presos.any():
                    print(f"[walkers] {_presos.sum()}/{len(_presos)} stuck walkers "
                          f"removed (indices {list(np.flatnonzero(_presos))})")
            kept_samples = chain3d.shape[0] * chain3d.shape[1]
            scale_corner = max(1, kept_samples // 10000)

            save_chain = chain3d[::scale_corner].reshape(-1, chain3d.shape[2])
            np.savetxt(f'./Output_Calibration/multi_chain_{group}{model_extension}.gz', save_chain)
            np.savez(f'./Output_Calibration/multi_ll_pars_{group}{model_extension}.npz',
                     max_ll=max(flat_ll), pars=best_pars,
                     n_stuck=int(nwalkers - chain3d.shape[1]))

            if MAKE_CORNER:
                corner_discard = total_iters // 2
                corner_kept = (total_iters - corner_discard) * nwalkers
                corner_thin = max(1, corner_kept // 5000)
                corner_chain = sampler.get_chain(discard=corner_discard,
                                                 flat=True, thin=corner_thin)
                corner.corner(corner_chain, labels=labels)
                plt.savefig('./Output_Calibration/multi_corner_' + group + model_extension + '.png',
                            format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
                plt.close()

            # Trace plot: thinned to avoid MemoryError with long chains
            full_samples = sampler.get_chain()
            trace_thin = max(1, full_samples.shape[0] // 5000)
            samples = full_samples[::trace_thin]
            del full_samples
            fig, axes = plt.subplots(nrows=len(labels), ncols=1,
                                     figsize=(10, len(labels) * 2), sharex=True)
            axes = axes.ravel()
            for i in range(len(labels)):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel(f"step (thinned every {trace_thin})")
            plt.savefig('./Output_Calibration/multi_chain_' + group + model_extension + '.png',
                        format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close('all')
            del samples
            gc.collect()

            plot_maxll_solution_global(
                './Output_Calibration/multi_ll_pars_' + group + model_extension + '.npz',
                full_data, show=False, nCols=4,
                figure_name='./Output_Calibration/multi_max_ll_' + group + model_extension,
            )
            plt.close('all')
            gc.collect()

        # === After both groups of the model: combined figures and posterior correlations ===
        if BACKEND_ONLY:
            continue

        plot_maxll_solution_global(
            f'./Output_Calibration/multi_ll_pars_*{model_extension}.npz',
            full_data, show=False, nCols=4,
            figure_name=f'./Output_Calibration/multi_max_ll_both{model_extension}',
        )
        posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_b16' + model_extension + '.gz',
            model_extension=model_extension, full_data=full_data,
            group='control_b16', n_samp=500, seed=42,
        )
        posterior_corr_from_chain(
            chain_path='./Output_Calibration/multi_chain_control_pdx' + model_extension + '.gz',
            model_extension=model_extension, full_data=full_data,
            group='control_pdx', n_samp=500, seed=42,
        )

    # === Convergence diagnostics (end of post-processing) ===
    if not BACKEND_ONLY:
        try:
            from convergence_check import run_convergence_check
            print('\n=== CONVERGENCE DIAGNOSTICS ===')
            run_convergence_check(
                results_dir='.', models=MODELS_TO_FIT,
                groups=CONTROL_GROUPS, save_csv=True, print_table=True,
            )
        except Exception as _e:
            print(f'[WARN] convergence_check failed: {_e.__class__.__name__}: {_e}')
