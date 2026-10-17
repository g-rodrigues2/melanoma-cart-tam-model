"""Driver of the MCMC calibration of the CAR T-cell groups.

Loads the treatment data (data_loader), defines the bounds (bounds), builds the emcee sampler with the
log-probability (inference), writes the HDF5 backend, optionally compresses it, and produces the figures
(postprocess) and the convergence diagnostics (convergence_check).

Same structure as ../03_growth_model_selection/pipeline_control_parallel/global.py, with
  - groups cart_ss_b16, cart_sm_b16, cart_sl_b16 (panel 3b)
  - the 18 treatment models of models.py
  - data from load_treatment_data()

Configured with CALIB_* environment variables (see the README).
"""
import gc
import os
from multiprocessing import cpu_count, get_context

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data
from inference import log_probability
from postprocess import compress_h5_inplace, plot_maxll_solution_global


# Output folders
os.makedirs('./Output_Calibration', exist_ok=True)
os.makedirs('./reinitiate_files', exist_ok=True)

# Treatment data (panel 3b: cart_ss/sm/sl)
full_data = load_treatment_data()
max_ic = find_max_initial_condition(full_data)[0]


if __name__ == "__main__":
    for g, mice in full_data.items():
        print(f"{g}: {len(mice)} mice -> {[m['name'] for m in mice]}")

    # --- Settings from environment variables ---
    MODELS_TO_FIT = os.environ.get(
        'CALIB_MODELS',
        # _exp-based models (16): r = R_B16_FIXED, 15 empirical killing terms
        '_exp,'
        '_exp_ED,_exp_CD,_exp_AccelED,_exp_AccelCD,'
        '_exp_MM,_exp_LS,_exp_PF,'
        '_exp_dePR,_exp_dePRDecay,'
        '_exp_CDwithLog,_exp_CDwithExp,_exp_CDwithSin,'
        '_exp_EDwithLog,_exp_EDwithExp,_exp_EDwithSin,'
        # _mac (2D, T+M without CAR T-cells) and _mac_treat (3D, full T+M+C)
        '_mac,_mac_treat'
    ).split(',')
    GROUP_LIST     = os.environ.get(
        'CALIB_GROUPS', 'cart_ss_b16,cart_sm_b16,cart_sl_b16').split(',')
    CHAIN_SIZE     = int(os.environ.get('CALIB_CHAIN_SIZE', '300000'))
    USE_POOL       = os.environ.get('CALIB_USE_POOL', '1') == '1'
    N_CORES        = int(os.environ.get('CALIB_N_CORES', str(cpu_count())))
    BACKEND_ONLY   = os.environ.get('CALIB_BACKEND_ONLY', '0') == '1'
    COMPRESS_H5    = os.environ.get('CALIB_COMPRESS_H5', '0') == '1'
    COMPRESS_LEVEL = int(os.environ.get('CALIB_COMPRESS_LEVEL', '4'))
    MAKE_CORNER    = os.environ.get('CALIB_CORNER', '1') == '1'
    WALKERS_PER_DIM = int(os.environ.get('CALIB_WALKERS_PER_DIM', '6'))

    # Test mode
    TEST_MODE = os.environ.get('CALIB_TEST', '0') == '1'
    if TEST_MODE:
        MODELS_TO_FIT = ['_exp_ED']
        GROUP_LIST    = ['cart_sl_b16']
        CHAIN_SIZE    = 50
        BACKEND_ONLY  = True
        print("[TEST MODE] _exp_ED + cart_sl_b16 + 50 steps + backend-only; "
              "files prefixed with 'TEST_'.")
    REINIT_PREFIX = "TEST_" if TEST_MODE else ""

    from models import SOLVER, SOLVER_METHOD
    print(f"[config] models={MODELS_TO_FIT} groups={GROUP_LIST} "
          f"chain={CHAIN_SIZE} pool={USE_POOL} n_cores={N_CORES} "
          f"walkers_per_dim={WALKERS_PER_DIM} solver={SOLVER}"
          + (f"({SOLVER_METHOD})" if SOLVER == 'solve_ivp' else "")
          + f" backend_only={BACKEND_ONLY} compress_h5={COMPRESS_H5}"
          + (f" (level={COMPRESS_LEVEL})" if COMPRESS_H5 else "")
          + f" corner={MAKE_CORNER}"
          + (" [TEST]" if TEST_MODE else ""))

    for model_extension in MODELS_TO_FIT:
        print(f"Solving model {model_extension}")
        for group in GROUP_LIST:
            print(f"Solving group {group}")
            group_size = len(full_data[group])
            if group_size == 0:
                print(f"[skip] {group}: empty")
                continue
            l_bound, u_bound, labels = define_bounds_labels(
                model_extension, group_size=group_size, max_ic=max_ic,
                bool_multiple_mice=True,
            )
            ndim = len(l_bound)
            nwalkers = WALKERS_PER_DIM * ndim

            filename = f"./reinitiate_files/reinitiate_{REINIT_PREFIX}{group}{model_extension}.h5"

            # gzip from creation: emcee 3.1.6 applies self.compression in reset(), so the datasets are
            # compressed from the start and incremental writing never reaches the uncompressed size (about
            # 4 GB per _mac_treat file, which exceeded the disk quota on TACC; total peak reduced from ~112 GB
            # to ~43 GB over the 54 pairs). When an existing backend is resumed the compression setting is
            # ignored (it applies only to reset). The level is COMPRESS_LEVEL, as in compress_h5_inplace.
            if not os.path.exists(filename):
                print("HDF5 file not found. Creating a new file...")
                backend = emcee.backends.HDFBackend(
                    filename, compression='gzip', compression_opts=COMPRESS_LEVEL)
                backend.reset(nwalkers, ndim)
                pos = np.zeros((nwalkers, ndim))
                for i in range(ndim):
                    pos[:, i] = np.random.uniform(low=l_bound[i], high=u_bound[i],
                                                  size=nwalkers)
            else:
                print(f"HDF5 file found: {filename}")
                backend = emcee.backends.HDFBackend(
                    filename, compression='gzip', compression_opts=COMPRESS_LEVEL)
                print(f"Current progress on the backend: {backend.iteration} iterations.")
                pos = None

            additional_chain_size = max(0, CHAIN_SIZE - backend.iteration)
            SKIPPED_MCMC = (additional_chain_size == 0)

            # Mixture of moves: StretchMove (emcee default) and DEMove. DEMove proposes jumps along the
            # directions spanned by the walker population, which helps with funnel-shaped posteriors (strongly
            # correlated alpha1/K1/beta1 in _mac and _mac_treat) and does not harm the _exp_* models. Moves are
            # a sampler setting, not stored in the backend, so they can be used when resuming a backend.
            SAMPLER_MOVES = [
                (emcee.moves.StretchMove(), 0.7),
                (emcee.moves.DEMove(), 0.3),
            ]

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
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability,
                    args=(l_bound, u_bound, model_extension, full_data, group),
                    backend=backend, moves=SAMPLER_MOVES,
                )
            else:
                print(f'Calibrating {group} with {model_extension}: '
                      f'{additional_chain_size} new steps '
                      f'(backend={backend.iteration} -> target={CHAIN_SIZE})')

                def _run_sampler(pool):
                    sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability,
                        args=(l_bound, u_bound, model_extension, full_data, group),
                        pool=pool, backend=backend, moves=SAMPLER_MOVES,
                    )
                    if backend.iteration > 0:
                        sampler.run_mcmc(None, additional_chain_size, progress=True)
                    else:
                        sampler.run_mcmc(pos, additional_chain_size, progress=True)
                    return sampler

                if USE_POOL:
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
            kept_samples = (total_iters - discard_burnin) * nwalkers
            scale_corner = max(1, kept_samples // 10000)

            save_chain = sampler.get_chain(discard=discard_burnin, flat=True,
                                           thin=scale_corner)
            np.savetxt(f'./Output_Calibration/multi_chain_{group}{model_extension}.gz', save_chain)
            np.savez(f'./Output_Calibration/multi_ll_pars_{group}{model_extension}.npz',
                     max_ll=max(flat_ll), pars=best_pars)

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

            # Trace plot
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
                group_name=group, model=model_extension,
            )
            plt.close('all')
            gc.collect()

        # No combined figure over the groups: the glob '*_exp.npz' would also match '_exp_ED.npz' and the other
        # models sharing the prefix; the figures of each (group, model) pair are used instead.
        if BACKEND_ONLY:
            continue

    # === Convergence diagnostics ===
    if not BACKEND_ONLY:
        try:
            from convergence_check import run_convergence_check
            print('\n=== CONVERGENCE DIAGNOSTICS ===')
            run_convergence_check(
                results_dir='.', models=MODELS_TO_FIT,
                groups=GROUP_LIST, save_csv=True, print_table=True,
            )
        except Exception as _e:
            print(f'[WARN] convergence_check failed: {_e.__class__.__name__}: {_e}')
