"""Trace plots and corner plots of every (group, treatment model) pair.

Used instead of the figures of global.py for the treatment models, whose names contain underscores that
plot_maxll_solution_global does not parse (rsplit '_' in '_exp_*').
"""
import gc
import os

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from bounds import define_bounds_labels, find_max_initial_condition
from data_loader import load_treatment_data

REINIT_DIR = './reinitiate_files'
OUT_DIR    = './Output_Calibration'

full_data = load_treatment_data()
max_ic = find_max_initial_condition(full_data)[0]
GROUPS = sorted(full_data.keys())


def parse_filename(filename):
    stem = filename.replace('reinitiate_', '').replace('.h5', '')
    for g in GROUPS:
        if stem.startswith(g):
            return g, stem[len(g):]
    return None, None


def gen_trace(backend, labels, out_path):
    """Trace plot, one panel per parameter, against the step number (0..total_iters). Thinned every N steps to
    avoid MemoryError with long chains.
    """
    full_samples = backend.get_chain()  # shape (total_iters, nwalkers, ndim)
    total_iters = full_samples.shape[0]
    trace_thin = max(1, total_iters // 5000)
    samples = full_samples[::trace_thin]
    x = np.arange(0, total_iters, trace_thin)[:len(samples)]
    del full_samples

    fig, axes = plt.subplots(nrows=len(labels), ncols=1,
                             figsize=(10, len(labels) * 2), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(x, samples[:, :, i], 'k', alpha=0.3)
        ax.set_xlim(0, total_iters)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel(f'step (total={total_iters}; thinned every {trace_thin} for plotting)')
    plt.savefig(out_path, format='png', dpi=200,
                bbox_inches='tight', pad_inches=0.02)
    plt.close('all')
    del samples


def gen_corner(backend, labels, out_path):
    """Corner plot of the model parameters (initial conditions and std excluded)."""
    # model parameters only (not ic*, not std)
    keep = [i for i, lbl in enumerate(labels)
            if not lbl.startswith('ic') and lbl != 'std']
    if not keep:
        return  # _exp baseline (so std + ICs)

    total_iters = int(backend.iteration)
    discard = total_iters // 2
    chain = backend.get_chain(discard=discard, flat=True)
    kept = chain.shape[0]
    thin = max(1, kept // 5000)
    chain = chain[::thin]

    sub = chain[:, keep]
    sub_labels = [labels[i] for i in keep]

    fig = corner.corner(sub, labels=sub_labels, show_titles=True,
                        title_fmt='.4f', title_kwargs={'fontsize': 10})
    fig.savefig(out_path, format='png', dpi=200,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    files = sorted(f for f in os.listdir(REINIT_DIR)
                   if f.startswith('reinitiate_') and f.endswith('.h5')
                   and 'TEST_' not in f)
    print(f"Trace and corner plots for {len(files)} pairs...")

    for fname in files:
        group, model = parse_filename(fname)
        if group is None:
            continue

        backend = emcee.backends.HDFBackend(
            os.path.join(REINIT_DIR, fname), read_only=True)
        group_size = len(full_data[group])
        _, _, labels = define_bounds_labels(
            model, group_size=group_size, max_ic=max_ic,
            bool_multiple_mice=True,
        )

        trace_path  = os.path.join(OUT_DIR, f'multi_chain_{group}{model}.png')
        corner_path = os.path.join(OUT_DIR, f'multi_corner_{group}{model}.png')

        gen_trace(backend, labels, trace_path)
        gen_corner(backend, labels, corner_path)

        n_params = sum(1 for l in labels if not l.startswith('ic') and l != 'std')
        print(f"  {group}{model:20s} trace + corner ({n_params} model pars, "
              f"{len(labels)} total)")
        gc.collect()


if __name__ == '__main__':
    main()
