"""Posterior histograms of the calibrated parameters, with the maximum-likelihood sample marked.

For each calibrated model parameter (initial conditions and std excluded) the figure shows
  - the histogram (density)
  - a dashed vertical line at the maximum-likelihood sample (MAP; uniform priors)
  - dotted lines at the 95% interval and a solid line at the median

Accepts chains of the sharing scenarios and of the treatment calibration. Labels starting with 'ic' and
'std' are skipped.

From Python:
    from posterior_histogram import plot_posterior_histograms
    plot_posterior_histograms(
        chain_path='Output_Calibration/multi_chain__exp_s1.gz',
        labels_path_or_list=['r_b16', 'r_pdx'] + [f'ic{i}' for i in range(46)] + ['std'],
        npz_path='Output_Calibration/multi_ll_pars__exp_s1.npz',
        figure_basename='Output_Calibration/posterior_hist__exp_s1',
    )

Command line:
    python posterior_histogram.py --chain Output_Calibration/multi_chain__exp_s1.gz \
                                   --npz   Output_Calibration/multi_ll_pars__exp_s1.npz \
                                   --out   Output_Calibration/posterior_hist__exp_s1
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def _is_model_param(label):
    """True for a model parameter (not an initial condition nor std)."""
    s = str(label).lower()
    if s == 'std' or s.startswith('std'):
        return False
    if s.startswith('ic'):
        return False
    return True


def plot_posterior_histograms(chain_path, labels, npz_path=None,
                              figure_basename='posterior_hist',
                              bins=50, save=True, show=False):
    """One histogram per model parameter (one figure per parameter).

    Arguments:
      chain_path        : .gz chain (n_samples x n_dim)
      labels            : labels, len == n_dim
      npz_path          : .npz with 'pars' (MAP); None omits the dashed line
      figure_basename   : output prefix (no extension); '_<label>.png/.pdf' is appended
      bins              : number of histogram bins
      save / show       : output control

    Returns the paths written.
    """
    chain = np.loadtxt(chain_path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)

    assert chain.shape[1] == len(labels), \
        f'chain has {chain.shape[1]} dimensions; labels has {len(labels)}'

    map_pars = None
    if npz_path is not None and os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        map_pars = npz['pars']

    model_param_indices = [i for i, lbl in enumerate(labels) if _is_model_param(lbl)]
    if not model_param_indices:
        print(f'[posterior_hist] nothing to plot in {chain_path} (only initial conditions/std)')
        return []

    os.makedirs(os.path.dirname(figure_basename) or '.', exist_ok=True)
    paths = []

    for i in model_param_indices:
        samples = chain[:, i]
        med  = np.median(samples)
        q025 = np.percentile(samples, 2.5)
        q975 = np.percentile(samples, 97.5)
        map_val = float(map_pars[i]) if map_pars is not None else None

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(samples, bins=bins, color='steelblue', alpha=0.55,
                edgecolor='white', density=True)

        # 95% interval (thin dotted lines)
        ax.axvline(q025, color='gray', linestyle=':', linewidth=1.0,
                   label=f'95% CI [{q025:.4f}, {q975:.4f}]')
        ax.axvline(q975, color='gray', linestyle=':', linewidth=1.0)

        # median (thin solid line)
        ax.axvline(med, color='black', linestyle='-', linewidth=1.2,
                   alpha=0.6, label=f'median = {med:.4f}')

        # MAP (dashed red line)
        if map_val is not None:
            ax.axvline(map_val, color='crimson', linestyle='--', linewidth=1.8,
                       label=f'MAP = {map_val:.4f}')

        ax.set_xlabel(labels[i], fontsize=13)
        ax.set_ylabel('density', fontsize=13)
        ax.set_title(f'Posterior of {labels[i]}', fontsize=13)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

        if save:
            label_safe = labels[i].replace('/', '_').replace(' ', '_')
            path_png = f"{figure_basename}_{label_safe}.png"
            path_pdf = f"{figure_basename}_{label_safe}.pdf"
            plt.savefig(path_png, dpi=200, bbox_inches='tight', pad_inches=0.02)
            plt.savefig(path_pdf,         bbox_inches='tight', pad_inches=0.02)
            paths.append(path_png)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return paths


# === Labels of the sharing-scenario chains ===

def _shared_labels_for(model_extension, scenario_id, full_data, groups):
    """Labels of a sharing-scenario chain in the order of theta: [model parameters (global or per group),
    initial conditions per mouse, std].
    """
    # imported here so that this module stays standalone
    import _path_setup  # noqa: F401
    from scenarios import MODEL_PARAM_NAMES, SCENARIOS

    scenario = SCENARIOS[model_extension][scenario_id]
    param_names = MODEL_PARAM_NAMES[model_extension]

    labels = []
    for p in param_names:
        kind = scenario[p]
        if kind == 'global':
            labels.append(p)
        else:
            for g in groups:
                # e.g. 'r_control_b16' -> 'r_b16'
                short = g.replace('control_', '')
                labels.append(f'{p}_{short}')

    for g in groups:
        for i, _mouse in enumerate(full_data[g]):
            labels.append(f'ic_{g.replace("control_", "")}_{i+1}')

    labels.append('std')
    return labels


# === Labels of the treatment-calibration chains ===

def _treatment_labels_for(model_extension, group_size):
    """Labels of a treatment-calibration chain (from bounds.define_bounds_labels)."""
    from bounds import define_bounds_labels  # bounds of the treatment calibration
    _, _, labels = define_bounds_labels(
        model_extension, group_size=group_size, max_ic=1.0,
        bool_multiple_mice=True,
    )
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', required=True, help='.gz chain')
    parser.add_argument('--npz',   default=None,  help='.npz with pars (MAP)')
    parser.add_argument('--out',   required=True, help='prefix of the PNG/PDF files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='explicit labels; otherwise detected from the file name')
    parser.add_argument('--bins',  type=int, default=50)
    args = parser.parse_args()

    if args.labels is None:
        # detect model and scenario from the file name
        base = os.path.basename(args.chain)
        # e.g. 'multi_chain__exp_s1.gz'
        if '_s' in base:
            import _path_setup as ps
            from data_loader import load_control_data
            full = load_control_data(data_dir=ps.DATA_DIR)
            tokens = base.replace('multi_chain_', '').replace('.gz', '').split('_s')
            mext = tokens[0] if tokens[0].startswith('_') else '_' + tokens[0]
            sid  = int(tokens[1])
            groups = ['control_b16', 'control_pdx']
            args.labels = _shared_labels_for(mext, sid, full, groups)
        else:
            raise SystemExit("--labels is required for chains other than the sharing scenarios")

    paths = plot_posterior_histograms(
        chain_path=args.chain, labels=args.labels, npz_path=args.npz,
        figure_basename=args.out, bins=args.bins,
    )
    print(f"{len(paths)} figures written to {os.path.dirname(args.out) or '.'}")


if __name__ == '__main__':
    main()
