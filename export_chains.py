"""Exports the MCMC chains of the selected models of Chapter 3 for the data deposit (Zenodo).

Exported chains:
  - B16 growth rate: exponential model with group-specific rates (sharing scenario 1, `_exp_s1`);
  - empirical treatment models selected by BIC: no treatment term (`_exp`) for the short- and medium-hinge
    groups and the sinusoidally modulated term (`_exp_EDwithSin`) for the long-hinge group;
  - mechanistic T-M-C model (`_mac_treat`) for the three CAR T-cell groups.

Each HDF5 backend becomes one compressed file

    <out>/<step folder>/<backend name>.npz
        chain       (steps/thin, walkers, parameters) float32, every `thin`-th step
        log_prob    (steps/thin, walkers) float32
        thin        thinning interval, the largest of 10, the value that keeps at most ~5,000 stored
                    steps, and the value that keeps the stored arrays under 90 MiB
        n_steps     number of steps of the original chain
        burn_in     first quarter of the steps, in original steps

The chains are read with the thinning applied inside the HDF5 file, so long chains never reach memory in full.
Read a file with `read_exported_chain`.

    python export_chains.py --out <folder>
"""
import argparse
import json
import os

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import emcee
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = '03_growth_model_selection/pipeline_shared_params_parallel/reinitiate_files'
TREATMENT = '04_treatment_calibration/reinitiate_files'
CHAINS = [
    f'{SHARED}/reinitiate__exp_s1.h5',
    f'{TREATMENT}/reinitiate_cart_ss_b16_exp.h5',
    f'{TREATMENT}/reinitiate_cart_sm_b16_exp.h5',
    f'{TREATMENT}/reinitiate_cart_sl_b16_exp_EDwithSin.h5',
    f'{TREATMENT}/reinitiate_cart_ss_b16_mac_treat.h5',
    f'{TREATMENT}/reinitiate_cart_sm_b16_mac_treat.h5',
    f'{TREATMENT}/reinitiate_cart_sl_b16_mac_treat.h5',
]


MAX_BYTES = 90 * 1024 ** 2   # a file in the repository must stay below 100 MiB


def export(rel, out):
    backend = emcee.backends.HDFBackend(os.path.join(HERE, *rel.split('/')), read_only=True)
    steps = int(backend.iteration)
    walkers, ndim = backend.shape
    thin = max(10, steps // 5000)
    while 4 * (steps // thin) * walkers * (ndim + 1) > MAX_BYTES:
        thin *= 2            # wide chains need the thinning the stored size asks for, not only the step count
    chain = backend.get_chain(thin=thin).astype(np.float32)
    log_prob = backend.get_log_prob(thin=thin).astype(np.float32)
    target = os.path.join(out, rel.split('/')[0], os.path.basename(rel).replace('.h5', '.npz'))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    np.savez_compressed(target, chain=chain, log_prob=log_prob, thin=thin, n_steps=steps, burn_in=steps // 4)
    return dict(file=os.path.relpath(target, out).replace('\\', '/'), steps=steps, thin=thin,
                walkers=int(chain.shape[1]), parameters=int(chain.shape[2]),
                megabytes=round(os.path.getsize(target) / 1e6, 1))


def read_exported_chain(path):
    """(chain, log_prob, info) of an exported file; chain and log_prob hold every `thin`-th step."""
    z = np.load(path)
    return z['chain'], z['log_prob'], dict(thin=int(z['thin']), n_steps=int(z['n_steps']), burn_in=int(z['burn_in']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    manifest = []
    for rel in CHAINS:
        row = export(rel, args.out)
        manifest.append(row)
        print(row, flush=True)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(f'{len(manifest)} chains, {sum(r["megabytes"] for r in manifest):.1f} MB')


if __name__ == '__main__':
    main()
