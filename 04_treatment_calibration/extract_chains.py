"""Chains and maximum-likelihood samples of the 48 treatment backends.

Used instead of the post-processing of global.py for the treatment models, whose suffixes contain underscores
('_exp_ED', '_exp_dePRDecay') that plot_maxll_solution_global does not parse. For each .h5 file:
  1. loads the backend and discards the burn-in (25%)
  2. writes the thinned chain as multi_chain_<group>_<model>.gz
  3. writes max_ll and pars as multi_ll_pars_<group>_<model>.npz
"""
import gc
import os

import emcee
import numpy as np

from data_loader import load_treatment_data

REINIT_DIR = './reinitiate_files'
OUT_DIR    = './Output_Calibration'
os.makedirs(OUT_DIR, exist_ok=True)


full_data = load_treatment_data()
GROUPS = sorted(full_data.keys())  # ['cart_sl_b16', 'cart_sm_b16', 'cart_ss_b16']


def parse_filename(filename):
    """Decodifica reinitiate_<group>_<model>.h5 em (group, model_extension)."""
    stem = filename.replace('reinitiate_', '').replace('.h5', '')
    for g in GROUPS:
        if stem.startswith(g):
            model_token = stem[len(g):]
            return g, model_token
    raise ValueError(f"Could not identify the group of {filename}")


def main():
    files = sorted(f for f in os.listdir(REINIT_DIR)
                   if f.startswith('reinitiate_') and f.endswith('.h5')
                   and 'TEST_' not in f)
    print(f"Processing {len(files)} files...")

    for fname in files:
        try:
            group, model = parse_filename(fname)
        except ValueError as e:
            print(f"  [skip] {fname}: {e}")
            continue

        path = os.path.join(REINIT_DIR, fname)
        backend = emcee.backends.HDFBackend(path, read_only=True)
        nwalkers, ndim = backend.shape
        total_iters = int(backend.iteration)

        flat_ll = backend.get_log_prob(flat=True)
        flat_chain = backend.get_chain(flat=True)
        idx_best = int(np.argmax(flat_ll))
        best_pars = flat_chain[idx_best]
        max_ll = float(flat_ll[idx_best])

        # Chain afinada
        discard_burnin = total_iters // 4
        kept = (total_iters - discard_burnin) * nwalkers
        thin = max(1, kept // 10000)
        save_chain = backend.get_chain(discard=discard_burnin, flat=True, thin=thin)

        np.savetxt(os.path.join(OUT_DIR, f'multi_chain_{group}{model}.gz'),
                   save_chain)
        np.savez(os.path.join(OUT_DIR, f'multi_ll_pars_{group}{model}.npz'),
                 max_ll=max_ll, pars=best_pars)

        print(f"  {group}{model:20s}  max_ll={max_ll:9.2f}  "
              f"chain={save_chain.shape}  ndim={ndim}  nwalkers={nwalkers}")

        del flat_ll, flat_chain, save_chain
        gc.collect()


if __name__ == '__main__':
    main()
