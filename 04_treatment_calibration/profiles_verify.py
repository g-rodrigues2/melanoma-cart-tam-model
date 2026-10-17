"""Verification pass of the long-hinge profile likelihoods outside their 95% intervals.

profiles_bounded.py propagates improvements between neighbouring grid values, so a region of a profile far from the
maximum can remain under-optimized when the estimate improves late in the sweeps. The confidence decision depends on
exactly these values (a drop larger than 1.92 excludes the grid value). Here every grid value of the long-hinge
profiles of log10(beta1) and mu1 (T-M-C model) whose drop exceeds the threshold is re-optimized from the solutions of
the other profile that lie inside the threshold; these solutions span the likelihood ridge and are the natural
starting points. The fixed parameter of each start is replaced by the grid value.

Improved values replace those in profile_bounded_grid.csv, profile_bounded_theta.npz and profile_bounded_summary.csv
(the latter recomputed for the modified profiles); every change is written to profiles_verify.log.
"""
import csv
import os
from multiprocessing import Pool

import numpy as np

from lsq_fit import Problem
from profiles_bounded import INDEX, LR, TMC, interval, non_monotone

GROUP = 'cart_sl_b16'
MODEL = '_mac_treat'
TOL = 0.01
WORKERS = 15


def load():
    rows = list(csv.DictReader(open(os.path.join(TMC, 'profile_bounded_grid.csv'), newline='')))
    z = dict(np.load(os.path.join(TMC, 'profile_bounded_theta.npz')))
    return rows, z


def refit(args):
    pname, i, value, starts = args
    ll, th = Problem(MODEL, GROUP, {INDEX[pname]: value}).fit_many(starts)
    return pname, i, ll, th


def main():
    rows, z = load()
    prof = {}
    for p in INDEX:
        grid = z[f'{p}__{MODEL}__{GROUP}__grid']
        lls = np.array([float(r['profile_ll']) for r in rows
                        if r['parameter'] == p and r['model'] == MODEL and r['group'] == GROUP])
        prof[p] = (grid, lls, z[f'{p}__{MODEL}__{GROUP}__theta'])
    ll_max = max(v[1].max() for v in prof.values())
    jobs = []
    for p, other in (('log10_beta1', 'mu1'), ('mu1', 'log10_beta1')):
        grid, lls, _ = prof[p]
        og, oll, oth = prof[other]
        starts = [oth[j] for j in np.where(oll >= ll_max - LR)[0]]
        for i in np.where(lls < ll_max - LR)[0]:
            jobs.append((p, int(i), float(grid[i]), starts))
    print(f'l_max={ll_max:.3f}; {len(jobs)} grid values outside the intervals re-optimized', flush=True)

    changes = []
    with Pool(WORKERS) as pool:
        for p, i, ll, th in pool.imap_unordered(refit, jobs):
            grid, lls, ths = prof[p]
            if ll > lls[i] + TOL:
                changes.append((p, grid[i], lls[i], ll))
                lls[i] = ll
                ths[i] = th
    with open(os.path.join(TMC, 'profiles_verify.log'), 'w') as f:
        f.write(f'l_max={ll_max:.4f}; re-optimized {len(jobs)} grid values; improved {len(changes)}\n')
        for p, v, old, new in sorted(changes):
            f.write(f'{p} {v:.4f}: {old:.4f} -> {new:.4f}\n')
    print(f'{len(changes)} grid values improved', flush=True)
    if not changes:
        return

    for r in rows:
        if r['model'] == MODEL and r['group'] == GROUP:
            grid, lls, ths = prof[r['parameter']]
            i = int(np.argmin(np.abs(grid - float(r['value']))))
            r['profile_ll'] = f'{lls[i]:.4f}'
            r['sigma'] = f'{ths[i][-1]:.3f}'
            for k in range(8):
                r[f'p{k}'] = f'{ths[i][k]:.6g}'
    with open(os.path.join(TMC, 'profile_bounded_grid.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for p in INDEX:
        z[f'{p}__{MODEL}__{GROUP}__theta'] = prof[p][2]
    np.savez(os.path.join(TMC, 'profile_bounded_theta.npz'), **z)

    summ = list(csv.DictReader(open(os.path.join(TMC, 'profile_bounded_summary.csv'), newline='')))
    ll_max = max(v[1].max() for v in prof.values())
    for r in summ:
        if r['model'] == MODEL and r['group'] == GROUP:
            grid, lls, _ = prof[r['parameter']]
            llm, lo, hi, cont = interval(grid, lls)
            r.update(argmax=f'{grid[int(np.argmax(lls))]:.4f}', ll_max=f'{llm:.3f}', CI95_lo=f'{lo:.4f}',
                     CI95_hi=f'{hi:.4f}', CI_contiguous=int(cont), drop_lower_edge=f'{llm - lls[0]:.3f}',
                     drop_upper_edge=f'{llm - lls[-1]:.3f}', non_monotone_points=non_monotone(lls))
            print(r, flush=True)
    with open(os.path.join(TMC, 'profile_bounded_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)


if __name__ == '__main__':
    main()
