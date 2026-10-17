"""Profile likelihoods of beta1 (T-M and T-M-C models) and mu1 (T-M-C model), with the prior bounds enforced.

For each grid value of the profiled parameter, the remaining model parameters and the initial volumes are
re-estimated inside the prior bounds by bounded least squares, with the residual standard deviation concentrated
out (lsq_fit.py). The likelihood of the T-M-C model is multimodal and concentrated on narrow curved ridges, so a
single optimization per grid value is not enough; the profile is built in three stages.

  1 continuation  from the maximum-likelihood estimate (mle_bounded.py) outwards in both directions; each grid
                  value starts from the solution of the previous one and from a secant extrapolation of the last
                  two solutions, so the optimizer follows the ridge
  2 sweeps        every grid value is re-optimized from the solutions of its neighbours (two on each side) whose
                  profile value is not more than NEIGHBOUR_WINDOW below its own, and from the best solution found for
                  the same model and group; repeated until no grid value improves by more than TOL
  3 estimate      when a grid value exceeds the current maximum-likelihood estimate, all parameters are released
                  from that solution; an improved estimate is saved, its value is added to the grid, and the sweeps
                  resume

Profile values remain lower bounds of the exact profile. The number of residual local increases of the profile
away from its maximum (non_monotone_points) is reported as a diagnostic.

Outputs (Output_Calibration/TMC):
  profile_bounded_grid.csv     one row per (parameter, model, group, grid value), with the full estimate
  profile_bounded_summary.csv  maximum, 95% interval, contiguity, drop at the prior edges, diagnostics
  profile_bounded_theta.npz    full parameter vector at every grid value
"""
import csv
import os
from multiprocessing import Pool

import numpy as np

from lsq_fit import Problem

TMC = './Output_Calibration/TMC'
LR = 1.9207                                   # half of the 95% quantile of chi-square with 1 degree of freedom
GROUPS = ['cart_ss_b16', 'cart_sm_b16', 'cart_sl_b16']
BETA1_GRID = np.round(np.linspace(-6.0, -2.0, 49), 4)
MU1_GRID = np.round(np.concatenate([np.arange(0.0, 1.0001, 0.025), np.arange(1.25, 5.0001, 0.25)]), 4)
PROBLEMS = [(p, m, g) for g in GROUPS for (p, m) in
            [('log10_beta1', '_mac'), ('log10_beta1', '_mac_treat'), ('mu1', '_mac_treat')]]
INDEX = {'log10_beta1': 2, 'mu1': 3}
NEIGHBOUR_WINDOW = 1.0
TOL = 0.01
MAX_SWEEPS = 8
WORKERS = 15


def mle_path(model, group):
    return os.path.join(TMC, f'mle_bounded_{group}.npz' if model == '_mac_treat' else f'mle_bounded_{group}{model}.npz')


def load_mle(model, group):
    z = np.load(mle_path(model, group))
    return float(z['ll']), np.asarray(z['theta'], float)


def continuation(args):
    """Walk along the grid from the estimate; returns [(grid index, ll, theta)]."""
    pname, model, group, grid, order, start = args
    idx = INDEX[pname]
    out, prev = [], [start]
    for i in order:
        cand = [prev[-1]]
        if len(prev) >= 2:
            p0 = Problem(model, group)
            u1, u0 = p0.to_unit(prev[-1]), p0.to_unit(prev[-2])
            pred = prev[-1].copy()
            pred[p0.free] = p0.lb[p0.free] + p0.span * np.clip(2 * u1 - u0, 0, 1)
            cand.append(pred)
        ll, th = Problem(model, group, {idx: float(grid[i])}).fit_many(cand)
        out.append((i, ll, th))
        prev.append(th)
    return pname, model, group, out


def refit(args):
    pname, model, group, i, value, starts = args
    ll, th = Problem(model, group, {INDEX[pname]: value}).fit_many(starts)
    return pname, model, group, i, ll, th


def release(args):
    model, group, starts = args
    return model, group, Problem(model, group).fit_many(starts)


def interval(grid, prof):
    llmax = prof.max()
    thr = llmax - LR
    inside = np.where(prof >= thr)[0]

    def cross(i, j):
        return grid[i] + (grid[j] - grid[i]) * (prof[i] - thr) / (prof[i] - prof[j])

    lo = grid[inside.min()] if inside.min() == 0 else cross(inside.min(), inside.min() - 1)
    hi = grid[inside.max()] if inside.max() == len(grid) - 1 else cross(inside.max(), inside.max() + 1)
    return llmax, lo, hi, bool(np.all(np.diff(inside) == 1))


def non_monotone(prof, tol=0.05):
    """Grid values where the profile increases by more than tol while moving away from its maximum."""
    k = int(np.argmax(prof))
    right = np.sum(np.diff(prof[k:]) > tol)
    left = np.sum(np.diff(prof[:k + 1][::-1]) > tol)
    return int(left + right)


def main():
    grids, sol = {}, {}
    mle = {(m, g): load_mle(m, g) for m in ('_mac', '_mac_treat') for g in GROUPS}
    for p, m, g in PROBLEMS:
        base = BETA1_GRID if p == 'log10_beta1' else MU1_GRID
        grids[(p, m, g)] = np.unique(np.round(np.append(base, mle[(m, g)][1][INDEX[p]]), 6))

    with Pool(WORKERS) as pool:
        jobs = []
        for key in PROBLEMS:
            grid, (ll0, th0) = grids[key], mle[key[1:]]
            k = int(np.argmin(np.abs(grid - th0[INDEX[key[0]]])))
            sol[key + (k,)] = (ll0, th0)
            jobs.append(key + (grid, list(range(k + 1, len(grid))), th0))
            jobs.append(key + (grid, list(range(k - 1, -1, -1)), th0))
        for p, m, g, out in pool.imap_unordered(continuation, [j for j in jobs if j[4]]):
            for i, ll, th in out:
                if (p, m, g, i) not in sol or ll > sol[(p, m, g, i)][0]:
                    sol[(p, m, g, i)] = (ll, th)
        print('continuation done', flush=True)

        tried = set()
        for sweep in range(1, MAX_SWEEPS + 1):
            best = {}
            for (p, m, g, i), (ll, th) in sol.items():
                if (m, g) not in best or ll > best[(m, g)][0]:
                    best[(m, g)] = (ll, th, (p, i))
            jobs = []
            for key in PROBLEMS:
                p, m, g = key
                grid = grids[key]
                for i, v in enumerate(grid):
                    ll_i = sol[key + (i,)][0]
                    starts, tags = [], []
                    for j in (i - 2, i - 1, i + 1, i + 2):
                        if 0 <= j < len(grid) and sol[key + (j,)][0] > ll_i - NEIGHBOUR_WINDOW:
                            tag = (key, i, 'nb', j, round(sol[key + (j,)][0], 4))
                            if tag not in tried:
                                starts.append(sol[key + (j,)][1]), tags.append(tag)
                    bll, bth, bsrc = best[(m, g)]
                    tag = (key, i, 'best', bsrc, round(bll, 4))
                    if bll > ll_i + TOL and tag not in tried:
                        starts.append(bth), tags.append(tag)
                    if starts:
                        tried.update(tags)
                        jobs.append(key + (i, float(v), starts))
            improved = 0
            for p, m, g, i, ll, th in pool.imap_unordered(refit, jobs):
                if ll > sol[(p, m, g, i)][0] + TOL:
                    improved += 1
                    sol[(p, m, g, i)] = (ll, th)
            print(f'sweep {sweep}: {len(jobs)} grid values re-optimized, {improved} improved', flush=True)

            rel_jobs = []
            for m, g in mle:
                cands = sorted([(ll, th) for (p, mm, gg, i), (ll, th) in sol.items() if mm == m and gg == g],
                               key=lambda r: -r[0])
                if cands[0][0] > mle[(m, g)][0] + TOL:
                    rel_jobs.append((m, g, [c[1] for c in cands[:3]]))
            new_mle = 0
            for m, g, (ll, th) in pool.map(release, rel_jobs):
                if ll > mle[(m, g)][0] + TOL:
                    new_mle += 1
                    print(f'  improved estimate {g}{m}: {mle[(m, g)][0]:.3f} -> {ll:.3f}', flush=True)
                    mle[(m, g)] = (ll, th)
                    old = np.load(mle_path(m, g))
                    np.savez(mle_path(m, g), theta=th, ll=ll, labels=old['labels'], map_ll=old['map_ll'],
                             start_lls=old['start_lls'], source='profile release')
                    for key in PROBLEMS:
                        if key[1:] != (m, g):
                            continue
                        v = round(float(th[INDEX[key[0]]]), 6)
                        grid = grids[key]
                        if np.min(np.abs(grid - v)) < 1e-6:
                            k = int(np.argmin(np.abs(grid - v)))
                            sol[key + (k,)] = (ll, th)
                            continue
                        new = np.sort(np.append(grid, v))
                        remap = {}
                        for i, gv in enumerate(grid):
                            remap[int(np.where(new == gv)[0][0])] = sol.pop(key + (i,))
                        for i, val in remap.items():
                            sol[key + (i,)] = val
                        sol[key + (int(np.where(new == v)[0][0]),)] = (ll, th)
                        grids[key] = new
                        tried = {t for t in tried if t[0] != key}
            if improved == 0 and new_mle == 0:
                break

    thetas = {}
    with open(os.path.join(TMC, 'profile_bounded_grid.csv'), 'w', newline='') as fg, \
            open(os.path.join(TMC, 'profile_bounded_summary.csv'), 'w', newline='') as fs:
        wg, ws = csv.writer(fg), csv.writer(fs)
        wg.writerow(['parameter', 'model', 'group', 'value', 'profile_ll', 'sigma'] + [f'p{i}' for i in range(8)])
        ws.writerow(['parameter', 'model', 'group', 'argmax', 'll_max', 'mle_ll', 'CI95_lo', 'CI95_hi',
                     'CI_contiguous', 'drop_lower_edge', 'drop_upper_edge', 'n_grid', 'non_monotone_points'])
        for key in PROBLEMS:
            p, m, g = key
            grid = grids[key]
            prof = np.array([sol[key + (i,)][0] for i in range(len(grid))])
            ths = np.array([sol[key + (i,)][1] for i in range(len(grid))])
            thetas[f'{p}__{m}__{g}__grid'], thetas[f'{p}__{m}__{g}__theta'] = grid, ths
            n_par = 8 if m == '_mac_treat' else 3
            for i, v in enumerate(grid):
                wg.writerow([p, m, g, f'{v:.4f}', f'{prof[i]:.4f}', f'{ths[i][-1]:.3f}'] +
                            [f'{x:.6g}' for x in ths[i][:n_par]] + [''] * (8 - n_par))
            llmax, lo, hi, cont = interval(grid, prof)
            nm = non_monotone(prof)
            ws.writerow([p, m, g, f'{grid[int(np.argmax(prof))]:.4f}', f'{llmax:.3f}', f'{mle[(m, g)][0]:.3f}',
                         f'{lo:.4f}', f'{hi:.4f}', int(cont), f'{llmax - prof[0]:.3f}', f'{llmax - prof[-1]:.3f}',
                         len(grid), nm])
            print(f'{p}{m} {g}: argmax={grid[int(np.argmax(prof))]:.3f} llmax={llmax:.2f} CI95=[{lo:.3f},{hi:.3f}] '
                  f'contiguous={cont} drops=({llmax - prof[0]:.2f},{llmax - prof[-1]:.2f}) non-monotone={nm}',
                  flush=True)
    np.savez(os.path.join(TMC, 'profile_bounded_theta.npz'), **thetas)


if __name__ == '__main__':
    main()
