"""Dummy-parameter significance test for the eFAST indices of Section 3.3.3 (Marino et al., 2008).

The fixed 0.05 cutoff used in the chapter is a screening convention, not a statistical test. The test of
Marino et al. replaces it by a comparison against an artificial parameter that, by construction, has no
influence on the model: the eFAST design is repeated over NR independent search curves (different random
phase shifts), which yields NR estimates of every index, and each parameter is compared against the dummy
by a one-sided two-sample t-test with a Bonferroni correction for the ten comparisons.

The sampling design, the parameter ranges and the compiled integrator are taken from `efast_sensitivity`;
the only change is the eleventh coordinate of the sampling matrix, which the right-hand side never reads.

The test answers "is this index distinguishable from the estimator noise?", which is not the question the
0.05 cutoff answers ("is this index large enough to matter?"). Because the variance compared by the t-test
is the variance across search curves at fixed N, it shrinks as N grows, so at the N = 10,000 of the chapter
the test is close to fully powered and declares significant indices that are numerically negligible. Both
criteria are therefore reported side by side, and neither is presented as a replacement for the other.

Writes results/dummy_test_end.csv, results/dummy_test_time.csv and results/dummy_test_replicates.npz.
Run after (or independently of) efast_sensitivity.py.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from SALib.sample import fast_sampler
from SALib.analyze import fast

import efast_sensitivity as efast

# NR = 5 search curves, as in Marino et al.; the seeds are the phase shifts of the eFAST design.
NR = 5
SEEDS = [1000 + r for r in range(NR)]

ALPHA = 0.05                       # nominal level of each t-test, before the correction
OUTPUTS = ('T', 'M', 'C')
CUTOFF = 0.05                      # the screening threshold of the chapter, reported alongside the test

problem = {
    'num_vars': efast.problem['num_vars'] + 1,
    'names': efast.param_names + ['dummy'],
    # the dummy varies over the same +/-20% relative range as the other parameters, around a unit reference
    'bounds': np.vstack([efast.bounds, np.array([[0.8, 1.2]])]),
}

param_names = efast.param_names
n_par = len(param_names)
i_dummy = problem['names'].index('dummy')
alpha_bonferroni = ALPHA / n_par


def _indices_of_one_curve(seed, t_out):
    """eFAST first-order and total-effect indices of one search curve, for every output and time."""
    X = fast_sampler.sample(problem, efast.N, M=efast.M_fast, seed=seed)
    Y = efast.solve_all(np.array(efast.TMC0, dtype=float), X, t_out, efast.RTOL, efast.ATOL)
    n_t = t_out.shape[0]
    S1 = np.empty((n_t, len(OUTPUTS), problem['num_vars']))
    ST = np.empty((n_t, len(OUTPUTS), problem['num_vars']))
    for k in range(n_t):
        for j in range(len(OUTPUTS)):
            # the bootstrap confidence intervals of SALib are not used here (the spread across search
            # curves is what the t-test consumes), so the cheapest resampling that keeps SALib from
            # computing a standard deviation over a single value is enough
            Si = fast.analyze(problem, Y[:, k, j], M=efast.M_fast, num_resamples=2,
                              print_to_console=False, seed=seed)
            S1[k, j, :] = Si['S1']
            ST[k, j, :] = Si['ST']
    return X.shape[0], S1, ST


def dummy_test(index_by_curve):
    """One-sided Welch t-test of every parameter against the dummy, over the NR search curves.

    `index_by_curve` has shape (NR, n_par + 1); returns the p-values of the n_par real parameters.
    """
    p = np.empty(n_par)
    for i in range(n_par):
        a, b = index_by_curve[:, i], index_by_curve[:, i_dummy]
        if np.allclose(a, b):
            p[i] = 1.0                 # identical columns: no evidence of an effect
        else:
            p[i] = stats.ttest_ind(a, b, equal_var=False, alternative='greater').pvalue
    return p


def main():
    os.makedirs('results', exist_ok=True)

    # the same analyzed times as efast_sensitivity.py, without t = 0, where the output is deterministic
    # (zero variance) and no sensitivity index is defined
    time_indices = np.arange(0, efast.n_t, 20)[1:]
    t_sel = efast.t[time_indices]
    t_out = np.unique(np.concatenate((t_sel, efast.t[-1:])))
    i_end = int(np.searchsorted(t_out, efast.t[-1]))

    print(f"Dummy-parameter test: {NR} search curves, {problem['num_vars']} parameters "
          f"(the last one is the dummy), N = {efast.N}, M = {efast.M_fast}")
    print(f"Bonferroni-corrected level: {ALPHA} / {n_par} = {alpha_bonferroni:.4f}")

    S1 = np.empty((NR, t_out.shape[0], len(OUTPUTS), problem['num_vars']))
    ST = np.empty_like(S1)
    for r, seed in enumerate(SEEDS):
        n_samples, S1[r], ST[r] = _indices_of_one_curve(seed, t_out)
        print(f"  search curve {r + 1}/{NR} (seed {seed}): {n_samples} samples integrated", flush=True)

    np.savez_compressed(
        'results/dummy_test_replicates.npz',
        S1=S1, ST=ST, t=t_out, seeds=np.array(SEEDS),
        param_names=np.array(problem['names']), outputs=np.array(OUTPUTS)
    )

    # --- final time ---
    rows = []
    for j, out in enumerate(OUTPUTS):
        p_S1 = dummy_test(S1[:, i_end, j, :])
        p_ST = dummy_test(ST[:, i_end, j, :])
        for i, name in enumerate(param_names):
            s1, st = S1[:, i_end, j, i], ST[:, i_end, j, i]
            rows.append({
                'output': out,
                'parameter': name,
                'S1_mean': s1.mean(), 'S1_sd': s1.std(ddof=1), 'S1_p': p_S1[i],
                'S1_significant': bool(p_S1[i] < alpha_bonferroni),
                'S1_above_cutoff': bool(s1.mean() > CUTOFF),
                'ST_mean': st.mean(), 'ST_sd': st.std(ddof=1), 'ST_p': p_ST[i],
                'ST_significant': bool(p_ST[i] < alpha_bonferroni),
                'ST_above_cutoff': bool(st.mean() > CUTOFF),
            })
        # the dummy itself, as the reference level of the comparison
        rows.append({
            'output': out, 'parameter': 'dummy',
            'S1_mean': S1[:, i_end, j, i_dummy].mean(), 'S1_sd': S1[:, i_end, j, i_dummy].std(ddof=1),
            'S1_p': np.nan, 'S1_significant': False, 'S1_above_cutoff': False,
            'ST_mean': ST[:, i_end, j, i_dummy].mean(), 'ST_sd': ST[:, i_end, j, i_dummy].std(ddof=1),
            'ST_p': np.nan, 'ST_significant': False, 'ST_above_cutoff': False,
        })
    df_end = pd.DataFrame(rows)
    df_end.to_csv('results/dummy_test_end.csv', index=False)

    # --- time-dependent ---
    rows = []
    for k, tk in enumerate(t_out):
        for j, out in enumerate(OUTPUTS):
            p_S1 = dummy_test(S1[:, k, j, :])
            p_ST = dummy_test(ST[:, k, j, :])
            for i, name in enumerate(param_names):
                rows.append({
                    'time': tk, 'output': out, 'parameter': name,
                    'S1_mean': S1[:, k, j, i].mean(), 'S1_p': p_S1[i],
                    'S1_significant': bool(p_S1[i] < alpha_bonferroni),
                    'ST_mean': ST[:, k, j, i].mean(), 'ST_p': p_ST[i],
                    'ST_significant': bool(p_ST[i] < alpha_bonferroni),
                })
    pd.DataFrame(rows).to_csv('results/dummy_test_time.csv', index=False)

    # --- summary printed to the log ---
    print(f"\nFinal time t = {efast.t[-1]:.0f} days "
          f"(significance against the dummy vs. the {CUTOFF} cutoff)")
    for out in OUTPUTS:
        sub = df_end[(df_end['output'] == out) & (df_end['parameter'] != 'dummy')]
        dum = df_end[(df_end['output'] == out) & (df_end['parameter'] == 'dummy')].iloc[0]
        print(f"\n=== {out}  (dummy: S1 = {dum['S1_mean']:.2e}, ST = {dum['ST_mean']:.2e}) ===")
        print(f"{'param':8s} {'S1':>10s} {'p(S1)':>10s} {'sig':>4s} {'>cut':>5s} "
              f"{'ST':>10s} {'p(ST)':>10s} {'sig':>4s} {'>cut':>5s}  disagreement")
        for _, row in sub.iterrows():
            flip = [k for k, a, b in (('S1', 'S1_significant', 'S1_above_cutoff'),
                                      ('ST', 'ST_significant', 'ST_above_cutoff')) if row[a] != row[b]]
            print(f"{row['parameter']:8s} {row['S1_mean']:10.6f} {row['S1_p']:10.2e} "
                  f"{'yes' if row['S1_significant'] else 'no':>4s} "
                  f"{'yes' if row['S1_above_cutoff'] else 'no':>5s} "
                  f"{row['ST_mean']:10.6f} {row['ST_p']:10.2e} "
                  f"{'yes' if row['ST_significant'] else 'no':>4s} "
                  f"{'yes' if row['ST_above_cutoff'] else 'no':>5s}  {','.join(flip) if flip else '-'}")
        n_sig = int(sub['S1_significant'].sum()), int(sub['ST_significant'].sum())
        n_cut = int(sub['S1_above_cutoff'].sum()), int(sub['ST_above_cutoff'].sum())
        print(f"influential parameters out of {n_par}: "
              f"test S1 {n_sig[0]}, cutoff S1 {n_cut[0]}; test ST {n_sig[1]}, cutoff ST {n_cut[1]}")

    print("\nFiles 'dummy_test_end.csv', 'dummy_test_time.csv' and "
          "'dummy_test_replicates.npz' were saved.")


if __name__ == '__main__':
    main()
