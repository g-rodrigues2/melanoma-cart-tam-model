"""Does the eFAST ranking depend on the horizon at which the output is taken?

The design of `efast_sensitivity.py` is held fixed, with the same bounds, base sample size, harmonic
order, seed and initial condition; only the time at which the scalar output is read varies, from the
sixty days used in the chapter out to four hundred. The motivation is that beta1 acts slowly near the
Hopf threshold, since the amplitude of the cycle it creates opens as the square root of the distance
to the threshold and the rate of approach to the attractor vanishes there, so sixty days may simply
be too short for it to show.

The sixty-day window is the one the chapter reports, and it is also the only one of experimental
relevance, since the B16 experiment of Jilani et al. runs for less than thirty days. The longer
horizons are statements about the asymptotic behavior of the model and not about anything that can be
measured, which is why they are reported in Section 3.3.3 as a check on the scope of the ranking and
not as results in their own right.

Outputs (results/): horizon_ST.csv, horizon_S1.csv and horizon_diagnostics.txt.

Run: python efast_horizon.py
"""
import os
import time

import numpy as np
import pandas as pd
from SALib.analyze import fast
from SALib.sample import fast_sampler

import efast_sensitivity as design

HORIZONS = [60.0, 100.0, 150.0, 200.0, 300.0, 400.0]
N_RESAMPLES = 20


def oscillatory_mask(X):
    """Samples whose coexistence equilibrium is unstable (Q < 0), with mu3 = 0 as in the design."""
    a1, a2, a3, b1, K1, K2, K3, u1, k2, k3 = (X[:, i] for i in range(10))
    Ts = k3 * K3 / (a3 - k3)
    Ms = K2 * (1 - k2 / (a2 * Ts))
    Cs = a1 / u1 * (1 - Ts / K1) * (1 + b1 * Ms)
    J11 = a1 * (1 - 2 * Ts / K1) * (1 + b1 * Ms) - u1 * Cs
    J12 = a1 * Ts * (1 - Ts / K1) * b1
    J13 = -u1 * Ts
    J21 = a2 * Ms * (1 - Ms / K2)
    J22 = a2 * Ts * (1 - 2 * Ms / K2) - k2
    J31 = Cs * a3 * K3 / (K3 + Ts) ** 2
    Q = (-(J11 + J22)) * (J11 * J22 - J12 * J21 - J13 * J31) - J13 * J22 * J31
    return Q < 0


def analyze(y, seed, n_resamples=N_RESAMPLES):
    si = fast.analyze(design.problem, y, M=design.M_fast, num_resamples=n_resamples,
                      print_to_console=False, seed=seed)
    return {k: np.asarray(v) for k, v in si.items()}


def main():
    os.makedirs("results", exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    t0 = time.time()
    names = design.param_names
    X = np.ascontiguousarray(fast_sampler.sample(design.problem, design.N, M=design.M_fast,
                                                 seed=design.seed))
    t_out = np.ascontiguousarray(np.array(HORIZONS))
    Y = design.solve_all(np.array(design.TMC0), X, t_out, design.RTOL, design.ATOL)
    osc = oscillatory_mask(X)
    K1 = X[:, names.index('K1')]

    say("=" * 96)
    say("eFAST RANKING AGAINST THE HORIZON (design of efast_sensitivity.py, output T at each time)")
    say("=" * 96)
    say(f"  {X.shape[0]} samples, p = {X.shape[1]}, N = {design.N}, M = {design.M_fast}, "
        f"seed = {design.seed}")
    say(f"  E* unstable (oscillatory attractor) in {100 * osc.mean():.1f}% of the samples")
    say("")
    say(f"  {'day':>5s} {'decades':>8s} {'CV':>6s} {'% at K1':>8s}  {'ST(beta1)':>10s}  "
        f"{'top 5 by ST':46s}")

    ST, S1 = {}, {}
    for k, h in enumerate(HORIZONS):
        y = Y[:, k, 0]
        si = analyze(y, design.seed)
        si_log = analyze(np.log10(np.maximum(y, 1e-300)), design.seed)
        ST[h], S1[h] = si['ST'], si['S1']
        top = [names[i] for i in np.argsort(-si['ST'])][:5]
        top_log = [names[i] for i in np.argsort(-si_log['ST'])][:5]
        say(f"  {h:5.0f} {np.log10(y.max() / max(y.min(), 1e-300)):8.2f} "
            f"{y.std() / y.mean():6.2f} {100 * np.mean(y > 0.99 * K1):8.1f}  "
            f"{si['ST'][names.index('beta1')]:10.4f}  {', '.join(top):46s}")
        if k == len(HORIZONS) - 1:
            say("")
            say(f"  at {h:.0f} days the top five by ST are, on Y     : {', '.join(top)}")
            say(f"                                     and on log Y : {', '.join(top_log)}")
            say(f"  the two lists hold the same five parameters: {sorted(top) == sorted(top_log)}")

    say("")
    say("  first-order and total-effect indices at the two ends")
    say(f"  {'parameter':10s} {'S1 60d':>9s} {'ST 60d':>9s} {'S1 400d':>9s} {'ST 400d':>9s}")
    for i in np.argsort(-ST[400.0]):
        say(f"  {names[i]:10s} {S1[60.0][i]:9.4f} {ST[60.0][i]:9.4f} "
            f"{S1[400.0][i]:9.4f} {ST[400.0][i]:9.4f}")
    say(f"  sum of S1: {S1[60.0].sum():.3f} at 60 days, {S1[400.0].sum():.3f} at 400 days")
    say(f"  sum of ST: {ST[60.0].sum():.3f} at 60 days, {ST[400.0].sum():.3f} at 400 days")

    amp = ((Y[:, -2:, 0].max(axis=1) - Y[:, -2:, 0].min(axis=1))
           / np.maximum(Y[:, -1, 0], 1.0))
    say("")
    say("  relative excursion of T between days 300 and 400, by attractor")
    say(f"    stable E*   median {np.median(amp[~osc]):.3f}")
    say(f"    unstable E* median {np.median(amp[osc]):.3f}")

    pd.DataFrame(ST, index=names).T.to_csv(os.path.join("results", "horizon_ST.csv"),
                                           index_label="t")
    pd.DataFrame(S1, index=names).T.to_csv(os.path.join("results", "horizon_S1.csv"),
                                           index_label="t")
    say("")
    say(f"  total time {time.time() - t0:.0f} s")
    with open(os.path.join("results", "horizon_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nresults in results/horizon_*")


if __name__ == "__main__":
    main()
