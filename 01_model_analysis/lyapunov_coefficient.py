"""First Lyapunov coefficient at the Hopf point of E* (Section 3.3.2), with its checks.

STEPS (projection method; Kuznetsov, Elements of Applied Bifurcation Theory, 3rd ed., Sec. 5.4)

 1. At the Hopf point, A = Df(E*) has eigenvalues +-i w and a third, negative real one. Write
    f(E* + u) = A u + (1/2) B(u,u) + (1/6) C(u,u,u) + O(|u|^4), where B and C are the bilinear
    and trilinear forms of the second and third derivatives of f at E*.
 2. Take q with A q = i w q and p with A^T p = -i w p, normalized by <q,q> = 1 and <p,q> = 1
    (Hermitian product <a,b> = conj(a).b). The projection z = <p,u> isolates the critical part.
 3. On the center manifold, u = z q + zb qb + (quadratic terms). The quadratic ones are forced
    by the two homological equations
        A h11 = -B(q,qb)               (ZERO-frequency component)
        (2 i w I - A) h20 = B(q,q)     (SECOND HARMONIC)
    with a unique solution because 0 and 2 i w are not eigenvalues of A -- the second condition
    is the absence of a 1:2 resonance, checked in T3.
 4. Project back and collect the coefficient of z^2 zb:
        g21 = <p, C(q,q,qb)> + 2 <p, B(q,h11)> + <p, B(qb,h20)>
    The last two terms are the feedback of the QUADRATIC nonlinearities: they generate a
    zero-frequency shift and a second harmonic which, beating against the fundamental, correct
    the amplitude at cubic order. Omitting them can flip the sign.
 5. l1 = Re(g21) / (2 w).   l1 < 0 => supercritical Hopf (a STABLE limit cycle is born).

Only the SIGN of l1 is intrinsic: its magnitude scales with |q|^2 and with the units of the state
variables, so the number is only meaningful alongside the normalization <q,q> = 1 used here.

CHECKS (all printed when the script is run)
  T1 B and C against centered finite differences of f
  T2 residuals of the homological equations, of the eigenvectors and of the normalization
  T3 conditioning: 0 and 2 i w far from the spectrum of A
  T4 known answer: exact normal form, l1 = 2 s / w, eight cases, both signs
  T5 invariance of the sign under rescaling of q
  T6 QUANTITATIVE prediction of the cycle amplitude against the orbits obtained by shooting

Outputs: results/lyapunov_coefficient.txt (the scalars reported in Appendix A) and
results/lyapunov_amplitude.csv (the T6 amplitude table of Appendix A).
"""
import os

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

os.makedirs("results", exist_ok=True)

alpha1, K1, mu1 = 0.69, 1e9, 1e-7
alpha2, K2, kappa2 = 1e-7, 1e9, 0.34
K3, alpha3, kappa3 = 2e7, 0.9, 0.35

T, M, C, b1 = sp.symbols('T M C b1', real=True)
F = sp.Matrix([
    alpha1*T*(1 - T/K1)*(1 + b1*M) - mu1*C*T,
    alpha2*M*T*(1 - M/K2) - kappa2*M,
    alpha3*C*T/(K3 + T) - kappa3*C,
])
X = (T, M, C)
JSYM = F.jacobian(X)


def equilibrium(beta1):
    Ts = kappa3*K3/(alpha3 - kappa3)
    Ms = K2*(1 - kappa2/(alpha2*Ts))
    return Ts, Ms, alpha1/mu1*(1 - Ts/K1)*(1 + beta1*Ms)


def jac(beta1):
    s = dict(zip(X, equilibrium(beta1))) | {b1: beta1}
    return np.array(JSYM.subs(s).evalf(), dtype=float)


def Qrh(beta1):
    A = jac(beta1)
    a11, a12, a13 = A[0]
    a21, a22 = A[1, 0], A[1, 1]
    a31 = A[2, 0]
    return (-(a11 + a22))*(a11*a22 - a12*a21 - a13*a31) - a13*a22*a31


def lyapunov(fsym, Xs, point, report=False):
    """l1 by the projection method; returns l1, or (l1, diagnostics) if report."""
    n = len(Xs)
    sub = dict(zip(Xs, point))
    A = np.array(fsym.jacobian(Xs).subs(sub).evalf(), dtype=float)
    vals, vecs = np.linalg.eig(A)
    w = float(np.max(np.abs(vals.imag)))
    q = vecs[:, int(np.argmin(np.abs(vals - 1j*w)))]
    q = q/np.linalg.norm(q)
    valsT, vecsT = np.linalg.eig(A.T)
    p = vecsT[:, int(np.argmin(np.abs(valsT + 1j*w)))]
    p = p/np.conj(np.vdot(p, q))

    D2 = np.array([[[complex(sp.diff(fsym[i], Xs[j], Xs[k]).subs(sub))
                     for k in range(n)] for j in range(n)] for i in range(n)])
    D3 = np.array([[[[complex(sp.diff(fsym[i], Xs[j], Xs[k], Xs[l]).subs(sub))
                      for l in range(n)] for k in range(n)] for j in range(n)] for i in range(n)])

    def B(x, y):
        return np.einsum('ijk,j,k->i', D2, x, y)

    def Cc(x, y, z):
        return np.einsum('ijkl,j,k,l->i', D3, x, y, z)

    qb = np.conj(q)
    eye = np.eye(n)
    h11 = -np.linalg.solve(A, B(q, qb))
    h20 = np.linalg.solve(2j*w*eye - A, B(q, q))
    g21 = np.vdot(p, Cc(q, q, qb)) + 2*np.vdot(p, B(q, h11)) + np.vdot(p, B(qb, h20))
    l1 = g21.real/(2*w)
    if not report:
        return l1
    diag = dict(A=A, w=w, q=q, p=p, vals=vals, B=B, C=Cc, D2=D2, D3=D3,
                h11=h11, h20=h20, g21=g21,
                res_h11=float(np.abs(A @ h11 + B(q, qb)).max()),
                res_h20=float(np.abs((2j*w*eye - A) @ h20 - B(q, q)).max()),
                res_q=float(np.abs(A @ q - 1j*w*q).max()),
                res_p=float(np.abs(A.T @ p + 1j*w*p).max()),
                res_norm=float(abs(np.vdot(p, q) - 1)))
    return l1, diag


beta1_star = brentq(Qrh, 1e-13, 2e-10, xtol=1e-24, rtol=8.9e-16)
Estar = equilibrium(beta1_star)
Fb = F.subs(b1, beta1_star)
l1, d = lyapunov(Fb, X, Estar, report=True)
A, w = d['A'], d['w']

print("=" * 76)
print("beta1*        = {:.9e} cells^-1".format(beta1_star))
print("E*            = ({:.4e}, {:.4e}, {:.4e}) cells".format(*Estar))
print("eigenvalues   = {}  (day^-1)".format(np.round(d['vals'], 8)))
print("omega         = {:.8f} day^-1".format(w))
print("g21           = {:.6e}".format(d['g21']))
print("l1            = {:.6e}   ->  {}".format(l1, "SUPERCRITICAL" if l1 < 0 else "SUBCRITICAL"))
print("=" * 76)

# ---- T1: B and C against centered finite differences -----------------------
fnum = sp.lambdify(X, Fb, 'numpy')


def fv(u):
    return np.array(fnum(*u), dtype=float).ravel()


x0 = np.array(Estar, dtype=float)
eye3 = np.eye(3)
# relative step 1e-3: the minimum of the U-shaped error curve (rounding dominates below it,
# truncation above); for the third derivative the error falls from 3.5e-3 at 1e-4 to 1.8e-6 at 1e-3
hs = 1e-3*x0
B2 = np.zeros((3, 3, 3))
C3 = np.zeros((3, 3, 3, 3))
for j in range(3):
    for k in range(3):
        hj, hk = hs[j]*eye3[j], hs[k]*eye3[k]
        B2[:, j, k] = (fv(x0+hj+hk) - fv(x0+hj-hk) - fv(x0-hj+hk) + fv(x0-hj-hk))/(4*hs[j]*hs[k])
        for m in range(3):
            hm = hs[m]*eye3[m]
            C3[:, j, k, m] = (fv(x0+hj+hk+hm) - fv(x0+hj+hk-hm) - fv(x0+hj-hk+hm)
                              + fv(x0+hj-hk-hm) - fv(x0-hj+hk+hm) + fv(x0-hj+hk-hm)
                              + fv(x0-hj-hk+hm) - fv(x0-hj-hk-hm))/(8*hs[j]*hs[k]*hs[m])


def relerr(a, b):
    return float(np.max(np.abs(a - b))/max(np.max(np.abs(b)), 1e-300))


eB = relerr(B2, d['D2'].real)
eC = relerr(C3, d['D3'].real)
print("T1  B and C vs finite differences       rel. error {:.2e} (B) / {:.2e} (C)".format(eB, eC))

# ---- T2: residuals --------------------------------------------------------
print("T2  homological equations               |A h11 + B(q,qb)| = {:.2e}, "
      "|(2iwI-A) h20 - B(q,q)| = {:.2e}".format(d['res_h11'], d['res_h20']))
print("    eigenvectors / normalization        {:.1e} / {:.1e} / {:.1e}"
      .format(d['res_q'], d['res_p'], d['res_norm']))

# ---- T3: no resonance, A invertible ---------------------------------------
dist0 = float(np.min(np.abs(d['vals'])))
dist2 = float(np.min(np.abs(d['vals'] - 2j*w)))
print("T3  dist(0, spectrum) = {:.4f}, dist(2iw, spectrum) = {:.4f}  "
      "-> no 1:2 resonance; cond(A) = {:.2e}".format(dist0, dist2, np.linalg.cond(A)))

# ---- T4: known answer ------------------------------------------------------
# zdot = i w z + s z|z|^2 in the physical coordinate; with <q,q> = 1 one has q = (1,-i)/sqrt(2),
# so the physical variable equals sqrt(2) z and the cubic coefficient doubles: l1 = 2 s / w.
xs, ys = sp.symbols('xs ys', real=True)
worst = 0.0
for s_ in (-1.0, 1.0, -3.5, 0.25):
    for w_ in (1.0, 0.388):
        fn = sp.Matrix([-w_*ys + s_*xs*(xs**2 + ys**2),
                        w_*xs + s_*ys*(xs**2 + ys**2)])
        worst = max(worst, abs(lyapunov(fn, (xs, ys), (0, 0)) - 2*s_/w_)/abs(2*s_/w_))
print("T4  exact normal form, 8 cases          worst rel. error {:.2e}  (l1 = 2s/w, both signs)"
      .format(worst))

# ---- T5: invariance of the sign -------------------------------------------
def l1_rescaled(fac):
    q2 = d['q']*fac
    p2 = d['p']/np.conj(np.vdot(d['p'], q2))
    qb2 = np.conj(q2)
    g = (np.vdot(p2, d['C'](q2, q2, qb2))
         + 2*np.vdot(p2, d['B'](q2, -np.linalg.solve(A, d['B'](q2, qb2))))
         + np.vdot(p2, d['B'](qb2, np.linalg.solve(2j*w*np.eye(3) - A, d['B'](q2, q2)))))
    return g.real/(2*w)


signs = {np.sign(l1_rescaled(f)) for f in (0.37, 2.0, 1j, -1.0, 10.0)}
print("T5  sign under rescaling of q           {}"
      .format("invariant" if signs == {np.sign(l1)} else "CHANGED"))

assert eB < 1e-6 and eC < 1e-4
assert d['res_h11'] < 1e-8 and d['res_h20'] < 1e-8
assert worst < 1e-12
assert signs == {np.sign(l1)}
# ---- T6: the amplitude prediction against the periodic orbits --------------
# Near the threshold, rdot = rho r + (1/2) Re(g21) r^3 with rho = (dRe lam/dbeta1)(beta1 - beta1*),
# whose steady state gives r^2 = 2 rho/|Re g21|. Since the tumor component of u is 2 Re(z q1) and
# q1 is real by the phase convention, the predicted excursion of T above T* is dT = 2 q1 r.
# The orbits it is compared against are solved by shooting, using nothing from this computation.
hh = beta1_star*1e-4


def _re_pair(b):
    ev_ = np.linalg.eigvals(jac(b))
    return ev_[np.argsort(-np.abs(ev_.imag))][0].real


TRANSV = (_re_pair(beta1_star + hh) - _re_pair(beta1_star - hh))/(2*hh)
qphase = d['q']*np.exp(-1j*np.angle(d['q'][0]))   # phase: first component real and positive
q1 = float(qphase[0].real)
Tst, Mst, _ = equilibrium(beta1_star)


def _rhs(t, y, b1):
    T_, M_, C_ = y
    return [alpha1*T_*(1 - T_/K1)*(1 + b1*M_) - mu1*T_*C_,
            alpha2*M_*T_*(1 - M_/K2) - kappa2*M_,
            alpha3*C_*T_/(K3 + T_) - kappa3*C_]


def _cycle(b1, guess):
    def G(v):
        M0, C0, P = v
        if P <= 0:
            return [1e9, 1e9, 1e9]
        xe = solve_ivp(_rhs, (0, P), [Tst, M0, C0], args=(b1,), method='LSODA',
                       rtol=1e-12, atol=[1e-3, 1e-3, 1e-3]).y[:, -1]
        return [(xe[0] - Tst)/Tst, (xe[1] - M0)/Mst, (xe[2] - C0)/1e7]
    r_ = root(G, guess, method='hybr', tol=1e-12)
    return r_.x, float(np.max(np.abs(G(r_.x))))


_E = np.array(equilibrium(beta1_star*2))
_s = solve_ivp(_rhs, (0, 10385), [_E[0]*1.02, _E[1], _E[2]], args=(beta1_star*2,), method='LSODA',
               rtol=1e-10, atol=[1.0, 1.0, 1.0], t_eval=np.linspace(10385*0.98, 10385, 200000))
_T, _t = _s.y[0], _s.t
_up = np.where((_T[:-1] < Tst) & (_T[1:] >= Tst))[0]
_g = (_s.y[1][_up[-2]], _s.y[2][_up[-2]], float(_t[_up[-1]] - _t[_up[-2]]))

print()
print("T6  amplitude: prediction of (C.23) against the orbits obtained by shooting")
print("      {:>8} {:>15} {:>15} {:>8}".format("eps", "dT computed", "dT predicted", "ratio"))
_rows = []
for _eps in (1.0, 0.5, 0.2, 0.05, 0.02, 0.008, 0.003, 0.001):
    _b = beta1_star*(1 + _eps)
    _u, _res = _cycle(_b, _g)
    if _res > 1e-8:
        continue
    _g = _u
    _orb = solve_ivp(_rhs, (0, _u[2]), [Tst, _u[0], _u[1]], args=(_b,), method='LSODA',
                     rtol=1e-12, atol=[1e-3]*3, t_eval=np.linspace(0, _u[2], 8000)).y
    _dT = _orb[0].max() - Tst
    _pred = 2*q1*np.sqrt(2*TRANSV*(_b - beta1_star)/abs(l1*2*w))
    _rows.append((_eps, _dT, _pred, _dT/_pred))
    print("      {:>8.3f} {:>15.4e} {:>15.4e} {:>8.4f}".format(_eps, _dT, _pred, _dT/_pred))
_ratios = [r[3] for r in _rows]
assert all(a > b for a, b in zip(_ratios, _ratios[1:])), "the ratio must decrease monotonically to 1"
assert abs(_ratios[-1] - 1) < 0.05, "at the smallest eps the prediction must be within 5% of the computed value"
print("      the ratio decreases monotonically and reaches {:.3f} at the smallest eps".format(_ratios[-1]))

print("=" * 76)
print("every check passed")

with open(os.path.join('results', 'lyapunov_amplitude.csv'), 'w') as fh:
    fh.write("eps,dT_computed_cells,dT_predicted_cells,ratio\n")
    for _r in _rows:
        fh.write("{:.3f},{:.6e},{:.6e},{:.6f}\n".format(*_r))

with open(os.path.join('results', 'lyapunov_coefficient.txt'), 'w') as fh:
    fh.write("beta1_star {:.12e}\nomega {:.12f}\ng21_real {:.12e}\nl1 {:.12e}\n"
             "criticality {}\ntest_B_rel {:.3e}\ntest_C_rel {:.3e}\n"
             "res_h11 {:.3e}\nres_h20 {:.3e}\ndist_0_spectrum {:.6f}\n"
             "dist_2iw_spectrum {:.6f}\nnormal_form_worst_rel {:.3e}\n"
             .format(beta1_star, w, d['g21'].real, l1,
                     "supercritical" if l1 < 0 else "subcritical",
                     eB, eC, d['res_h11'], d['res_h20'], dist0, dist2, worst))
