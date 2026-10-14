"""Stage M, task M1(b): second-order variation around the optimum of the coupled pendulums (eps = 0.1, 0.3, 1; sigma = 1e-5, 1e-4).
The optimum is taken from a full optimisation in the scaled frame of l3_coupled (V+, W+, q-, p-; scales sigma, sqrt sigma, sqrt sigma, 1;
lower-triangular factor with log-diagonal; the mean of q free), started from the leading-order ansatz and from random starts.
Parameters (14): the q-mean (2, units of sigma), the p-mean (2), the 10 factor entries.  The p-mean does not enter law 1 at all
(E[Df] depends only on the law of q, F only on S): two exact zero modes.  For the other 12 the gradient and the Hessian of
(kappa - R)/sigma are computed by central differences (steps 3e-3 and 1e-3; at 1e-4 the rounding noise of the gap, ~1e-12, divided by h^2 is ~1e-4).  Parameters that sit on the
box (V+ log-width at its lower end = rank-one limit, p- log-length at its upper end = infinitely long) are reported with their
gradient (sign = direction in which the gap decreases) and are excluded from the reduced Hessian, whose eigenvalues and
eigenvectors are reported.  Every near-flat mode (eigenvalue < 1e-3 x the largest) is scanned along the line x + t v, |t| <= 1,
and the lowest value found is compared with the optimum (a negative difference beyond the rounding level would mean a saddle).   -> results/m1_hessian.json"""
import numpy as np, json
from multiprocessing import Pool
from scipy.optimize import minimize
from l3_coupled import obj, unpack, kappa, ansatz_x, gap_scaled

NAMES = ['dq1', 'dq2'] + [f'L{i}{j}' for i, j in zip(*np.tril_indices(4))]   # frame order (V+, W+, q-, p-)
LOG = [2, 4, 7, 11]


def optimum(sig, eps, seed=0):
    rng = np.random.default_rng(seed); best = None
    starts = [ansatz_x(eps)] + [ansatz_x(eps) + rng.normal(0, 0.3, 12) for _ in range(5)]
    for x0 in starts:
        f = lambda x: obj(x, sig, eps)
        r = minimize(f, x0, method='Powell', options={'xtol': 1e-12, 'ftol': 1e-15, 'maxfev': 40000})
        r = minimize(f, r.x, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-16, 'maxfev': 40000, 'maxiter': 40000, 'adaptive': True})
        if best is None or r.fun < best.fun: best = r
    return best.x, float(best.fun)


def hess(f, x, h):
    n = len(x); H = np.zeros((n, n)); g = np.zeros(n); f0 = f(x)
    for i in range(n):
        e = np.zeros(n); e[i] = h; g[i] = (f(x + e) - f(x - e)) / (2 * h); H[i, i] = (f(x + e) - 2 * f0 + f(x - e)) / h ** 2
        for j in range(i):
            d = np.zeros(n); d[j] = h
            H[i, j] = H[j, i] = (f(x + e + d) - f(x + e - d) - f(x - e + d) + f(x - e - d)) / (4 * h * h)
    return g, H


def job(a):
    sig, eps = a; x, fx = optimum(sig, eps); f = lambda z: gap_scaled(z, sig, eps)
    onbox = [i for i in LOG if x[i] <= -14 + 1e-3 or x[i] >= 8 - 1e-3]
    free = [i for i in range(12) if i not in onbox]
    res = {'sigma': sig, 'eps': eps, 'kappa': kappa(eps), 'gap': fx, 'x': x.tolist(), 'on_box': {NAMES[i]: float(x[i]) for i in onbox}}
    for h in [3e-3, 1e-3]:
        g, H = hess(f, x, h); Hr = H[np.ix_(free, free)]; ev, U = np.linalg.eigh(Hr)
        soft = []
        for kk in range(len(ev)):
            if ev[kk] < 1e-3 * ev[-1]:                          # near-flat: scan the line x + t v, |t| <= 1
                v = np.zeros(12); v[free] = U[:, kk]; ts = np.linspace(-1, 1, 41); fs = [f(x + t * v) for t in ts]
                soft.append({'eig': float(ev[kk]), 'vector': {NAMES[free[j]]: round(float(U[j, kk]), 3) for j in range(len(free)) if abs(U[j, kk]) > 0.15},
                             'scan_min_minus_f0': float(min(fs) - fx), 'scan_max_minus_f0': float(max(fs) - fx)})
        res[f'h{h:g}'] = {'grad': dict(zip(NAMES, g.tolist())), 'grad_free_max': float(np.max(np.abs(g[free]))), 'eig_reduced': ev.tolist(), 'soft_modes': soft}
    print(sig, eps, fx, 'on box', res['on_box'], 'eig', np.round(res['h0.003']['eig_reduced'], 6), 'soft', [(round(m['eig'], 7), m['vector'], m['scan_min_minus_f0']) for m in res['h0.003']['soft_modes']], flush=True)
    return res


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(job, [(s, e) for e in [0.1, 0.3, 1.0] for s in [1e-4, 1e-5]])
    json.dump(out, open('results/m1_hessian.json', 'w'), indent=1)
