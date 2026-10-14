"""Stage M, task M1(a): assumption-free optimisation of the best Gaussian state of the coupled pendulums.
All parameters are free: the mean (q1, q2, p1, p2) anywhere (not fixed at the top; (0, pi), (pi, 0) and any other point allowed) and
the covariance S = L L^T with L a FULL 4 x 4 matrix in the plain coordinates (q1, q2, p1, p2): 4 + 16 numbers for the 4 + 10 physical
parameters (the 6 extra ones are the rotations L -> L O, which leave S unchanged).  No frame, no scales, no assumed shape.
(A first version used a lower-triangular factor with log-diagonal; its log-diagonal ran off to -1e5 on plateaus and most starts
stalled at (kappa - R)/sigma ~ 6 already at sigma = 0.1.  The full factor has no such plateau.)
Random start: mean q uniform on the torus, mean p ~ N(0, 1), L entries ~ N(0, 0.5^2).
Continuation in sigma: 1e-1 -> 1e-2 -> 1e-3 -> 1e-4 -> 1e-5 (L-BFGS-B, central-difference gradient), each result starting the next;
the values at every sigma are recorded.  Objective (kappa - R)/sigma by m_tools.gap (QR evaluation).
Results are written to results/m1_free_eps<eps>.json after every start (restartable: finished starts are skipped).
Usage: python m1_free.py <eps> [n_starts] [workers]"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import minimize
from m_tools import coupled, find_kappa, gap, coefficient

SIGS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def obj(x, sysm, sig, k):
    try:
        v = gap(sysm, x[:2], x[4:].reshape(4, 4), sig, k)
    except (np.linalg.LinAlgError, ValueError):
        return 1e8
    return v if np.isfinite(v) else 1e8


def grad(x, sysm, sig, k, h=1e-7):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h; g[i] = (obj(x + e, sysm, sig, k) - obj(x - e, sysm, sig, k)) / (2 * h)
    return g


def opt(x, sysm, sig, k):
    r = minimize(obj, x, args=(sysm, sig, k), jac=grad, method='L-BFGS-B', bounds=[(None, None)] * 4 + [(-1e3, 1e3)] * 16,
                 options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-11})
    return r.x, float(r.fun)


def job(a):
    eps, i, k = a; sysm = coupled(eps); rng = np.random.default_rng(1000 * int(round(eps * 10 + 50)) + i)
    x = np.concatenate([rng.uniform(-np.pi, np.pi, 2), rng.normal(0, 1, 2), rng.normal(0, 0.5, 16)]); out = {'start': i, 'x0': x.tolist(), 'gap': {}}
    for s in SIGS:
        x, f = opt(x, sysm, s, k); out['gap'][f'{s:g}'] = f
    out['x'] = x.tolist(); out['mq'] = (np.mod(x[:2] + np.pi, 2 * np.pi) - np.pi).tolist()
    S = x[4:].reshape(4, 4) @ x[4:].reshape(4, 4).T; out['S'] = S.tolist()
    return out


if __name__ == '__main__':
    eps = float(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 60; W = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    fn = f'results/m1_free_eps{eps:g}.json'
    k, tops = find_kappa(coupled(eps), starts=100)
    res = json.load(open(fn)) if os.path.exists(fn) else {'eps': eps, 'kappa': k, 'tops': [t.tolist() for t in tops], 'formula': coefficient(coupled(eps), tops[0], k)['coef'], 'runs': []}
    done = {r['start'] for r in res['runs']}
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, [(eps, i, k) for i in range(n) if i not in done]):
            res['runs'].append(r); json.dump(res, open(fn, 'w'), indent=1)
            print(f"eps {eps:g} start {r['start']}: " + ' '.join(f"{s} {v:.7f}" for s, v in r['gap'].items()) + f"  mq {np.round(r['mq'], 4)}  (formula {res['formula']:.6f})", flush=True)
