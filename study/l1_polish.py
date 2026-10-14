"""Stage L, task L1(a), family 1 (continued): gradient polish of the free mixtures.
The random hill climbing of l1_free.py was still improving after 2500 + 5000 steps, so the best K = 16 state of each sigma is
polished with L-BFGS-B (forward-difference gradient, step 1e-5; objective (1 - R)/sigma by i_tools.fields with per_sd 3, which agrees
with per_sd 6 and with a >= 401-point grid to ~1e-10 on these states), then grown to K = 24 and K = 32 by adding copies of existing
components at a small weight (log-weight - 5, so the grown state starts at the K = 16 value) and polished again.
Box: centres |x| <= 20 (V in units of sigma, W in units of 2 sqrt(sigma)), V-width log in [-10, 2], W-length log in [-10, 2],
|shear| <= 6 (the box of l1_free); no lower limit on the weights.
Every final state is diagnosed with l_tools.diag (161- and 401-point grids) and re-evaluated at the other sigma.
Usage: python l1_polish.py [maxiter]  -> results/l1_polish_sigma<sigma>.json"""
import numpy as np, json, sys, glob
from multiprocessing import Pool
from scipy.optimize import minimize
from h2_adversarial import unpack
from i_tools import fields
from l_tools import diag


def gap(x, K, sig, per_sd=3):
    ws, mus, Ss = unpack(x, K, sig)
    try: r = fields(ws, mus, Ss, sig, per_sd=per_sd)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return 10.0
    return (1 - r['R']) / sig if abs(r['mass'] - 1) < 1e-6 and np.isfinite(r['R']) else 10.0


def bounds(K):
    return [(-30, 5)] * K + [(-20, 20)] * (2 * K) + [(-10, 2), (-6, 6), (-10, 2)] * K


def grow(x, K, Knew, rng):
    w, c, s = list(x[:K]), list(x[K:3 * K].reshape(K, 2)), list(x[3 * K:].reshape(K, 3))
    order = np.argsort(-np.array(w))
    for i in range(Knew - K):
        j = order[i % K]; w.append(w[j] - 5.0); c.append(c[j] + rng.normal(0, 0.3, 2)); s.append(np.clip(s[j] + rng.normal(0, 0.2, 3), [-10, -6, -10], [2, 6, 2]))
    return np.concatenate([w, np.ravel(c), np.ravel(s)])


def polish(x, K, sig, maxiter):
    x = x.copy(); x[:K] -= x[:K].max()                       # weights are relative: put the largest log-weight at 0
    b = bounds(K); x = np.clip(x, [l for l, h in b], [h for l, h in b])
    r = minimize(gap, x, args=(K, sig), method='L-BFGS-B', bounds=b, options={'maxiter': maxiter, 'maxfun': 10 ** 7, 'eps': 1e-5, 'ftol': 1e-13, 'gtol': 1e-9})
    return r.x, r.fun, int(r.nit), str(r.message)


def chain(sig, seed, maxiter):
    rng = np.random.default_rng(seed)
    cand = [o for f in glob.glob(f'results/l1_free_s*_sigma{sig:g}.json') for o in json.load(open(f)) if o['K'] == 16]
    o = min(cand, key=lambda o: o['diag']['gap_over_sigma']); x, K = np.array(o['x']), 16; rec = []
    for Knew in [16, 24, 32]:
        if Knew > K: x = grow(x, K, Knew, rng); K = Knew
        g0 = gap(x, K, sig); x, fx, nit, msg = polish(x, K, sig, maxiter)
        ws, mus, Ss = unpack(x, K, sig); d = diag(ws, mus, Ss, sig); dfine = diag(ws, mus, Ss, sig, min_axis=401)
        other = 1e-2 if sig == 1e-3 else 1e-3; dother = diag(*unpack(x, K, other), other)
        rec.append({'K': K, 'sigma': sig, 'gap_start': g0, 'gap_search': fx, 'nit': nit, 'message': msg, 'diag': d,
                    'gap_fine_grid': dfine['gap_over_sigma'], 'diag_other_sigma': dother, 'x': x.tolist()})
        json.dump(rec, open(f'results/l1_polish_sigma{sig:g}.json', 'w'), indent=1, default=float)
        print(f"sigma {sig:g} K={K}: start {g0:.5f} -> (1-R)/sigma {d['gap_over_sigma']:.5f} (fine grid {dfine['gap_over_sigma']:.5f}; at sigma {other:g}: "
              f"{dother['gap_over_sigma']:.5f}) nit {nit}  Theta' {d['ThetaP']:.4f} rho {d['rho']:.4f}  [{msg}]", flush=True)
    return rec


if __name__ == '__main__':
    maxiter = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with Pool(2) as pool:
        pool.starmap(chain, [(1e-3, 61, maxiter), (1e-2, 62, maxiter)])
