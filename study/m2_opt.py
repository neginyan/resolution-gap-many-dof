"""Stage M, task M2: full-parameter optimisation of the best Gaussian state, AFTER the predictions of m2_predict.py were written
(results/m2_prediction.json, SHA-256 in logs/m2_prediction_hash.txt; this script checks the hash before it starts).
Parametrisation as in m1_free (no assumed shape): mean of q (n) free, S = L L^T with L a full 2n x 2n matrix in the plain coordinates.
(B) precise optimum: full-parameter optimisation in the scaled frame at the top (m_tools.frame_optimise: coordinates and units only,
every S representable; L-BFGS-B + Powell, restarted until no improvement), started from the leading-order state, at sigma = 1e-4,
1e-5, 1e-6; quadratic extrapolation to sigma -> 0 compared with the prediction.
(A) basins: `n_random` assumption-free random starts in plain coordinates (mean q uniform on the torus, L entries N(0, 0.5^2)),
continuation sigma = 1e-1 -> ... -> 1e-5; counted: how many reach the optimum of (B) (within 1e-4) and whether any goes below it.
(A first version used only (A) with the ansatz as one start: the plain-coordinate L-BFGS stopped early for n = 3 (1e-4 relative).)
Shape of the optimum at sigma = 1e-5: s.d. along (V+, W+, e_k, (0, e_k)) and the F-weights sigma^2 (S + sigma^2)^-1 along them.
Restartable: results/m2_opt.json is written after every system.   Usage: python m2_opt.py [n_random] [workers]"""
import numpy as np, json, os, sys, hashlib
from multiprocessing import Pool
from scipy.optimize import minimize
from m_tools import gap, ansatz_state, frame
from m2_predict import SYSTEMS

SIGS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
HASH = 'd84ba44aab9c4e5b09a2079589db0f6f69770ff314f08a92f7562139cfed7664'


def obj(x, s, sig, k):
    n = s.n
    try: v = gap(s, x[:n], x[n:].reshape(2 * n, 2 * n), sig, k)
    except (np.linalg.LinAlgError, ValueError): return 1e8
    return v if np.isfinite(v) else 1e8


def grad(x, s, sig, k, h=1e-7):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h; g[i] = (obj(x + e, s, sig, k) - obj(x - e, s, sig, k)) / (2 * h)
    return g


def run(x, s, k):
    n = s.n; vals = {}
    for sig in SIGS:
        r = minimize(obj, x, args=(s, sig, k), jac=grad, method='L-BFGS-B', bounds=[(None, None)] * n + [(-1e3, 1e3)] * (4 * n * n),
                     options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-11})
        x = r.x; vals[f'{sig:g}'] = float(r.fun)
    return x, vals


def shape(s, x, q0, sig):
    """s.d. of the state and F-weights sigma^2 e^T F e along the frame directions; F from the QR factor (stable; a plain inverse of
    Fr^T S Fr + sigma^2 gave a negative weight for n = 4)."""
    n = s.n; L = x[n:].reshape(2 * n, 2 * n); Fr, _ = frame(s, q0)
    names = ['V+', 'W+'] + [f'q{k}' for k in range(n - 1)] + [f'p{k}' for k in range(n - 1)]
    R = np.linalg.qr(np.vstack([L.T, sig * np.eye(2 * n)]), mode='r'); Ri = np.linalg.inv(R)
    return {nm: {'sd_over_sqrt_sigma': float(np.linalg.norm(L.T @ Fr[:, i]) / np.sqrt(sig)), 'F_times_sigma2': float(np.sum((Ri.T @ Fr[:, i]) ** 2) * sig ** 2)}
            for i, nm in enumerate(names)}


def job(a):
    idx, nrand = a; kind, n, e, s = SYSTEMS[idx]; P = json.load(open('results/m2_prediction.json'))['systems'][idx]; k = P['kappa']; q0 = np.array(P['tops'][0])
    from m_tools import frame_series
    fr, _ = frame_series(s, q0, k, (1e-4, 1e-5)); _, (mq5, L5) = fr, _   # (B) full-parameter optimisation in the scaled frame
    fr6, _ = frame_series(s, q0, k); fr = fr6
    sg = np.array([1e-4, 1e-5, 1e-6]); gv = np.array([fr[f'{v:g}'] for v in sg]); p = np.polyfit(sg, gv, 2)
    rng = np.random.default_rng(77 + idx); rand = []      # (A) assumption-free random starts (plain coordinates), basins
    for i in range(nrand):
        x = np.concatenate([rng.uniform(-np.pi, np.pi, n), rng.normal(0, 0.5, 4 * n * n)]); xr, vr = run(x, s, k)
        rand.append({'vals': vr, 'mq': (np.mod(xr[:n] + np.pi, 2 * np.pi) - np.pi).tolist()})
    x5 = np.concatenate([mq5, L5.ravel()])
    return {'kind': kind, 'n': n, 'eps': e, 'name': s.name, 'kappa': k, 'prediction': P['prediction'], 'frame_opt': fr,
            'extrap_sigma0': float(p[-1]), 'extrap_slope': float(p[-2]), 'opt_rel_diff_extrap_vs_prediction': float((p[-1] - P['prediction']) / P['prediction']),
            'random': rand, 'random_below_frame_opt_1e-5': int(sum(r['vals']['1e-05'] < fr['1e-05'] * (1 - 1e-7) for r in rand)),
            'random_reaching_1e-4_rel': int(sum(abs(r['vals']['1e-05'] - fr['1e-05']) < 1e-4 * fr['1e-05'] for r in rand)),
            'shape_1e-5': shape(s, x5, q0, 1e-5)}


if __name__ == '__main__':
    h = hashlib.sha256(open('results/m2_prediction.json', 'rb').read()).hexdigest()
    assert h == HASH, 'prediction file changed after registration'
    nrand = int(sys.argv[1]) if len(sys.argv) > 1 else 8; W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    fn = 'results/m2_opt.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['kind'], o['n'], o['eps']) for o in out}
    todo = [i for i, (kind, n, e, s) in enumerate(SYSTEMS) if (kind, n, e) not in done]
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, [(i, nrand) for i in todo]):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1)
            print(f"{r['name']}: prediction {r['prediction']:.7f}  optimum {r['frame_opt']}  extrapolated {r['extrap_sigma0']:.7f} "
                  f"(rel {r['opt_rel_diff_extrap_vs_prediction']:.1e});  random starts: {r['random_reaching_1e-4_rel']}/{len(r['random'])} reach it, {r['random_below_frame_opt_1e-5']} below it", flush=True)
