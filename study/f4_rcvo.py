"""Stage F, task 4: maximise rho * CV * Omega over states whose law term sits at lambda_rho (law gap = lambda_rho - law <= eps),
distance 0.1-0.6, pendulum.  Because fraction = [rho CV Omega (1 - lambda_rho) + D] / distance and distance = (1 - lambda_rho) + gap,
fraction > 1 needs rho CV Omega > 1 + (gap - D) / (1 - lambda_rho).  Chains start from the best stage-F states; eps = 0.01 and 0.003.
Usage: python f4_rcvo.py [n_steps].  Writes results/f4_rcvo.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from e1_local import pack, unpack
from f1_chain import evaluate, factors


def chain(args):
    name, x0, K, eps, step0, n_it, seed = args
    rng = np.random.default_rng(seed)
    def obj(x):
        r = evaluate(x, K)
        if r is None: return -np.inf, None
        fc = factors(r)
        if not (0.1 <= fc['dist'] <= 0.6): return -10 - abs(fc['dist'] - np.clip(fc['dist'], 0.1, 0.6)), r
        if fc['law_gap'] > eps: return -5 - (fc['law_gap'] - eps) / eps, r
        return fc['rhoCVOmega'], r
    x = np.array(x0); fx, rx = obj(x); track = []; step = step0; acc = 0
    for it in range(1, n_it + 1):
        y = x + rng.normal(0, step, x.size); fy, ry = obj(y)
        if fy > fx: x, fx, rx = y, fy, ry; acc += 1
        if it % 50 == 0: step = float(np.clip(step * (1.5 if acc > 10 else 0.7), 1e-4, 0.3)); acc = 0
        if it % 20 == 0 and rx is not None: track.append(dict(step=it, value=fx, **factors(rx)))
    ws, mus, Ss, sig = unpack(x, K)
    r10 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r20 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=20)
    return {'name': name, 'K': K, 'eps': eps, 'best_value': fx, 'track': track, 'final': factors(r10), 'R_per_sd20': float(r20['R']),
            'sigma': sig, 'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss], 'mass': float(r10['mass'])}


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    F = sorted(json.load(open('results/f1_chains.json')), key=lambda r: -r['final']['ratio'])
    jobs = []
    for i, r in enumerate(F[:2]):
        st = pack(r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']], r['sigma'])
        for j, eps in enumerate([0.003]):
            jobs.append((f"{r['name']}_eps{eps}", st, len(r['ws']), eps, 0.03, n, 40 + 10 * i + j))
    with Pool(2) as pool:
        out = pool.map(chain, jobs, chunksize=1)
    json.dump(out, open('results/f4_rcvo.json', 'w'), indent=1, default=float)
    for r in out:
        f = r['final']
        print(f"{r['name']:<36} rhoCVOmega {f['rhoCVOmega']:.4f} (rho {f['rho']:.3f} CV {f['CV']:.3f} Omega {f['Omega']:.3f}) gap {f['law_gap']:.4f} "
              f"ratio {f['ratio']:.4f} dist {f['dist']:.3f} R {f['R']:.5f}")
