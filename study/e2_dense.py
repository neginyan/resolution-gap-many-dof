"""Stage E, task 2: dense re-search of the largest excess at distances 0.0003 ... 0.003 from the top (pendulum, kappa = 1).

8 target distances (log-spaced), K = 2 and 3, 8 restarts per target: 4 from the feasible seed of d3_peak.seed_state, 2 from the
best stage-D states closest to the target (rescaled to the target), 2 random; 100 hill-climbing steps each (sigma free).
Distance constraint: kappa - law <= target.  Best of every restart is kept, so the spread between restarts is visible.
Writes results/e2_dense.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from d3_peak import unpack, evaluate, seed_state

TARGETS = list(np.logspace(np.log10(3e-4), np.log10(3e-3), 8))
KEEP = ['R', 'law_term', 'f', 'D', 'H_nonPD_mass', 'abar_var', 'omega_rms', 'u_cv', 'u_mean', 'cov_cs_bound', 'focus_mismatch', 'mass']


def pack(ws, mus, Ss, sig, scale):
    x = list(np.log(ws))
    for m in mus: x += list((np.array(m) - np.array([np.pi, 0.0])) / scale)
    for S in Ss:
        L = np.linalg.cholesky(np.array(S)) / scale; x += [np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])]
    return np.array(x + [np.log(sig)])


def run(args):
    K, d, seed, starts = args
    rng = np.random.default_rng(seed); scale = np.sqrt(d)
    def obj(x):
        r = evaluate(x[:6 * K], K, scale, float(np.exp(np.clip(x[-1], np.log(1e-5), 0))))
        if r is None: return -np.inf, None
        dist = 1 - r['law_term']
        if dist > d: return -10 - (dist - d) / d, r
        return r['R'] - r['law_term'], r
    results = []
    for kind, x0 in starts:
        x = x0 if x0 is not None else np.concatenate([rng.normal(0, 0.5, K), rng.normal(0, 0.5, 2 * K),
                                                       np.ravel([[rng.uniform(-3, 0), rng.normal(0, 0.3), rng.uniform(-1, 0.5)] for _ in range(K)]),
                                                       [np.log(d) + rng.uniform(-3, 0)]])
        if kind == 'seed': x = seed_state(d, K, rng, scale)
        fx, _ = obj(x); step = 0.15
        for it in range(100):
            y = x + rng.normal(0, step, x.size); fy, _ = obj(y)
            if fy > fx: x, fx = y, fy
            else: step = max(step * 0.98, 0.01)
        rec = {'K': K, 'target': d, 'start': kind, 'value': fx}
        if np.isfinite(fx) and fx > -10:
            sig = float(np.exp(np.clip(x[-1], np.log(1e-5), 0))); ws, mus, Ss = unpack(x[:6 * K], K, scale)
            r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r20 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=20)
            rec.update({k: float(r[k]) for k in KEEP})
            rec.update({'sigma': sig, 'dist': 1 - r['law_term'], 'excess': r['R'] - r['law_term'], 'ratio': (r['R'] - r['law_term']) / (1 - r['law_term']),
                        'R_per_sd20': float(r20['R']), 'feasible': True, 'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss]})
        else:
            rec['feasible'] = False
        results.append(rec)
    return results


if __name__ == '__main__':
    prev = [r for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) if r.get('feasible')]
    jobs = []
    for K in [2, 3]:
        for i, d in enumerate(TARGETS):
            near = sorted([r for r in prev if r['K'] == K], key=lambda r: abs(np.log(r['dist'] / d)))[:2]
            starts = [('seed', None)] * 4 + [('previous', pack(r['ws'], r['mus'], r['Ss'], r['sigma'], np.sqrt(d))) for r in near] + [('random', None)] * 2
            jobs.append((K, d, 9000 + 100 * K + i, starts))
    with Pool(2) as pool:
        out = sum(pool.map(run, jobs, chunksize=1), [])
    json.dump(out, open('results/e2_dense.json', 'w'), indent=1, default=float)
    for d in TARGETS:
        v = [r for r in out if r['target'] == d and r.get('feasible')]
        b = max(v, key=lambda r: r['excess'])
        print(f"target {d:.2e}: {len(v)} feasible; best excess {b['excess']:.3e} at dist {b['dist']:.3e} (ratio {b['ratio']:.3f}, K={b['K']}, {b['start']})")
