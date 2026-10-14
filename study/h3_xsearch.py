"""Stage H, task 2 (continued): how far can a mixture go beyond the best single Gaussian?  Long searches of
g* = (R - R*(sigma))/(1 - R*(sigma)) in the scale-invariant regime (sigma = 1e-3; the parametrisation of h2_adversarial.unpack is
scale free: widths along V in units of sigma, along W in units of 2 sqrt(sigma)), K = 2, 3, 4, started from the best states found
by h2_adversarial (and the probe), from crossing needles ("X" shapes: two rank-one needles with opposite tilts through one point),
and at random.  Shapes are kept within e^{+-}... of the needle scale (a, c in [-10, 2]) so the grids stay small.
The best state of each K is then re-evaluated over sigma = 1e-5 ... 0.3 (same x) to test scale invariance.
Usage: python h3_xsearch.py [steps] [restarts]  -> results/h3_xsearch.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from h2_adversarial import unpack, evaluate, needle_params
from h1_rstar import rstar
from obs_eval2 import evaluate_full

SIG = 1e-3


def seeds_from_files():
    xs = []
    try:
        for r in json.load(open('results/h2_adversarial.json')):
            if r['gstar'] > 0: xs.append((r['K'], np.array(r['x'])))
    except FileNotFoundError:
        pass
    try:
        xs.append((2, np.array(json.load(open('results/h2_probe.json'))['x'])))
    except FileNotFoundError:
        pass
    return xs


def grow(x, K, Knew, rng):
    """add components (copies of existing ones, perturbed) to go from K to Knew."""
    w, c, s = list(x[:K]), list(x[K:3 * K].reshape(K, 2)), list(x[3 * K:].reshape(K, 3))
    while len(w) < Knew:
        i = rng.integers(len(w)); w.append(w[i] - 1.0); c.append(c[i] + rng.normal(0, 0.3, 2)); s.append(s[i] + rng.normal(0, 0.3, 3))
    return np.concatenate([w, np.ravel(c), np.ravel(s)])


def job(args):
    K, seed, steps, restarts = args
    rng = np.random.default_rng(seed); Rs = rstar(SIG)['Rstar']; a0, b0, c0 = needle_params(SIG)
    pool = [(k, x) for k, x in seeds_from_files()]
    def ok(x):
        sh = x[3 * K:].reshape(K, 3)
        return not (np.any(np.abs(x[K:3 * K]) > 20) or np.any(sh[:, [0, 2]] > 2) or np.any(sh[:, [0, 2]] < -10) or np.any(np.abs(sh[:, 1]) > 6))
    def obj(x):
        if not ok(x): return -np.inf
        r = evaluate(x, K, SIG); return -np.inf if r is None else (r['R'] - Rs) / (1 - Rs)
    runs = []
    for rs in range(restarts):
        kind = rs % 3
        if kind == 0 and pool:                         # from a stored positive state (grown/trimmed to K)
            k, x0 = pool[rng.integers(len(pool))]
            if k < K: x = grow(x0, k, K, rng)
            elif k > K:
                idx = np.argsort(-x0[:k])[:K]; x = np.concatenate([x0[:k][idx], x0[k:3 * k].reshape(k, 2)[idx].ravel(), x0[3 * k:].reshape(k, 3)[idx].ravel()])
            else: x = x0.copy()
            x = x + rng.normal(0, 0.05, x.size); kind_name = 'stored'
        elif kind == 1:                                # crossing needles: opposite tilts, common centre slightly off the top along V
            shp = [[rng.uniform(-3, -0.5), b0 * (-1) ** k * rng.uniform(0.3, 1.2) + (0 if k % 2 == 0 else 0), rng.uniform(-3, 0)] for k in range(K)]
            off = rng.normal(0, 0.5); x = np.concatenate([rng.normal(0, 0.3, K), np.ravel([[off + rng.normal(0, 0.1), rng.normal(0, 0.05)] for _ in range(K)]), np.ravel(shp)])
            kind_name = 'crossing'
        else:
            shp = [[rng.normal(-1, 1), rng.normal(0, 0.8), rng.normal(-0.5, 0.7)] for _ in range(K)]
            x = np.concatenate([rng.normal(0, 0.3, K), rng.normal(0, 1.0, 2 * K), np.ravel(shp)]); kind_name = 'random'
        x = np.where(np.isfinite(x), x, 0.0)
        fx = obj(x); t = 0
        while not np.isfinite(fx) and t < 30: x = x + rng.normal(0, 0.1, x.size); x[3 * K:] = np.clip(x[3 * K:], -9.5, 1.9); fx = obj(x); t += 1
        start = fx; step = 0.2; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.5); fy = obj(y)
            if fy > fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        runs.append({'restart': rs, 'kind': kind_name, 'start_gstar': start, 'gstar': fx, 'x': x.tolist()})
    best = max(runs, key=lambda r: r['gstar']); x = np.array(best['x'])
    scale = []
    for s in [1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3]:
        ws, mus, Ss = unpack(x, K, s); r = evaluate_full(ws, mus, Ss, s, grid='vw', per_sd=10); r16 = evaluate_full(ws, mus, Ss, s, grid='vw', per_sd=16)
        Rss = rstar(s)['Rstar']; scale.append({'sigma': s, 'R': r['R'], 'R16': r16['R'], 'Rstar': Rss, 'gstar': (r['R'] - Rss) / (1 - Rss),
                                              'one_minus_R_over_sigma': (1 - r['R']) / s, 'mass': r['mass']})
    ws, mus, Ss = unpack(x, K, SIG)
    return {'K': K, 'seed': seed, 'gstar': scale[2]['gstar'], 'runs': [{k: v for k, v in u.items() if k != 'x'} for u in runs], 'x': best['x'],
            'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss], 'scale': scale}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600; restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    jobs = [(K, 7000 + 10 * K + j, steps, restarts) for K in [2, 3, 4] for j in range(2)]
    out = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); print(f"K={r['K']} seed {r['seed']}: best g* {r['gstar']:+.5f}; restarts {[(u['kind'][0], round(u['gstar'], 4)) for u in r['runs']]}; "
                                 f"scale g* {[round(v['gstar'], 4) for v in r['scale']]}", flush=True)
            json.dump(out, open('results/h3_xsearch.json', 'w'), indent=1, default=float)
