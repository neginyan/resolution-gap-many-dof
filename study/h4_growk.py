"""Stage H, task 2 (continued): does the advantage of mixtures over the best single Gaussian keep growing with the number of
components K?  Starting from the best K = 4 state of h3_xsearch (sigma = 1e-3, scale-invariant regime), add one component at a time
(a perturbed copy of an existing one) and hill-climb g* = (R - R*)/(1 - R*) again: K = 5, 6, 7, 8.  Two independent chains.
Usage: python h4_growk.py [steps]   -> results/h4_growk.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from h2_adversarial import unpack, evaluate
from h3_xsearch import grow
from h1_rstar import rstar
from obs_eval2 import evaluate_full

SIG = 1e-3


def climb(x, K, steps, rng, Rs):
    def obj(x):
        sh = x[3 * K:].reshape(K, 3)
        if np.any(np.abs(x[K:3 * K]) > 20) or np.any(sh[:, [0, 2]] > 2) or np.any(sh[:, [0, 2]] < -10) or np.any(np.abs(sh[:, 1]) > 6): return -np.inf
        r = evaluate(x, K, SIG); return -np.inf if r is None else (r['R'] - Rs) / (1 - Rs)
    fx = obj(x); step = 0.15; acc = 0; t = 0
    while not np.isfinite(fx) and t < 30: x = x + rng.normal(0, 0.05, x.size); fx = obj(x); t += 1
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.4); fy = obj(y)
        if fy > fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.6)); acc = 0
    return x, fx


def chain(seed, steps):
    rng = np.random.default_rng(seed); Rs = rstar(SIG)['Rstar']
    h3 = [r for r in json.load(open('results/h3_xsearch.json')) if r['K'] == 4]
    x = np.array(max(h3, key=lambda r: r['gstar'])['x']); K = 4
    x, fx = climb(x, K, steps // 2, rng, Rs); rec = [{'K': K, 'gstar': fx, 'x': x.tolist()}]
    print(f'seed {seed} K=4 g* {fx:+.5f}', flush=True)
    for Knew in [5, 6, 7, 8]:
        x = grow(x, K, Knew, rng); K = Knew; x, fx = climb(x, K, steps, rng, Rs)
        rec.append({'K': K, 'gstar': fx, 'x': x.tolist()}); print(f'seed {seed} K={K} g* {fx:+.5f}', flush=True)
    for r in rec:          # final checks at per_sd 10 / 16 and at sigma = 1e-4 (scale invariance)
        ws, mus, Ss = unpack(np.array(r['x']), r['K'], SIG)
        a = evaluate_full(ws, mus, Ss, SIG, grid='vw', per_sd=10); b = evaluate_full(ws, mus, Ss, SIG, grid='vw', per_sd=16)
        ws4, mus4, Ss4 = unpack(np.array(r['x']), r['K'], 1e-4); c = evaluate_full(ws4, mus4, Ss4, 1e-4, grid='vw', per_sd=10); R4 = rstar(1e-4)['Rstar']
        r.update({'R': a['R'], 'R16': b['R'], 'gstar10': (a['R'] - Rs) / (1 - Rs), 'gstar_sigma1e-4': (c['R'] - R4) / (1 - R4), 'mass': a['mass']})
    return {'seed': seed, 'chain': rec}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    with Pool(2) as pool:
        out = pool.starmap(chain, [(9001, steps), (9002, steps)])
    json.dump(out, open('results/h4_growk.json', 'w'), indent=1, default=float)
    for c in out: print(c['seed'], [(r['K'], round(r['gstar10'], 5), round(r['gstar_sigma1e-4'], 5)) for r in c['chain']])
