"""Stage I, task 3 (continued): the twist flow, where two-component mixtures beat the best single Gaussian by ~20 % of its gap
(i3_flows search).  Does the advantage keep growing with the number of components (which could change the power of the gap),
and is it scale invariant?  Two chains at sigma = 1e-3 start from the best K = 2 state of i3_search and add components one at a
time (K = 2, 3, 4, 6, 8), hill-climbing g* = (R - R*)/(kappa - R*) after each step (Gauss-Hermite with 4 nodes per axis: the
posterior is narrower than sigma and f is smooth on scale 1, so 4 nodes agree with 20 nodes to 1e-12).  Each best state is then
re-evaluated (same scale-free parameters) at sigma = 1e-4, 1e-3, 1e-2 (6 nodes, per_sd 10; check: 4 nodes, per_sd 14), and its
(kappa - R)/sigma recorded.  (A first run evaluated this with 16 nodes and unchunked memory and was killed for lack of memory.)
Usage: python i4_twist.py [steps]  -> results/i4_twist.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from i_tools import fields, KAPPA
from i3_flows import unpack, scales, rstar_flow
from h3_xsearch import grow

KIND = 'twist'; SIG = 1e-3


def gfun(x, K, sig, sv0, sw0, Rs, gh=4, per_sd=6):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 40) or np.any(sh[:, [0, 2]] > 2) or np.any(sh[:, [0, 2]] < -10) or np.any(np.abs(sh[:, 1]) > 8): return -np.inf, None
    ws, mus, Ss = unpack(x, K, KIND, sv0, sw0)
    if min(ws) < 0.01: return -np.inf, None
    try: r = fields(ws, mus, Ss, sig, kind=KIND, per_sd=per_sd, gh=gh)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return -np.inf, None
    if abs(r['mass'] - 1) > 1e-6: return -np.inf, None
    return (r['R'] - Rs) / (KAPPA[KIND] - Rs), r


def climb(x, K, steps, rng, sv0, sw0, Rs):
    fx = gfun(x, K, SIG, sv0, sw0, Rs)[0]; step = 0.15; acc = 0; t = 0
    while not np.isfinite(fx) and t < 30: x = x + rng.normal(0, 0.05, x.size); fx = gfun(x, K, SIG, sv0, sw0, Rs)[0]; t += 1
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.4); fy = gfun(y, K, SIG, sv0, sw0, Rs)[0]
        if fy > fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.6)); acc = 0
    return x, fx


def chain(seed, steps):
    rng = np.random.default_rng(seed); Rs = rstar_flow(KIND, SIG); sv0, sw0 = scales(KIND, SIG)
    S = [r for r in json.load(open('results/i3_search.json')) if r['kind'] == KIND and r['K'] == 2]
    x = np.array(max(S, key=lambda r: r['gstar'])['x']); K = 2; rec = []
    for Knew in [2, 3, 4, 6, 8]:
        if Knew > K: x = grow(x, K, Knew, rng); K = Knew
        x, fx = climb(x, K, steps, rng, sv0, sw0, Rs); rec.append({'K': K, 'gstar': fx, 'x': x.tolist()}); print(f'seed {seed} K={K} g* {fx:+.5f}', flush=True)
        json.dump(rec, open(f'results/i4_twist_chain{seed}.json', 'w'), indent=1, default=float)        # incremental save
    for r in rec:
        sc = []
        for s in [1e-4, 1e-3, 1e-2]:
            Rs_s = rstar_flow(KIND, s); a, b = scales(KIND, s)
            g, rr = gfun(np.array(r['x']), r['K'], s, a, b, Rs_s, gh=6, per_sd=10); g2, rr2 = gfun(np.array(r['x']), r['K'], s, a, b, Rs_s, gh=4, per_sd=14)
            sc.append({'sigma': s, 'gstar': g, 'gstar_per_sd14': g2, 'R': rr['R'], 'gap_over_sigma': (KAPPA[KIND] - rr['R']) / s, 'Rstar_gap_over_sigma': (KAPPA[KIND] - Rs_s) / s})
        r['scale'] = sc
    return {'seed': seed, 'chain': rec}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    with Pool(2) as pool:
        out = pool.starmap(chain, [(501, steps), (502, steps)])
    json.dump(out, open('results/i4_twist.json', 'w'), indent=1, default=float)
    for c in out:
        print(c['seed'], [(r['K'], round(r['gstar'], 4), [round(v['gap_over_sigma'], 4) for v in r['scale']]) for r in c['chain']])
