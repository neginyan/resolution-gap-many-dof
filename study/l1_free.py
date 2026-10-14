"""Stage L, task L1(a), family 1: free mixtures with K = 8, 12, 16 components, adversarial MINIMISATION of (1 - R)/sigma
(pendulum, kappa = 1) at sigma = 1e-3 and 1e-2.  Scale-free parametrisation of h2_adversarial.unpack (V widths in units of sigma,
W lengths in units of 2 sqrt(sigma)).  Each chain starts from the best K = 8 state of stage H (h4_growk), hill-climbs, then adds
components one at a time up to 12 and 16 (h3_xsearch.grow) and hill-climbs again.  'cont' mode: continue the best K = 16 state (more steps) and grow it to K = 24, 32 (convergence in K).  Evaluation: i_tools.fields (per_sd 6) during
the search; the final state of every K is diagnosed with l_tools.diag (grid of 161 and of >= 401 points per axis) and re-evaluated at the other sigma
(scale invariance).  Usage: python l1_free.py [steps]  -> results/l1_free_s<seed>_sigma<sigma>.json (written after every K), results/l1_free.json (all)
Two independent chains (seeds) per sigma."""
import numpy as np, json, sys
from multiprocessing import Pool
from h2_adversarial import unpack
from h3_xsearch import grow
from i_tools import fields
from l_tools import diag


def gap(x, K, sig):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 20) or np.any(sh[:, [0, 2]] > 2) or np.any(sh[:, [0, 2]] < -10) or np.any(np.abs(sh[:, 1]) > 6): return np.inf
    ws, mus, Ss = unpack(x, K, sig)
    if min(ws) < 0.005: return np.inf
    try: r = fields(ws, mus, Ss, sig, per_sd=6)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return np.inf
    if abs(r['mass'] - 1) > 1e-6: return np.inf
    return (1 - r['R']) / sig


def climb(x, K, sig, steps, rng):
    fx = gap(x, K, sig); step = 0.1; acc = 0
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = gap(y, K, sig)
        if fy < fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.5)); acc = 0
    return x, fx


def chain(sig, seed, steps, start=None, Ks=(8, 12, 16)):
    rng = np.random.default_rng(seed)
    if start is None:
        c = json.load(open('results/h4_growk.json')); best = max([x for ch in c for x in ch['chain'] if x['K'] == 8], key=lambda x: x['gstar10'])
        x = np.array(best['x']); K = 8
    else:
        x, K = np.array(start[0]), start[1]
    rec = []
    for Knew in Ks:
        if Knew > K: x = grow(x, K, Knew, rng); K = Knew
        x, fx = climb(x, K, sig, steps, rng)
        ws, mus, Ss = unpack(x, K, sig); d6 = diag(ws, mus, Ss, sig); d10 = diag(ws, mus, Ss, sig, min_axis=401)
        other = 1e-2 if sig == 1e-3 else 1e-3; wo, mo, So = unpack(x, K, other); dother = diag(wo, mo, So, other)
        rec.append({'K': K, 'sigma': sig, 'gap_search': fx, 'diag': d6, 'gap_fine_grid': d10['gap_over_sigma'], 'diag_other_sigma': dother, 'x': x.tolist()})
        json.dump(rec, open(f'results/l1_free_s{seed}_sigma{sig:g}.json', 'w'), indent=1, default=float)
        print(f"seed {seed} sigma {sig:g} K={K}: (1-R)/sigma {d6['gap_over_sigma']:.5f} (fine grid {d10['gap_over_sigma']:.5f}; at sigma {other:g}: {dother['gap_over_sigma']:.5f})  "
              f"Theta' {d6['ThetaP']:.4f} rho {d6['rho']:.4f}", flush=True)
    return rec


def best_of(sig, K):
    import glob
    r = [o for f in glob.glob(f'results/l1_free_s3[1-4]_sigma{sig:g}.json') for o in json.load(open(f)) if o['K'] == K]
    o = min(r, key=lambda o: o['diag']['gap_over_sigma']); return o['x'], K


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[1] == 'cont':
    # continuation: the best K = 16 state of the first run for each sigma, climbed further at K = 16, then grown to 24 and 32
    steps = int(sys.argv[2])
    with Pool(2) as pool:
        out = pool.starmap(chain, [(1e-3, 41, steps, best_of(1e-3, 16), (16, 24, 32)), (1e-2, 42, steps, best_of(1e-2, 16), (16, 24, 32))])
elif __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    with Pool(2) as pool:
        out = pool.starmap(chain, [(1e-3, 31, steps), (1e-2, 32, steps), (1e-3, 33, steps), (1e-2, 34, steps)])
    json.dump([r for c in out for r in c], open('results/l1_free.json', 'w'), indent=1, default=float)
