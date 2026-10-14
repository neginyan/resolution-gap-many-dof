"""Stage E, task 4: how far do P_v >= 0 and P_w >= 0 hold?  Adversarial search that minimises P_v (or P_w) directly
(pendulum, kappa = 1), K = 2 and 3, sigma in {0.03, 0.1, 0.3, 1}; parametrisation and constraints as in c1_adversarial.py,
evaluated on the grid aligned with V, W.  Writes results/e4_valley.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from c1_adversarial import unpack

KEEP = ['R', 'Pv', 'Pw', 'D', 'H_nonPD_mass', 'mass_Hvv_neg', 'mass_Hww_neg', 'Pv_neg_part', 'Pw_neg_part', 'valley_depth_mean',
        'valley_depth_max', 'law_term', 'mass']


def obj(x, K, sig, target):
    ws, mus, Ss = unpack(x, K)
    ev = [np.linalg.eigvalsh(S) for S in Ss]
    if min(e.min() for e in ev) < 1e-3 or max(e.max() for e in ev) > 1.0 or min(ws) < 0.02: return -np.inf, None
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=6)
    if abs(r['mass'] - 1) > 1e-6: return -np.inf, None
    return -r[target], r


def search(args):
    K, sig, target, seed = args
    rng = np.random.default_rng(seed); best = (-np.inf, None)
    for rs in range(4):
        x = np.concatenate([rng.normal(0, 0.5, K), np.ravel([[rng.uniform(0, 2 * np.pi), rng.normal(0, 0.7)] for _ in range(K)]),
                            np.ravel([[rng.uniform(-2.5, -0.5), rng.normal(0, 0.5), rng.uniform(-2.5, -0.5)] for _ in range(K)])])
        fx, _ = obj(x, K, sig, target); step = 0.4
        for it in range(70):
            y = x + rng.normal(0, step, x.size); fy, _ = obj(y, K, sig, target)
            if fy > fx: x, fx = y, fy
            else: step = max(step * 0.97, 0.02)
        if fx > best[0]: best = (fx, x)
    ws, mus, Ss = unpack(best[1], K)
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r20 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=20)
    out = {'K': K, 'sigma': sig, 'target': target, 'x': best[1].tolist(), target + '_per_sd20': float(r20[target])}
    out.update({k: float(r[k]) for k in KEEP}); return out


if __name__ == '__main__':
    jobs = [(K, s, t, 3000 + 10 * K + int(100 * s) + (0 if t == 'Pv' else 5)) for t in ['Pv', 'Pw'] for K in [2, 3] for s in [0.03, 0.1, 0.3, 1.0]]
    with Pool(2) as pool:
        out = pool.map(search, jobs, chunksize=1)
    json.dump(out, open('results/e4_valley.json', 'w'), indent=1)
    for r in out:
        print(f"min {r['target']} K={r['K']} sigma={r['sigma']}: Pv {r['Pv']:+.4f} Pw {r['Pw']:+.4f} R {r['R']:.4f} valley {r['H_nonPD_mass']:.3f} "
              f"(v {r['mass_Hvv_neg']:.3f}, w {r['mass_Hww_neg']:.3f}) depth mean {r['valley_depth_mean']:.3f} max {r['valley_depth_max']:.2f}")
