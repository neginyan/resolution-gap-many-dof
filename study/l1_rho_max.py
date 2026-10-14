"""Stage L, task L1(b): is the focus-slope ratio rho = -E[u d_w w]/E[w d_w u] bounded below 1 over ALL states?
Adversarial MAXIMISATION of rho (and, separately, of r_sup = E|u d_w w|/(j sup|u|)) over free K = 2, 3, 4 mixtures at sigma = 1e-3
(random hill climbing from random needle-like starts near the top, l1_free box).  The gap (1 - R)/sigma, Theta' and the state-wise
bound of the found states are recorded: a state with large rho but a large gap shows that a bound on rho is not needed for the gap,
only for this route of proof.   -> results/l1_rho_max.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from h2_adversarial import unpack
from l_tools import diag

SIG = 1e-3


def val(x, K, key):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 20) or np.any(sh[:, [0, 2]] > 2) or np.any(sh[:, [0, 2]] < -10) or np.any(np.abs(sh[:, 1]) > 6) or np.any(np.abs(x[:K]) > 15): return -np.inf, None
    try: d = diag(*unpack(x, K, SIG), SIG)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return -np.inf, None
    v = d[key]
    return (v if np.isfinite(v) else -np.inf), d


def job(a):
    K, key, seed, steps = a; rng = np.random.default_rng(seed); best = None
    for restart in range(3):
        x = np.concatenate([rng.normal(0, 1, K), rng.normal(0, 1.0, 2 * K), np.ravel([[rng.uniform(-4, 1), rng.normal(0, 1), rng.uniform(-6, -1)] for _ in range(K)])])
        fx, dx = val(x, K, key); step = 0.3; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.4); fy, dy = val(y, K, key)
            if fy > fx: x, fx, dx = y, fy, dy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 1.0)); acc = 0
        if dx is not None and (best is None or fx > best[0]): best = (fx, x, dx)
    fx, x, d = best
    return {'K': K, 'key': key, 'value': fx, 'x': x.tolist(), 'diag': d}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    jobs = [(K, key, 70 + 10 * K + i, steps) for K in [2, 3, 4] for i, key in enumerate(['rho', 'r_sup'])]
    out = []
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(out, open('results/l1_rho_max.json', 'w'), indent=1, default=float); d = r['diag']
            print(f"K={r['K']} max {r['key']}: {r['value']:.4f}   gap {d['gap_over_sigma']:.4f}  Theta' {d['ThetaP']:.4f}  geo {d['geo']:.4f}  rho {d['rho']:.4f}  "
                  f"eps~/sigma {d['eps_tilde_over_sigma']:.4f}  bound {d['bound_over_sigma']:.4f}", flush=True)
