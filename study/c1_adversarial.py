"""Stage C, proposal 1: adversarial search for the largest R / kappa (pendulum, kappa = 1), aimed at the valleys
(regions where the local Fisher matrix H is not positive definite), where the bound (f) <= kappa is no longer guaranteed.

Objectives (maximised by hill climbing with random restarts over weights, means and Cholesky factors):
  'R'      : R itself (= R / kappa)
  'excess' : covariance term + D  (= R - law term), the part of R not controlled by lambda_rho
  'room'   : (cov + D) / (kappa - law term); Conjecture 1 <=> room <= 1 (usage: python c1_adversarial.py room)
K = 2 and 3 components, sigma in {0.03, 0.1, 0.3, 1.0}. Half of the restarts start near the hyperbolic point q = pi.
Covariances: eigenvalues in [1e-3, 1] (s.d. 0.03 ... 1). The search uses a grid capped at 501 points per axis; the best
state of every run is re-evaluated at full adaptive resolution and R is checked at 1.5x that resolution.
Writes results/c1_adversarial.json.
"""
import numpy as np, json, os, sys, itertools
from multiprocessing import Pool
from obs_eval import evaluate, mixture_grid
from obs_eval2 import evaluate_full

SIGMAS = [0.03, 0.1, 0.3, 1.0]


def unpack(x, K):
    w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum())
    mus, Ss = [], []
    for k in range(K):
        mus.append(np.array([x[K + 2 * k], x[K + 2 * k + 1]]))
        a, b, c = x[3 * K + 3 * k: 3 * K + 3 * k + 3]
        L = np.array([[np.exp(a), 0], [b, np.exp(c)]]); Ss.append(L @ L.T + 1e-6 * np.eye(2))
    return ws, mus, Ss


def bhattacharyya(m1, S1, m2, S2, sig):
    T1, T2 = S1 + sig ** 2 * np.eye(2), S2 + sig ** 2 * np.eye(2); Tm = (T1 + T2) / 2; d = m1 - m2
    DB = d @ np.linalg.solve(Tm, d) / 8 + 0.5 * np.log(np.linalg.det(Tm) / np.sqrt(np.linalg.det(T1) * np.linalg.det(T2)))
    return float(np.exp(-DB))


def objective(x, K, sig, target, n_cap=501):
    ws, mus, Ss = unpack(x, K)
    ev = [np.linalg.eigvalsh(S) for S in Ss]
    if min(e.min() for e in ev) < 1e-3 or max(e.max() for e in ev) > 1.0 or min(ws) < 0.02:
        return -np.inf, None
    n = min(mixture_grid(ws, mus, Ss, sig)[3], n_cap)
    r = evaluate_full(ws, mus, Ss, sig, n=n)
    val = {'R': r['R'], 'excess': r['R'] - r['law_term'], 'room': (r['R'] - r['law_term']) / (1.0 - r['law_term'])}[target]
    return val, r


def search(args):
    K, sig, target, seed = args
    rng = np.random.default_rng(seed); best = (-np.inf, None)
    for rs in range(4):
        x = np.concatenate([rng.normal(0, 0.5, K),
                            np.ravel([[np.pi + rng.normal(0, 0.7) if rs % 2 == 0 else rng.uniform(0, 2 * np.pi), rng.normal(0, 0.7)] for _ in range(K)]),
                            np.ravel([[rng.uniform(-2.5, -0.5), rng.normal(0, 0.5), rng.uniform(-2.5, -0.5)] for _ in range(K)])])
        fx, _ = objective(x, K, sig, target); step = 0.4
        for it in range(70):
            y = x + rng.normal(0, step, x.size)
            fy, _ = objective(y, K, sig, target)
            if fy > fx: x, fx = y, fy
            else: step = max(step * 0.97, 0.02)
        if fx > best[0]: best = (fx, x)
    return {'K': K, 'sigma': sig, 'target': target, 'x': best[1].tolist(), 'search_value': best[0]}


def finalize(rec):
    K, sig = rec['K'], rec['sigma']; ws, mus, Ss = unpack(np.array(rec['x']), K)
    r = evaluate_full(ws, mus, Ss, sig)
    n15 = int(1.5 * r['n']); r15 = evaluate(ws, mus, Ss, sig, n=n15)
    ov = max(bhattacharyya(mus[i], Ss[i], mus[j], Ss[j], sig) for i, j in itertools.combinations(range(K), 2))
    keep = ['R', 'law_term', 'f_cov_part', 'D', 'f', 'e', 'lam_rho_direct', 'lam_comp', 'H_nonPD_mass', 'A', 'R_switch', 'n', 'mass', 'tweedie_err']
    rec.update({k: float(r[k]) for k in keep})
    rec.update({'cov': float(r['f'] - r['law_term']), 'excess': float(r['R'] - r['law_term']), 'overlap_max': ov,
                'room': float((r['R'] - r['law_term']) / (1.0 - r['law_term'])), 'R_n1.5': float(r15['R']),
                'ws': [float(w) for w in ws], 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss]})
    return rec


if __name__ == '__main__':
    targets = sys.argv[1:] or ['R', 'excess']
    seed_off = {'R': 0, 'excess': 7, 'room': 13}
    jobs = [(K, s, t, 1000 * K + int(100 * s) + seed_off[t]) for K in [2, 3] for s in SIGMAS for t in targets]
    with Pool(2) as pool:
        found = pool.map(search, jobs, chunksize=1)
    out = [finalize(r) for r in found]                 # full-resolution re-evaluation, one at a time (memory)
    os.makedirs('results', exist_ok=True)
    fn = 'results/c1_adversarial.json' if targets == ['R', 'excess'] else 'results/c1_adversarial_' + '_'.join(targets) + '.json'
    json.dump(out, open(fn, 'w'), indent=1)
    for r in sorted(out, key=lambda r: -r['R']):
        print(f"K={r['K']} sigma={r['sigma']:<5} target={r['target']:<6} R={r['R']:.4f} (1.5x {r['R_n1.5']:.4f}) law={r['law_term']:.4f} "
              f"cov={r['cov']:+.4f} D={r['D']:+.4f} room={r['room']:.3f} valley={r['H_nonPD_mass']:.3f} overlap={r['overlap_max']:.3f} lam_rho={r['lam_rho_direct']:.3f}")
