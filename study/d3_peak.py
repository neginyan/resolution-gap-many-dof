"""Stage D, task 3: states close to the top of the stretch (distance kappa - law term small), pendulum, kappa = 1.

Part 'ratio' : maximise excess / (kappa - law) over states with distance < 0.3, K = 2, 3, sigma in {0.03, 0.1, 0.3, 1}
               (excess = R - law term = covariance term + D).  States that cannot reach distance < 0.3 are reported as such.
Part 'sweep' : for target distances d = 0.3 ... 0.001, maximise the excess over states with distance <= d (K = 2, 3; sigma is a
               search variable, 1e-4 ... 1), and record the spread of the local stretch Var(abar) and the focus mismatch
               E||H - F|| / tr F of the best state.  Slope of log(max excess) vs log(d): >= 1 consistent with Conjecture 1 near
               the top, ~ 1/2 would make the top a candidate for a violation.
Part 'sweep2': the same, but three of the four restarts start from a feasible seed (seed_state) and use smaller steps; needed
because random starts did not reach distances <= 0.01 (part 'sweep').
Hill climbing with random restarts (means near the hyperbolic point (pi, 0), sizes scaled with sqrt(d)); grid aligned with V, W
(per_sd = 6 during the search); best states re-evaluated at per_sd = 10 and checked at per_sd = 20.
Writes results/d3_<part>.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full, V, W

DISTS = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
KEEP = ['R', 'law_term', 'f', 'D', 'H_nonPD_mass', 'abar_var', 'focus_mismatch', 'B_balance', 'B_prime', 'mass', 'lam_rho_direct']


def unpack(x, K, scale):
    w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum())
    mus, Ss = [], []
    for k in range(K):
        mus.append(np.array([np.pi, 0.0]) + scale * np.array([x[K + 2 * k], x[K + 2 * k + 1]]))
        a, b, c = x[3 * K + 3 * k: 3 * K + 3 * k + 3]
        L = scale * np.array([[np.exp(a), 0], [b, np.exp(c)]]); Ss.append(L @ L.T + 1e-12 * np.eye(2))
    return ws, mus, Ss


def evaluate(x, K, scale, sig, per_sd=6):
    ws, mus, Ss = unpack(x, K, scale)
    ev = [np.linalg.eigvalsh(S) for S in Ss]
    if min(ws) < 0.02 or max(e.max() for e in ev) > 1.0 or min(e.min() for e in ev) < (1e-3 * scale) ** 2:
        return None
    sds = [np.sqrt(e.max()) for e in ev]
    if max(sds) / min(sds) > 30: return None                       # keep the grid resolvable
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=per_sd)
    if abs(r['mass'] - 1) > 1e-6: return None
    return r


def seed_state(d, K, rng, scale):
    """A feasible starting point near the top: K copies of the single Gaussian at (pi, 0) that is narrow along V and long along W,
    with sigma = d/4, s_v = sigma/2, s_w^2 = 4 sqrt(s_v^2 + sigma^2) (distance ~ 1.1 sigma for one Gaussian), slightly perturbed."""
    sig = d / 4; sv = sig / 2; sw = np.sqrt(4 * np.sqrt(sv ** 2 + sig ** 2))
    x = list(rng.normal(0, 0.2, K)) + list(rng.normal(0, 0.05 * sv / scale, 2 * K))
    for _ in range(K):
        S = (sv * rng.uniform(0.7, 1.5)) ** 2 * np.outer(V, V) + (sw * rng.uniform(0.7, 1.3)) ** 2 * np.outer(W, W)
        L = np.linalg.cholesky(S) / scale; x += [np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])]
    return np.array(x + [np.log(sig)])


def search(args):
    part, K, sig_fixed, d, seed = args
    rng = np.random.default_rng(seed)
    seeded = part == 'sweep2'
    if seeded: part = 'sweep'
    scale = np.sqrt(d) if part == 'sweep' else 0.5
    nx = 6 * K + (1 if sig_fixed is None else 0)
    def sig_of(x): return sig_fixed if sig_fixed is not None else float(np.exp(np.clip(x[-1], np.log(1e-4), 0)))
    def obj(x):
        r = evaluate(x[:6 * K], K, scale, sig_of(x))
        if r is None: return -np.inf, None
        dist = 1 - r['law_term']; exc = r['R'] - r['law_term']
        lim = d if part == 'sweep' else 0.3
        if dist > lim:
            return -10 - (dist - lim) / lim, r                      # infeasible: push towards the target distance
        return (exc if part == 'sweep' else exc / dist), r
    best = (-np.inf, None, None)
    for rs in range(4):
        x = np.concatenate([rng.normal(0, 0.5, K), rng.normal(0, 0.5, 2 * K),
                            np.ravel([[rng.uniform(-3, 0), rng.normal(0, 0.3), rng.uniform(-1, 0.5)] for _ in range(K)])])
        if sig_fixed is None: x = np.append(x, np.log(d) + rng.uniform(-3, 0))
        if seeded and rs < 3: x = seed_state(d, K, rng, scale)
        fx, rx = obj(x); step = 0.15 if seeded else 0.4
        for it in range(70):
            y = x + rng.normal(0, step, x.size); fy, ry = obj(y)
            if fy > fx: x, fx, rx = y, fy, ry
            else: step = max(step * 0.97, 0.02)
        if fx > best[0]: best = (fx, x, rx)
    fx, x, _ = best
    rec = {'part': 'sweep2' if seeded else part, 'K': K, 'sigma_fixed': sig_fixed, 'target_dist': d, 'search_value': fx, 'x': None if x is None else x.tolist(),
           'scale': scale}
    if x is None or not np.isfinite(fx):
        rec['feasible'] = False; return rec
    sig = sig_of(x); ws, mus, Ss = unpack(x[:6 * K], K, scale)
    r10 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r20 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=20)
    rec.update({k: float(r10[k]) for k in KEEP})
    rec.update({'sigma': sig, 'dist': 1 - r10['law_term'], 'excess': r10['R'] - r10['law_term'], 'cov': r10['f'] - r10['law_term'],
                'R_per_sd20': float(r20['R']), 'feasible': bool(fx > -10), 'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss]})
    rec['ratio'] = rec['excess'] / rec['dist']
    return rec


if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'ratio':
        jobs = [('ratio', K, s, 0.3, 100 * K + int(1000 * s)) for K in [2, 3] for s in [0.03, 0.1, 0.3, 1.0]]
    else:
        jobs = [(part, K, None, d, (500 if part == 'sweep' else 700) * K + i) for K in [2, 3] for i, d in enumerate(DISTS)]
    with Pool(2) as pool:
        out = pool.map(search, jobs, chunksize=1)
    json.dump(out, open(f'results/d3_{part}.json', 'w'), indent=1, default=float)
    for r in out:
        if r.get('feasible'):
            print(f"{r['part']} K={r['K']} target {r['target_dist']} sigma={r['sigma']:.4g}: dist={r['dist']:.4g} excess={r['excess']:+.4g} "
                  f"ratio={r['ratio']:+.3f} cov={r['cov']:+.3g} D={r['D']:+.2g} var(abar)={r['abar_var']:.3g} mismatch={r['focus_mismatch']:.3g} "
                  f"R20-R10={r['R_per_sd20'] - r['R']:.1e}")
        else:
            print(f"{r['part']} K={r['K']} sigma={r['sigma_fixed']} target {r['target_dist']}: no feasible state found")
