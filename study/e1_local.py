"""Stage E, task 1: local search that maximises the fraction of the room used, excess / (kappa - law), at mid distances
(0.1 <= kappa - law <= 0.3), pendulum, kappa = 1.

Seeds: (A) the state with fraction 0.830 (stage D sweep, K = 3, distance 0.288);
       (B) the state with R = 0.946 (stage C adversarial, K = 3, distance 0.111).
Free parameters: weights, means, Cholesky factors of every component, and sigma.  Chains: step sizes 0.01, 0.03, 0.1, each with
the seed's K and with K + 1 (the heaviest component split in two).  Grid aligned with V, W (per_sd = 6 during the search); every
chain's best state is re-evaluated at per_sd = 10 and checked at per_sd = 20.  Writes results/e1_local.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from c1_adversarial import unpack as unpack_c1


def pack(ws, mus, Ss, sig):
    x = list(np.log(ws))
    for m in mus: x += list(m)
    for S in Ss:
        L = np.linalg.cholesky(S); x += [np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])]
    return np.array(x + [np.log(sig)])


def unpack(x, K):
    w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum())
    mus = [np.array(x[K + 2 * k: K + 2 * k + 2]) for k in range(K)]; Ss = []
    for k in range(K):
        a, b, c = x[3 * K + 3 * k: 3 * K + 3 * k + 3]
        L = np.array([[np.exp(a), 0], [b, np.exp(c)]]); Ss.append(L @ L.T + 1e-14 * np.eye(2))
    return ws, mus, Ss, float(np.exp(x[-1]))


def split(ws, mus, Ss, sig, rng):
    k = int(np.argmax(ws)); ws = list(ws); w = ws[k] / 2; ws[k] = w; ws.append(w)
    ev, U = np.linalg.eigh(Ss[k]); d = 0.3 * np.sqrt(ev[-1]) * U[:, -1]
    mus = list(mus); m = mus[k]; mus[k] = m - d; mus.append(m + d)
    Ss = list(Ss); Ss.append(Ss[k] * rng.uniform(0.3, 0.8)); return ws, mus, Ss, sig


def evaluate(x, K, per_sd=6):
    ws, mus, Ss, sig = unpack(x, K)
    ev = [np.linalg.eigvalsh(S) for S in Ss]
    if min(ws) < 0.01 or max(e.max() for e in ev) > 1.0 or min(e.min() for e in ev) < 1e-10 or not (1e-4 <= sig <= 1):
        return None
    sds = [np.sqrt(e.max()) for e in ev]
    if max(sds) / min(sds) > 40: return None
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=per_sd)
    if abs(r['mass'] - 1) > 1e-6: return None
    return r


def objective(x, K):
    r = evaluate(x, K)
    if r is None: return -np.inf, None
    dist = 1 - r['law_term']; exc = r['R'] - r['law_term']
    if dist < 0.1: return -10 - (0.1 - dist), r
    if dist > 0.3: return -10 - (dist - 0.3), r
    return exc / dist, r


def chain(args):
    seed_name, x0, K, step0, n_it, seed = args
    rng = np.random.default_rng(seed); x = np.array(x0); fx, _ = objective(x, K); hist = [fx]; step = step0; acc = 0
    for it in range(n_it):
        y = x + rng.normal(0, step, x.size); fy, _ = objective(y, K)
        if fy > fx: x, fx = y, fy; acc += 1
        hist.append(fx)
        if (it + 1) % 50 == 0:                                   # simple step adaptation
            step = step * (1.5 if acc > 10 else 0.7); acc = 0
    ws, mus, Ss, sig = unpack(x, K)
    r10 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r20 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=20)
    dist = 1 - r10['law_term']; exc = r10['R'] - r10['law_term']
    keep = ['R', 'law_term', 'D', 'f', 'H_nonPD_mass', 'Pv', 'Pw', 'B_prime', 'abar_var', 'omega_rms', 'u_cv', 'cov_cs_bound', 'mass', 'lam_rho_direct']
    out = {'seed': seed_name, 'K': K, 'step0': step0, 'start_value': hist[0], 'best_value': fx, 'history': hist[::10],
           'sigma': sig, 'dist': dist, 'excess': exc, 'ratio': exc / dist, 'cov': r10['f'] - r10['law_term'], 'R_per_sd20': r20['R'],
           'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss]}
    out.update({k: float(r10[k]) for k in keep})
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(2028)
    best_d = max([r for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) if r.get('feasible')],
                 key=lambda r: r['ratio'])
    sA = (best_d['ws'], [np.array(m) for m in best_d['mus']], [np.array(S) for S in best_d['Ss']], best_d['sigma'])
    T = json.load(open('results/c1_table.json')); bt = max(T, key=lambda r: r['R'])
    src = next(c for c in json.load(open('results/c1_adversarial.json')) if c['K'] == bt['K'] and c['sigma'] == bt['sigma'] and c['target'] == bt['target'])
    wsB, musB, SsB = unpack_c1(np.array(src['x']), src['K']); sB = (wsB, musB, SsB, src['sigma'])
    jobs = []
    for name, st in [('A_ratio0.830', sA), ('B_R0.946', sB)]:
        K = len(st[0])
        for j, step in enumerate([0.01, 0.03, 0.1]):
            jobs.append((name, pack(*st), K, step, 250, 10 * j + (0 if name[0] == 'A' else 100)))
            st2 = split(*st, rng); jobs.append((name + '+split', pack(*st2), K + 1, step, 250, 10 * j + 5 + (0 if name[0] == 'A' else 100)))
    with Pool(2) as pool:
        out = pool.map(chain, jobs, chunksize=1)
    json.dump(out, open('results/e1_local.json', 'w'), indent=1, default=float)
    for r in out:
        print(f"{r['seed']:<18} K={r['K']} step={r['step0']:<5} start {r['start_value']:+.4f} -> best {r['best_value']:+.4f} | per_sd10: ratio {r['ratio']:.4f} "
              f"dist {r['dist']:.3f} R {r['R']:.4f} sigma {r['sigma']:.4g} valley {r['H_nonPD_mass']:.3f} (sd20 diff {r['R_per_sd20'] - r['R']:.1e})")
