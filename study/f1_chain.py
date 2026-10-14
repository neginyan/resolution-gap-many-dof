"""Stage F, tasks 1-2: long local searches of the fraction excess / (kappa - law) with the distance band widened to 0.1-0.6
(pendulum, kappa = 1), started from the best states of stage E.  Along every chain the factors of the exact relation
    fraction = [rho * CV * Omega * (1 - lambda_rho) + D] / distance,   law gap = lambda_rho - law
are recorded every 20 steps (state accepted so far).  Grid aligned with V, W (per_sd = 6 during the search); final states
re-evaluated at per_sd = 10 and 20.  Usage: python f1_chain.py [n_steps] [lo] [hi] [out.json]."""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from e1_local import pack, unpack, split

LO, HI = 0.1, 0.6


def evaluate(x, K, per_sd=6):
    ws, mus, Ss, sig = unpack(x, K)
    ev = [np.linalg.eigvalsh(S) for S in Ss]
    if min(ws) < 0.01 or max(e.max() for e in ev) > 1.0 or min(e.min() for e in ev) < 1e-10 or not (1e-4 <= sig <= 1): return None
    sds = [np.sqrt(e.max()) for e in ev]
    if max(sds) / min(sds) > 40: return None
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=per_sd)
    if abs(r['mass'] - 1) > 1e-6: return None
    return r


def factors(r):
    dist = 1 - r['law_term']; lr = r['lam_rho_direct']
    rho = r['f_cov_part'] / r['cov_cs_bound'] if r['cov_cs_bound'] > 0 else 0.0
    return {'ratio': (r['R'] - r['law_term']) / dist, 'dist': dist, 'R': r['R'], 'rho': rho, 'CV': r['u_cv'], 'Omega': r['omega_rms'],
            'rhoCVOmega': rho * r['u_cv'] * r['omega_rms'], 'law_gap': lr - r['law_term'], 'one_minus_lam': 1 - lr, 'D': r['D'],
            'valley': r['H_nonPD_mass'], 'Pv': r['Pv'], 'Pw': r['Pw']}


def objective(x, K, lo, hi):
    r = evaluate(x, K)
    if r is None: return -np.inf, None
    dist = 1 - r['law_term']
    if dist < lo: return -10 - (lo - dist), r
    if dist > hi: return -10 - (dist - hi), r
    return (r['R'] - r['law_term']) / dist, r


def chain(args):
    name, x0, K, step0, n_it, seed, lo, hi = args
    rng = np.random.default_rng(seed); x = np.array(x0); fx, rx = objective(x, K, lo, hi)
    track = [dict(step=0, **factors(rx))]; step = step0; acc = 0
    for it in range(1, n_it + 1):
        y = x + rng.normal(0, step, x.size); fy, ry = objective(y, K, lo, hi)
        if fy > fx: x, fx, rx = y, fy, ry; acc += 1
        if it % 50 == 0: step = float(np.clip(step * (1.5 if acc > 10 else 0.7), 1e-4, 0.3)); acc = 0
        if it % 20 == 0: track.append(dict(step=it, **factors(rx)))
    ws, mus, Ss, sig = unpack(x, K)
    r10 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); r20 = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=20)
    out = {'name': name, 'K': K, 'step0': step0, 'n_steps': n_it, 'band': [lo, hi], 'best_value': fx, 'track': track,
           'final': factors(r10), 'R_per_sd20': float(r20['R']), 'R_per_sd10': float(r10['R']), 'sigma': sig, 'x': list(map(float, x)),
           'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss], 'mass': float(r10['mass'])}
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else LO; hi = float(sys.argv[3]) if len(sys.argv) > 3 else HI
    fn = sys.argv[4] if len(sys.argv) > 4 else 'results/f1_chains.json'
    rng = np.random.default_rng(2029)
    E = sorted(json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')), key=lambda r: -r['ratio'])
    st = lambda r: (r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']], r['sigma'])
    b0, b1, b2 = E[0], E[1], E[2]
    jobs = [('best_K3_step0.01', pack(*st(b0)), 3, 0.01, n, 1, lo, hi), ('best_K3_step0.03', pack(*st(b0)), 3, 0.03, n, 2, lo, hi),
            ('best_K3_step0.1', pack(*st(b0)), 3, 0.1, n, 3, lo, hi), ('best_split_K4_step0.03', pack(*split(*st(b0), rng)), 4, 0.03, n, 4, lo, hi),
            (f"second_K{len(b1['ws'])}_step0.03", pack(*st(b1)), len(b1['ws']), 0.03, n, 5, lo, hi),
            (f"third_K{len(b2['ws'])}_step0.03", pack(*st(b2)), len(b2['ws']), 0.03, n, 6, lo, hi)]
    with Pool(2) as pool:
        out = pool.map(chain, jobs, chunksize=1)
    json.dump(out, open(fn, 'w'), indent=1, default=float)
    for r in out:
        f = r['final']
        print(f"{r['name']:<24} best {r['best_value']:.5f} | ratio {f['ratio']:.5f} dist {f['dist']:.4f} R {f['R']:.5f} rho {f['rho']:.3f} CV {f['CV']:.3f} "
              f"Omega {f['Omega']:.3f} rhoCVO {f['rhoCVOmega']:.4f} gap {f['law_gap']:.4f} D {f['D']:.1e} sigma {r['sigma']:.4g} (sd20 diff {r['R_per_sd20'] - r['R_per_sd10']:.1e})")
