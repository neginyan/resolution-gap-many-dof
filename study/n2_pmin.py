"""Stage N, task N2(a): minimise P = E[w u^2] J_ww / j^2 directly (pendulum; free mixtures, h2_adversarial parametrisation),
sigma = 1e-3 and 1e-2, K = 8, 16, 24.  NOT the gap: the question is whether the "uncertainty product" P (focus spread x W-sharpness,
P = d^2 y/sigma) has a positive infimum, and whether gap >~ sqrt(P) survives at the P-minimising states.
  w = sigma^2 (H_vv)_+ clipped to [0, 1], u = sqrt2 (m_q - pi), j = E[w], J_ww = E[H_ww]  (j2_smooth.arrays, per_sd 3 in the search, 8 final).
Starts: the stage-L polished minimum-gap states (K = 16, 24; for K = 8 the 8 heaviest components of the K = 16 state) and the
stage-H best K = 8 state.  Procedure: adaptive random hill climbing (`steps`) then L-BFGS-B (forward differences, `maxiter`).
Checkpoints in results/n2_ckpt/.   Final: P, the gap, Theta', rho, d, y/sigma on the standard grid and on a finer grid.
Restricted variant (4th argument Y): penalty 100 (y/sigma - Y)_+^2, i.e. P minimised among states with y/sigma <= Y.
(The unrestricted search drives P to ~1e-4 at y/sigma ~ 100, where the gap is ~100: P alone has no positive infimum.)
Usage: python n2_pmin.py <steps> <maxiter> [K list] [Y]   -> results/n2_pmin.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import minimize
from h2_adversarial import unpack
from j2_smooth import arrays
from l_tools import diag


def pvals(x, K, sig, per_sd=3):
    ws, mus, Ss = unpack(x, K, sig)
    A = arrays(ws, mus, Ss, sig, per_sd=per_sd, max_points=int(min(2_500_000, 2.4e7 / K))); p = A['pw']
    w, u = A['w'], A['u']; j = np.sum(p * w); Z = np.sum(p * w * u * u); Jww = np.sum(p * A['Hww'])
    return {'P': float(Z * Jww / j ** 2), 'gap': float((1 - A['R']) / sig), 'd': float(np.sqrt(Z / (j * sig))), 'y_over_sigma': float(sig * Jww / j), 'j': float(j)}


YMAX = None                                    # restricted search: penalty for y/sigma > YMAX (set from the command line)


def obj(x, K, sig):
    sh = x[3 * K:].reshape(K, 3)
    if np.any(np.abs(x[K:3 * K]) > 30) or np.any(sh[:, [0, 2]] > 3) or np.any(sh[:, [0, 2]] < -12) or np.any(np.abs(sh[:, 1]) > 8): return 1e6
    try: r = pvals(x, K, sig); v = r['P']
    except (MemoryError, ValueError, np.linalg.LinAlgError, ZeroDivisionError): return 1e6
    if YMAX is not None: v += 100 * max(0.0, r['y_over_sigma'] - YMAX) ** 2
    return v if np.isfinite(v) and v > 0 else 1e6


def starts(K, sig):
    out = []
    for f in [f'results/l1_polish_sigma{sig:g}.json', 'results/l1_polish_sigma0.001.json']:
        if os.path.exists(f):
            for r in json.load(open(f)):
                x = np.array(r['x']); Kr = r['K']
                if Kr == K: out.append(('l1_polish', x))
                elif Kr > K:
                    o = np.argsort(x[:Kr])[::-1][:K]; out.append((f'l1_polish_K{Kr}_heaviest', np.concatenate([x[:Kr][o], x[Kr:3 * Kr].reshape(Kr, 2)[o].ravel(), x[3 * Kr:].reshape(Kr, 3)[o].ravel()])))
            if out: break
    if K == 8:
        c = json.load(open('results/h4_growk.json')); b = max([x for ch in c for x in ch['chain'] if x['K'] == 8], key=lambda x: x['gstar10']); out.append(('h4_growk', np.array(b['x'])))
    return out[:2]


def job(a):
    K, sig, steps, maxiter = a; tag = f'K{K}_s{sig:g}' + ('' if YMAX is None else f'_y{YMAX:g}'); ck = f'results/n2_ckpt/{tag}.json'; os.makedirs('results/n2_ckpt', exist_ok=True)
    f = lambda z: obj(z, K, sig)
    if os.path.exists(ck): st = json.load(open(ck)); x = np.array(st['x']); stage = st['stage']; done = st.get('it', 0); src = st['src']; P0 = st['P0']
    else:
        cands = starts(K, sig); vals = [f(x) for _, x in cands]; i = int(np.argmin(vals)); src, x = cands[i][0], cands[i][1]; P0 = vals[i]; stage = 'climb'; done = 0
    if stage == 'climb':
        rng = np.random.default_rng(7 + K); fx = f(x); step = 0.1; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = f(y)
            if fy < fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.5)); acc = 0
        stage = 'lbfgs'; json.dump({'x': x.tolist(), 'stage': stage, 'it': 0, 'src': src, 'P0': P0}, open(ck, 'w'))
    while done < maxiter:
        n = min(25, maxiter - done)
        r = minimize(f, x, method='L-BFGS-B', options={'maxiter': n, 'maxfun': 10 ** 7, 'eps': 1e-5, 'ftol': 1e-14, 'gtol': 1e-10})
        x = r.x; done += n; json.dump({'x': x.tolist(), 'stage': 'lbfgs', 'it': done, 'src': src, 'P0': P0}, open(ck, 'w'))
        if r.nit < n: break
    v = pvals(x, K, sig); vf = pvals(x, K, sig, per_sd=8); d = diag(*unpack(x, K, sig), sig)
    return {'K': K, 'sigma': sig, 'ymax': YMAX, 'start': src, 'P_start': P0, **v, 'P_fine': vf['P'], 'gap_fine': vf['gap'], 'ThetaP': d['ThetaP'], 'rho': d['rho'],
            'eps_tilde_over_sigma': d['eps_tilde_over_sigma'], 'D_over_sigma': d['D_over_sigma'], 'bound_over_sigma': d['bound_over_sigma'],
            'gap_over_sqrtP': v['gap'] / np.sqrt(v['P']), 'iterations': done, 'x': x.tolist()}


if __name__ == '__main__':
    steps, maxiter = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else '8,16,24').split(',')]
    YMAX = float(sys.argv[4]) if len(sys.argv) > 4 else None
    fn = 'results/n2_pmin.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['K'], o['sigma'], o.get('ymax')) for o in out}
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [(K, s, steps, maxiter) for K in Ks for s in [1e-3, 1e-2] if (K, s, YMAX) not in done]):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float)
            print(f"[y/sigma <= {YMAX}] K={r['K']} sigma {r['sigma']:g} (from {r['start']}): P {r['P_start']:.4f} -> {r['P']:.4f} (fine {r['P_fine']:.4f});  gap {r['gap']:.4f}, "
                  f"gap/sqrtP {r['gap_over_sqrtP']:.4f}, Theta' {r['ThetaP']:.4f}, rho {r['rho']:.3f}, d {r['d']:.3f}, y/sigma {r['y_over_sigma']:.3f}", flush=True)
