"""Adversarial search against the observed relation R <= lam_comp = max_k lambda_max(Sym E_k[Df]) for two-component
mixtures (pendulum); R <= lam_rho is already violated by random states (mixtures.py). The grid is capped at 601
points per axis during the search; the best state is re-evaluated at full and at 1.5x resolution. Hill climbing with random restarts over weights, means and covariances
(Cholesky factors, log-scale), at fixed sigma. usage: python adversarial.py [ratio] [sigma ...] (target 'comp': maximise R - lam_comp; 'ratio': maximise
R / lam_comp over states with lam_comp >= 0.2) -> results/adversarial_<sigmas>.json."""
import numpy as np, json, os, sys
from obs_eval import evaluate
from mixtures import references

def unpack(x):
    w = 1 / (1 + np.exp(-x[0])); ws = [w, 1 - w]
    mus = [np.array([x[1], x[2]]), np.array([x[3], x[4]])]
    Ss = []
    for k in range(2):
        a, b, c = x[5 + 3 * k: 8 + 3 * k]
        L = np.array([[np.exp(a), 0], [b, np.exp(c)]])
        Ss.append(L @ L.T + 1e-6 * np.eye(2))
    return ws, mus, Ss

def objective(x, sig, target, n_cap=601):
    ws, mus, Ss = unpack(x)
    if min(np.linalg.eigvalsh(S).min() for S in Ss) < 1e-3 or max(np.linalg.eigvalsh(S).max() for S in Ss) > 4:
        return -np.inf, None
    from obs_eval import mixture_grid
    n = min(mixture_grid(ws, mus, Ss, sig)[3], n_cap)
    r = evaluate(ws, mus, Ss, sig, n=n)
    lam_rho, lam_comp, R_sep, ov = references(ws, mus, Ss, sig)
    info_ = None
    if target == 'ratio':                      # R / lam_comp, restricted to non-trivial states lam_comp >= 0.2
        if lam_comp < 0.2:
            return -np.inf, None
        return r['R'] / lam_comp, {'R': r['R'], 'lam_comp': lam_comp, 'lam_rho': lam_rho, 'R_sep': R_sep, 'overlap': ov}
    ref = lam_comp if target == 'comp' else lam_rho
    return r['R'] - ref, {'R': r['R'], 'lam_comp': lam_comp, 'lam_rho': lam_rho, 'R_sep': R_sep, 'overlap': ov}

def search(sig, target, restarts=4, iters=45, seed=0):
    rng = np.random.default_rng(seed); best = (-np.inf, None, None)
    for rs in range(restarts):
        x = np.concatenate([[rng.normal()], [rng.uniform(0, 2 * np.pi), rng.normal(0, 1)], [rng.uniform(0, 2 * np.pi), rng.normal(0, 1)],
                            rng.normal(-1.5, 0.7, 6) * np.array([1, 0.3, 1, 1, 0.3, 1])])
        fx, info = objective(x, sig, target); step = 0.5
        for it in range(iters):
            y = x + rng.normal(0, step, x.size)
            fy, iy = objective(y, sig, target)
            if fy > fx: x, fx, info = y, fy, iy
            else: step *= 0.97
        if fx > best[0]: best = (fx, x, info)
        print(f'  sigma={sig} target={target} restart {rs}: margin R-ref = {fx:+.4f}  {info}', flush=True)
    # re-evaluate the best state at full resolution (adaptive grid, no cap) and at 1.5x that resolution
    fx_full, info_full = objective(best[1], sig, target, n_cap=10 ** 9)
    ws, mus, Ss = unpack(best[1])
    from obs_eval import mixture_grid
    r15 = evaluate(ws, mus, Ss, sig, n=int(1.5 * mixture_grid(ws, mus, Ss, sig)[3]))
    return (fx_full, best[1].tolist(), info_full, {'R_capped_grid': best[2]['R'], 'R_n1.5': r15['R']})

if __name__ == '__main__':
    args = sys.argv[1:]
    targets = ['ratio'] if args and args[0] == 'ratio' else ['comp']
    if args and args[0] == 'ratio': args = args[1:]
    sigs = [float(a) for a in args] or [0.1, 0.3, 1.0]
    out = {}
    for sig in sigs:
        for target in targets:
            b = search(sig, target, seed=int(sig * 100) + (0 if target == 'comp' else 7))
            out[f'{target}_{sig}'] = {'best_margin': b[0], 'x': b[1], 'info': b[2], 'resolution_check': b[3]}
            print(f'sigma={sig} target={target}: best R - ref = {b[0]:+.4f}', flush=True)
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/adversarial_' + targets[0] + '_' + '_'.join(str(x) for x in sigs) + '.json', 'w'), indent=1, default=float)
