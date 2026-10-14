"""Stage G: the best single Gaussian at a fixed window sigma, R*(sigma) = sup over Gaussian states of law 1
R = tr(F Sym Jbar)/tr F (pendulum), and the test "no mixture beats the best single Gaussian at the same sigma": R <= R*(sigma).
R*(sigma) is found by maximising law 1 over the centre (q), the two widths and the tilt (Nelder-Mead from several starts).
Writes results/g3_best_single.json."""
import numpy as np, json
from scipy.optimize import minimize
from obs_eval2 import V, W

Rm = np.stack([V, W], 1)


def law1_params(x, sig):
    q, lsv, lsw, th = x
    lsv, lsw = np.clip(lsv, np.log(1e-9), np.log(1e3)), np.clip(lsw, np.log(1e-9), np.log(1e3))
    c, s = np.cos(th), np.sin(th); U = Rm @ np.array([[c, -s], [s, c]])
    S = U @ np.diag([np.exp(2 * lsv), np.exp(2 * lsw)]) @ U.T
    J = np.array([[0, 1.0], [-np.cos(q) * np.exp(-S[0, 0] / 2), 0]])
    try: F = np.linalg.inv(S + sig ** 2 * np.eye(2))
    except np.linalg.LinAlgError: return -np.inf
    if not np.all(np.isfinite(F)): return -np.inf
    return np.trace(F @ (J + J.T) / 2) / np.trace(F)


def Rstar(sig, full=False):
    """2-D search over the widths with the centre at the top and the axes along V, W (full=True: also centre and tilt free)."""
    best = None
    for lsv0, lsw0 in [(np.log(sig / 2), 0.5 * np.log(4 * sig)), (np.log(sig / 10), 0.5 * np.log(sig)), (np.log(sig), 0.5 * np.log(10 * sig)), (np.log(sig / 10), np.log(10.0)), (np.log(sig / 10), np.log(100.0))]:
        if full:
            f = lambda x: -law1_params(x, sig); x0 = [np.pi, lsv0, lsw0, 0.0]
        else:
            f = lambda x: -law1_params([np.pi, x[0], x[1], 0.0], sig); x0 = [lsv0, lsw0]
        r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 8000, 'maxfev': 8000})
        if best is None or r.fun < best.fun: best = r
    return float(-best.fun), list(map(float, best.x))


if __name__ == '__main__':
    from f6_components import Rk_max
    rows = []
    for fn in ['results/g2_search.json', 'results/g2_map.json', 'results/g2_map2.json', 'results/g2_scale.json', 'results/g2_shift.json', 'results/g1_families.json']:
        for r in json.load(open(fn)): rows.append((fn.split('/')[-1], r['R'], r['sigma']))
    for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) + json.load(open('results/e2_dense.json')):
        if r.get('feasible'): rows.append(('near-top', r['R'], r['sigma']))
    for r in json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')): rows.append(('e1', r['R'], r['sigma']))
    for r in json.load(open('results/f1_chains.json')) + json.load(open('results/f4_rcvo.json')): rows.append(('f', r['final']['R'], r['sigma']))
    for r in json.load(open('results/c1_adversarial.json')) + json.load(open('results/c1_adversarial_room.json')): rows.append(('c1', r['R'], r['sigma']))
    for m, r in zip(json.load(open('results/mixtures.json')), json.load(open('results/m2_mix.json'))): rows.append(('random', r['R'], m['sigma']))
    for r in json.load(open('results/f5_needle.json'))['rows']: rows.append(('f5-needle', r['R'], r['sigma']))
    cache = {}
    out = []
    for tag, R, s in rows:
        key = float('%.12g' % s)
        if key not in cache: cache[key] = Rstar(s, full=True)
        out.append({'tag': tag, 'R': R, 'sigma': s, 'Rstar': cache[key][0], 'margin': cache[key][0] - R, 'gstar': (R - cache[key][0]) / (1 - cache[key][0])})
    res = {'n': len(out), 'n_R_gt_Rstar': sum(o['margin'] < -1e-9 for o in out), 'min_margin': min(o['margin'] for o in out),
           'max_gstar': max(o['gstar'] for o in out), 'Rstar_table': {str(k): v for k, v in sorted(cache.items())}, 'rows': out}
    json.dump(res, open('results/g3_best_single.json', 'w'), indent=1)
    print({k: v for k, v in res.items() if k not in ['rows', 'Rstar_table']})
    for o in sorted(out, key=lambda o: o['margin'])[:8]: print(o)
