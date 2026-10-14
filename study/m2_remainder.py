"""Measurement 2 (stage B): the remainder R - e.

Exact split (obs_eval2.py):  R - e = (f - e) + D,
  f - e = E[s . div G] / tr F,  G = Sym E[Df|Y]   ("stretch-gradient" term: how fast the local stretch changes across the window)
  D     = R - f = posterior Stein defect  (zero when the posterior X|Y is Gaussian, e.g. a single Gaussian state, or f linear)
Scale candidates: K_sin * sigma^2, K_cos * sigma^2 (K = E_{p_Y}|a'|, E_{p_Y}|a''|), and the spread of the local Fisher
matrix H (std of log eigen-ratio, non-positive-definite mass, std of the V/W anisotropy alpha).
Part 'single': all single-Gaussian states of stage A (R, e from results/sweep*.json; K by 1-D quadrature; H is constant, so
the H-spread candidates are exactly 0).  Part 'mix': the 1000 random two-component states of results/mixtures.json,
re-evaluated with evaluate_full.  Writes results/m2_single.json, results/m2_mix.json.
"""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full


def K_single(mu_q, var):
    x = np.linspace(-10, 10, 4001); w = np.exp(-x ** 2 / 2); w /= w.sum()
    q = mu_q + np.sqrt(var) * x
    return float(np.sum(w * np.abs(np.sin(q))) / 2), float(np.sum(w * np.abs(np.cos(q))) / 2)


def run_mix(m):
    r = evaluate_full(m['ws'], [np.array(x) for x in m['mus']], [np.array(S) for S in m['Ss']], m['sigma'])
    r = {k: float(v) for k, v in r.items()}
    r.update({'i': m['i'], 'overlap': m['overlap'], 'lam_rho': m['lam_rho']})
    return r


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'single'
    if part == 'single':
        D = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
        out = []
        for rec in D:
            S = np.array(rec['S'])
            for r in rec['rows']:
                ks, kc = K_single(rec['q0'], S[0, 0] + r['sigma'] ** 2)
                out.append({'shape': rec['shape'], 'q0': rec['q0'], 'sigma': r['sigma'], 'R': r['R'], 'e': r['cand_e'],
                            'K_sin': ks, 'K_cos': kc})
        json.dump(out, open('results/m2_single.json', 'w'), indent=1)
        print('single', len(out))
    else:
        M = json.load(open('results/mixtures.json'))
        # one 2401^2 evaluation needs ~3 GB: light states run two at a time, heavy states (n > 1500) one at a time
        light = [m for m in M if m['n'] <= 1500]; heavy = [m for m in M if m['n'] > 1500]
        with Pool(2) as pool:
            out = pool.map(run_mix, light, chunksize=4)
        out += [run_mix(m) for m in heavy]
        out.sort(key=lambda r: (r['i'], r['sigma']))
        json.dump(out, open('results/m2_mix.json', 'w'), indent=1)
        print('mix', len(out))
