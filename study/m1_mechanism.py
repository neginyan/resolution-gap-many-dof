"""Measurement 1 (stage B): take the counterexample R > lambda_comp apart.

Base family (simplified from the adversarial counterexample, which sat near the stable point with both components long
along the contracting direction W = (1,-1)/sqrt2):
  equal weights, centre c0 = (0, 0) (stable point), sigma = 0.1,
  component 1: width n1 = 0.09 along V (stretching direction), length l1 = 1.9 along W,
  component 2: width n2 = ratio * n1 along V, length l2 = 0.63 * l1 along W, optionally rotated by theta,
  means c0 -+ (delta/2) u, with u = W (default) or V.
One quantity is changed at a time; every record stores R, R_sep, lambda_comp, e, f, A, R_switch, overlap, ...
Writes results/m1_slices.json and results/m1_maps.json.
"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from mixtures import references

V = np.array([1.0, 1.0]) / np.sqrt(2); W = np.array([1.0, -1.0]) / np.sqrt(2)
N1, L1, RATIO, SIG = 0.09, 1.9, 0.4 / 0.09, 0.1


def shape(n, l, rot=0.0):
    c, s = np.cos(rot), np.sin(rot); Rm = np.array([[c, -s], [s, c]])
    v, w = Rm @ V, Rm @ W
    return n ** 2 * np.outer(v, v) + l ** 2 * np.outer(w, w)


def state(delta, ratio=RATIO, theta=0.0, l1=L1, sep='W', c0=(0.0, 0.0)):
    u = W if sep == 'W' else V
    c0 = np.array(c0)
    return [0.5, 0.5], [c0 - delta / 2 * u, c0 + delta / 2 * u], [shape(N1, l1), shape(ratio * N1, 0.63 * l1, theta)]


def run(job):
    tag, kw, sig = job
    ws, mus, Ss = state(**kw)
    r = evaluate_full(ws, mus, Ss, sig)
    lam_rho, lam_comp, R_sep, ov = references(ws, mus, Ss, sig)
    r = {k: float(v) for k, v in r.items()}
    r.update({'tag': tag, 'params': kw, 'lam_rho': float(lam_rho), 'overlap': float(ov), 'R_sep_ref': float(R_sep),
              'elongation1': kw.get('l1', L1) / N1})
    return r


DELTAS = list(np.linspace(0, 3.0, 16))                          # delta / sigma = 0 ... 30

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'slices'
    jobs = []
    if which == 'slices':
        for d in DELTAS:
            jobs.append(('base_sepW', {'delta': d}, SIG))
            jobs.append(('base_sepV', {'delta': d, 'sep': 'V'}, SIG))
            for ra in [1.0, 2.0, 8.0]:
                jobs.append((f'ratio_{ra}', {'delta': d, 'ratio': ra}, SIG))
            for th in [22.5, 45.0, 90.0]:
                jobs.append((f'tilt_{th}', {'delta': d, 'theta': np.radians(th)}, SIG))
    else:
        for d in DELTAS:
            for ra in np.logspace(0, 1, 12):
                jobs.append(('map_ratio', {'delta': d, 'ratio': float(ra)}, SIG))
            for l1 in np.logspace(np.log10(0.09), np.log10(2.7), 12):     # elongation l1/n1 = 1 ... 30
                jobs.append(('map_elong', {'delta': d, 'l1': float(l1)}, SIG))
    with Pool(2) as pool:
        out = pool.map(run, jobs, chunksize=1)
    os.makedirs('results', exist_ok=True)
    json.dump(out, open(f'results/m1_{which}.json', 'w'), indent=1)
    print(which, len(out), 'max R/lam_comp', max(o['R'] / o['lam_comp'] for o in out))
