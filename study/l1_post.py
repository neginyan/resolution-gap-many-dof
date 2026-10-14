"""Stage L, task L1(a): final diagnostics of every state found by the L1 searches (l1_free, l1_polish, l1_fan_ext, l1_rho_max).
For each state: l_tools.diag on the standard grid (per_sd 6, >= 161 points per axis) and on a fine grid (per_sd 10, >= 401 points),
the same state at the other sigma (scale invariance), and the gap relative to the best single Gaussian, (1 - R)/(1 - R*(sigma))
(which removes the common O(sigma) drift 1 - 7/8 sigma of the single Gaussian).  For the worst state of each sigma, R is also
computed with the independent evaluators of verification/indep_eval.py (R_indep: Gauss-Hermite posterior; R_brute: direct kernel
summation, sigma = 1e-2 only).   -> results/l1_post.json"""
import numpy as np, json, glob, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verification'))
from h2_adversarial import unpack
from h1_rstar import rstar
from l_tools import diag

RS = {s: rstar(s)['Rstar'] for s in [1e-3, 1e-2]}


def states():
    out = []
    for f in sorted(glob.glob('results/l1_free_s*_sigma*.json')):
        run = os.path.basename(f).split('_sigma')[0]
        for r in json.load(open(f)): out.append({'family': 'free (hill climb)', 'run': run, 'K': r['K'], 'sigma': r['sigma'], 'x': r['x']})
    for f in sorted(glob.glob('results/l1_polish_sigma*.json')):
        for r in json.load(open(f)): out.append({'family': 'free (L-BFGS polish)', 'run': 'l1_polish', 'K': r['K'], 'sigma': r['sigma'], 'x': r['x']})
    if os.path.exists('results/l1_fan_ext.json'):
        for r in json.load(open('results/l1_fan_ext.json')): out.append({'family': r['kind'], 'run': 'l1_fan_ext', 'K': 40, 'sigma': r['sigma'], 'x': r['x']})
    return out


def build(st, sig):
    x = np.array(st['x'])
    if st['family'] in ('fan', 'ext'):
        from l1_fan_ext import fan22
        from i1_fan import fan as fan17
        return fan22(x, 40, sig) if st['family'] == 'ext' else fan17(x, 40, sig)
    return unpack(x, st['K'], sig)


def run(st):
    sig = st['sigma']; other = 1e-2 if sig == 1e-3 else 1e-3
    d = diag(*build(st, sig), sig); dfine = diag(*build(st, sig), sig, per_sd=10, min_axis=401); dother = diag(*build(st, other), other)
    return {**{k: st[k] for k in ['family', 'run', 'K', 'sigma']}, 'diag': d, 'gap_fine': dfine['gap_over_sigma'], 'diag_fine': dfine,
            'gap_other_sigma': dother['gap_over_sigma'], 'diag_other': dother,
            'gap_rel_Rstar': (1 - d['R']) / (1 - RS[sig]), 'gap_rel_Rstar_other': (1 - dother['R']) / (1 - RS[other]), 'x': st['x']}


if __name__ == '__main__':
    S = states(); print(len(S), 'states', flush=True)
    with Pool(2) as pool: out = pool.map(run, S)
    from indep_eval import R_indep, R_brute
    for s in [1e-3, 1e-2]:
        b = min([o for o in out if o['sigma'] == s], key=lambda o: o['diag']['gap_over_sigma']); st = [x for x in S if x['x'] == b['x']][0]
        ws, mus, Ss = build(st, s); Ri, mass = R_indep(ws, mus, Ss, s, 500, 500, gh=16)
        b['indep'] = {'R_indep': Ri, 'gap_indep': (1 - Ri) / s, 'mass': mass}
        if s == 1e-2: Rb = R_brute(ws, mus, Ss, s, ny=181); b['indep'].update({'R_brute': Rb, 'gap_brute': (1 - Rb) / s})
        print(s, b['family'], b['K'], b['diag']['gap_over_sigma'], b['indep'], flush=True)
    json.dump(out, open('results/l1_post.json', 'w'), indent=1, default=float)
    for o in sorted(out, key=lambda o: (o['sigma'], o['diag']['gap_over_sigma'])):
        d = o['diag']; print(f"{o['sigma']:g} {o['family']:22s} {o['run']:12s} K={o['K']:2d}  gap {d['gap_over_sigma']:.5f} fine {o['gap_fine']:.5f} other-sigma {o['gap_other_sigma']:.5f}  "
                             f"rel R* {o['gap_rel_Rstar']:.5f}/{o['gap_rel_Rstar_other']:.5f}  Theta' {d['ThetaP']:.4f} eps~ {d['eps_tilde_over_sigma']:.4f} D {d['D_over_sigma']:.5f} rho {d['rho']:.4f}")
