"""Stage L, task L2 (side check): the continuum limit of the stage-I bent needle.
Stage I optimised the arc of i5_arc at n = 24 segments and reported 0.509 sigma at n = 48; at n = 96 the same shape gives 0.523 sigma
(stage I noted the drop of g* at 96 segments).  A discrete chain of segments is itself a mixture, so the question here is what the
CONTINUOUS curved needle reaches.  The six arc parameters are re-optimised (Nelder-Mead, Gauss-Hermite 3 nodes, per_sd 5) at
n = 64 and the optimum evaluated at n = 64, 128, 192 (Gauss-Hermite 6, per_sd 8) and at sigma = 1e-4 (scale invariance); the n = 24
optimum of stage I is evaluated at the same n for comparison.   -> results/l2_arc_continuum.json"""
import numpy as np, json
from scipy.optimize import minimize
from i5_arc import arc
from i_tools import fields

SIG = 1e-3


def gap(x, n, sig=SIG, gh=3, per_sd=5):
    x = np.asarray(x)
    if abs(x[2]) > 50 or not (-3 < x[3] < 3) or not (-12 < x[4] < 1) or not (-3 < x[5] < 3): return np.inf
    try: r = fields(*arc(x, n, sig), sig, kind='twist', per_sd=per_sd, gh=gh)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return np.inf
    return (0.25 - r['R']) / sig if abs(r['mass'] - 1) < 1e-6 else np.inf


if __name__ == '__main__':
    b5 = max(json.load(open('results/i5_arc.json')), key=lambda o: o['gstar_n48']); x5 = np.array(b5['x'])
    out = {'stageI_x': x5.tolist(), 'stageI': {str(n): gap(x5, n, gh=6, per_sd=8) for n in [24, 48, 64, 128, 192]}}
    print('stage-I arc:', out['stageI'], flush=True)
    r = minimize(lambda z: gap(z, 64), x5, method='Nelder-Mead', options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-7})
    x = r.x; out['opt_x_n64'] = x.tolist(); out['opt_search_n64'] = float(r.fun)
    out['opt_n64'] = {str(n): gap(x, n, gh=6, per_sd=8) for n in [64, 128, 192]}
    out['opt_n64_sigma1e-4'] = gap(x, 128, sig=1e-4, gh=6, per_sd=8)
    out['opt_curvature'] = float(x[2]); out['turning_rate'] = float(1 / np.sqrt(2))
    print('re-optimised at n = 64:', out['opt_n64'], 'curvature', x[2], flush=True)
    json.dump(out, open('results/l2_arc_continuum.json', 'w'), indent=1, default=float)


def more(ns=(384, 768)):
    """continuation: both shapes at larger n (pieces shorter than sigma), Gauss-Hermite 4, per_sd 6.  Appends to the json."""
    out = json.load(open('results/l2_arc_continuum.json'))
    for n in ns:
        out['stageI'][str(n)] = gap(out['stageI_x'], n, gh=4, per_sd=6); out['opt_n64'][str(n)] = gap(out['opt_x_n64'], n, gh=4, per_sd=6)
        print(n, out['stageI'][str(n)], out['opt_n64'][str(n)], flush=True)
        json.dump(out, open('results/l2_arc_continuum.json', 'w'), indent=1, default=float)


def brute(ns=(24, 96), sig=1e-2):
    """independent check of the n-dependence: the stage-I arc with n segments at sigma = 1e-2 by the grid evaluator and by the direct
    kernel summation (l2_check.ns['R_brute'], twist flow).  Appends to the json under 'brute'."""
    from l2_check import ns as NS
    out = json.load(open('results/l2_arc_continuum.json')); out['brute'] = {}
    for n in ns:
        ws, mus, Ss = arc(np.array(out['stageI_x']), n, sig)
        g = gap(out['stageI_x'], n, sig=sig, gh=6, per_sd=8); b = (0.25 - NS['R_brute'](ws, mus, Ss, sig, ny=161)) / sig
        out['brute'][str(n)] = {'grid': g, 'brute': b}; print(n, g, b, flush=True)
    json.dump(out, open('results/l2_arc_continuum.json', 'w'), indent=1, default=float)
