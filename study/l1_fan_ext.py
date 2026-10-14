"""Stage L, task L1(a), families 2-4: continuous fans (stage I functional family), the fan extended by a position shift along W
as a function of the angle (3 Chebyshev terms) and a rigid shift of the whole fan off the hyperbolic point (dv in units of sigma,
dw in units of 2 sqrt(sigma)), minimising (1 - R)/sigma (pendulum).
  'fan'      : the 17-parameter family of i1_fan, re-optimised for the minimum of (1 - R)/sigma at sigma = 1e-3 and 1e-2
  'ext'      : 22 parameters (fan + W-offset function + rigid shift), started from the best fan
  'shiftscan': the best extended fan with its rigid shift moved by dv = -2..2 (sigma units) or dw = -0.5..0.5 (2 sqrt(sigma) units)
Search: adaptive random hill climbing (n = 20 segments, per_sd 3: agrees with per_sd 6 to 1e-10 on the stage-I fan) followed by Nelder-Mead; final states re-evaluated with
n = 40, a grid of >= 401 points per axis, the other sigma, and diagnosed with l_tools.diag.  -> results/l1_fan_ext.json, results/l1_shiftscan.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from numpy.polynomial import chebyshev as Ch
from scipy.optimize import minimize
from i1_fan import fan as fan17
from i_tools import fields, V, W
from l_tools import diag

TOP = np.array([np.pi, 0.0])


def fan22(x, n, sig):
    ws, mus, Ss = fan17(x[:17], n, sig)
    lT = x[0]; xs = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    lw = Ch.chebval(xs, np.concatenate([[0.0], x[1:7]])); w = np.exp(lw - lw.max()); w = w / w.sum(); keep = w > 1e-6; xs = xs[keep]
    offw = 2 * np.sqrt(sig) * Ch.chebval(xs, x[17:20]); shift = x[20] * sig * V + x[21] * 2 * np.sqrt(sig) * W
    mus = [m + ow * W + shift for m, ow in zip(mus, offw)]
    return ws, mus, Ss


def gap(x, sig, n=20, per_sd=3):
    if np.any(np.abs(x) > 12): return np.inf
    ws, mus, Ss = fan22(x, n, sig) if len(x) == 22 else fan17(x, n, sig)
    try: r = fields(ws, mus, Ss, sig, per_sd=per_sd)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return np.inf
    return (1 - r['R']) / sig if abs(r['mass'] - 1) < 1e-6 else np.inf


def optimise(x, sig, steps, seed):
    rng = np.random.default_rng(seed); fx = gap(x, sig); step = 0.05; acc = 0
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.35); fy = gap(y, sig)
        if fy < fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.4)); acc = 0
    r = minimize(lambda z: gap(z, sig), x, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-9})
    return (r.x, r.fun) if r.fun < fx else (x, fx)


def job(a):
    kind, sig, seed, steps = a
    i1 = json.load(open('results/i1_fan.json')); x0 = np.array(max(i1, key=lambda r: r['checks']['n40'])['x'])
    if kind == 'ext': x0 = np.concatenate([x0, np.zeros(5)])
    x, fx = optimise(x0, sig, steps, seed)
    other = 1e-2 if sig == 1e-3 else 1e-3
    ws, mus, Ss = fan22(x, 40, sig) if kind == 'ext' else fan17(x, 40, sig)
    d = diag(ws, mus, Ss, sig); d10 = diag(ws, mus, Ss, sig, min_axis=401)
    wo, mo, So = fan22(x, 40, other) if kind == 'ext' else fan17(x, 40, other)
    return {'kind': kind, 'sigma': sig, 'gap_search_n20': fx, 'gap_n40': d['gap_over_sigma'], 'gap_n40_fine_grid': d10['gap_over_sigma'],
            'gap_other_sigma': diag(wo, mo, So, other)['gap_over_sigma'], 'diag': d, 'x': list(x)}


def scan(args):
    x, sig, which, val = args; x = np.array(x).copy(); x[20 if which == 'v' else 21] += val
    ws, mus, Ss = fan22(x, 40, sig); d = diag(ws, mus, Ss, sig)
    return {'sigma': sig, 'which': which, 'shift': val, 'gap': d['gap_over_sigma'], 'ThetaP': d['ThetaP'], 'rho': d['rho']}


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'opt'
    if part == 'opt':
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        jobs = [('fan', 1e-3, 51, steps), ('fan', 1e-2, 52, steps), ('ext', 1e-3, 53, steps), ('ext', 1e-2, 54, steps)]
        out = []
        with Pool(2) as pool:
            for r in pool.imap_unordered(job, jobs):
                out.append(r); json.dump(out, open('results/l1_fan_ext.json', 'w'), indent=1, default=float)
                print(f"{r['kind']} sigma {r['sigma']:g}: (1-R)/sigma n20 {r['gap_search_n20']:.5f}, n40 {r['gap_n40']:.5f} (fine grid {r['gap_n40_fine_grid']:.5f}), "
                      f"other sigma {r['gap_other_sigma']:.5f}; Theta' {r['diag']['ThetaP']:.4f} rho {r['diag']['rho']:.4f}", flush=True)
    else:
        E = [r for r in json.load(open('results/l1_fan_ext.json')) if r['kind'] == 'ext']
        jobs = [(r['x'], r['sigma'], wh, v) for r in E for wh, vals in [('v', np.linspace(-2, 2, 9)), ('w', np.linspace(-0.5, 0.5, 9))] for v in vals]
        with Pool(2) as pool: out = pool.map(scan, jobs)
        json.dump(out, open('results/l1_shiftscan.json', 'w'), indent=1, default=float)
        for o in out: print(o)
