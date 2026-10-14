"""Stage N, task N3: the "tangent bundle" (bent needle made of straight pieces of FIXED length) in the twist flow.
Model (i5_arc.arc): n pieces along an arc through (1 + dr sigma, 0); piece s.d. along its own tangent = l (the piece length), set
through c_len = l / spacing, spacing = 6 Ls/(n - 1); n is chosen so that spacing <= l/1.2 (smooth superposition, no beads).
Part 'ell' (twist flow, kappa = 1/4, sigma = 1e-3): for each l/sqrt(sigma) in ELLS the other five parameters (dr, tilt, curvature,
Ls, thickness) are optimised (Nelder-Mead from the stage-M optimum, Gauss-Hermite 3, per_sd 5), final value with Gauss-Hermite 4,
per_sd 6.  -> the curve gap(l), its minimum and its curvature there.
Part 'm' (generalised twist flows f = (p, -q)/(1 + r^(2m)), kappa = m/4 on r = 1, turning rate 1/sqrt2 for every m):
all six parameters (l free) optimised for m = 0.7, 1.4 (and 1), compared with the estimate
   gap/sigma -> 2 sqrt(2 beta_rad kappa) = m^2/2   (beta_rad = m^3/8: radial fall of the stretch along W; the turning part removed)
and with the straight Gaussian needle 2 sqrt(2 (beta_rad + beta_turn) kappa) = sqrt(m^4/4 + m^2/2)  (beta_turn = kappa = m/4).
Restartable: results/n3_tangent.json written after every point.   Usage: python n3_tangent.py ell|m"""
import numpy as np, json, os, sys
from scipy.optimize import minimize
from i5_arc import arc
from i_tools import fields

SIG = 1e-3
ELLS = [0.1, 0.2, 0.3, 0.44, 0.6, 0.8, 1.0, 1.4]


def n_for(ell, Ls_over_sqrt):
    return int(max(64, np.ceil(6 * Ls_over_sqrt * 1.2 / ell) + 1))


def model(z, ell, n):
    """z = (dr, tilt, k, log Ls/sqrt sigma, log thickness/sigma); l fixed -> arc parameters."""
    Ls = np.exp(z[3]); ds = 6 * Ls / (n - 1)                     # in units of sqrt(sigma)
    return np.array([z[0], z[1], z[2], z[3], z[4], np.log(ell / ds)])


def gapf(x, n, kind, kap, sig=SIG, gh=3, per_sd=5):
    if abs(x[2]) > 50 or not (-3 < x[3] < 3) or not (-12 < x[4] < 1) or not (-4 < x[5] < 6): return np.inf
    try: r = fields(*arc(x, n, sig), sig, kind=kind, per_sd=per_sd, gh=gh)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return np.inf
    return (kap - r['R']) / sig if abs(r['mass'] - 1) < 1e-6 else np.inf


if __name__ == '__main__':
    part = sys.argv[1]; fn = 'results/n3_tangent.json'; out = json.load(open(fn)) if os.path.exists(fn) else {'ell': [], 'm': []}
    x0 = np.array(json.load(open('results/m5_arc.json'))['by_n']['128']['x'])   # stage-M optimum (l = 0.44 sqrt sigma)
    if part == 'ell':
        done = {o['ell'] for o in out['ell']}; z = x0[:5].copy()
        for ell in ELLS:
            if ell in done: continue
            n = n_for(ell, np.exp(z[3]))
            r = minimize(lambda q: gapf(model(q, ell, n), n, 'twist', 0.25), z, method='Nelder-Mead', options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-7})
            xf = model(r.x, ell, n); g = gapf(xf, n, 'twist', 0.25, gh=4, per_sd=6)
            out['ell'].append({'ell': ell, 'n': n, 'gap': g, 'gap_search': float(r.fun), 'x': xf.tolist(), 'curvature': float(r.x[2]), 'Ls_over_sqrt_sigma': float(np.exp(r.x[3])),
                               'thickness_over_sigma': float(np.exp(r.x[4]))})
            json.dump(out, open(fn, 'w'), indent=1, default=float); print(ell, n, g, r.x[2], flush=True)
    else:
        done = {o['m'] for o in out['m']}
        for m in [1.0, 0.7, 1.4]:
            if m in done: continue
            kind = f'twistm{m:g}'; kap = m / 4; n = 64; x = x0.copy(); x[5] -= np.log(127 / 63)      # same l at n = 64
            r = minimize(lambda q: gapf(q, n, kind, kap), x, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-7})
            g = gapf(r.x, n, kind, kap, gh=4, per_sd=6); Ls = np.exp(r.x[3]); ell = np.exp(r.x[5]) * 6 * Ls / (n - 1)
            out['m'].append({'m': m, 'kappa': kap, 'gap': g, 'x': list(r.x), 'ell_over_sqrt_sigma': float(ell), 'curvature': float(r.x[2]),
                             'estimate_tangent_bundle': m * m / 2, 'estimate_straight_gaussian': float(np.sqrt(m ** 4 / 4 + m ** 2 / 2))})
            json.dump(out, open(fn, 'w'), indent=1, default=float); print(m, g, m * m / 2, ell, r.x[2], flush=True)
