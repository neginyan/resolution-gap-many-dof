"""Stage I, task 3 (continued): a CURVED needle for the twist flow.  The stretching direction of the twist flow turns with the
polar angle (Sym Df at polar angle phi is the one at phi = 0 rotated by phi), so a straight segment along the local contracting
direction W sees its stretching direction turn along its length.  For a single straight Gaussian the loss of stretch along the
segment is kappa - beta t^2 with beta = 3/8 = 1/8 (radial fall of a = r^2/(1+r^2)^2) + 1/4 (turning of the direction).
A curved needle whose normal follows the turning should keep only the radial part: beta_arc = 1/8, and the gap coefficient
2 sqrt(2 beta kappa) would drop from sqrt(3)/2 = 0.866 to 1/2.
Model: n short segments along an arc through (r0, 0): arc-length s in [-3 Ls, 3 Ls], direction angle theta(s) = theta0 + k s with
theta0 = angle of W (-pi/4) + tilt; weights ~ exp(-s^2/(2 Ls^2)); each segment's s.d. along the arc = c_len * spacing, across = thickness.
Parameters (r0 - 1, tilt, curvature k, Ls, thickness, c_len) are optimised (Nelder-Mead) at sigma = 1e-3 with n = 24 and the
optimum re-evaluated with n = 16 ... 96 (continuum), per_sd 14 and sigma = 1e-4, 1e-2 (scale invariance);
the optimum curvature is compared with the turning rate of the stretching direction along W (1/sqrt 2).
Writes results/i5_arc.json."""
import numpy as np, json
from multiprocessing import Pool
from scipy.optimize import minimize
from i_tools import fields, KAPPA
from i3_flows import rstar_flow

SIG = 1e-3; KIND = 'twist'


def arc(x, n, sig=SIG):
    dr, tilt, k, lLs, lth, lcl = x
    Ls = np.sqrt(sig) * np.exp(lLs); th = sig * np.exp(lth)
    s = np.linspace(-3 * Ls, 3 * Ls, n); ds = s[1] - s[0]
    th0 = -np.pi / 4 + tilt * sig
    ang = th0 + k * s
    # position by exact integration of a circular arc (constant curvature k)
    if abs(k) > 1e-12: pos = np.stack([(np.sin(th0 + k * s) - np.sin(th0)) / k, -(np.cos(th0 + k * s) - np.cos(th0)) / k], -1)
    else: pos = np.stack([s * np.cos(th0), s * np.sin(th0)], -1)
    c0 = np.array([1 + dr * sig, 0.0]); w = np.exp(-s ** 2 / (2 * Ls ** 2)); w /= w.sum()
    cl = np.exp(lcl) * ds; mus, Ss = [], []
    for p, a in zip(pos, ang):
        t = np.array([np.cos(a), np.sin(a)]); nr = np.array([-np.sin(a), np.cos(a)])
        mus.append(c0 + p); Ss.append(cl ** 2 * np.outer(t, t) + th ** 2 * np.outer(nr, nr))
    return list(w), mus, Ss


def gval(x, n, sig=SIG, gh=3, per_sd=5):
    if np.any(np.abs(np.array(x)[[0, 1]]) > 200) or abs(x[2]) > 50 or not (-3 < x[3] < 3) or not (-12 < x[4] < 1) or not (-3 < x[5] < 3): return -np.inf, None
    ws, mus, Ss = arc(x, n, sig)
    try: r = fields(ws, mus, Ss, sig, kind=KIND, per_sd=per_sd, gh=gh)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return -np.inf, None
    if abs(r['mass'] - 1) > 1e-6: return -np.inf, None
    Rs = rstar_flow(KIND, sig); return (r['R'] - Rs) / (KAPPA[KIND] - Rs), r


def job(args):
    """optimise at n = 24 (2 Nelder-Mead starts, 300 iterations each), then evaluate the optimum with n = 16, 24, 48, 96."""
    k0, seed = args; best = None
    starts = {0: [[0.0, 0.0, k0, np.log(0.6), np.log(0.01), 0.3], [1.0, 0.5, k0, np.log(0.5), np.log(0.03), 0.0]],
              1: [[0.0, 0.0, 0.0, np.log(0.6), np.log(0.01), 0.3], [-1.0, -0.5, k0, np.log(0.7), np.log(0.01), 0.5]]}[seed]
    for x0 in starts:
        r = minimize(lambda z: -gval(z, 24)[0], x0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-8})
        if best is None or r.fun < best.fun: best = r
    x = best.x; res = {'k0': k0, 'seed': seed, 'x': list(x), 'curvature': x[2], 'turning_rate_along_W': 1 / np.sqrt(2), 'gstar_search_n24': -best.fun}
    for n in [16, 24, 48, 96]:
        g, rr = gval(x, n, gh=6, per_sd=10); res[f'gstar_n{n}'] = g; res[f'gap_over_sigma_n{n}'] = (0.25 - rr['R']) / SIG
    res['gstar_n48_per_sd14'] = gval(x, 48, gh=4, per_sd=14)[0]
    for s in [1e-4, 1e-2]:
        g, rr = gval(x, 48, sig=s, gh=6, per_sd=10); res[f'gstar_n48_sigma{s:g}'] = g; res[f'gap_over_sigma_n48_sigma{s:g}'] = (0.25 - rr['R']) / s
    return res


if __name__ == '__main__':
    out = []
    with Pool(2) as pool:
        for o in pool.imap_unordered(job, [(-0.7, 0), (0.7, 0), (-0.7, 1), (0.7, 1)]):
            out.append(o); json.dump(out, open('results/i5_arc.json', 'w'), indent=1, default=float)
            print(f"start k0 {o['k0']:+.1f} / {o['seed']}: g* n16 {o['gstar_n16']:+.4f} n24 {o['gstar_n24']:+.4f} n48 {o['gstar_n48']:+.4f} n96 {o['gstar_n96']:+.4f} "
                  f"(per_sd14 {o['gstar_n48_per_sd14']:+.4f}; sigma 1e-4 {o['gstar_n48_sigma0.0001']:+.4f}, 1e-2 {o['gstar_n48_sigma0.01']:+.4f}); "
                  f"(kappa - R)/sigma {o['gap_over_sigma_n48']:.4f}; curvature {o['curvature']:+.4f} (turning rate 0.7071)", flush=True)
