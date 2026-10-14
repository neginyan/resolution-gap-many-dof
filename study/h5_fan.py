"""Stage H, task 2 (continued): the many-component limit of the structure found by the searches -- a FAN of zero-width segments
through the top (q = pi), all tilted slightly differently from W.  A fan with n segments:
    angle_j = (sqrt(sigma)/2) (t0 + dt z_j)   [so the ends of a segment move by sigma (t0 + dt z_j) l along V: the scale-invariant
                                              choice; with angles proportional to sigma the fan is not scale invariant]
    z_j on a uniform grid in [-3, 3],  weight_j ~ exp(-z_j^2/2) (1 + h z_j)_+ ,
    half-length_j = 2 sqrt(sigma) l0 (1 + l2 z_j^2)
(rank-one components; a tiny thickness 1e-4 sigma keeps the matrices regular).  Five parameters (t0, log dt, h, log l0, l2) are
optimised with Nelder-Mead for n = 8, 16, 32 at sigma = 1e-3, to see whether g* = (R - R*)/(1 - R*) keeps growing as the fan fills in.
Writes results/h5_fan.json.  (A first run used angle_j = sigma (t0 + dt z_j) with starts t0 in {0.25, -0.5, 0}, dt in {4, 8, 2}:
results/h5_fan_sigma_scaled_angles.json -- g* ~ 0.044 at sigma = 1e-3 but -0.006 for the same shape at 1e-4.)"""
import numpy as np, json
from scipy.optimize import minimize
from multiprocessing import Pool
from obs_eval2 import evaluate_full, V, W
from h1_rstar import rstar

TOP = np.array([np.pi, 0.0]); SIG = 1e-3


def fan(params, n, sig=SIG):
    t0, ldt, h, ll0, l2 = params
    z = np.linspace(-3, 3, n) if n > 1 else np.zeros(1); w = np.exp(-z ** 2 / 2) * np.maximum(1 + h * z, 0); w = w / w.sum()
    keep = w > 1e-4; z, w = z[keep], w[keep] / w[keep].sum()
    ang = np.sqrt(sig) / 2 * (t0 + np.exp(ldt) * z); hl = 2 * np.sqrt(sig) * np.exp(ll0) * np.maximum(1 + l2 * z ** 2, 0.05)
    Ss = []
    for a, l in zip(ang, hl):
        d = np.cos(a) * W - np.sin(a) * V                 # tilt from W towards -V by angle a
        Ss.append(l ** 2 * np.outer(d, d) + (1e-4 * sig) ** 2 * np.eye(2))
    return list(w), [TOP] * len(w), Ss


def gstar(params, n, sig=SIG, per_sd=6):
    ws, mus, Ss = fan(params, n, sig)
    r = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=per_sd); Rs = rstar(sig)['Rstar']
    return (r['R'] - Rs) / (1 - Rs), r


def job(n):
    best = None
    for x0 in [[0.0, np.log(0.5), 0.0, 0.1, 0.0], [0.1, np.log(1.0), 0.2, 0.2, 0.05], [0.0, np.log(0.25), -0.2, 0.0, 0.0]]:
        f = lambda x: -gstar(x, n)[0]
        r = minimize(f, x0, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-7})
        if best is None or r.fun < best.fun: best = r
    g10, r10 = gstar(best.x, n, per_sd=10); g16, r16 = gstar(best.x, n, per_sd=16); g4, _ = gstar(best.x, n, sig=1e-4, per_sd=10); g2, _ = gstar(best.x, n, sig=1e-2, per_sd=10)
    return {'n': n, 'params': list(best.x), 'gstar': g10, 'gstar_per_sd16': g16, 'gstar_sigma1e-4': g4, 'gstar_sigma1e-2': g2, 'R': r10['R'], 'mass': r10['mass'],
            'law': r10['law_term'], 'cov': r10['f'] - r10['law_term'], 'D': r10['D']}


if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(job, [1, 8, 16, 32])
    json.dump(out, open('results/h5_fan.json', 'w'), indent=1, default=float)
    for o in out: print(f"n = {o['n']}: g* {o['gstar']:+.5f} (per_sd 16 {o['gstar_per_sd16']:+.5f}, sigma 1e-4 {o['gstar_sigma1e-4']:+.5f}, 1e-2 {o['gstar_sigma1e-2']:+.5f}); params {np.round(o['params'], 3)}")
