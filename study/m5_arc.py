"""Stage M, task M5 (partial; continuum() added): the bent needle with n straight pieces, re-optimised at n = 128 (Nelder-Mead from the n = 64 optimum
of stage L, Gauss-Hermite 3, per_sd 5; 250 iterations), then evaluated at n = 128, 256 (Gauss-Hermite 4, per_sd 6), sigma = 1e-3.
Written after every step to results/m5_arc.json (restartable from the stored best).  n = 512 and the extrapolation n -> infinity
were not affordable here (one evaluation at n = 512 takes minutes)."""
import numpy as np, json, os
from scipy.optimize import minimize
from l2_arc_continuum import gap

if __name__ == '__main__':
    fn = 'results/m5_arc.json'
    A = json.load(open('results/l2_arc_continuum.json'))
    out = json.load(open(fn)) if os.path.exists(fn) else {'start': 'stage L opt_x_n64', 'x0': A['opt_x_n64'], 'by_n': {}}
    for n, it in [(128, 250)]:
        if str(n) in out['by_n']: continue
        x0 = np.array(out['by_n'].get(str(n // 2), {}).get('x', out['x0']))
        r = minimize(lambda z: gap(z, n), x0, method='Nelder-Mead', options={'maxiter': it, 'xatol': 1e-4, 'fatol': 1e-7})
        out['by_n'][str(n)] = {'x': r.x.tolist(), 'opt_search': float(r.fun), 'gap_n128': gap(r.x, 128, gh=4, per_sd=6), 'gap_n256': gap(r.x, 256, gh=4, per_sd=6),
                               'start_value_n128': gap(x0, 128, gh=4, per_sd=6), 'curvature': float(r.x[2])}
        json.dump(out, open(fn, 'w'), indent=1); print(n, out['by_n'][str(n)], flush=True)


def continuum():
    """the n = 128 optimum keeps its pieces long: piece s.d. = c_len x spacing = 13.8 sigma (0.44 sqrt sigma), the same absolute length
    as stage I's n = 24 optimum.  Continuum test: n = 64, 128, 256 with the SAME absolute piece length (c_len rescaled with the
    spacing), Richardson extrapolation in 1/n, and sigma = 1e-4 (scale invariance).  Appends 'fixed_piece_length' to the json."""
    out = json.load(open('results/m5_arc.json')); x = np.array(out['by_n']['128']['x']); res = {}
    for n in [64, 128, 256]:
        y = x.copy(); y[5] += np.log((n - 1) / 127); res[str(n)] = gap(y, n, gh=4, per_sd=6); print(n, res[str(n)], flush=True)
    g = [res['64'], res['128'], res['256']]
    out['fixed_piece_length'] = {'by_n': res, 'extrap_richardson_1_over_n': 2 * g[2] - g[1], 'sigma1e-4_n128': gap(x, 128, sig=1e-4, gh=4, per_sd=6),
                                 'piece_sd_over_sigma': float(np.exp(x[5]) * 6 * np.sqrt(1e-3) * np.exp(x[3]) / 127 / 1e-3)}
    json.dump(out, open('results/m5_arc.json', 'w'), indent=1, default=float); print(out['fixed_piece_length'])
