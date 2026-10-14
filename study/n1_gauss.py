"""Stage N, task N1(b): the best GAUSSIAN as a product state (K = 1 of n_tools, same procedure as n1_mix), for the decomposition
of the mixture gain: parts (num2, extra, J2, J_b), s_b and theta of the Gaussian optimum.  Its gap must equal the 4-d optimum of
m1_free at sigma = 1e-3 (the product family contains the best Gaussian).   -> results/n1_gauss.json"""
import numpy as np, json
from n_tools import Flow, R_product, unpack, pack, gap, gauss_start
from n1_mix import rstar_gap, SIG
from scipy.optimize import minimize

if __name__ == '__main__':
    out = []
    for kind in ['1d', '0.1', '0.3']:
        eps = None if kind == '1d' else float(kind); fl = Flow(eps, SIG); x = pack(*gauss_start(fl), fl); f = lambda z: gap(z, 1, fl, per_sd=6)
        for _ in range(3):
            r = minimize(f, x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-13, 'adaptive': True}); x = r.x
            r = minimize(f, x, method='L-BFGS-B', options={'maxiter': 500, 'eps': 1e-7, 'ftol': 1e-15, 'gtol': 1e-12}); x = r.x
        ws, mus, Ss, s_b, th = unpack(x, 1, fl); R, parts = R_product(fl, ws, mus, Ss, s_b, th, per_sd=8, parts=True); g = (fl.k - R) / SIG
        out.append({'kind': kind, 'gap': g, 'gap_reference': rstar_gap(eps), 'parts': parts, 's_b_over_sigma': None if s_b is None else s_b / SIG, 'theta': th,
                    'sd_a_over_sqrt_sigma': float(np.sqrt(Ss[0][1, 1] / SIG)), 'x': x.tolist()})
        print(kind, g, rstar_gap(eps), out[-1]['s_b_over_sigma'], th, flush=True)
    json.dump(out, open('results/n1_gauss.json', 'w'), indent=1, default=float)
