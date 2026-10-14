"""Stage N, task N1(b): does leaving the product family help?  Each product optimum of n1_mix (coupled pendulums, K = 2, 4) is
written as a general 3-d mixture (every component its own 3 x 3 covariance and its own mean in (V+, W+, q-)) and polished with
L-BFGS-B (forward differences, step 1e-5) on the 3-d evaluator m3_mix.R3 (per_sd 3; final per_sd 5).
Reported: the gain over the product optimum, and per component the q--direction spread (in units of sqrt sigma) and the correlation
of the q- coordinate with W+, against the values of the Gaussian optimum.   -> results/n1_nonproduct.json (checkpoint per job)"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import minimize
from n_tools import Flow, unpack as unpack_p
from m3_mix import Setup, gapf, pack, unpack

SIG = 1e-3


def to3d(o):
    fl = Flow(float(o['kind']), SIG); ws, mus, Ss, s_b, th = unpack_p(np.array(o['x']), o['K'], fl)
    c, s = np.cos(th), np.sin(th); Rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    m3 = [Rot @ np.array([m[0], m[1], 0]) for m in mus]; S3 = [Rot @ np.block([[S, np.zeros((2, 1))], [np.zeros((1, 2)), np.array([[s_b]])]]) @ Rot.T for S in Ss]
    return list(ws), m3, S3


def stats(ws, mus, Ss):
    return [{'w': float(w), 'sd_q_over_sqrt_sigma': float(np.sqrt(S[2, 2] / SIG)), 'corr_wq': float(S[1, 2] / np.sqrt(S[1, 1] * S[2, 2])),
             'sd_w_over_sqrt_sigma': float(np.sqrt(S[1, 1] / SIG)), 'mean_q_over_sqrt_sigma': float(m[2] / np.sqrt(SIG))} for w, m, S in zip(ws, mus, Ss)]


def job(a):
    o, maxiter = a; eps = float(o['kind']); K = o['K']; st = Setup(eps, SIG); tag = f"{o['kind']}_K{K}"; ck = f'results/n1_ckpt/nonprod_{tag}.json'
    ws, mus, Ss = to3d(o); x0 = pack(ws, mus, Ss, SIG); g0 = gapf(st, x0, K, per_sd=5)
    x = np.array(json.load(open(ck))['x']) if os.path.exists(ck) else x0.copy(); done = json.load(open(ck))['it'] if os.path.exists(ck) else 0
    f = lambda z: gapf(st, z, K, per_sd=3)
    while done < maxiter:
        r = minimize(f, x, method='L-BFGS-B', options={'maxiter': 10, 'maxfun': 10 ** 6, 'eps': 1e-5, 'ftol': 1e-14, 'gtol': 1e-10})
        x = r.x; done += 10; json.dump({'x': x.tolist(), 'it': done}, open(ck, 'w'))
        if r.nit < 10: break
    g = gapf(st, x, K, per_sd=5); w2, m2, S2 = unpack(x, K, SIG)
    return {'kind': o['kind'], 'K': K, 'gap_product': o['gap'], 'gap_product_R3': g0, 'gap_nonproduct': g, 'gain_rel': (g0 - g) / g0, 'iterations': done,
            'product_components': stats(ws, mus, Ss), 'nonproduct_components': stats(w2, m2, S2), 'x': x.tolist()}


if __name__ == '__main__':
    maxiter = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    src = [o for o in json.load(open('results/n1_mix.json')) if o['kind'] != '1d' and o['K'] <= 4]
    fn = 'results/n1_nonproduct.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['kind'], o['K']) for o in out}
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [(o, maxiter) for o in src if (o['kind'], o['K']) not in done]):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float)
            print(f"{r['kind']} K={r['K']}: product {r['gap_product_R3']:.6f} -> non-product {r['gap_nonproduct']:.6f} (gain {100 * r['gain_rel']:.3f}%)", flush=True)
