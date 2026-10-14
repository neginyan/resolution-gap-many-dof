"""Stage N, task N1(a), second round (towards the infimum): for every result of n1_mix
  (i)  'cont':  L-BFGS-B continued from the stored optimum for `extra` more iterations (same settings);
  (ii) 'from1d' (coupled only): a second start, the POLISHED 1-d optimum of the same K embedded in the (V+, W+) plane with the b-law of
       the Gaussian optimum, then L-BFGS-B for (400 + extra) iterations, i.e. the same total gradient effort as the 1-d run.
The best of all runs is the reported value.  Checkpoints in results/n1_ckpt/, restartable.   -> results/n1_more.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import minimize
from n_tools import Flow, R_product, unpack, pack, gap, gauss_start
from n1_mix import rstar_gap, SIG


def lbfgs(x, f, maxiter, ck):
    done = 0
    if os.path.exists(ck): st = json.load(open(ck)); x = np.array(st['x']); done = st['it']
    while done < maxiter:
        n = min(25, maxiter - done)
        r = minimize(f, x, method='L-BFGS-B', options={'maxiter': n, 'maxfun': 10 ** 7, 'eps': 1e-5, 'ftol': 1e-14, 'gtol': 1e-10})
        x = r.x; done += n; json.dump({'x': x.tolist(), 'it': done, 'f': float(r.fun)}, open(ck, 'w'))
        if r.nit < n: break
    return x, done


def job(a):
    mode, kind, K, iters = a; eps = None if kind == '1d' else float(kind); fl = Flow(eps, SIG); f = lambda z: gap(z, K, fl, per_sd=3)
    src = {(o['kind'], o['K']): o for o in json.load(open('results/n1_mix.json'))}
    if mode == 'cont':
        x0 = np.array(src[(kind, K)]['x'])
    else:
        o1 = src[('1d', K)]; ws, mus, Ss, _, _ = unpack(np.array(o1['x']), K, Flow(None, SIG)); g = gauss_start(fl); x0 = pack(ws, mus, Ss, g[3], g[4], fl)
    f0 = f(x0); x, it = lbfgs(x0, f, iters, f'results/n1_ckpt/more_{mode}_{kind}_K{K}.json')
    ws, mus, Ss, s_b, th = unpack(x, K, fl); R6, parts = R_product(fl, ws, mus, Ss, s_b, th, per_sd=6, parts=True); g6 = (fl.k - R6) / SIG
    return {'mode': mode, 'kind': kind, 'K': K, 'gap_start': f0, 'gap': g6, 'iterations': it, 'ratio': g6 / rstar_gap(eps), 'parts': parts,
            's_b_over_sigma': None if s_b is None else s_b / SIG, 'theta': th, 'x': x.tolist()}


if __name__ == '__main__':
    extra = int(sys.argv[1]); Ks = [int(v) for v in sys.argv[2].split(',')]
    fn = 'results/n1_more.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['mode'], o['kind'], o['K']) for o in out}
    jobs = [('cont', kd, K, extra) for K in Ks for kd in ['1d', '0.1', '0.3']] + [('from1d', kd, K, 400 + extra) for K in Ks for kd in ['0.1', '0.3']]
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [j for j in jobs if j[:3] not in done]):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float)
            print(f"{r['mode']} {r['kind']} K={r['K']}: start {r['gap_start']:.5f} -> {r['gap']:.5f} ({r['iterations']} it)  ratio {r['ratio']:.5f}", flush=True)
