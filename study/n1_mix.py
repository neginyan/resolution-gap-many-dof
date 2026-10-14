"""Stage N, task N1(a): mixtures beating the best Gaussian, 1-d pendulum versus coupled pendulums (eps = 0.1, 0.3), sigma = 1e-3,
with the SAME procedure and the SAME effort for both.
Family for the coupled pendulums: product states of n_tools (a 2-d Gaussian mixture in (V+, a) times a common Gaussian of width s_b
in the rotated q- direction b, rotation angle theta free; p- infinitely long).  The best Gaussian of the 4-d problem is in this family
(theta = principal angle of T*), so the ratio is measured against the true best Gaussian.  Non-product refinements: n1_nonproduct.py.
Start (both cases): the K heaviest components of the best 1-d state of stage H (K = 8, h4_growk) for K = 2, 4, 8, and the stage-L
polished K = 16 state for K = 16, in the (V+, W+) plane; for the coupled pendulums the b-law (s_b, theta) of the Gaussian optimum.
Procedure: adaptive random hill climbing (`steps`), then L-BFGS-B with a forward-difference gradient (step 1e-5) for `maxiter`
iterations (the method that reached 0.916 in stage L), objective (kappa - R)/sigma with n_tools.R_product (per_sd 3; agrees with
per_sd 6 to ~1e-9 on these states).  Checkpoints every 25 iterations (results/n1_ckpt/), restartable.
Final: per_sd 6, the parts (num2, extra, J2, J_b) for the decomposition of N1(b), and the 3-d evaluator m3_mix.R3 on the same state.
Ratio = (kappa - R)/(kappa - R*), R* the best Gaussian: 1-d exact (h1_rstar), coupled: m1_free/m1_frame optimum at sigma = 1e-3.
Usage: python n1_mix.py <steps> <maxiter> <K list> [workers] [kinds, e.g. 1d,0.1,0.3]  -> results/n1_mix.json"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import minimize
from n_tools import Flow, R_product, unpack, pack, gap, gauss_start

SIG = 1e-3


def rstar_gap(eps):
    if eps is None:
        from h1_rstar import rstar
        return (1 - rstar(SIG)['Rstar']) / SIG
    fn = f'results/m1_free_eps{eps:g}.json'
    if os.path.exists(fn): return min(r['gap']['0.001'] for r in json.load(open(fn))['runs'])
    from m_tools import coupled, find_kappa, frame_series                     # eps without an m1_free run (stage N1(c) test)
    s = coupled(eps); k, t = find_kappa(s, starts=100); return frame_series(s, t[0], k, sigs=(1e-3,))[0]['0.001']


def start(fl, K):
    from h2_adversarial import unpack as u1, TOP, V as V1, W as W1
    R2 = np.stack([V1, W1], 1)
    if K <= 8:
        c = json.load(open('results/h4_growk.json')); best = max([x for ch in c for x in ch['chain'] if x['K'] == 8], key=lambda x: x['gstar10'])
        ws, mus, Ss = u1(np.array(best['x']), 8, SIG); o = np.argsort(ws)[::-1][:K]
    else:
        r = json.load(open('results/l1_polish_sigma0.001.json'))[0]; ws, mus, Ss = u1(np.array(r['x']), 16, SIG); o = np.argsort(ws)[::-1][:K]
    ws = np.array(ws)[o] / np.sum(np.array(ws)[o]); m2 = [R2.T @ (mus[i] - TOP) for i in o]; S2 = [R2.T @ Ss[i] @ R2 for i in o]
    if fl.oned: return pack(ws, m2, S2, None, 0.0, fl)
    g = gauss_start(fl); return pack(ws, m2, S2, g[3], g[4], fl)


def job(a):
    kind, K, steps, maxiter = a; eps = None if kind == '1d' else float(kind); fl = Flow(eps, SIG)
    tag = f'{kind}_K{K}'; ck = f'results/n1_ckpt/{tag}.json'; os.makedirs('results/n1_ckpt', exist_ok=True)
    f = lambda z: gap(z, K, fl, per_sd=3)
    if os.path.exists(ck):
        st = json.load(open(ck)); x = np.array(st['x']); stage = st['stage']; it_done = st.get('it', 0); f0 = st['f0']
    else:
        x = start(fl, K); f0 = f(x); stage = 'climb'; it_done = 0
    if stage == 'climb':
        rng = np.random.default_rng(100 + K); fx = f(x); step = 0.1; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = f(y)
            if fy < fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.5)); acc = 0
        stage = 'lbfgs'; json.dump({'x': x.tolist(), 'stage': stage, 'it': 0, 'f0': f0, 'after_climb': fx}, open(ck, 'w'))
    after_climb = json.load(open(ck)).get('after_climb')
    if stage == 'lbfgs':
        left = maxiter - it_done
        while left > 0:
            n = min(25, left)
            r = minimize(f, x, method='L-BFGS-B', options={'maxiter': n, 'maxfun': 10 ** 7, 'eps': 1e-5, 'ftol': 1e-14, 'gtol': 1e-10})
            x = r.x; left -= n; it_done += n
            json.dump({'x': x.tolist(), 'stage': 'lbfgs', 'it': it_done, 'f0': f0, 'after_climb': after_climb, 'f': float(r.fun)}, open(ck, 'w'))
            if r.nit < n: break
    ws, mus, Ss, s_b, th = unpack(x, K, fl)
    R6, parts = R_product(fl, ws, mus, Ss, s_b, th, per_sd=6, parts=True); g6 = (fl.k - R6) / SIG; gs = rstar_gap(eps)
    out = {'kind': kind, 'K': K, 'gap_start': f0, 'gap_after_climb': after_climb, 'gap': g6, 'gap_per_sd3': f(x), 'gap_gauss': gs, 'ratio': g6 / gs,
           'lbfgs_iterations': it_done, 'parts': parts, 's_b_over_sigma': None if s_b is None else s_b / SIG, 'theta': th, 'x': x.tolist()}
    if not fl.oned and K <= 8:
        from m3_mix import Setup, R3
        st = Setup(eps, SIG); c, s = np.cos(th), np.sin(th); Rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        m3 = [Rot @ np.array([m[0], m[1], 0]) for m in mus]; S3 = [Rot @ np.block([[S, np.zeros((2, 1))], [np.zeros((1, 2)), np.array([[s_b]])]]) @ Rot.T for S in Ss]
        R3v, mass, n = R3(st, list(ws), m3, S3, per_sd=5); out['gap_3d_evaluator'] = (st.k - R3v) / SIG
    return out


if __name__ == '__main__':
    steps, maxiter = int(sys.argv[1]), int(sys.argv[2]); Ks = [int(v) for v in sys.argv[3].split(',')]; W = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    KINDS = sys.argv[5].split(',') if len(sys.argv) > 5 else ['1d', '0.1', '0.3']
    fn = 'results/n1_mix.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['kind'], o['K']) for o in out}
    jobs = [(kd, K, steps, maxiter) for K in Ks for kd in KINDS if (kd, K) not in done]
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1, default=float)
            print(f"{r['kind']} K={r['K']}: start {r['gap_start']:.5f} climb {r['gap_after_climb']:.5f} -> {r['gap']:.5f} (L-BFGS {r['lbfgs_iterations']} it)  "
                  f"Gauss {r['gap_gauss']:.5f}  ratio {r['ratio']:.5f}" + (f"  3-d check {r['gap_3d_evaluator']:.6f}" if 'gap_3d_evaluator' in r else ''), flush=True)
