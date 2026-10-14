"""Stage M, task M1(c): the coupled pendulums outside the range used so far: eps = -0.3, -0.1 (coupling of the opposite sign), 2, 3.
First: where is kappa attained (m_tools.find_kappa: 400 random starts; the candidates (pi, 0), (0, pi), (pi, pi) are evaluated
explicitly)?  Then B and the formula at the new top, and a full optimisation (no assumed shape; m2_opt.run: ansatz start + random
starts, continuation to sigma = 1e-5) compared with the formula.   -> results/m1_range.json (written after every eps; restartable)
Usage: python m1_range.py [n_random] [workers]"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from m_tools import coupled, find_kappa, coefficient, ansatz_state
from m2_opt import run, shape

EPS = [-0.3, -0.1, 2.0, 3.0]


def job(a):
    e, nrand = a; s = coupled(e); k, tops = find_kappa(s, starts=400, seed=3)
    cand = {name: float(s.lam(np.array(q))) for name, q in [('(pi,0)', [np.pi, 0]), ('(0,pi)', [0, np.pi]), ('(pi,pi)', [np.pi, np.pi]), ('(0,0)', [0, 0])]}
    q0 = tops[0]; c = coefficient(s, q0, k)
    m, L = ansatz_state(s, q0, k, 1e-1, wide=300.0); xa, va = run(np.concatenate([m, L.ravel()]), s, k)
    rng = np.random.default_rng(91 + int(10 * e)); rand = []
    for i in range(nrand):
        x = np.concatenate([rng.uniform(-np.pi, np.pi, 2), rng.normal(0, 0.5, 16)]); xr, vr = run(x, s, k)
        rand.append({'vals': vr, 'mq': (np.mod(xr[:2] + np.pi, 2 * np.pi) - np.pi).tolist()})
    best = min([va] + [r['vals'] for r in rand], key=lambda v: v['1e-05'])
    ext = best['1e-05'] - (best['0.0001'] - best['1e-05']) / 9
    return {'eps': e, 'kappa': k, 'tops': [t.tolist() for t in tops], 'lambda_at_candidates': cand, 'formula': c['coef'], 'B': c['B'],
            'eig_M_top': c['eig_M_top'], 'from_ansatz': va, 'random': rand, 'best': best, 'extrap_sigma0': ext,
            'opt_rel_diff_extrap_vs_formula': (ext - c['coef']) / c['coef'], 'shape_from_ansatz_run': shape(s, xa, q0, 1e-5)}


if __name__ == '__main__':
    nrand = int(sys.argv[1]) if len(sys.argv) > 1 else 4; W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    fn = 'results/m1_range.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {o['eps'] for o in out}
    with Pool(W) as pool:
        for r in pool.imap_unordered(job, [(e, nrand) for e in EPS if e not in done]):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1)
            print(f"eps {r['eps']:g}: kappa {r['kappa']:.8f} at {np.round(r['tops'], 5).tolist()}  formula {r['formula']:.6f}  optimum 1e-5 {r['best']['1e-05']:.6f} "
                  f"extrapolated {r['extrap_sigma0']:.6f} (rel {r['opt_rel_diff_extrap_vs_formula']:.1e})", flush=True)
