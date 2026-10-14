"""Stage C, proposal 4: the same analysis for two other flows f = (p, g(q)):
  quartic oscillator  g = -q^3       (Sym Df = a(q)[[0,1],[1,0]], a = (1 - 3q^2)/2; the stretch is unbounded, kappa = infinity)
  double well         g = q - q^3    (a = (2 - 3q^2)/2; hyperbolic point q = 0, stable points q = +-1)
Parts
  single : random single Gaussians x 5 sigmas: Gaussian law R = tr(F Sym Jbar)/tr F, and the general remainder formula
           R - e = -E_rho[g'''(q)] K_qq K_qp / tr F   (pendulum: g''' = cos q; quartic and double well: g''' = -6)
  mix    : random two-component mixtures x 4 sigmas: three-term split, R vs lambda_rho / lambda_comp, where the excess comes from
  family : the measurement-1 family (concentric, narrow + wide, long along W) centred at q = 0 (quartic) and q = 1 (double well),
           width ratio 1, 2, 4.4, 8
Writes results/c4_<part>.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from obs_eval2 import evaluate_full, V, W

KINDS = ['quartic', 'double_well']
SIGS = [0.03, 0.1, 0.3, 1.0]
KEEP = ['R', 'e', 'f', 'D', 'law_term', 'lam_rho_direct', 'lam_comp', 'A', 'R_sep', 'R_switch', 'H_nonPD_mass', 'n', 'mass',
        'tweedie_err', 'D_within', 'D_between', 'resp_var']


def rand_state(rng, K):
    ws = list(rng.dirichlet([2] * K)) if K > 1 else [1.0]
    mus = [np.array([rng.normal(0, 0.8), rng.normal(0, 0.8)]) for _ in range(K)]
    Ss = []
    for _ in range(K):
        A = rng.normal(0, 1, (2, 2)) * 10 ** rng.uniform(-1.3, -0.4); Ss.append(A @ A.T + 1e-3 * np.eye(2))
    return ws, mus, Ss


def run(job):
    tag, kind, ws, mus, Ss, sig, extra = job
    r = evaluate_full(ws, mus, Ss, sig, kind=kind)
    out = {k: float(r[k]) for k in KEEP}
    out.update({'tag': tag, 'kind': kind, 'sigma': sig, 'ws': [float(w) for w in ws], 'mus': [m.tolist() for m in mus],
                'Ss': [S.tolist() for S in Ss]}); out.update(extra)
    return out


if __name__ == '__main__':
    part = sys.argv[1]
    rng = np.random.default_rng(404)
    jobs = []
    if part == 'single':
        for kind in KINDS:
            for i in range(30):
                ws, mus, Ss = rand_state(rng, 1)
                for sig in [0.03, 0.1, 0.3, 1.0, 2.0]:
                    jobs.append(('single', kind, ws, mus, Ss, sig, {'i': i}))
    elif part == 'mix':
        for kind in KINDS:
            for i in range(100):
                ws, mus, Ss = rand_state(rng, 2)
                if i % 2 == 0: mus[1] = mus[0] + rng.normal(0, 0.3, 2)          # half of the states forced to overlap
                for sig in SIGS:
                    jobs.append(('mix', kind, ws, mus, Ss, sig, {'i': i}))
    else:
        for kind, c0 in [('quartic', 0.0), ('double_well', 1.0)]:
            for ratio in [1.0, 2.0, 0.4 / 0.09, 8.0]:
                for l1 in [0.5, 1.0]:
                    n1 = 0.09
                    S1 = n1 ** 2 * np.outer(V, V) + l1 ** 2 * np.outer(W, W)
                    S2 = (ratio * n1) ** 2 * np.outer(V, V) + (0.63 * l1) ** 2 * np.outer(W, W)
                    jobs.append(('family', kind, [0.5, 0.5], [np.array([c0, 0.0])] * 2, [S1, S2], 0.1, {'ratio': ratio, 'l1': l1}))
    from obs_eval import mixture_grid
    big = [mixture_grid(j[2], j[3], j[4], j[5])[3] > 1500 for j in jobs]       # ~3 GB per 2401^2 grid: run those one at a time
    with Pool(2) as pool:
        out = pool.map(run, [j for j, b in zip(jobs, big) if not b], chunksize=2)
    out += [run(j) for j, b in zip(jobs, big) if b]
    json.dump(out, open(f'results/c4_{part}.json', 'w'), indent=1)
    print(part, len(out))
