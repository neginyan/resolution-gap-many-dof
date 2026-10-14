"""Stage C+: the balance bound and the curvature picture, re-evaluated on every state of stages B and C.

For each state: R, law term, covariance term, D, and
  B_balance      = E[|a_eff| tr|H|] / E[tr H]   (tr(HG) <= lambda_max(G) tr H_+ - lambda_min(G) tr H_-, eigenvalues of G = +-|a_eff|)
                   so f <= B_balance and R <= B_balance + D (exact inequalities)
  B_kappa_factor = E[tr|H|] / E[tr H]  (= 1 without valleys; then f <= kappa)
  Ea2_rho        = E_rho[a''(q)]  (curvature of the stretch felt by the state; pendulum: = 1/2 - lambda_rho exactly)
  Ea2_pY         = E_{p_Y}[a''(q)]
  cov_pos_curv / cov_neg_curv: covariance term from points where the local curvature E[a''(X)|y] is > 0 / <= 0
Sets: random pendulum mixtures (1000), measurement-1 states (512), stage-C adversarial states (24),
      quartic / double-well mixtures (800) and family (16).  Writes results/c5_balance.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval import mixture_grid
from obs_eval2 import evaluate_full
from m1_mechanism import state
from c1_adversarial import unpack

KEEP = ['R', 'f', 'e', 'D', 'law_term', 'lam_rho_direct', 'lam_comp', 'H_nonPD_mass', 'B_balance', 'B_kappa_factor', 'Ea2_rho', 'Ea2_pY',
        'cov_pos_curv', 'cov_neg_curv', 'n', 'mass', 'sigma']


def run(job):
    tag, kind, ws, mus, Ss, sig = job
    r = evaluate_full(ws, mus, Ss, sig, kind=kind)
    out = {k: float(r[k]) for k in KEEP}; out.update({'tag': tag, 'kind': kind, 'cov': float(r['f'] - r['law_term'])})
    return out


if __name__ == '__main__':
    A = lambda L: [np.array(x) for x in L]
    jobs = [('random', 'pendulum', m['ws'], A(m['mus']), A(m['Ss']), m['sigma']) for m in json.load(open('results/mixtures.json'))]
    for fn in ['results/m1_slices.json', 'results/m1_maps.json']:
        jobs += [('m1', 'pendulum', *state(**d['params']), 0.1) for d in json.load(open(fn))]
    for fn in ['results/c1_adversarial.json', 'results/c1_adversarial_room.json']:
        jobs += [('c1', 'pendulum', *unpack(np.array(c['x']), c['K']), c['sigma']) for c in json.load(open(fn))]
    for fn in ['results/c4_mix.json', 'results/c4_family.json']:
        jobs += [('c4', x['kind'], x['ws'], A(x['mus']), A(x['Ss']), x['sigma']) for x in json.load(open(fn))]
    big = [mixture_grid(j[2], j[3], j[4], j[5])[3] > 1500 for j in jobs]
    with Pool(2) as pool:
        out = pool.map(run, [j for j, b in zip(jobs, big) if not b], chunksize=4)
    out += [run(j) for j, b in zip(jobs, big) if b]               # ~3 GB each: one at a time
    json.dump(out, open('results/c5_balance.json', 'w'), indent=1)
    print(len(out))
