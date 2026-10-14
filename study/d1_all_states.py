"""Stage D, tasks 1 and 2 (re-used in stage E with extra quantities: omega_rms, u = 1 - abar, valley depth): Conjecture 1 in flow coordinates (V, W) and the tightened bound B', on every state of stages B-C.

Pendulum (kappa = 1, 0 <= abar <= 1), with abar = E[a|Y], H_vv = v^T H v, H_ww = w^T H w (2 H_qp = H_vv - H_ww):
  f tr F = E[abar (H_vv - H_ww)],  tr F = E[H_vv + H_ww]
  R <= 1  <=>  M = E[(1 - abar) H_vv] + E[(1 + abar) H_ww] - D tr F >= 0,   and exactly M / tr F = 1 - R.
  Parts: Pv = E[(1-abar) H_vv]/trF, Pw = E[(1+abar) H_ww]/trF, and -D.
Quartic / double well (kappa = infinity): kappa replaced by kappa_loc = max |abar(y)| over the region the state occupies
  (p_Y >= 1e-6 max); M_loc / tr F = kappa_loc - R.
B' = E[|abar| (tr|H| - 2 (H_ww)_+)] / E[tr H] (as proposed); B'_gen uses H_vv instead of H_ww where abar < 0.
All states are evaluated on the grid aligned with V and W (obs_eval2.vw_grid); R is compared with the earlier (q, p)-grid values.
Writes results/d1_all.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from m1_mechanism import state
from c1_adversarial import unpack

KEEP = ['R', 'f', 'D', 'law_term', 'H_nonPD_mass', 'B_balance', 'B_prime', 'B_prime_gen', 'B_prime_pointwise_violation_mass',
        'Pv', 'Pw', 'M_over_trF', 'kappa_loc', 'PvL', 'PwL', 'ML_over_trF', 'mass_Hvv_neg', 'mass_Hww_neg', 'Pv_neg_part', 'Pw_neg_part',
        'abar_var', 'focus_mismatch', 'Ea2_rho', 'sigma', 'mass', 'lam_rho_direct',
        'omega_rms', 'mismatch2', 'u_mean', 'u_std', 'u_cv', 'abar_kurtosis', 'cov_cs_bound', 'valley_depth_mean', 'valley_depth_max', 'f_cov_part']


def run(job):
    tag, kind, ws, mus, Ss, sig, R_old = job
    r = evaluate_full(ws, mus, Ss, sig, kind=kind, grid='vw')
    out = {k: float(r[k]) for k in KEEP}; out.update({'tag': tag, 'kind': kind, 'R_qpgrid': R_old, 'n_vw': list(r['n'])})
    return out


if __name__ == '__main__':
    A = lambda L: [np.array(x) for x in L]
    # R_old: the earlier (q, p)-grid value of the same state (m2_mix.json is in the order of mixtures.json)
    mx = json.load(open('results/mixtures.json')); m2 = json.load(open('results/m2_mix.json'))
    assert all(a['i'] == b['i'] and a['sigma'] == b['sigma'] for a, b in zip(mx, m2))
    jobs = [('random', 'pendulum', m['ws'], A(m['mus']), A(m['Ss']), m['sigma'], r['R']) for m, r in zip(mx, m2)]
    for fn in ['results/m1_slices.json', 'results/m1_maps.json']:
        jobs += [('m1', 'pendulum', *state(**d['params']), 0.1, d['R']) for d in json.load(open(fn))]
    for fn in ['results/c1_adversarial.json', 'results/c1_adversarial_room.json']:
        jobs += [('c1', 'pendulum', *unpack(np.array(c['x']), c['K']), c['sigma'], c['R']) for c in json.load(open(fn))]
    for fn in ['results/c4_mix.json', 'results/c4_family.json']:
        jobs += [('c4', x['kind'], x['ws'], A(x['mus']), A(x['Ss']), x['sigma'], x['R']) for x in json.load(open(fn))]
    with Pool(2) as pool:
        out = pool.map(run, jobs, chunksize=8)
    import sys
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'results/d1_all.json', 'w'), indent=1)
    print(len(out))
