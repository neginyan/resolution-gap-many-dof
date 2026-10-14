"""Verification for stage C+ (balance bound and curvature picture). Exit code 0 iff all checks pass."""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
from obs_eval2 import evaluate_full
from m1_mechanism import state

LOOSE = {'spearman': ('abs', 0.02), 'slope': 1e-6}  # rank correlations of tied data (CPU-dependent order of values at rounding level): absolute 0.02; log-log fit slopes: 1e-6
from vtools import check, finish, reproduce

rng = np.random.default_rng(31)
def rnd(K):
    ws = list(rng.dirichlet([2] * K)); m0 = np.array([rng.uniform(0, 2 * np.pi), rng.normal(0, 0.8)])
    mus = [m0 + (rng.normal(0, 0.3, 2) if k else 0) for k in range(K)]; Ss = []
    for _ in range(K):
        A = rng.normal(size=(2, 2)) * rng.uniform(0.1, 0.6); Ss.append(A @ A.T + 0.005 * np.eye(2))
    return ws, mus, Ss

# 1. inequalities and identities on fresh states (three flows)
w = {'fB': -1, 'RBD': -1, 'split': 0, 'ident': 0, 'nov': -1}
for kind in ['pendulum', 'quartic', 'double_well']:
    for K, sig in [(2, 0.1), (3, 0.3), (2, 1.0), (1, 0.3)]:
        ws, mus, Ss = rnd(K) if K > 1 else ([1.0], [np.array([rng.uniform(0, 6), 0.1])], [np.diag([0.2, 0.05])])
        r = evaluate_full(ws, mus, Ss, sig, kind=kind, n=601)
        w['fB'] = max(w['fB'], r['f'] - r['B_balance']); w['RBD'] = max(w['RBD'], r['R'] - r['B_balance'] - r['D'])
        w['split'] = max(w['split'], abs(r['cov_pos_curv'] + r['cov_neg_curv'] - r['f_cov_part']))
        if kind == 'pendulum':
            w['ident'] = max(w['ident'], abs(r['Ea2_rho'] - (0.5 - r['lam_rho_direct'])))
            if r['B_kappa_factor'] < 1 + 1e-12: w['nov'] = max(w['nov'], r['B_balance'] - 1)
check('f <= B and R <= B + D on fresh states (three flows, incl. single Gaussians)', w['fB'] <= 1e-10 and w['RBD'] <= 1e-10, value=max(w['fB'], w['RBD']), tol=1e-10, info=
      f"(max f - B {w['fB']:+.2e}, max R - B - D {w['RBD']:+.2e})")
check('covariance term = part from local curvature > 0 + part from <= 0', w['split'] < 1e-12, f"({w['split']:.1e})", value=w['split'], tol=1e-12)
check("pendulum: E_rho[a''] = 1/2 - lambda_rho exactly (a + a'' = 1/2 pointwise)", w['ident'] < 1e-14, f"({w['ident']:.1e})", value=w['ident'], tol=1e-14)
check('pendulum without valley: B <= kappa', w['nov'] <= 1e-12, value=w['nov'], tol=1e-12)

# 2. independent check of B: H by finite differences of log p_Y on the grid, a_eff from the fields
ws, mus, Ss = state(0.0); r = evaluate_full(ws, mus, Ss, 0.1, n=1201, return_fields=True); F = r['fields']
hq, hp = F['q'][1] - F['q'][0], F['p'][1] - F['p'][0]; p = np.nan_to_num(F['pY']); L = np.log(np.maximum(p, 1e-300))
sq, sp = np.gradient(L, hq, axis=0), np.gradient(L, hp, axis=1)
Hqq, Hpp, Hqp = -np.gradient(sq, hq, axis=0), -np.gradient(sp, hp, axis=1), -np.gradient(sq, hp, axis=1)
m = p > p.max() * 1e-8; I = (slice(3, -3), slice(3, -3)); m = m & np.pad(np.ones((p.shape[0] - 6, p.shape[1] - 6), bool), 3)
tr = Hqq + Hpp; det = Hqq * Hpp - Hqp ** 2; disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0)); l1, l2 = tr / 2 + disc, tr / 2 - disc
Bfd = np.sum((p * np.abs(np.nan_to_num(F['a_eff'])) * (np.abs(l1) + np.abs(l2)))[m]) / np.sum((p * tr)[m])
check('B of the base state reproduced by finite differences of log p_Y', abs(Bfd - r['B_balance']) < 2e-3, f"({Bfd:.5f} vs {r['B_balance']:.5f})",
      value=abs(Bfd - r['B_balance']), tol=2e-3)

# 3. summary reproducible and quoted numbers
ok_rep, dev_rep, where_rep, s = reproduce('results/summary_C5.json', 'c5_analyze.py', loose=LOOSE)
check('summary_C5.json is reproduced from the raw result files (relative 1e-9; rank correlations 0.02 absolute, fit slopes 1e-6; the stored file is left unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
f2, f3 = (lambda x: '%.2f' % x), (lambda x: '%.3f' % x)
db = s['dist_bins']
quoted = {
    'states: 1536 pendulum, 816 quartic / double well': (s['n_pendulum'], s['n_other']) == (1536, 816),
    'sign(cov) = sign(E_rho a\'\') in 59% (1535 non-zero); 254 neg-curvature/pos-cov, 368 pos-curvature/neg-cov':
        ('%.0f' % (100 * s['sign_cov_eq_sign_Ea2rho']), s['n_cov_nonzero'], s['n_neg_curv_pos_cov'], s['n_pos_curv_neg_cov']) == ('59', 1535, 254, 368),
    'largest covariance term at negative felt curvature: 0.603': f3(s['max_cov_at_neg_curv']) == '0.603',
    'other flows (a\'\' = -3): covariance positive in 418, negative in 396': (s['other_n_cov_pos'], s['other_n_cov_neg']) == (418, 396),
    'positive covariance comes from local curvature <= 0 (median share 1.06)': f2(s['cov_from_neg_local_curv_share_of_pos_cov']) == '1.06',
    'excess <= kappa - law in all; max ratio 0.714 (at E a\'\' -0.174, no overlap, no valley)':
        (s['all_excess_le_dist'], f3(s['max_ratio']), f3(s['max_ratio_state']['Ea2_rho']), s['max_ratio_state']['H_nonPD_mass'] < 1e-9) == (True, '0.714', '-0.174', True),
    'by curvature bin: max ratio 0.515 / 0.714 / 0.633 / 0.581':
        [f3(s['max_ratio_curv_bins'][k]['max_ratio']) for k in ['-0.5..-0.25', '-0.25..0.0', '0.0..0.25', '0.25..0.5']] == ['0.515', '0.714', '0.633', '0.581'],
    'by distance bin: n 6 / 24 / 950 / 556, max excess 0.057 / 0.320 / 0.490 / 0.603, max ratio 0.515 / 0.714 / 0.633 / 0.359':
        ([db[k]['n'] for k in db], [f3(db[k]['max_excess']) for k in db], [f3(db[k]['max_ratio']) for k in db]) ==
        ([6, 24, 950, 556], ['0.057', '0.320', '0.490', '0.603'], ['0.515', '0.714', '0.633', '0.359']),
    'min distance from the peak 0.111; distance >= 1/2 + E a\'\' in all': (f3(s['min_dist']), s['dist_ge_half_plus_Ea2_all']) == ('0.111', True),
    'balance bound: f <= B and R <= B + D in all; B <= 1 in 1515, B + D <= 1 in 1513 of 1536':
        (s['f_le_B_all'], s['R_le_B_plus_D_all'], s['n_B_le_1'], s['n_B_plus_D_le_1'], s['other_f_le_B']) == (True, True, 1515, 1513, True),
    'B > 1 in 21 states (all with valley mass >= 0.012), max B 1.121; no-valley max B 0.992; 314 states without valley':
        (s['n_B_gt_1'], f3(s['min_valley_mass_B_gt_1']), f3(s['max_B']), f3(s['max_B_no_valley']), s['n_no_valley']) == (21, '0.012', '1.121', '0.992', 314),
    'the largest-R state (0.946) is among those with B > 1': f3(s['max_R_where_B_gt_1']) == '0.946',
    'looseness B - f median 0.26; Spearman with valley mass -0.30': (f2(s['B_minus_f_median']), f2(s['spearman_gap_valley'])) == ('0.26', '-0.30'),
}
X = [x for x in json.load(open('results/c5_balance.json')) if x['kind'] == 'pendulum']
exc = np.array([x['R'] - x['law_term'] for x in X]); dist = np.array([1 - x['law_term'] for x in X])
quoted['max-ratio state at distance 0.37; all 5 states with excess > 0.4 have distance > 0.6'] = \
    (f2(dist[np.argmax(exc / dist)]), int((exc > 0.4).sum()), bool(np.all(dist[exc > 0.4] > 0.6))) == ('0.37', 5, True)
for k, v in quoted.items():
    check('summary number: ' + k, v)
finish()
