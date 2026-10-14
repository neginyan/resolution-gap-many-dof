"""Verification for stage E. Exit code 0 iff all checks pass; failed checks are listed by name, and checks that use more than
10 % of their tolerance are listed as well (vtools.finish)."""
import sys, os, json, shutil, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
LOOSE = {'spearman': ('abs', 0.02), 'slope': 1e-6}  # rank correlations of tied data (CPU-dependent order of values at rounding level): absolute 0.02; log-log fit slopes: 1e-6
from vtools import check, finish, reproduce
from obs_eval2 import evaluate_full
from c1_adversarial import unpack as unpack_c1

# 0. the reproducibility check tolerates last-bit differences and never changes the stored file
tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json'); tmp.close(); shutil.copyfile('results/summary_D.json', tmp.name)
s = json.load(open('results/summary_D.json')); s['task1']['min_M'] *= 1 + 1e-15; json.dump(s, open('results/summary_D.json', 'w'), indent=1)
ok, dev, where, _ = reproduce('results/summary_D.json', 'd_analyze.py', loose=LOOSE)
kept = json.load(open('results/summary_D.json'))['task1']['min_M'] == s['task1']['min_M']
shutil.copyfile(tmp.name, 'results/summary_D.json'); os.unlink(tmp.name)
check('reproducibility check: a 1e-15 change in a stored summary passes and the stored file is not rewritten', ok and kept and dev > 0,
      f'(detected deviation {dev:.1e} at {where})')

# 1. identities of the scaffold on fresh states
rng = np.random.default_rng(1234)
w = {'um': 0, 'sv': 0, 'cs': -1}
for K, sig in [(1, 0.3), (2, 0.05), (2, 0.3), (3, 0.1), (3, 1.0)]:
    ws = list(rng.dirichlet([2] * K)) if K > 1 else [1.0]; m0 = np.array([rng.uniform(0, 2 * np.pi), rng.normal(0, 0.5)])
    mus = [m0 + (rng.normal(0, 0.3, 2) if k else 0) for k in range(K)]; Ss = []
    for _ in range(K):
        A = rng.normal(size=(2, 2)) * rng.uniform(0.1, 0.6); Ss.append(A @ A.T + 0.003 * np.eye(2))
    r = evaluate_full(ws, mus, Ss, sig, grid='vw')
    w['um'] = max(w['um'], abs(r['u_mean'] - (1 - r['lam_rho_direct']))); w['sv'] = max(w['sv'], abs(np.sqrt(r['abar_var']) - r['u_cv'] * r['u_mean']))
    w['cs'] = max(w['cs'], abs(r['f_cov_part']) - r['cov_cs_bound'])
check('E[1 - abar] = 1 - lambda_rho (fresh states)', w['um'] < 1e-12, f"({w['um']:.1e})", value=w['um'], tol=1e-12)
check('sqrt(Var abar) = CV(u) x E[u] (fresh states)', w['sv'] < 1e-12, f"({w['sv']:.1e})", value=w['sv'], tol=1e-12)
check('|covariance term| <= sqrt(Var abar) x Omega (Cauchy-Schwarz, fresh states)', w['cs'] <= 1e-12, f"(max |cov| - bound {w['cs']:+.1e})", value=w['cs'], tol=1e-12)

# 2. the best mid-distance state: fresh evaluation on two finer grids
EL = json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')); b = max(EL, key=lambda r: r['ratio'])
ws, mus, Ss = b['ws'], [np.array(m) for m in b['mus']], [np.array(S) for S in b['Ss']]
r15 = evaluate_full(ws, mus, Ss, b['sigma'], grid='vw', per_sd=15); r25 = evaluate_full(ws, mus, Ss, b['sigma'], grid='vw', per_sd=25)
dd = max(abs(r15['R'] - b['R']), abs(r25['R'] - b['R']))
ratio = (r25['R'] - r25['law_term']) / (1 - r25['law_term'])
check('best mid-distance state: R, distance and fraction reproduced on finer grids (per_sd 15 and 25)',
      dd < 1e-9 and abs(ratio - b['ratio']) < 1e-8 and 0.1 <= 1 - r25['law_term'] <= 0.3 and abs(r25['mass'] - 1) < 1e-9,
      f"(R {r25['R']:.6f}, fraction {ratio:.6f}, diff {dd:.1e})", value=dd, tol=1e-9)
rho = r25['f_cov_part'] / r25['cov_cs_bound']
dev = abs(ratio - (rho * r25['u_cv'] * r25['omega_rms'] * r25['u_mean'] + r25['D']) / (1 - r25['law_term']))
# limited by the quadrature identity R = f + D (integration by parts on the grid), ~1e-12 here: tolerance 1e-10
check('best state: fraction = (rho CV Omega (1 - lambda_rho) + D) / distance', dev < 1e-10, f'({dev:.1e})', value=dev, tol=1e-10)

# 3. near-top dense search and valley search: fresh re-evaluation of their extreme states
E2 = [r for r in json.load(open('results/e2_dense.json')) if r.get('feasible')]
for lab, r in [('smallest distance', min(E2, key=lambda r: r['dist'])), ('largest fraction', max(E2, key=lambda r: r['ratio']))]:
    a = evaluate_full(r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']], r['sigma'], grid='vw', per_sd=15)
    d = abs(a['R'] - r['R'])
    check(f'dense search ({lab}): fresh evaluation reproduces R and the distance target is met', d < 1e-9 and 1 - a['law_term'] <= r['target'] * (1 + 1e-9),
          f"(dist {1 - a['law_term']:.3g}, diff {d:.1e})", value=d, tol=1e-9)
V4 = json.load(open('results/e4_valley.json'))
for t in ['Pv', 'Pw']:
    r = min([x for x in V4 if x['target'] == t], key=lambda x: x[t])
    ws, mus, Ss = unpack_c1(np.array(r['x']), r['K'])
    a = evaluate_full(ws, mus, Ss, r['sigma'], grid='vw', per_sd=15); d = abs(a[t] - r[t])
    check(f'valley search: smallest {t} reproduced on a finer grid and still positive', d < 1e-9 and a[t] > 0, f'({t} = {a[t]:.5f}, diff {d:.1e})', value=d, tol=1e-9)

# 4. summary reproducible (tolerant, non-destructive) and quoted numbers
ok_rep, dev_rep, where_rep, s = reproduce('results/summary_E.json', 'e_analyze.py', loose=LOOSE)
check('summary_E.json is reproduced from the raw result files (relative 1e-9; rank correlations 0.02 absolute, fit slopes 1e-6; the stored file is left unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
t1, t2, t3, t4 = s['task1'], s['task2'], s['task3'], s['task4']; bm = t1['best_margins']
f2, f3, f4 = (lambda x: '%.2f' % x), (lambda x: '%.3f' % x), (lambda x: '%.4f' % x)
quoted = {
    'task 1: fraction 0.925 after 250 steps, 0.9515 after continuation; R 0.9854; distance 0.300 (upper edge), sigma 0.0054, K = 3':
        (f3(t1['max_ratio_250']), f4(t1['max_ratio']), f4(t1['max_R']), f3(t1['best']['dist']), f4(t1['best']['sigma']), t1['best']['K']) ==
        ('0.925', '0.9515', '0.9854', '0.300', '0.0054', 3),
    'task 1: best state valley 0.062, P_v 0.0083, P_w 0.0063, D 8e-07; 9 of 15 chains end at distance > 0.29':
        (f3(t1['best']['H_nonPD_mass']), f4(t1['best']['Pv']), f4(t1['best']['Pw']), '%.0e' % t1['best']['D'], t1['n_dist_at_upper_edge']) == ('0.062', '0.0083', '0.0063', '8e-07', 9),
    'task 1: gains in the last 100 continuation steps 0.0002 / 0.0013 / 0.0010':
        [f4(x) for x in t1['gain_last_100_of_continuation']] == ['0.0002', '0.0013', '0.0010'],
    'task 1: best state law gap 0.0049, CV 1.127, Omega 1.184, rho 0.724, rho CV Omega 0.967; scaffold certificate 1.094 (fails)':
        (f4(bm['law_gap']), f3(bm['CV']), f3(bm['Omega']), f3(bm['rho']), f3(bm['rho_CV_Omega']), f3(bm['certificate_upper_R'])) ==
        ('0.0049', '1.127', '1.184', '0.724', '0.967', '1.094'),
    'task 2: 128 restarts, 85 feasible; fraction at distance <= 3e-3 at most 0.090; envelope slope 1.26 overall, 0.98 for 3e-4-3e-3; point fit 1.10':
        (t2['n_dense_restarts'], t2['n_dense_feasible'], f3(t2['max_ratio_dist_le_3e-3_all']), f2(t2['slope_envelope_all']), f2(t2['slope_envelope_dense']),
         f2(t2['slope_points_dense'])) == (128, 85, '0.090', '1.26', '0.98', '1.10'),
    'task 3: first-moment form fails in 113 states (up to 2.12x); Cauchy-Schwarz form holds in all (max 0.99998)':
        (t3['user_form_violations'], f2(t3['user_form_max_ratio']), t3['cs_violations'], '%.5f' % t3['cs_max_ratio']) == (113, '2.12', 0, '0.99998'),
    'task 3: CV(u) 0.004-2.37 (median 0.41); Omega 0.001-2.13 (median 0.33); CV Omega < 1 in 1524 of 1536; certificate < 1 in all 1536 (max 0.9945)':
        ([f3(x) for x in t3['cv_range']], [f3(x) for x in t3['omega_range']], t3['n_cv_omega_lt_1'], t3['n_certified'], f4(t3['max_certificate'])) ==
        (['0.004', '0.406', '2.374'], ['0.001', '0.330', '2.128'], 1524, 1536, '0.9945'),
    'task 3: searched states: certificate < 1 in 86 of 100 (max 1.228); certificate >= R everywhere':
        (t3['extra_n'], t3['extra_n_certified'], f3(t3['extra_max_certificate']), t3['extra_cert_ge_R'], t3['cert_ge_R_all']) == (100, 86, '1.228', True, True),
    'task 3: Spearman CV vs kurtosis -0.43; rho median 0.17; max rho CV Omega 2.17 (all) / 0.967 (searched)':
        (f2(t3['spearman_cv_kurtosis']), f2(t3['rho_range'][1]), f2(t3['max_rho_CV_Omega_all']), f3(t3['max_rho_CV_Omega_extra'])) == ('-0.43', '0.17', '2.17', '0.967'),
    'task 4: smallest P_v 0.0027 / 0.00062 (search), smallest P_w 0.0079 / 0.0176 (search); valleys up to mass 0.34, mean depth 0.36':
        (f4(t4['all_min_Pv']), '%.5f' % t4['adv_min_Pv'], f4(t4['all_min_Pw']), f4(t4['adv_min_Pw']), f2(t4['max_valley_mass_all']), f2(t4['max_depth_mean_all'])) ==
        ('0.0027', '0.00062', '0.0079', '0.0176', '0.34', '0.36'),
    'task 4: with valley mass > 0.1 along v / w: smallest P_v 0.011, P_w 0.0079':
        (f3(t4['min_Pv_with_valley_v_gt_0.1']), f4(t4['min_Pw_with_valley_w_gt_0.1'])) == ('0.011', '0.0079'),
    'task 1: chains from the R = 0.946 state reach at most 0.885 (start 0.515); best split (K = 4) chain 0.942 after continuation':
        (f3(max(c['best_value'] for c in t1['chains'] if c['seed'].startswith('B'))), f3(min(c['start_value'] for c in t1['chains'] if c['seed'].startswith('B') and c['K'] == 3)),
         f3(max(c['ratio'] for c in t1['chains'] if c['K'] == 4))) == ('0.885', '0.515', '0.942'),
}
for k, v in quoted.items():
    check('summary number: ' + k, v)
finish()
