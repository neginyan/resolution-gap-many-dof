"""Verification for stage D (flow-coordinate form of Conjecture 1, tightened bound B', states near the top). Exit 0 iff all pass."""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
from obs_eval2 import evaluate_full, V, W
from d3_peak import unpack as unpack3

LOOSE = {'spearman': ('abs', 0.02), 'slope': 1e-6}  # rank correlations of tied data (CPU-dependent order of values at rounding level): absolute 0.02; log-log fit slopes: 1e-6
from vtools import check, finish, reproduce

rng = np.random.default_rng(99)
def rnd(K, kind):
    ws = list(rng.dirichlet([2] * K)) if K > 1 else [1.0]
    m0 = np.array([rng.uniform(0, 2 * np.pi) if kind == 'pendulum' else rng.normal(0, 0.8), rng.normal(0, 0.8)])
    mus = [m0 + (rng.normal(0, 0.3, 2) if k else 0) for k in range(K)]; Ss = []
    for _ in range(K):
        A = rng.normal(size=(2, 2)) * rng.uniform(0.1, 0.5); Ss.append(A @ A.T + 0.005 * np.eye(2))
    return ws, mus, Ss

# 1. the grid aligned with V, W reproduces the (q, p) grid; identities M / tr F = 1 - R and M_loc / tr F = kappa_loc - R
wg, wM, wB, wnv = 0, 0, -1, 0
for kind in ['pendulum', 'quartic', 'double_well']:
    for K, sig in [(1, 0.3), (2, 0.1), (3, 0.3), (2, 1.0)]:
        ws, mus, Ss = rnd(K, kind)
        a = evaluate_full(ws, mus, Ss, sig, kind=kind, n=901); b = evaluate_full(ws, mus, Ss, sig, kind=kind, grid='vw')
        wg = max(wg, abs(a['R'] - b['R']), abs(a['f'] - b['f']), abs(a['D'] - b['D']))
        if kind == 'pendulum':
            wM = max(wM, abs(b['M_over_trF'] - (1 - b['R'])), abs(b['Pv'] + b['Pw'] - b['D'] - (1 - b['R'])))
            wB = max(wB, b['f'] - b['B_prime'])
            if b['H_nonPD_mass'] < 1e-12: wnv = max(wnv, abs(b['B_prime'] - b['f']))
        else:
            wM = max(wM, abs(b['ML_over_trF'] - (b['kappa_loc'] - b['R'])))
check('V/W grid reproduces the (q, p) grid for R, f, D (fresh states, three flows)', wg < 1e-8, f'({wg:.1e})', value=wg, tol=1e-8)
check('M / tr F = P_v + P_w - D = 1 - R (pendulum) and M_loc / tr F = kappa_loc - R (other flows)', wM < 1e-10, f'({wM:.1e})', value=wM, tol=1e-10)
check("pendulum: f <= B' (fresh states)", wB <= 1e-10, f'(max f - B\' {wB:+.1e})', value=wB, tol=1e-10)
check("pendulum without valley: B' = f exactly", wnv < 1e-10, f'({wnv:.1e})', value=wnv, tol=1e-10)

# 2. the algebraic counterexample to B' when abar < 0 (quartic / double well), and the pointwise inequality for abar >= 0
abar = -0.5; Hvv, Hww = 1.0, 5.0                                  # H diagonal in (v, w): positive definite
lhs = abar * (Hvv - Hww); rhs = abs(abar) * ((Hvv + Hww) - 2 * max(Hww, 0)); rhs_gen = abs(abar) * ((Hvv + Hww) - 2 * max(Hvv, 0))
check("B' pointwise counterexample for abar < 0 (abar = -0.5, H_vv = 1, H_ww = 5): local f = 2.0 > local B' = -2.0; corrected version holds",
      lhs > rhs and lhs <= rhs_gen + 1e-15, f'(f {lhs:+.2f}, B\' {rhs:+.2f}, B\'_gen {rhs_gen:+.2f})')
worst = 0
for _ in range(20000):                                            # random symmetric H, abar in [0, 1]
    A = rng.normal(size=(2, 2)); H = (A + A.T) / 2; ab = rng.uniform(0, 1)
    hvv, hww = V @ H @ V, W @ H @ W; tra = np.abs(np.linalg.eigvalsh(H)).sum()
    worst = max(worst, ab * (hvv - hww) - ab * (tra - 2 * max(hww, 0)))
check("pendulum (0 <= abar): abar (H_vv - H_ww) <= abar (tr|H| - 2 (H_ww)_+) for 20000 random symmetric H", worst <= 1e-12, f'(max {worst:+.1e})', value=worst, tol=1e-12)

# 3. states near the top: fresh re-evaluation of the closest one and of the largest-ratio one
SW = json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json'))
rows = [r for r in SW if r.get('feasible')]
for lab, r in [('closest to the top', min(rows, key=lambda r: r['dist'])), ('largest fraction of the room', max(rows, key=lambda r: r['ratio']))]:
    ws, mus, Ss = r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']]
    a = evaluate_full(ws, mus, Ss, r['sigma'], grid='vw', per_sd=15)
    check(f"near-top state ({lab}): fresh evaluation at a finer V/W grid reproduces R, distance and excess",
          abs(a['R'] - r['R']) < 1e-9 and abs((1 - a['law_term']) - r['dist']) < 1e-9 and r['dist'] <= r['target_dist'] and abs(a['mass'] - 1) < 1e-9,
          f"(dist {r['dist']:.3g}, excess {r['excess']:.3g}, diff {abs(a['R'] - r['R']):.1e})", value=abs(a['R'] - r['R']), tol=1e-9)

# 4. summary reproducible and quoted numbers
ok_rep, dev_rep, where_rep, s = reproduce('results/summary_D.json', 'd_analyze.py', loose=LOOSE)
t1, t2, t3 = s['task1'], s['task2'], s['task3']
check('summary_D.json is reproduced from the raw result files (relative 1e-9; rank correlations 0.02 absolute, fit slopes 1e-6; the stored file is left unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
f2, f3, f4 = (lambda x: '%.2f' % x), (lambda x: '%.3f' % x), (lambda x: '%.4f' % x)
c0 = t1['closest'][0]
quoted = {
    'all 2352 states: V/W grid vs earlier (q, p) grid < 1e-10; M identity < 1e-10': t1['grid_check_max_abs_R_vw_minus_qp'] < 1e-10 and t1['identity_M_eq_1_minus_R_maxerr'] < 1e-10 and t1['identity_ML_eq_kloc_minus_R_maxerr'] < 1e-10,
    'pendulum: min M/trF 0.0538; P_v >= 0.0027, P_w >= 0.0079 (never negative); D > 0 in 580 (max 0.068)':
        (f4(t1['min_M']), f4(t1['min_Pv']), f4(t1['min_Pw']), t1['n_Pv_neg'], t1['n_Pw_neg'], t1['n_D_pos'], f3(t1['max_D'])) == ('0.0538', '0.0027', '0.0079', 0, 0, 580, '0.068'),
    'negative parts: P_v down to -0.166, P_w down to -0.300; Spearman with valley mass along v / w 0.98 / 0.91; P_v < P_w in 977':
        (f3(t1['min_Pv_neg_part']), f3(t1['min_Pw_neg_part']), f2(t1['spearman_Pvneg_massHvvneg']), f2(t1['spearman_Pwneg_massHwwneg']), t1['n_Pv_lt_Pw']) == ('-0.166', '-0.300', '0.98', '0.91', 977),
    'closest state: R 0.9462, P_v 0.0275, P_w 0.0264, D 0.0000, valley along v / w 0.035 / 0.040':
        (f4(c0['R']), f4(c0['Pv']), f4(c0['Pw']), f4(abs(c0['D'])), f3(c0['mass_Hvv_neg']), f3(c0['mass_Hww_neg'])) == ('0.9462', '0.0275', '0.0264', '0.0000', '0.035', '0.040'),
    'other flows: R <= kappa_loc in all 816; min M_loc/trF 0.139': (t1['other_n_R_gt_kappa_loc'], f3(t1['other_min_ML'])) == (0, '0.139'),
    "B': exact for the pendulum; B' = f without valley (261 states); B' > 1 only for the largest-R state (1.0067); 20 of the 21 B > 1 states have B' <= 1":
        (t2['pend_f_le_Bp_all'], t2['pend_pointwise_violation_max'], t2['n_no_valley'], t2['no_valley_max_abs_Bp_minus_f'] < 1e-10, t2['n_Bp_gt_1'],
         t2['n_Bp_plus_D_gt_1'], f4(t2['max_Bp']), t2['n_Bp_le_1_among_B_gt_1']) == (True, 0.0, 261, True, 1, 1, '1.0067', 20),
    "B' - f: median 0.0012, valley states median 0.0047, max 0.178; largest-R state 0.061 (valley 0.150)":
        (f4(t2['Bp_minus_f_median']), f4(t2['valley_median_Bp_minus_f']), f3(t2['max_Bp_minus_f']), f3(t2['top_Bp_minus_f']), f3(t2['top_valley_mass'])) == ('0.0012', '0.0047', '0.178', '0.061', '0.150'),
    "other flows: B' pointwise violated in 725 states, f > B' in 318; corrected B'_gen holds in all":
        (t2['other_n_states_with_pointwise_violation'], t2['other_n_f_gt_Bp'], t2['other_f_le_Bgen_all']) == (725, 318, True),
    'task 3 (fixed sigma, distance < 0.3): max ratio 0.403; sigma = 1 never below 0.3':
        (f3(t3['ratio_part_max_ratio']), any(r['sigma_fixed'] == 1.0 and r['feasible'] for r in t3['ratio_part'])) == ('0.403', False),
    'sweep: 9 unseeded searches did not reach their target; closest distance 0.0004; resolution < 1e-11; mass 1':
        (t3['n_unseeded_infeasible'], f4(t3['min_dist_reached']), t3['sweep_resolution'] < 1e-11, t3['sweep_mass_dev'] < 1e-9) == (9, '0.0004', True, True),
    'sweep: slope 1.43 (all), 1.17 (distance <= 0.05); last two points 0.27':
        (f2(t3['slope_excess']), f2(t3['slope_excess_small']), f2(t3['local_slope_last_two'])) == ('1.43', '1.17', '0.27'),
    'sweep: Var(abar) slope 2.35, focus mismatch slope 0.23':
        (f2(t3['slope_abar_var']), f2(t3['slope_mismatch'])) == ('2.35', '0.23'),
    'sweep: max fraction 0.830 (distance 0.288, sigma 0.0063, K = 3); at distance <= 0.02 at most 0.082 (11 states)':
        (f3(t3['max_ratio_sweep']), f3(t3['max_ratio_sweep_state']['dist']), f4(t3['max_ratio_sweep_state']['sigma']), t3['max_ratio_sweep_state']['K'],
         f3(t3['max_ratio_dist_le_0.02']), t3['n_dist_le_0.02']) == ('0.830', '0.288', '0.0063', 3, '0.082', 11),
    'sweep: excess / (sqrt Var(abar) x mismatch) between 0.34 and 0.69 for the best states':
        (f2(min(t3['cs_ratio'])), f2(max(t3['cs_ratio']))) == ('0.34', '0.69'),
}
for k, v in quoted.items():
    check('summary number: ' + k, v)
finish()
