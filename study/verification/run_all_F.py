"""Verification for stage F (does the fraction exceed 1?).  Exit 0 iff all checks pass; failures are listed by name."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from obs_eval2 import evaluate_full, vw_grid, V, W
from f5_needle import composite

F = json.load(open('results/f1_chains.json')); b = max(F, key=lambda r: r['final']['ratio'])
A = lambda r: (r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']], r['sigma'])
ws, mus, Ss, sig = A(b)
trk = [w * np.trace(np.linalg.inv(S + sig ** 2 * np.eye(2))) for w, S in zip(ws, Ss)]; k = int(np.argmax(trk))

# 1. nested grid: exact partition of unity (mass), convergence under refinement, agreement with a fully resolved uniform grid
r1 = evaluate_full(ws, mus, Ss, sig, points=composite(ws, mus, Ss, sig, k))
r2 = evaluate_full(ws, mus, Ss, sig, points=composite(ws, mus, Ss, sig, k, fine=20, coarse=18, box=16.0, taper=4.0))
check('nested grid: mass = 1 and R converged under refinement (best stage-F state)', abs(r1['mass'] - 1) < 1e-10 and abs(r1['R'] - r2['R']) < 1e-8,
      f"(mass - 1 = {r1['mass'] - 1:.1e}, refine diff {abs(r1['R'] - r2['R']):.1e})", value=abs(r1['R'] - r2['R']), tol=1e-8)
# a wider needle (x20 along V, x sqrt20 along W, sigma x20) is resolved by the uniform grid without hitting the cap: both must agree
Rm = np.stack([V, W], 1); Svw = Rm.T @ Ss[k] @ Rm; Dg = np.diag([400.0, 20.0]); Sk = Rm @ (np.sqrt(Dg) @ Svw @ np.sqrt(Dg)) @ Rm.T
Ss3 = list(Ss); Ss3[k] = Sk; s3 = 20 * sig
_, _, n10, _ = vw_grid(ws, mus, Ss3, s3, per_sd=10); _, _, n20, _ = vw_grid(ws, mus, Ss3, s3, per_sd=20)
u10 = evaluate_full(ws, mus, Ss3, s3, grid='vw', per_sd=10); u20 = evaluate_full(ws, mus, Ss3, s3, grid='vw', per_sd=20)
nn = evaluate_full(ws, mus, Ss3, s3, points=composite(ws, mus, Ss3, s3, k))
check('nested grid = uniform grid on a state the uniform grid resolves without hitting its cap',
      max(n20) < 2401 and abs(u10['R'] - u20['R']) < 1e-9 and abs(nn['R'] - u20['R']) < 1e-8,
      f"(uniform per_sd 10/20: {n10}/{n20}, diff {abs(u10['R'] - u20['R']):.1e}; nested - uniform {nn['R'] - u20['R']:.1e})", value=abs(nn['R'] - u20['R']), tol=1e-8)
# the uniform grid is capped for the best state: its per_sd 10 / 20 comparison is not a resolution test there; the error is 1e-7
_, _, nb10, _ = vw_grid(ws, mus, Ss, sig, per_sd=10)
check('uniform grid hits its 2401-point cap for the best state (documented); its error vs the nested grid is below 1e-6',
      nb10[0] == 2401 and abs(r1['R'] - b['R_per_sd10']) < 1e-6, f"(grid {nb10}, R uniform - nested = {b['R_per_sd10'] - r1['R']:.1e})")

# 2. the decisive inequalities along the chains and the needle sequence
check('no chain state has fraction > 1 or R > kappa (uniform grid values, error < 1e-6)', all(r['final']['ratio'] < 1 - 1e-4 and r['final']['R'] < 1 - 1e-4 for r in F))
N = json.load(open('results/f5_needle.json'))
check('needle sequence: R < R(needle alone) < lambda(needle) < 1 and fraction < 1 at every t (nested grid)',
      all(r['R'] < r['R_needle_alone'] < r['lam_needle'] < 1 and r['fraction'] < 1 for r in N['rows']))
check('needle sequence: nested-grid R converged (refined grid) and mass = 1 at every t',
      all(abs(r['R_refined'] - r['R']) < 1e-8 and abs(r['mass'] - 1) < 1e-9 for r in N['rows']),
      f"(max refine diff {max(abs(r['R_refined'] - r['R']) for r in N['rows']):.1e})", value=max(abs(r['R_refined'] - r['R']) for r in N['rows']), tol=1e-8)
# fresh recomputation of the smallest needle
r0 = N['rows'][-1]; t = r0['t']; Dg = np.diag([t ** 2, t]); Skt = Rm @ (np.sqrt(Dg) @ Svw @ np.sqrt(Dg)) @ Rm.T
Sst = list(Ss); Sst[k] = Skt; rt = evaluate_full(ws, mus, Sst, sig * t, points=composite(ws, mus, Sst, sig * t, k))
check(f'smallest needle (t = {t}): fresh nested-grid evaluation reproduces R and the fraction < 1', abs(rt['R'] - r0['R']) < 1e-9 and (rt['R'] - rt['law_term']) / (1 - rt['law_term']) < 1,
      f"(R {rt['R']:.8f}, fraction {(rt['R'] - rt['law_term']) / (1 - rt['law_term']):.8f})", value=abs(rt['R'] - r0['R']), tol=1e-9)

# 3. stage-E best state re-evaluated on the nested grid (its uniform-grid checks were also capped)
E = json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')); eb = max(E, key=lambda r: r['ratio'])
we, me, Se, se = A(eb); ke = int(np.argmax([w * np.trace(np.linalg.inv(S + se ** 2 * np.eye(2))) for w, S in zip(we, Se)]))
re_ = evaluate_full(we, me, Se, se, points=composite(we, me, Se, se, ke))
check('stage-E best state on the nested grid: R agrees with the reported 0.98545 to 1e-6', abs(re_['R'] - eb['R']) < 1e-6,
      f"(nested {re_['R']:.8f}, reported {eb['R']:.8f})", value=abs(re_['R'] - eb['R']), tol=1e-6)

# 3b. the largest near-top exceedance of the best single component, recomputed on the nested grid (it is real, and R < 1)
from f6_components import Rk_max
Enear = [r for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) + json.load(open('results/e2_dense.json')) if r.get('feasible')]
rx = max(Enear, key=lambda r: r['R'] - Rk_max(r['ws'], r['mus'], r['Ss'], r['sigma']))
wx, mx_, Sx, sx = A(rx); kx = int(np.argmax([w * np.trace(np.linalg.inv(S + sx ** 2 * np.eye(2))) for w, S in zip(wx, Sx)]))
ax_ = evaluate_full(wx, mx_, Sx, sx, points=composite(wx, mx_, Sx, sx, kx)); mk = Rk_max(rx['ws'], rx['mus'], rx['Ss'], rx['sigma'])
check('near-top state where R exceeds its best component alone: confirmed on the nested grid (excess 7.8e-6) and R < 1',
      abs((ax_['R'] - mk) - 7.83e-6) < 1e-7 and ax_['R'] < 1, f"(R {ax_['R']:.9f}, best component {mk:.9f})")

# 4. summary reproducible and quoted numbers
# plateau fits (curve_fit, iterative and nonlinear) are compared with separate, looser tolerances: their error estimates a_err come
# from the fit covariance and moved by a relative 1e-3 on another machine; the plateau values themselves are not used in any conclusion
FIT_TOL = {'a_err': 0.1, '/plateau/': 1e-2, '/rcvo_plateau/': 1e-2}
ok_rep, dev_rep, where_rep, s = reproduce('results/summary_F.json', 'f_analyze.py', loose=FIT_TOL)
check('summary_F.json is reproduced from the raw result files (relative 1e-9; fit outputs: a_err 10 %, other fit values 1 %; stored file unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
f2, f3, f4, f6 = (lambda x: '%.2f' % x), (lambda x: '%.3f' % x), (lambda x: '%.4f' % x), (lambda x: '%.6f' % x)
nd = s['needle']; bf = s['best']; ch = {c['name']: c for c in s['chains']}
quoted = {
    'chains: best fraction 0.9838 (R 0.9928, distance 0.445, sigma 0.0030, K = 3); none above 1':
        (f4(s['max_ratio']), f4(s['max_R']), f3(bf['dist']), f4(bf['sigma']), s['any_ratio_gt_1']) == ('0.9838', '0.9928', '0.445', '0.0030', False),
    'best chain factors: rho 0.704, CV 0.895, Omega 1.568, rho CV Omega 0.988, law gap 0.0019, valley 0.068':
        (f3(bf['rho']), f3(bf['CV']), f3(bf['Omega']), f3(bf['rhoCVOmega']), f4(bf['law_gap']), f3(bf['valley'])) == ('0.704', '0.895', '1.568', '0.988', '0.0019', '0.068'),
    'best chain from step 0 to 2400: distance 0.300 -> 0.445, CV x0.79, Omega x1.32, rho x0.97, law gap x0.39':
        (f3(s['best_factor_change']['dist']['start']), f3(s['best_factor_change']['dist']['end']), f2(1 + s['best_factor_change']['CV']['rel']),
         f2(1 + s['best_factor_change']['Omega']['rel']), f2(1 + s['best_factor_change']['rho']['rel']), f2(1 + s['best_factor_change']['law_gap']['rel'])) ==
        ('0.300', '0.445', '0.79', '1.32', '0.97', '0.39'),
    'needle sequence: fraction 0.983816 (t = 1) ... 0.999847 (t = 0.01); slope of 1 - fraction vs t 1.0x; all below 1':
        (f6(nd['rows'][0]['fraction']), f6(nd['rows'][-1]['fraction']), f2(nd['slope_one_minus_fraction_vs_t'])[:3], nd['all_fraction_lt_1'], nd['all_R_lt_Rneedle_lt_lamneedle_lt_1']) ==
        ('0.983816', '0.999847', '1.0', True, True),
    'needle check at t = 1: uniform 0.9928021779 vs nested 0.9928022918': ('%.10f' % nd['check_t1']['R_uniform'], '%.10f' % nd['check_t1']['R_nested']) == ('0.9928021779', '0.9928022918'),
    'chains: last-400-step gains 0.0008-0.0025; plateau fits 0.99-1.21 (undetermined); final distances 0.34-0.54':
        (f4(min(p['gain_last_400'] for p in s['plateau'].values())), f4(max(p['gain_last_400'] for p in s['plateau'].values())),
         f2(min(p['a'] for p in s['plateau'].values())), f2(max(p['a'] for p in s['plateau'].values())),
         f2(min(c['dist'] for c in s['chains'])), f2(max(c['dist'] for c in s['chains']))) == ('0.0008', '0.0025', '0.99', '1.21', '0.34', '0.54'),
    'needle: R(needle alone) - R(mixture) 6.1e-4 at t = 1 and 6.2e-8 at t = 0.01; best chain D 8.1e-7 -> 1.5e-7':
        ('%.1e' % (nd['rows'][0]['R_needle_alone'] - nd['rows'][0]['R']), '%.1e' % (nd['rows'][-1]['R_needle_alone'] - nd['rows'][-1]['R']),
         '%.1e' % s['best_factor_change']['D']['start'], '%.1e' % s['best_factor_change']['D']['end']) == ('6.1e-04', '6.2e-08', '8.1e-07', '1.5e-07'),
}
if 'rcvo' in s:
    quoted['constrained search (law gap <= 0.003): rho CV Omega 0.9908 at most (below 1); needle sequence 0.99988 at t = 0.01'] = \
        (f4(s['rcvo_max']), '%.5f' % s['needle']['max_rhoCVOmega']) == ('0.9908', '0.99988')
C6 = json.load(open('results/f6_components.json'))
quoted['mixture vs best component alone: R exceeds it in 382 of 1671 states (max 0.264); with best component R >= 0.95 in 2 of 126 (max 7.8e-6)'] = \
    (C6['n'], C6['n_R_gt_maxRk'], f3(C6['max_R_minus_maxRk']), C6['bins']['0.95-1.0']['n'], C6['bins']['0.95-1.0']['n_exceed'], '%.1e' % C6['bins']['0.95-1.0']['max_R_minus_maxRk']) == \
    (1671, 382, '0.264', 126, 2, '7.8e-06')
for kq, v in quoted.items():
    check('summary number: ' + kq, v)
finish()
