"""Verification for stage C (proposals 1-4). Exit code 0 iff all checks pass."""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
from obs_eval2 import evaluate_full, FLOWS, V, W
from c1_adversarial import unpack
import fast_eval as fe

LOOSE = {'spearman': ('abs', 0.02), 'slope': 1e-6}  # rank correlations of tied data (CPU-dependent order of values at rounding level): absolute 0.02; log-log fit slopes: 1e-6
from vtools import check, finish, reproduce

rng = np.random.default_rng(2027)
def rnd(K, spread=0.3):
    ws = list(rng.dirichlet([2] * K)) if K > 1 else [1.0]
    m0 = np.array([rng.normal(0, 0.8), rng.normal(0, 0.8)])
    mus = [m0 + (rng.normal(0, spread, 2) if k else 0) for k in range(K)]
    Ss = []
    for _ in range(K):
        A = rng.normal(size=(2, 2)) * rng.uniform(0.1, 0.5); Ss.append(A @ A.T + 0.005 * np.eye(2))
    return ws, mus, Ss

# 1. quartic: evaluate_full reproduces fast_eval.py (independent reference code) on fresh states
worst = 0
for K in [1, 2, 3]:
    for sig in [0.1, 0.5]:
        ws, mus, Ss = rnd(K); r = evaluate_full(ws, mus, Ss, sig, kind='quartic', n=701)
        _, trF, Sig = fe.dSdt(ws, mus, Ss, sig, 0.0, 'quartic', n=701)
        worst = max(worst, abs(r['R'] + Sig / (sig ** 2 * trF)))
check('quartic: R of evaluate_full equals fast_eval.py (fresh K = 1, 2, 3 states)', worst < 1e-10, f'({worst:.1e})', value=worst, tol=1e-10)

# 2. identities for all three flows on fresh overlapping states (incl. the new D split and the field integrals)
w = {'fD': 0, 'A': 0, 'split': 0, 'tw': 0, 'fields': 0, 'law': -1}
for kind in ['pendulum', 'quartic', 'double_well']:
    for K, sig in [(2, 0.1), (3, 0.3), (2, 1.0)]:
        ws, mus, Ss = rnd(K); r = evaluate_full(ws, mus, Ss, sig, kind=kind, n=601, return_fields=True); F = r['fields']
        dA = (F['q'][1] - F['q'][0]) * (F['p'][1] - F['p'][0])
        w['fD'] = max(w['fD'], abs(r['f'] + r['D'] - r['R'])); w['A'] = max(w['A'], abs(r['A'] * r['R_sep'] + r['R_switch'] - r['R']))
        w['split'] = max(w['split'], abs(r['D_within'] + r['D_between'] - r['D'])); w['tw'] = max(w['tw'], r['tweedie_err'])
        w['fields'] = max(w['fields'], abs(np.nansum(F['law_density']) * dA - r['law_term']), abs(np.nansum(F['cov_density']) * dA - (r['f'] - r['law_term'])),
                          abs(np.nansum(F['D_density']) * dA - r['D']))
        w['law'] = max(w['law'], r['law_term'] - r['lam_rho_direct'])
check('R = f + D and R = A R_sep + R_switch for pendulum, quartic, double well', max(w['fD'], w['A']) < 1e-10, f"({max(w['fD'], w['A']):.1e})", value=max(w['fD'], w['A']), tol=1e-10)
check('D = D_within + D_between (exact split)', w['split'] < 1e-10, f"({w['split']:.1e})", value=w['split'], tol=1e-10)
check('second-order Tweedie identity on the grid (all flows)', w['tw'] < 1e-10, f"({w['tw']:.1e})", value=w['tw'], tol=1e-10)
check('maps integrate to law term, covariance term and D', w['fields'] < 1e-10, f"({w['fields']:.1e})", value=w['fields'], tol=1e-10)
check('law term <= lambda_max(Sym E_rho Df) (all flows)', w['law'] <= 1e-12, f"(max {w['law']:+.3f})", value=w['law'], tol=1e-12)

# 3. double well: independent brute force (x-quadrature, finite differences) for an overlapping mixture
def brute(ws, mus, Ss, sig, g, gp, ny=241, nx=181):
    lo = np.min([m - 6 * np.sqrt(np.diag(S) + sig ** 2) for m, S in zip(mus, Ss)], 0); hi = np.max([m + 6 * np.sqrt(np.diag(S) + sig ** 2) for m, S in zip(mus, Ss)], 0)
    yq, yp = np.linspace(lo[0], hi[0], ny), np.linspace(lo[1], hi[1], ny)
    xlo = np.min([m - 7 * np.sqrt(np.diag(S)) for m, S in zip(mus, Ss)], 0); xhi = np.max([m + 7 * np.sqrt(np.diag(S)) for m, S in zip(mus, Ss)], 0)
    xq, xp = np.linspace(xlo[0], xhi[0], nx), np.linspace(xlo[1], xhi[1], nx)
    X = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1).reshape(-1, 2); dx = (xq[1] - xq[0]) * (xp[1] - xp[0])
    rho = sum(wk * np.exp(-0.5 * np.einsum('ni,ij,nj->n', X - m, np.linalg.inv(S), X - m)) / (2 * np.pi * np.sqrt(np.linalg.det(S))) for wk, m, S in zip(ws, mus, Ss)) * dx
    fX = np.stack([X[:, 1], g(X[:, 0])], -1); aX = (1 + gp(X[:, 0])) / 2
    Y = np.stack(np.meshgrid(yq, yp, indexing='ij'), -1).reshape(-1, 2)
    p = np.zeros(len(Y)); mf = np.zeros((len(Y), 2)); ma = np.zeros(len(Y))
    for c in range(0, len(Y), 2000):
        y = Y[c:c + 2000]; Kr = np.exp(-((y[:, None, :] - X[None]) ** 2).sum(-1) / (2 * sig ** 2)) * rho[None]
        p[c:c + 2000] = Kr.sum(1); mf[c:c + 2000] = Kr @ fX; ma[c:c + 2000] = Kr @ aX
    p = p.reshape(ny, ny) / (2 * np.pi * sig ** 2); mf = mf.reshape(ny, ny, 2) / (p[..., None] * 2 * np.pi * sig ** 2); ma = ma.reshape(ny, ny) / (p * 2 * np.pi * sig ** 2)
    hq, hp = yq[1] - yq[0], yp[1] - yp[0]; L = np.log(p)
    sq, sp = np.gradient(L, hq, axis=0), np.gradient(L, hp, axis=1)
    Hqp = -np.gradient(sq, hp, axis=1); Hqq = -np.gradient(sq, hq, axis=0); Hpp = -np.gradient(sp, hp, axis=1)
    I = slice(3, -3); wg = (p * hq * hp)[I, I]
    sq, sp, Hqp, Hqq, Hpp, mf, ma = sq[I, I], sp[I, I], Hqp[I, I], Hqq[I, I], Hpp[I, I], mf[I, I], ma[I, I]
    trF = np.sum(wg * (sq ** 2 + sp ** 2)); R = np.sum(wg * (sq * mf[..., 0] + sp * mf[..., 1])) / (sig ** 2 * trF)
    return R, np.sum(wg * 2 * ma * Hqp) / np.sum(wg * (Hqq + Hpp))
bs = ([0.5, 0.5], [np.array([0.8, 0.3]), np.array([1.2, -0.3])],
      [0.1 ** 2 * np.outer(V, V) + 0.5 ** 2 * np.outer(W, W), 0.4 ** 2 * np.outer(V, V) + 0.3 ** 2 * np.outer(W, W)], 0.7)
rb = brute(*bs, g=lambda q: q - q ** 3, gp=lambda q: 1 - 3 * q ** 2); rf = evaluate_full(*bs, kind='double_well')
dd = (abs(rb[0] - rf['R']), abs(rb[1] - rf['f']), abs((rb[0] - rb[1]) - rf['D']))
check('double well: brute force reproduces R, (f) and D of an overlapping mixture', max(dd) < 1e-4 and abs(rf['D']) > 10 * dd[2], value=max(dd), tol=1e-4, info=
      f'(diffs {dd[0]:.1e}, {dd[1]:.1e}, D {dd[2]:.1e}; D = {rf["D"]:+.4f})')

# 4. generalised single-Gaussian remainder formula R - e = -E_rho[g'''] K_qq K_qp / tr F on fresh states, all three flows
g3 = {'pendulum': lambda m, v: np.cos(m) * np.exp(-v / 2), 'quartic': lambda m, v: -6.0, 'double_well': lambda m, v: -6.0}
worst = 0
for kind in g3:
    for _ in range(6):
        ws, mus, Ss = rnd(1); sig = [0.05, 0.3, 1.0][_ % 3]
        r = evaluate_full(ws, mus, Ss, sig, kind=kind); S = Ss[0]; F = np.linalg.inv(S + sig ** 2 * np.eye(2)); K = S @ F
        worst = max(worst, abs(r['R'] - r['e'] + g3[kind](mus[0][0], S[0, 0]) * K[0, 0] * K[0, 1] / np.trace(F)))
check("single Gaussian: R - e = -E_rho[g'''] K_qq K_qp / tr F (fresh states, three flows)", worst < 1e-9, f'({worst:.1e})', value=worst, tol=1e-9)

# 5. proposal 1: the largest-R state, fresh full-resolution evaluation and 1.5x grid
T = json.load(open('results/c1_table.json')); b = max(T, key=lambda r: r['R'])
ST = json.load(open('results/c1_adversarial.json')) + json.load(open('results/c1_adversarial_room.json'))
src = next(c for c in ST if c['K'] == b['K'] and c['sigma'] == b['sigma'] and c['target'] == b['target'])
ws, mus, Ss = unpack(np.array(src['x']), src['K'])
r = evaluate_full(ws, mus, Ss, src['sigma'])
from obs_eval import evaluate                                     # lighter evaluator for the finer grid (memory)
r2 = evaluate(ws, mus, Ss, src['sigma'], n=int(1.25 * r['n']))
check('largest-R adversarial state: fresh evaluation equals the table, stable on a 1.25x grid, and R < kappa = 1',
      abs(r['R'] - b['R']) < 1e-10 and abs(r2['R'] - r['R']) < 1e-8 and r['R'] < 1, value=max(abs(r['R'] - b['R']) / 1e-10, abs(r2['R'] - r['R']) / 1e-8), tol=1.0,
      info=f"(R = {r['R']:.6f}, vs table {abs(r['R'] - b['R']):.1e}, 1.25x diff {abs(r2['R'] - r['R']):.1e})")

# 6. summary_C reproducible, quoted numbers
ok_rep, dev_rep, where_rep, s = reproduce('results/summary_C.json', 'summarize_C.py', loose=LOOSE)
check('summary_C.json is reproduced from the raw result files (relative 1e-9; rank correlations 0.02 absolute, fit slopes 1e-6; the stored file is left unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
f2, f3, f4 = (lambda x: '%.2f' % x), (lambda x: '%.3f' % x), (lambda x: '%.4f' % x)
c1, c2, c3, c4 = s['c1'], s['c2'], s['c3'], s['c4']; bb = c1['best']
quoted = {
    'C1: 24 states; max R/kappa 0.9462 (K=3, sigma 0.03): law 0.889, cov +0.057, D 0.0000, valley 0.150, overlap 0.65':
        (c1['n_states'], f4(bb['R']), bb['K'], bb['sigma'], f3(bb['law_term']), f3(bb['cov']), f4(abs(bb['D'])), f3(bb['H_nonPD_mass']), f2(bb['overlap_max'])) ==
        (24, '0.9462', 3, 0.03, '0.889', '0.057', '0.0000', '0.150', '0.65'),
    'C1: max R/kappa below the stage-A single-Gaussian value 0.9495': c1['max_R_over_kappa'] < c1['stageA_single_max_R'] and f4(c1['stageA_single_max_R']) == '0.9495',
    'C1: max excess 0.603, max room 0.714 (K=2, sigma 0.03, overlap 0, valley 0)':
        (f3(c1['max_excess']), f3(c1['max_room']), c1['room_best']['K'], c1['room_best']['sigma'], f3(c1['room_best']['overlap_max']), f3(c1['room_best']['H_nonPD_mass'])) ==
        ('0.603', '0.714', 2, 0.03, '0.000', '0.000'),
    'C1: all states resolution-stable (1.5x) and field grids agree (< 1e-13)': c1['all_resolution_ok'] and c1['max_fieldgrid_dev'] < 1e-13,
    'C1: covariance inside valleys: sigma 0.03 -0.137..+0.017, 0.1 -0.037..0, 0.3 -0.003..+0.071, 1.0 +0.047..+0.110':
        [[f3(x) for x in c1['cov_in_valley_by_sigma'][k]] for k in ['0.03', '0.1', '0.3', '1.0']] ==
        [['-0.137', '0.017'], ['-0.037', '-0.000'], ['-0.003', '0.071'], ['0.047', '0.110']],
    'C3: A: cov outer half +0.099 / inner -0.005; valley share -0.069; B: +0.086 / +0.014, valley -0.056; C: valley -0.137':
        (f3(c3['A']['cov_outer_half']), f3(c3['A']['cov_inner_half']), f3(c3['A']['cov_in_valley']), f3(c3['B']['cov_outer_half']),
         f3(c3['B']['cov_inner_half']), f3(c3['B']['cov_in_valley']), f3(c3['C']['cov_in_valley'])) ==
        ('0.099', '-0.005', '-0.069', '0.086', '0.014', '-0.056', '-0.137'),
    'C3: p_Y-weighted correlation of local stretch and focus excess: A 0.33, B 0.36, C 0.54':
        (f2(c3['A']['corr_a_aniso']), f2(c3['B']['corr_a_aniso']), f2(c3['C']['corr_a_aniso'])) == ('0.33', '0.36', '0.54'),
    'C2: 1152 states (867 with |D| > 1e-8); split exact; Spearman |D| vs S_D 0.90, vs product 0.65':
        (c2['n'], c2['n_nonzero'], c2['split_max_err'] < 1e-10, f2(c2['spearman_absD_SD']), f2(c2['spearman_absD_rvgap'])) == (1152, 867, True, '0.90', '0.65'),
    'C2: |D|/S_D max 0.72 / 0.69 / 0.66 / 2.20 for sigma 0.03 / 0.1 / 0.3 / 1; 6 cases above 1, all D_between-dominated':
        ([f2(c2['ratio_D_SD_max_by_sigma'][k]) for k in ['0.03', '0.1', '0.3', '1.0']], c2['n_ratio_gt_1'], c2['failures_between_dominated']) ==
        (['0.72', '0.69', '0.66', '2.20'], 6, True),
    'C2: within share median 0.32; sign(D) = sign(D_between) in 74%; |D|/product up to 938':
        (f2(c2['within_share_median']), '%.0f' % (100 * c2['sign_D_eq_sign_between']), '%.0f' % c2['ratio_D_rvgap_max']) == ('0.32', '74', '938'),
    'C4: single law and remainder formula hold to < 1e-10 (300 cases)': c4['single_n'] == 300 and c4['single_law_max_err'] < 1e-10 and c4['single_remainder_formula_max_err'] < 1e-10,
    'C4: quartic and double-well covariance term and D identical in the family (< 1e-13)': c4['quartic_eq_doublewell_cov_maxdiff'] < 1e-13,
    'C4: family covariance: pendulum +0.0006/+0.036/+0.095/+0.099; quartic l1=0.5: -0.004/-0.026/-0.022/+0.085, l1=1: -0.005/-0.115/-0.265/-0.188':
        ([f3(c4['pendulum_family_cov_l19'][k]) for k in ['1.0', '2.0', '4.44', '8.0']] ==
         ['0.001', '0.036', '0.095', '0.099'] and
         [f3(x['cov']) for x in c4['family'] if x['kind'] == 'quartic' and x['l1'] == 0.5] == ['-0.004', '-0.026', '-0.022', '0.085'] and
         [f3(x['cov']) for x in c4['family'] if x['kind'] == 'quartic' and x['l1'] == 1.0] == ['-0.005', '-0.115', '-0.265', '-0.188']),
    'C4: 800 mixtures: identities < 1e-10, law <= lam_rho, R > lam_rho in 32, R > lam_comp in 0; max R/lam_comp 0.905 / 0.770':
        (c4['mix_n'], c4['mix_identity_err'] < 1e-10, c4['mix_law_le_lam_rho'], c4['mix_n_R_gt_lam_rho'], c4['mix_n_R_gt_lam_comp'],
         f3(c4['mix_by_kind']['quartic']['max_R_over_lam_comp']), f3(c4['mix_by_kind']['double_well']['max_R_over_lam_comp'])) == (800, True, True, 32, 0, '0.905', '0.770'),
    'C4: median |R-f| 0.0022 vs |R-e| 0.0086 where D is non-zero':
        (f4(c4['mix_median_abs_R_f_vs_e_overlap'][0]), f4(c4['mix_median_abs_R_f_vs_e_overlap'][1])) == ('0.0022', '0.0086'),
}
for k, v in quoted.items():
    check('summary number: ' + k, v)
finish()
