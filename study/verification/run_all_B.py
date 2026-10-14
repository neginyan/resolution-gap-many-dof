"""Verification for stage B (measurements 1 and 2). Exit code 0 iff all checks pass.
Recomputes the new identities on fresh states, checks them against an independent brute-force computation,
and checks every number quoted in the stage-B summary."""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
from obs_eval import evaluate
from obs_eval2 import evaluate_full
from mixtures import references, random_state
from m1_mechanism import state

LOOSE = {'spearman': ('abs', 0.02), 'slope': 1e-6}  # rank correlations of tied data (CPU-dependent order of values at rounding level): absolute 0.02; log-log fit slopes: 1e-6
from vtools import check, finish, reproduce

rng = np.random.default_rng(77)
fresh = []
for _ in range(6):
    ws, mus, Ss = random_state(rng)
    mus[1] = mus[0] + rng.normal(0, 0.3, 2)                    # force overlap
    fresh.append((ws, mus, Ss, [0.05, 0.3, 1.0][_ % 3]))

# 1. stage-B evaluator agrees with the stage-A evaluator (R and e)
worst = max(max(abs(evaluate_full(*s[:3], s[3])['R'] - evaluate(*s[:3], s[3])['R']),
                abs(evaluate_full(*s[:3], s[3])['e'] - evaluate(*s[:3], s[3])['cand_e'])) for s in fresh)
check('evaluate_full reproduces R and (e) of the stage-A evaluator (6 fresh overlapping mixtures)', worst < 1e-12, f'({worst:.1e})', value=worst, tol=1e-12)

# 2. exact identities on fresh states
err = {'A': 0, 'fD': 0, 'tw': 0, 'trF': 0, 'law': 0, 'sep': 0, 'lawlam': -1}
for ws, mus, Ss, sig in fresh:
    r = evaluate_full(ws, mus, Ss, sig); lam_rho, lam_comp, R_sep, ov = references(ws, mus, Ss, sig)
    err['A'] = max(err['A'], abs(r['A'] * r['R_sep'] + r['R_switch'] - r['R']))
    err['fD'] = max(err['fD'], abs(r['f'] + r['D'] - r['R']))
    err['tw'] = max(err['tw'], r['tweedie_err'])
    err['trF'] = max(err['trF'], abs(r['trF_components'] - r['fisher_loss'] - r['trF']) / r['trF'])
    err['law'] = max(err['law'], abs(r['law_term'] - r['f_mean_part']))
    err['sep'] = max(err['sep'], abs(r['R_sep'] - R_sep))
    err['lawlam'] = max(err['lawlam'], r['law_term'] - lam_rho)
check('R = A R_sep + R_switch (fresh states)', err['A'] < 1e-10, f"({err['A']:.1e})", value=err['A'], tol=1e-10)
check('R = f + D (posterior Stein defect, fresh states)', err['fD'] < 1e-10, f"({err['fD']:.1e})", value=err['fD'], tol=1e-10)
check('second-order Tweedie Cov(X|y) = s2 I - s4 H(y) on the grid', err['tw'] < 1e-10, f"({err['tw']:.1e})", value=err['tw'], tol=1e-10)
check('tr F = sum w_k tr F_k - Fisher loss', err['trF'] < 1e-10, f"({err['trF']:.1e})", value=err['trF'], tol=1e-10)
check('law term = mean part of f, and R_sep matches the closed form', err['law'] < 1e-10 and err['sep'] < 1e-10, value=max(err['law'], err['sep']), tol=1e-10)
check('law term <= lambda_max(Sym E_rho Df)', err['lawlam'] <= 1e-12, f"(max law - lam_rho {err['lawlam']:+.3f})", value=err['lawlam'], tol=1e-12)

# 3. single Gaussians: f = R, D = 0; closed form of R - e and the curvature bound (fresh states)
w1, w2, w3 = 0, 0, 0
for _ in range(12):
    A = rng.normal(size=(2, 2)) * rng.uniform(0.1, 0.8); S = A @ A.T + 0.005 * np.eye(2)
    mu = np.array([rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1)]); sig = [0.05, 0.3, 1.0, 2.0][_ % 4]
    r = evaluate_full([1.0], [mu], [S], sig)
    T = S + sig ** 2 * np.eye(2); K = S @ np.linalg.inv(T); trF = np.trace(np.linalg.inv(T))
    pred = -np.cos(mu[0]) * np.exp(-S[0, 0] / 2) * K[0, 0] * K[0, 1] / trF
    w1 = max(w1, abs(r['R'] - r['f']) + abs(r['D'])); w2 = max(w2, abs(r['R'] - r['e'] - pred))
    w3 = max(w3, abs(r['R'] - r['e']) - sig ** 2 * abs(np.cos(mu[0])) * np.exp(-S[0, 0] / 2) / 2)
check('single Gaussian: (f) = R and D = 0 (12 fresh states)', w1 < 1e-9, f'({w1:.1e})', value=w1, tol=1e-9)
check('single Gaussian: R - e = -E_rho[cos q] K_qq K_qp / tr F (12 fresh states)', w2 < 1e-9, f'({w2:.1e})', value=w2, tol=1e-9)
check('single Gaussian: |R - e| <= sigma^2 |E_rho a\'\'| (12 fresh states)', w3 < 1e-9, value=w3, tol=1e-9)

# 4. independent brute force: posterior by direct x-quadrature, score and H by finite differences of log p_Y
def brute(ws, mus, Ss, sig, ny=181, nx=181):
    lo = np.min([m - 6 * np.sqrt(np.diag(S) + sig ** 2) for m, S in zip(mus, Ss)], 0)
    hi = np.max([m + 6 * np.sqrt(np.diag(S) + sig ** 2) for m, S in zip(mus, Ss)], 0)
    yq, yp = np.linspace(lo[0], hi[0], ny), np.linspace(lo[1], hi[1], ny)
    xlo = np.min([m - 7 * np.sqrt(np.diag(S)) for m, S in zip(mus, Ss)], 0); xhi = np.max([m + 7 * np.sqrt(np.diag(S)) for m, S in zip(mus, Ss)], 0)
    xq, xp = np.linspace(xlo[0], xhi[0], nx), np.linspace(xlo[1], xhi[1], nx)
    X = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1).reshape(-1, 2); dx = (xq[1] - xq[0]) * (xp[1] - xp[0])
    rho = sum(w * np.exp(-0.5 * np.einsum('ni,ij,nj->n', X - m, np.linalg.inv(S), X - m)) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
              for w, m, S in zip(ws, mus, Ss)) * dx
    fX = np.stack([X[:, 1], -np.sin(X[:, 0])], -1); aX = (1 - np.cos(X[:, 0])) / 2
    Y = np.stack(np.meshgrid(yq, yp, indexing='ij'), -1).reshape(-1, 2)
    p = np.zeros(len(Y)); mf = np.zeros((len(Y), 2)); ma = np.zeros(len(Y))
    for c in range(0, len(Y), 2000):
        y = Y[c:c + 2000]
        Kr = np.exp(-((y[:, None, :] - X[None]) ** 2).sum(-1) / (2 * sig ** 2)) / (2 * np.pi * sig ** 2) * rho[None]
        p[c:c + 2000] = Kr.sum(1); mf[c:c + 2000] = Kr @ fX; ma[c:c + 2000] = Kr @ aX
    p = p.reshape(ny, ny); mf = mf.reshape(ny, ny, 2) / p[..., None]; ma = ma.reshape(ny, ny) / p
    hq, hp = yq[1] - yq[0], yp[1] - yp[0]; L = np.log(p)
    sq, sp = np.gradient(L, hq, axis=0), np.gradient(L, hp, axis=1)
    Hqp = -np.gradient(sq, hp, axis=1); Hqq = -np.gradient(sq, hq, axis=0); Hpp = -np.gradient(sp, hp, axis=1)
    wgt = p * hq * hp; I = slice(3, -3)
    wgt, sq, sp, Hqp, Hqq, Hpp, mf, ma = wgt[I, I], sq[I, I], sp[I, I], Hqp[I, I], Hqq[I, I], Hpp[I, I], mf[I, I], ma[I, I]
    trF = np.sum(wgt * (sq ** 2 + sp ** 2)); R = np.sum(wgt * (sq * mf[..., 0] + sp * mf[..., 1])) / (sig ** 2 * trF)
    f = np.sum(wgt * 2 * ma * Hqp) / np.sum(wgt * (Hqq + Hpp)); e = np.sum(wgt * 2 * ma * sq * sp) / trF
    return R, f, e
Vv = np.array([1, 1]) / np.sqrt(2); Wv = np.array([1, -1]) / np.sqrt(2)
bs = ([0.5, 0.5], [np.array([2.0, 0.3]), np.array([2.4, -0.3])],
      [0.1 ** 2 * np.outer(Vv, Vv) + 0.8 ** 2 * np.outer(Wv, Wv), 0.6 ** 2 * np.outer(Vv, Vv) + 0.3 ** 2 * np.outer(Wv, Wv)], 1.0)
rb = brute(*bs, ny=241, nx=181); rf = evaluate_full(*bs)
diffs = (abs(rb[0] - rf['R']), abs(rb[1] - rf['f']), abs(rb[2] - rf['e']), abs((rb[0] - rb[1]) - rf['D']))
check('brute force (x-quadrature + finite differences) reproduces R, (f), (e) and D = R - f of an overlapping mixture',
      max(diffs) < 2e-5 and abs(rf['D']) > 100 * max(diffs), value=max(diffs), tol=2e-5,
      info=f'(diffs {diffs[0]:.1e}, {diffs[1]:.1e}, {diffs[2]:.1e}, D: {diffs[3]:.1e}; D = {rf["D"]:+.4f})')

# 5. measurement-1 base state: grid convergence and the exceedance itself
r0 = evaluate_full(*state(0.0), 0.1); r1 = evaluate_full(*state(0.0), 0.1, n=int(1.5 * r0['n']))
cv = max(abs(r0[k] - r1[k]) for k in ['R', 'f', 'D', 'A', 'R_switch', 'law_term'])
check('base concentric state: converged in the grid (1.5x changes R, f, D, A, R_switch, law term by < 1e-8)', cv < 1e-8, f'({cv:.1e})', value=cv, tol=1e-8)
check('base concentric state: R > lambda_comp > lambda_rho >= law term', r0['R'] > r0['lam_comp'] > references(*state(0.0), 0.1)[0] >= r0['law_term'])

# 6. summary_B reproducible and quoted numbers
ok_rep, dev_rep, where_rep, s = reproduce('results/summary_B.json', 'summarize_B.py', loose=LOOSE)
check('summary_B.json is reproduced from the raw result files (relative 1e-9; rank correlations 0.02 absolute, fit slopes 1e-6; the stored file is left unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
f2, f3, f4 = (lambda x: '%.2f' % x), (lambda x: '%.3f' % x), (lambda x: '%.4f' % x)
m1 = s['m1_base_concentric']; m2s = s['m2_single']; mx = s['m2_mix']
quoted = {
    'concentric R/lam_comp vs width ratio 1/2/4.4/8: 0.745 / 0.861 / 1.060 / 1.117':
        [f3(s['m1_concentric_ratio'][k]) for k in ['1', '2', '4.4', '8']] == ['0.745', '0.861', '1.060', '1.117'],
    'concentric R/lam_comp vs tilt 0/22.5/45/90: 1.060 / 1.093 / 1.068 / 1.009':
        [f3(s['m1_concentric_tilt'][k]) for k in ['0', '22.5', '45', '90']] == ['1.060', '1.093', '1.068', '1.009'],
    'separation along W keeps R/lam_comp >= 1.060': f3(s['m1_sepW_min_ratio']) == '1.060',
    'separation along V: above 1 up to delta/sigma = 10, below at 12 (0.977, overlap 0.11)':
        (s['m1_sepV_last_above'], round(s['m1_sepV_first_below']['delta_over_sigma']), f3(s['m1_sepV_first_below']['ratio']),
         f2(s['m1_sepV_first_below']['overlap'])) == (10.0, 12, '0.977', '0.11'),
    'base concentric: R 0.3156, lam_comp 0.2976, R_sep 0.2772, A 1.535, switch -0.1099, law 0.2210, cov 0.0947, D -0.0001, overlap 0.73':
        (f4(m1['R']), f4(m1['lam_comp']), f4(m1['R_sep']), f3(m1['A']), f4(m1['R_switch']), f4(m1['law_term']), f4(m1['cov']),
         f4(m1['D']), f2(m1['overlap'])) == ('0.3156', '0.2976', '0.2772', '1.535', '-0.1099', '0.2210', '0.0947', '-0.0001', '0.73'),
    'ratio 1 concentric: covariance part 0.0006': f4(s['m1_ratio1_concentric_cov']) == '0.0006',
    'm1: 512 states, 227 exceed; all with switch < 0, A 1.14-1.58, cov >= 0.037, overlap >= 0.19':
        (s['m1_n_states'], s['m1_n_exceed'], s['m1_exceed_all_switch_negative'], f2(s['m1_exceed_A_range'][0]), f2(s['m1_exceed_A_range'][1]),
         f3(s['m1_exceed_min_cov']), f2(s['m1_exceed_min_overlap'])) == (512, 227, True, '1.14', '1.58', '0.037', '0.19'),
    'm1: max R/lam_comp 1.117; law term never above lam_rho; max |R-f| 0.0017, max |R-e| 0.0040':
        (f3(s['m1_max_ratio']), s['m1_law_minus_lam_rho_max'] < 0, f4(s['m1_max_abs_R_minus_f']), f4(s['m1_max_abs_R_minus_e'])) == ('1.117', True, '0.0017', '0.0040'),
    'maps: exceed only for width ratio >= 2.85, elongation >= 11.9 (overlap >= 0.34)':
        (f2(s['m1_map_ratio_min_ratio_exceed']), f3(s['m1_map_elong_min_elongation_exceed'])[:4], f2(s['m1_map_elong_min_overlap_exceed'])) == ('2.85', '11.8', '0.34'),
    'm1 identities hold to < 1e-12': max(s['m1_identity_errors'].values()) < 1e-12,
    'single: 1675 cases, closed form 4.3e-12, bound ratio max 0.960, sign rule 800/800, zero at cos q0 = 0':
        (m2s['n'], '%.1e' % m2s['closed_form_max_err'], f3(m2s['bound_max_ratio']), m2s['sign_rule_ok'], int(m2s['n_cos_nonzero']),
         m2s['max_abs_rem_at_cos0'] < 1e-11) == (1675, '4.3e-12', '0.960', 800, 800, True),
    'mixtures: 1000 cases, 432 overlapping, R = f + D to < 1e-10, mass 1':
        (mx['n'], mx['n_overlapping'], mx['identity_R_eq_f_plus_D_max_err'] < 1e-10, abs(mx['mass_min'] - 1) < 1e-9) == (1000, 432, True, True),
    'mixtures: median |R-e| 0.0048 vs |R-f| 0.0020 (overlapping); f closer in 885; max |R-e| 0.063, |R-f| 0.057':
        (f4(mx['median_abs_R_minus_e_overlapping']), f4(mx['median_abs_R_minus_f_overlapping']), mx['n_f_closer_than_e'],
         f3(mx['max_abs_R_minus_e']), f3(mx['max_abs_R_minus_f'])) == ('0.0048', '0.0020', 885, '0.063', '0.057'),
    'mixtures: |f-e|/(sigma^2 E|a\'\'|) max 5.9 overall, 1.02 without overlap; Spearman 0.74 (cos) / 0.70 (sin)':
        (f2(mx['fe_over_curv_max'])[:3], f2(mx['fe_over_curv_max_nonoverlap']), f2(mx['spearman_absfe_curv']), f2(mx['spearman_absfe_curvsin'])) == ('5.8', '1.02', '0.74', '0.70'),
    'mixtures: Spearman |D| vs overlap 0.89, nonPD 0.70, log-ratio spread 0.07, alpha spread -0.02':
        (f2(mx['spearman_absD_overlap']), f2(mx['spearman_absD_nonPD']), f2(mx['spearman_absD_logratio_std']), f2(mx['spearman_absD_alpha_std'])) == ('0.89', '0.70', '0.07', '-0.02'),
    'mixtures: Spearman |f - law| vs alpha spread 0.64, log-ratio spread 0.19':
        (f2(mx['spearman_abscov_alpha_std']), f2(mx['spearman_abscov_logratio_std'])) == ('0.64', '0.19'),
    'mixtures: D without overlap median 3e-12 (max 1.4e-4); D with H > 0 everywhere up to 0.012 (359 states)':
        ('%.0e' % mx['D_nonoverlap_median'], '%.1e' % mx['D_nonoverlap_max'], f3(mx['D_when_H_PD_everywhere_max']), mx['n_H_PD_everywhere']) == ('3e-12', '1.4e-04', '0.012', 359),
    'mixtures: max |D| by sigma 0.0003 / 0.0015 / 0.0076 / 0.057; per-state slope median 2.3':
        ([f4(mx['D_by_sigma_max'][k]) for k in ['0.03', '0.1', '0.3']] + [f3(mx['D_by_sigma_max']['1.0'])], f2(mx['D_slope_003_01']['median'])) ==
        (['0.0003', '0.0015', '0.0076', '0.057'], '2.30'),
    'mixtures: D sign 401 + / 411 -; law term <= lam_rho in all; R > lam_rho in 11': (mx['n_D_pos'], mx['n_D_neg'], mx['law_le_lam_rho_all'], mx['n_R_gt_lam_rho']) == (401, 411, True, 11),
    'mixtures: median |f-e| 0.0027, median |D| 0.0020 (overlapping)': (f4(mx['median_abs_fe_overlapping']), f4(mx['median_abs_D_overlapping'])) == ('0.0027', '0.0020'),
}
for k, v in quoted.items():
    check('summary number: ' + k, v)
finish()
