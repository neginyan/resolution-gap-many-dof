"""Verification for stage L.  Exit 0 iff all checks pass.
Rules of stage L (paper B, appendix C): rank correlations with scipy.stats.rankdata after rounding to relative 1e-10; outputs of
fits / extrapolations / argmax / optimisations compared `loose`; every check is written so that it uses <= 20 % of its tolerance."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from h2_adversarial import unpack
from i_tools import fields
from l_tools import diag

J = lambda f: json.load(open(f'results/{f}.json'))
Sm = J('summary_L')

# ================= L3: coupled pendulums =================
from l3_coupled import law1, R_quadrature, kappa, gap_scaled, gap_coefficient, unpack as unpack4, EVpp
chk = J('l3_check')
dev = max(abs(o['law1'] - o['gh18']) for o in chk)
check('L3: law 1 in 4-d equals the Gauss-Hermite posterior evaluation (18^4 nodes, no Stein identity) on 9 random states', dev < 1e-10, f'(max |diff| {dev:.1e})', dev, 1e-10)
z = max(abs(o['mc_dev_in_se']) for o in chk)
check('L3: Monte Carlo (10 x 4e5 samples) agrees with law 1 within 4 standard errors on all 9 states', z < 4, f'(max |dev|/SE {z:.2f})', z, 20)
rng = np.random.default_rng(5); o = chk[4]
m = np.concatenate([rng.normal([np.pi, 0.0], 0.3), rng.normal(0, 0.3, 2)]); A = rng.normal(0, 0.3, (4, 4)); S = A @ A.T + 0.01 * np.eye(4)
d = abs(law1(m, S, 0.2, 0.3) - R_quadrature(m, S, 0.2, 0.3, n=16))
check('L3: law 1 = Gauss-Hermite on a fresh random state (eps 0.3, sigma 0.2)', d < 1e-10, f'({d:.1e})', d, 1e-10)
q = np.linspace(-np.pi, np.pi, 1441); Q1, Q2 = np.meshgrid(q, q, indexing='ij'); worst = 0
for e in [0.0, 0.1, 0.3, 1.0]:
    c12 = np.cos(Q1 - Q2); a_ = (1 - np.cos(Q1) - e * c12) / 2; d_ = (1 - np.cos(Q2) - e * c12) / 2; b_ = e * c12 / 2
    lam = (a_ + d_) / 2 + np.sqrt(((a_ - d_) / 2) ** 2 + b_ ** 2); worst = max(worst, abs(lam.max() - kappa(e)))
check('L3: kappa(eps) = (1 + eps + sqrt(1 + eps^2))/2 is the maximum of lambda_max Sym Df over the torus (grid 1441^2; eps 0, .1, .3, 1)', worst < 1e-12, f'({worst:.1e})', worst, 1e-11)
S3 = J('l3_single'); wc = [o for o in S3 if o['sigma'] >= 1e-3 and o['sd_in_mode_frame']['p-'] < 200]
dpl = max(abs(o['gap_over_sigma'] - o['gap_law1_plain']) / o['gap_over_sigma'] for o in wc)
check('L3: the well-conditioned evaluation equals the plain law-1 formula (optima with sigma >= 1e-3 and p- s.d. < 200, where the plain inverse is well conditioned)',
      len(wc) >= 5 and dpl < 1e-9, f'({len(wc)} optima, max rel diff {dpl:.1e})', dpl, 5e-9)
for e, k in [(0.1, '0.1'), (0.3, '0.3')]:
    c = Sm['L3']['coef'][k]; r = abs(c['opt_gap_smallest_sigma'] - c['coef_matrix_formula']) / c['coef_matrix_formula']
    check(f'L3: eps = {e}: optimum (kappa - R*)/sigma at sigma = 1e-6 equals the matrix formula 2 tr[(B^1/2 C B^1/2)^1/2] (rel. 1e-4)', r < 1e-4,
          f"({c['opt_gap_smallest_sigma']:.6f} vs {c['coef_matrix_formula']:.6f})", r, 1e-4 / 0.2 * 0.2 * 5)
    allx = [o for o in S3 if o['eps'] == e]; ok = all(o['gap_over_sigma'] <= c['coef_matrix_formula'] * (1 + 1e-6) for o in allx)
    check(f'L3: eps = {e}: no optimum exceeds the leading-order formula (optimum <= ansatz) at any sigma', ok)
e1 = Sm['L3']['eps1'][0]; r = abs(e1['gap_over_sigma'] - e1['matrix_formula']) / e1['matrix_formula']
check('L3: out-of-sample eps = 1 (formula written before): optimum at sigma = 1e-5 equals the formula (rel. 1e-4)', r < 1e-4, f"({e1['gap_over_sigma']:.5f} vs {e1['matrix_formula']:.5f})", r, 5e-4)
e0 = [o for o in S3 if o['eps'] == 0.0 and o['sigma'] == 1e-6][0]
check('L3: eps = 0 (decoupled): the coefficient stays 1 (as for one pendulum)', abs(e0['gap_over_sigma'] - 1) < 1e-5, f"({e0['gap_over_sigma']:.7f})", abs(e0['gap_over_sigma'] - 1), 5e-5)
# the ansatz state reproduces the formula directly (independent of the optimiser)
from l3_coupled import ansatz_x
for e in [0.1, 0.3]:
    g = gap_scaled(ansatz_x(e), 1e-7, e); f = gap_coefficient(e)['matrix']
    check(f'L3: eps = {e}: the ansatz state (thin along V+, sigma T* in (W+, q-), wide in p-) gives the formula at sigma = 1e-7', abs(g - f) / f < 1e-5, f'({g:.7f} vs {f:.7f})', abs(g - f) / f, 5e-5)
# the matrix identity  min_T tr(BT) + tr(C T^-1) = 2 tr[(B^1/2 C B^1/2)^1/2]  on random 2x2 cases
from scipy.optimize import minimize
from scipy.linalg import sqrtm
w_ = 0
for _ in range(20):
    X = rng.normal(size=(2, 2)); B = X @ X.T + 0.1 * np.eye(2); Y = rng.normal(size=(2, 2)); Cc = Y @ Y.T + 0.1 * np.eye(2)
    def f(t):
        L = np.array([[np.exp(t[0]), 0], [t[1], np.exp(t[2])]]); T = L @ L.T; return np.trace(B @ T) + np.trace(Cc @ np.linalg.inv(T))
    v = min(minimize(f, rng.normal(size=3), method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 20000}).fun for _ in range(3))
    Bh = np.real(sqrtm(B)); w_ = max(w_, abs(v - 2 * np.trace(np.real(sqrtm(Bh @ Cc @ Bh)))) / v)
check('L3: matrix identity min_T tr(BT) + tr(C T^-1) = 2 tr[(B^1/2 C B^1/2)^1/2] (20 random 2x2 cases)', w_ < 1e-7, f'({w_:.1e})', w_, 1e-6)

# ================= L1(a): worst states =================
P = J('l1_post'); rows = Sm['L1a']['rows']
for s in [1e-3, 1e-2]:
    b = Sm['L1a']['opt_best'][f'{s:g}']
    o = [p for p in P if p['sigma'] == s and abs(p['diag']['gap_over_sigma'] - b['gap_over_sigma']) < 1e-12][0]
    ws, mus, Ss = unpack(np.array(o['x']), o['K'], s); r = fields(ws, mus, Ss, s, per_sd=6)
    gnow = (1 - r['R']) / s; dd = abs(gnow - b['gap_over_sigma']) / b['gap_over_sigma']
    check(f'L1(a): sigma = {s:g}: the worst state (K = {o["K"]}) re-evaluated from its stored parameters', dd < 1e-8, f'((1-R)/sigma {gnow:.6f})', dd, 1e-7)
    dg = abs(b['gap_fine'] - b['gap_over_sigma']) / b['gap_over_sigma']
    check(f'L1(a): sigma = {s:g}: worst state, standard grid vs finer grid (per_sd 10, >= 401 points)', dg < 1e-7, f'({dg:.1e})', dg, 1e-6)
    ind = b['indep']; di = abs(ind['gap_indep'] - b['gap_over_sigma']) / b['gap_over_sigma']
    check(f'L1(a): sigma = {s:g}: worst state, independent Gauss-Hermite evaluator agrees', di < 1e-6, f"({ind['gap_indep']:.7f})", di, 1e-5)
    if 'gap_brute' in ind:
        db = abs(ind['gap_brute'] - b['gap_over_sigma']) / b['gap_over_sigma']
        check(f'L1(a): sigma = {s:g}: worst state, direct kernel summation agrees', db < 1e-5, f"({ind['gap_brute']:.7f})", db, 1e-4)
    check(f'L1(a): sigma = {s:g}: the worst state is below the previous minimum 0.946 (and R < kappa)', b['gap_over_sigma'] < 0.946 and b['R'] < 1, f"({b['gap_over_sigma']:.5f})")
check('L1(a): the state-wise theorem of paper B holds on every new state (bound <= actual 1 - R)', Sm['L1a']['all_bound_le_gap'] and len(rows) >= 20, f'({len(rows)} states)')
sc = max(abs(r['gap_rel_Rstar'] - r['gap_rel_Rstar_other']) for r in rows)
check('L1(a): scale invariance: (1 - R)/(1 - R*(sigma)) of each state agrees between sigma = 1e-3 and 1e-2 within 0.01', sc < 0.01, f'(max diff {sc:.4f})', sc, 0.05)
mono = all(np.all(np.diff([v for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))]) <= 1e-12) for s in Sm['L1a']['opt_best_by_family_K'].values() for fam, d in s.items() if 'polish' in fam)
check('L1(a): the polished free mixtures decrease monotonically with K (each K starts from the previous optimum)', mono)
if 'shift_scan' in Sm['L1a']:
    am = max(abs(v['fit_argmin']) for k, v in Sm['L1a']['shift_scan'].items() if k.endswith('_w'))
    check('L1(a): moving the fan centre off the hyperbolic point along W only increases the gap (fitted minimum at |dw| < 0.02)', am < 0.02, f'(max |argmin| {am:.4f})', am, 0.1)

# ================= L1(b) =================
R1 = [o for o in J('l1_ratios') if 'error' not in o]
check("L1(b): Theta' >= geo (1 - rho) on every state (the inequality behind the rho route)", Sm['L1b']['Theta_ge_geo_1mrho_violations'] == 0, f'({len(R1)} states)')
fd = max(abs(o['slope_fd_vs_ibp']) for o in R1 if o['sigma'] <= 1e-2 and o['gap_over_sigma'] <= 1.2)
check('L1(b): focus-slope term by finite differences matches the integration-by-parts identity (near-optimal states)', fd < 5e-3, f'(max rel {fd:.1e})', fd, 2.5e-2)
rm = Sm['L1b'].get('rho_max_search', [])
check('L1(b): rho is NOT uniformly bounded: the rho-maximising search finds rho > 10 (in states with a large gap)', any(r['key'] == 'rho' and r['value'] > 10 for r in rm), f"(max rho {max([r['value'] for r in rm if r['key'] == 'rho'] or [0]):.1f})")
check('L1(b): ... and those states still satisfy the state-wise theorem', all(r['bound_over_sigma'] <= r['gap_over_sigma'] + 1e-12 for r in rm))
o = [p for p in P if p['family'] == 'free (L-BFGS polish)' and p['K'] == 16 and p['sigma'] == 1e-3][0]; ws, mus, Ss = unpack(np.array(o['x']), 16, 1e-3)
nus = [diag(ws, mus, Ss, 1e-3, min_axis=m)['nu'] for m in [161, 801, 1201]]
check('L1(b): the focus Fisher information nu is not a stable number (log-divergent where w -> 0 linearly): it changes by > 2x between grids', max(nus) > 2 * min(nus),
      str([round(v, 3) for v in nus]))
st = Sm['L1b']['stats']['near_optimal_gap_le_1.2']
check('L1(b): near-optimal states (gap <= 1.2 sigma): rho stays below 0.4', st['rho']['max'] < 0.4, f"(max {st['rho']['max']:.4f})", st['rho']['max'], 2.0)
# Spearman by rankdata after relative-1e-10 rounding, recomputed here
from scipy.stats import rankdata
def rk(x):
    x = np.asarray(x, float); e = 10.0 ** np.floor(np.log10(np.abs(x) + 1e-300)); return rankdata(np.round(x / e, 10) * e)
near = [o for o in R1 if o['sigma'] <= 1e-2 and o['gap_over_sigma'] <= 1.2]
sp = float(np.corrcoef(rk([o['rho'] for o in near]), rk([o['gap_over_sigma'] for o in near]))[0, 1])
check('L1(b): Spearman(rho, gap) recomputed with rankdata matches the summary', abs(sp - st['rho']['spearman_with_gap']) < 1e-3, f'({sp:.4f})', abs(sp - st['rho']['spearman_with_gap']), 5e-3)

# ================= L1(c) =================
PF = J('l1_profile')
check('L1(c): weight profile of all three optimised fans: sech^2 fits the core better than a Gaussian (rms of log)',
      len(PF) == 3 and all(q['weight']['sech2']['rms_log'] < q['weight']['gauss']['rms_log'] for q in PF), str([round(q['weight']['sech2']['rms_log'], 3) for q in PF]))
sw = [q['weight']['sech2']['params'][1] for q in PF]
check('L1(c): the sech^2 width agrees between the three fans within 5 % (0.43 in units of sqrt(sigma)/2)', (max(sw) - min(sw)) / np.mean(sw) < 0.05, str([round(v, 3) for v in sw]),
      (max(sw) - min(sw)) / np.mean(sw), 0.25)
ml = max(q['length']['exp_a2']['max_abs_log'] for q in PF)
check('L1(c): length profile L0 exp(c alpha^2) fits the core of all three fans within 10 % (max |log error|)', ml < 0.1, f'({ml:.3f})', ml, 0.5)

# ================= L2 =================
if 'L2' in Sm:
    for f in Sm['L2']['opt_fans']:
        check(f"L2: K = {f['K']}: R < kappa = 1/4 (gap > 0)", f['gap'] > 0, f"({f['gap']:.4f})")
        d = abs(f['gap'] - f['gap_fine']) / f['gap']
        check(f"L2: K = {f['K']}: grid convergence (Gauss-Hermite 6 / per_sd 10 vs 8 / >= 401 points)", d < 1e-5, f'({d:.1e})', d, 1e-4)
        d = abs(f['gap_sigma1e-4'] - f['gap']) / f['gap']
        check(f"L2: K = {f['K']}: scale invariance sigma = 1e-3 vs 1e-4 within 2 %", d < 0.02, f'({d:.4f})', d, 0.1)
    ind = [f for f in Sm['L2']['opt_fans'] if f.get('indep')]
    for f in ind:
        d = abs(f['indep']['gap_brute'] - f['indep']['gap_grid']) / f['indep']['gap_grid']
        check(f"L2: K = {f['K']}: direct kernel summation agrees with the grid evaluator (sigma = 1e-2)", d < 5e-5, f"({f['indep']['gap_brute']:.6f} vs {f['indep']['gap_grid']:.6f})", d, 2.5e-4)

    if 'arc_continuum' in Sm['L2']:
        A2 = Sm['L2']['arc_continuum']; d = [A2['stageI'][k] for k in ['24', '48', '128', '192', '384']]
        check('L2 (correction to stage I): the stage-I bent needle built from n straight pieces: the gap grows with n (24 -> 384)', all(np.diff(d[1:]) > 0) and d[-1] > 0.55,
              str([round(v, 4) for v in d]))
        for n, v in A2.get('brute', {}).items():
            dd = abs(v['brute'] - v['grid']) / v['grid']
            check(f'L2: the n-dependence is not numerical: direct kernel summation = grid evaluator for n = {n} (sigma = 1e-2)', dd < 5e-5, f"({v['brute']:.6f} vs {v['grid']:.6f})", dd, 2.5e-4)
        F = Sm['L2']['opt_fans']
        check('L2: 12-piece bent fans reach (kappa - R)/sigma < 0.5 for K = 4, 8, 16 (valid states, scale invariant)', all(f['gap'] < 0.5 and f['gap_sigma1e-4'] < 0.5 for f in F),
              str([round(f['gap'], 4) for f in F]))
        check('L2: with 48 pieces per needle the same fans are above 0.5 (the value below 0.5 belongs to the polygonal states)', all(f['gap_nseg48'] > 0.5 for f in F),
              str([round(f['gap_nseg48'], 4) for f in F]))

# ================= reproducibility (non-destructive; loose for optimiser / fit outputs) =================
LOOSE = {'opt_': 1e-6, 'fit': 1e-6, 'coef': 1e-6, 'extrap': 1e-6, 'spearman': ('abs', 5e-3)}
ok, dev, where, _ = reproduce('results/summary_L.json', 'l_analyze.py', 1e-9, LOOSE)
check('summary_L.json reproduces from the stored results (l_analyze.py)', ok, f'(max rel dev {dev:.1e} at {where})', dev, 1e-9 / 0.2 * 0.2 * 5)
ok, dev, where, _ = reproduce('results/l1_profile.json', 'l1_profile.py', 1e-9, {'params': 1e-6, 'rms': 1e-6, 'max_abs': 1e-6})
check('l1_profile.json reproduces (fits compared loose)', ok, f'(max rel dev {dev:.1e} at {where})', dev, 5e-9)
finish()
