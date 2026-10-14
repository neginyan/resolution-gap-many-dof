"""Verification for stage G (two needles near the top; best single Gaussian; nested-grid recheck; coefficient c).
Exit 0 iff all checks pass; failures are listed by name."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from obs_eval2 import evaluate_full, V, W
from g1_two_needles import make, law1
from g2_concentric import shape, TOP
from g3_best_single import law1_params, Rstar

L = lambda f: json.load(open(f'results/{f}.json'))
slope = lambda x, y: float(np.polyfit(np.log(np.abs(x)), np.log(np.abs(y)), 1)[0])
Rm = np.stack([V, W], 1)
rng = np.random.default_rng(7)

# ---- 0. law 1 used for single components and for R*: against the full quadrature evaluator
dev = 0.0
for _ in range(6):
    sig = 10 ** rng.uniform(-2, -0.5); x = [np.pi + rng.normal(0, 0.5), np.log(sig) + rng.normal(0, 1), np.log(sig) / 2 + rng.normal(0, 0.5), rng.uniform(0, np.pi)]
    c, s = np.cos(x[3]), np.sin(x[3]); U = Rm @ np.array([[c, -s], [s, c]]); S = U @ np.diag(np.exp(2 * np.array(x[1:3]))) @ U.T
    r = evaluate_full([1.0], [np.array([x[0], 0.3])], [S], sig, grid='vw', per_sd=10)
    dev = max(dev, abs(r['R'] - law1_params(x, sig)), abs(r['R'] - law1(np.array([x[0], 0.3]), S, sig)[0]))
check('law 1 (both implementations) = quadrature R for 6 random single Gaussians', dev < 1e-9, f'(max dev {dev:.1e})', value=dev, tol=1e-9)

# ---- 1. two needles: same shape
g1 = L('g1_families')
check('same-shape needles: mixing never raises R above the sharpest needle (all 176 states, excess <= 1e-10)',
      all(r['excess'] <= 1e-10 for r in g1), f"(max excess {max(r['excess'] for r in g1):.1e})", value=max(max(r['excess'] for r in g1), 0), tol=1e-10)
ok = True; info = []
for s in [1e-3, 1e-2]:
    for f, key in [('sym', 'off2'), ('one', 'off2'), ('alongW', 'off2'), ('one', 'd_abar'), ('alongW', 'd_abar'), ('width', 'd_abar')]:
        v = [r for r in g1 if r['family'] == f and r['sigma'] == s and r['delta'] < 0.3]
        k = slope([r[key] for r in v], [r['excess'] for r in v]); ok &= abs(k - 1) < 0.05; info.append(f'{f}/{key}/{s:g}:{k:.3f}')
check('same-shape needles: |excess| is first order in offset^2 and in d_abar at small offsets (log-log slope 1 +- 0.05)', ok, '(' + ', '.join(info) + ')')
v = [r for r in g1 if r['family'] == 'sym' and r['sigma'] == 1e-3][0]
ws, mus, Ss = make('sym', v['delta'], 1e-3); rr = evaluate_full(ws, mus, Ss, 1e-3, grid='vw', per_sd=12)
check('same-shape needles: fresh evaluation (per_sd 12) reproduces a stored excess', abs(rr['R'] - v['R']) < 1e-10, f"(diff {rr['R'] - v['R']:.1e})", value=abs(rr['R'] - v['R']), tol=1e-10)
check('same-shape needles: grid resolution (per_sd 10 vs 16) below 1e-9 in every state', max(abs(r['R_per_sd16'] - r['R']) for r in g1) < 1e-9,
      f"({max(abs(r['R_per_sd16'] - r['R']) for r in g1):.1e})", value=max(abs(r['R_per_sd16'] - r['R']) for r in g1), tol=1e-9)

# ---- 2. concentric needles of different shape
M = L('g2_map') + L('g2_map2'); b = max(M, key=lambda r: r['g'])
S1 = shape(1e-3)[0]; S2 = shape(1e-3, b['rv'], b['rw'])[0]
rr = evaluate_full([0.5, 0.5], [TOP, TOP], [S1, S2], 1e-3, grid='vw', per_sd=14); mR = max(law1(TOP, S, 1e-3)[0] for S in [S1, S2])
gg = (rr['R'] - mR) / (1 - mR)
check('concentric needles: fresh evaluation of the best map point reproduces g = 0.0269 (positive, < 1)', abs(gg - b['g']) < 1e-6 and 0 < gg < 1,
      f"(g {gg:.6f} vs stored {b['g']:.6f})", value=abs(gg - b['g']), tol=1e-6)
sc = L('g2_scale'); sg = [r['sigma'] for r in sc]
ks = [slope(sg, [r[k] for r in sc]) for k in ['excess', 'gap', 'd_abar']]
check('concentric needles: excess, gap and d_abar all scale as sigma^1 (slopes 1 +- 0.01) and g is constant (spread < 1e-3)',
      all(abs(k - 1) < 0.01 for k in ks) and np.ptp([r['g'] for r in sc]) < 1e-3, f"(slopes {[round(k, 4) for k in ks]}, g spread {np.ptp([r['g'] for r in sc]):.1e})")
check('concentric needles: grid resolution below 1e-9 (map, scale)', max(abs(r['R_per_sd16'] - r['R']) for r in M + sc) < 1e-9,
      f"({max(abs(r['R_per_sd16'] - r['R']) for r in M + sc):.1e})", value=max(abs(r['R_per_sd16'] - r['R']) for r in M + sc), tol=1e-9)
sh = L('g2_shift'); info = []; ok = True
for dr in 'VW':
    v = [r for r in sh if r['direction'] == dr and r['delta'] < 0.3]; k = slope([r['delta'] for r in v], [b['g'] - r['g'] for r in v]); ok &= abs(k - 2) < 0.05; info.append(f'{dr}: {k:.3f}')
check('shifting the second needle: g falls as delta^2 (slope 2 +- 0.05)', ok, '(' + ', '.join(info) + ')')
se = L('g2_search')
check('adversarial near-top search: g < 1 and R < 1 in every run (grid resolution < 1e-8)', all(r['g'] < 1 and r['R'] < 1 and abs(r['R_per_sd16'] - r['R']) < 1e-8 for r in se),
      f"(max g {max(r['g'] for r in se):.4f}, max R {max(r['R'] for r in se):.6f})")
r = max(se, key=lambda r: r['g']); rr = evaluate_full(r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']], r['sigma'], grid='vw', per_sd=12)
check('adversarial search: fresh evaluation of the largest-g state reproduces R', abs(rr['R'] - r['R']) < 1e-8, f"(diff {rr['R'] - r['R']:.1e})", value=abs(rr['R'] - r['R']), tol=1e-8)

# ---- 3. best single Gaussian R*(sigma)
g3 = L('g3_best_single')
check('no stored state exceeds the best single Gaussian at its window: R <= R*(sigma) for all states',
      g3['n_R_gt_Rstar'] == 0 and g3['min_margin'] > 0 and g3['n'] >= 1591, f"(n {g3['n']}, min margin {g3['min_margin']:.2e}, max g* {g3['max_gstar']:.4f})")
fresh = {s: Rstar(s, full=True)[0] for s in [1e-3, 1e-2, 0.1]}
check('R*(sigma) recomputed from scratch matches the stored table', max(abs(fresh[s] - g3['Rstar_table'][k][0]) for s, k in [(1e-3, '0.001'), (1e-2, '0.01'), (0.1, '0.1')]) < 1e-10)
best_rand = -np.inf
for _ in range(20000):
    x = [np.pi + rng.normal(0, 0.3), np.log(1e-3) + rng.normal(0, 3), np.log(0.06) + rng.normal(0, 1.5), rng.normal(0, 0.3)]
    best_rand = max(best_rand, law1_params(x, 1e-3))
check('random search over 20000 single Gaussians at sigma = 1e-3 never beats R*(1e-3)', best_rand <= fresh[1e-3] + 1e-12, f"(best random {best_rand:.8f}, R* {fresh[1e-3]:.8f})")
from scipy.optimize import minimize_scalar
def aligned(s):
    f = lambda Lg: -(1 + np.exp(-np.exp(Lg) / 4)) / 2 * np.exp(Lg) / (np.exp(Lg) + 2 * s ** 2)
    return -minimize_scalar(f, bounds=(np.log(s ** 2) - 5, 8), method='bounded', options={'xatol': 1e-12}).fun
dv = max(abs(1 - aligned(s) - s) / s for s in [1e-5, 1e-4, 1e-3])
check('closed form at the top (needle along V/W, s_v -> 0): R = (1 + e^{-s_w^2/4})/2 * s_w^2/(s_w^2 + 2 sigma^2); 1 - R* = sigma (1 + O(sigma))',
      dv < 2e-3 and all(aligned(s) <= Rstar(s, full=True)[0] + 1e-12 for s in [1e-4, 1e-3, 1e-2]), f'(max |(1-R)/sigma - 1| {dv:.1e} for sigma <= 1e-3)')
check('R*(1) is below 1/2 (the sup 1/2 is reached only by an infinitely long needle)', g3['Rstar_table']['1.0'][0] < 0.5, f"(R*(1) = {g3['Rstar_table']['1.0'][0]:.8f})")
c = max(g3['rows'], key=lambda o: o['gstar']); v = [r for r in L('g2_map2') if abs(r['R'] - c['R']) < 1e-15][0]
rr = evaluate_full([0.5, 0.5], [TOP, TOP], [shape(1e-3)[0], shape(1e-3, v['rv'], v['rw'])[0]], 1e-3, grid='vw', per_sd=14)
check('closest state to R*: fresh evaluation stays below R*(1e-3)', rr['R'] < fresh[1e-3] and abs(rr['R'] - c['R']) < 1e-9, f"(R {rr['R']:.9f}, R* {fresh[1e-3]:.9f})")

# ---- 4. coefficient c (law 1, needle limit)
g5 = L('g5_coefficient'); rows = g5['rows']
check('c from law 1 matches the measured slope of 1 - rho CV Omega at small t (relative 1e-3)',
      abs(g5['c'] / ((rows[-1]['measured'] - g5['c0']) / rows[-1]['t']) - 1) < 1e-3, f"(c {g5['c']:.6f}, measured {(rows[-1]['measured'] - g5['c0']) / rows[-1]['t']:.6f})")
err = [abs(r['prediction_exact_law1'] / r['measured'] - 1) for r in rows]
check('law-1 prediction of 1 - rho CV Omega: relative error falls with t (monotone) and is below 3e-4 at t = 0.01', all(np.diff(err) < 0) and err[-1] < 3e-4, f"(errors {[f'{e:.1e}' for e in err]})")
# independent: slope of the exact law-1 prediction at tiny t by finite differences
from g5_coefficient import lam
F = L('f1_chains'); bb = max(F, key=lambda r: r['final']['ratio']); ws, mus, Ss, sig0 = bb['ws'], [np.array(m) for m in bb['mus']], [np.array(S) for S in bb['Ss']], bb['sigma']
k = g5['needle_index']; Svw = Rm.T @ Ss[k] @ Rm
def P(t):
    D = np.diag([t ** 2, t]); Sk = Rm @ (np.sqrt(D) @ Svw @ np.sqrt(D)) @ Rm.T; T = Rm.T @ (Sk + (sig0 * t) ** 2 * np.eye(2)) @ Rm
    phi = (T[1, 1] - T[0, 0]) / (T[1, 1] + T[0, 0]); ln = lam(mus[k], Sk); lr = sum(w * lam(m, Sk if i == k else S) for i, (w, m, S) in enumerate(zip(ws, mus, Ss)))
    return (1 - phi) + phi * (1 - ln) / (1 - lr)
fd = (P(2e-6) - P(1e-6)) / 1e-6
check('closed-form c = 2(S_vv + sigma0^2)/S_ww + (-cos q_k) S_ww/(8(1 - lambda_rho0)) equals the finite-difference slope of the exact law-1 prediction',
      abs(fd / g5['c'] - 1) < 1e-4 and abs(P(1e-9) - g5['c0']) < 1e-10, f"(fd {fd:.6f}, c {g5['c']:.6f})")
check('needle alone: law 1 = grid within 1e-8 along the sequence', max(abs(r['R_needle_law1'] - r['R_needle_grid']) for r in rows) < 1e-8,
      f"({max(abs(r['R_needle_law1'] - r['R_needle_grid']) for r in rows):.1e})", value=max(abs(r['R_needle_law1'] - r['R_needle_grid']) for r in rows), tol=1e-8)
ok, d, where, _ = reproduce('results/g5_coefficient.json', 'g5_coefficient.py')
check('g5_coefficient.json reproducible (relative 1e-9; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')

# ---- 5. nested-grid recheck of all capped states
g4 = L('g4_recheck'); cap = [r for r in g4 if 'R_nested' in r]
fb = [r for r in g4 if 'failed' in r]
check('recheck covers every capped state: nested grid, or (if it did not fit in memory) qp n = 1200/1800 + uncapped V/W grid agreeing to 1e-10',
      len(cap) + len(fb) == sum(r['capped'] for r in g4) and all(r.get('diff_alt', 1) < 1e-10 and max(r['n_alt_vw6']) < 2401 for r in fb),
      f"(states {len(g4)}, capped {sum(r['capped'] for r in g4)}, nested {len(cap)}, fallback {len(fb)})")
mx = max(abs(r['diff']) for r in cap)
check('nested grid agrees with every stored R to 1e-6', mx < 1e-6, f'(max |diff| {mx:.1e})', value=mx, tol=1e-6)
mr = max(abs(r['R_nested_refined'] - r['R_nested']) for r in cap)
check('nested grid converged under refinement (max change < 1e-7) and mass = 1 to 1e-9', mr < 1e-7 and max(abs(r['mass_nested'] - 1) for r in cap) < 1e-9,
      f"(refine {mr:.1e}, mass {max(abs(r['mass_nested'] - 1) for r in cap):.1e})", value=mr, tol=1e-7)
check('no capped state reaches R >= 1 on the nested grid', max(r['R_nested'] for r in cap) < 1, f"(max {max(r['R_nested'] for r in cap):.8f})")

ok, d, where, _ = reproduce('results/summary_G.json', 'g_analyze.py')
check('summary_G.json reproducible (relative 1e-9; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')
finish()
