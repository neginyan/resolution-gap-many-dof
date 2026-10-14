"""Verification for stage H (true R*(sigma), its expansion, mixtures that beat it).  Exit 0 iff all checks pass."""
import sys, os, json
import numpy as np
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from obs_eval2 import evaluate_full, V, W
from h1_rstar import rstar, gaussian, one_minus_R, series, sigma_c
from h2_adversarial import unpack
from g3_best_single import law1_params
from g1_two_needles import law1
from indep_eval import R_indep, R_brute

L = lambda f: json.load(open(f'results/{f}.json'))
Rm = np.stack([V, W], 1); TOP = np.array([np.pi, 0.0]); rng = np.random.default_rng(11)

# ---- 1. the best single Gaussian
dev = 0.0
for s in [1e-3, 1e-2, 0.1, 0.3]:
    t = rstar(s); r = evaluate_full([1.0], [TOP], [gaussian(s, t['u'], t['p'])], s, grid='vw', per_sd=12); dev = max(dev, abs(r['R'] - t['Rstar']))
check('R*(sigma) = quadrature R of the optimal rank-one segment (sigma = 1e-3, 1e-2, 0.1, 0.3)', dev < 1e-9, f'(max dev {dev:.1e})', value=dev, tol=1e-9)
ok = True
for _ in range(200):
    s = 10 ** rng.uniform(-3, -0.5); Svw = np.array([[rng.uniform(0, 1), 0], [0, rng.uniform(0, 1)]]) * np.array([s ** 2, 4 * s]) * rng.uniform(0.1, 3)
    c = rng.uniform(-1, 1) * np.sqrt(Svw[0, 0] * Svw[1, 1]); A = Svw.copy(); A[0, 1] = A[1, 0] = c; B = Svw.copy(); B[0, 1] = B[1, 0] = -np.sqrt(Svw[0, 0] * Svw[1, 1])
    if Svw[1, 1] > Svw[0, 0]:          # phi > 0 (longer along W than along V), needed for R > 0; for phi < 0 the other extreme wins
        ok &= law1(TOP, Rm @ B @ Rm.T, s)[0] >= law1(TOP, Rm @ A @ Rm.T, s)[0] - 1e-14
check('at fixed S_vv < S_ww the extreme S_vw = -sqrt(S_vv S_ww) (rank one) maximises law 1 (random shapes with phi > 0)', ok)
mx = 0.0
for _ in range(200):
    s = 10 ** rng.uniform(-3, -0.3); u = rng.uniform(1, 8)
    f = lambda p: (2 * s * (1 + s * p ** 2 / u)) / (u + 2 * s * (p + 1) + 2 * s ** 2 * p ** 2 / u)
    ps = np.linspace(0, 3, 30001); pn = ps[np.argmin(f(ps))]; pc = 2 / (1 + np.sqrt(1 + 4 * s / u)); mx = max(mx, abs(pn - pc))
check('closed-form best inner parameter p* = 2/(1 + sqrt(1 + 4 sigma/u)) matches a direct scan (200 cases)', mx < 2e-4, f'(max |diff| {mx:.1e})')
best = {}
for s in [1e-3, 0.1, 0.3]:
    b = -np.inf
    for _ in range(20000):
        x = [np.pi + rng.normal(0, 0.2), np.log(s) + rng.normal(0, 3), np.log(2 * np.sqrt(s)) + rng.normal(0, 1), rng.normal(0, 0.5)]
        b = max(b, law1_params(x, s))
    best[s] = b
check('20000 random single Gaussians per sigma (1e-3, 0.1, 0.3) never beat R*(sigma)', all(best[s] <= rstar(s)['Rstar'] + 1e-12 for s in best),
      '(' + ', '.join('%g: best %.6f vs R* %.6f' % (s, best[s], rstar(s)['Rstar']) for s in best) + ')')
g3 = L('g3_best_single'); dv = max(abs(rstar(float(k))['Rstar'] - v[0]) for k, v in g3['Rstar_table'].items() if float(k) < 0.9)
check('the stage-G 4-parameter optimisation had already found the true R* (148 windows < 0.9)', dv < 1e-10, f'(max dev {dv:.1e})', value=dv, tol=1e-10)

# ---- 2. the expansion
coef, w = series(True); coef_u, _ = series(False)
check('exact series: 1 - R* = s - 7/8 s^2 + 65/96 s^3 - 61/128 s^4 + ...; untilted: s - 3/4 s^2 + 11/24 s^3 (hand derivation)',
      coef[1:5] == ['1', '-7/8', '65/96', '-61/128'] and coef_u[1:4] == ['1', '-3/4', '11/24'], f'({coef[1:6]})')
ok = True; info = []
for m in [1, 2, 3, 5, 7]:
    s = 1e-2; err = abs(sum(float(Fr(c)) * s ** k for k, c in enumerate(coef[:m + 1])) - rstar(s)['one_minus_Rstar']); pred = abs(float(Fr(coef[m + 1]))) * s ** (m + 1)
    if pred > 1e-15: ok &= abs(err / pred - 1) < 0.05; info.append(f'{m}:{err / pred:.3f}')
check('series truncated at s^m: error = next term (ratio 1 +- 0.05) at sigma = 0.01', ok, '(' + ', '.join(info) + ')')
e11 = abs(sum(float(Fr(c)) * 0.3 ** k for k, c in enumerate(coef)) - rstar(0.3)['one_minus_Rstar'])
check('series to s^11 at sigma = 0.3: error below 1e-8', e11 < 1e-8, f'({e11:.1e})', value=e11, tol=1e-8)
sc = sigma_c(); h1 = L('h1_rstar')
check('sigma_c = 0.93909: interior optimum = 1/2 there; R* is attained below it and is the long-needle limit 1/2 above it',
      abs(sc - 0.939088) < 1e-5 and rstar(0.93)['attained'] and rstar(0.93)['Rstar'] > 0.5 and not rstar(0.95)['attained'], f'(sigma_c {sc:.6f})')
# the location of a smooth maximum (u*, p* and quantities derived from them) is only determined to ~sqrt(machine eps) by the 1-d
# optimiser, so it gets its own tolerance; the maximum values themselves are compared at 1e-9
ARGMAX_TOL = {'/u': 1e-6, '/p': 1e-6, 'tilt_angle': 1e-6, 'length': 1e-6, '/params': 1e-6}
ok, d, where, _ = reproduce('results/h1_rstar.json', 'h1_rstar.py', loose=ARGMAX_TOL)
check('h1_rstar.json reproducible (relative 1e-9; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')

# ---- 3. mixtures that beat R*
h2 = L('h2_adversarial')
check('adversarial search: g* > 0 in all 8 (K, sigma) runs, grid-converged (per_sd 10 vs 16 < 1e-12), mass 1',
      all(r['gstar'] > 0 and abs(r['R_per_sd16'] - r['R']) < 1e-12 and abs(r['mass'] - 1) < 1e-9 for r in h2),
      f"(g* {[round(r['gstar'], 4) for r in h2]})")
check('all states still have R < 1 = kappa', all(r['R'] < 1 for r in h2), f"(max R {max(r['R'] for r in h2):.6f})")
r = max([r for r in h2 if r['sigma'] == 1e-3], key=lambda r: r['gstar'])
Ri, m = R_indep(r['ws'], [np.array(x) for x in r['mus']], [np.array(S) for S in r['Ss']], 1e-3, 400, 400, gh=20)
check('independent evaluator (Gauss-Hermite posterior, no closed forms for E[f|y]) reproduces the sigma = 1e-3 violation', abs(Ri - r['R']) < 1e-10 and Ri > r['Rstar'],
      f"(diff {Ri - r['R']:.1e}; R - R* = {Ri - r['Rstar']:.2e})", value=abs(Ri - r['R']), tol=1e-10)
br = L('h2_brute')
check('direct-summation evaluator (kernel sums over prior nodes) reproduces the sigma = 0.1, 0.3 violations to 1e-7',
      all(abs(b['brute_minus_stored']) < 1e-7 and b['brute_minus_Rstar'] > 1e-3 for b in br), f"(max |diff| {max(abs(b['brute_minus_stored']) for b in br):.1e})")
r = [r for r in h2 if r['sigma'] == 0.3 and r['K'] == 2][0]
Rb = R_brute(r['ws'], [np.array(x) for x in r['mus']], [np.array(S) for S in r['Ss']], 0.3, ny=121)
check('fresh direct summation, sigma = 0.3, K = 2: R > R*', abs(Rb - r['R']) < 1e-7 and Rb > r['Rstar'], f"(R {Rb:.8f}, R* {r['Rstar']:.8f})")
h3 = L('h3_xsearch'); b = max(h3, key=lambda r: r['gstar'])
Ri, m = R_indep(b['ws'], [np.array(x) for x in b['mus']], [np.array(S) for S in b['Ss']], 1e-3, 400, 400, gh=20)
check('long search: the best state (K = 4) reproduced by the independent evaluator', abs(Ri - b['scale'][2]['R']) < 1e-10, f"(g* {b['gstar']:.4f})")
sm = [v for r in h3 for v in r['scale'] if v['sigma'] <= 1e-3]
check('scale invariance: each long-search state has the same g* at sigma = 1e-5, 1e-4, 1e-3 (spread < 2e-3)',
      all(np.ptp([v['gstar'] for v in r['scale'] if v['sigma'] <= 1e-3]) < 2e-3 for r in h3), f"(spreads {[round(float(np.ptp([v['gstar'] for v in r['scale'] if v['sigma'] <= 1e-3])), 5) for r in h3]})")
check('the gap stays proportional to sigma: (1 - R)/sigma >= 0.94 for every searched state at sigma <= 1e-3',
      min(v['one_minus_R_over_sigma'] for v in sm) >= 0.94, f"(min {min(v['one_minus_R_over_sigma'] for v in sm):.4f})")
h4 = L('h4_growk'); allk = [x for c in h4 for x in c['chain']]
check('adding components (K = 4..8): g* grows slowly and stays < 0.06; R < 1; grid-converged',
      all(x['gstar10'] < 0.06 and x['R'] < 1 and abs(x['R16'] - x['R']) < 1e-11 for x in allk), f"(g* by K {[(x['K'], round(x['gstar10'], 4)) for x in allk]})")
check('K = 4..8 states: g* at sigma = 1e-4 equals g* at 1e-3 (scale invariance, |diff| < 2e-3)', all(abs(x['gstar_sigma1e-4'] - x['gstar10']) < 2e-3 for x in allk))
h5 = L('h5_fan'); f = {o['n']: o for o in h5}
check('fan of n segments: n = 1 reproduces R* (|g*| < 1e-6); n = 8, 16, 32 give g* = 0.044 - 0.043 (no growth with n), scale invariant and grid-converged',
      abs(f[1]['gstar']) < 1e-6 and all(0.04 < f[n]['gstar'] < 0.05 and abs(f[n]['gstar_sigma1e-4'] - f[n]['gstar']) < 1e-3 and abs(f[n]['gstar_per_sd16'] - f[n]['gstar']) < 1e-9 for n in [8, 16, 32])
      and f[32]['gstar'] <= f[16]['gstar'] <= f[8]['gstar'], f"(g* {[round(f[n]['gstar'], 5) for n in [1, 8, 16, 32]]})")
from h5_fan import gstar as fan_g
gf = fan_g(f[16]['params'], 16, per_sd=10)[0]
check('fan n = 16 re-evaluated from its parameters', abs(gf - f[16]['gstar']) < 1e-10, f'(g* {gf:.6f})')
ok, d, where, _ = reproduce('results/summary_H.json', 'h_analyze.py', loose=ARGMAX_TOL)
check('summary_H.json reproducible (relative 1e-9; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')
finish()
