"""Verification for stage I (continuous fan, lower-bound chain, other flows).  Exit 0 iff all checks pass."""
import sys, os, json
import numpy as np
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from obs_eval2 import evaluate_full
from i_tools import fields, KAPPA

L = lambda f: json.load(open(f'results/{f}.json'))
rng = np.random.default_rng(5)

# ---- 0. the lean evaluator
dev = 0.0
for kind in ['pendulum', 'double_well', 'quartic']:
    for _ in range(3):
        ws = [0.6, 0.4]; mus = [rng.normal(0, 0.5, 2), rng.normal(0, 0.5, 2)]; Ss = []
        for _ in range(2): A = rng.normal(0, 0.3, (2, 2)); Ss.append(A @ A.T + 0.01 * np.eye(2))
        e = evaluate_full(ws, mus, Ss, 0.1, kind=kind, grid='vw'); a = fields(ws, mus, Ss, 0.1, kind=kind); g = fields(ws, mus, Ss, 0.1, kind=kind, gh=16)
        dev = max(dev, abs(a['R'] - e['R']), abs(g['R'] - e['R']))
check('lean evaluator (closed forms and Gauss-Hermite) = stage-B evaluator, 9 random mixtures, three flows', dev < 1e-11, f'({dev:.1e})', value=dev, tol=1e-11)
# twist flow: Gauss-Hermite posterior vs direct summation over prior nodes (no shared formula)
from indep_eval import R_brute
import indep_eval
ws = [0.5, 0.5]; mus = [np.array([1.0, 0.05]), np.array([0.98, -0.03])]; Ss = [np.diag([0.02, 0.01]), np.array([[0.01, 0.004], [0.004, 0.02]])]
r_gh = fields(ws, mus, Ss, 0.1, kind='twist', gh=24)['R']
# R_brute is written for the pendulum: swap its flow for the twist flow temporarily
src = open(os.path.join(HERE, 'indep_eval.py')).read().replace("fX = np.stack([Xn[:, 1], -np.sin(Xn[:, 0])], -1)",
                                                                 "fX = np.stack([Xn[:, 1], -Xn[:, 0]], -1) / (1 + Xn[:, 0] ** 2 + Xn[:, 1] ** 2)[:, None]")
ns = {}; exec(compile(src, 'indep_eval_twist', 'exec'), ns); r_b = ns['R_brute'](ws, mus, Ss, 0.1, ny=121)
check('twist flow: Gauss-Hermite posterior evaluator = direct summation over prior nodes', abs(r_gh - r_b) < 1e-7, f'(R {r_gh:.9f} vs {r_b:.9f})', value=abs(r_gh - r_b), tol=1e-7)

# ---- 1. continuous fan
fan = L('i1_fan'); best = max(fan, key=lambda r: r['checks']['n40'])
c = best['checks']
check('functional fan: continuum reached (n = 20, 40, 80 agree to 1e-4) and grid-converged (per_sd 8 vs 12 < 1e-8)',
      max(abs(c['n20'] - c['n40']), abs(c['n80'] - c['n40'])) < 1e-4 and abs(c['n40_per_sd12'] - c['n40']) < 1e-8, f"(g* {c['n20']:.5f}, {c['n40']:.5f}, {c['n80']:.5f})")
check('functional fan: scale invariant in the small-sigma regime (sigma = 1e-4 vs 1e-3 within 2e-3)', abs(c['n40_sigma1e-4'] - c['n40']) < 2e-3, f"({c['n40_sigma1e-4']:.5f} vs {c['n40']:.5f})")
from i1_fan import gstar as fan_g
gf, rf = fan_g(np.array(best['x']), n=40, per_sd=8)
check('functional fan: best state re-evaluated from its parameters', abs(gf - c['n40']) < 1e-10, f'(g* {gf:.6f})')
check('functional fan: g* stays below 0.06 and R < 1', all(r['checks']['n40'] < 0.06 for r in fan) and rf['R'] < 1, f"(best {c['n40']:.4f}, R {rf['R']:.7f})")
from indep_eval import R_indep
from i1_fan import fan as fan_state
from h1_rstar import rstar
ws, mus, Ss = fan_state(np.array(best['x']), 20); Ri, _ = R_indep(ws, mus, Ss, 1e-3, 300, 300, gh=16); g20 = (Ri - rstar(1e-3)['Rstar']) / (1 - rstar(1e-3)['Rstar'])
check('functional fan (n = 20): independent Gauss-Hermite evaluator agrees', abs(g20 - c['n20']) < 1e-8, f'(g* {g20:.6f})')

# ---- 2. lower-bound chain
B = [o for o in L('i2_bound') if 'error' not in o]; small = [o for o in B if o['sigma'] <= 1e-2]
check('exact identity J_tot (1 - f) = E[g H_vv] - E[g H_ww] + 2 J_ww on every state', max(abs(o['exact_identity'] - o['one_minus_f']) for o in B) < 1e-12,
      f"({len(B)} states, max err {max(abs(o['exact_identity'] - o['one_minus_f']) for o in B):.1e})")
check('bathtub bound LB <= 1 - f on every state (rigorous chain)', all(o['LB'] <= o['one_minus_f'] + 1e-12 for o in B))
seg = [o for o in B if o['tag'] == 'Rstar_segment']
check('the chain is tight for the optimal single segment (bound / actual > 0.99 for sigma <= 0.03)', all(o['LB_over_sigma'] / o['gap_over_sigma'] > 0.99 for o in seg if o['sigma'] <= 0.03),
      f"({[round(o['LB_over_sigma'] / o['gap_over_sigma'], 4) for o in seg]})")
check('line-wise Cramer-Rao along W: Q(1) J_ww >= (1 - lambda_rho)/(2 (E delta^2 + sigma^2)) on every state', all(o['Q1Jww'] >= o['CR_bound'] - 1e-12 for o in B))
mb = min(o['LB_over_sigma'] for o in small); mg = min(o['gap_over_sigma'] for o in small)
check('over all states with sigma <= 1e-2: bound >= 0.75 sigma, actual >= 0.94 sigma', mb >= 0.75 and mg >= 0.94, f'(bound {mb:.4f}, actual {mg:.4f})')
# direct check of the CR inequality on a fresh mixture: J_ww E_p[(w + v)^2] >= 1 (V/W coordinates centred at the top)
ws = [0.7, 0.3]; mus = [np.array([np.pi + 0.01, 0.0]), np.array([np.pi - 0.02, 0.01])]; Ss = [np.diag([0.003, 0.002]), np.array([[0.002, -0.001], [-0.001, 0.003]])]
r = fields(ws, mus, Ss, 0.01, need_H=True); Y = r['Y'] - np.array([np.pi, 0.0]); p = r['fields']['p']
from i_tools import V, W
v = Y @ V; w = Y @ W; Jww = W @ r['J'] @ W
check('Cramer-Rao along W (fresh mixture): J_ww * E[(w + v)^2] >= 1', Jww * np.sum(p * (w + v) ** 2) >= 1, f'({Jww * np.sum(p * (w + v) ** 2):.4f})')

# ---- 3. other flows
S1 = L('i3_single'); summ = L('summary_I')
from i3_flows import series_poly, rstar_poly, POLY
coef, _ = series_poly(10)
check('universal polynomial-top series: Psi(eps) = 2 eps - 5/2 eps^2 + 7/4 eps^3 - 1/8 eps^4 + ...', coef[1:5] == ['2', '-5/2', '7/4', '-1/8'], f'({coef[1:7]})')
from i3_flows import phi_star
dv = 0.0
for k in POLY:
    for s in [1e-3, 1e-2, 0.1, 0.3]:
        t = rstar_poly(k, s); e = t['eps']; v = np.sqrt(4 + 9 * e * e) - 3 * e
        dv = max(dv, abs((1 - t['Rstar'] / POLY[k][0]) - (1 - (1 - e * v / 2) * phi_star(e / v)[0])))
check('polynomial tops in closed form: v* = sqrt(4 + 9 eps^2) - 3 eps (read off the exact series 2 - 3 eps + 9/4 eps^2 - 81/64 eps^4 + ...) gives 1 - R*/kappa exactly',
      dv < 1e-13 and coef[0] == '0' and series_poly(10)[1][1:5] == ['-3', '9/4', '0', '-81/64'], f'(max dev {dv:.1e})')
ok = all(r['series_err'] < 1e-8 for k in POLY for r in S1['flows'][k]['rows'])
check('polynomial tops: series to eps^10 reproduces the 1-d optimum (error < 1e-8 for sigma <= 0.1)', ok)
ch = L('i3_check')
check('law 1 / R* = grid evaluation of the optimal segment (double well, quartic, twist; sigma = 1e-3, 1e-2, 0.1)', max(abs(c['diff']) for c in ch) < 1e-10,
      f"(max {max(abs(c['diff']) for c in ch):.1e})", value=max(abs(c['diff']) for c in ch), tol=1e-10)
pred, meas = summ['predicted_gap_coefficient'], summ['measured_gap_coefficient_sigma0']
check('gap coefficient (kappa - R*)/sigma -> 2 sqrt(2 beta kappa) in all four flows (relative 1e-3)', all(abs(meas[k] / pred[k] - 1) < 1e-3 for k in pred),
      f"({', '.join(f'{k}: {meas[k]:.4f} vs {pred[k]:.4f}' for k in pred)})")
check('beta values: pendulum 1/8, double well 3/4, quartic 3/4, twist 3/8', all(abs(summ['beta'][k] - b) < 1e-6 for k, b in [('pendulum', 0.125), ('double_well', 0.75), ('quartic', 0.75), ('twist', 0.375)]),
      f"({summ['beta']})")
from i3_flows import rstar_twist
t = rstar_twist(1e-3); rb = -np.inf
from i3_flows import law1_twist
for _ in range(3000):
    x = [1 + rng.normal(0, 0.01), np.log(1e-3) + rng.normal(0, 3), np.log(0.035) + rng.normal(0, 0.5), rng.normal(0, 0.01)]; rb = max(rb, law1_twist(x, 1e-3, gh=16)[0])
check('twist R*(1e-3): 3000 random single Gaussians never beat it', rb <= t['Rstar'] + 1e-10, f"(best random {rb:.8f}, R* {t['Rstar']:.8f})")
try:
    SE = L('i3_search')
    check('mixture searches near the top of the other flows: R < kappa, grid/quadrature converged (< 1e-8)', all(r['R'] < r['kappa'] and abs(r['R_check'] - r['R']) < 1e-8 for r in SE),
          f"(g* {[(r['kind'][:5], r['K'], r['sigma'], round(r['gstar'], 4)) for r in SE]})")
    pol = [r for r in SE if r['kind'] != 'twist']; tw = [r for r in SE if r['kind'] == 'twist']
    check('double well, quartic: mixtures near the top gain at most a few % of the gap: (kappa - R)/sigma >= 0.95 x the single-Gaussian coefficient',
          all(r['gap_over_sigma'] >= 0.95 * r['Rstar_gap_over_sigma'] for r in pol), f"(min ratio {min(r['gap_over_sigma'] / r['Rstar_gap_over_sigma'] for r in pol):.4f})")
    check('twist: mixtures gain more (bending), but (kappa - R)/sigma stays >= 0.5 x the single coefficient and is the same at sigma = 1e-3 and 1e-2 (+-5 %)',
          all(r['gap_over_sigma'] >= 0.5 * r['Rstar_gap_over_sigma'] for r in tw)
          and all(abs(a['gap_over_sigma'] / b['gap_over_sigma'] - 1) < 0.05 for a in tw for b in tw if a['K'] == b['K']), f"({[(r['K'], r['sigma'], round(r['gap_over_sigma'], 4)) for r in tw]})")
except FileNotFoundError:
    check('i3_search.json present', False)
I5 = L('i5_arc'); b5 = max(I5, key=lambda o: o['gstar_n48'])
check('twist, curved needle: optimal curvature = turning rate of the stretching direction along W, 1/sqrt 2 (within 0.01), in all 4 starts',
      all(abs(abs(o['curvature']) - 1 / np.sqrt(2)) < 0.01 for o in I5), f"(curvatures {[round(o['curvature'], 4) for o in I5]})")
check('twist, curved needle: (kappa - R)/sigma = 0.51 (prediction 1/2 for a needle that follows the turning), scale invariant, grid-converged',
      abs(b5['gap_over_sigma_n48'] - 0.5) < 0.02 and abs(b5['gstar_n48_sigma0.0001'] - b5['gstar_n48']) < 3e-3 and abs(b5['gstar_n48_per_sd14'] - b5['gstar_n48']) < 1e-8,
      f"(gap/sigma {b5['gap_over_sigma_n48']:.4f}; g* {b5['gstar_n48']:.4f}, at 1e-4 {b5['gstar_n48_sigma0.0001']:.4f})")
from i5_arc import gval
g5, r5 = gval(np.array(b5['x']), 48, gh=6, per_sd=10)
check('twist, curved needle: re-evaluated from its parameters (to 1e-8; the quadrature itself is accurate to ~5e-8)', abs(g5 - b5['gstar_n48']) < 1e-8, f'(g* {g5:.6f}, diff {abs(g5 - b5["gstar_n48"]):.1e})', value=abs(g5 - b5['gstar_n48']), tol=1e-8)
from i5_arc import arc
ws, mus, Ss = arc(np.array(b5['x']), 16, sig=1e-2); rb = ns['R_brute'](ws, mus, Ss, 1e-2, ny=101); rg = fields(ws, mus, Ss, 1e-2, kind='twist', per_sd=10, gh=6)['R']
check('twist, curved needle (16 segments, sigma = 1e-2): direct summation over prior nodes agrees (its accuracy is ~5e-8)', abs(rb - rg) < 1e-7, f'(R {rg:.9f} vs {rb:.9f})', value=abs(rb - rg), tol=1e-7)
I4 = L('i4_twist')
check('twist, components added one by one: (kappa - R)/sigma decreases towards the bent-needle value, never below 0.49; R < kappa',
      all(r['scale'][1]['gap_over_sigma'] > 0.49 and r['scale'][1]['R'] < 0.25 for c in I4 for r in c['chain']),
      f"({[[round(r['scale'][1]['gap_over_sigma'], 3) for r in c['chain']] for c in I4]})")
ok, d, where, _ = reproduce('results/summary_I.json', 'i_analyze.py', loose={'/x': 1e-6, 'coefficient_sigma0': 1e-6, '/beta/': 1e-6, 'predicted_gap_coefficient': 1e-6})
check('summary_I.json reproducible (relative 1e-9; extrapolated coefficients, fitted beta and the coefficient predicted from it 1e-6; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')
finish()
