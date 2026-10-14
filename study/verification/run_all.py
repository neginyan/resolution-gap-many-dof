"""Verification for stage A of the nonlinear critical-rate observation. Exit code 0 iff all checks pass.
Recomputes the key facts independently and checks every number quoted in the stage-A summary."""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT)
from obs_eval import evaluate, cand_c, cand_d, mixture_grid, SHAPES
import fast_eval as fe
from mixtures import references
from adversarial import unpack

from vtools import check, finish, reproduce

def law(mu, S, sig, kind='pendulum'):
    if kind == 'pendulum': J = np.array([[0, 1.0], [-np.cos(mu[0]) * np.exp(-S[0, 0] / 2), 0]])
    else: J = np.array([[0, 1.0], [-3 * (mu[0] ** 2 + S[0, 0]), 0]])
    SJ = (J + J.T) / 2; F = np.linalg.inv(S + sig ** 2 * np.eye(2))
    return np.trace(F @ SJ) / np.trace(F), np.linalg.eigvalsh(SJ).max()

rng = np.random.default_rng(11)
def rand_gauss():
    A = rng.normal(size=(2, 2)) * rng.uniform(0.1, 0.8); return np.array([rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1)]), A @ A.T + 0.005 * np.eye(2)

# 1. evaluator reproduces fast_eval.py of the arrow-of-time repository exactly
w1 = 0
for _ in range(3):
    mu, S = rand_gauss()
    for sig in [0.05, 0.5, 1.5]:
        r = evaluate([1], [mu], [S], sig); _, trF, Sg = fe.dSdt([1], [mu], [S], sig, 0, 'pendulum', n=r['n'])
        w1 = max(w1, abs(Sg - r['Sigma']) / max(1.0, abs(Sg)), abs(trF - r['trF']) / trF)
check('evaluator reproduces fast_eval (Sigma, tr F; relative difference < 1e-10)', w1 < 1e-10, f'({w1:.1e})', value=w1, tol=1e-10)
# 2. grid convergence of the sweep values
D = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
worst = 0
for i in rng.choice(len(D), 8, replace=False):
    rec = D[i]; r = rec['rows'][rng.integers(0, 25)]
    rr = evaluate([1], [np.array([rec['q0'], 0.0])], [np.array(rec['S'])], r['sigma'], n=int(1.5 * r['n']))
    worst = max(worst, abs(rr['R'] - r['R']))
check('sweep values converged in the grid (1.5x resolution changes R by < 1e-8)', worst < 1e-8, f'({worst:.1e})', value=worst, tol=1e-8)
# 3. linear hyperbolic flow: exact formula
w3 = 0
for sig in [0.05, 0.5, 1.5]:
    S = np.array([[0.3, 0.1], [0.1, 0.05]]); r = evaluate([1], [np.array([np.pi, 0.2])], [S], sig, kind='hyperbolic_linear')
    F = np.linalg.inv(S + sig ** 2 * np.eye(2)); w3 = max(w3, abs(r['R'] - np.trace(np.array([[0, 1.], [1, 0]]) @ F) / np.trace(F)), abs(r['cand_e'] - r['R']))
check('linear flow: R = tr(Sym B F)/tr F, and candidate (e) equals R', w3 < 1e-9, f'({w3:.1e})', value=w3, tol=1e-9)
# 4-5. Gaussian law on fresh random states (pendulum and quartic) and its consequence R <= lambda_max(Sym Jbar) <= kappa
okp, okq, okb, worst_p, worst_q = True, True, True, 0, 0
for _ in range(15):
    mu, S = rand_gauss(); sig = 10 ** rng.uniform(-1.5, 0.3)
    r = evaluate([1], [mu], [S], sig); L, lam = law(mu, S, sig)
    worst_p = max(worst_p, abs(r['R'] - L)); okb &= r['R'] <= lam + 1e-12 and lam <= 1
    mu2 = np.array([rng.uniform(-2, 2), rng.uniform(-1, 1)])
    _, trF, Sg = fe.dSdt([1], [mu2], [S], sig, 0, 'quartic', n=1201); Rq = -Sg / (sig ** 2 * trF)
    worst_q = max(worst_q, abs(Rq - law(mu2, S, sig, 'quartic')[0]) / max(1e-12, abs(Rq)))
check('single Gaussian, pendulum: R = tr(F Sym Jbar)/tr F (15 fresh random states)', worst_p < 1e-8, f'(max diff {worst_p:.1e})', value=worst_p, tol=1e-8)
check('single Gaussian, quartic oscillator: same law (15 fresh random states)', worst_q < 1e-8, f'(max rel diff {worst_q:.1e})', value=worst_q, tol=1e-8)
check('single Gaussian: R <= lambda_max(Sym Jbar) <= kappa = 1 (pendulum)', okb)
# 6. momentum-reversal symmetric states have R = 0; reflected pair has opposite R
w6 = 0
for q0 in [np.pi, 2.0, 0.3]:
    for S in [SHAPES['isotropic small (0.1)'], SHAPES['elongated along q (0.8 x 0.1)']]:
        w6 = max(w6, abs(evaluate([1], [np.array([q0, 0.0])], [S], 0.4)['R']))
r1 = evaluate([1], [np.array([2.0, 0.0])], [SHAPES['sharp along stretching (0.05 x 0.5)']], 0.3)['R']
r2 = evaluate([1], [np.array([2.0, 0.0])], [SHAPES['sharp along contracting (0.5 x 0.05)']], 0.3)['R']
w6 = max(w6, abs(r1 + r2))
check('momentum-reversal symmetric states: R = 0; p-reflected shapes: opposite R', w6 < 1e-10 and abs(r1) > 0.1, f'({w6:.1e})', value=w6, tol=1e-10)
# 7. Theorem 9: R -> (c) with error O(sigma^2)
mu = np.array([2.2, 0.0]); S = SHAPES['sharp along stretching (0.05 x 0.5)']; c = cand_c([1], [mu], [S])
e1 = evaluate([1], [mu], [S], 0.02)['R'] - c; e2 = evaluate([1], [mu], [S], 0.01)['R'] - c
check('Theorem 9: R(sigma) -> (c), error ratio for sigma halved ~ 4 (O(sigma^2))', 3.8 < e1 / e2 < 4.2, f'(ratio {e1 / e2:.3f})', value=abs(e1 / e2 - 4), tol=0.2)
# 8. candidate (d) analytic formula vs direct quadrature over p_Y
mu = np.array([1.3, 0.2]); S = SHAPES['sharp along stretching (0.05 x 0.5)']; sig = 0.7
T = S + sig ** 2 * np.eye(2); qs = np.linspace(mu[0] - 10 * np.sqrt(T[0, 0]), mu[0] + 10 * np.sqrt(T[0, 0]), 20001)
pq = np.exp(-(qs - mu[0]) ** 2 / (2 * T[0, 0])) / np.sqrt(2 * np.pi * T[0, 0])
w8 = abs(np.trapezoid(pq * (1 - np.cos(qs)) / 2, qs) - cand_d([1], [mu], [S], sig))
check('candidate (d): analytic E_{p_Y}[a(q)] equals quadrature', w8 < 1e-10, f'({w8:.1e})', value=w8, tol=1e-10)
# 9. the mixture counterexample, recomputed at two resolutions
adv = json.load(open('results/adversarial_ratio_0.1.json'))['ratio_0.1']
ws, mus, Ss = unpack(np.array(adv['x'])); n0 = mixture_grid(ws, mus, Ss, 0.1)[3]
Ra = evaluate(ws, mus, Ss, 0.1, n=n0, pad=10.0)['R']; Rb = evaluate(ws, mus, Ss, 0.1, n=2 * n0, pad=10.0)['R']
lam_rho, lam_comp, R_sep, ov = references(ws, mus, Ss, 0.1)
check('two-component counterexample: R > max_k lambda_max(Sym J_k) > lambda_max(Sym J_rho), and R < kappa (two resolutions)',
      Ra > lam_comp > lam_rho and abs(Ra - Rb) < 1e-10 and Ra < 1, value=abs(Ra - Rb), tol=1e-10, info=f'(R={Ra:.4f}, lam_comp={lam_comp:.4f}, lam_rho={lam_rho:.4f}, overlap={ov:.2f})')
# 10. summary file is reproducible from the raw results, and the quoted numbers match it
ok_rep, dev_rep, where_rep, after = reproduce('results/summary.json', 'summarize.py')
check('summary.json is reproduced from the raw result files (relative 1e-9; the stored file is left unchanged)', ok_rep,
      f'(max relative deviation {dev_rep:.1e} at {where_rep or "-"})', value=dev_rep, tol=1e-9)
s = after; f4 = lambda x: '%.4f' % x
quoted = {
    'single: cases, states': (s['single']['n_cases'], s['single']['n_states']) == (1675, 67),
    'single: max |R - law| < 1e-11': s['single']['max_abs_R_minus_law'] < 1e-11,
    'single: all R <= lambda(Sym Jbar) <= 1': s['single']['all_R_le_lam_Jbar'] and s['single']['all_lam_Jbar_le_1'],
    'R(0.01)/c = 0.9992 at every position': all(f4(x) == '0.9992' for x in s['theorem9_R001_over_c']),
    'example q0 = pi: c 0.9502, R 0.9495 / 0.4933 / 0.0291, d 0.9694 / 0.9427 / 0.5635':
        (f4(s['example_pi']['c']), f4(s['example_pi']['R']['0.01']), f4(s['example_pi']['R']['0.342']), f4(s['example_pi']['R']['2.0']),
         f4(s['example_pi']['d']['0.01']), f4(s['example_pi']['d']['0.342']), f4(s['example_pi']['d']['2.0'])) ==
        ('0.9502', '0.9495', '0.4933', '0.0291', '0.9694', '0.9427', '0.5635'),
    'example q0 = pi: (e) deviates by +5.8% at sigma = 2': '%.1f' % (100 * s['example_pi']['e_rel_dev_sigma2']) == '5.8',
    'example q0 = 0: R +0.0009, (e) -0.0008 at sigma = 2': ('%+.4f' % s['example_q0']['R_sigma2'], '%+.4f' % s['example_q0']['e_sigma2']) == ('+0.0009', '-0.0008'),
    'mixtures: 1000 cases, max R 0.771, all <= 1, 11 above lam_rho, 0 above lam_comp':
        (s['mixtures_random']['n'], '%.3f' % s['mixtures_random']['max_R'], s['mixtures_random']['all_R_le_1'],
         s['mixtures_random']['n_R_gt_lam_rho'], s['mixtures_random']['n_R_gt_lam_comp']) == (1000, '0.771', True, 11, 0),
    'mixtures: |R - R_sep| median 4e-11 / max 0.0014 (overlap < 1e-3), max 0.268 (overlap >= 0.3)':
        ('%.0e' % s['mixtures_random']['dev_by_overlap']['0-0.001']['median'], '%.4f' % s['mixtures_random']['dev_by_overlap']['0-0.001']['max'],
         '%.3f' % s['mixtures_random']['dev_by_overlap']['0.3-1.01']['max']) == ('4e-11', '0.0014', '0.268'),
    'counterexample: R 0.3488, lam_comp 0.3372, lam_rho 0.2697, ratio 1.0345, overlap 0.63':
        (f4(s['adversarial']['ratio_0.1']['R']), f4(s['adversarial']['ratio_0.1']['lam_comp']), f4(s['adversarial']['ratio_0.1']['lam_rho']),
         f4(s['adversarial']['ratio_0.1']['best']), '%.2f' % s['adversarial']['ratio_0.1']['overlap']) == ('0.3488', '0.3372', '0.2697', '1.0345', '0.63'),
    'mixtures (e): 200 cases, 91 overlapping, median |R-e| 0.0040 vs |R-R_sep| 0.0297, max |R-e| 0.0633, R > e in 94, counterexample e 0.3515':
        (s['mixtures_e']['n'], s['mixtures_e']['n_overlapping'], f4(s['mixtures_e']['median_abs_R_minus_e_overlapping']),
         f4(s['mixtures_e']['median_abs_R_minus_Rsep_overlapping']), f4(s['mixtures_e']['max_abs_R_minus_e']), s['mixtures_e']['n_R_gt_e'],
         f4(s['mixtures_e']['counterexample_e'])) == (200, 91, '0.0040', '0.0297', '0.0633', 94, '0.3515'),
    'candidate (e) never exceeds kappa (it is an average of lambda_max <= kappa)': s['mixtures_e']['max_e'] <= 1,
    'max R over everything computed < 1 (0.949)': s['max_R_everywhere'] < 1 and '%.3f' % s['max_R_everywhere'] == '0.949',
    'collapse of R/c onto the shape formula: max diff 4.1e-11': '%.1e' % s['collapse_max_diff'] == '4.1e-11',
    'symmetric states max |R| 5e-14; p-reflection antisymmetry 6e-15':
        ('%.0e' % s['symmetric_states_max_abs_R'], '%.0e' % s['stretch_contract_antisymmetry_max']) == ('5e-14', '6e-15'),
    'Theorem 9: (R - c)/sigma^2 ratio between the two smallest sigmas = 1.0004 at every position':
        all(f4(x) == '1.0004' for x in s['theorem9_ratio_of_scaled_errors']),
    'mixtures (e): max e 0.663': '%.3f' % s['mixtures_e']['max_e'] == '0.663',
}
_D = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
_nm = sum(any(abs(a['R']) < abs(b['R']) - 1e-12 for a, b in zip(rec['rows'], rec['rows'][1:])) for rec in _D)
quoted['single: |R| never increases with sigma in the 67 measured states'] = (len(_D), _nm) == (67, 0)
for k, v in quoted.items():
    check('summary number: ' + k, v)
finish()
