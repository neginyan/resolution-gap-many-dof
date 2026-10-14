"""Verification for stage M.  Exit 0 iff all checks pass.
Rules (paper B, appendix C): rank correlations with scipy.stats.rankdata after rounding to relative 1e-10; outputs of fits /
extrapolations / optimisations compared `loose`; every check uses <= 20 % of its tolerance."""
import sys, os, json, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from m_tools import coupled, find_kappa, gap, coefficient, ansatz_state, B_matrix, System, pendulum_chain
from l3_coupled import law1, R_quadrature, gap_coefficient

J = lambda f: json.load(open(f'results/{f}.json'))
Sm = J('summary_M'); rng = np.random.default_rng(12)

# ================= the evaluator =================
worst = 0
for e in [0.1, 0.3, -0.3]:
    for _ in range(3):
        m = np.concatenate([rng.normal([np.pi, 0.0], 0.3), rng.normal(0, 0.3, 2)]); A = rng.normal(0, 0.3, (4, 4)); S = A @ A.T + 0.01 * np.eye(4); sig = 0.2
        k = 1.0; g = gap(coupled(e), m[:2], np.linalg.cholesky(S), sig, k); R = k - g * sig
        worst = max(worst, abs(R - law1(m, S, sig, e)), abs(R - R_quadrature(m, S, sig, e, n=14)))
check('m_tools.gap (QR evaluation) = law 1 (plain inverse) = 4-d Gauss-Hermite posterior evaluation on 9 random states', worst < 1e-10, f'({worst:.1e})', worst, 5e-10)
m = np.array([3.0, 0.2, 0.0, 0.0]); A = rng.normal(0, 0.3, (4, 4)); S = A @ A.T + 0.01 * np.eye(4)
d = abs(R_quadrature(m, S, 0.2, 0.3, n=14) - R_quadrature(m + np.array([0, 0, 0.7, -1.3]), S, 0.2, 0.3, n=14))
check('the mean momentum does not enter R (exact zero modes of M1(b)), checked with the Gauss-Hermite evaluator', d < 1e-12, f'({d:.1e})', d, 5e-12)
wb = 0
for e in [0.1, 0.3, 1.0]:
    s = coupled(e); k, t = find_kappa(s, starts=30); B, _ = B_matrix(s, t[0]); Bfd = np.array(gap_coefficient(e)['B'])
    wb = max(wb, np.max(np.abs(np.abs(B) - np.abs(Bfd))))
check('analytic B (first-order perturbation) = finite-difference B of stage L (up to the sign convention of e-)', wb < 1e-6, f'({wb:.1e})', wb, 5e-6)
for e in [0.3, 2.0]:
    s = coupled(e); k, t = find_kappa(s, starts=30); c = coefficient(s, t[0], k)['coef']; g = gap(s, *ansatz_state(s, t[0], k, 1e-7), 1e-7, k)
    check(f'eps = {e}: the leading-order state evaluated at sigma = 1e-7 reproduces the formula (QR evaluation stays accurate)', abs(g - c) / c < 2e-6, f'({g:.7f} vs {c:.7f})', abs(g - c) / c, 1e-5)

# ================= M1 =================
q = np.linspace(-np.pi, np.pi, 1441); Q1, Q2 = np.meshgrid(q, q, indexing='ij'); mx = 0
for o in Sm['M1']['frame_opt']:
    e = o['eps']; c12 = np.cos(Q1 - Q2); a_ = (1 - np.cos(Q1) - e * c12) / 2; d_ = (1 - np.cos(Q2) - e * c12) / 2; b_ = e * c12 / 2
    lam = (a_ + d_) / 2 + np.sqrt(((a_ - d_) / 2) ** 2 + b_ ** 2)
    mx = max(mx, abs(lam.max() - o['kappa']), abs(coupled(e).lam(np.array(o['top'])) - o['kappa']))
check('M1: kappa and its location for eps = -0.3 ... 3 agree with a 1441^2 torus grid', mx < 1e-9, f'({mx:.1e})', mx, 5e-9)
neg = [o for o in Sm['M1']['frame_opt'] if o['eps'] < 0]
check('M1(c): for eps < 0 the top moves to (pi, pi) and kappa = 1 - eps', all(np.allclose(np.abs(o['top']), [np.pi, np.pi], atol=1e-5) and abs(o['kappa'] - (1 - o['eps'])) < 1e-12 for o in neg))
pos = [o for o in Sm['M1']['frame_opt'] if o['eps'] > 0]
check('M1(c): for eps > 0 (up to 3) the top stays at (pi, 0) / (0, pi)', all(np.allclose(sorted(np.abs(o['top'])), [0, np.pi], atol=1e-5) for o in pos))
dev = max(abs(o['opt_rel_diff_extrap_vs_formula']) for o in Sm['M1']['frame_opt'])
check('M1: full optimisation (sigma = 1e-4, 1e-5, 1e-6, extrapolated) = formula for all 7 eps (rel. 1e-7)', dev < 1e-7, f'(max {dev:.1e})', dev, 5e-7)
for e, F in Sm['M1']['free'].items():
    check(f'M1(a): eps = {e}: >= 50 assumption-free random starts; >= half reach the optimum', F['n_starts'] >= 50 and F['n_reach_opt'] >= F['n_starts'] / 2, f"({F['n_reach_opt']}/{F['n_starts']})")
    ex = (F['frame_opt_1e-5'] - F['opt_min_free_1e-5']) / F['frame_opt_1e-5']
    check(f'M1(a): eps = {e}: no random start goes below the optimum by more than the optimiser precision (1e-6 rel.)', ex < 1e-6, f'(lowest free minus optimum: {-ex:.1e} rel.)', max(ex, 0), 5e-6)
FRo = {o['eps']: o['opt']['1e-05'] for o in Sm['M1']['frame_opt']}
ex = max((FRo[o['eps']] - min(o['opt_random_1e-5'] + [o['opt_from_ansatz']['1e-05']])) / FRo[o['eps']] for o in Sm['M1']['range'])
check('M1(c): eps = -0.3, -0.1, 2, 3: no start (ansatz or random) goes below the optimum by more than the optimiser precision (1e-6 rel.)', ex < 1e-6, f'({max(ex, 0):.1e})', max(ex, 0), 5e-6)
for h in Sm['M1']['hessian']:
    ok = h['grad_max'] < 2e-5 and h['eig_min'] > -1e-5 * h['eig_max'] and h['scan_min_rel'] > -1e-7
    check(f"M1(b): eps = {h['eps']}, sigma = {h['sigma']:g}: stationary, no negative curvature beyond noise, flat modes do not lower the gap", ok,
          f"(grad {h['grad_max']:.1e}, min eig {h['eig_min']:.1e} of max {h['eig_max']:.2f}, scan {h['scan_min_rel']:.1e})", max(h['grad_max'] / 2e-5, -h['eig_min'] / (1e-5 * h['eig_max']), -h['scan_min_rel'] / 1e-7, 0) * 0.2, 1.0)

# ================= M2 =================
h = hashlib.sha256(open('results/m2_prediction.json', 'rb').read()).hexdigest(); logged = open('logs/m2_prediction_hash.txt').read()
check('M2: the prediction file is unchanged since registration (SHA-256 equals the logged one)', h in logged and h == Sm['M2']['prediction_sha256'], f'({h[:16]}...)')
P = J('m2_prediction')['systems']; Mo = Sm['M2']['systems']
for s in Mo:
    pr = [p for p in P if (p['kind'], p['n'], p['eps']) == (s['kind'], s['n'], s['eps'])][0]
    check(f"M2: {s['name']}: optimum (extrapolated) = registered prediction {pr['prediction']:.6f} (rel. 1e-7)", abs(s['opt_rel_diff_extrap_vs_prediction']) < 1e-7 and pr['prediction'] == s['prediction'],
          f"({s['extrap_sigma0']:.7f})", abs(s['opt_rel_diff_extrap_vs_prediction']), 5e-7)
    pw = max(v['F_times_sigma2'] for k, v in s['shape_1e-5'].items() if k.startswith('p'))
    check(f"M2: {s['name']}: momentum-only directions cost nothing (F weight sigma^2 F < 1e-6 along them at the optimum)", pw < 1e-6, f'({pw:.1e})', pw, 5e-6)
    lo = min(s['opt_random_1e-5']); ex = (s['frame_opt']['1e-05'] - lo) / s['frame_opt']['1e-05']
    check(f"M2: {s['name']}: no random start below the optimum by more than the optimiser precision (2e-6 rel.)", ex < 2e-6, f'({max(ex, 0):.1e})', max(ex, 0), 1e-5)
# the pre-registered numbers can be recomputed from the definitions
from m2_predict import SYSTEMS
dmax = 0
for (kind, n, e, s), p in zip(SYSTEMS, P):
    dmax = max(dmax, abs(coefficient(s, np.array(p['tops'][0]), p['kappa'])['coef'] - p['prediction']))
check('M2: the registered predictions are recomputed from kappa, the top and B', dmax < 1e-12, f'({dmax:.1e})', dmax, 5e-12)

# ================= M3 =================
if 'M3' in Sm:
    from m3_mix import Setup, R3
    from m_tools import frame
    w3 = 0
    for e in [0.1, 0.3]:
        st = Setup(e, 1e-3)
        A = rng.normal(0, 1, (3, 3)) * np.array([1e-3, 0.03, 0.03])[:, None]; S = A @ A.T + np.diag([1e-7, 1e-5, 1e-5]); mu = rng.normal(0, 1, 3) * np.array([1e-3, 0.03, 0.03])
        R, mass, n = R3(st, [1.0], [mu], [S], per_sd=6)
        Fr, _ = frame(st.s, st.q0); B4 = Fr[:, :3]; L4 = np.zeros((4, 4)); L4[:, :3] = B4 @ np.linalg.cholesky(S); L4[:, 3] = 1e3 * Fr[:, 3]
        g4 = gap(st.s, st.q0 + (B4 @ mu)[:2], L4, 1e-3, st.k); w3 = max(w3, abs((st.k - R) / 1e-3 - g4) / g4)
    check('M3: the 3-d reduction (p- infinitely long) equals 4-d law 1 for random single Gaussians', w3 < 1e-7, f'({w3:.1e})', w3, 5e-7)
    for o in Sm['M3']:
        if o['kind'] == '3d':
            d = abs(o['gap'] - o['gap_per_sd7']) / o['gap']
            check(f"M3: eps = {o['eps']}, K = {o['K']}: grid convergence (per_sd 5 vs 7)", d < 2e-5, f'({d:.1e})', d, 1e-4)
        check(f"M3: {o['kind']} eps = {o['eps']} K = {o['K']}: the mixture beats the best Gaussian (ratio < 1)", o['ratio'] < 1, f"({o['ratio']:.4f})")

# ================= M4 =================
if 'M4' in Sm:
    check('M4: the Z-bound (no Cramer-Rao step) is a valid lower bound on every state', Sm['M4']['Zbound_le_gap_all'], f"({Sm['M4']['n']} states)")
    check("M4: P = d^2 y/sigma >= Theta'^2 on every state (the weighted Cramer-Rao step of paper B, in the two-variable form)", Sm['M4']['P_ge_ThetaP2_all'],
          f"(min P/Theta'^2 = {Sm['M4']['P_over_ThetaP2_min']:.4f})")
    check('M4: the two-variable form is tight: median (form / actual gap) within 1e-3 of 1', abs(Sm['M4']['two_variable_over_gap'][1] - 1) < 1e-3,
          f"(median {Sm['M4']['two_variable_over_gap'][1]:.6f})", abs(Sm['M4']['two_variable_over_gap'][1] - 1), 5e-3)

# ================= M5 =================
if 'M5' in Sm and Sm['M5'].get('fixed_piece_length'):
    F5 = Sm['M5']['fixed_piece_length']; v = [F5['by_n'][k] for k in ['64', '128', '256']]
    d1, d2 = v[1] - v[0], v[2] - v[1]
    check('M5: bent needle with a FIXED absolute piece length: converges in n (successive differences halve: 1/n convergence)', 0 < d2 < 0.7 * d1 and d2 < 2e-4, f'({[round(x, 5) for x in v]})')
    check('M5: its continuum limit (Richardson in 1/n) is within 1 % of the prediction 1/2 and scale invariant (sigma = 1e-4 within 0.5 %)',
          abs(F5['extrap_richardson_1_over_n'] - 0.5) < 0.01 and abs(F5['sigma1e-4_n128'] - v[1]) / v[1] < 5e-3, f"({F5['extrap_richardson_1_over_n']:.5f}; 1e-4: {F5['sigma1e-4_n128']:.5f})",
          abs(F5['extrap_richardson_1_over_n'] - 0.5), 0.05)
    from l2_arc_continuum import gap as garc
    x5 = np.array(J('m5_arc')['by_n']['128']['x']); y5 = x5.copy(); y5[5] -= np.log(127 / 63); g64 = garc(y5, 64, gh=4, per_sd=6)
    check('M5: the n = 64 value is reproduced from the stored parameters', abs(g64 - F5['by_n']['64']) < 1e-9, f'({g64:.8f})', abs(g64 - F5['by_n']['64']), 5e-9)

# ================= reproducibility =================
LOOSE = {'opt': 1e-6, 'extrap': 1e-6, 'fit': 1e-6, 'coef': 1e-6, 'spearman': ('abs', 5e-3), 'ratio': 1e-6, 'gap': 1e-6, 'eig': ('abs', 1e-6), 'scan': ('abs', 1e-6), 'grad': ('abs', 1e-6)}
ok, dev, where, _ = reproduce('results/summary_M.json', 'm_analyze.py', 1e-9, LOOSE)
check('summary_M.json reproduces from the stored results (m_analyze.py)', ok, f'(max rel dev {dev:.1e} at {where})', dev, 5e-9)
finish()
