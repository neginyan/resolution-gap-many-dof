"""Verification for stage N.  Exit 0 iff all checks pass.
Rules: rank correlations with rankdata after relative-1e-10 rounding (none used here); fit / optimisation outputs compared `loose`;
every check uses <= 20 % of its tolerance."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from n_tools import Flow, R_product, unpack, gap as gapN
from m3_mix import Setup, R3
from m_tools import gap as gap4, frame

J = lambda f: json.load(open(f'results/{f}.json'))
Sm = J('summary_N'); rng = np.random.default_rng(21); SIG = 1e-3

# ================= the product evaluator (N1) =================
w = 0
for e in [0.1, 0.3]:
    fl = Flow(e, SIG); st = Setup(e, SIG)
    for th in [0.0, 0.5, -1.1]:
        K = 3; ws = rng.dirichlet(np.ones(K)); mus = [rng.normal(0, 1, 2) * np.array([SIG, 0.03]) for _ in range(K)]
        Ss = [(lambda A: A @ A.T + np.diag([1e-8, 1e-5]))(rng.normal(0, 1, (2, 2)) * np.array([SIG, 0.03])[:, None]) for _ in range(K)]; sb = 3e-3
        R2, _ = R_product(fl, ws, mus, Ss, sb, th, per_sd=6)
        c, s = np.cos(th), np.sin(th); Rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        m3 = [Rot @ np.array([m[0], m[1], 0]) for m in mus]; S3 = [Rot @ np.block([[S, np.zeros((2, 1))], [np.zeros((1, 2)), np.array([[sb]])]]) @ Rot.T for S in Ss]
        R3v, mass, n = R3(st, list(ws), m3, S3, per_sd=6); w = max(w, abs(R2 - R3v) / SIG)
check('N1: product evaluator (2-d grid + closed-form b-integrals) = 3-d evaluator, random product mixtures, theta = 0, 0.5, -1.1', w < 2e-6, f'(max |diff| in gap/sigma {w:.1e})', w, 1e-5)
w = 0
for e in [0.1, 0.3]:
    fl = Flow(e, SIG); st = Setup(e, SIG); Fr, _ = frame(st.s, st.q0)
    S2 = np.array([[4e-9, 1e-7], [1e-7, 5e-3]]); sb = 6e-3; th = 0.3; c, s = np.cos(th), np.sin(th); Rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    S3 = Rot @ np.block([[S2, np.zeros((2, 1))], [np.zeros((1, 2)), np.array([[sb]])]]) @ Rot.T; mu3 = Rot @ np.array([2e-4, 3e-3, 0])
    R2, _ = R_product(fl, [1.0], [mu3[:2] * 0 + np.array([2e-4, 3e-3])], [S2], sb, th, per_sd=6)
    L4 = np.zeros((4, 4)); L4[:, :3] = Fr[:, :3] @ np.linalg.cholesky(S3); L4[:, 3] = 1e3 * Fr[:, 3]
    g4 = gap4(st.s, st.q0 + (Fr[:, :3] @ mu3)[:2], L4, SIG, st.k); w = max(w, abs((st.k - R2) / SIG - g4) / g4)
check('N1: for a single Gaussian the product evaluator equals 4-d law 1 (p- infinitely long)', w < 1e-6, f'({w:.1e})', w, 5e-6)

# ================= N1 results =================
rows = {(r['kind'], r['K']): r for r in Sm['N1']['rows']}
for (kind, K), r in sorted(rows.items()):
    check(f"N1: {kind} K = {K}: the mixture beats the best Gaussian (ratio < 1)", r['opt_ratio'] < 1, f"({r['opt_ratio']:.5f})")
src = [o for o in J('n1_mix')] + (J('n1_more') if os.path.exists('results/n1_more.json') else [])
o = min([o for o in src if o['kind'] == '0.1' and o['K'] == 8], key=lambda o: o['gap']); fl = Flow(0.1, SIG)
g3 = gapN(np.array(o['x']), 8, fl, per_sd=6)
check('N1: eps = 0.1, K = 8: the stored optimum is re-evaluated from its parameters', abs(g3 - o['gap']) / o['gap'] < 1e-8, f'({g3:.7f})', abs(g3 - o['gap']) / o['gap'], 5e-8)
d3 = [x for x in J('n1_mix') if 'gap_3d_evaluator' in x]
dd = max(abs(x['gap_3d_evaluator'] - x['gap']) / x['gap'] for x in d3)
check('N1: every product optimum (K <= 8) is confirmed by the 3-d evaluator', dd < 2e-6, f'({len(d3)} states, max rel {dd:.1e})', dd, 1e-5)
for k in ['1d', '0.1', '0.3']:
    gp = Sm['N1']['gauss_product'][k]; dv = (gp['gap'] - gp['gap_reference']) / gp['gap_reference']
    check(f'N1: {k}: the best Gaussian of the product family is within 2e-4 of the true best Gaussian (and not below it)', -1e-8 < dv < 2e-4, f'({dv:.1e})', max(dv, 0), 1e-3)
for e in ['0.1', '0.3']:
    for K in [8, 16]:
        if (e, K) in rows and ('1d', K) in rows:
            check(f'N1: eps = {e}, K = {K}: the multi-dimensional gain is smaller than the 1-d gain', rows[(e, K)]['gain'] < rows[('1d', K)]['gain'],
                  f"({100 * rows[(e, K)]['gain']:.2f}% vs {100 * rows[('1d', K)]['gain']:.2f}%)")
if 'decomp' in Sm['N1']:
    w = max(max(abs(o['gain_total'] - o['gain_plane'] - o['gain_b']), abs(o['gain_plane'] - o['d_plane'] - o['d_cross'])) for o in Sm['N1']['decomp'])
    check('N1(b): the hybrid decomposition adds up (total = plane + q- re-tuning, plane = d_plane + d_cross)', w < 2e-12, f'({w:.1e})', w, 1e-11)
    for o in Sm['N1']['decomp']:
        if o['K'] == 8:
            fl = Flow(float(o['eps']), SIG); M = min([x for x in src if x['kind'] == o['eps'] and x['K'] == 8], key=lambda x: x['gap'])
            wm, mm, Sm_, sbm, thm = unpack(np.array(M['x']), 8, fl); Gx = [g for g in J('n1_gauss') if g['kind'] == o['eps']][0]
            _, _, _, sbg, thg = unpack(np.array(Gx['x']), 1, fl); Rh, _ = R_product(fl, wm, mm, Sm_, sbg, thg, per_sd=6); gH = (fl.k - Rh) / SIG
            d = abs(gH - o['gap_H']) / o['gap_H']
            check(f"N1(b): eps = {o['eps']}, K = 8: the hybrid state (plane mixture + Gaussian q- law) is re-evaluated", d < 1e-9, f'({gH:.7f})', d, 5e-9)
            check(f"N1(b): eps = {o['eps']}, K = 8: the plane mixture raises the q- cost (d_cross < 0) and carries most of the gain (> 85 %)",
                  o['d_cross'] < 0 and o['gain_plane'] / o['gain_total'] > 0.85, f"(d_cross {100 * o['d_cross']:.3f} %, plane share of gain {o['gain_plane'] / o['gain_total']:.3f})")
if 'nonproduct' in Sm['N1']:
    for o in Sm['N1']['nonproduct']:
        d = abs(o['gap_product_R3'] - [r for r in J('n1_mix') if r['kind'] == o['kind'] and r['K'] == o['K']][0]['gap']) / o['gap_product_R3']
        check(f"N1(b): eps = {o['kind']}, K = {o['K']}: the product start is reproduced by the 3-d evaluator", d < 2e-6, f'({d:.1e})', d, 1e-5)
        check(f"N1(b): eps = {o['kind']}, K = {o['K']}: leaving the product family never makes the state worse", o['gap_nonproduct'] <= o['gap_product_R3'] * (1 + 1e-9))

# ================= N2 =================
if 'N2' in Sm:
    from n2_pmin import pvals
    for o in J('n2_pmin'):
        v = pvals(np.array(o['x']), o['K'], o['sigma'], per_sd=8); d = abs(v['P'] - o['P_fine']) / o['P_fine']
        check(f"N2: K = {o['K']}, sigma = {o['sigma']:g}, y/sigma <= {o.get('ymax')}: P re-evaluated from the stored state", d < 1e-8, f"({v['P']:.6f})", d, 5e-8)
        dg = abs(o['P'] - o['P_fine']) / o['P_fine']; tag = f"K = {o['K']}, sigma = {o['sigma']:g}, y/sigma <= {o.get('ymax')}"
        if o.get('ymax') is None and o['P_fine'] < 1e-2:
            check(f"N2: {tag}: the collapsed state (P -> 0 with gap ~ 100) is real: P < 1e-3 and gap > 50 on the fine grid", o['P_fine'] < 1e-3 and o['gap_fine'] > 50, f"(P {o['P_fine']:.1e}, gap {o['gap_fine']:.1f})")
            check(f"N2: {tag}: search grid and fine grid agree for the collapsed state (relative 2e-3)", dg < 2e-3, f'({dg:.1e})', dg, 1e-2)
        else:
            check(f"N2: {tag}: P on the search grid (per_sd 3) and the fine grid (per_sd 8) agree (relative 1e-4)", dg < 1e-4, f'({dg:.1e})', dg, 5e-4)
        check(f"N2: K = {o['K']}, sigma = {o['sigma']:g}, y/sigma <= {o.get('ymax')}: the state-wise theorem still holds at the P-minimising state", o['bound_over_sigma'] <= o['gap'] + 1e-9)
        if o.get('ymax') is not None:
            check(f"N2: {tag}: the restriction is respected (y/sigma <= Y + 1e-3) and P stays >= 0.84", o['y_over_sigma'] <= o['ymax'] + 1e-3 and o['P_fine'] >= 0.84, f"(y/sigma {o['y_over_sigma']:.4f}, P {o['P_fine']:.4f})")

# ================= N3 =================
if 'N3' in Sm:
    E = Sm['N3']['ell_curve']; g = [o['gap'] for o in E]; i = int(np.argmin(g))
    check('N3: the gap as a function of the piece length has an interior minimum (between 0.3 and 0.6 sqrt sigma)', 0 < i < len(E) - 1 and 0.3 <= E[i]['ell'] <= 0.6, f"(min at l = {E[i]['ell']})")
    cv = [o['curvature'] for o in E if o['ell'] <= 0.44]; dv = max(abs(abs(c) * np.sqrt(2) - 1) for c in cv)
    check('N3: for l <= 0.44 sqrt(sigma) the optimal curvature equals the turning rate 1/sqrt2 within 1 % (it rises for longer pieces)', dv < 0.01,
          f"({[round(abs(c), 4) for c in cv]})", dv, 0.05)
    for o in Sm['N3']['by_m']:
        check(f"N3: m = {o['m']}: the tangent bundle is within 2 % of the estimate m^2/2 and far below the straight Gaussian", abs(o['gap_over_estimate'] - 1) < 0.02 and o['gap'] < 0.8 * o['estimate_straight_gaussian'],
              f"({o['gap']:.5f} vs {o['estimate_tangent_bundle']:.4f})", abs(o['gap_over_estimate'] - 1), 0.1)
    ls = [o['ell_over_sqrt_sigma'] for o in Sm['N3']['by_m']]
    check('N3: the optimal piece length does not depend on m (same turning rate): spread < 5 %', (max(ls) - min(ls)) / np.mean(ls) < 0.05, f'({[round(v, 3) for v in ls]})',
          (max(ls) - min(ls)) / np.mean(ls), 0.25)
    from n3_tangent import gapf
    o = [x for x in J('n3_tangent')['ell'] if x['ell'] == 0.44][0]; gv = gapf(np.array(o['x']), o['n'], 'twist', 0.25, gh=4, per_sd=6)
    check('N3: the l = 0.44 point is re-evaluated from its parameters', abs(gv - o['gap']) < 1e-9, f'({gv:.8f})', abs(gv - o['gap']), 5e-9)

ok, dev, where, _ = reproduce('results/summary_N.json', 'n_analyze.py', 1e-9, {'opt': 1e-6, 'fit': 1e-5, 'extrap': 1e-6, 'coef': 1e-6, 'gain': 1e-6, 'ratio': 1e-6, 'gap': 1e-6, 'scaling': 1e-6})
check('summary_N.json reproduces from the stored results (n_analyze.py)', ok, f'(max rel dev {dev:.1e} at {where})', dev, 5e-9)
finish()
