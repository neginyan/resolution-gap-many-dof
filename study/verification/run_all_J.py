"""Verification for stage J (weighted Cramer-Rao lower bound).  Exit 0 iff all checks pass."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from j1_weighted_cr import chain
from i2_bound import states
from i_tools import fields, V, W

J = [o for o in json.load(open('results/j1_weighted_cr.json')) if 'error' not in o]; S = states()
check('all 290 states evaluated without error', len(J) == len(S) == 290, f'({len(J)} of {len(S)})')
# ---- each step of the chain, on every state
check('(E) exact identity J_tot(1 - f) = E[g H_vv] - E[g H_ww] + 2 J_ww on every state', max(abs(o['checks']['identity']) for o in J) < 1e-13,
      f"(max err {max(abs(o['checks']['identity']) for o in J):.1e})")
check('(2) g >= u^2/8 + Var(X_q|y)/4 - E[delta^4|y]/48 at every grid point of every state (slack >= -1e-12)', min(o['checks']['step2_min_slack'] for o in J) >= -1e-12,
      f"(min slack {min(o['checks']['step2_min_slack'] for o in J):.1e})")
check('(3) Cauchy-Schwarz E[w u^2] >= Lambda^2 / E[w s_w^2] on every state', min(o['checks']['step3_slack'] for o in J) >= 0)
check('(4) E[w s_w^2] <= J_ww on every state (w <= 1 from H <= I/sigma^2)', min(o['checks']['step4_slack'] for o in J) >= 0)
check('(5) J_vv <= j/sigma^2 on every state (relative rounding < 1e-12)', min(o['checks']['step5_slack'] / (o['Jvv_s2'] / o['sigma'] ** 2) for o in J) > -1e-12)
check('exact-form bound LB2 <= 1 - f and closed-form bound <= actual on every state', all(o['checks']['LB2_le_1mf'] <= 1e-12 and o['c_state'] <= o['gap_over_sigma'] + 1e-12 for o in J))
# ---- (6): min over y of (T^2 s^2/(8y) + 2y)/(1+y) >= T s/(1 + T s/2), and the sine inequality
rng = np.random.default_rng(3); worst = np.inf
for _ in range(2000):
    T, s = rng.uniform(0.05, 3), 10 ** rng.uniform(-5, -0.3); y = np.logspace(-12, 4, 20001)
    worst = min(worst, np.min((T * T * s * s / (8 * y) + 2 * y) / (1 + y)) / (T * s / (1 + T * s / 2)))
check('(6) min_y [T^2 sigma^2/(8y) + 2y]/(1+y) >= T sigma/(1 + T sigma/2) (2000 random cases, dense y)', worst >= 1 - 1e-9, f'(min ratio {worst:.6f})')
x = np.linspace(-50, 50, 2_000_001)
check('sin^2(x/2) >= x^2/4 - x^4/48 for all x (|x| <= 50, 2e6 points)', np.min(np.sin(x / 2) ** 2 - (x ** 2 / 4 - x ** 4 / 48)) >= -1e-12)
# ---- fresh recomputation of the two decisive states
small = [o for o in J if o['sigma'] <= 1e-2]
for key in ['LB2_over_sigma', 'c_state']:
    o = min(small, key=lambda o: o[key]); st = [s for s in S if s[0] == o['tag'] and abs(s[4] - o['sigma']) < 1e-15]
    fr = [chain(*s[1:]) for s in st]; m = min(fr, key=lambda r: abs(r[key] - o[key]))
    check(f'fresh recomputation of the state with the smallest {key} ({o["tag"]}, sigma {o["sigma"]:g})', abs(m[key] - o[key]) < 1e-9, f'({m[key]:.6f})')
# ---- integration by parts: Lambda = E[w du/dw] + E[u dw/dw], with both derivatives by finite differences on the grid
st = [s for s in S if s[0] == 'h4_growk'][-1]; ws, mus, Ss, sig = st[1:]
def ibp(per_sd):
    r = fields(ws, mus, Ss, sig, per_sd=per_sd, need_H=True); F = r['fields']; n = r['n']; s2 = sig * sig
    p = F['p'].reshape(n); s = F['s'].reshape(n + (2,)); H = F['H'].reshape(n + (2, 2)); Y = r['Y'].reshape(n + (2,))
    w = np.clip(s2 * np.maximum(np.einsum('i,abij,j->ab', V, H, V), 0), 0, 1); u = np.sqrt(2) * (Y[..., 0] + s2 * s[..., 0] - np.pi); sw = s @ W
    hw = np.linalg.norm(Y[0, 1] - Y[0, 0])                               # grid step along W (axis 1)
    dwu = np.gradient(u, hw, axis=1); dww = np.gradient(w, hw, axis=1); M = p.sum()
    Lam = -np.sum(p * w * u * sw) / M; parts = (np.sum(p * w * dwu) + np.sum(p * u * dww)) / M
    return Lam, parts, np.sum(p * u * dww) / np.sum(p * w)
L1, P1, sl = ibp(10); L2, P2, _ = ibp(20)
# finite differences of w (which has kinks where H_vv changes sign) converge at first-to-second order: require 1e-3 and convergence
check('integration by parts along W: -E[w u s_w] = E[w d_w u] + E[u d_w w] (finite differences; per_sd 10 and 20, K = 8 state)',
      abs(L2 - P2) < 1e-3 * abs(L2) and abs(L2 - P2) < abs(L1 - P1), f'(rel. mismatch {abs(L1 - P1) / abs(L1):.1e} -> {abs(L2 - P2) / abs(L2):.1e})')
check('the focus-slope part is negative for this fan-like state (the escape route of mixtures)', sl < -0.1, f'({sl:+.4f})')
ok, d, where, _ = reproduce('results/summary_J.json', 'j_analyze.py')
check('summary_J.json reproducible (relative 1e-9; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')
finish()
