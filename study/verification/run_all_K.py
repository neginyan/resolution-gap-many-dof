"""Verification for stage K (corrected stage-J theorem; smoothed focus weight).  Exit 0 iff all checks pass."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
from vtools import check, finish, reproduce
from j2_smooth import evaluate, minorant, LAMBDAS
from i2_bound import states

J = [o for o in json.load(open('results/j1_weighted_cr.json')) if 'error' not in o]
S2 = [o for o in json.load(open('results/j2_smooth.json')) if 'error' not in o]; St = states()
L = [str(l) for l in LAMBDAS]
# ---- corrected stage-J theorem
check('corrected stage-J theorem (eps divided by J_tot exactly): bound <= actual 1 - R on all 290 states', len(J) == 290 and all(o['c_state'] <= o['gap_over_sigma'] + 1e-12 for o in J))
fac = [o['eps_factor'] for o in J]
check('the factor j/(sigma^2 J_tot) multiplying eps lies in [0.93, 1.16] (as found independently)', 0.93 < min(fac) and max(fac) < 1.16, f'({min(fac):.4f} .. {max(fac):.4f})')
check('the stage-J form (whole numerator divided by j/sigma^2 + J_ww) was valid on every state: positive part >= 1.6 x eps j/sigma^2',
      min(o['pos_part_over_eps'] for o in J if o['eps_over_sigma'] > 0) > 1.6, f"(min ratio {min(o['pos_part_over_eps'] for o in J if o['eps_over_sigma'] > 0):.3f})")
# ---- refined step (6): min_y [T^2 s^2/(8y) + k y]/(1+y) >= T'/(1 + T'/k),  T' = T s sqrt(k/2)
rng = np.random.default_rng(4); worst = np.inf
for _ in range(2000):
    T, s, k = rng.uniform(0.05, 3), 10 ** rng.uniform(-5, -0.3), rng.uniform(0.2, 2.5); y = np.logspace(-12, 4, 20001)
    Tp = T * s * np.sqrt(k / 2); worst = min(worst, np.min((T * T * s * s / (8 * y) + k * y) / (1 + y)) / (Tp / (1 + Tp / k)))
check('refined step (6): min_y [Theta^2 sigma^2/(8y) + kappa2 y]/(1+y) >= T/(1 + T/kappa2), T = Theta sigma sqrt(kappa2/2) (2000 random cases)', worst >= 1 - 1e-9, f'(min ratio {worst:.6f})')
# ---- the minorant: chi <= w, sqrt(chi) (L/2)-Lipschitz, and maximal (brute force on small random arrays)
ok = True
for _ in range(200):
    w = rng.random((3, 40)) ** 3 * (rng.random((3, 40)) > 0.2); Lr, h = rng.uniform(0.1, 5), 0.1
    c = minorant(w, Lr, h); bf = np.min(np.sqrt(w)[:, None, :] + (Lr / 2) * h * np.abs(np.arange(40)[:, None] - np.arange(40)[None, :])[None], axis=2) ** 2
    ok &= np.all(c <= w + 1e-15) and np.all(np.abs(np.diff(np.sqrt(c), axis=1)) <= Lr / 2 * h + 1e-12) and np.allclose(c, bf, atol=1e-14)
check('minorant: chi <= w, sqrt(chi) (L/2)-Lipschitz, equal to the brute-force largest such function (200 random arrays)', ok)
# ---- the steps of the smoothed chain on every state and every lambda
ck = lambda key: min(o['by_lambda'][l][key] for o in S2 for l in L)
check('all 290 states evaluated', len(S2) == 290)
check("E[g w] >= E[g chi] (chi <= w, g >= 0) on every state and lambda", ck('check_Egw_ge_Egchi') >= -1e-15, f"(min {ck('check_Egw_ge_Egchi'):.1e})")
check("(4') |E[u d_w chi]| <= L E[|u| sqrt chi] on every state and lambda (finite-difference derivative)", ck('check_slope_le_L') >= 0, f"(min slack {ck('check_slope_le_L'):.1e})")
check("(3')+(4') Z = E[chi u^2] >= (j_chi G_chi)^2/(sqrt A_chi + L sqrt P_chi)^2 on every state and lambda", ck('check_Z_ge_Zlow') >= 0, f"(min slack {ck('check_Z_ge_Zlow'):.1e})")
check("A_chi <= J_ww on every state and lambda", ck('check_A_le_Jww') >= 0)
check("discrete Lipschitz constant of sqrt(chi) <= L/2 on every state and lambda", max(o['by_lambda'][l]['lipschitz_ratio'] for o in S2 for l in L) <= 1 + 1e-9)
check('smoothed bound (refined) <= actual 1 - R on every state and lambda', all(o['by_lambda'][l]['c_bound2'] <= o['gap_over_sigma'] + 1e-12 for o in S2 for l in L))
sm = json.load(open('results/summary_K.json'))
check('concentrated class (eps2 + D <= 0.05 sigma at lambda = 1; 236 states): smoothed bound >= 0.38 sigma at lambda = 1',
      sm['n_concentrated'] == 236 and sm['by_lambda']['1.0']['min_bound2_concentrated'] >= 0.38, f"({sm['by_lambda']['1.0']['min_bound2_concentrated']:.4f})")
o = min([x for x in S2 if x['sigma'] <= 1e-2], key=lambda x: x['gap_over_sigma']); st = [s for s in St if s[0] == o['tag'] and abs(s[4] - o['sigma']) < 1e-15]
fr = min((evaluate(*s[1:]) for s in st), key=lambda r: abs(r['gap_over_sigma'] - o['gap_over_sigma']))
check(f"fresh recomputation of the smallest-gap state ({o['tag']}): smoothed quantities at lambda = 1 reproduced",
      all(abs(fr['by_lambda']['1.0'][k] - o['by_lambda']['1.0'][k]) < 1e-9 for k in ['alpha', 'G', 'Theta_chi', 'c_bound2']), f"(bound {fr['by_lambda']['1.0']['c_bound2']:.6f})")
ok, d, where, _ = reproduce('results/summary_K.json', 'k_analyze.py')
check('summary_K.json reproducible (relative 1e-9; stored file restored)', ok, f'(max rel dev {d:.1e} at {where})')
finish()
