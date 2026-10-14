"""Stage J: tightening step (B) of the lower-bound chain with a WEIGHTED Cramer-Rao inequality along W (pendulum, kappa = 1).

Notation: Y = X + sigma Z, p = p_Y, s = grad log p, H = -grad^2 log p, posterior mean m(y) = y + sigma^2 s(y) (Tweedie),
posterior covariance sigma^2 I - sigma^4 H >= 0, so H <= I/sigma^2.  V/W components; delta = X_q - pi; e_q = (V + W)/sqrt 2.
  g(y)  = 1 - abar(y) = E[sin^2(delta/2) | y]                              (missing stretch felt at y)
  w(y)  = sigma^2 (H_vv(y))_+  in [0, 1]                                    (V-focus weight; <= 1 by H <= I/sigma^2)
  u(y)  = sqrt 2 (m_q(y) - pi)                                              (posterior-mean distance from the top, in q)

Chain (every step rigorous for every state; only integrability is assumed):
 (E)  J_tot (1 - f) = E[g H_vv] - E[g H_ww] + 2 J_ww                                          [exact identity]
 (1)  E[g H_vv] = E[g w]/sigma^2 - N,  N = E[g (H_vv)_-]                                      [split of H_vv]
 (2)  g >= u^2/8 + Var(X_q|y)/4 - E[delta^4|y]/48                                             [sin^2(x/2) >= x^2/4 - x^4/48, all x;
                                                                                                E[delta^2|y] = u^2/2 + Var(X_q|y)]
 (3)  E[w u^2] >= Lambda^2 / E[w s_w^2],   Lambda = -E[w u s_w]                                 [Cauchy-Schwarz]
 (4)  (not needed) E[w s_w^2] <= J_ww since w <= 1; instead keep the ratio r = J_ww / E[w s_w^2] >= 1
 (5)  J_vv <= E[(H_vv)_+] = j/sigma^2,  j = E[w]                                                [definition]
 =>  with Theta' = (|Lambda|/j) sqrt(r),  x = J_ww,  eps = (E[w (E[delta^4|y]/48 - Var(X_q|y)/4)] + sigma^2 N + sigma^2 E[g H_ww]) / j:
      1 - f >= [Theta'^2 sigma^2/(8 y) + 2 y - eps] / (1 + y),   y = sigma^2 x / j
 (6)  split the numerator into its positive part P = Theta'^2 j^2/(8 sigma^2 x) + 2x >= 0 and the rest -(j/sigma^2) eps;
      divide P by J_tot <= j/sigma^2 + x (valid because P >= 0) and the rest by J_tot exactly;
      min over y > 0 (AM-GM on y <= Theta' sigma/2, 2y/(1+y) beyond):  P/(j/sigma^2 + x) >= Theta' sigma/(1 + Theta' sigma/2)
 =>  1 - R  >=  Theta' sigma/(1 + Theta' sigma/2) - eps_tilde - D,   eps_tilde = eps j/(sigma^2 J_tot)   (STATE-WISE THEOREM)
 (stage J wrote max(eps, 0) instead of eps_tilde; that form needs the whole numerator to be positive -- true on all 290 states,
  where the positive part exceeds eps j/sigma^2 by a factor >= 1/0.6, but not guaranteed; corrected in stage K.)
 The exact-form bound LB2 (steps (E), (1)-(3) and (5) only, J_ww and E[w s_w^2] kept as they are) is also reported.
Integration by parts along W (boundary terms vanish) splits Theta:
      Theta j = -E[w u s_w] = E[w d_w u] + E[u d_w w],   d_w u = 1 - sqrt 2 sigma^2 e_q^T H W
  "geometric part"  E[w d_w u]/j   (= 1 when sigma^2 H is small: the single-Gaussian value)
  "focus-slope part" E[u d_w w]/j  (how the V-focus weight falls along W, away from the top; the fan makes it negative).
A uniform constant c needs a lower bound on Theta' over all states (and eps, D = O(sigma^2)) -- that is the open step.
The script evaluates every quantity on every stored pendulum state (the i2_bound list).  Writes results/j1_weighted_cr.json."""
import numpy as np, json, sys
from multiprocessing import Pool
from i_tools import grid, V, W
from i2_bound import states



def chain(ws, mus, Ss, sig, per_sd=6, chunk=150_000):
    Y, dA, n = grid(ws, mus, Ss, sig, per_sd=per_sd); s2 = sig * sig; K = len(ws)
    T = [S + s2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) - np.log(2 * np.pi) for w, t in zip(ws, T)]
    Kg = [S @ ti for S, ti in zip(Ss, Ti)]; Cp = [S - k @ S for S, k in zip(Ss, Kg)]
    acc = {k: 0.0 for k in ['mass', 'num', 'Jvv', 'Jww', 'Jvw', 'Egw', 'N', 'Bw', 'j', 'Ews2', 'Lam', 'geo', 'd4', 'varq', 'gHvv', 'f_num', 'trH', 'wu2']}
    minGap2 = np.inf                                                     # min over the grid of g - (bound (2)), must be >= 0
    for c in range(0, len(Y), chunk):
        y = Y[c:c + chunk]
        logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', y - m, ti, y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
        mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True)
        u_k = np.stack([-(y - m) @ ti for m, ti in zip(mus, Ti)], 1); s = np.einsum('nk,nkd->nd', pi, u_k)
        H = np.einsum('nk,kij->nij', pi, np.array(Ti)) - (np.einsum('nk,nki,nkj->nij', pi, u_k, u_k) - np.einsum('ni,nj->nij', s, s))
        Ef = np.zeros_like(y); abar = np.zeros(len(y)); Ed2 = np.zeros(len(y)); Ed4 = np.zeros(len(y))
        for k in range(K):
            mp = mus[k] + (y - mus[k]) @ Kg[k].T; v = Cp[k][0, 0]; mu = mp[:, 0] - np.pi
            Ef += pi[:, k:k + 1] * np.stack([mp[:, 1], -np.sin(mp[:, 0]) * np.exp(-v / 2)], -1)
            abar += pi[:, k] * (1 - np.cos(mp[:, 0]) * np.exp(-v / 2)) / 2
            Ed2 += pi[:, k] * (mu ** 2 + v); Ed4 += pi[:, k] * (mu ** 4 + 6 * mu ** 2 * v + 3 * v ** 2)
        pw = p * dA
        Hvv = np.einsum('i,nij,j->n', V, H, V); Hww = np.einsum('i,nij,j->n', W, H, W); Hqw = np.einsum('i,nij,j->n', np.array([1.0, 0.0]), H, W)
        g = np.clip(1 - abar, 0, 1); w = np.clip(s2 * np.maximum(Hvv, 0), 0, 1)
        mq = y[:, 0] + s2 * s[:, 0]; u = np.sqrt(2) * (mq - np.pi); varq = Ed2 - (mq - np.pi) ** 2
        sw = s @ W; dwu = 1 - np.sqrt(2) * s2 * Hqw
        acc['mass'] += pw.sum(); acc['num'] += np.sum(pw * (s * Ef).sum(1))
        acc['Jvv'] += np.sum(pw * Hvv); acc['Jww'] += np.sum(pw * Hww); acc['Egw'] += np.sum(pw * g * w); acc['N'] += np.sum(pw * g * np.maximum(-Hvv, 0))
        acc['Bw'] += np.sum(pw * g * Hww); acc['j'] += np.sum(pw * w); acc['Ews2'] += np.sum(pw * w * sw ** 2); acc['Lam'] += -np.sum(pw * w * u * sw)
        acc['geo'] += np.sum(pw * w * dwu); acc['d4'] += np.sum(pw * w * Ed4); acc['varq'] += np.sum(pw * w * varq)
        acc['wu2'] += np.sum(pw * w * u ** 2); keep = pw > 1e-300
        minGap2 = min(minGap2, float(np.min((g - (u ** 2 / 8 + varq / 4 - Ed4 / 48))[keep])))
        acc['gHvv'] += np.sum(pw * g * Hvv); acc['f_num'] += np.sum(pw * abar * (Hvv - Hww)); acc['trH'] += np.sum(pw * (Hvv + Hww))
    M = acc['mass']; a = {k: v / M for k, v in acc.items() if k != 'mass'}
    Jtot = a['Jvv'] + a['Jww']
    R = a['num'] / (s2 * Jtot)                       # R = E[s . E[f|y]]/(sigma^2 E|s|^2), E|s|^2 = E tr H = J_tot
    f = a['f_num'] / Jtot; D = R - f
    j = a['j']; Theta = abs(a['Lam']) / j; ThetaP = Theta * np.sqrt(a['Jww'] / a['Ews2'])     # step (4) not needed: Theta' >= Theta
    eps = (a['d4'] / 48 - a['varq'] / 4 + s2 * a['N'] + s2 * a['Bw']) / j
    # exact-form bound before the y-minimisation (uses (1)-(5) only):
    x = a['Jww']; num = (a['Lam'] ** 2 / (8 * a['Ews2']) - a['d4'] / 48 + a['varq'] / 4) / s2 - a['N'] - a['Bw'] + 2 * x
    LB2 = num / Jtot
    eps_t = eps * j / (s2 * Jtot)       # stage K correction: the eps part is divided by J_tot exactly (no sign condition needed)
    cstate = ThetaP / (1 + ThetaP * sig / 2) - (eps_t + D) / sig
    cstate_old = ThetaP / (1 + ThetaP * sig / 2) - (max(eps, 0) + D) / sig
    return {'sigma': sig, 'R': R, 'f': f, 'D': D, 'gap_over_sigma': (1 - R) / sig, 'j': j, 'Theta': Theta, 'ThetaP': ThetaP, 'info_ratio': a['Jww'] / a['Ews2'],
            'Theta_signed': a['Lam'] / j, 'Theta_geometric': a['geo'] / j, 'Theta_slope': (a['Lam'] - a['geo']) / j,
            'eps_over_sigma': eps / sig, 'eps_tilde_over_sigma': eps_t / sig, 'eps_factor': j / (s2 * Jtot), 'c_state_stageJ': cstate_old,
            'pos_part_over_eps': (a['Lam'] ** 2 / (8 * a['Ews2']) / s2 + 2 * x) / (eps * j / s2) if eps != 0 else np.inf, 'D_over_sigma': D / sig, 'LB2_over_sigma': (LB2 - D) / sig, 'c_state': cstate,
            'Jvv_s2': a['Jvv'] * s2, 'Jww': x, 'mass': M, 'n': n,
            'checks': {'identity': (a['gHvv'] - a['Bw'] + 2 * x) / Jtot - (1 - f), 'step2_min_slack': minGap2,
                       'step3_slack': a['wu2'] - a['Lam'] ** 2 / a['Ews2'], 'step4_slack': x - a['Ews2'], 'step5_slack': j / s2 - a['Jvv'],
                       'w_max_excess': 0.0, 'LB2_le_1mf': LB2 - (1 - f)}}


def run(st):
    tag, ws, mus, Ss, sig = st
    try: o = chain(ws, mus, Ss, sig)
    except (MemoryError, ValueError) as e: return {'tag': tag, 'sigma': sig, 'error': str(e)}
    o['tag'] = tag; return o


if __name__ == '__main__':
    S = states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/j1_weighted_cr.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o]; small = [o for o in ok if o['sigma'] <= 1e-2]
    print('errors', len(out) - len(ok), '| identity max err', max(abs(o['checks']['identity']) for o in ok))
    for key in ['gap_over_sigma', 'LB2_over_sigma', 'c_state', 'ThetaP', 'Theta', 'Theta_geometric', 'Theta_slope']:
        v = min(small, key=lambda o: o[key]); print(f'min {key}: {v[key]:.4f} ({v["tag"]}, sigma {v["sigma"]:g}, j {v["j"]:.3f})')
    for key in ['eps_over_sigma', 'D_over_sigma']:
        v = max(small, key=lambda o: o[key]); print(f'max {key}: {v[key]:.4f} ({v["tag"]}, sigma {v["sigma"]:g})')
