"""Stage M, task M4: towards the dichotomy for a uniform gap (pendulum, kappa = 1).
On every stored state (stages A-K: i2_bound.states(); stage L: l1_ratios.new_states()) measure, besides the quantities of l_tools.diag:
  d       = sqrt(E[w u^2] / (j sigma))         rms distance of the focus weight w from the top, in units of sqrt(sigma)
                                               (u = sqrt2 (m_q - pi) the posterior-mean distance, j = E[w]);
  law_def = (kappa - lambda_rho)/sigma,  lambda_rho = E_rho[(1 - cos q)/2]  (the law-term eigenvalue of the PRIOR; kappa - lambda_rho
            is the "deficit of the law term");
  Zbound  = the part of the stage-J chain that uses only the focus distance (no Cramer-Rao step):
            1 - f >= (E[g w]/sigma^2 - N + kappa2 J_ww)/J_tot  with  E[g w] >= Z/8 - E[w (E[delta^4|y]/48 - Var(X_q|y)/4)],
            kappa2 = 2 - E[g H_ww]/J_ww;  reported as (value - D)/sigma, a rigorous state-wise lower bound on (1 - R)/sigma.
  y = sigma^2 J_ww / j (W-information relative to the focus), P = Z J_ww / j^2 = d^2 (y/sigma)  ("focus spread x W-sharpness");
  the Z-bound written with these variables:  (1 - f)/sigma >= [d^2/8 - corr/(j sigma) + kappa2 y/sigma - N sigma/j]/(1 + y)  (when
  J_tot = j/sigma^2 + J_ww, i.e. no negative H_vv; reported as 'two_variable_form').  The weighted Cramer-Rao step of paper B is
  P >= Theta'^2; a state-independent lower bound P >= P0 would give, with kappa2 ~ 2, the uniform constant c = sqrt(P0) (AM-GM).
The candidate dichotomy:  Theta' >= geo (1 - rho)  (proved, stage L)  handles states with small rho; the Z-bound handles states whose
focus sits far from the top (large d).  It closes if rho <= a + b d  (or any increasing function of d) held for all states.
-> results/m4_dichotomy.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from j2_smooth import arrays
from l_tools import diag


def extra(ws, mus, Ss, sig):
    A = arrays(ws, mus, Ss, sig, per_sd=6, max_points=int(min(2_500_000, 2.4e7 / len(ws)))); p = A['pw']; s2 = A['s2']
    w, u, g = A['w'], A['u'], A['g']; j = np.sum(p * w); Z = np.sum(p * w * u * u)
    Jww = np.sum(p * A['Hww']); Jtot = np.sum(p * A['Hvv']) + Jww
    N = np.sum(p * g * np.maximum(-A['Hvv'], 0)); k2 = 2 - np.sum(p * g * A['Hww']) / Jww
    corr = np.sum(p * w * (A['Ed4'] / 48 - A['varq'] / 4))
    zb = ((Z / 8 - corr) / s2 - N + k2 * Jww) / Jtot
    lam = sum(wk * (1 - np.cos(m[0]) * np.exp(-S[0, 0] / 2)) / 2 for wk, m, S in zip(ws, mus, Ss))
    y = s2 * Jww / j
    return {'d': float(np.sqrt(Z / (j * sig))), 'y_over_sigma': float(y / sig), 'P': float(Z * Jww / (j * j)), 'law_def': float((1 - lam) / sig),
            'Zbound_over_sigma': float((zb - (A['R'] - A['f'])) / sig), 'kappa2': float(k2),
            'two_variable_form': float(((Z / 8 - corr) / (j * sig) + k2 * y / sig - N * s2 / j / sig) / (1 + y) - (A['R'] - A['f']) / sig)}


def run(st):
    tag, ws, mus, Ss, sig = st
    try: d = diag(ws, mus, Ss, sig); d.update(extra(ws, mus, Ss, sig))
    except (MemoryError, ValueError) as e: return {'tag': tag, 'sigma': sig, 'error': str(e)}
    d['tag'] = tag; d.pop('n', None); return d


if __name__ == '__main__':
    from i2_bound import states
    from l1_ratios import new_states
    S = states() + new_states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool: out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/m4_dichotomy.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o and o['sigma'] <= 1e-2]
    print('errors', sum('error' in o for o in out))
    for t in [0.2, 0.25, 0.3, 0.5]:
        v = [o for o in ok if o['rho'] > t]
        if v: print(f"rho > {t}: {len(v)} states, min d {min(o['d'] for o in v):.3f}, min gap {min(o['gap_over_sigma'] for o in v):.3f}, min law_def {min(o['law_def'] for o in v):.3f}")
    print('min Zbound', min(o['Zbound_over_sigma'] for o in ok))
