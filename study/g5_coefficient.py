"""Stage G, task 2: the coefficient c in rho*CV*Omega = 1 - c t along the needle sequence of stage F (f5_needle.py), from law 1.

Pendulum facts used (exact):
  * Sym E[Df] = lambda * diag(1, -1) in the V/W basis for ANY state (the qp entry of Sym J is (1 - E cos q)/2 = E a = lambda).
  * Law 1 for a single Gaussian: R_n = lambda_n * phi_n,  phi_n = (F_vv - F_ww)/tr F = (T_ww - T_vv)/(T_ww + T_vv),  T = S + sigma^2 I
    in V/W coordinates (the V/W off-diagonal of T drops out).
Needle limit (the needle carries all the Fisher information, D -> 0, the mixture's f -> R_needle):
    law -> lambda_rho * phi_n,   rho CV Omega = (f - law)/(1 - lambda_rho) -> phi_n (lambda_n - lambda_rho)/(1 - lambda_rho)
    =>  1 - rho CV Omega  ~  (1 - phi_n) + phi_n (1 - lambda_n)/(1 - lambda_rho)                         (prediction P)
Scaling of f5 (sigma -> t sigma0, needle T_vv -> t^2 (S_vv + sigma0^2), S_ww -> t S_ww, centre fixed at q_k):
    1 - phi_n = 2 T_vv/(T_vv + T_ww) = 2 (S_vv + sigma0^2) t / S_ww + O(t^2)
    1 - lambda_n = (1 + cos q_k)/2 + (-cos q_k) S_qq t / 4 + O(t^2),  S_qq = (S_vv t + 2 S_vw sqrt(t) + S_ww)/2 * t -> S_ww t / 2
    =>  1 - rho CV Omega = c0 + c t + O(t^2),  c0 = (1 + cos q_k)/(2 (1 - lambda_rho0)),
        c = 2 (S_vv + sigma0^2)/S_ww + (-cos q_k) S_ww / (8 (1 - lambda_rho0))
    (lambda_rho0 = lambda_rho at t -> 0; the needle's own S_vv, S_ww at t = 1).
The same picture gives 1 - fraction ~ (1 - R_needle)/(1 - law): the ruler effect.
Writes results/g5_coefficient.json."""
import numpy as np, json
from obs_eval2 import V, W

Rm = np.stack([V, W], 1)


def lam(mu, S):
    return (1 - np.cos(mu[0]) * np.exp(-S[0, 0] / 2)) / 2


if __name__ == '__main__':
    F = json.load(open('results/f1_chains.json')); b = max(F, key=lambda r: r['final']['ratio'])
    ws, mus, Ss, sig0 = b['ws'], [np.array(m) for m in b['mus']], [np.array(S) for S in b['Ss']], b['sigma']
    k = int(np.argmax([w * np.trace(np.linalg.inv(S + sig0 ** 2 * np.eye(2))) for w, S in zip(ws, Ss)]))
    Svw = Rm.T @ Ss[k] @ Rm; qk = mus[k][0]
    rows = json.load(open('results/f5_needle.json'))['rows']

    def pieces(t):
        D = np.diag([t ** 2, t]); Sk = Rm @ (np.sqrt(D) @ Svw @ np.sqrt(D)) @ Rm.T; s = sig0 * t
        T = Rm.T @ (Sk + s ** 2 * np.eye(2)) @ Rm; phi = (T[1, 1] - T[0, 0]) / (T[1, 1] + T[0, 0])
        lam_n = lam(mus[k], Sk); lam_rho = sum(w * lam(m, (Sk if i == k else S)) for i, (w, m, S) in enumerate(zip(ws, mus, Ss)))
        return phi, lam_n, lam_rho

    lam_rho0 = sum(w * (lam(m, S) if i != k else (1 - np.cos(qk)) / 2) for i, (w, m, S) in enumerate(zip(ws, mus, Ss)))
    c0 = (1 + np.cos(qk)) / (2 * (1 - lam_rho0))
    c_phi = 2 * (Svw[0, 0] + sig0 ** 2) / Svw[1, 1]; c_lam = -np.cos(qk) * Svw[1, 1] / (8 * (1 - lam_rho0)); c = c_phi + c_lam
    out = []
    for r in rows:
        t = r['t']; phi, ln, lr = pieces(t)
        P = (1 - phi) + phi * (1 - ln) / (1 - lr)
        fr = r['factors']
        out.append({'t': t, 'measured': 1 - fr['rhoCVOmega'], 'prediction_exact_law1': P, 'prediction_linear': c0 + c * t,
                    'phi_n': phi, 'lam_n': ln, 'lam_rho': lr, 'lam_rho_measured': 1 - fr['one_minus_lam'],
                    'one_minus_fraction': 1 - fr['ratio'], 'ruler_prediction': (1 - r['R_needle_alone']) / (1 - r['law']),
                    'R_needle_law1': ln * phi, 'R_needle_grid': r['R_needle_alone']})
    res = {'needle_index': k, 'q_k': qk, 'S_vv': Svw[0, 0], 'S_ww': Svw[1, 1], 'S_vw': Svw[0, 1], 'sigma0': sig0, 'lam_rho0': lam_rho0,
           'c0': c0, 'c': c, 'c_phi_part': c_phi, 'c_lam_part': c_lam, 'rows': out}
    json.dump(res, open('results/g5_coefficient.json', 'w'), indent=1, default=float)
    print(f"needle q_k - pi = {qk - np.pi:.3e}; c0 = {c0:.3e}; c = {c:.5f} (focus part {c_phi:.5f}, stretch part {c_lam:.5f}); lambda_rho0 = {lam_rho0:.5f}")
    for o in out:
        print(f"t={o['t']:<5} 1-rCVO measured {o['measured']:.4e}  law-1 prediction {o['prediction_exact_law1']:.4e}  c0+ct {o['prediction_linear']:.4e}"
              f"  | 1-fraction {o['one_minus_fraction']:.4e} ruler {o['ruler_prediction']:.4e} | R_needle law1-grid {o['R_needle_law1'] - o['R_needle_grid']:.1e}")
