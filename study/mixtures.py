"""Preliminary look at two-component Gaussian mixtures (pendulum), where the single-Gaussian law no longer applies.

For each state and sigma we record the exact R (quadrature) and three reference quantities:
  lam_rho  = lambda_max(Sym E_rho[Df])              (data-averaged Jacobian of the whole mixture)
  lam_comp = max_k lambda_max(Sym E_k[Df])            (largest component-averaged stretch)
  R_sep    = sum_k w_k tr(F_k Sym J_k) / sum_k w_k tr F_k,  F_k = (S_k + sigma^2 I)^{-1}
             (the single-Gaussian law applied component by component; exact if the components do not overlap)
  overlap  = Bhattacharyya coefficient of the two blurred components (0: disjoint, 1: identical).
Writes results/mixtures.json."""
import numpy as np, json, os
from obs_eval import evaluate

def jbar(mu, S):
    return np.array([[0, 1.0], [-np.cos(mu[0]) * np.exp(-S[0, 0] / 2), 0]])

def sym(J):
    return (J + J.T) / 2

def references(ws, mus, Ss, sig):
    T = [S + sig ** 2 * np.eye(2) for S in Ss]
    J = [jbar(m, S) for m, S in zip(mus, Ss)]
    Jrho = sum(w * j for w, j in zip(ws, J))
    lam_rho = np.linalg.eigvalsh(sym(Jrho)).max()
    lam_comp = max(np.linalg.eigvalsh(sym(j)).max() for j in J)
    Fk = [np.linalg.inv(t) for t in T]
    R_sep = sum(w * np.trace(F @ sym(j)) for w, F, j in zip(ws, Fk, J)) / sum(w * np.trace(F) for w, F in zip(ws, Fk))
    Tm = (T[0] + T[1]) / 2; d = mus[0] - mus[1]
    DB = d @ np.linalg.solve(Tm, d) / 8 + 0.5 * np.log(np.linalg.det(Tm) / np.sqrt(np.linalg.det(T[0]) * np.linalg.det(T[1])))
    return lam_rho, lam_comp, R_sep, float(np.exp(-DB))

def random_state(rng):
    ws = rng.dirichlet([2, 2])
    mus = [np.array([rng.uniform(0, 2 * np.pi), rng.normal(0, 1.0)]) for _ in range(2)]
    Ss = []
    for _ in range(2):
        A = rng.normal(0, 1, (2, 2)) * 10 ** rng.uniform(-1.3, -0.2)
        Ss.append(A @ A.T + 1e-4 * np.eye(2))
    return list(ws), mus, Ss

if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    out = []
    for i in range(250):
        ws, mus, Ss = random_state(rng)
        for sig in [0.03, 0.1, 0.3, 1.0]:
            r = evaluate(ws, mus, Ss, sig)
            lam_rho, lam_comp, R_sep, ov = references(ws, mus, Ss, sig)
            out.append({'i': i, 'sigma': sig, 'R': r['R'], 'lam_rho': lam_rho, 'lam_comp': lam_comp, 'R_sep': R_sep,
                        'overlap': ov, 'n': r['n'], 'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss]})
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/mixtures.json', 'w'), indent=1, default=float)
    R = np.array([o['R'] for o in out]); lr = np.array([o['lam_rho'] for o in out]); lc = np.array([o['lam_comp'] for o in out])
    Rs = np.array([o['R_sep'] for o in out]); ov = np.array([o['overlap'] for o in out])
    print('states x sigmas:', len(out), ' max R:', R.max(), ' all R <= 1:', bool((R <= 1 + 1e-9).all()))
    print('R > lam_rho in', int((R > lr + 1e-9).sum()), 'cases;  R > lam_comp in', int((R > lc + 1e-9).sum()), 'cases')
    for lo, hi in [(0, 1e-3), (1e-3, 0.05), (0.05, 0.3), (0.3, 1.01)]:
        m = (ov >= lo) & (ov < hi)
        if m.any():
            print(f'overlap in [{lo},{hi}): n={m.sum():4d}  max|R - R_sep| = {np.abs(R - Rs)[m].max():.4f}  median = {np.median(np.abs(R - Rs)[m]):.2e}')
