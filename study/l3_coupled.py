"""Stage L, task L3: the best Gaussian state for two coupled pendulums (4-d phase space),
    H = (p1^2 + p2^2)/2 - cos q1 - cos q2 - eps cos(q1 - q2),   x = (q1, q2, p1, p2),  f = (p, -grad V(q)).
Df = [[0, I], [-V''(q), 0]], Sym Df = [[0, M], [M, 0]] with M(q) = (I - V''(q))/2; its eigenvalues are +- eig M(q).
  kappa = max_q lambda_max M(q) = (1 + eps + sqrt(1 + eps^2))/2, attained at (q1, q2) = (pi, 0) (and (0, pi)) for eps > 0;
  for eps = 0, M = diag(a(q1), a(q2)) and kappa = 1 on the lines q1 = pi or q2 = pi.
Law 1 (paper B, theorem 3; valid in any dimension for divergence-free f and a Gaussian state N(m, S)):
  R = tr(F Sym Jbar)/tr F,  F = (S + sigma^2 I)^{-1},  Jbar = E[Df],  with the Gaussian averages
  E cos q_i = cos m_i e^{-S_ii/2},  E cos(q1 - q2) = cos(m1 - m2) e^{-(S11 + S22 - 2 S12)/2}.
Independent check: R = E_Y[s . E[f|Y]] / (sigma^2 E|s|^2) computed by a 4-d Gauss-Hermite grid over Y ~ N(m, S + sigma^2 I)
(posterior means of sin q_i, sin(q1 - q2) in closed form; no Stein identity used), and by Monte Carlo.
Parts:  check | single (R*(sigma) by full optimisation over m and the 10 entries of S) -> results/l3_<part>.json"""
import numpy as np, json, sys
from scipy.optimize import minimize
from multiprocessing import Pool

I2 = np.eye(2)


def kappa(eps):
    return (1 + eps + np.sqrt(1 + eps * eps)) / 2


def EVpp(m, S, eps):
    """E[V''(q)] for q ~ N(m, S_qq)."""
    c1 = np.cos(m[0]) * np.exp(-S[0, 0] / 2); c2 = np.cos(m[1]) * np.exp(-S[1, 1] / 2)
    c12 = np.cos(m[0] - m[1]) * np.exp(-(S[0, 0] + S[1, 1] - 2 * S[0, 1]) / 2)
    return np.array([[c1 + eps * c12, -eps * c12], [-eps * c12, c2 + eps * c12]])


def law1(m, S, sig, eps):
    M = (I2 - EVpp(m, S, eps)) / 2; SJ = np.block([[np.zeros((2, 2)), M], [M, np.zeros((2, 2))]])
    F = np.linalg.inv(S + sig ** 2 * np.eye(4)); return float(np.trace(F @ SJ) / np.trace(F))


def R_quadrature(m, S, sig, eps, n=12, mc=0, seed=0):
    """E_Y[s . E[f|Y]]/(sigma^2 E|s|^2) by an n^4 Gauss-Hermite grid over Y (or Monte Carlo with mc samples)."""
    T = S + sig ** 2 * np.eye(4); Ti = np.linalg.inv(T); K = S @ Ti; C = S - K @ S
    if mc:
        Y = np.random.default_rng(seed).multivariate_normal(m, T, size=mc); wts = np.full(mc, 1 / mc)
    else:
        x, w = np.polynomial.hermite_e.hermegauss(n); w = w / w.sum(); L = np.linalg.cholesky(T)
        Z = np.stack(np.meshgrid(x, x, x, x, indexing='ij'), -1).reshape(-1, 4); wts = np.einsum('a,b,c,d->abcd', w, w, w, w).ravel()
        Y = m + Z @ L.T
    s = -(Y - m) @ Ti; mp = m + (Y - m) @ K.T
    e1 = np.exp(-C[0, 0] / 2); e2 = np.exp(-C[1, 1] / 2); e12 = np.exp(-(C[0, 0] + C[1, 1] - 2 * C[0, 1]) / 2)
    s1 = np.sin(mp[:, 0]) * e1; s2 = np.sin(mp[:, 1]) * e2; s12 = np.sin(mp[:, 0] - mp[:, 1]) * e12
    Ef = np.stack([mp[:, 2], mp[:, 3], -(s1 + eps * s12), -(s2 - eps * s12)], -1)
    return float(np.sum(wts * (s * Ef).sum(1)) / (sig ** 2 * np.sum(wts * (s * s).sum(1))))


# ---------- optimisation of R over Gaussian states ----------
def top(eps):
    return np.array([np.pi, 0.0]) if eps > 0 else np.array([np.pi, np.pi])


def mode_frame(eps):
    """orthonormal frame (V+, W+, V-, W-) at the top: V+/W+ = stretching/contracting directions of the most stretching mode."""
    q0 = top(eps); c = np.cos(q0[0] - q0[1])
    Mt = (I2 - np.array([[np.cos(q0[0]) + eps * c, -eps * c], [-eps * c, np.cos(q0[1]) + eps * c]])) / 2
    ev, U = np.linalg.eigh(Mt); ep, em = U[:, 1], U[:, 0]
    return np.stack([np.concatenate([ep, ep]), np.concatenate([ep, -ep]), np.concatenate([em, em]), np.concatenate([em, -em])], 1) / np.sqrt(2), ev


def opt_frame(eps):
    """orthonormal frame (V+, W+, q-, p-): V+/W+ = stretching/contracting directions of the most stretching mode at the top,
    q- = (e-, 0) and p- = (0, e-) the configuration and momentum directions of the other mode (e+- = eigenvectors of M at the top)."""
    Fr, ev = mode_frame(eps); em = Fr[:2, 2] * np.sqrt(2); z = np.zeros(2)
    return np.stack([Fr[:, 0], Fr[:, 1], np.concatenate([em, z]), np.concatenate([z, em])], 1), ev


def unpack(x, sig, eps, scaled=False):
    """x = [m1 - q1*, m2 - q2* (units of sigma), 10 entries of a lower-triangular factor Lm in the frame (V+, W+, q-, p-), log diagonal].
    Axis scales sigma, sqrt(sigma), sqrt(sigma), 1 (the 1-d optimum is sigma-thin across V and ~2 sqrt(sigma) long; p- is free).
    S = Fr D Lm Lm^T D Fr^T.  With scaled=True also returns (Fr, D, Lm) for the well-conditioned evaluation."""
    m = np.concatenate([top(eps) + x[:2] * sig, [0.0, 0.0]])
    Fr, _ = opt_frame(eps); sc = np.array([sig, np.sqrt(sig), np.sqrt(sig), 1.0])
    Lm = np.zeros((4, 4)); idx = np.tril_indices(4); Lm[idx] = x[2:12]
    d = np.diag_indices(4); Lm[d] = np.exp(x[[2, 4, 7, 11]])
    S = Fr @ (sc[:, None] * (Lm @ Lm.T) * sc[None, :]) @ Fr.T
    return (m, S, (Fr, sc, Lm)) if scaled else (m, S)


def gap_scaled(x, sig, eps):
    """(kappa - R)/sigma by law 1, evaluated in the scaled frame so that it stays accurate for sigma down to 1e-7:
    F = Fr D^-1 (Lm Lm^T + sigma^2 D^-2)^-1 D^-1 Fr^T,   kappa - R = tr(F (kappa I - Sym Jbar)) / tr F."""
    m, S, (Fr, sc, Lm) = unpack(x, sig, eps, scaled=True)
    G = np.linalg.inv(Lm @ Lm.T + np.diag(sig ** 2 / sc ** 2)); F = Fr @ (G / sc[:, None] / sc[None, :]) @ Fr.T
    k = kappa(eps); M = (I2 - EVpp(m, S, eps)) / 2; SJ = np.block([[np.zeros((2, 2)), M], [M, np.zeros((2, 2))]])
    return float(np.trace(F @ (k * np.eye(4) - SJ)) / np.trace(F)) / sig


def obj(x, sig, eps):
    # bounded search box (log-diagonal in [-14, 8] in units of the axis scales, |off-diagonal| <= 50); singular cases rejected
    if np.any(x[[2, 4, 7, 11]] < -14) or np.any(x[[2, 4, 7, 11]] > 8) or np.any(np.abs(x[[3, 5, 6, 8, 9, 10]]) > 50) or np.any(np.abs(x[:2]) > 1e3): return 1e6
    try:
        v = gap_scaled(x, sig, eps)
    except np.linalg.LinAlgError:
        return 1e6
    return v if np.isfinite(v) else 1e6


def rstar(sig, eps, starts=6, seed=0):
    rng = np.random.default_rng(seed); best = None
    for k in range(starts):
        x0 = np.zeros(12); x0[[2, 4, 7, 11]] = np.array([-3.0, np.log(2.0), np.log(2.0), 3.0]) + rng.normal(0, 0.7 if k else 0.0, 4)
        x0[[3, 5, 6, 8, 9, 10]] = rng.normal(0, 0.3 if k else 0.0, 6)
        if k == 1 and eps > 0: x0 = ansatz_x(eps)
        f = lambda x: obj(x, sig, eps)
        r = minimize(f, x0, method='Powell', options={'xtol': 1e-10, 'ftol': 1e-14, 'maxfev': 20000})
        r = minimize(f, r.x, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-16, 'maxfev': 20000, 'maxiter': 20000})
        if best is None or r.fun < best.fun: best = r
    m, S = unpack(best.x, sig, eps)
    return kappa(eps) - best.fun * sig, m, S, best.x, best.fun


def shape(S, eps):
    """standard deviations of the optimal S along the frame (V+, W+, q-, p-) of opt_frame, the correlation of W+ and q-, and the
    principal axes (eigenvalues of S)."""
    Fr, ev = opt_frame(eps); P = Fr.T @ S @ Fr; sd = np.sqrt(np.maximum(np.diag(P), 0))
    return {'V+': float(sd[0]), 'W+': float(sd[1]), 'q-': float(sd[2]), 'p-': float(sd[3]), 'corr_W+_q-': float(P[1, 2] / (sd[1] * sd[2]))}, ev, Fr


def ansatz_x(eps):
    """start point: the leading-order optimum of gap_coefficient (V+ thin, (W+, q-) covariance sigma T*, p- wide)."""
    from scipy.linalg import sqrtm
    g = gap_coefficient(eps); B = np.array(g['B']); Cm = np.diag([2 * g['kappa'], g['kappa']])
    Bh = np.real(sqrtm(B)); Bhi = np.linalg.inv(Bh); T = Bhi @ np.real(sqrtm(Bh @ Cm @ Bh)) @ Bhi
    P = np.zeros((4, 4)); P[0, 0] = np.exp(-20); P[1:3, 1:3] = T; P[3, 3] = np.exp(6)
    L = np.linalg.cholesky(P); x = np.zeros(12); x[2:12] = L[np.tril_indices(4)]; x[[2, 4, 7, 11]] = np.log(np.diag(L))
    x[2] = -10.0
    return x


def gap_coefficient(eps, t=1e-6):
    """Leading-order prediction of (kappa - R*)/sigma as sigma -> 0 (stage L; a leading-order DERIVATION under the ansatz that the
    optimum is sigma-thin along V+, free along the momentum-only direction p-, and of size sqrt(sigma) along the two configuration
    directions W+ and q- (mode-(-) direction in q-space); not a theorem).  With S2 the covariance in the plane (W+, q-):
        (kappa - R)/sigma^2 ~ tr(B S2)/... :  lambda_max of the averaged Sym Jbar falls by tr(B S2) (B = stretch-decay matrix,
        computed here by finite differences of lambda_max M(E V'') in S), and every direction e of S2 costs c_e/(e.S2 e) in tr F,
        with c = 2 kappa (W+, contracting: Sym J = -kappa) and kappa (q-: Sym J = 0).
    min over S2 > 0 of tr(B S2)/sigma + sigma tr(C S2^{-1}) = 2 tr[(B^1/2 C B^1/2)^1/2]   (standard matrix identity; exact).
    Returns the matrix form, the diagonal form sum_e 2 sqrt(B_ee c_e) (ignores the cross term), and B."""
    from scipy.linalg import sqrtm
    Fr, ev = mode_frame(eps); k = kappa(eps); em = Fr[:2, 2] * np.sqrt(2)
    dirs = [Fr[:, 1], np.concatenate([em, [0, 0]])]
    m0 = np.concatenate([top(eps), [0, 0]])
    lp = lambda S: np.linalg.eigvalsh((I2 - EVpp(m0, S, eps)) / 2)[-1]
    B = np.zeros((2, 2))
    for i in range(2): B[i, i] = -(lp(t * np.outer(dirs[i], dirs[i])) - k) / t
    B[0, 1] = B[1, 0] = -(lp(t * (np.outer(dirs[0], dirs[1]) + np.outer(dirs[1], dirs[0])) / 2) - k) / t
    C = np.diag([2 * k, k]); Bh = np.real(sqrtm(B))
    return {'matrix': float(2 * np.trace(np.real(sqrtm(Bh @ C @ Bh)))), 'diagonal': float(sum(2 * np.sqrt(B[i, i] * C[i, i]) for i in range(2))),
            'B': B.tolist(), 'kappa': float(k)}


def job(a):
    s, e = a; R, m, S, x, gap = rstar(s, e); sh, ev, _ = shape(S, e)
    return {'sigma': s, 'eps': e, 'kappa': kappa(e), 'Rstar': R, 'gap_over_sigma': gap, 'gap_law1_plain': (kappa(e) - law1(m, S, s, e)) / s, 'm': m.tolist(), 'S': S.tolist(),
            'x': list(x), 'sd_in_mode_frame': sh, 'eig_S': np.linalg.eigvalsh(S).tolist(), 'M_top_eig': ev.tolist()}


if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'check':
        rng = np.random.default_rng(1); out = []
        for eps in [0.0, 0.1, 0.3]:
            for k in range(3):
                m = np.concatenate([rng.normal([np.pi, 0.0], 0.3), rng.normal(0, 0.3, 2)]); A = rng.normal(0, 0.3, (4, 4)); S = A @ A.T + 0.01 * np.eye(4)
                sig = [0.3, 0.1, 0.5][k]
                a = law1(m, S, sig, eps); b = R_quadrature(m, S, sig, eps, n=14); c = R_quadrature(m, S, sig, eps, n=18)
                mcs = [R_quadrature(m, S, sig, eps, mc=400_000, seed=100 * k + b_) for b_ in range(10)]   # 10 batches -> standard error
                d, se = float(np.mean(mcs)), float(np.std(mcs, ddof=1) / np.sqrt(len(mcs)))
                out.append({'eps': eps, 'sigma': sig, 'law1': a, 'gh14': b, 'gh18': c, 'mc': d, 'mc_se': se, 'mc_dev_in_se': (d - a) / se}); print(out[-1], flush=True)
        json.dump(out, open('results/l3_check.json', 'w'), indent=1)
    elif part == 'eps1':   # out-of-sample test of the matrix formula (written after eps = 0.1, 0.3 were seen): eps = 1
        with Pool(2) as pool:
            out = pool.map(job, [(1e-5, 1.0), (1e-4, 1.0), (1e-3, 1.0)])
        for o in out: o['matrix_formula'] = gap_coefficient(1.0)['matrix']; print(o['sigma'], o['gap_over_sigma'], o['matrix_formula'], flush=True)
        json.dump(out, open('results/l3_eps1.json', 'w'), indent=1, default=float)
    else:
        jobs = [(s, e) for e in [0.0, 0.1, 0.3] for s in [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]]
        with Pool(2) as pool:
            out = pool.map(job, jobs)
        json.dump(out, open('results/l3_single.json', 'w'), indent=1, default=float)
        for o in out: print(o['eps'], o['sigma'], round(o['gap_over_sigma'], 6), {k: round(v / (np.sqrt(o['sigma']) if k in ['W+', 'q-'] else o['sigma'] if k == 'V+' else 1), 4) for k, v in o['sd_in_mode_frame'].items()})
