"""Stage M, task M3: Gaussian MIXTURES for the coupled pendulums (eps = 0.1, 0.3), sigma = 1e-3, in the 3-d frame (V+, W+, q-)
with the momentum-only direction p- = (0, e-) infinitely long for every component.
Reduction (exact): if X_p- is independent of the other coordinates with an infinitely wide flat law, then the score has no p-
component, E[f . Q- | Y] = E[X_p- | Y] = Y_p- is independent of the other coordinates, so it drops out of E[s . E[f|Y]], and
R = E_Y[s . E[g|Y]] / (sigma^2 E|s|^2) on the 3-d coordinates y = (v, w, x3) along (V+, W+, Q-) with the effective field
    g_v = ((v - w)/sqrt2 - G)/sqrt2,   g_w = ((v - w)/sqrt2 + G)/sqrt2,   g_3 = 0,   G = grad V(q) . e+,
    q = q* + (v + w)/sqrt2 e+ + x3 e-   (coordinates relative to the top q*, e+- eigenvectors of M(q*)).
Posterior expectations in closed form per component (E sin(a.q) = sin(a.m) exp(-a^T C a/2)); 3-d midpoint grid aligned with
(V+, W+, Q-), spacing = (smallest posterior-free s.d. per axis)/per_sd, extent +-pad s.d.; the number of grid points is capped.
Check: for K = 1 this equals law 1 in 4-d with p- infinitely long (verified in run_all_M).
Parametrisation of a K-component state (scale-free units, as in stage H): weights (log), means (v in sigma, w and x3 in sqrt sigma),
lower-triangular factor of each covariance in the same units (log-diagonal).
Optimisation: start from the 1-d best state (stage H, h4_growk, K = 8 needles in the V/W plane; for K = 2, 4 its K heaviest
components) placed in the (V+, W+) plane, each component given the q- spread of the Gaussian optimum (ansatz T*); then adaptive
random hill climbing + Nelder-Mead on all parameters.  The same procedure is run for the 1-d pendulum (sigma = 1e-3) as the baseline.
Output: (kappa - R)/(kappa - R*_Gauss) for K = 2, 4 (8 if affordable) -> results/m3_mix.json (written after every job)."""
import numpy as np, json, os, sys, time
from multiprocessing import Pool
from scipy.optimize import minimize
from m_tools import coupled, find_kappa, frame, B_matrix, coefficient

SQ2 = np.sqrt(2)


class Setup:
    def __init__(self, eps, sig):
        self.s = coupled(eps); self.k, tops = find_kappa(self.s, starts=100); self.q0 = tops[0]; self.sig = sig
        ev, U = np.linalg.eigh(self.s.M(self.q0)); self.ep, self.em = U[:, -1], U[:, 0]
        self.alpha = np.stack([(self.s.A @ self.ep) / SQ2, (self.s.A @ self.ep) / SQ2, self.s.A @ self.em], 1)   # a_k . q as a function of y
        self.phase = self.s.A @ self.q0; self.ce = self.s.c * (self.s.A @ self.ep)                              # G = sum ce_k sin(phase_k + alpha_k . y)


def R3(st, ws, mus, Ss, per_sd=5, pad=7.0, max_points=1_500_000, chunk=60_000):
    sig = st.sig; s2 = sig * sig; T = [S + s2 * np.eye(3) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    lo = np.min([m - pad * np.sqrt(np.diag(t)) for m, t in zip(mus, T)], 0); hi = np.max([m + pad * np.sqrt(np.diag(t)) for m, t in zip(mus, T)], 0)
    step = np.array([min(1 / np.sqrt(ti[i, i]) for ti in Ti) / per_sd for i in range(3)])
    n = np.maximum(np.ceil((hi - lo) / step).astype(int), 21)
    while np.prod(n) > max_points: n = np.maximum((n * 0.93).astype(int), 21)
    h = (hi - lo) / n; ax = [lo[i] + (np.arange(n[i]) + 0.5) * h[i] for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    lg = [np.log(w) - 0.5 * np.log(np.linalg.det(t)) for w, t in zip(ws, T)]
    Kg = [S @ ti for S, ti in zip(Ss, Ti)]; Cp = [S - kk @ S for S, kk in zip(Ss, Kg)]
    damp = [np.exp(-0.5 * np.einsum('ki,ij,kj->k', st.alpha, C, st.alpha)) for C in Cp]
    num = den = mass = 0.0
    for c0 in range(0, len(G), chunk):
        Y = G[c0:c0 + chunk]
        logc = np.stack([l - 0.5 * np.einsum('ni,ij,nj->n', Y - m, ti, Y - m) for l, m, ti in zip(lg, mus, Ti)], 1)
        mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True)
        sc = np.zeros_like(Y); Eg = np.zeros_like(Y)
        for k in range(len(ws)):
            d = Y - mus[k]; sc += pi[:, k:k + 1] * (-d @ Ti[k]); mp = mus[k] + d @ Kg[k].T
            Gk = (np.sin(st.phase[None, :] + mp @ st.alpha.T) * damp[k][None, :]) @ st.ce
            lin = (mp[:, 0] - mp[:, 1]) / SQ2
            Eg[:, 0] += pi[:, k] * (lin - Gk) / SQ2; Eg[:, 1] += pi[:, k] * (lin + Gk) / SQ2
        num += np.sum(p * (sc * Eg).sum(1)); den += np.sum(p * (sc ** 2).sum(1)); mass += np.sum(p)
    dV = np.prod(h) / (2 * np.pi) ** 1.5
    return num / (s2 * den), mass * dV, tuple(n)


def unpack(x, K, sig):
    sc = np.array([sig, np.sqrt(sig), np.sqrt(sig)]); w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum()); mus, Ss = [], []
    for k in range(K):
        mus.append(x[K + 3 * k: K + 3 * k + 3] * sc)
        l = x[4 * K + 6 * k: 4 * K + 6 * k + 6]; L = np.zeros((3, 3)); L[np.tril_indices(3)] = l; L[np.diag_indices(3)] = np.exp(l[[0, 2, 5]])
        L = sc[:, None] * L; Ss.append(L @ L.T)
    return ws, mus, Ss


def pack(ws, mus, Ss, sig):
    sc = np.array([sig, np.sqrt(sig), np.sqrt(sig)]); K = len(ws); x = np.zeros(10 * K); x[:K] = np.log(ws)
    for k in range(K):
        x[K + 3 * k: K + 3 * k + 3] = mus[k] / sc
        L = np.linalg.cholesky(Ss[k] / np.outer(sc, sc) + 1e-14 * np.eye(3)); l = L[np.tril_indices(3)].copy(); l[[0, 2, 5]] = np.log(np.diag(L)); x[4 * K + 6 * k: 4 * K + 6 * k + 6] = l
    return x


def start_state(st, K):
    """the 1-d best K = 8 state of stage H (h4_growk) in the (V+, W+) plane, K heaviest components, rescaled to this flow:
    the 1-d units (V in sigma, W in 2 sqrt(sigma)) are kept; each component gets the q- variance of the Gaussian optimum."""
    from h2_adversarial import unpack as u1
    c = json.load(open('results/h4_growk.json')); best = max([x for ch in c for x in ch['chain'] if x['K'] == 8], key=lambda x: x['gstar10'])
    ws, mus, Ss = u1(np.array(best['x']), 8, st.sig); o = np.argsort(ws)[::-1][:K]
    from h2_adversarial import V as V1, W as W1, TOP
    R2 = np.stack([V1, W1], 1)
    B, _ = B_matrix(st.s, st.q0); var3 = st.sig * np.sqrt(st.k / B[1, 1])            # 1-d optimum of B22 s + sigma^2 kappa / s
    W_, M_, S_ = [], [], []
    for i in o:
        m2 = R2.T @ (mus[i] - TOP); S2 = R2.T @ Ss[i] @ R2
        S3 = np.zeros((3, 3)); S3[:2, :2] = S2; S3[2, 2] = var3; W_.append(ws[i]); M_.append(np.array([m2[0], m2[1], 0.0])); S_.append(S3)
    W_ = list(np.array(W_) / np.sum(W_)); return W_, M_, S_


def gauss_best(st):
    """best single Gaussian of the 3-d problem (from the 4-d optimum structure: thin V+, (W+, q-) covariance sigma T*)."""
    from scipy.linalg import sqrtm
    B, _ = B_matrix(st.s, st.q0); C = np.diag([2 * st.k, st.k]); Bh = np.real(sqrtm(B)); Bhi = np.linalg.inv(Bh); T = Bhi @ np.real(sqrtm(Bh @ C @ Bh)) @ Bhi
    S = np.zeros((3, 3)); S[1:, 1:] = st.sig * T; S[0, 0] = (1e-3 * st.sig) ** 2
    x = pack([1.0], [np.zeros(3)], [S], st.sig)
    f = lambda z: gapf(st, z, 1)
    r = minimize(f, x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
    return float(r.fun), r.x


def gapf(st, x, K, per_sd=5):
    if np.any(np.abs(x[K:4 * K]) > 30) or np.any(np.abs(x[:K]) > 20): return 1e6
    ws, mus, Ss = unpack(x, K, st.sig)
    try: R, mass, n = R3(st, ws, mus, Ss, per_sd=per_sd)
    except (np.linalg.LinAlgError, ValueError, MemoryError): return 1e6
    return (st.k - R) / st.sig if abs(mass - 1) < 1e-5 and np.isfinite(R) else 1e6


def climb(st, x, K, steps, seed):
    rng = np.random.default_rng(seed); fx = gapf(st, x, K); step = 0.1; acc = 0
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = gapf(st, y, K)
        if fy < fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.5)); acc = 0
    r = minimize(lambda z: gapf(st, z, K), x, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
    return (r.x, float(r.fun)) if r.fun < fx else (x, fx)


def rstar_gauss(eps, sig=1e-3):
    """best single Gaussian of the 4-d problem at sigma (= the 3-d problem with p- infinitely long): the smallest value over the
    assumption-free optimisations of m1_free at this sigma (the gap is the same in the 3-d reduction)."""
    fn = f'results/m1_free_eps{eps:g}.json'
    return min(r['gap'][f'{sig:g}'] for r in json.load(open(fn))['runs'])


def job(a):
    kind, eps, K, steps = a; sig = 1e-3
    if kind == '1d':
        from l1_free import gap as g1
        from h2_adversarial import unpack as u1
        from h1_rstar import rstar
        c = json.load(open('results/h4_growk.json')); best = max([x for ch in c for x in ch['chain'] if x['K'] == 8], key=lambda x: x['gstar10'])
        x8 = np.array(best['x']); ws, _, _ = u1(x8, 8, sig); o = np.argsort(ws)[::-1][:K]
        x = np.concatenate([x8[:8][o], x8[8:24].reshape(8, 2)[o].ravel(), x8[24:].reshape(8, 3)[o].ravel()])
        rng = np.random.default_rng(5 + K); fx = g1(x, K, sig); step = 0.1; acc = 0; f0 = fx
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = g1(y, K, sig)
            if fy < fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.5)); acc = 0
        r = minimize(lambda z: g1(z, K, sig), x, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
        if r.fun < fx: x, fx = r.x, float(r.fun)
        gs = (1 - rstar(sig)['Rstar']) / sig
        return {'kind': '1d', 'eps': None, 'K': K, 'gap_start': f0, 'gap': fx, 'gap_gauss': gs, 'ratio': fx / gs, 'x': x.tolist()}
    st = Setup(eps, sig); ws, mus, Ss = start_state(st, K); x = pack(ws, mus, Ss, sig); f0 = gapf(st, x, K, per_sd=3)
    g3 = lambda z: gapf(st, z, K, per_sd=3)
    rng = np.random.default_rng(11 + K); fx = f0; step = 0.1; acc = 0
    for it in range(1, steps + 1):
        y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.3); fy = g3(y)
        if fy < fx: x, fx = y, fy; acc += 1
        if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.5)); acc = 0
    r = minimize(g3, x, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
    if r.fun < fx: x, fx = r.x, float(r.fun)
    fine = gapf(st, x, K, per_sd=5); gs = rstar_gauss(eps, sig)
    return {'kind': '3d', 'eps': eps, 'K': K, 'gap_start': f0, 'gap_search': fx, 'gap': fine, 'gap_per_sd7': gapf(st, x, K, per_sd=7),
            'gap_gauss': gs, 'ratio': fine / gs, 'x': x.tolist()}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500; Ks = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else '2,4').split(',')]
    fn = 'results/m3_mix.json'; out = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(o['kind'], o['eps'], o['K']) for o in out}
    jobs = [(kd, e, K, steps) for K in Ks for kd, e in [('1d', None), ('3d', 0.1), ('3d', 0.3)] if (kd, e, K) not in done]
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r); json.dump(out, open(fn, 'w'), indent=1)
            print(f"{r['kind']} eps {r['eps']} K={r['K']}: start {r['gap_start']:.5f} -> {r['gap']:.5f}  Gaussian {r['gap_gauss']:.5f}  ratio {r['ratio']:.5f}", flush=True)
