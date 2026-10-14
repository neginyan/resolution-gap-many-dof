"""Stage N: fast exact evaluator for PRODUCT mixtures in the reduced frame of the coupled pendulums (and the 1-d pendulum).
Reduced 3-d coordinates (stage M, m3_mix): y = (v, w, x3) along (V+, W+, Q-), p- infinitely long, effective field
    g_v = (L - G)/sqrt2,  g_w = (L + G)/sqrt2,  g_3 = 0,   L = (v - w)/sqrt2,   G = sum_t ce_t sin(ph_t + alpha_t . y).
Rotate the (w, x3) plane by theta:  a = c w + s x3,  b = -s w + c x3  (c = cos theta, s = sin theta; the noise stays isotropic).
PRODUCT states:  X = (X_v, X_a) ~ rho (any 2-d Gaussian mixture)  independent of  X_b ~ N(0, s_b).
Exact reduction (derived in stage N; Y_b-integrals done in closed form):
    R = [num2 + extra] / (sigma^2 (J2 + 1/T_b)),   T_b = s_b + sigma^2,
    num2  = E_Y2[ s . E[g2 | Y2] ]  for the 2-d field  g2 = ((L' - G')/sqrt2, c (L' + G')/sqrt2),  L' = (v - c a)/sqrt2,
            G' = sum_t ce_t exp(-gamma_t^2 s_b/2) sin(ph_t + beta_t . (v, a)),
            beta_t = (alpha_tv, alpha_tw c + alpha_t3 s),  gamma_t = -alpha_tw s + alpha_t3 c;
    J2    = E|s2|^2 (2-d Fisher information of the smoothed rho);
    extra = s^2 s_b/(2 T_b) + (s/sqrt2) sum_t ce_t (gamma_t s_b/T_b) exp(-gamma_t^2 s_b/2) Re[exp(i ph_t) phi(beta_t)],
            phi = characteristic function of rho.
For theta = 0 the extra term vanishes; the 1-d pendulum is the case without b (no 1/T_b, no extra).
2-d midpoint grid aligned with (v, a); posterior expectations per component in closed form; vectorised over components.
Checked against the 3-d evaluator m3_mix.R3 (product states, any theta) and, for K = 1, against 4-d law 1 (run_all_N)."""
import numpy as np
from m_tools import System, coupled, find_kappa, B_matrix

SQ2 = np.sqrt(2)


class Flow:
    def __init__(self, eps=None, sig=1e-3):
        """eps = None: the 1-d pendulum (kappa = 1, top pi); otherwise the coupled pendulums at their top."""
        self.sig = sig; self.eps = eps
        if eps is None:
            self.s = System(1, [(1.0, [1.0])]); self.k = 1.0; self.q0 = np.array([np.pi]); ep = np.array([1.0]); em = np.zeros(1)
        else:
            self.s = coupled(eps); self.k, tops = find_kappa(self.s, starts=100); self.q0 = tops[0]
            ev, U = np.linalg.eigh(self.s.M(self.q0)); ep, em = U[:, -1], U[:, 0]
        A = self.s.A
        self.alpha = np.stack([(A @ ep) / SQ2, (A @ ep) / SQ2, A @ em], 1); self.ph = A @ self.q0; self.ce = self.s.c * (A @ ep)
        self.oned = eps is None


def R_product(fl, ws, mus, Ss, s_b=None, theta=0.0, per_sd=5, pad=7.0, max_points=400_000, parts=False):
    """ws, mus (2-vectors in (v, a)), Ss (2 x 2) of the 2-d mixture; s_b, theta: the common b-law (ignored for the 1-d pendulum)."""
    sig = fl.sig; s2 = sig * sig; K = len(ws); c, s = np.cos(theta), np.sin(theta)
    al = fl.alpha; beta = np.stack([al[:, 0], al[:, 1] * c + al[:, 2] * s], 1); gam = -al[:, 1] * s + al[:, 2] * c
    if fl.oned: damp = np.ones(len(fl.ce)); Tb = None; c, s = 1.0, 0.0; beta = al[:, :2]
    else: damp = np.exp(-gam ** 2 * s_b / 2); Tb = s_b + s2
    ce = fl.ce * damp
    mus = np.array(mus); Ss = np.array(Ss); ws = np.array(ws)
    T = Ss + s2 * np.eye(2); Ti = np.linalg.inv(T); Kg = Ss @ Ti; Cp = Ss - Kg @ Ss
    sd = np.sqrt(np.einsum('kii->ki', T)); lo = np.min(mus - pad * sd, 0); hi = np.max(mus + pad * sd, 0)
    step = np.min(1 / np.sqrt(np.einsum('kii->ki', Ti)), 0) / per_sd
    n = np.maximum(np.ceil((hi - lo) / step).astype(int), 41)
    while n[0] * n[1] > max_points: n = np.maximum((n * 0.95).astype(int), 41)
    h = (hi - lo) / n; ax = [lo[i] + (np.arange(n[i]) + 0.5) * h[i] for i in range(2)]
    Y = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 2)
    D = Y[:, None, :] - mus[None]                                          # (N, K, 2)
    Q = np.einsum('nki,kij,nkj->nk', D, Ti, D)
    lg = np.log(ws) - 0.5 * np.log(np.linalg.det(T)) - np.log(2 * np.pi)
    logc = lg[None] - 0.5 * Q; mx = logc.max(1); e = np.exp(logc - mx[:, None]); p = e.sum(1) * np.exp(mx); pi = e / e.sum(1, keepdims=True)
    uk = -np.einsum('kij,nkj->nki', Ti, D); sc = np.einsum('nk,nki->ni', pi, uk)
    mp = mus[None] + np.einsum('kij,nkj->nki', Kg, D)                       # posterior means (N, K, 2)
    dmp = np.exp(-0.5 * np.einsum('ti,kij,tj->kt', beta, Cp, beta))         # (K, terms)
    Gk = np.einsum('nkt,kt,t->nk', np.sin(fl.ph[None, None, :] + mp @ beta.T), dmp, ce)
    Lk = (mp[:, :, 0] - c * mp[:, :, 1]) / SQ2
    Eg = np.stack([np.sum(pi * (Lk - Gk), 1) / SQ2, c * np.sum(pi * (Lk + Gk), 1) / SQ2], 1)
    dA = h[0] * h[1]; num2 = np.sum(p * (sc * Eg).sum(1)) * dA; J2 = np.sum(p * (sc ** 2).sum(1)) * dA; mass = np.sum(p) * dA
    if fl.oned:
        extra = 0.0; den = s2 * J2
    else:
        phi = np.sum(ws[:, None] * np.exp(1j * mus @ beta.T - 0.5 * np.einsum('ti,kij,tj->kt', beta, Ss, beta)), 0)   # (terms,)
        extra = s * s * s_b / (2 * Tb) + (s / SQ2) * np.sum(fl.ce * (gam * s_b / Tb) * damp * np.real(np.exp(1j * fl.ph) * phi))
        den = s2 * (J2 + 1 / Tb)
    R = (num2 + extra) / den
    if parts: return R, {'num2': num2, 'extra': extra, 'J2': J2, 'Jb': (0.0 if fl.oned else 1 / Tb), 'mass': mass, 'n': tuple(n)}
    return R, mass


# ---------- parametrisation (scale-free units: v in sigma, a in sqrt sigma; s_b in sigma; theta free) ----------
def unpack(x, K, fl):
    sig = fl.sig; sc = np.array([sig, np.sqrt(sig)]); w = np.exp(x[:K] - x[:K].max()); ws = w / w.sum()
    mus = x[K:3 * K].reshape(K, 2) * sc
    l = x[3 * K:6 * K].reshape(K, 3); L = np.zeros((K, 2, 2)); L[:, 0, 0] = np.exp(l[:, 0]); L[:, 1, 0] = l[:, 1]; L[:, 1, 1] = np.exp(l[:, 2])
    L = sc[None, :, None] * L; Ss = L @ np.transpose(L, (0, 2, 1))
    if fl.oned: return ws, mus, Ss, None, 0.0
    return ws, mus, Ss, sig * np.exp(x[6 * K]), x[6 * K + 1]


def pack(ws, mus, Ss, s_b, theta, fl):
    sig = fl.sig; sc = np.array([sig, np.sqrt(sig)]); K = len(ws); x = [np.log(ws), (np.array(mus) / sc).ravel()]
    ll = []
    for S in Ss:
        L = np.linalg.cholesky(np.array(S) / np.outer(sc, sc) + 1e-18 * np.eye(2)); ll.append([np.log(L[0, 0]), L[1, 0], np.log(L[1, 1])])
    x.append(np.ravel(ll))
    if not fl.oned: x.append([np.log(s_b / sig), theta])
    return np.concatenate(x)


def gap(x, K, fl, per_sd=4):
    if np.any(np.abs(x[K:3 * K]) > 40) or np.any(np.abs(x[:K]) > 40) or np.any(x[3 * K:6 * K].reshape(K, 3)[:, [0, 2]] > 4) or np.any(x[3 * K:6 * K].reshape(K, 3)[:, [0, 2]] < -12): return 1e6
    if not fl.oned and not (-12 < x[6 * K] < 12): return 1e6
    try:
        R, mass = R_product(fl, *unpack(x, K, fl), per_sd=per_sd)
    except (np.linalg.LinAlgError, ValueError, MemoryError, FloatingPointError):
        return 1e6
    return (fl.k - R) / fl.sig if abs(mass - 1) < 1e-6 and np.isfinite(R) else 1e6


def gauss_start(fl):
    """the leading-order Gaussian optimum written as a product state: theta = principal angle of T*, s_b = the q-like eigenvalue."""
    from scipy.linalg import sqrtm
    sig = fl.sig
    if fl.oned:
        S = np.array([[1e-6 * sig ** 2, 0], [0, 4 * sig]]); return [1.0], [np.zeros(2)], [S], None, 0.0
    B, _ = B_matrix(fl.s, fl.q0); C = np.diag([2 * fl.k, fl.k]); Bh = np.real(sqrtm(B)); Bhi = np.linalg.inv(Bh); T = sig * Bhi @ np.real(sqrtm(Bh @ C @ Bh)) @ Bhi
    ev, U = np.linalg.eigh(T); u = U[:, np.argmax(np.abs(U[0]))] * np.sign(U[np.argmax(np.abs(U[0])), np.argmax(np.abs(U[0]))])  # axis closest to W+
    th = np.arctan2(u[1], u[0]); a_var = u @ T @ u; b_var = np.trace(T) - a_var
    S = np.array([[1e-6 * sig ** 2, 0], [0, a_var]]); return [1.0], [np.zeros(2)], [S], b_var, th
