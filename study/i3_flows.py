"""Stage I, task 3: the same questions for other flows -- double well, quartic oscillator (near q = 0) and the twist flow
H = ln(1 + q^2 + p^2)/2 (see i_tools).  Parts:

  single : best single Gaussian R*(sigma) at the top.
      Flows f = (p, g(q)) with a = (1 + g')/2: law 1 gives R = lambda * phi exactly as for the pendulum (Sym E[Df] = lambda diag(1,-1)
      in V/W), so the optimum is again a zero-width segment through the top, tilted towards -V, with the same inner optimum p*.
      For a polynomial top a = kappa - b q^2 (double well kappa = 1, b = 3/2; quartic kappa = 1/2, b = 3/2; pendulum ~ b = 1/4)
      lambda = kappa - b (m_q^2 + S_qq) is LINEAR in S_qq, and with eps = sigma sqrt(b/kappa), u = sqrt(kappa/b) v:
          R*/kappa = max_v (1 - eps v/2) phi*(eps/v)     -- one universal function of eps for every polynomial top;
          1 - R*/kappa = Psi(eps) = 2 eps - ... (exact rational series below).
      Twist flow: law 1 with E_rho[Df] by Gauss-Hermite quadrature; maximised over centre radius, two widths and the tilt.
  check  : law 1 / R* against the grid evaluator (i_tools.fields) for the optimal segment of each flow.
  search : adversarial search for mixtures near the top beating R*(sigma) (K = 2, 3; sigma = 1e-3, 1e-2) and the gap (kappa - R)/sigma.
Writes results/i3_<part>.json."""
import numpy as np, json, sys
from fractions import Fraction as Fr
from multiprocessing import Pool
from scipy.optimize import minimize, minimize_scalar
from i_tools import fields, stretch, KAPPA, TOPS, V, W, RVW
from h1_rstar import S2, one_minus_R

POLY = {'double_well': (1.0, 1.5), 'quartic': (0.5, 1.5)}          # kappa, b  (a = kappa - b q^2)


# ---------- single Gaussians ----------
def phi_star(eps_over_v):
    x = eps_over_v; p = 2 / (1 + np.sqrt(1 + 4 * x)); M = (1 + 2 * x * p) / (2 - p); return 1 - 2 * x / (M + 2 * x), p


def rstar_poly(kind, sig):
    kap, b = POLY[kind]; eps = sig * np.sqrt(b / kap)
    f = lambda L: -(1 - eps * np.exp(L) / 2) * phi_star(eps / np.exp(L))[0]
    Ls = np.linspace(np.log(1e-3), np.log(2 / eps), 3001); i = int(np.argmin([f(L) for L in Ls])); i = min(max(i, 1), len(Ls) - 2)
    r = minimize_scalar(f, bracket=(Ls[i - 1], Ls[i], Ls[i + 1]), tol=1e-14); v = np.exp(r.x)
    return {'Rstar': -r.fun * kap, 'v': v, 'p': phi_star(eps / v)[1], 'eps': eps, 'u': v * np.sqrt(kap / b)}


def segment(sig, u, p, top):
    """rank-one optimum: S_ww - type segment; d^2 = sigma u, alpha = p sigma^2/d (same construction as h1_rstar.gaussian)."""
    d = np.sqrt(u * sig); alpha = p * sig ** 2 / d; beta = alpha + d; vec = -alpha * V + beta * W; return np.outer(vec, vec)


def series_poly(N=10):
    """exact rational series of Psi(eps) = 1 - R*/kappa for a polynomial top, with v = 2 + w."""
    import h1_rstar as h; h.N = N
    s = S2({(1, 0): 1}); w = S2({(0, 1): 1}); v = 2 + w; iv = v.inv()
    x = s * iv; r = (4 * x).sqrt1(); p = 2 / (1 + r); M = (1 + 2 * x * p) / (2 - p); one_m_phi = 2 * x / (M + 2 * x)
    lam = 1 - s * v * Fr(1, 2); F = 1 - lam * (1 - one_m_phi)
    Fw = F.dw(); wk = [Fr(0)] * (N + 1)
    def compose(G, wk, order):
        out = [Fr(0)] * (order + 1); powers = [[Fr(1)] + [Fr(0)] * order]
        for j in range(1, N + 1):
            prev = powers[-1]; powers.append([sum(prev[a] * wk[b - a] for a in range(b + 1)) for b in range(order + 1)])
        for (i, j), c in G.c.items():
            for b in range(order + 1 - i): out[i + b] += c * powers[j][b]
        return out
    for k in range(1, N - 1):
        base = compose(Fw, wk, k + 1)[k + 1]; t = list(wk); t[k] = Fr(1); slope = compose(Fw, t, k + 1)[k + 1] - base; wk[k] = -base / slope
    return [str(c) for c in compose(F, wk, N)], [str(c) for c in wk[:N - 1]]


def law1_twist(x, sig, gh=24):
    """law 1 for the twist flow: centre (r0, 0), widths along the local V/W, tilt th.  x = [r0, log s_v, log s_w, th]."""
    r0, lsv, lsw, th = x; c, s = np.cos(th), np.sin(th); U = RVW @ np.array([[c, -s], [s, c]])
    S = U @ np.diag([np.exp(2 * np.clip(lsv, -25, 3)), np.exp(2 * np.clip(lsw, -25, 3))]) @ U.T
    g, w = np.polynomial.hermite_e.hermegauss(gh); w = w / w.sum()
    ev, Q = np.linalg.eigh(S); L = Q * np.sqrt(np.maximum(ev, 0))
    Z = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2); Wt = np.outer(w, w).ravel()
    pts = np.array([r0, 0.0]) + Z @ L.T
    Msym = np.einsum('k,kij->ij', Wt, stretch('twist', pts)[2])
    F = np.linalg.inv(S + sig ** 2 * np.eye(2)); return float(np.trace(F @ Msym) / np.trace(F)), S


def rstar_twist(sig):
    best = None
    for x0 in [[1.0, np.log(sig / 3), np.log(2 * np.sqrt(sig)), 0.0], [1.0, np.log(sig / 10), np.log(3 * np.sqrt(sig)), 0.05],
               [1.0, np.log(sig), np.log(1.5 * np.sqrt(sig)), -0.05], [1.02, np.log(sig / 30), np.log(2.5 * np.sqrt(sig)), 0.01]]:
        r = minimize(lambda x: -law1_twist(x, sig)[0], x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-15, 'maxiter': 6000, 'maxfev': 6000})
        if best is None or r.fun < best.fun: best = r
    R, S = law1_twist(best.x, sig)
    ev, U = np.linalg.eigh(RVW.T @ S @ RVW); u = U[:, 1] * np.sign(U[1, 1])
    return {'Rstar': R, 'x': list(best.x), 'S': S.tolist(), 'centre_r': best.x[0], 'tilt_from_W': float(np.arctan2(u[0], u[1])),
            'length': float(np.sqrt(ev[1])), 'thickness': float(np.sqrt(max(ev[0], 0)))}


from functools import lru_cache


@lru_cache(maxsize=256)
def rstar_flow(kind, sig):
    if kind == 'pendulum':
        from h1_rstar import rstar; return rstar(sig)['Rstar']
    if kind in POLY: return rstar_poly(kind, sig)['Rstar']
    return rstar_twist(sig)['Rstar']


# ---------- mixture search near the top ----------
def scales(kind, sig):
    """width / length scales of the optimal segment (for a scale-free parametrisation of the search)."""
    if kind in POLY: t = rstar_poly(kind, sig); return sig, np.sqrt(t['u'] * sig)
    if kind == 'twist': t = rstar_twist(sig); return sig, t['length']
    from h1_rstar import rstar; t = rstar(sig); return sig, np.sqrt(t['u'] * sig)


def unpack(x, K, kind, sv0, sw0):
    w = np.exp(x[:K] - x[:K].max()); ws = list(w / w.sum()); mus, Ss = [], []
    for k in range(K):
        mus.append(TOPS[kind] + x[K + 2 * k] * sv0 * V + x[K + 2 * k + 1] * sw0 * W)
        a, b, c = x[3 * K + 3 * k: 3 * K + 3 * k + 3]
        L = np.array([[sv0 * np.exp(a), 0], [b * sw0, sw0 * np.exp(c)]]); Ss.append(RVW @ (L @ L.T) @ RVW.T)
    return ws, mus, Ss


def job_search(args):
    kind, K, sig, seed, steps, gh = args
    rng = np.random.default_rng(seed); Rs = rstar_flow(kind, sig); kap = KAPPA[kind]; sv0, sw0 = scales(kind, sig)
    def obj(x):
        sh = x[3 * K:].reshape(K, 3)
        if np.any(np.abs(x[K:3 * K]) > 20) or np.any(sh[:, [0, 2]] > 2) or np.any(sh[:, [0, 2]] < -10) or np.any(np.abs(sh[:, 1]) > 6): return -np.inf
        ws, mus, Ss = unpack(x, K, kind, sv0, sw0)
        if min(ws) < 0.02: return -np.inf
        try: r = fields(ws, mus, Ss, sig, kind=kind, per_sd=6, gh=gh)
        except (MemoryError, ValueError, np.linalg.LinAlgError): return -np.inf
        if abs(r['mass'] - 1) > 1e-6: return -np.inf
        return (r['R'] - Rs) / (kap - Rs)
    runs = []
    for rs in range(4):
        shp = [[rng.normal(-1.5, 1), rng.normal(0, 0.6) * (-1) ** k, rng.normal(-0.2, 0.4)] for k in range(K)]
        x = np.concatenate([rng.normal(0, 0.3, K), rng.normal(0, 0.3 if rs < 2 else 1.0, 2 * K), np.ravel(shp)]); fx = obj(x); t = 0
        while not np.isfinite(fx) and t < 20: x = x + rng.normal(0, 0.1, x.size); fx = obj(x); t += 1
        step = 0.2; acc = 0
        for it in range(1, steps + 1):
            y = x + rng.normal(0, step, x.size) * (rng.random(x.size) < 0.5); fy = obj(y)
            if fy > fx: x, fx = y, fy; acc += 1
            if it % 25 == 0: step = float(np.clip(step * (1.4 if acc > 4 else 0.6), 1e-3, 0.8)); acc = 0
        runs.append({'gstar': fx, 'x': x.tolist()})
    b = max(runs, key=lambda r: r['gstar']); ws, mus, Ss = unpack(np.array(b['x']), K, kind, sv0, sw0)
    hi = None if gh is None else 2 * gh
    r1 = fields(ws, mus, Ss, sig, kind=kind, per_sd=10, gh=gh); r2 = fields(ws, mus, Ss, sig, kind=kind, per_sd=14, gh=hi)
    return {'kind': kind, 'K': K, 'sigma': sig, 'Rstar': Rs, 'kappa': kap, 'R': r1['R'], 'R_check': r2['R'], 'gstar': (r1['R'] - Rs) / (kap - Rs),
            'gap_over_sigma': (kap - r1['R']) / sig, 'Rstar_gap_over_sigma': (kap - Rs) / sig, 'runs': [u['gstar'] for u in runs], 'x': b['x'],
            'ws': ws, 'mus': [m.tolist() for m in mus], 'Ss': [S.tolist() for S in Ss], 'mass': r1['mass']}


if __name__ == '__main__':
    part = sys.argv[1]
    SIGS = [1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    if part == 'single':
        coef, wser = series_poly(10)
        out = {'Psi_coef': coef, 'v_series': wser, 'flows': {}}
        print('Psi(eps) =', ' + '.join(f'({c}) e^{k}' for k, c in enumerate(coef) if c != '0'))
        for kind in ['double_well', 'quartic']:
            kap, b = POLY[kind]; rows = []
            for s in SIGS:
                t = rstar_poly(kind, s); ser = sum(float(Fr(c)) * t['eps'] ** k for k, c in enumerate(coef))
                rows.append({'sigma': s, **t, 'gap_over_sigma': (kap - t['Rstar']) / s, 'rel_gap_over_sigma': (1 - t['Rstar'] / kap) / s, 'series_err': abs(ser - (1 - t['Rstar'] / kap))})
            out['flows'][kind] = {'kappa': kap, 'b': b, 'predicted_gap_coefficient': 2 * np.sqrt(b * kap), 'rows': rows}
            print(kind, [(r['sigma'], round(r['gap_over_sigma'], 5), f"{r['series_err']:.1e}") for r in rows])
        rows = []
        for s in SIGS:
            t = rstar_twist(s); rows.append({'sigma': s, **t, 'gap_over_sigma': (0.25 - t['Rstar']) / s, 'rel_gap_over_sigma': (1 - t['Rstar'] / 0.25) / s})
            print('twist', s, round(t['Rstar'], 10), round(rows[-1]['gap_over_sigma'], 5), 'centre r', round(t['centre_r'], 6), 'tilt/sigma', round(t['tilt_from_W'] / s, 4),
                  'length/sqrt(sigma)', round(t['length'] / np.sqrt(s), 4), 'thickness/sigma', round(t['thickness'] / s, 6), flush=True)
        out['flows']['twist'] = {'kappa': 0.25, 'rows': rows}
        json.dump(out, open('results/i3_single.json', 'w'), indent=1, default=float)
    elif part == 'check':
        out = []
        for kind in ['double_well', 'quartic', 'twist']:
            for s in [1e-3, 1e-2, 0.1]:
                if kind in POLY: t = rstar_poly(kind, s); S = segment(s, t['u'], t['p'], TOPS[kind]); Rs = t['Rstar']
                else: t = rstar_twist(s); S = np.array(t['S']); Rs = t['Rstar']
                mu = TOPS[kind] if kind in POLY else np.array([t['centre_r'], 0.0])
                r = fields([1.0], [mu], [S + 1e-30 * np.eye(2)], s, kind=kind, per_sd=10, gh=(24 if kind == 'twist' else None))
                r2 = fields([1.0], [mu], [S + 1e-30 * np.eye(2)], s, kind=kind, per_sd=10, gh=24)
                out.append({'kind': kind, 'sigma': s, 'Rstar': Rs, 'R_grid': r['R'], 'R_grid_gh': r2['R'], 'diff': r['R'] - Rs, 'diff_gh': r2['R'] - Rs})
                print(out[-1], flush=True)
        json.dump(out, open('results/i3_check.json', 'w'), indent=1, default=float)
    else:
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        jobs = [(kind, K, s, 100 * K + int(-np.log10(s)) + 7 * i, steps, (10 if kind == 'twist' else None))
                for i, kind in enumerate(['double_well', 'quartic', 'twist']) for K in [2, 3] for s in [1e-3, 1e-2]]
        out = []
        with Pool(2) as pool:
            for r in pool.imap_unordered(job_search, jobs):
                out.append(r); json.dump(out, open('results/i3_search.json', 'w'), indent=1, default=float)
                print(f"{r['kind']:<11} K={r['K']} sigma={r['sigma']:g}: g* {r['gstar']:+.4f}  (kappa-R)/sigma {r['gap_over_sigma']:.4f}  (kappa-R*)/sigma {r['Rstar_gap_over_sigma']:.4f}  check {r['R_check'] - r['R']:.1e}", flush=True)
