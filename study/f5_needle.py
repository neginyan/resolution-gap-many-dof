"""Stage F: why the fraction approaches 1.  The best states consist of one needle at the hyperbolic point (narrow along V, long
along W) that carries ~99.9 % of the Fisher information but only ~30 % of the mass, plus broad components far from the top.
R is then close to the needle's own R (single Gaussian: R_k <= lambda_k < 1), while the room 1 - law is set by the mass-weighted
lambda_rho.  Test: shrink the needle and the window together (sigma -> t sigma, needle width along V -> t, length along W -> sqrt(t),
the scaling that keeps a single Gaussian at its best distance) with the broad components unchanged, and follow R, the needle's
own R, and the fraction.  A single uniform grid cannot resolve a needle 100x narrower than the other components (the first
attempt gave a 1.1 error at t = 0.5), so a composite grid is used: a fine V/W box around the needle (+-12 s.d., 12 points per s.d.)
plus a coarse V/W grid of the whole state with the box cut out; checked against the uniform grid at t = 1 and by refining both parts.
Writes results/f5_needle.json."""
import numpy as np, json
from obs_eval2 import evaluate_full, V, W, vw_grid
from f1_chain import factors

Rm = np.stack([V, W], 1)


def composite(ws, mus, Ss, sig, k, fine=12, coarse=10, box=12.0, taper=3.0, pad=8.0, refine=8.0):
    """Nested smooth partition of unity in V/W coordinates around the needle k.  Level 0 is a coarse midpoint grid of the whole
    state; level l (l = 1..n) a midpoint grid with steps h_l = max(h_(l-1)/refine, needle step) on a box around the needle.
    chi_l(v, w) = b_l(v) b_l(w), b_l = erfc taper of length taper*h_(l-1) around a plateau of half-width
    max(box * needle s.d., 5 * taper length).  Weights: level l: chi_l - chi_(l+1) (chi_0 = 1, chi_(n+1) = 0), which sum to 1 exactly.
    Every weighted integrand is smooth and decays, so every midpoint sum is spectrally accurate.  (A hard-edged box gave an
    O(h^2) error of ~1e-5, a plateau shorter than the taper gave nonsense, and weights chi_l (1 - chi_(l+1)) do not telescope:
    all three found by the mass and convergence checks and fixed while building this.)"""
    from scipy.special import erfc
    s2 = sig ** 2; T = [Rm.T @ (S + s2 * np.eye(2)) @ Rm for S in Ss]; M = [Rm.T @ m for m in mus]
    others = [i for i in range(len(ws)) if i != k]
    h = [np.array([min(1 / np.sqrt(np.linalg.inv(T[i])[j, j]) for i in others) / coarse for j in range(2)])]
    hf = np.array([1 / np.sqrt(np.linalg.inv(T[k])[j, j]) / fine for j in range(2)])
    while np.any(h[-1] > hf * 1.0001):
        h.append(np.maximum(h[-1] / refine, hf))
    c = M[k]; sdk = np.sqrt(np.diag(T[k]))
    def chi(l, Yvw):                                         # level-l bump (l >= 1)
        L = taper * h[l - 1]; P = np.maximum(box * sdk, 5 * L)
        return np.prod([0.5 * erfc((np.abs(Yvw[:, j] - c[j]) - P[j]) / L[j]) for j in range(2)], 0), P + 6 * L
    lo = np.min([m - pad * np.sqrt(np.diag(t)) for m, t in zip(M, T)], 0); hi = np.max([m + pad * np.sqrt(np.diag(t)) for m, t in zip(M, T)], 0)
    Ys, Ws = [], []
    for l in range(len(h)):
        if l == 0:
            g = [lo[j] + (np.arange(int(np.ceil((hi[j] - lo[j]) / h[0][j]))) + 0.5) * h[0][j] for j in range(2)]
        else:
            ext = chi(l, np.zeros((1, 2)) + c)[1]
            g = [c[j] - ext[j] + (np.arange(int(np.ceil(2 * ext[j] / h[l][j]))) + 0.5) * h[l][j] for j in range(2)]
        A, B = np.meshgrid(g[0], g[1], indexing='ij'); Yl = np.stack([A.ravel(), B.ravel()], -1)
        cl = chi(l, Yl)[0] if l >= 1 else np.ones(len(Yl)); cn = chi(l + 1, Yl)[0] if l + 1 < len(h) else np.zeros(len(Yl))
        wl = h[l][0] * h[l][1] * (cl - cn)                  # telescoping: sum over levels of (chi_l - chi_(l+1)) = 1 exactly
        keep = np.abs(wl) > 1e-300; Ys.append(Yl[keep]); Ws.append(wl[keep])
    Yvw = np.concatenate(Ys); return Yvw[:, :1] * V + Yvw[:, 1:] * W, np.concatenate(Ws)


if __name__ == '__main__':
    F = json.load(open('results/f1_chains.json')); b = max(F, key=lambda r: r['final']['ratio'])
    ws, mus, Ss, sig = b['ws'], [np.array(m) for m in b['mus']], [np.array(S) for S in b['Ss']], b['sigma']
    trk = [w * np.trace(np.linalg.inv(S + sig ** 2 * np.eye(2))) for w, S in zip(ws, Ss)]; k = int(np.argmax(trk))
    Svw = Rm.T @ Ss[k] @ Rm
    ru = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=10); rc = evaluate_full(ws, mus, Ss, sig, points=composite(ws, mus, Ss, sig, k))
    rc2 = evaluate_full(ws, mus, Ss, sig, points=composite(ws, mus, Ss, sig, k, fine=20, coarse=18, box=16.0, taper=4.0))
    print(f"t = 1 check: uniform V/W grid R = {ru['R']:.10f}, nested grid R = {rc['R']:.10f} / refined {rc2['R']:.10f}; uniform - nested {ru['R'] - rc['R']:.1e}")
    check1 = {'R_uniform': ru['R'], 'R_nested': rc['R'], 'R_nested_refined': rc2['R'], 'mass_nested': rc['mass']}
    out = []
    for t in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
        D = np.diag([t ** 2, t]); Sk = Rm @ (np.sqrt(D) @ Svw @ np.sqrt(D)) @ Rm.T
        Ss2 = list(Ss); Ss2[k] = Sk; s2 = sig * t
        r = evaluate_full(ws, mus, Ss2, s2, points=composite(ws, mus, Ss2, s2, k)); rn = evaluate_full([1.0], [mus[k]], [Sk], s2, grid='vw', per_sd=10)
        rr = evaluate_full(ws, mus, Ss2, s2, points=composite(ws, mus, Ss2, s2, k, fine=16, coarse=14, box=14.0))
        trk2 = [w * np.trace(np.linalg.inv(S + s2 ** 2 * np.eye(2))) for w, S in zip(ws, Ss2)]
        rec = {'R_refined': rr['R'], 't': t, 'sigma': s2, 'R': r['R'], 'R_needle_alone': rn['R'], 'lam_needle': rn['lam_rho_direct'],
               'law': r['law_term'], 'lam_rho': r['lam_rho_direct'], 'fraction': (r['R'] - r['law_term']) / (1 - r['law_term']),
               'fisher_share_needle': trk2[k] / sum(trk2), 'mass': r['mass'], 'factors': factors(r)}
        out.append(rec)
        print(f"t={t:<5} sigma={s2:.2e} R={r['R']:.6f} (needle alone {rn['R']:.6f}, lambda_needle {rn['lam_rho_direct']:.6f}) fraction={rec['fraction']:.6f} "
              f"Fisher share {rec['fisher_share_needle']:.5f} (refined composite diff {rr['R'] - r['R']:.1e})")
    json.dump({'check_t1': check1, 'rows': out}, open('results/f5_needle.json', 'w'), indent=1)
