"""Independent evaluators of R used by the stage-H checks (no code shared with obs_eval2):
  R_indep : analytic p_Y and score; posterior expectation E[f(X)|y] by 2-d Gauss-Hermite quadrature over each component's
            posterior (instead of the closed form E[sin q] = sin m e^{-C/2}); y-quadrature on a V/W-aligned midpoint grid.
  R_brute : direct summation over prior nodes (kernel sums for p_Y, its gradient and E[f|y]); no closed forms."""
import numpy as np
V = np.array([1, 1]) / np.sqrt(2); W = np.array([1, -1]) / np.sqrt(2)


def R_indep(ws, mus, Ss, sig, nv=500, nw=500, ext=9.0, gh=24):
    T = [S + sig ** 2 * np.eye(2) for S in Ss]; Ti = [np.linalg.inv(t) for t in T]
    c = sum(w * m for w, m in zip(ws, mus))
    lov = min(V @ (m - c) - ext * np.sqrt(V @ t @ V) for m, t in zip(mus, T)); hiv = max(V @ (m - c) + ext * np.sqrt(V @ t @ V) for m, t in zip(mus, T))
    low = min(W @ (m - c) - ext * np.sqrt(W @ t @ W) for m, t in zip(mus, T)); hiw = max(W @ (m - c) + ext * np.sqrt(W @ t @ W) for m, t in zip(mus, T))
    hv = (hiv - lov) / nv; hw = (hiw - low) / nw
    a = lov + (np.arange(nv) + 0.5) * hv; b = low + (np.arange(nw) + 0.5) * hw
    A, B = np.meshgrid(a, b, indexing='ij'); Y = c + A.reshape(-1, 1) * V + B.reshape(-1, 1) * W
    xg, wg = np.polynomial.hermite_e.hermegauss(gh); wg = wg / wg.sum()
    comps = []
    for w, m, S, ti, t in zip(ws, mus, Ss, Ti, T):
        d = Y - m; dens = w * np.exp(-0.5 * np.einsum('ni,ij,nj->n', d, ti, d)) / (2 * np.pi * np.sqrt(np.linalg.det(t)))
        K = S @ ti; mpost = m + d @ K.T; C = S - K @ S
        ev, U = np.linalg.eigh(C); ev = np.maximum(ev, 0); L = U * np.sqrt(ev)
        Ef = np.zeros_like(Y)
        for i in range(gh):
            for j in range(gh):
                X = mpost + L @ np.array([xg[i], xg[j]]); Ef += wg[i] * wg[j] * np.stack([X[:, 1], -np.sin(X[:, 0])], -1)
        comps.append((dens, -d @ ti, Ef))
    p = sum(cc[0] for cc in comps); ok = p > 0                       # cells where p_Y underflows to 0 carry no weight
    p = np.where(ok, p, 1e-300); s = sum(cc[0][:, None] * cc[1] for cc in comps) / p[:, None]; Ef = sum(cc[0][:, None] * cc[2] for cc in comps) / p[:, None]
    p = np.where(ok, p, 0.0)
    wgt = p * hv * hw; trF = np.sum(wgt * (s ** 2).sum(1))
    return float(np.sum(wgt * (s * Ef).sum(1)) / (sig ** 2 * trF)), float(wgt.sum())


def R_brute(ws, mus, Ss, sig, ny=241):
    """direct summation: p_Y(y), grad p_Y(y) and E[f|y] as sums of the kernel N(y; x, sig^2) (and its exact derivative) over prior
    nodes; y-quadrature on a uniform V/W-aligned grid (spectrally accurate for these smooth integrands);
    R = E[s . E[f|y]]/(sig^2 E|s|^2).  Shares no formula with obs_eval2 (no closed-form posterior, no analytic score)."""
    Rm = np.stack([V, W], 1)
    c = sum(w * m for w, m in zip(ws, mus))
    def box(ext, add):
        lo = np.min([Rm.T @ (m - c) - ext * np.sqrt(np.diag(Rm.T @ S @ Rm) + add) for m, S in zip(mus, Ss)], 0)
        hi = np.max([Rm.T @ (m - c) + ext * np.sqrt(np.diag(Rm.T @ S @ Rm) + add) for m, S in zip(mus, Ss)], 0); return lo, hi
    ylo, yhi = box(6.0, sig ** 2)
    yv, yw = np.linspace(ylo[0], yhi[0], ny), np.linspace(ylo[1], yhi[1], ny)
    # the prior may be (nearly) degenerate: each component on its own uniform grid along its principal axes over +-8 s.d.
    # with spacing <= sigma/3 (the kernel width), weights = Gaussian density x spacing (spectrally accurate for these sums)
    nodes, wts = [], []
    for w, m, S in zip(ws, mus, Ss):
        ev, U = np.linalg.eigh(S); sd = np.sqrt(np.maximum(ev, 0))
        zs = []
        for k in range(2):
            n = int(max(41, np.ceil(16 * sd[k] / (sig / 3)))) | 1          # spacing <= sd/2.5 and <= sigma/3
            z = np.linspace(-8, 8, n); g = np.exp(-z ** 2 / 2); zs.append((z, g / g.sum()))
        Z1, Z2 = np.meshgrid(zs[0][0], zs[1][0], indexing='ij'); G = np.outer(zs[0][1], zs[1][1]).ravel()
        Xc = m + np.stack([Z1.ravel() * sd[0], Z2.ravel() * sd[1]], -1) @ U.T
        nodes.append(Xc); wts.append(w * G)
    Xn = np.concatenate(nodes); wn = np.concatenate(wts); fX = np.stack([Xn[:, 1], -np.sin(Xn[:, 0])], -1)
    Y = c + np.stack(np.meshgrid(yv, yw, indexing='ij'), -1).reshape(-1, 2) @ Rm.T
    p = np.zeros(len(Y)); mf = np.zeros((len(Y), 2)); gp = np.zeros((len(Y), 2))
    ch = int(max(20, min(1000, 1e7 / len(Xn))))                                     # bound the memory of the kernel matrix
    for k in range(0, len(Y), ch):
        y = Y[k:k + ch]; Kr = np.exp(-((y[:, None, :] - Xn[None]) ** 2).sum(-1) / (2 * sig ** 2)) / (2 * np.pi * sig ** 2) * wn[None]
        p[k:k + ch] = Kr.sum(1); mf[k:k + ch] = Kr @ fX
        gp[k:k + ch] = (Kr @ Xn - Kr.sum(1)[:, None] * y) / sig ** 2              # grad p_Y = sum K (x - y)/sigma^2 (exact kernel derivative)
    ok = p > 1e-300 * p.max()
    s_ = gp[ok] / p[ok, None]; Ef = mf[ok] / p[ok, None]; wgt = p[ok]               # uniform y-grid: the cell area cancels in the ratio
    trF = np.sum(wgt * (s_ ** 2).sum(1))
    return float(np.sum(wgt * (s_ * Ef).sum(1)) / (sig ** 2 * trF))
