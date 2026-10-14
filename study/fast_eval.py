"""Exact evaluation of dS_C/dt at t = 0 for Gaussian-mixture initial states (no time stepping, no nodes).

Y = X + N, N ~ N(0, sigma^2 I).   dS/dt = r sigma^2 tr F + Sigma,   Sigma = -E[s(Y) . f(X)],
with s = grad log p_Y (score identity), Cdot = 2 r sigma^2 I.
For each Gaussian component (mu, S): p_k(y) = N(y; mu, S + sigma^2 I) and X | Y = y is Gaussian with
mean mu + S (S + sigma^2 I)^{-1} (y - mu) and covariance S - S (S + sigma^2 I)^{-1} S, so
E[f(X) | Y] is analytic for f = (p, -sin q) (pendulum) and f = (p, -q^3) (quartic).
Only the final y-integral is done by quadrature on a grid.
"""
import numpy as np


def cond_force(kind, mq, vq):
    if kind == 'pendulum':
        return -np.sin(mq) * np.exp(-vq / 2)
    if kind == 'quartic':
        return -(mq ** 3 + 3 * mq * vq)
    if kind == 'harmonic':
        return -mq
    raise ValueError(kind)


def dSdt(ws, mus, Ss, sig, r, kind, n=401, pad=8.0):
    s2 = sig * sig
    T = [S + s2 * np.eye(2) for S in Ss]
    sd = [np.sqrt(np.diag(t)) for t in T]
    lo = np.min([m - pad * d for m, d in zip(mus, sd)], 0)
    hi = np.max([m + pad * d for m, d in zip(mus, sd)], 0)
    xq = np.linspace(lo[0], hi[0], n); xp = np.linspace(lo[1], hi[1], n)
    dA = (xq[1] - xq[0]) * (xp[1] - xp[0])
    Y = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1)
    p = 0; g = 0; m = 0
    for w, mu, S, Tk in zip(ws, mus, Ss, T):
        Ti = np.linalg.inv(Tk)
        d = Y - mu
        pk = w * np.exp(-0.5 * np.einsum('...i,ij,...j->...', d, Ti, d)) / (2 * np.pi * np.sqrt(np.linalg.det(Tk)))
        K = S @ Ti
        mpost = mu + d @ K.T
        Vpost = S - K @ S
        fq = mpost[..., 1]                                   # E[p | Y]
        fp = cond_force(kind, mpost[..., 0], Vpost[0, 0])    # E[force(q) | Y]
        p = p + pk
        g = g + pk[..., None] * (-(d @ Ti.T))
        m = m + pk[..., None] * np.stack([fq, fp], -1)
    ok = p > 1e-300
    trF = np.sum(np.sum(g[ok] ** 2, -1) / p[ok]) * dA
    Sigma = -np.sum(np.sum(m[ok] * g[ok], -1) / p[ok]) * dA
    return r * s2 * trF + Sigma, trF, Sigma
