"""Stage L, task L2: a FAN of bent needles for the twist flow H = ln(1 + q^2 + p^2)/2 (kappa = 1/4 on the circle r = 1).
Stage I: straight Gaussian needle gap (sqrt 3/2) sigma; bent needle (arc of segments, curvature = turning rate 1/sqrt 2) 0.509 sigma
(prediction sigma/2).  Here K bent needles (each an arc of n = 12 segments, i5_arc.arc) share the centre and the curvature profile
and differ in their tilt:  angle_j = angle + Delta sqrt(sigma) z_j  (fan angles scale as sqrt(sigma), as found for the pendulum in stage H),  z_j uniform in [-1, 1],  weights ~ exp(-z_j^2/(2 zeta^2)),  curvature
k_j = k + kp z_j.  Parameters: the six of i5_arc (dr, tilt, k, log Ls, log thickness, log c_len) + (log Delta, log zeta, kp).
Minimise (kappa - R)/sigma at sigma = 1e-3 (Nelder-Mead from the best bent needle with Delta small, two starts), K = 4, 8, 16.
Final states: Gauss-Hermite 6 nodes and per_sd 10 (grid check: Gauss-Hermite 8, >= 401 points per axis), sigma = 1e-4 and 1e-2 (scale invariance), and the
direct-summation evaluator at sigma = 1e-2 (independent check, small K only).  -> results/l2_bentfan.json"""
import numpy as np, json, sys
from multiprocessing import Pool
from scipy.optimize import minimize
from i5_arc import arc
from i_tools import fields

SIG = 1e-3; KIND = 'twist'; NSEG = 12


def bentfan(x, K, sig=SIG, nseg=NSEG):
    base = np.array(x[:6]); lD, lz, kp = x[6:9]
    z = np.linspace(-1, 1, K) if K > 1 else np.zeros(1); wz = np.exp(-z ** 2 / (2 * np.exp(lz) ** 2)); wz = wz / wz.sum()
    ws, mus, Ss = [], [], []
    for zj, wj in zip(z, wz):
        b = base.copy(); b[1] = base[1] + np.exp(lD) * zj / np.sqrt(sig); b[2] = base[2] + kp * zj   # tilt in units of sigma -> spread exp(lD) sqrt(sigma) rad
        a_ws, a_mus, a_Ss = arc(b, nseg, sig)
        ws += [wj * w for w in a_ws]; mus += a_mus; Ss += a_Ss
    return ws, mus, Ss


def gap(x, K, sig=SIG, gh=3, per_sd=5, min_axis=161, nseg=NSEG):
    if abs(x[6]) > 12 or abs(x[7]) > 6 or abs(x[8]) > 20 or not (-3 < x[3] < 3) or not (-12 < x[4] < 1) or not (-3 < x[5] < 3): return np.inf
    ws, mus, Ss = bentfan(x, K, sig, nseg)
    try: r = fields(ws, mus, Ss, sig, kind=KIND, per_sd=per_sd, gh=gh, min_axis=min_axis)
    except (MemoryError, ValueError, np.linalg.LinAlgError): return np.inf
    return (0.25 - r['R']) / sig if abs(r['mass'] - 1) < 1e-6 else np.inf


def job(a):
    K, starts = a
    b5 = max(json.load(open('results/i5_arc.json')), key=lambda o: o['gstar_n48']); x5 = np.array(b5['x'])
    best = None
    for lD in starts:
        x0 = np.concatenate([x5, [lD, 0.0, 0.0]])
        r = minimize(lambda z: gap(z, K), x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-7})
        if best is None or r.fun < best.fun: best = r
    x = best.x; res = {'K': K, 'x': list(x), 'gap_search': best.fun}
    res['gap'] = gap(x, K, gh=6, per_sd=10); res['gap_fine'] = gap(x, K, gh=8, per_sd=10, min_axis=401); res['gap_nseg24'] = gap(x, K, gh=6, per_sd=10, nseg=24)
    res['gap_sigma1e-4'] = gap(x, K, sig=1e-4, gh=6, per_sd=10); res['gap_sigma1e-2'] = gap(x, K, sig=1e-2, gh=6, per_sd=10)
    ws, mus, Ss = bentfan(x, K); res['R'] = 0.25 - res['gap'] * SIG; res['n_components'] = len(ws)
    res['single_arc_gap'] = (0.25 - fields(*arc(x[:6], NSEG, SIG), SIG, kind=KIND, per_sd=10, gh=6)['R']) / SIG
    res['single_arc_gap_nseg24'] = (0.25 - fields(*arc(x[:6], 24, SIG), SIG, kind=KIND, per_sd=10, gh=6)['R']) / SIG
    res['stageI_best_arc_nseg24'] = (0.25 - fields(*arc(x5, 24, SIG), SIG, kind=KIND, per_sd=10, gh=6)['R']) / SIG
    return res


if __name__ == '__main__':
    out = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, [(4, [np.log(0.3), np.log(1.0)]), (8, [np.log(0.3), np.log(1.0)]), (16, [np.log(0.5)])]):
            out.append(r); json.dump(out, open('results/l2_bentfan.json', 'w'), indent=1, default=float)
            print(f"K = {r['K']}: (kappa - R)/sigma {r['gap']:.4f} (search {r['gap_search']:.4f}, fine {r['gap_fine']:.4f}, 24 segments {r['gap_nseg24']:.4f}, sigma 1e-4 {r['gap_sigma1e-4']:.4f}, "
                  f"1e-2 {r['gap_sigma1e-2']:.4f}); its central arc alone {r['single_arc_gap']:.4f}", flush=True)
