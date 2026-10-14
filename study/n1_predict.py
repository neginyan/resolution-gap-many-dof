"""Stage N, task N1(c): SEALED prediction (written before the runs) of the mixture gain at K = 8 for two coupled pendulums not used
in the fit:  eps = 1 (top (pi, 0)) and eps = -0.3 (top (pi, pi)).
Rule (an ESTIMATE fitted on eps = 0.1, 0.3, K = 8):   gain_3d = gain_1d x f^a,   f = 2 sqrt(2 B11 kappa) / Gamma
(the share of the 1-d term in the Gaussian coefficient), a = least-squares exponent through the origin in log-log (both eps).
Also written: the range from the two individual exponents, and the raw "effective beta" model (B11 -> B11 (1 - g1)^2, B12 -> B12 (1 - g1)),
which over-predicted the gain on eps = 0.1, 0.3 (by a factor 1.4-1.9).
Writes results/n1_prediction.json (read-only afterwards) and appends its SHA-256 with the UTC time to logs/n1_prediction_hash.txt."""
import numpy as np, json, hashlib, datetime
from scipy.linalg import sqrtm
from m_tools import coupled, find_kappa, coefficient, B_matrix

if __name__ == '__main__':
    runs = json.load(open('results/n1_mix.json')) + json.load(open('results/n1_more.json')); best = {}
    for o in runs:
        k = (o['kind'], o['K'])
        if k not in best or o['gap'] < best[k]['gap']: best[k] = o
    g1 = 1 - best[('1d', 8)]['ratio']; fit = {}
    def share(e):
        s = coupled(e); k, t = find_kappa(s, starts=100); c = coefficient(s, t[0], k); return c['diag_terms'][0] / c['coef'], s, k, t[0]
    for e in [0.1, 0.3]:
        f, *_ = share(e); fit[e] = (np.log(f), np.log((1 - best[(f'{e:g}', 8)]['ratio']) / g1))
    xs, ys = np.array(list(fit.values())).T; a = float(np.sum(xs * ys) / np.sum(xs * xs)); a_each = (ys / xs).tolist()
    pred = []
    for e in [1.0, -0.3]:
        f, s, k, q0 = share(e); B, _ = B_matrix(s, q0); C = np.diag([2 * k, k])
        G = lambda Bm: 2 * np.trace(np.real(sqrtm(np.real(sqrtm(Bm)) @ C @ np.real(sqrtm(Bm)))))
        D = np.diag([1 - g1, 1]); eff = 1 - G(D @ B @ D) / G(B)
        pred.append({'eps': e, 'top': q0.tolist(), 'kappa': k, 'plane_share_f': f, 'predicted_gain': g1 * f ** a, 'predicted_ratio': 1 - g1 * f ** a,
                     'predicted_ratio_range': sorted([1 - g1 * f ** min(a_each), 1 - g1 * f ** max(a_each)]), 'effective_beta_gain': eff})
        print(pred[-1], flush=True)
    txt = json.dumps({'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gain_1d_K8': g1, 'a_fit': a, 'a_each': a_each, 'K': 8, 'predictions': pred}, indent=1)
    open('results/n1_prediction.json', 'w').write(txt); h = hashlib.sha256(txt.encode()).hexdigest()
    line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}  sha256(results/n1_prediction.json) = {h}\n"; open('logs/n1_prediction_hash.txt', 'a').write(line); print(line)
