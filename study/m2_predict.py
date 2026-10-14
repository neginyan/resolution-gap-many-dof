"""Stage M, task M2: PREDICTIONS written before any optimisation of the systems below (pre-registration).
For each system: kappa = max_q lambda_max M(q) and the maximising points (m_tools.find_kappa, 400 random starts + Nelder-Mead, checked
on random samples), the simple-eigenvalue condition at the top, B (analytic, m_tools.B_matrix), and the predicted coefficient
    lim_{sigma -> 0} (kappa - R*)/sigma = 2 tr[(B^1/2 C B^1/2)^1/2],   C = diag(2 kappa, kappa, ..., kappa).
Also the predicted shape: thin along V+, covariance sigma T* in (W+, e_k), unbounded along the momentum-only directions (0, e_k).
Systems:
  (a) open pendulum chains  H = sum p_i^2/2 - sum cos q_i - eps sum cos(q_i - q_{i+1}),  n = 3, 4,  eps = 0.1, 0.3
  (b) a system of a different form (substitute for Henon-Heiles, see the note):  two unequal pendulums with a product coupling
      H = (p1^2 + p2^2)/2 - cos q1 - 2 cos q2 - eps cos q1 cos q2,  eps = 0.3, 1
      (cos q1 cos q2 = [cos(q1 + q2) + cos(q1 - q2)]/2).
Note on Henon-Heiles  H = (px^2 + py^2)/2 + (x^2 + y^2)/2 + x^2 y - y^3/3:  V'' = [[1 + 2y, 2x], [2x, 1 - 2y]], so
M = (I - V'')/2 = [[-y, -x], [-x, y]] with eigenvalues +-sqrt(x^2 + y^2): the stretching grows without bound away from the origin,
kappa = infinity, and the saddle points (0, 1), (+-sqrt3/2, -1/2) (lambda_max = 1) are not maxima of the stretching (its gradient
there is nonzero).  The formula, which assumes a quadratic maximum, does not apply; no prediction is made for it.
Writes results/m2_prediction.json and prints its SHA-256 (also appended, with the UTC time, to logs/m2_prediction_hash.txt)."""
import numpy as np, json, hashlib, datetime
from m_tools import System, pendulum_chain, find_kappa, coefficient, frame


def product_coupled(eps):
    return System(2, [(1.0, [1, 0]), (2.0, [0, 1]), (eps / 2, [1, 1]), (eps / 2, [1, -1])], f'unequal pendulums, product coupling eps={eps}')


SYSTEMS = [('chain', n, e, pendulum_chain(n, e)) for n in [3, 4] for e in [0.1, 0.3]] + [('product', 2, e, product_coupled(e)) for e in [0.3, 1.0]]

if __name__ == '__main__':
    out = []
    for kind, n, e, s in SYSTEMS:
        k, tops = find_kappa(s, starts=400, seed=1)
        rng = np.random.default_rng(2); samp = max(s.lam(q) for q in rng.uniform(-np.pi, np.pi, (20000, n)))
        c = coefficient(s, tops[0], k); Fr, ev = frame(s, tops[0])
        coefs_all_tops = [coefficient(s, t, k)['coef'] for t in tops]
        out.append({'kind': kind, 'n': n, 'eps': e, 'name': s.name, 'kappa': k, 'tops': [t.tolist() for t in tops], 'max_on_random_samples': float(samp),
                    'prediction': c['coef'], 'prediction_all_tops': coefs_all_tops, 'B': c['B'], 'B_eigvals': c['B_eigvals'], 'C_diag': c['C_diag'],
                    'eig_M_top': c['eig_M_top'], 'top_eigen_gap': c['top_eigen_gap'], 'diag_terms': c['diag_terms'],
                    'predicted_shape': 'thin along V+ (width -> 0), covariance sigma T* in (W+, e_k), unbounded along (0, e_k) (momentum-only: cost 0)'})
        print(f"{s.name}: kappa {k:.10f} at {[np.round(t, 6).tolist() for t in tops]}  prediction {c['coef']:.8f}  (all tops {np.round(coefs_all_tops, 8).tolist()})", flush=True)
    txt = json.dumps({'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'systems': out}, indent=1)
    open('results/m2_prediction.json', 'w').write(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}  sha256(results/m2_prediction.json) = {h}\n"
    open('logs/m2_prediction_hash.txt', 'a').write(line); print(line)
