"""Stage A sweep: single Gaussian states moved from the hyperbolic point q = pi to the stable point q = 0,
five shapes, 25 values of sigma in [0.01, 2]. Writes results/sweep.json."""
import numpy as np, json, os, time
from obs_eval import evaluate, cand_b, cand_c, cand_d, SHAPES, Q0, SIGMAS

os.makedirs('results', exist_ok=True)
out = []
t0 = time.time()
for shape, S in SHAPES.items():
    for q0 in Q0:
        mu = np.array([q0, 0.0])
        rec = {'shape': shape, 'q0': q0, 'q0_over_pi': q0 / np.pi, 'S': S.tolist(),
               'b': cand_b(mu), 'c': cand_c([1], [mu], [S]), 'rows': []}
        for sig in SIGMAS:
            r = evaluate([1], [mu], [S], sig)
            r['d'] = cand_d([1], [mu], [S], sig)
            rec['rows'].append(r)
        out.append(rec)
        print(f"{shape:40s} q0/pi={q0/np.pi:.1f}  c={rec['c']:+.4f}  R(0.01)={rec['rows'][0]['R']:+.4f}  "
              f"R(0.3)={rec['rows'][12]['R']:+.4f}  R(2)={rec['rows'][-1]['R']:+.4f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open('results/sweep.json', 'w'), indent=1, default=float)
