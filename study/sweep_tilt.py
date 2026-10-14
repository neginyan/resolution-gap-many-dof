"""Additional shapes: anisotropic Gaussians (0.05 x 0.5) tilted at 22.5 and 67.5 degrees, i.e. not aligned with the
stretching direction (45 degrees). Writes results/sweep_tilt.json (same format as results/sweep.json)."""
import numpy as np, json
from obs_eval import evaluate, cand_b, cand_c, cand_d, SIGMAS
out = []
for deg in [22.5, 67.5]:
    th = np.deg2rad(deg); u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    S = 0.05 ** 2 * np.outer(u, u) + 0.5 ** 2 * np.outer(v, v)      # narrow along u (angle deg)
    for k in [10, 8, 6, 4, 2, 0]:
        q0 = np.pi * k / 10; mu = np.array([q0, 0.0])
        rec = {'shape': f'tilted {deg} deg (0.05 x 0.5)', 'q0': q0, 'q0_over_pi': k / 10, 'S': S.tolist(),
               'b': cand_b(mu), 'c': cand_c([1], [mu], [S]), 'rows': []}
        for sig in SIGMAS:
            r = evaluate([1], [mu], [S], sig); r['d'] = cand_d([1], [mu], [S], sig); rec['rows'].append(r)
        out.append(rec); print(rec['shape'], k / 10, round(rec['c'], 4), round(rec['rows'][0]['R'], 4), round(rec['rows'][-1]['R'], 4), flush=True)
json.dump(out, open('results/sweep_tilt.json', 'w'), indent=1, default=float)
