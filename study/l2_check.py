"""Stage L, task L2: independent check of the bent fans.  For each optimised bent fan (results/l2_bentfan.json, K = 4 and 8) the
gap at sigma = 1e-2 is computed (a) with the grid evaluator i_tools.fields (Gauss-Hermite posterior, 8 nodes, per_sd 10) and (b) with
the direct kernel summation R_brute of verification/indep_eval.py with the pendulum flow replaced by the twist flow (no closed forms,
no Gauss-Hermite).  Results are added to results/l2_bentfan.json under 'indep'.
Also the continuum check: every fan and its central arc with 48 segments per needle ('gap_nseg48', 'single_arc_gap_nseg48')."""
import numpy as np, json, os
from l2_bentfan import bentfan
from i_tools import fields

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'verification', 'indep_eval.py')).read().replace("fX = np.stack([Xn[:, 1], -np.sin(Xn[:, 0])], -1)",
                                                                                "fX = np.stack([Xn[:, 1], -Xn[:, 0]], -1) / (1 + Xn[:, 0] ** 2 + Xn[:, 1] ** 2)[:, None]")
ns = {}; exec(compile(src, 'indep_eval_twist', 'exec'), ns)

if __name__ == '__main__':
    res = json.load(open('results/l2_bentfan.json')); sig = 1e-2
    from l2_bentfan import gap, SIG
    from i5_arc import arc
    for r in res:      # continuum check: 48 segments per needle (stage I's arc changes from 0.503 (24) to 0.509 (48))
        x = np.array(r['x']); r['gap_nseg48'] = gap(x, r['K'], gh=6, per_sd=8, nseg=48)
        r['single_arc_gap_nseg48'] = (0.25 - fields(*arc(x[:6], 48, SIG), SIG, kind='twist', per_sd=8, gh=6)['R']) / SIG
        print(r['K'], 'nseg48', r['gap_nseg48'], 'arc', r['single_arc_gap_nseg48'], flush=True)
        json.dump(res, open('results/l2_bentfan.json', 'w'), indent=1, default=float)
    for r in res:
        if r['K'] > 8: continue
        ws, mus, Ss = bentfan(np.array(r['x']), r['K'], sig)
        g = (0.25 - fields(ws, mus, Ss, sig, kind='twist', per_sd=10, gh=8)['R']) / sig
        b = (0.25 - ns['R_brute'](ws, mus, Ss, sig, ny=161)) / sig
        r['indep'] = {'sigma': sig, 'gap_grid': g, 'gap_brute': b}; print(r['K'], g, b, flush=True)
    json.dump(res, open('results/l2_bentfan.json', 'w'), indent=1, default=float)
