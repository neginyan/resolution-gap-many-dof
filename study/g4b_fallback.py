"""Stage G, task 3 (supplement): states whose nested grid did not fit in the 3 GB worker limit of g4_recheck.py are checked
instead on the (q, p) grid at n = 1200 and 1800 and on an uncapped coarser V/W grid (per_sd = 6).  Updates results/g4_recheck.json."""
import json, numpy as np
from g4_recheck import states
from obs_eval2 import evaluate_full

if __name__ == '__main__':
    d = json.load(open('results/g4_recheck.json')); S = states()
    for i, r in enumerate(d):
        if r.get('failed'):
            tag, ws, mus, Ss, sig, R_old = S[i]
            a = evaluate_full(ws, mus, Ss, sig, n=1200); b = evaluate_full(ws, mus, Ss, sig, n=1800); c = evaluate_full(ws, mus, Ss, sig, grid='vw', per_sd=6)
            r.update({'R_alt_qp1200': a['R'], 'R_alt_qp1800': b['R'], 'R_alt_vw6': c['R'], 'n_alt_vw6': list(c['n']),
                      'diff_alt': max(abs(x - R_old) for x in [a['R'], b['R'], c['R']])})
            print(i, tag, r['diff_alt'])
    json.dump(d, open('results/g4_recheck.json', 'w'), indent=1)
