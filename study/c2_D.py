"""Stage C, proposal 2: what sets the posterior Stein defect D?

Exact split (obs_eval2.py), with posterior "guesses" k (weights pi_k(y), means x_k(y), covariances V_k):
  D = D_within + D_between
  D_within  = -E[sum_k pi_k (g'_k - gbar') V_k,qp] / (sigma^4 tr F)                 (the guesses have different widths)
  D_between = -E[sum_k pi_k (g_k - gbar - gbar' dx_kq) dx_kp] / (sigma^4 tr F)      (the flow bends between the guesses)
Candidate scales: S_D (curvature sup|g''| = 1 times the spread of the guesses, see obs_eval2.py), and the simple product
"responsibility variance x stretch gap between components" E[sum pi_k (1 - pi_k)] * max_k,j |abar_k - abar_j|.
States: the 1000 random mixtures, the 128 measurement-1 slice states and the stage-C adversarial states.
Writes results/c2_D.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full
from m1_mechanism import state
from c1_adversarial import unpack

KEEP = ['R', 'f', 'D', 'D_within', 'D_between', 'S_D', 'resp_var', 'stretch_gap', 'H_nonPD_mass', 'law_term', 'sigma', 'n', 'mass']


def run(job):
    tag, ws, mus, Ss, sig, extra = job
    r = evaluate_full(ws, mus, Ss, sig)
    out = {k: float(r[k]) for k in KEEP}; out.update({'tag': tag}); out.update(extra)
    return out


if __name__ == '__main__':
    jobs = []
    for m in json.load(open('results/mixtures.json')):
        jobs.append(('random', m['ws'], [np.array(x) for x in m['mus']], [np.array(S) for S in m['Ss']], m['sigma'],
                     {'overlap': m['overlap'], 'n_grid': m['n']}))
    for d in json.load(open('results/m1_slices.json')):
        p = dict(d['params']); jobs.append(('m1', *state(**p), 0.1, {'overlap': d['overlap'], 'n_grid': int(d['n'])}))
    for fn in ['results/c1_adversarial.json', 'results/c1_adversarial_room.json']:
        for c in json.load(open(fn)):
            jobs.append(('c1', *unpack(np.array(c['x']), c['K']), c['sigma'], {'overlap': c['overlap_max'], 'n_grid': int(c['n'])}))
    light = [j for j in jobs if j[5]['n_grid'] <= 1500]; heavy = [j for j in jobs if j[5]['n_grid'] > 1500]
    with Pool(2) as pool:
        out = pool.map(run, light, chunksize=4)
    out += [run(j) for j in heavy]                     # ~3 GB each: one at a time
    json.dump(out, open('results/c2_D.json', 'w'), indent=1)
    print(len(out))
