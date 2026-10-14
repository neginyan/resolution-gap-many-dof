"""Candidate (e) on the random two-component mixtures and on the counterexample (preliminary hint for stage B).
Writes results/mixtures_e.json."""
import json, numpy as np
from obs_eval import evaluate
from adversarial import unpack
M = json.load(open('results/mixtures.json'))
out = []
for m in M[::5]:
    r = evaluate(m['ws'], [np.array(x) for x in m['mus']], [np.array(S) for S in m['Ss']], m['sigma'])
    out.append({'sigma': m['sigma'], 'R': r['R'], 'e': r['cand_e'], 'R_sep': m['R_sep'], 'overlap': m['overlap']})
adv = json.load(open('results/adversarial_ratio_0.1.json'))['ratio_0.1']
ws, mus, Ss = unpack(np.array(adv['x'])); r = evaluate(ws, mus, Ss, 0.1)
res = {'sample': out, 'counterexample': {'R': r['R'], 'e': r['cand_e']}}
json.dump(res, open('results/mixtures_e.json', 'w'), indent=1, default=float)
R = np.array([o['R'] for o in out]); e = np.array([o['e'] for o in out]); Rs = np.array([o['R_sep'] for o in out]); ov = np.array([o['overlap'] for o in out])
k = ov >= 0.05
print('n', len(out), ' overlap>=0.05:', k.sum())
print('overlapping: median |R-e| %.4f  max %.4f   |  median |R-R_sep| %.4f  max %.4f' % (np.median(abs(R-e)[k]), abs(R-e)[k].max(), np.median(abs(R-Rs)[k]), abs(R-Rs)[k].max()))
print('all: max |R-e| %.4f; R > e in %d of %d' % (abs(R-e).max(), (R > e + 1e-12).sum(), len(R)))
print('counterexample: R %.4f  e %.4f' % (r['R'], r['cand_e']))
