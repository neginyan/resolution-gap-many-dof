"""Stage F: compare the R of a mixture with the best R of its components taken alone (single-Gaussian law 1, same sigma),
over all stored pendulum states (random, measurement-1, stage-C/E/F searches, near-top searches).  Writes results/f6_components.json."""
import json, numpy as np
from m1_mechanism import state
from c1_adversarial import unpack


def Rk_max(ws, mus, Ss, sig):
    best = -9
    for mu, S in zip(mus, Ss):
        S = np.array(S); J = np.array([[0, 1.0], [-np.cos(mu[0]) * np.exp(-S[0, 0] / 2), 0]]); F = np.linalg.inv(S + sig ** 2 * np.eye(2))
        best = max(best, np.trace(F @ (J + J.T) / 2) / np.trace(F))
    return float(best)


if __name__ == '__main__':
    rows = []
    for m, r in zip(json.load(open('results/mixtures.json')), json.load(open('results/m2_mix.json'))): rows.append(('random', r['R'], Rk_max(m['ws'], m['mus'], m['Ss'], m['sigma'])))
    for fn in ['results/m1_slices.json', 'results/m1_maps.json']:
        for d in json.load(open(fn)): rows.append(('m1', d['R'], Rk_max(*state(**d['params']), 0.1)))
    for fn in ['results/c1_adversarial.json', 'results/c1_adversarial_room.json']:
        for c in json.load(open(fn)): rows.append(('c1', c['R'], Rk_max(*unpack(np.array(c['x']), c['K']), c['sigma'])))
    for r in json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')): rows.append(('e1', r['R'], Rk_max(r['ws'], r['mus'], r['Ss'], r['sigma'])))
    for r in json.load(open('results/f1_chains.json')) + json.load(open('results/f4_rcvo.json')): rows.append(('f', r['final']['R'], Rk_max(r['ws'], r['mus'], r['Ss'], r['sigma'])))
    for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) + json.load(open('results/e2_dense.json')):
        if r.get('feasible'): rows.append(('near-top', r['R'], Rk_max(r['ws'], r['mus'], r['Ss'], r['sigma'])))
    R = np.array([r[1] for r in rows]); M = np.array([r[2] for r in rows])
    out = {'n': len(rows), 'n_R_gt_maxRk': int((R > M + 1e-9).sum()), 'max_R_minus_maxRk': float((R - M).max()),
           'bins': {f'{lo}-{hi}': {'n': int(((M >= lo) & (M < hi)).sum()), 'n_exceed': int(((M >= lo) & (M < hi) & (R > M + 1e-9)).sum()),
                                   'max_R_minus_maxRk': float((R - M)[(M >= lo) & (M < hi)].max())} for lo, hi in [(-1, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]},
           'rows': [{'tag': t, 'R': a, 'maxRk': b} for t, a, b in rows]}
    json.dump(out, open('results/f6_components.json', 'w'), indent=1)
    print({k: v for k, v in out.items() if k != 'rows'})
