"""Stage G, task 3: re-check every stored pendulum state whose uniform V/W grid hits its size cap (2401 points per axis or
3e6 points in total), with the nested partition-of-unity grid of f5_needle.composite refined around the component that carries
the most Fisher information.  For each capped state: R on the nested grid, on a refined nested grid, and the stored R.
Writes results/g4_recheck.json."""
import numpy as np, json
from multiprocessing import Pool
from obs_eval2 import evaluate_full, vw_grid
from f5_needle import composite
from m1_mechanism import state
from c1_adversarial import unpack as unpack_c1

A = lambda L: [np.array(x) for x in L]


def states():
    out = []
    for m, r in zip(json.load(open('results/mixtures.json')), json.load(open('results/m2_mix.json'))): out.append(('random', m['ws'], A(m['mus']), A(m['Ss']), m['sigma'], r['R']))
    for fn in ['results/m1_slices.json', 'results/m1_maps.json']:
        for d in json.load(open(fn)): out.append(('m1', *state(**d['params']), 0.1, d['R']))
    for fn in ['results/c1_adversarial.json', 'results/c1_adversarial_room.json']:
        for c in json.load(open(fn)): out.append(('c1', *unpack_c1(np.array(c['x']), c['K']), c['sigma'], c['R']))
    for r in json.load(open('results/e1_local.json')) + json.load(open('results/e1b_continue.json')): out.append(('e1', r['ws'], A(r['mus']), A(r['Ss']), r['sigma'], r['R']))
    for r in json.load(open('results/f1_chains.json')) + json.load(open('results/f4_rcvo.json')): out.append(('f', r['ws'], A(r['mus']), A(r['Ss']), r['sigma'], r['final']['R']))
    for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) + json.load(open('results/e2_dense.json')):
        if r.get('feasible'): out.append(('near-top', r['ws'], A(r['mus']), A(r['Ss']), r['sigma'], r['R']))
    for r in json.load(open('results/g2_search.json')): out.append(('g2', r['ws'], A(r['mus']), A(r['Ss']), r['sigma'], r['R']))
    return out


def limit_memory(gb=3.0):
    import resource; resource.setrlimit(resource.RLIMIT_AS, (int(gb * 2 ** 30), int(gb * 2 ** 30)))


SETTINGS = [({}, dict(fine=16, coarse=14, box=14.0)), (dict(fine=8, coarse=6), dict(fine=10, coarse=8)), (dict(fine=6, coarse=4, box=8.0), dict(fine=8, coarse=6, box=10.0))]


def run(job):
    tag, ws, mus, Ss, sig, R_old = job
    _, _, n, _ = vw_grid(ws, mus, Ss, sig, per_sd=10)
    capped = max(n) >= 2401 or n[0] * n[1] >= 2_850_000
    rec = {'tag': tag, 'sigma': sig, 'R_stored': R_old, 'n_vw': list(n), 'capped': bool(capped)}
    if capped and len(ws) > 1:
        k = int(np.argmax([w * np.trace(np.linalg.inv(S + sig ** 2 * np.eye(2))) for w, S in zip(ws, Ss)]))
        for level, (a, b) in enumerate(SETTINGS):     # memory-capped worker: fall back to coarser nested grids if a grid does not fit
            try:
                r1 = evaluate_full(ws, mus, Ss, sig, points=composite(ws, mus, Ss, sig, k, **a))
                r2 = evaluate_full(ws, mus, Ss, sig, points=composite(ws, mus, Ss, sig, k, **b))
                rec.update({'R_nested': r1['R'], 'R_nested_refined': r2['R'], 'mass_nested': r1['mass'], 'diff': r1['R'] - R_old, 'grid_level': level})
                break
            except MemoryError:
                continue
        else:
            rec['failed'] = 'memory'
    return rec


if __name__ == '__main__':
    S = states()
    import sys
    out = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1, maxtasksperchild=20, initializer=limit_memory) as pool:
        for i, r in enumerate(pool.imap(run, S, chunksize=1)):
            out.append(r)
            if r.get('R_nested') is not None or i % 100 == 0: print(i, len(S), r['tag'], r['capped'], r.get('diff'), flush=True)
    json.dump(out, open('results/g4_recheck.json', 'w'), indent=1)
    cap = [r for r in out if 'R_nested' in r]; print('failed', sum('failed' in r for r in out), 'fallback levels', [r['grid_level'] for r in cap if r['grid_level']])
    print('states', len(out), 'capped', sum(r['capped'] for r in out), 'rechecked', len(cap))
    if cap:
        print('max |nested - stored|', max(abs(r['diff']) for r in cap), 'max |refined - nested|', max(abs(r['R_nested_refined'] - r['R_nested']) for r in cap),
              'max mass dev', max(abs(r['mass_nested'] - 1) for r in cap), 'max nested R', max(r['R_nested'] for r in cap))
