"""Stage L, task L1(b): the three candidate ratios for an upper bound on the focus-slope term, on every stored state (stages A-K,
290 states) and on the new states of L1 (free mixtures, polished mixtures, fans, the rho-maximising states of l1_rho_max), with l_tools.diag:
    rho     = -E[u d_w w] / E[w d_w u]
    r_sup   = E[|u d_w w|] / (j sup|u|)
    r_sigma = E[|d_w w|] sigma / j
and the uniform constant that a bound rho <= rho0 would give:  Theta' >= Theta = (geo/j)(1 - rho) (since Lambda/j = geo/j + slope/j
= (geo/j)(1 - rho) and Theta' >= Theta),  so  c = G0 (1 - rho0) / (1 + ...) - eta.   -> results/l1_ratios.json"""
import numpy as np, json, glob, sys
from multiprocessing import Pool
from i2_bound import states
from l_tools import diag
from h2_adversarial import unpack


def new_states():
    out = []
    for f in sorted(glob.glob('results/l1_free_s*_sigma*.json')) + sorted(glob.glob('results/l1_polish_sigma*.json')):
        for r in json.load(open(f)):
            ws, mus, Ss = unpack(np.array(r['x']), r['K'], r['sigma']); out.append((f.split('/')[-1].split('_sigma')[0] + f"_K{r['K']}", ws, mus, Ss, r['sigma']))
    try:
        from l1_fan_ext import fan22
        from i1_fan import fan as fan17
        for r in json.load(open('results/l1_fan_ext.json')):
            x = np.array(r['x']); ws, mus, Ss = fan22(x, 40, r['sigma']) if r['kind'] == 'ext' else fan17(x, 40, r['sigma'])
            out.append(('l1_' + r['kind'], ws, mus, Ss, r['sigma']))
    except FileNotFoundError:
        pass
    try:
        for r in json.load(open('results/l1_rho_max.json')):
            ws, mus, Ss = unpack(np.array(r['x']), r['K'], 1e-3); out.append((f"l1_rho_max_{r['key']}_K{r['K']}", ws, mus, Ss, 1e-3))
    except FileNotFoundError:
        pass
    return out


def run(st):
    tag, ws, mus, Ss, sig = st
    try: d = diag(ws, mus, Ss, sig)
    except (MemoryError, ValueError) as e: return {'tag': tag, 'sigma': sig, 'error': str(e)}
    d['tag'] = tag; return d


if __name__ == '__main__':
    S = states() + new_states(); print(len(S), 'states', flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as pool:
        out = pool.map(run, S, chunksize=4)
    json.dump(out, open('results/l1_ratios.json', 'w'), indent=1, default=float)
    ok = [o for o in out if 'error' not in o]; small = [o for o in ok if o['sigma'] <= 1e-2]
    print('errors', len(out) - len(ok))
    for k in ['rho', 'r_sup', 'r_sigma', 'nu']:
        v = max(small, key=lambda o: o[k]); print(f'max {k}: {v[k]:.4f} ({v["tag"]}, sigma {v["sigma"]:g}, gap {v["gap_over_sigma"]:.4f})')
    print('min geo', min(o['geo'] for o in small), ' min gap', min(o['gap_over_sigma'] for o in small))
