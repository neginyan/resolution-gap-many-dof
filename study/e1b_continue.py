"""Stage E, task 1 (continued): the e1_local chains were still improving after 250 steps, so the three best states are
continued for 800 more steps each (step 0.03 adapted every 50 steps), same objective and constraints (distance 0.1-0.3).
Writes results/e1b_continue.json."""
import numpy as np, json
from multiprocessing import Pool
from e1_local import chain, pack

if __name__ == '__main__':
    E = sorted(json.load(open('results/e1_local.json')), key=lambda r: -r['ratio'])[:3]
    jobs = [(f"continue_{r['seed']}_K{r['K']}_step{r['step0']}", pack(r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']], r['sigma']),
             len(r['ws']), 0.03, 800, 777 + i) for i, r in enumerate(E)]
    with Pool(2) as pool:
        out = pool.map(chain, jobs, chunksize=1)
    json.dump(out, open('results/e1b_continue.json', 'w'), indent=1, default=float)
    for r in out:
        print(f"{r['seed']}: start {r['start_value']:.4f} -> best {r['best_value']:.4f}; ratio {r['ratio']:.4f} dist {r['dist']:.3f} R {r['R']:.4f} "
              f"sigma {r['sigma']:.4g} valley {r['H_nonPD_mass']:.3f} Pv {r['Pv']:.4f} Pw {r['Pw']:.4f} (sd20 diff {r['R_per_sd20'] - r['R']:.1e})")
