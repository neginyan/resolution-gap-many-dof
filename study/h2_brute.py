"""Stage H: direct-summation check (verification/indep_eval.R_brute, no closed forms) of the sigma = 0.1 and 0.3 violations of
R <= R*(sigma) found by h2_adversarial.py, at 121 and 161 y-grid points per axis.  (K = 3, sigma = 0.3 is skipped: its wide
components need too many prior nodes for this check.)  Writes results/h2_brute.json."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verification'))
from indep_eval import R_brute

if __name__ == '__main__':
    out = []
    for r in json.load(open('results/h2_adversarial.json')):
        if r['sigma'] in (0.1, 0.3) and not (r['K'] == 3 and r['sigma'] == 0.3):
            ws, mus, Ss = r['ws'], [np.array(m) for m in r['mus']], [np.array(S) for S in r['Ss']]
            for ny in [121, 161]:
                Rb = R_brute(ws, mus, Ss, r['sigma'], ny=ny)
                out.append({'K': r['K'], 'sigma': r['sigma'], 'ny': ny, 'brute_minus_stored': Rb - r['R'], 'brute_minus_Rstar': Rb - r['Rstar'],
                            'stored_minus_Rstar': r['R'] - r['Rstar']}); print(out[-1], flush=True)
    json.dump(out, open('results/h2_brute.json', 'w'), indent=1)
