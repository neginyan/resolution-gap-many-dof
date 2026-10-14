"""Collect every number quoted in the stage-A summary from the result files -> results/summary.json"""
import json, numpy as np, glob

def law(S, q0, sig):
    S = np.array(S); ec = np.cos(q0) * np.exp(-S[0, 0] / 2)
    SJ = np.array([[0, (1 - ec) / 2], [(1 - ec) / 2, 0]]); F = np.linalg.inv(S + sig ** 2 * np.eye(2))
    return float(np.trace(F @ SJ) / np.trace(F)), float(np.linalg.eigvalsh(SJ).max())

D = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
out = {}
diffs, c_diffs, R_le_lam, lam_le_1 = [], [], [], []
for rec in D:
    for r in rec['rows']:
        L, lam = law(rec['S'], rec['q0'], r['sigma']); diffs.append(abs(r['R'] - L)); R_le_lam.append(r['R'] <= lam + 1e-12); lam_le_1.append(lam <= 1)
    c_diffs.append(abs(rec['c'] - law(rec['S'], rec['q0'], 0.0)[0]))
out['single'] = {'n_cases': len(diffs), 'n_states': len(D), 'max_abs_R_minus_law': max(diffs), 'max_abs_c_minus_law0': max(c_diffs),
                 'all_R_le_lam_Jbar': all(R_le_lam), 'all_lam_Jbar_le_1': all(lam_le_1),
                 'max_R': max(r['R'] for rec in D for r in rec['rows'])}
sym = [abs(r['R']) for rec in D if rec['shape'].startswith(('isotropic', 'elongated')) for r in rec['rows']]
out['symmetric_states_max_abs_R'] = max(sym)
st = {rec['q0_over_pi']: rec for rec in D if rec['shape'].startswith('sharp along stretching')}
co = {rec['q0_over_pi']: rec for rec in D if rec['shape'].startswith('sharp along contracting')}
out['stretch_contract_antisymmetry_max'] = max(abs(a['R'] + b['R']) for q in st for a, b in zip(st[q]['rows'], co[q]['rows']))
s1, s2 = 0.05 ** 2, 0.5 ** 2
out['collapse_max_diff'] = max(abs(r['R'] / rec['c'] - (s1 + s2) / (s1 + s2 + 2 * r['sigma'] ** 2))
                               for rec in st.values() for r in rec['rows'])
# Theorem 9: (R - c)/sigma^2 at the two smallest sigmas (should be roughly constant -> error O(sigma^2))
t9 = []
for rec in st.values():
    r0, r1 = rec['rows'][0], rec['rows'][1]
    t9.append(((r0['R'] - rec['c']) / r0['sigma'] ** 2, (r1['R'] - rec['c']) / r1['sigma'] ** 2))
out['theorem9_ratio_of_scaled_errors'] = [float(a / b) for a, b in t9 if abs(b) > 0]
out['theorem9_R001_over_c'] = [float(rec['rows'][0]['R'] / rec['c']) for rec in st.values()]
pi_rec = st[1.0]; z_rec = st[0.0]
idx = {round(r['sigma'], 3): r for r in pi_rec['rows']}
out['example_pi'] = {'c': pi_rec['c'], 'b': pi_rec['b'],
                     'R': {str(k): idx[k]['R'] for k in [0.01, 0.342, 2.0]}, 'd': {str(k): idx[k]['d'] for k in [0.01, 0.342, 2.0]},
                     'e_rel_dev_sigma2': (idx[2.0]['cand_e'] - idx[2.0]['R']) / idx[2.0]['R']}
zi = {round(r['sigma'], 3): r for r in z_rec['rows']}
out['example_q0'] = {'c': z_rec['c'], 'R_sigma2': zi[2.0]['R'], 'e_sigma2': zi[2.0]['cand_e'], 'd_sigma2': zi[2.0]['d']}
M = json.load(open('results/mixtures.json'))
R = np.array([m['R'] for m in M]); lr = np.array([m['lam_rho'] for m in M]); lc = np.array([m['lam_comp'] for m in M])
Rs = np.array([m['R_sep'] for m in M]); ov = np.array([m['overlap'] for m in M])
bins = {}
for lo, hi in [(0, 1e-3), (1e-3, 0.05), (0.05, 0.3), (0.3, 1.01)]:
    k = (ov >= lo) & (ov < hi)
    bins[f'{lo}-{hi}'] = {'n': int(k.sum()), 'max': float(np.abs(R - Rs)[k].max()), 'median': float(np.median(np.abs(R - Rs)[k]))}
out['mixtures_random'] = {'n': len(M), 'max_R': float(R.max()), 'all_R_le_1': bool((R <= 1).all()),
                          'n_R_gt_lam_rho': int((R > lr + 1e-9).sum()), 'n_R_gt_lam_comp': int((R > lc + 1e-9).sum()), 'dev_by_overlap': bins}
adv = {}
for f in sorted(glob.glob('results/adversarial_*.json')):
    for k, v in json.load(open(f)).items():
        adv[k] = {'best': v['best_margin'], 'R': v['info']['R'], 'lam_comp': v['info']['lam_comp'], 'lam_rho': v['info']['lam_rho'],
                  'R_sep': v['info']['R_sep'], 'overlap': v['info']['overlap'], 'R_n1.5': v['resolution_check']['R_n1.5']}
out['adversarial'] = adv
ME = json.load(open('results/mixtures_e.json'))
Re = np.array([o['R'] for o in ME['sample']]); ee = np.array([o['e'] for o in ME['sample']]); Rse = np.array([o['R_sep'] for o in ME['sample']])
ove = np.array([o['overlap'] for o in ME['sample']]); kk = ove >= 0.05
out['mixtures_e'] = {'n': len(Re), 'n_overlapping': int(kk.sum()), 'median_abs_R_minus_e_overlapping': float(np.median(np.abs(Re - ee)[kk])),
                     'max_abs_R_minus_e': float(np.abs(Re - ee).max()), 'median_abs_R_minus_Rsep_overlapping': float(np.median(np.abs(Re - Rse)[kk])),
                     'n_R_gt_e': int((Re > ee + 1e-12).sum()), 'max_e': float(ee.max()), 'counterexample_e': ME['counterexample']['e']}
out['max_R_everywhere'] = float(max(out['single']['max_R'], R.max(), max(v['R'] for v in adv.values())))
json.dump(out, open('results/summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:3000])
