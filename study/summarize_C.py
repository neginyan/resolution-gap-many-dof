"""Collect every number quoted in the stage-C summary -> results/summary_C.json"""
import json, numpy as np, collections
from obs_eval2 import FLOWS


def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


out = {}
# ---------------- C1: adversarial search
T = json.load(open('results/c1_table.json'))
best = max(T, key=lambda r: r['R'])
out['c1'] = {'n_states': len(T), 'max_R_over_kappa': best['R'], 'best': best,
             'max_room': max(r['room'] for r in T), 'room_best': max(T, key=lambda r: r['room']),
             'max_excess': max(r['R'] - r['law_term'] for r in T), 'max_law_plus_excess': max(r['R'] for r in T),
             'all_resolution_ok': all(abs(r['R'] - r['R_n1.5']) < 1e-8 for r in T),
             'max_fieldgrid_dev': max(abs(r['R_fieldgrid_minus_full']) for r in T),
             'cov_in_valley_by_sigma': {str(s): [min(r['cov_in_valley'] for r in T if r['sigma'] == s), max(r['cov_in_valley'] for r in T if r['sigma'] == s)]
                                        for s in [0.03, 0.1, 0.3, 1.0]},
             'stageA_single_max_R': json.load(open('results/summary.json'))['single']['max_R']}
out['c1']['table'] = sorted([{k: r[k] for k in ['K', 'sigma', 'target', 'R', 'law_term', 'cov', 'D', 'H_nonPD_mass', 'overlap_max', 'room',
                                                'cov_in_valley', 'D_in_valley']} for r in T], key=lambda r: -r['R'])
# ---------------- C3: spatial
out['c3'] = json.load(open('results/c3_spatial.json'))
# ---------------- C2: D
D = json.load(open('results/c2_D.json'))
d = np.array([x['D'] for x in D]); dw = np.array([x['D_within'] for x in D]); db = np.array([x['D_between'] for x in D])
SD = np.array([x['S_D'] for x in D]); rv = np.array([x['resp_var'] for x in D]); gap = np.array([x['stretch_gap'] for x in D])
sig = np.array([x['sigma'] for x in D]); k = np.abs(d) > 1e-8
out['c2'] = {'n': len(D), 'n_tags': dict(collections.Counter(x['tag'] for x in D)), 'n_nonzero': int(k.sum()),
             'split_max_err': float(np.abs(d - dw - db).max()),
             'spearman_absD_SD': spear(np.abs(d[k]), SD[k]), 'spearman_absD_rvgap': spear(np.abs(d[k]), (rv * gap)[k]),
             'spearman_absD_rv': spear(np.abs(d[k]), rv[k]),
             'ratio_D_SD_max_by_sigma': {str(s): float((np.abs(d) / SD)[k & (sig == s)].max()) for s in [0.03, 0.1, 0.3, 1.0]},
             'ratio_D_SD_median': float(np.median((np.abs(d) / SD)[k])),
             'ratio_D_rvgap_max': float((np.abs(d) / (rv * gap))[k & (rv * gap > 0)].max()),
             'within_share_median': float(np.median((np.abs(dw) / (np.abs(dw) + np.abs(db)))[k])),
             'sign_D_eq_sign_between': float(np.mean(np.sign(d[k]) == np.sign(db[k]))),
             'n_ratio_gt_1': int(((np.abs(d) / SD) > 1)[k].sum()),
             'failures_between_dominated': bool(all(abs(x['D_between']) > abs(x['D_within']) for x in D if abs(x['D']) > 1e-8 and abs(x['D']) / x['S_D'] > 1))}
# ---------------- C4: other flows
S = json.load(open('results/c4_single.json')); wl = wr = 0
for x in S:
    mu = np.array(x['mus'][0]); Sx = np.array(x['Ss'][0]); s_ = x['sigma']
    F = np.linalg.inv(Sx + s_ ** 2 * np.eye(2)); K = Sx @ F
    a = (1 + FLOWS[x['kind']][1](mu[0], Sx[0, 0])) / 2
    wl = max(wl, abs(a * 2 * F[0, 1] / np.trace(F) - x['R'])); wr = max(wr, abs(x['R'] - x['e'] - 6.0 * K[0, 0] * K[0, 1] / np.trace(F)))
Fm = json.load(open('results/c4_family.json')); M = json.load(open('results/c4_mix.json'))
pend = {lab: d_ for lab, d_ in [('1', None)]}
m1 = collections.defaultdict(list)
for x in json.load(open('results/m1_slices.json')): m1[x['tag']].append(x)
pend_cov = {str(round(r, 2)): min(m1[t], key=lambda x: x['params']['delta'])['f'] - min(m1[t], key=lambda x: x['params']['delta'])['law_term']
            for r, t in [(1.0, 'ratio_1.0'), (2.0, 'ratio_2.0'), (0.4 / 0.09, 'base_sepW'), (8.0, 'ratio_8.0')]}
fam = [{'kind': x['kind'], 'ratio': x['ratio'], 'l1': x['l1'], 'R': x['R'], 'lam_comp': x['lam_comp'], 'law': x['law_term'],
        'cov': x['f'] - x['law_term'], 'D': x['D'], 'valley': x['H_nonPD_mass']} for x in Fm]
Rm = np.array([x['R'] for x in M]); lc = np.array([x['lam_comp'] for x in M]); lr = np.array([x['lam_rho_direct'] for x in M])
law = np.array([x['law_term'] for x in M]); cov = np.array([x['f'] - x['law_term'] for x in M]); Dm = np.array([x['D'] for x in M])
ex = Rm > lc + 1e-9
out['c4'] = {'single_n': len(S), 'single_law_max_err': wl, 'single_remainder_formula_max_err': wr,
             'family': fam, 'pendulum_family_cov_l19': pend_cov,
             'quartic_eq_doublewell_cov_maxdiff': max(abs(a['cov'] - b['cov']) + abs(a['D'] - b['D']) for a in fam for b in fam
                                                      if a['kind'] == 'quartic' and b['kind'] == 'double_well' and a['ratio'] == b['ratio'] and a['l1'] == b['l1']),
             'mix_n': len(M), 'mix_identity_err': max(max(abs(x['A'] * x['R_sep'] + x['R_switch'] - x['R']), abs(x['f'] + x['D'] - x['R'])) for x in M),
             'mix_tweedie_err': max(x['tweedie_err'] for x in M), 'mix_mass_min': min(x['mass'] for x in M),
             'mix_law_le_lam_rho': bool(np.all(law <= lr + 1e-12)),
             'mix_n_R_gt_lam_rho': int((Rm > lr + 1e-9).sum()), 'mix_n_R_gt_lam_comp': int(ex.sum()),
             'mix_exceed_cov_positive': bool(np.all(cov[ex] > 0)) if ex.any() else None,
             'mix_exceed_max_ratio': float((Rm / lc)[ex].max()) if ex.any() else None,
             'mix_by_kind': {kd: {'n_R_gt_lam_comp': int((ex & np.array([x['kind'] == kd for x in M])).sum()),
                                  'max_R_over_lam_comp': float(max(x['R'] / x['lam_comp'] for x in M if x['kind'] == kd))} for kd in ['quartic', 'double_well']},
             'mix_median_abs_R_f_vs_e_overlap': [float(np.median(np.abs(Rm - np.array([x['f'] for x in M]))[np.abs(Dm) > 1e-6])),
                                                 float(np.median(np.abs(Rm - np.array([x['e'] for x in M]))[np.abs(Dm) > 1e-6]))]}
json.dump(out, open('results/summary_C.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'c1'}, indent=1, default=float)[:6000])
print(json.dumps({k: v for k, v in out['c1'].items() if k not in ['table', 'best', 'room_best']}, indent=1, default=float))
