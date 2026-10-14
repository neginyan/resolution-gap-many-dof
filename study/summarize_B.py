"""Collect every number quoted in the stage-B summary -> results/summary_B.json"""
import json, numpy as np, collections

out = {}
# ---------------- measurement 1
S = json.load(open('results/m1_slices.json')); M = json.load(open('results/m1_maps.json')); ALL = S + M
g = collections.defaultdict(list)
for d in S: g[d['tag']].append(d)
for v in g.values(): v.sort(key=lambda d: d['params']['delta'])
r = lambda d: d['R'] / d['lam_comp']
out['m1_concentric_ratio'] = {lab: r(g[t][0]) for lab, t in [('1', 'ratio_1.0'), ('2', 'ratio_2.0'), ('4.4', 'base_sepW'), ('8', 'ratio_8.0')]}
out['m1_concentric_tilt'] = {lab: r(g[t][0]) for lab, t in [('0', 'base_sepW'), ('22.5', 'tilt_22.5'), ('45', 'tilt_45.0'), ('90', 'tilt_90.0')]}
w = g['base_sepW']; v = g['base_sepV']
out['m1_sepW_min_ratio'] = min(r(d) for d in w)
first_below = next(d for d in v if r(d) < 1)
out['m1_sepV_first_below'] = {'delta_over_sigma': first_below['params']['delta'] / 0.1, 'ratio': r(first_below), 'overlap': first_below['overlap']}
out['m1_sepV_last_above'] = max(d['params']['delta'] / 0.1 for d in v if r(d) > 1)
ex = [d for d in ALL if d['R'] > d['lam_comp']]
out['m1_n_states'] = len(ALL); out['m1_n_exceed'] = len(ex)
out['m1_exceed_min_overlap'] = min(d['overlap'] for d in ex)
out['m1_exceed_min_cov'] = min(d['f'] - d['law_term'] for d in ex)
out['m1_exceed_all_switch_negative'] = all(d['R_switch'] < 0 for d in ex)
out['m1_exceed_A_range'] = [min(d['A'] for d in ex), max(d['A'] for d in ex)]
out['m1_max_ratio'] = max(r(d) for d in ALL)
out['m1_law_minus_lam_rho_max'] = max(d['law_term'] - d['lam_rho'] for d in ALL)
out['m1_max_abs_R_minus_f'] = max(abs(d['R'] - d['f']) for d in ALL)
out['m1_max_abs_R_minus_e'] = max(abs(d['R'] - d['e']) for d in ALL)
ma = [d for d in M if d['tag'] == 'map_ratio' and d['R'] > d['lam_comp']]
out['m1_map_ratio_min_ratio_exceed'] = min(d['params']['ratio'] for d in ma)
me = [d for d in M if d['tag'] == 'map_elong' and d['R'] > d['lam_comp']]
out['m1_map_elong_min_elongation_exceed'] = min(d['params'].get('l1', 1.9) / 0.09 for d in me)
out['m1_map_elong_min_overlap_exceed'] = min(d['overlap'] for d in me)
out['m1_identity_errors'] = {
    'R = A R_sep + R_switch': max(abs(d['A'] * d['R_sep'] + d['R_switch'] - d['R']) for d in ALL),
    'R = f + D': max(abs(d['f'] + d['D'] - d['R']) for d in ALL),
    'R_sep (grid) = R_sep (closed form)': max(abs(d['R_sep'] - d['R_sep_ref']) for d in ALL),
    'Tweedie Cov(X|y) = s2 I - s4 H': max(d['tweedie_err'] for d in ALL),
    'law term = mean part of f': max(abs(d['law_term'] - d['f_mean_part']) for d in ALL)}
cx = g['base_sepW'][0]
out['m1_base_concentric'] = {k: cx[k] for k in ['R', 'lam_comp', 'R_sep', 'A', 'R_switch', 'law_term', 'f', 'D', 'e', 'overlap', 'lam_rho']}
out['m1_base_concentric']['cov'] = cx['f'] - cx['law_term']
r1 = g['ratio_1.0'][0]; out['m1_ratio1_concentric_cov'] = r1['f'] - r1['law_term']

# ---------------- measurement 2: single Gaussians
SG = json.load(open('results/m2_single.json')); SW = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
Sm = {(rec['shape'], rec['q0']): np.array(rec['S']) for rec in SW}
cf, ratio, viol = 0, 0, 0
for d in SG:
    Sx = Sm[(d['shape'], d['q0'])]; T = Sx + d['sigma'] ** 2 * np.eye(2); K = Sx @ np.linalg.inv(T); trF = np.trace(np.linalg.inv(T))
    pred = -np.cos(d['q0']) * np.exp(-Sx[0, 0] / 2) * K[0, 0] * K[0, 1] / trF
    cf = max(cf, abs(d['R'] - d['e'] - pred))
    b = d['sigma'] ** 2 * abs(np.cos(d['q0'])) * np.exp(-Sx[0, 0] / 2) / 2
    if b > 1e-12: ratio = max(ratio, abs(d['R'] - d['e']) / b)
    viol = max(viol, abs(d['R'] - d['e']) - b)
nz = [d for d in SG if abs(d['R']) > 1e-9]
sign_ok = sum(np.sign(d['R'] - d['e']) == np.sign(d['R'] * np.cos(d['q0'])) for d in nz if abs(np.cos(d['q0'])) > 1e-6)
out['m2_single'] = {'n': len(SG), 'closed_form_max_err': cf, 'bound_max_ratio': ratio, 'bound_max_violation': viol,
                    'n_nonzero': len(nz), 'n_cos_nonzero': sum(abs(np.cos(d['q0'])) > 1e-6 for d in nz), 'sign_rule_ok': int(sign_ok),
                    'max_abs_rem_at_cos0': max(abs(d['R'] - d['e']) for d in SG if abs(np.cos(d['q0'])) < 1e-6),
                    'max_abs_rem': max(abs(d['R'] - d['e']) for d in SG)}

# ---------------- measurement 2: mixtures
MX = json.load(open('results/m2_mix.json'))
R = np.array([d['R'] for d in MX]); e = np.array([d['e'] for d in MX]); f = np.array([d['f'] for d in MX]); D = np.array([d['D'] for d in MX])
sig = np.array([d['sigma'] for d in MX]); ov = np.array([d['overlap'] for d in MX]); law = np.array([d['law_term'] for d in MX])
curv = np.array([d['sigma'] ** 2 * d['K_cos'] for d in MX]); curv_sin = np.array([d['sigma'] ** 2 * d['K_sin'] for d in MX])
npd = np.array([d['H_nonPD_mass'] for d in MX]); lrs = np.array([d['H_logratio_std'] for d in MX]); als = np.array([d['H_alpha_std'] for d in MX])
k = ov >= 0.05
def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
out['m2_mix'] = {
    'n': len(MX), 'n_overlapping': int(k.sum()),
    'identity_R_eq_f_plus_D_max_err': float(np.abs(R - f - D).max()),
    'mass_min': min(d['mass'] for d in MX),
    'median_abs_R_minus_e_overlapping': float(np.median(np.abs(R - e)[k])), 'median_abs_R_minus_f_overlapping': float(np.median(np.abs(R - f)[k])),
    'max_abs_R_minus_e': float(np.abs(R - e).max()), 'max_abs_R_minus_f': float(np.abs(R - f).max()),
    'n_f_closer_than_e': int((np.abs(R - f) < np.abs(R - e)).sum()),
    'fe_over_curv_max': float(np.max(np.abs(f - e) / curv)), 'fe_over_curv_median': float(np.median(np.abs(f - e) / curv)),
    'fe_over_curvsin_max': float(np.max(np.abs(f - e) / curv_sin)),
    'spearman_absfe_curv': spear(np.abs(f - e), curv), 'spearman_absfe_curvsin': spear(np.abs(f - e), curv_sin),
    'spearman_absD_nonPD': spear(np.abs(D), npd), 'spearman_absD_logratio_std': spear(np.abs(D), lrs), 'spearman_absD_alpha_std': spear(np.abs(D), als),
    'spearman_absD_overlap': spear(np.abs(D), ov),
    'D_when_H_PD_everywhere_max': float(np.abs(D)[npd < 1e-6].max()), 'n_H_PD_everywhere': int((npd < 1e-6).sum()),
    'D_when_nonPD_gt_0.05_median': float(np.median(np.abs(D)[npd > 0.05])),
    'D_by_sigma_max': {str(s_): float(np.abs(D)[sig == s_].max()) for s_ in [0.03, 0.1, 0.3, 1.0]},
    'n_D_pos': int((D > 1e-12).sum()), 'n_D_neg': int((D < -1e-12).sum()),
    'law_le_lam_rho_all': bool(np.all(law <= np.array([d['lam_rho'] for d in MX]) + 1e-12)),
    'n_R_gt_lam_rho': int((R > np.array([d['lam_rho'] for d in MX]) + 1e-9).sum()),
    'max_f': float(f.max()), 'max_R': float(R.max()),
    'fe_over_curv_max_nonoverlap': float(np.max((np.abs(f - e) / curv)[ov < 1e-3])),
    'median_abs_fe_overlapping': float(np.median(np.abs(f - e)[k])), 'median_abs_D_overlapping': float(np.median(np.abs(D)[k])),
    'D_nonoverlap_median': float(np.median(np.abs(D)[ov < 1e-3])), 'D_nonoverlap_max': float(np.abs(D)[ov < 1e-3].max()),
    'spearman_abscov_alpha_std': spear(np.abs(f - law), als), 'spearman_abscov_logratio_std': spear(np.abs(f - law), lrs)}
gi = collections.defaultdict(dict)
for d in MX: gi[d['i']][d['sigma']] = d['D']
sl = [np.log(abs(v[0.1] / v[0.03])) / np.log(0.1 / 0.03) for v in gi.values() if abs(v[0.03]) > 1e-10 and abs(v[0.1]) > 1e-10]
out['m2_mix']['D_slope_003_01'] = {'n': len(sl), 'p10': float(np.percentile(sl, 10)), 'median': float(np.median(sl)), 'p90': float(np.percentile(sl, 90))}
json.dump(out, open('results/summary_B.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
