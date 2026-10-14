"""Stage M analysis: results/summary_M.json and figures stageM_formula / stageM_free / stageM_mix / stageM_dichotomy.
Outputs of fits, extrapolations and optimisations carry 'opt', 'extrap', 'fit', 'coef' in their JSON paths (compared `loose`);
rank correlations ('spearman') use scipy.stats.rankdata after rounding to relative 1e-10."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
J = lambda f: json.load(open(f'results/{f}.json'))
have = lambda f: os.path.exists(f'results/{f}.json')
out = {}


def rk(x):
    x = np.asarray(x, float); e = 10.0 ** np.floor(np.log10(np.abs(x) + 1e-300)); return rankdata(np.round(x / e, 10) * e)


def spearman(a, b):
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


# ---------------- M1 ----------------
FR = {o['eps']: o for o in J('m1_frame')}
M1 = {'frame_opt': [{k: o[k] for k in ['eps', 'kappa', 'top', 'formula', 'opt', 'extrap_sigma0', 'opt_rel_diff_extrap_vs_formula']} for o in J('m1_frame')], 'free': {}}
for e in [0.1, 0.3, 1.0]:
    d = J(f'm1_free_eps{e:g}'); ref = FR[e]['opt']
    g5 = np.array([r['gap']['1e-05'] for r in d['runs']]); g4 = np.array([r['gap']['0.0001'] for r in d['runs']])
    reach = np.abs(g5 - ref['1e-05']) < 1e-4 * ref['1e-05']
    near_top = [bool(np.allclose(np.abs(np.cos(np.array(r['mq']))), 1, atol=1e-3)) for r in d['runs']]
    M1['free'][f'{e:g}'] = {'n_starts': len(g5), 'n_reach_opt': int(reach.sum()), 'n_below_opt': int(np.sum(g5 < ref['1e-05'] * (1 - 1e-7))),
                            'opt_min_free_1e-5': float(g5.min()), 'opt_min_free_1e-4': float(g4.min()), 'frame_opt_1e-5': ref['1e-05'], 'frame_opt_1e-4': ref['0.0001'],
                            'others_1e-5_sorted': sorted(float(v) for v in g5[~reach])[:20], 'n_stuck_O1': int(np.sum(g5 > 100)),
                            'n_mean_at_a_top': int(sum(near_top))}
if have('m1_range'):
    M1['range'] = [{'eps': o['eps'], 'kappa': o['kappa'], 'tops': o['tops'], 'lambda_at_candidates': o['lambda_at_candidates'], 'formula': o['formula'],
                    'opt_from_ansatz': o['from_ansatz'], 'opt_random_1e-5': [r['vals']['1e-05'] for r in o['random']]} for o in J('m1_range')]
if have('m1_hessian'):
    H = []
    for o in J('m1_hessian'):
        h = o['h0.003']; ev = np.array(h['eig_reduced'])
        H.append({'eps': o['eps'], 'sigma': o['sigma'], 'opt_gap': o['gap'], 'grad_max': h['grad_free_max'], 'eig_min': float(ev.min()), 'eig_max': float(ev.max()),
                  'eig_min_h1e-3': float(min(o['h0.001']['eig_reduced'])), 'n_soft': len(h['soft_modes']),
                  'soft_modes': [{'eig': m['eig'], 'vector': m['vector'], 'scan_min_minus_f0': m['scan_min_minus_f0']} for m in h['soft_modes']],
                  'scan_min_rel': float(min(m['scan_min_minus_f0'] for m in h['soft_modes']) / o['gap'])})
    M1['hessian'] = H
out['M1'] = M1

# ---------------- M2 ----------------
P = J('m2_prediction'); M2 = {'prediction_written_utc': P['written_utc'], 'prediction_sha256': open('logs/m2_prediction_hash.txt').read().split('=')[-1].strip().split()[0], 'systems': []}
if have('m2_opt'):
    for o in sorted(J('m2_opt'), key=lambda o: (o['kind'], o['n'], o['eps'])):
        M2['systems'].append({k: o[k] for k in ['kind', 'n', 'eps', 'name', 'kappa', 'prediction', 'frame_opt', 'extrap_sigma0', 'opt_rel_diff_extrap_vs_prediction',
                                                'random_below_frame_opt_1e-5', 'random_reaching_1e-4_rel', 'shape_1e-5']} | {'opt_random_1e-5': [r['vals']['1e-05'] for r in o['random']]})
out['M2'] = M2

# ---------------- M3 ----------------
if have('m3_mix'):
    out['M3'] = [{k: o.get(k) for k in ['kind', 'eps', 'K', 'gap_start', 'gap', 'gap_per_sd7', 'gap_gauss', 'ratio']} for o in sorted(J('m3_mix'), key=lambda o: (o['kind'], o['eps'] or 0, o['K']))]

# ---------------- M4 ----------------
if have('m4_dichotomy'):
    D = [o for o in J('m4_dichotomy') if 'error' not in o and o['sigma'] <= 1e-2]
    st = {}
    for t in [0.2, 0.25, 0.3, 0.5]:
        v = [o for o in D if o['rho'] > t]
        st[f'rho>{t}'] = {'n': len(v), 'min_d': min((o['d'] for o in v), default=None), 'min_gap': min((o['gap_over_sigma'] for o in v), default=None),
                          'min_law_def': min((o['law_def'] for o in v), default=None), 'min_Zbound': min((o['Zbound_over_sigma'] for o in v), default=None)}
    near = [o for o in D if o['gap_over_sigma'] <= 1.2]
    out['M4'] = {'n': len(D), 'by_rho': st, 'spearman_rho_d_near': spearman([o['rho'] for o in near], [o['d'] for o in near]),
                 'spearman_rho_lawdef_near': spearman([o['rho'] for o in near], [o['law_def'] for o in near]),
                 'min_Zbound_all': min(o['Zbound_over_sigma'] for o in D), 'Zbound_le_gap_all': all(o['Zbound_over_sigma'] <= o['gap_over_sigma'] + 1e-9 for o in D),
                 'd_range_near': [min(o['d'] for o in near), max(o['d'] for o in near)],
                 'P_min': min(o['P'] for o in D), 'P_argmin_tag': min(D, key=lambda o: o['P'])['tag'], 'sqrt_P_min': float(np.sqrt(min(o['P'] for o in D))),
                 'P_ge_ThetaP2_all': all(o['P'] >= o['ThetaP'] ** 2 * (1 - 1e-9) for o in D), 'P_over_ThetaP2_min': min(o['P'] / o['ThetaP'] ** 2 for o in D),
                 'two_variable_over_gap': [min(o['two_variable_form'] / o['gap_over_sigma'] for o in D), float(np.median([o['two_variable_form'] / o['gap_over_sigma'] for o in D]))],
                 'Zbound_over_gap': [min(o['Zbound_over_sigma'] / o['gap_over_sigma'] for o in D), float(np.median([o['Zbound_over_sigma'] / o['gap_over_sigma'] for o in D]))],
                 'spearman_P_gap_near': spearman([o['P'] for o in near], [o['gap_over_sigma'] for o in near]),
                 'worst_P_states': [{k: o[k] for k in ['tag', 'sigma', 'P', 'gap_over_sigma', 'ThetaP', 'rho', 'd', 'y_over_sigma']} for o in sorted(D, key=lambda o: o['P'])[:5]]}
if have('m5_arc'):
    A5 = J('m5_arc'); out['M5'] = {'opt_n128': A5['by_n']['128']['opt_search'], 'same_shape_n256': A5['by_n']['128']['gap_n256'], 'curvature': A5['by_n']['128']['curvature'],
                                   'fixed_piece_length': A5.get('fixed_piece_length')}
json.dump(out, open('results/summary_M.json', 'w'), indent=1, default=float)

# ================= figures =================
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
a = ax[0]; pts = [(o['formula'], o['extrap_sigma0'], 'coupled, ε = %g' % o['eps']) for o in J('m1_frame')]
pts2 = [(o['prediction'], o['extrap_sigma0'], o['name']) for o in J('m2_opt')] if have('m2_opt') else []
a.scatter([p[0] for p in pts], [p[1] for p in pts], color=C[0], s=30, label='M1: coupled pendulums (ε = −0.3 … 3)')
a.scatter([p[0] for p in pts2], [p[1] for p in pts2], color=C[1], s=40, marker='D', label='M2: predicted before optimising')
x = np.linspace(1, 8, 10); a.plot(x, x, '-', color=INK, lw=1)
a.set_xlabel('formula 2 tr[(B^½ C B^½)^½]'); a.set_ylabel('optimum, extrapolated to σ → 0'); a.legend(fontsize=7, frameon=False); a.set_title('the formula against the full optimisation', fontsize=9)
a = ax[1]; allp = [(o['formula'], o['opt_rel_diff_extrap_vs_formula']) for o in J('m1_frame')] + ([(o['prediction'], o['opt_rel_diff_extrap_vs_prediction']) for o in J('m2_opt')] if have('m2_opt') else [])
a.scatter([p[0] for p in allp], [max(abs(p[1]), 1e-12) for p in allp], color=INK, s=20); a.set_yscale('log'); a.set_ylim(1e-12, 1e-4)
a.set_xlabel('coefficient'); a.set_ylabel('|relative difference|'); a.set_title('agreement (limited by the optimiser, ~1e-7)', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageM_formula.png', dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, e in zip(ax, [0.1, 0.3, 1.0]):
    d = J(f'm1_free_eps{e:g}'); ref = FR[e]['opt']['1e-05']; g5 = np.array([r['gap']['1e-05'] for r in d['runs']])
    rel = (g5 - ref) / ref; a.hist(np.log10(np.maximum(rel, 1e-9)), bins=40, color=C[0])
    a.axvline(np.log10(1e-4), color=MUTED, ls=':', lw=1)
    a.set_xlabel('log10 [(value − optimum)/optimum]  at σ = 1e-5'); a.set_title(f'ε = {e:g}: 60 random starts (none below the optimum)', fontsize=8.5)
plt.tight_layout(); plt.savefig('figures/stageM_free.png', dpi=150); plt.close()

if 'M3' in out:
    fig, a = plt.subplots(figsize=(6, 3.8))
    for i, (kd, e, lab) in enumerate([('1d', None, '1 degree of freedom (pendulum)'), ('3d', 0.1, 'coupled, ε = 0.1'), ('3d', 0.3, 'coupled, ε = 0.3')]):
        v = sorted([o for o in out['M3'] if o['kind'] == kd and o['eps'] == e], key=lambda o: o['K'])
        if v: a.plot([o['K'] for o in v], [o['ratio'] for o in v], 'o-', color=C[i], label=lab)
    a.axhline(1, color=MUTED, lw=1); a.set_xscale('log', base=2); a.set_xlabel('K'); a.set_ylabel('(κ − R)/(κ − R*_Gauss)'); a.legend(fontsize=7, frameon=False)
    a.set_title('M3: advantage of mixtures over the best Gaussian (σ = 1e-3)', fontsize=9); plt.tight_layout(); plt.savefig('figures/stageM_mix.png', dpi=150); plt.close()

if 'M4' in out:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); D = [o for o in J('m4_dichotomy') if 'error' not in o and o['sigma'] <= 1e-2]
    a = ax[0]; v = [o for o in D if o['gap_over_sigma'] < 50]
    sc = a.scatter([o['d'] for o in v], [o['y_over_sigma'] for o in v], c=[min(o['gap_over_sigma'], 3) for o in v], s=10, cmap='viridis')
    dd = np.logspace(-0.5, 1.5, 100)
    for P0, ls in [(Sm := min(o['P'] for o in D), '-'), (1.0, ':')]: a.plot(dd, P0 / dd ** 2, ls, color=INK, lw=1, label=f'd² y/σ = {P0:.3f}')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('d = rms distance of the focus from the top (units √σ)'); a.set_ylabel('y/σ = σ J_ww / j')
    plt.colorbar(sc, ax=a, label='(1 − R)/σ (capped at 3)'); a.legend(fontsize=7, frameon=False); a.set_title('focus spread × W-sharpness: all states lie above P = 0.858', fontsize=9)
    a = ax[1]; a.scatter([o['gap_over_sigma'] for o in D], [max(o['Zbound_over_sigma'], 1e-3) for o in D], s=10, color=C[1], label='Z-bound (no Cramér–Rao)')
    a.scatter([o['gap_over_sigma'] for o in D], [max(o['bound_over_sigma'], 1e-3) for o in D], s=10, color=C[0], label="Θ′ bound (paper B)")
    xx = np.array([0.8, 2e3]); a.plot(xx, xx, '-', color=INK, lw=1); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('(1 − R)/σ'); a.set_ylabel('bound / σ')
    a.legend(fontsize=7, frameon=False); a.set_title('the two halves of the dichotomy', fontsize=9)
    plt.tight_layout(); plt.savefig('figures/stageM_dichotomy.png', dpi=150); plt.close()
print(json.dumps({'M1_free': {k: (v['n_reach_opt'], v['n_below_opt']) for k, v in M1['free'].items()},
                  'M2': [(o['name'], round(o['prediction'], 7), round(o['extrap_sigma0'], 7)) for o in M2['systems']]}, indent=0))
