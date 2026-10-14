"""Stage K analysis: results/summary_K.json and figures/stageK_smooth.png."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
L = ['0.25', '0.5', '1.0', '2.0', '4.0', '8.0']
S = [o for o in json.load(open('results/j2_smooth.json')) if 'error' not in o]
J = [o for o in json.load(open('results/j1_weighted_cr.json')) if 'error' not in o]
small = [o for o in S if o['sigma'] <= 1e-2]
for o in S: o['best'] = max(o['by_lambda'][l]['c_bound2'] for l in L)
conc = [o for o in small if o['by_lambda']['1.0']['eps2_over_sigma'] + max(o['D_over_sigma'], 0) <= 0.05]
out = {'corrected_stageJ': {'eps_factor_range': [min(o['eps_factor'] for o in J), max(o['eps_factor'] for o in J)],
                            'min_positive_part_over_eps': min(o['pos_part_over_eps'] for o in J if o['eps_over_sigma'] > 0),
                            'min_c_state_corrected': min(o['c_state'] for o in J if o['sigma'] <= 1e-2),
                            'min_c_state_stageJ_form': min(o['c_state_stageJ'] for o in J if o['sigma'] <= 1e-2),
                            'all_le_gap': all(o['c_state'] <= o['gap_over_sigma'] + 1e-9 for o in J)},
       'by_lambda': {l: {'min_alpha': min(o['by_lambda'][l]['alpha'] for o in small), 'min_G': min(o['by_lambda'][l]['G'] for o in small),
                         'min_Theta_chi': min(o['by_lambda'][l]['Theta_chi'] for o in small), 'min_kappa2': min(o['by_lambda'][l]['kappa2'] for o in small),
                         'max_eps2_over_sigma': max(o['by_lambda'][l]['eps2_over_sigma'] for o in small),
                         'min_bound2_all': min(o['by_lambda'][l]['c_bound2'] for o in small), 'min_bound2_concentrated': min(o['by_lambda'][l]['c_bound2'] for o in conc),
                         'min_alpha_concentrated': min(o['by_lambda'][l]['alpha'] for o in conc)} for l in L},
       'n_small': len(small), 'n_concentrated': len(conc),
       'min_best_bound_all': min(o['best'] for o in small), 'min_best_bound_concentrated': min(o['best'] for o in conc),
       'checks': {'min_Z_minus_Zlow': min(o['by_lambda'][l]['check_Z_ge_Zlow'] for o in S for l in L),
                  'min_Egw_minus_Egchi': min(o['by_lambda'][l]['check_Egw_ge_Egchi'] for o in S for l in L),
                  'min_slope_slack': min(o['by_lambda'][l]['check_slope_le_L'] for o in S for l in L),
                  'max_lipschitz_ratio': max(o['by_lambda'][l]['lipschitz_ratio'] for o in S for l in L),
                  'min_A_slack': min(o['by_lambda'][l]['check_A_le_Jww'] for o in S for l in L),
                  'all_bound2_le_gap': all(o['by_lambda'][l]['c_bound2'] <= o['gap_over_sigma'] + 1e-9 for o in S for l in L)}}
tags = sorted(set(o['tag'] for o in small))
out['by_tag'] = {t: {'n': sum(o['tag'] == t for o in small), 'min_gap': min(o['gap_over_sigma'] for o in small if o['tag'] == t),
                     'min_best_bound': min(o['best'] for o in small if o['tag'] == t), 'min_alpha_lambda1': min(o['by_lambda']['1.0']['alpha'] for o in small if o['tag'] == t),
                     'max_eps2_lambda1': max(o['by_lambda']['1.0']['eps2_over_sigma'] for o in small if o['tag'] == t), 'max_K': max(o['K'] for o in small if o['tag'] == t)} for t in tags}
# example of the conditional uniform constant with the class values at lambda = 1
b1 = out['by_lambda']['1.0']; a0, G0, k0 = b1['min_alpha_concentrated'], min(o['by_lambda']['1.0']['G'] for o in conc), min(o['by_lambda']['1.0']['kappa2'] for o in conc)
eta = max(o['by_lambda']['1.0']['eps2_over_sigma'] + max(o['D_over_sigma'], 0) for o in conc)
T0 = a0 * G0 / 2 * np.sqrt(k0 / 2)
out['conditional_example_lambda1'] = {'alpha0': a0, 'G0': G0, 'kappa0': k0, 'eta': eta, 'Theta0': a0 * G0 / 2, 'c_leading': T0 - eta}
json.dump(out, open('results/summary_K.json', 'w'), indent=1, default=float)

COL = dict(zip(tags, C + ['#444', '#999', '#c7a', '#7ac', '#ac7', '#555']))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
lam = [float(l) for l in L]
a = ax[0]
for t in ['Rstar_segment', 'h4_growk', 'i1_fan', 'near-top']:
    v = [o for o in small if o['tag'] == t]
    a.plot(lam, [min(o['by_lambda'][l]['alpha'] for o in v) for l in L], 'o-', color=COL[t], label=f'α (min over {t})')
    a.plot(lam, [min(o['by_lambda'][l]['Theta_chi'] for o in v) for l in L], 's--', color=COL[t], alpha=0.7)
a.plot(lam, 1 / (1 + np.array(lam)), ':', color=INK, lw=1, label='1/(1 + λ): the price of smoothing')
a.set_xscale('log', base=2); a.set_xlabel('λ  (allowed rate of change of √χ along W, in units of √J_ww)'); a.set_ylim(0, 1.05)
a.legend(fontsize=6.5, frameon=False); a.set_title('(1) ● α = surviving focus,  ■ Θ_χ = α G/(1 + λ√P)', fontsize=8.5)
a = ax[1]
Jd = {(o['tag'], o['sigma'], round(o['gap_over_sigma'], 6)): o for o in J}
for t in tags:
    v = [o for o in small if o['tag'] == t]
    a.scatter([o['gap_over_sigma'] for o in v], [max(o['best'], 0.05) for o in v], s=11, color=COL[t], label=t)
    vj = [Jd.get((o['tag'], o['sigma'], round(o['gap_over_sigma'], 6))) for o in v]
    a.scatter([o['gap_over_sigma'] for o, q in zip(v, vj) if q], [q['c_state'] for q in vj if q], s=7, marker='x', color=COL[t], alpha=0.4)
x = np.array([0.8, 10]); a.plot(x, x, '-', color=INK, lw=1)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(0.85, 10); a.set_ylim(0.05, 10)
a.set_xlabel('actual (1 − R)/σ'); a.set_ylabel('bound / σ'); a.legend(fontsize=6, frameon=False, ncol=2, loc='lower right')
a.set_title('(2) ● smoothed weight (best λ),  × stage J (state-wise Θ′)', fontsize=8.5)
a = ax[2]
for t in tags:
    v = [o for o in small if o['tag'] == t]
    a.scatter([o['by_lambda']['1.0']['eps2_over_sigma'] + max(o['D_over_sigma'], 0) for o in v], [o['by_lambda']['1.0']['c_bound2'] for o in v], s=11, color=COL[t])
a.axvline(0.05, color=MUTED, ls='--', lw=1); a.axhline(0, color=INK, lw=0.8); a.set_xscale('symlog', linthresh=1e-3); a.set_xlim(left=0)
a.set_xlabel('(ε₂ + D)/σ at λ = 1'); a.set_ylabel('smoothed bound / σ at λ = 1')
a.set_title('(3) the bound is lost only where ε₂ (wide states far from the top) is large', fontsize=8.5)
plt.tight_layout(); plt.savefig('figures/stageK_smooth.png', dpi=150); plt.close()
print(json.dumps({k: v for k, v in out.items() if k not in ['by_tag']}, indent=1, default=float))
