"""Stage J analysis: results/summary_J.json and figures/stageJ_bound.png."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
J = [o for o in json.load(open('results/j1_weighted_cr.json')) if 'error' not in o]
B = {(o['tag'], o['sigma'], round(o["R"], 9)): o for o in json.load(open('results/i2_bound.json')) if 'error' not in o}
small = [o for o in J if o['sigma'] <= 1e-2]
for o in J:
    b = B.get((o['tag'], o['sigma'], round(o["R"], 9)))
    o['bathtub_over_sigma'] = b['LB_over_sigma'] if b else None
out = {'n': len(J), 'n_small': len(small),
       'min_gap': min(o['gap_over_sigma'] for o in small), 'min_LB2': min(o['LB2_over_sigma'] for o in small), 'min_c_state': min(o['c_state'] for o in small),
       'min_bathtub': min(o['bathtub_over_sigma'] for o in small if o['bathtub_over_sigma'] is not None),
       'min_ThetaP': min(o['ThetaP'] for o in small), 'min_Theta': min(o['Theta'] for o in small),
       'min_Theta_geometric': min(o['Theta_geometric'] for o in small), 'min_Theta_slope': min(o['Theta_slope'] for o in small),
       'max_eps_over_sigma': max(o['eps_over_sigma'] for o in small), 'max_D_over_sigma': max(o['D_over_sigma'] for o in small),
       'all_c_state_le_gap': all(o['c_state'] <= o['gap_over_sigma'] + 1e-9 for o in J), 'all_LB2_le_gap': all(o['LB2_over_sigma'] <= o['gap_over_sigma'] + 1e-9 for o in J),
       'max_identity_err': max(abs(o['checks']['identity']) for o in J), 'min_step2': min(o['checks']['step2_min_slack'] for o in J),
       'min_step3': min(o['checks']['step3_slack'] for o in J), 'min_step4': min(o['checks']['step4_slack'] for o in J),
       'min_step5_relative': min(o['checks']['step5_slack'] / (o['Jvv_s2'] / o['sigma'] ** 2) for o in J),
       'max_LB2_minus_1mf': max(o['checks']['LB2_le_1mf'] for o in J)}
tags = sorted(set(o['tag'] for o in small))
out['by_tag'] = {t: {'n': sum(o['tag'] == t for o in small), **{k: (min if k != 'eps' else max)(o[{'gap': 'gap_over_sigma', 'bathtub': 'bathtub_over_sigma', 'LB2': 'LB2_over_sigma', 'c_state': 'c_state', 'ThetaP': 'ThetaP', 'slope': 'Theta_slope', 'eps': 'eps_over_sigma'}[k]] for o in small if o['tag'] == t and o['bathtub_over_sigma'] is not None)
                     for k in ['gap', 'bathtub', 'LB2', 'c_state', 'ThetaP', 'slope', 'eps']}} for t in tags}
out['improvement'] = {'bathtub_min': out['min_bathtub'], 'weighted_CR_exact_form_min': out['min_LB2'], 'weighted_CR_closed_form_min': out['min_c_state'], 'actual_min': out['min_gap']}
seg = sorted([o for o in J if o['tag'] == 'Rstar_segment'], key=lambda o: o['sigma'])
out['single_segment'] = [{k: o[k] for k in ['sigma', 'gap_over_sigma', 'LB2_over_sigma', 'c_state', 'ThetaP', 'j']} for o in seg]
json.dump(out, open('results/summary_J.json', 'w'), indent=1, default=float)

COL = dict(zip(tags, C + ['#444', '#999', '#c7a', '#7ac', '#ac7', '#555']))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
a = ax[0]
for t in tags:
    v = [o for o in small if o['tag'] == t and o['bathtub_over_sigma'] is not None]
    a.scatter([o['gap_over_sigma'] for o in v], [o['bathtub_over_sigma'] for o in v], s=9, marker='x', color=COL[t], alpha=0.5)
    a.scatter([o['gap_over_sigma'] for o in v], [o['c_state'] for o in v], s=11, color=COL[t], label=t)
x = np.array([0.7, 4]); a.plot(x, x, '-', color=INK, lw=1)
a.axhline(out['min_c_state'], color=C[0], ls=':', lw=1); a.axhline(out['min_bathtub'], color=MUTED, ls=':', lw=1); a.axvline(out['min_gap'], color=INK, ls='--', lw=0.8)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(0.85, 4); a.set_ylim(0.7, 4)
a.set_xlabel('actual (1 − R)/σ'); a.set_ylabel('rigorous state-wise bound / σ'); a.legend(fontsize=6, frameon=False, ncol=2, loc='upper left')
a.set_title(f"(1) ● weighted Cramér–Rao (min {out['min_c_state']:.3f}),  × stage I bathtub (min {out['min_bathtub']:.3f})", fontsize=8.5)
a = ax[1]
for t in tags:
    v = [o for o in small if o['tag'] == t]
    a.scatter([o['Theta_slope'] for o in v], [o['ThetaP'] for o in v], s=11, color=COL[t])
a.axhline(1, color=MUTED, lw=0.8, ls='--'); a.axvline(0, color=MUTED, lw=0.8, ls='--')
a.set_xlabel('focus-slope part  E[u ∂_w w]/E[w]'); a.set_ylabel("Θ'  (the constant of the state-wise theorem)")
a.set_title("(2) Θ' stays ≥ 0.91 although the focus-slope part reaches −0.32", fontsize=8.5)
a = ax[2]
for t in tags:
    v = [o for o in small if o['tag'] == t]
    a.scatter([o['gap_over_sigma'] for o in v], [o['LB2_over_sigma'] / o['gap_over_sigma'] for o in v], s=11, color=COL[t])
    v2 = [o for o in v if o['bathtub_over_sigma'] is not None]
    a.scatter([o['gap_over_sigma'] for o in v2], [o['bathtub_over_sigma'] / o['gap_over_sigma'] for o in v2], s=8, marker='x', color=COL[t], alpha=0.4)
a.set_xscale('log'); a.set_xlabel('actual (1 − R)/σ'); a.set_ylabel('bound / actual'); a.set_ylim(0.3, 1.02)
a.set_title('(3) tightness: ● weighted CR (exact form), × bathtub', fontsize=8.5)
plt.tight_layout(); plt.savefig('figures/stageJ_bound.png', dpi=150); plt.close()
print(json.dumps({k: v for k, v in out.items() if k not in ['by_tag']}, indent=1, default=float))
