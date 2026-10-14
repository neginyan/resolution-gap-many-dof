"""Stage D analysis: numbers -> results/summary_D.json; figures stageD_M.png, stageD_Bprime.png, stageD_peak.png"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})


def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


X = json.load(open('results/d1_all.json'))
P = [x for x in X if x['kind'] == 'pendulum']; O = [x for x in X if x['kind'] != 'pendulum']
g = lambda L, k: np.array([x[k] for x in L])
R, Pv, Pw, D, M = g(P, 'R'), g(P, 'Pv'), g(P, 'Pw'), g(P, 'D'), g(P, 'M_over_trF')
B, Bp, f, val = g(P, 'B_balance'), g(P, 'B_prime'), g(P, 'f'), g(P, 'H_nonPD_mass')
out = {'task1': {}, 'task2': {}, 'task3': {}}
t1 = out['task1']
t1.update({'n_pendulum': len(P), 'n_other': len(O),
           'grid_check_max_abs_R_vw_minus_qp': float(np.abs(g(X, 'R') - g(X, 'R_qpgrid')).max()),
           'identity_M_eq_1_minus_R_maxerr': float(np.abs(M - (1 - R)).max()),
           'identity_ML_eq_kloc_minus_R_maxerr': float(np.abs(g(O, 'ML_over_trF') - (g(O, 'kappa_loc') - g(O, 'R'))).max()),
           'min_M': float(M.min()), 'min_Pv': float(Pv.min()), 'min_Pw': float(Pw.min()),
           'n_Pv_neg': int((Pv < 0).sum()), 'n_Pw_neg': int((Pw < 0).sum()), 'n_D_pos': int((D > 0).sum()), 'max_D': float(D.max()),
           'min_Pv_neg_part': float(g(P, 'Pv_neg_part').min()), 'min_Pw_neg_part': float(g(P, 'Pw_neg_part').min()),
           'max_mass_Hvv_neg': float(g(P, 'mass_Hvv_neg').max()), 'max_mass_Hww_neg': float(g(P, 'mass_Hww_neg').max()),
           'spearman_Pvneg_massHvvneg': spear(-g(P, 'Pv_neg_part'), g(P, 'mass_Hvv_neg')),
           'spearman_Pwneg_massHwwneg': spear(-g(P, 'Pw_neg_part'), g(P, 'mass_Hww_neg')),
           'n_Pv_lt_Pw': int((Pv < Pw).sum()),
           'closest': [{k: P[i][k] for k in ['tag', 'sigma', 'R', 'Pv', 'Pw', 'D', 'M_over_trF', 'Pv_neg_part', 'Pw_neg_part', 'mass_Hvv_neg', 'mass_Hww_neg', 'H_nonPD_mass']}
                       for i in np.argsort(M)[:6]],
           'other_min_ML': float(g(O, 'ML_over_trF').min()), 'other_min_PvL': float(g(O, 'PvL').min()), 'other_min_PwL': float(g(O, 'PwL').min()),
           'other_n_R_gt_kappa_loc': int((g(O, 'R') > g(O, 'kappa_loc')).sum())})
t2 = out['task2']
i21 = np.where(B > 1)[0]; itop = int(np.argmax(R))
t2.update({'pend_f_le_Bp_all': bool(np.all(f <= Bp + 1e-10)), 'pend_pointwise_violation_max': float(g(P, 'B_prime_pointwise_violation_mass').max()),
           'n_B_gt_1': int(len(i21)), 'n_Bp_le_1_among_B_gt_1': int((Bp[i21] <= 1).sum()), 'n_Bp_gt_1': int((Bp > 1).sum()),
           'n_Bp_plus_D_gt_1': int((Bp + D > 1).sum()), 'max_Bp': float(Bp.max()),
           'top_state': {'R': float(R[itop]), 'B': float(B[itop]), 'B_prime': float(Bp[itop]), 'D': float(D[itop])},
           'Bp_minus_f_median': float(np.median(Bp - f)), 'B_minus_Bp_median': float(np.median(B - Bp)),
           'other_n_states_with_pointwise_violation': int((g(O, 'B_prime_pointwise_violation_mass') > 0).sum()),
           'other_n_f_gt_Bp': int((g(O, 'f') > g(O, 'B_prime') + 1e-10).sum()),
           'other_f_le_Bgen_all': bool(np.all(g(O, 'f') <= g(O, 'B_prime_gen') + 1e-10)),
           'other_max_f_minus_Bp': float((g(O, 'f') - g(O, 'B_prime')).max()),
           'n_no_valley': int((val < 1e-12).sum()), 'no_valley_max_abs_Bp_minus_f': float(np.abs((Bp - f)[val < 1e-12]).max()),
           'valley_median_Bp_minus_f': float(np.median((Bp - f)[val >= 1e-12])), 'max_Bp_minus_f': float((Bp - f).max()),
           'top_Bp_minus_f': float(Bp[itop] - f[itop]), 'top_valley_mass': float(val[itop])})
# ---------------- task 3
RA = json.load(open('results/d3_ratio.json')); SW = json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json'))
t3 = out['task3']
t3['ratio_part'] = [{k: r.get(k) for k in ['K', 'sigma_fixed', 'feasible', 'dist', 'excess', 'ratio', 'cov', 'D', 'abar_var', 'focus_mismatch', 'H_nonPD_mass']} for r in RA]
t3['ratio_part_max_ratio'] = max(r['ratio'] for r in RA if r.get('feasible'))
t3['ratio_part_resolution'] = max(abs(r['R_per_sd20'] - r['R']) for r in RA if r.get('feasible'))
rows = [r for r in SW if r.get('feasible')]
t3['sweep'] = [{k: r.get(k) for k in ['part', 'K', 'target_dist', 'sigma', 'dist', 'excess', 'ratio', 'cov', 'D', 'abar_var', 'focus_mismatch', 'H_nonPD_mass', 'R']} for r in SW]
t3['sweep_resolution'] = max(abs(r['R_per_sd20'] - r['R']) for r in rows)
t3['sweep_mass_dev'] = max(abs(r['mass'] - 1) for r in rows)
t3['n_infeasible'] = sum(1 for r in SW if not r.get('feasible'))
best = {}
for r in rows:
    d = r['target_dist']
    if d not in best or r['excess'] > best[d]['excess']: best[d] = r
ds = sorted(best); dist = np.array([best[d]['dist'] for d in ds]); exc = np.array([best[d]['excess'] for d in ds])
var = np.array([best[d]['abar_var'] for d in ds]); mis = np.array([best[d]['focus_mismatch'] for d in ds])
pos = exc > 0
t3['best_by_target'] = [{'target': d, 'dist': best[d]['dist'], 'excess': best[d]['excess'], 'ratio': best[d]['ratio'], 'K': best[d]['K'],
                         'sigma': best[d]['sigma'], 'abar_var': best[d]['abar_var'], 'focus_mismatch': best[d]['focus_mismatch']} for d in ds]
fit = lambda x, y: float(np.polyfit(np.log(x), np.log(y), 1)[0])
t3['slope_excess'] = fit(dist[pos], exc[pos]) if pos.sum() >= 3 else None
t3['slope_excess_small'] = fit(dist[pos & (dist <= 0.05)], exc[pos & (dist <= 0.05)]) if (pos & (dist <= 0.05)).sum() >= 3 else None
t3['slope_abar_var'] = fit(dist, var); t3['slope_mismatch'] = fit(dist, mis)
t3['max_ratio_sweep'] = float(max(r['ratio'] for r in rows))
t3['max_ratio_sweep_state'] = {k: max(rows, key=lambda r: r['ratio'])[k] for k in ['part', 'K', 'sigma', 'dist', 'excess', 'H_nonPD_mass']}
small = [r for r in rows if r['dist'] <= 0.02]
t3['max_ratio_dist_le_0.02'] = float(max(r['ratio'] for r in small)); t3['n_dist_le_0.02'] = len(small)
t3['min_dist_reached'] = float(min(r['dist'] for r in rows))
t3['local_slope_last_two'] = float(np.log(exc[1] / exc[0]) / np.log(dist[1] / dist[0])) if len(ds) > 1 else None
t3['cs_ratio'] = [float(e_ / (np.sqrt(v_) * m_)) for e_, v_, m_ in zip(exc, var, mis)]     # excess / (sqrt Var(abar) * mismatch)
t3['n_unseeded_infeasible'] = sum(1 for r in SW if r['part'] == 'sweep' and not r.get('feasible'))
json.dump(out, open('results/summary_D.json', 'w'), indent=1, default=float)

# ---------------- figures
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
a = ax[0]
sc = a.scatter(Pv, Pw, c=val, cmap='viridis', s=6)
cl = np.argsort(M)[:6]; a.scatter(Pv[cl], Pw[cl], s=60, facecolor='none', edgecolor=C[1], linewidth=1.5, label='6 states closest to R = κ')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('$P_v$ = E[(1 − ā) $H_{vv}$]/tr F'); a.set_ylabel('$P_w$ = E[(1 + ā) $H_{ww}$]/tr F')
a.set_title('(1) both parts stay positive in all 1536 states', fontsize=9); a.legend(fontsize=8, frameon=False, loc='lower right')
fig.colorbar(sc, ax=a, fraction=0.046, label='valley mass')
a = ax[1]
a.plot(g(P, 'mass_Hvv_neg'), -g(P, 'Pv_neg_part'), 'o', ms=2.5, color=C[0], alpha=0.5, label='$P_v$: negative part vs mass where $H_{vv}$ < 0')
a.plot(g(P, 'mass_Hww_neg'), -g(P, 'Pw_neg_part'), 'o', ms=2.5, color=C[1], alpha=0.5, label='$P_w$: negative part vs mass where $H_{ww}$ < 0')
a.set_xlabel('mass of the valley in that direction'); a.set_ylabel('size of the negative contribution')
a.set_title('(2) the negative contributions come from the valleys', fontsize=9); a.legend(fontsize=7.5, frameon=False)
a = ax[2]
xx = np.arange(6); w_ = 0.6
a.bar(xx, Pv[cl], w_, color=C[0], label='$P_v$', edgecolor='white'); a.bar(xx, Pw[cl], w_, bottom=Pv[cl], color=C[1], label='$P_w$', edgecolor='white')
a.bar(xx, -D[cl], w_, bottom=Pv[cl] + Pw[cl], color=C[3], label='−D', edgecolor='white')
a.plot(xx, 1 - R[cl], 'D', color=INK, ms=5, label='1 − R')
a.set_xticks(xx); a.set_xticklabels([f"R={R[i]:.3f}" for i in cl], fontsize=7.5); a.set_ylabel('M / tr F')
a.set_title('(3) M = $P_v$ + $P_w$ − D for the closest states', fontsize=9); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageD_M.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
a = ax[0]
a.plot(B, Bp, 'o', ms=2.5, color=MUTED, alpha=0.5, label='pendulum states')
a.plot(B[i21], Bp[i21], 'o', ms=6, color=C[1], label='the 21 states with B > κ')
a.plot(B[itop], Bp[itop], '*', ms=14, color=C[0], mec=INK, label=f'largest R ({R[itop]:.3f})')
a.axhline(1, color=INK, lw=1); a.axvline(1, color=INK, lw=1, ls=':')
a.set_xlabel('B'); a.set_ylabel("B′"); a.set_title("(1) the tightened bound B′ vs B", fontsize=9); a.legend(fontsize=8, frameon=False)
a = ax[1]
a.plot(Bp + D, R, 'o', ms=2.5, color=C[0], alpha=0.5); lo_ = min((Bp + D).min(), R.min())
a.plot([lo_, 1.05], [lo_, 1.05], '-', color=INK, lw=1, label='R = B′ + D'); a.axvline(1, color=C[1], lw=1.5, label='κ = 1')
a.set_xlabel("B′ + D"); a.set_ylabel('R'); a.set_title("(2) R ≤ B′ + D; B′ + D > κ only for the largest-R state", fontsize=9); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageD_Bprime.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
a = ax[0]
for K, col in [(2, C[0]), (3, C[1])]:
    v = [r for r in rows if r['K'] == K and r['excess'] > 0]
    a.loglog([r['dist'] for r in v], [r['excess'] for r in v], 'o', ms=5, color=col, alpha=0.6, label=f'K = {K} (all searches)')
a.loglog(dist[pos], exc[pos], 'o', ms=9, mfc='none', mec=INK, label='largest per target distance')
if t3['slope_excess'] is not None:
    xx = np.logspace(np.log10(dist[pos].min()), np.log10(dist[pos].max()), 20)
    c0 = np.exp(np.polyfit(np.log(dist[pos]), np.log(exc[pos]), 1)[1])
    a.loglog(xx, c0 * xx ** t3['slope_excess'], '-', color=INK, lw=1.2, label=f"fit: slope {t3['slope_excess']:.2f}")
    a.loglog(xx, xx, '--', color=C[2], lw=1.2, label='excess = distance (R = κ)')
    a.loglog(xx, exc[pos][0] * np.sqrt(xx / dist[pos][0]), ':', color=C[4], lw=1.5, label='slope 1/2 (reference)')
a.set_xlabel('distance from the peak  κ − law'); a.set_ylabel('largest excess found'); a.legend(fontsize=8, frameon=False)
a.set_title('(1) largest excess vs distance from the peak', fontsize=9)
a = ax[1]
a.loglog(dist, var, 'o-', color=C[0], label=f"Var(ā)  (slope {t3['slope_abar_var']:.2f})")
a.loglog(dist, mis, 's-', color=C[1], label=f"focus mismatch E‖H − F‖/tr F  (slope {t3['slope_mismatch']:.2f})")
a.set_xlabel('distance from the peak'); a.set_title('(2) best states: spread of stretch and focus mismatch', fontsize=9); a.legend(fontsize=8, frameon=False)
a = ax[2]
for K, col in [(2, C[0]), (3, C[1])]:
    v = [r for r in rows if r['K'] == K]
    a.semilogx([r['dist'] for r in v], [r['ratio'] for r in v], 'o', ms=6, color=col, label=f'K = {K}')
a.axhline(1, color=INK, lw=1); a.set_xlabel('distance from the peak'); a.set_ylabel('excess / distance  (Conjecture 1 ⟺ ≤ 1)')
a.set_title('(3) fraction of the room used', fontsize=9); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageD_peak.png', dpi=150); plt.close(fig)
print(json.dumps(out['task3'], indent=1, default=float)[:5000])
