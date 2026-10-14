"""Stage E analysis: numbers -> results/summary_E.json; figures stageE_local.png, stageE_dense.png, stageE_scaffold.png, stageE_valley.png"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})


def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


out = {}
# ---------------- task 1: local search at mid distances
E1 = json.load(open('results/e1_local.json')); E1b = json.load(open('results/e1b_continue.json')); EL = E1 + E1b
best = max(EL, key=lambda r: r['ratio'])
def margins(r):                           # R = 1 - (lam_rho - law) - (1 - lam_rho)(1 - CV*Omega) + D  is an upper bound for R
    lr = r['lam_rho_direct']; cvo = r['u_cv'] * r['omega_rms']
    return {'law_gap': lr - r['law_term'], 'spread_focus_margin': (1 - lr) * (1 - cvo), 'D': r['D'], 'CV': r['u_cv'], 'Omega': r['omega_rms'],
            'CV_Omega': cvo, 'rho': r['cov'] / (r['u_cv'] * (1 - lr) * r['omega_rms']), 'rho_CV_Omega': r['cov'] / (1 - lr),
            'certificate_upper_R': 1 - (lr - r['law_term']) - (1 - lr) * (1 - cvo) + r['D']}
out['task1'] = {'chains': [{k: r[k] for k in ['seed', 'K', 'step0', 'start_value', 'best_value', 'ratio', 'dist', 'R', 'sigma', 'H_nonPD_mass', 'Pv', 'Pw', 'D']}
                           for r in EL],
                'max_ratio_250': max(r['ratio'] for r in E1), 'max_ratio': best['ratio'], 'best': {k: best[k] for k in ['seed', 'K', 'ratio', 'dist', 'R', 'sigma',
                'H_nonPD_mass', 'Pv', 'Pw', 'D', 'cov', 'law_term', 'lam_rho_direct']}, 'best_margins': margins(best),
                'max_R': max(r['R'] for r in EL), 'resolution': max(abs(r['R_per_sd20'] - r['R']) for r in EL),
                'n_dist_at_upper_edge': sum(r['dist'] > 0.29 for r in EL), 'gain_last_100_of_continuation': [float(r['history'][-1] - r['history'][-11]) for r in E1b],
                'mass_dev': max(abs(r['mass'] - 1) for r in EL)}
# ---------------- task 2: dense near-top
D3 = [r for r in json.load(open('results/d3_sweep.json')) + json.load(open('results/d3_sweep2.json')) if r.get('feasible')]
E2 = [r for r in json.load(open('results/e2_dense.json')) if r.get('feasible')]
pts = [(r['dist'], r['excess']) for r in D3 + E2 if r['excess'] > 0]
dd = np.array([p[0] for p in pts]); ee = np.array([p[1] for p in pts])
grid_d = np.logspace(np.log10(1.5e-4), np.log10(0.3), 25)
env = np.array([ee[dd <= x].max() if (dd <= x).any() else np.nan for x in grid_d])     # largest excess with distance <= x
ok = np.isfinite(env)
fit = lambda x, y: float(np.polyfit(np.log(x), np.log(y), 1)[0])
dense = (grid_d >= 3e-4) & (grid_d <= 3e-3) & ok
out['task2'] = {'n_dense_feasible': len(E2), 'n_dense_restarts': len(json.load(open('results/e2_dense.json'))),
                'resolution': max(abs(r['R_per_sd20'] - r['R']) for r in E2), 'mass_dev': max(abs(r['mass'] - 1) for r in E2),
                'max_ratio_dense': max(r['ratio'] for r in E2), 'max_ratio_dist_le_3e-3_all': float(max(e / d for d, e in pts if d <= 3e-3)),
                'slope_envelope_all': fit(grid_d[ok], env[ok]), 'slope_envelope_dense': fit(grid_d[dense], env[dense]),
                'slope_points_dense': fit(np.array([r['dist'] for r in E2 if r['excess'] > 0 and 3e-4 <= r['dist'] <= 3e-3]),
                                          np.array([r['excess'] for r in E2 if r['excess'] > 0 and 3e-4 <= r['dist'] <= 3e-3])),
                'best_by_target': [], 'starts_best': {}}
for d in sorted(set(r['target'] for r in E2)):
    v = [r for r in E2 if r['target'] == d]; b = max(v, key=lambda r: r['excess'])
    out['task2']['best_by_target'].append({'target': d, 'dist': b['dist'], 'excess': b['excess'], 'ratio': b['ratio'], 'K': b['K'], 'start': b['start'],
                                           'n_feasible': len(v), 'excess_spread': [min(r['excess'] for r in v), max(r['excess'] for r in v)]})
for s in ['seed', 'previous', 'random']:
    v = [r for r in E2 if r['start'] == s]; out['task2']['starts_best'][s] = {'n': len(v), 'max_ratio': max((r['ratio'] for r in v), default=None)}
# ---------------- task 3: scaffold
X = json.load(open('results/e_all.json')); P = [x for x in X if x['kind'] == 'pendulum']
g = lambda L, k: np.array([x[k] for x in L])
cov, va, mis, om, cs = g(P, 'f_cov_part'), g(P, 'abar_var'), g(P, 'focus_mismatch'), g(P, 'omega_rms'), g(P, 'cov_cs_bound')
cv, um, lr, law, D, R = g(P, 'u_cv'), g(P, 'u_mean'), g(P, 'lam_rho_direct'), g(P, 'law_term'), g(P, 'D'), g(P, 'R')
dist = 1 - law; cert = law + cv * om * um + D
extra = [dict(r, f_cov_part=r['f'] - r['law_term']) for r in E2] + [dict(r, f_cov_part=r['cov'], law_term=r['law_term']) for r in EL]
ex_cert = [r['law_term'] + r['u_cv'] * r['omega_rms'] * (1 - (r['u_mean'] if 'u_mean' in r else 1 - (1 - r['lam_rho_direct']))) * 0 for r in []]
def cert_of(r):
    umr = r['u_mean'] if 'u_mean' in r else 1 - r['lam_rho_direct']
    return r['law_term'] + r['u_cv'] * r['omega_rms'] * umr + r['D']
out['task3'] = {'n': len(P), 'grid_check': float(np.abs(g(X, 'R') - g(X, 'R_qpgrid')).max()),
                'user_form_violations': int((np.abs(cov) > np.sqrt(va) * mis + 1e-12).sum()), 'user_form_max_ratio': float((np.abs(cov) / (np.sqrt(va) * mis)).max()),
                'cs_violations': int((np.abs(cov) > cs * (1 + 1e-9) + 1e-14).sum()), 'cs_max_ratio': float((np.abs(cov) / cs).max()),
                'identity_umean_eq_1_minus_lamrho': float(np.abs(um - (1 - lr)).max()), 'identity_sqrtvar_eq_cv_umean': float(np.abs(np.sqrt(va) - cv * um).max()),
                'cv_range': [float(cv.min()), float(np.median(cv)), float(cv.max())], 'omega_range': [float(om.min()), float(np.median(om)), float(om.max())],
                'n_cv_omega_lt_1': int((cv * om < 1).sum()), 'max_cv_omega': float((cv * om).max()),
                'n_certified': int((cert < 1).sum()), 'max_certificate': float(cert.max()), 'cert_ge_R_all': bool(np.all(cert >= R - 1e-10)),
                'spearman_cv_kurtosis': spear(cv, g(P, 'abar_kurtosis')), 'spearman_cv_nonPD': spear(cv, g(P, 'H_nonPD_mass')),
                'extra_n': len(extra), 'extra_cs_violations': int(sum(abs(r['f_cov_part']) > r['cov_cs_bound'] * (1 + 1e-9) + 1e-14 for r in extra)),
                'extra_n_certified': int(sum(cert_of(r) < 1 for r in extra)), 'extra_max_certificate': float(max(cert_of(r) for r in extra)),
                'extra_cert_ge_R': bool(all(cert_of(r) >= r['R'] - 1e-10 for r in extra)),
                # exact: cov = rho * CV * Omega * (1 - lambda_rho), rho = correlation of abar and omega' under p_Y
                'rho_range': [float((cov / cs)[cs > 0].min()), float(np.median((cov / cs)[cs > 0])), float((cov / cs)[cs > 0].max())],
                'max_rho_CV_Omega_all': float((cov / um).max()),
                'max_rho_CV_Omega_extra': float(max(r['f_cov_part'] / (r['u_mean'] if 'u_mean' in r else 1 - r['lam_rho_direct']) for r in extra))}
# ---------------- task 4: valleys
V4 = json.load(open('results/e4_valley.json'))
Pv, Pw = g(P, 'Pv'), g(P, 'Pw')
out['task4'] = {'all_min_Pv': float(Pv.min()), 'all_min_Pw': float(Pw.min()),
                'adv_min_Pv': float(min(r['Pv'] for r in V4 if r['target'] == 'Pv')), 'adv_min_Pw': float(min(r['Pw'] for r in V4 if r['target'] == 'Pw')),
                'adv_resolution': float(max(abs(r[r['target'] + '_per_sd20'] - r[r['target']]) for r in V4)),
                'adv': [{k: r[k] for k in ['target', 'K', 'sigma', 'Pv', 'Pw', 'R', 'H_nonPD_mass', 'mass_Hvv_neg', 'mass_Hww_neg', 'valley_depth_mean', 'valley_depth_max']} for r in V4],
                'max_valley_mass_all': float(g(P, 'H_nonPD_mass').max()), 'max_depth_mean_all': float(g(P, 'valley_depth_mean').max()),
                'max_depth_max_all': float(g(P, 'valley_depth_max').max()),
                'min_Pv_with_valley_v_gt_0.1': float(Pv[g(P, 'mass_Hvv_neg') > 0.1].min()), 'min_Pw_with_valley_w_gt_0.1': float(Pw[g(P, 'mass_Hww_neg') > 0.1].min()),
                'spearman_Pv_massv': spear(Pv, g(P, 'mass_Hvv_neg')), 'spearman_Pw_massw': spear(Pw, g(P, 'mass_Hww_neg'))}
json.dump(out, open('results/summary_E.json', 'w'), indent=1, default=float)

# ---------------- figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
a = ax[0]
for r in E1:
    col = C[0] if r['seed'].startswith('A') else C[1]; ls = '-' if 'split' not in r['seed'] else ':'
    a.plot(np.arange(len(r['history'])) * 10, r['history'], ls, color=col, lw=1.3)
for r in E1b:
    a.plot(250 + np.arange(len(r['history'])) * 10, r['history'], '-', color=C[2], lw=1.8)
a.plot([], [], '-', color=C[0], label='from the 0.830 state'); a.plot([], [], '-', color=C[1], label='from the R = 0.946 state')
a.plot([], [], ':', color=MUTED, label='with one component split (K + 1)'); a.plot([], [], '-', color=C[2], label='continuation of the best three')
a.axhline(1, color=INK, lw=1); a.set_ylim(0.4, 1.02); a.set_xlabel('step'); a.set_ylabel('excess / distance (distance 0.1–0.3)')
a.set_title('(1) local search: fraction of the room used', fontsize=9); a.legend(fontsize=7.5, frameon=False, loc='lower right')
a = ax[1]
top = sorted(EL, key=lambda r: -r['ratio'])[:6]
xx = np.arange(len(top)); M = [margins(r) for r in top]
a.bar(xx, [m['law_gap'] for m in M], 0.6, color=C[0], label='λ_ρ − law (focus not fully along V)', edgecolor='white')
a.bar(xx, [m['spread_focus_margin'] for m in M], 0.6, bottom=[m['law_gap'] for m in M], color=C[2], label='(1 − λ_ρ)(1 − CV·Ω)', edgecolor='white')
a.plot(xx, [1 - r['R'] for r in top], 'D', color=INK, label='1 − R (actual room left)')
a.plot(xx, [m['D'] for m in M], 'x', color=C[1], ms=8, label='D')
a.set_xticks(xx); a.set_xticklabels([f"{r['ratio']:.3f}" for r in top], fontsize=8); a.set_xlabel('fraction used (best chains)')
a.set_title('(2) the two safety margins of the scaffold (1 − R ≥ margins − D)', fontsize=9); a.legend(fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageE_local.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
a = ax[0]
a.loglog([r['dist'] for r in D3 if r['excess'] > 0], [r['excess'] for r in D3 if r['excess'] > 0], 'o', ms=4, color=MUTED, alpha=0.6, label='stage D searches')
for s, col in [('seed', C[0]), ('previous', C[1]), ('random', C[2])]:
    v = [r for r in E2 if r['start'] == s and r['excess'] > 0]
    a.loglog([r['dist'] for r in v], [r['excess'] for r in v], 'o', ms=4, color=col, alpha=0.7, label=f'dense search, start: {s}')
a.loglog(grid_d[ok], env[ok], '-', color=INK, lw=1.8, label=f"envelope (slope {out['task2']['slope_envelope_all']:.2f}; 3e-4–3e-3: {out['task2']['slope_envelope_dense']:.2f})")
a.loglog(grid_d, grid_d, '--', color=C[4], lw=1.2, label='excess = distance (R = κ)')
a.set_xlabel('distance from the peak'); a.set_ylabel('excess'); a.legend(fontsize=7, frameon=False, loc='upper left')
a.set_title('(1) largest excess vs distance (envelope = best with distance ≤ x)', fontsize=9)
a = ax[1]
a.semilogx([r['dist'] for r in D3], [r['ratio'] for r in D3], 'o', ms=4, color=MUTED, alpha=0.6, label='stage D')
a.semilogx([r['dist'] for r in E2], [r['ratio'] for r in E2], 'o', ms=4, color=C[0], alpha=0.7, label='dense search')
a.semilogx([r['dist'] for r in EL], [r['ratio'] for r in EL], 's', ms=5, color=C[1], label='mid-distance local search')
a.axhline(1, color=INK, lw=1); a.set_xlabel('distance from the peak'); a.set_ylabel('excess / distance')
a.set_title('(2) fraction of the room used', fontsize=9); a.legend(fontsize=8, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig('figures/stageE_dense.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
a = ax[0]
a.loglog(np.sqrt(va) * mis + 1e-16, np.abs(cov) + 1e-16, 'o', ms=2.5, color=C[1], alpha=0.5, label='proposed: √Var(ā) × mismatch (first moment)')
a.loglog(cs + 1e-16, np.abs(cov) + 1e-16, 'o', ms=2.5, color=C[0], alpha=0.5, label='Cauchy–Schwarz: √Var(ā) × Ω (rms)')
a.loglog([1e-8, 10], [1e-8, 10], '-', color=INK, lw=1)
a.set_xlim(1e-7, 10); a.set_ylim(1e-9, 1); a.set_xlabel('bound'); a.set_ylabel('|covariance term|'); a.legend(fontsize=7.5, frameon=False)
a.set_title(f"(1) |cov| ≤ bound: {out['task3']['user_form_violations']} violations (first moment), {out['task3']['cs_violations']} (rms)", fontsize=9)
a = ax[1]
sc = a.scatter(cv, om, c=np.clip(R, -1, 1), cmap='viridis', s=6)
xx = np.logspace(-2.5, 0.5, 50); a.plot(xx, 1 / xx, '-', color=C[1], lw=1.5, label='CV·Ω = 1')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('CV(u), u = 1 − ā  (spread of the distance to the top)'); a.set_ylabel('Ω  (rms focus mismatch)')
a.set_title(f"(2) CV·Ω < 1 in {out['task3']['n_cv_omega_lt_1']} of {len(P)}; certificate < 1 in {out['task3']['n_certified']}", fontsize=9)
a.legend(fontsize=8, frameon=False); fig.colorbar(sc, ax=a, fraction=0.046, label='R')
a = ax[2]
a.loglog(g(P, 'abar_kurtosis'), cv, 'o', ms=2.5, color=C[2], alpha=0.5)
a.set_xlabel('kurtosis of ā'); a.set_ylabel('CV(u)'); a.set_title(f"(3) CV(u) vs kurtosis (Spearman {out['task3']['spearman_cv_kurtosis']:.2f})", fontsize=9)
fig.tight_layout(); fig.savefig('figures/stageE_scaffold.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for a, key, mkey, lab, col in [(ax[0], 'Pv', 'mass_Hvv_neg', 'P_v', C[0]), (ax[1], 'Pw', 'mass_Hww_neg', 'P_w', C[1])]:
    sc = a.scatter(g(P, mkey), g(P, key), c=np.log10(g(P, 'valley_depth_mean') + 1e-6), cmap='magma_r', s=6)
    adv = [r for r in V4 if r['target'] == key]
    a.scatter([r[mkey] for r in adv], [r[key] for r in adv], marker='*', s=120, color=col, edgecolor=INK, linewidth=0.6, label=f'searches minimising {lab}', zorder=5)
    a.axhline(0, color=INK, lw=1); a.set_yscale('symlog', linthresh=1e-3)
    a.set_xlabel(f'valley mass along {"v" if key == "Pv" else "w"}'); a.set_ylabel(lab); a.legend(fontsize=8, frameon=False)
    a.set_title(f'{lab} ≥ 0 vs the valley (colour: log10 mean depth)', fontsize=9); fig.colorbar(sc, ax=a, fraction=0.046)
fig.tight_layout(); fig.savefig('figures/stageE_valley.png', dpi=150); plt.close(fig)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ['chains', 'adv', 'best_by_target']} for k, v in out.items()}, indent=1, default=float))
