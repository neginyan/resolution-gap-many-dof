"""Stage C+ analysis: numbers -> results/summary_C5.json, figures -> figures/stageC5_curvature.png, figures/stageC5_balance.png"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
X = json.load(open('results/c5_balance.json'))
P = [x for x in X if x['kind'] == 'pendulum']; O = [x for x in X if x['kind'] != 'pendulum']
g = lambda L, k: np.array([x[k] for x in L])
R, law, cov, D, a2, a2y = g(P, 'R'), g(P, 'law_term'), g(P, 'cov'), g(P, 'D'), g(P, 'Ea2_rho'), g(P, 'Ea2_pY')
lr, B, Bk, val = g(P, 'lam_rho_direct'), g(P, 'B_balance'), g(P, 'B_kappa_factor'), g(P, 'H_nonPD_mass')
cpos, cneg = g(P, 'cov_pos_curv'), g(P, 'cov_neg_curv'); f = g(P, 'f')
exc = R - law; dist = 1.0 - law; ratio = exc / dist
tags = np.array([x['tag'] for x in P]); nz = np.abs(cov) > 1e-6

def frac(m): return float(m.sum()) / max(int(len(m)), 1)
out = {'n_pendulum': len(P), 'n_other': len(O),
       'identity_Ea2_rho_eq_half_minus_lam_rho_maxerr': float(np.abs(a2 - (0.5 - lr)).max()),
       'cov_split_maxerr': float(np.abs(cpos + cneg - cov).max()),
       # (A) curvature sign vs covariance sign
       'n_cov_nonzero': int(nz.sum()),
       'sign_cov_eq_sign_Ea2rho': frac((np.sign(cov) == np.sign(a2))[nz]),
       'sign_cov_eq_sign_Ea2pY': frac((np.sign(cov) == np.sign(a2y))[nz]),
       'n_neg_curv_pos_cov': int(((a2 < 0) & (cov > 1e-6)).sum()), 'max_cov_at_neg_curv': float(cov[a2 < 0].max()),
       'n_pos_curv_neg_cov': int(((a2 > 0) & (cov < -1e-6)).sum()),
       'cov_from_neg_local_curv_share_of_pos_cov': float(np.median((cneg / cov)[cov > 1e-3])),
       # (B) excess vs distance from the peak
       'all_excess_le_dist': bool(np.all(exc <= dist + 1e-12)), 'max_ratio': float(ratio.max()),
       'max_ratio_at_neg_curv': float(ratio[a2 < 0].max()), 'max_ratio_at_pos_curv': float(ratio[a2 > 0].max()),
       'max_ratio_curv_bins': {}, 'min_dist': float(dist.min()),
       # (C) balance bound
       'f_le_B_all': bool(np.all(f <= B + 1e-10)), 'R_le_B_plus_D_all': bool(np.all(R <= B + D + 1e-10)),
       'n_B_le_1': int((B <= 1).sum()), 'n_B_plus_D_le_1': int((B + D <= 1).sum()), 'max_B': float(B.max()), 'max_B_plus_D': float((B + D).max()),
       'max_Bk': float(Bk.max()), 'n_no_valley': int((Bk < 1 + 1e-9).sum()),
       'B_minus_f_median': float(np.median(B - f)), 'spearman_gap_valley': None,
       'max_B_no_valley': float(B[Bk < 1 + 1e-9].max()), 'n_B_gt_1': int((B > 1).sum()), 'min_valley_mass_B_gt_1': float(val[B > 1].min()),
       'max_R_where_B_gt_1': float(R[B > 1].max()),
       'dist_ge_half_plus_Ea2_all': bool(np.all(dist >= 0.5 + a2 - 1e-12)),
       'n_c4_family_cov_sign_rule': None}
for lo, hi in [(-0.5, -0.25), (-0.25, 0.0), (0.0, 0.25), (0.25, 0.5)]:
    m = (a2 >= lo) & (a2 < hi)
    out['max_ratio_curv_bins'][f'{lo}..{hi}'] = {'n': int(m.sum()), 'max_ratio': float(ratio[m].max()) if m.any() else None,
                                                 'max_cov': float(cov[m].max()) if m.any() else None}
out['dist_bins'] = {}
for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0)]:
    m = (dist >= lo) & (dist < hi)
    out['dist_bins'][f'{lo}..{hi}'] = {'n': int(m.sum()), 'max_excess': float(exc[m].max()), 'max_ratio': float(ratio[m].max()),
                                       'Ea2_range': [float(a2[m].min()), float(a2[m].max())]}
ra = np.argsort(np.argsort(B - f)); rb = np.argsort(np.argsort(val)); out['spearman_gap_valley'] = float(np.corrcoef(ra, rb)[0, 1])
i = int(np.argmax(B + D)); out['max_B_plus_D_state'] = {k: P[i][k] for k in ['tag', 'sigma', 'R', 'B_balance', 'D', 'B_kappa_factor', 'H_nonPD_mass', 'law_term']}
i = int(np.argmax(ratio)); out['max_ratio_state'] = {k: P[i][k] for k in ['tag', 'sigma', 'R', 'law_term', 'cov', 'D', 'Ea2_rho', 'H_nonPD_mass']}
# other flows: a'' = -3 everywhere
fam = [x for x in O if x['tag'] == 'c4']
co = g(O, 'cov'); out['other_n'] = len(O); out['other_n_cov_pos'] = int((co > 1e-6).sum()); out['other_n_cov_neg'] = int((co < -1e-6).sum())
out['other_f_le_B'] = bool(np.all(g(O, 'f') <= g(O, 'B_balance') + 1e-10))
json.dump(out, open('results/summary_C5.json', 'w'), indent=1, default=float)

# ---------------- figure: curvature picture
fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
mk = {'random': ('o', 2.5, 0.35), 'm1': ('s', 3, 0.6), 'c1': ('D', 6, 1.0)}
lab = {'random': 'random mixtures', 'm1': 'measurement-1 states', 'c1': 'stage-C adversarial'}
a = ax[0]
for t, (m_, ms, al) in mk.items():
    k = tags == t; a.plot(a2[k], cov[k], m_, ms=ms, alpha=al, color=C[['random', 'm1', 'c1'].index(t)], label=lab[t],
                         mec=INK if t == 'c1' else 'none', mew=0.6)
a.axhline(0, color=INK, lw=0.8); a.axvline(0, color=INK, lw=0.8)
a.set_xlabel("curvature felt by the state  $E_\\rho[a''(q)]$  (pendulum: = 1/2 − $\\lambda_\\rho$)"); a.set_ylabel('covariance term')
a.set_title('(1) covariance term vs felt curvature', fontsize=9); a.legend(fontsize=7.5, frameon=False, loc='upper left')
a = ax[1]
sc = a.scatter(dist, exc, c=a2, cmap='RdBu_r', norm=TwoSlopeNorm(0, -0.5, 0.5), s=[36 if t == 'c1' else 6 for t in tags],
               edgecolor=[INK if t == 'c1' else 'none' for t in tags], linewidth=0.6)
xx = np.linspace(0, dist.max(), 10); a.plot(xx, xx, '-', color=C[1], lw=2, label='excess = κ − law (R = κ)')
a.set_xlabel('distance from the peak  κ − law term'); a.set_ylabel('excess = covariance + D')
a.set_title('(2) excess vs distance from the peak (colour: felt curvature)', fontsize=9); a.legend(fontsize=8, frameon=False, loc='upper left')
fig.colorbar(sc, ax=a, fraction=0.046, label="$E_\\rho[a'']$")
a = ax[2]
a.scatter(a2, ratio, c=[C[['random', 'm1', 'c1'].index(t)] for t in tags], s=[36 if t == 'c1' else 6 for t in tags],
          edgecolor=[INK if t == 'c1' else 'none' for t in tags], linewidth=0.6)
a.axhline(1, color=C[1], lw=2); a.axhline(0, color=INK, lw=0.8); a.axvline(0, color=INK, lw=0.8)
a.set_xlabel("$E_\\rho[a'']$"); a.set_ylabel('excess / (κ − law)  (Conjecture 1 ⟺ ≤ 1)')
a.set_title('(3) how much of the room is used, vs felt curvature', fontsize=9); a.set_ylim(min(-0.5, ratio.min() * 1.05), 1.1)
a = ax[3]
a.plot(cneg, cpos, 'o', ms=3, color=C[0], alpha=0.5)
lim = max(np.abs(cpos).max(), np.abs(cneg).max()) * 1.05
a.plot([-lim, lim], [lim, -lim], ':', color=MUTED, lw=1, label='cov = 0'); a.axhline(0, color=INK, lw=0.8); a.axvline(0, color=INK, lw=0.8)
a.set_xlabel("from points with E[a''|y] ≤ 0 (near the stretch peak)"); a.set_ylabel("from points with E[a''|y] > 0")
a.set_title('(4) where the covariance term comes from', fontsize=9); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageC5_curvature.png', dpi=150); plt.close(fig)

# ---------------- figure: balance bound
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
a = ax[0]
a.plot(B + D, R, 'o', ms=3, color=C[0], alpha=0.5, label='pendulum states')
lo_ = min((B + D).min(), R.min()); hi_ = max((B + D).max(), 1.2)
a.plot([lo_, hi_], [lo_, hi_], '-', color=INK, lw=1, label='R = B + D'); a.axhline(1, color=C[1], lw=1.5, label='κ = 1'); a.axvline(1, color=C[1], lw=1.5, ls='--')
a.set_xlabel('balance bound  B + D,  B = E[|a| tr|H|]/E[tr H]'); a.set_ylabel('R'); a.legend(fontsize=8, frameon=False)
a.set_title('(1) R ≤ B + D holds; B + D > κ in some states', fontsize=9)
a = ax[1]
a.plot(val, B - f, 'o', ms=3, color=C[2], alpha=0.5)
a.set_xlabel('valley mass'); a.set_ylabel('looseness B − f'); a.set_title('(2) looseness is not mainly from the valley', fontsize=9)
a = ax[2]
a.plot(Bk, B, 'o', ms=3, color=C[3], alpha=0.5); a.axhline(1, color=C[1], lw=1.5)
a.set_xlabel('E tr|H| / E tr H  (1 = no valley)'); a.set_ylabel('B'); a.set_title('(3) B exceeds κ only with valleys', fontsize=9)
fig.tight_layout(); fig.savefig('figures/stageC5_balance.png', dpi=150); plt.close(fig)
print(json.dumps(out, indent=1, default=float))
