"""Stage F analysis: numbers -> results/summary_F.json; figures stageF_chains.png, stageF_rcvo.png, stageF_map.png"""
import json, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from obs_eval2 import evaluate_full

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
F = json.load(open('results/f1_chains.json'))
best = max(F, key=lambda r: r['final']['ratio'])
out = {'chains': [{'name': r['name'], 'K': r['K'], 'step0': r['step0'], 'start': r['track'][0]['ratio'], **r['final'], 'sigma': r['sigma'],
                   'R_per_sd20_minus_10': r['R_per_sd20'] - r['R_per_sd10'], 'mass': r['mass']} for r in F]}
out['max_ratio'] = best['final']['ratio']; out['max_R'] = max(r['final']['R'] for r in F); out['best'] = {'name': best['name'], **best['final'], 'sigma': best['sigma']}
out['any_ratio_gt_1'] = bool(any(r['final']['ratio'] > 1 for r in F))
out['resolution'] = max(abs(r['R_per_sd20'] - r['R_per_sd10']) for r in F)
# plateau estimate for every chain: ratio(step) = a - b * step^(-c) fitted on steps >= 400
def model(s, a, b, c): return a - b * s ** (-c)
plats = {}
for r in F:
    s = np.array([t['step'] for t in r['track']], float); y = np.array([t['ratio'] for t in r['track']]); k = s >= 400
    try:
        p, cov = curve_fit(model, s[k], y[k], p0=[y[-1] + 0.01, 1.0, 0.5], bounds=([y[-1], 0, 0.05], [2.0, 1e3, 5.0]), maxfev=20000)
        plats[r['name']] = {'a': float(p[0]), 'a_err': float(np.sqrt(cov[0, 0])), 'c': float(p[2]), 'last': float(y[-1]),
                            'gain_last_400': float(y[-1] - y[s <= s[-1] - 400][-1])}
    except Exception as e:
        plats[r['name']] = {'error': str(e), 'last': float(y[-1])}
out['plateau'] = plats
# which factor grows along the best chain: relative change from step 0 to the end
t0, t1 = best['track'][0], best['track'][-1]
out['best_factor_change'] = {k: {'start': t0[k], 'end': t1[k], 'rel': (t1[k] - t0[k]) / abs(t0[k]) if t0[k] != 0 else None}
                             for k in ['ratio', 'dist', 'rho', 'CV', 'Omega', 'rhoCVOmega', 'law_gap', 'one_minus_lam', 'D', 'valley']}
# required rho*CV*Omega for fraction 1 at the best state: 1 + (gap - D)/(1 - lambda_rho)
bf = best['final']; out['best_required_rhoCVOmega'] = 1 + (bf['law_gap'] - bf['D']) / bf['one_minus_lam']
if os.path.exists('results/f4_rcvo.json'):
    G = json.load(open('results/f4_rcvo.json'))
    out['rcvo'] = [{'name': r['name'], 'eps': r['eps'], **r['final'], 'R_per_sd20': r['R_per_sd20'], 'mass': r['mass'],
                    'required': 1 + (r['final']['law_gap'] - r['final']['D']) / r['final']['one_minus_lam']} for r in G]
    out['rcvo_max'] = max(r['final']['rhoCVOmega'] for r in G)
    out['rcvo_plateau'] = {}
    for r in G:
        s = np.array([t['step'] for t in r['track']], float); y = np.array([t['rhoCVOmega'] for t in r['track']]); ok = np.array([t['value'] > 0 for t in r['track']])
        k = (s >= 300) & ok
        try:
            p, cov = curve_fit(model, s[k], y[k], p0=[y[-1] + 0.01, 1.0, 0.5], bounds=([y[k][-1], 0, 0.05], [3.0, 1e3, 5.0]), maxfev=20000)
            out['rcvo_plateau'][r['name']] = {'a': float(p[0]), 'a_err': float(np.sqrt(cov[0, 0])), 'last': float(y[-1])}
        except Exception as e:
            out['rcvo_plateau'][r['name']] = {'error': str(e), 'last': float(y[-1])}
# needle sequence (f5_needle.py): shrink the needle and the window together
if os.path.exists('results/f5_needle.json'):
    N = json.load(open('results/f5_needle.json')); rows = N['rows']
    t = np.array([r['t'] for r in rows]); om = 1 - np.array([r['fraction'] for r in rows])
    out['needle'] = {'check_t1': N['check_t1'], 'rows': [{k: r[k] for k in ['t', 'sigma', 'R', 'R_refined', 'R_needle_alone', 'lam_needle', 'fraction', 'fisher_share_needle', 'mass']}
                                                         | {'rhoCVOmega': r['factors']['rhoCVOmega'], 'law_gap': r['factors']['law_gap']} for r in rows],
                     'all_R_lt_Rneedle_lt_lamneedle_lt_1': bool(all(r['R'] < r['R_needle_alone'] < r['lam_needle'] < 1 for r in rows)),
                     'all_fraction_lt_1': bool(all(r['fraction'] < 1 for r in rows)),
                     'slope_one_minus_fraction_vs_t': float(np.polyfit(np.log(t), np.log(om), 1)[0]),
                     'max_refine_diff': float(max(abs(r['R_refined'] - r['R']) for r in rows)), 'max_mass_dev': float(max(abs(r['mass'] - 1) for r in rows)),
                     'max_rhoCVOmega': float(max(r['factors']['rhoCVOmega'] for r in rows))}
json.dump(out, open('results/summary_F.json', 'w'), indent=1, default=float)

# ---------------- figures
if os.path.exists('results/f5_needle.json'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    a = ax[0]
    a.loglog(t, om, 'o-', color=C[0], label='1 − fraction')
    a.loglog(t, [1 - r['R'] for r in rows], 's-', color=C[1], label='1 − R (mixture)')
    a.loglog(t, [1 - r['R_needle_alone'] for r in rows], '^--', color=C[2], label='1 − R of the needle alone')
    a.loglog(t, [1 - r['lam_needle'] for r in rows], 'v:', color=C[3], label='1 − λ of the needle')
    a.set_xlabel('t (needle width along V ∝ t, length along W ∝ √t, window σ ∝ t)'); a.set_ylabel('distance to 1')
    a.set_title(f"(1) shrinking the needle: fraction → 1 from below (slope {out['needle']['slope_one_minus_fraction_vs_t']:.2f})", fontsize=9)
    a.legend(fontsize=8, frameon=False)
    a = ax[1]
    a.semilogx(t, [r['R_needle_alone'] - r['R'] for r in rows], 'o-', color=C[1], label='R(needle alone) − R(mixture)')
    a.semilogx(t, [r['lam_needle'] - r['R_needle_alone'] for r in rows], 's-', color=C[3], label='λ(needle) − R(needle alone)')
    a.axhline(0, color=INK, lw=1); a.set_yscale('symlog', linthresh=1e-6); a.set_xlabel('t'); a.legend(fontsize=8, frameon=False)
    a.set_title('(2) R stays below the needle\'s own R, which stays below its λ (< 1)', fontsize=9)
    fig.tight_layout(); fig.savefig('figures/stageF_needle.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for r, col in zip(F, C):
    s = [t['step'] for t in r['track']]
    ax[0].plot(s, [t['ratio'] for t in r['track']], '-', color=col, lw=1.6, label=r['name'])
    ax[1].plot(s, [t['dist'] for t in r['track']], '-', color=col, lw=1.6)
ax[0].axhline(1, color=INK, lw=1); ax[0].set_xlabel('step'); ax[0].set_ylabel('excess / distance'); ax[0].legend(fontsize=7, frameon=False, loc='lower right')
ax[0].set_title(f"(1) fraction of the room used (best {out['max_ratio']:.4f})", fontsize=9)
ax[1].axhline(0.3, color=MUTED, ls=':', lw=1); ax[1].set_xlabel('step'); ax[1].set_ylabel('distance κ − law'); ax[1].set_title('(2) distance along the chains (old upper edge 0.3 dotted)', fontsize=9)
a = ax[2]; s = np.array([t['step'] for t in best['track']])
for k, col, lab in [('rho', C[0], 'ρ'), ('CV', C[1], 'CV'), ('Omega', C[2], 'Ω'), ('rhoCVOmega', INK, 'ρ·CV·Ω'), ('one_minus_lam', C[3], '1 − λ_ρ'), ('law_gap', C[4], 'λ_ρ − law')]:
    v = np.array([t[k] for t in best['track']]); a.plot(s, v / v[0], '-', color=col, lw=2 if k == 'rhoCVOmega' else 1.4, label=lab)
a.axhline(1, color=MUTED, lw=0.8); a.set_xlabel('step'); a.set_ylabel('value / value at step 0'); a.legend(fontsize=8, frameon=False)
a.set_title(f"(3) factors along the best chain ({best['name']})", fontsize=9)
fig.tight_layout(); fig.savefig('figures/stageF_chains.png', dpi=150); plt.close(fig)

if os.path.exists('results/f4_rcvo.json'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for r, col in zip(G, C):
        tr = [t for t in r['track'] if t['value'] > 0]
        ax[0].plot([t['step'] for t in tr], [t['rhoCVOmega'] for t in tr], '-', color=col, lw=1.5, label=r['name'])
        ax[1].plot([t['step'] for t in tr], [t['ratio'] for t in tr], '-', color=col, lw=1.5)
    ax[0].axhline(1, color=INK, lw=1); ax[0].set_xlabel('step'); ax[0].set_ylabel('ρ·CV·Ω'); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title('(1) ρ·CV·Ω with the law term held at λ_ρ (gap ≤ ε)', fontsize=9)
    ax[1].axhline(1, color=INK, lw=1); ax[1].set_xlabel('step'); ax[1].set_ylabel('excess / distance'); ax[1].set_title('(2) fraction along the same chains', fontsize=9)
    fig.tight_layout(); fig.savefig('figures/stageF_rcvo.png', dpi=150); plt.close(fig)

# map of the best state on the V/W grid
r = evaluate_full(best['ws'], [np.array(m) for m in best['mus']], [np.array(S) for S in best['Ss']], best['sigma'], grid='vw', per_sd=10, return_fields=True)
Fd = r['fields']; Vv, Ww = np.meshgrid(Fd['q'], Fd['p'], indexing='ij'); pY = np.nan_to_num(Fd['pY']); m = pY > pY.max() * 1e-4
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
lev = pY.max() * np.array([0.01, 0.05, 0.2, 0.5, 0.8])
win = m.any(1); winw = m.any(0)
ext = lambda a: a.set(xlim=(Fd['q'][win].min(), Fd['q'][win].max()), ylim=(Fd['p'][winw].min(), Fd['p'][winw].max()))
a = ax[0]; im = a.pcolormesh(Vv, Ww, np.where(m, Fd['a_eff'], np.nan), cmap='Blues', vmin=0, vmax=1, shading='auto')
a.contour(Vv, Ww, pY, levels=lev, colors=INK, linewidths=0.6); ext(a); fig.colorbar(im, ax=a, fraction=0.046)
a.set_title('local stretch E[a|y] + density (lines)', fontsize=9); a.set_xlabel('coordinate along V (stretching)'); a.set_ylabel('coordinate along W (contracting)')
a = ax[1]; fa = np.where(m, Fd['focus_aniso'], np.nan)
im = a.pcolormesh(Vv, Ww, np.clip(fa, -1.5, 1.5), cmap='PuOr_r', norm=TwoSlopeNorm(0, -1.5, 1.5), shading='auto')
a.contourf(Vv, Ww, np.where(m, Fd['nonPD'], 0), levels=[0.5, 1.5], colors='none', hatches=['////'])
a.contour(Vv, Ww, np.where(m, Fd['nonPD'], 0), levels=[0.5], colors=INK, linewidths=1); ext(a); fig.colorbar(im, ax=a, fraction=0.046)
a.set_title('local focus anisotropy (hatched: valley)', fontsize=9); a.set_xlabel('coordinate along V')
a = ax[2]; cd = np.where(m, Fd['cov_density'], np.nan); vmax = np.nanmax(np.abs(cd))
im = a.pcolormesh(Vv, Ww, cd, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), shading='auto')
a.contour(Vv, Ww, pY, levels=lev, colors=MUTED, linewidths=0.5); ext(a); fig.colorbar(im, ax=a, fraction=0.046)
a.set_title(f"covariance-term density (red adds to R); R = {r['R']:.5f}", fontsize=9); a.set_xlabel('coordinate along V')
fig.suptitle(f"best state of stage F: fraction {best['final']['ratio']:.4f}, distance {best['final']['dist']:.3f}, σ = {best['sigma']:.4g}, K = {best['K']}", fontsize=10)
fig.tight_layout(); fig.savefig('figures/stageF_map.png', dpi=140); plt.close(fig)
out['map_R'] = float(r['R'])
json.dump(out, open('results/summary_F.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ['chains', 'rcvo']}, indent=1, default=float))
