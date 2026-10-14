"""Figures for stage B: figures/stageB_m1_slices.png, figures/stageB_m1_maps.png, figures/stageB_m2_remainder.png"""
import json, numpy as np, os, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
os.makedirs('figures', exist_ok=True)
SIG = 0.1
S = json.load(open('results/m1_slices.json'))
g = collections.defaultdict(list)
for d in S: g[d['tag']].append(d)
for v in g.values(): v.sort(key=lambda d: d['params']['delta'])
x = lambda v: np.array([d['params']['delta'] / SIG for d in v])
y = lambda v, k: np.array([d[k] for d in v])

# ---------------- Figure B1: slices vs delta/sigma
fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))
for a, tag, title in [(ax[0], 'base_sepW', 'base: separated along the contracting direction W'),
                      (ax[1], 'base_sepV', 'base: separated along the stretching direction V')]:
    v = g[tag]
    a.plot(x(v), y(v, 'R'), '-', color=INK, lw=2.4, label='R (exact)', zorder=5)
    a.plot(x(v), y(v, 'lam_comp'), '--', color=C[0], lw=1.6, label='$\\lambda_{comp}$ (largest component stretch)')
    a.plot(x(v), y(v, 'R_sep'), '-.', color=C[1], lw=1.6, label='$R_{sep}$ (component-wise law)')
    a.plot(x(v), y(v, 'e'), ls=(0, (4, 1, 1, 1)), color=C[3], lw=1.6, label='(e)')
    a.plot(x(v), y(v, 'f'), 'o', ms=4, mfc='none', color=C[2], label='(f) local-Fisher weighted stretch')
    a.plot(x(v), y(v, 'law_term'), ':', color=C[4], lw=2, label='law term tr(F Sym $E_\\rho Df$)/tr F')
    a.set_title(title, fontsize=9); a.set_xlabel('separation $\\delta/\\sigma$')
ax[0].set_ylabel('critical rate'); ax[0].legend(fontsize=7, frameon=False, loc='upper left')
for a, tags, labels, title in [
        (ax[2], ['ratio_1.0', 'ratio_2.0', 'base_sepW', 'ratio_8.0'], ['1', '2', '4.4 (base)', '8'], 'width ratio $n_2/n_1$ (V direction)'),
        (ax[3], ['base_sepW', 'tilt_22.5', 'tilt_45.0', 'tilt_90.0'], ['0° (base)', '22.5°', '45°', '90°'], 'tilt of component 2')]:
    for tag, lab, col, ls in zip(tags, labels, C, ['-', '--', '-.', ':']):
        v = g[tag]; a.plot(x(v), y(v, 'R') / y(v, 'lam_comp'), ls=ls, color=col, lw=2, label=lab)
    a.axhline(1, color=INK, lw=1)
    a.set_title('R / $\\lambda_{comp}$ — ' + title, fontsize=9); a.set_xlabel('separation $\\delta/\\sigma$ (along W)')
    a.legend(fontsize=8, frameon=False, title=title.split(' (')[0], title_fontsize=8)
fig.tight_layout(); fig.savefig('figures/stageB_m1_slices.png', dpi=150); plt.close(fig)

# ---------------- Figure B2: maps + decomposition bars
M = json.load(open('results/m1_maps.json'))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for a, tag, key, scale, ylab in [(ax[0], 'map_ratio', 'ratio', 1.0, 'width ratio $n_2/n_1$'),
                                 (ax[1], 'map_elong', 'l1', 0.09, 'elongation of component 1, $l_1/n_1$')]:
    v = [d for d in M if d['tag'] == tag]
    xs = sorted(set(round(d['params']['delta'], 9) for d in v)); ys = sorted(set(round(d['params'][key], 9) for d in v))
    Z = np.full((len(ys), len(xs)), np.nan); OV = Z.copy()
    for d in v:
        i, j = ys.index(round(d['params'][key], 9)), xs.index(round(d['params']['delta'], 9))
        Z[i, j] = d['R'] / d['lam_comp']; OV[i, j] = d['overlap']
    X = np.array(xs) / SIG; Yv = np.array(ys) / scale
    pc = a.pcolormesh(X, Yv, np.clip(Z, 0.5, 1.15), cmap='RdBu_r', norm=TwoSlopeNorm(1.0, 0.5, 1.15), shading='nearest')
    cs = a.contour(X, Yv, Z, levels=[1.0], colors=INK, linewidths=2)
    co = a.contour(X, Yv, OV, levels=[0.1, 0.3, 0.6], colors=MUTED, linewidths=0.9, linestyles='--')
    a.clabel(co, fmt=lambda t: f'overlap {t:g}', fontsize=7)
    a.set_yscale('log'); a.set_xlabel('separation $\\delta/\\sigma$ (along W)'); a.set_ylabel(ylab); a.grid(False)
    a.set_title('R / $\\lambda_{comp}$ (solid line: = 1; red: R exceeds every component)', fontsize=9)
    fig.colorbar(pc, ax=a, fraction=0.046, pad=0.02, extend='min')
a = ax[2]
tags = ['ratio_1.0', 'ratio_2.0', 'base_sepW', 'ratio_8.0']; labs = ['1', '2', '4.4', '8']
v0 = [g[t][0] for t in tags]                                     # delta = 0 (concentric)
law = np.array([d['law_term'] for d in v0]); cov = np.array([d['f'] - d['law_term'] for d in v0]); D = np.array([d['D'] for d in v0])
xx = np.arange(len(tags))
a.bar(xx, law, 0.6, color=C[0], label='law term (≤ $\\lambda_\\rho$)', edgecolor='white', linewidth=2)
a.bar(xx, cov, 0.6, bottom=law, color=C[1], label='focus–stretch covariance', edgecolor='white', linewidth=2)
a.bar(xx, D, 0.6, bottom=law + cov, color=C[3], label='posterior Stein defect D', edgecolor='white', linewidth=2)
for i, d in enumerate(v0):
    a.plot([i - 0.38, i + 0.38], [d['lam_comp']] * 2, '-', color=INK, lw=2)
    a.plot(i, d['R'], 'D', color=INK, ms=5)
a.plot([], [], '-', color=INK, lw=2, label='$\\lambda_{comp}$'); a.plot([], [], 'D', color=INK, ms=5, label='R')
a.set_xticks(xx); a.set_xticklabels(labs); a.set_xlabel('width ratio $n_2/n_1$ (concentric, $\\delta = 0$)')
a.set_title('Exact split: R = law term + covariance + D', fontsize=9); a.legend(fontsize=7.5, frameon=False, loc='upper left')
a.set_ylim(0, 0.42)
fig.tight_layout(); fig.savefig('figures/stageB_m1_maps.png', dpi=150); plt.close(fig)

# ---------------- Figure B3: remainder R - e
SG = json.load(open('results/m2_single.json')); SW = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
Sm = {(rec['shape'], rec['q0']): np.array(rec['S']) for rec in SW}
rem = np.array([d['R'] - d['e'] for d in SG])
bound = np.array([d['sigma'] ** 2 * abs(np.cos(d['q0'])) * np.exp(-Sm[(d['shape'], d['q0'])][0, 0] / 2) / 2 for d in SG])
MX = json.load(open('results/m2_mix.json'))
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
a = ax[0]; k = bound > 1e-12
a.loglog([1e-9, 1], [1e-9, 1], '-', color=INK, lw=1.2, label='equality (bound)')
a.loglog(bound[k], np.abs(rem[k]) + 1e-16, 'o', ms=2.5, color=C[0], alpha=0.5, label=f'single Gaussians ({k.sum()} cases)')
a.set_xlabel('$\\sigma^2\\,|E_\\rho[a\'\'(q)]|$  (curvature of stretch × window$^2$)'); a.set_ylabel('|R − e|')
a.set_title('(1) single Gaussian: |R − e| ≤ $\\sigma^2|E_\\rho a\'\'|$ (exact formula)', fontsize=9)
a.set_xlim(1e-9, 1); a.set_ylim(1e-9, 1); a.legend(fontsize=8, frameon=False)
sig = np.array([d['sigma'] for d in MX]); ov = np.array([d['overlap'] for d in MX])
fe = np.array([d['f'] - d['e'] for d in MX]); Dm = np.array([d['D'] for d in MX])
curv = np.array([d['sigma'] ** 2 * d['K_cos'] for d in MX])
cov = np.array([d['f'] - d['law_term'] for d in MX]); als = np.array([d['H_alpha_std'] for d in MX])
a = ax[1]
a.loglog([1e-9, 1], [1e-9, 1], '-', color=INK, lw=1.2, label='equality')
for sel, col, lab in [(ov < 1e-3, C[0], 'overlap < 1e-3'), ((ov >= 1e-3) & (ov < 0.05), C[3], '1e-3 ≤ overlap < 0.05'), (ov >= 0.05, C[1], 'overlap ≥ 0.05')]:
    a.loglog(curv[sel], np.abs(fe[sel]) + 1e-16, 'o', ms=2.5, color=col, alpha=0.6, label=lab)
a.set_xlabel('$\\sigma^2 E_{p_Y}|a\'\'(q)|$'); a.set_ylabel('|f − e|  (stretch-gradient part)')
a.set_title('(2) mixtures: the bound holds only without overlap', fontsize=9); a.legend(fontsize=8, frameon=False)
a.set_xlim(1e-7, 1); a.set_ylim(1e-12, 1)
a = ax[2]
for s_, col in zip([0.03, 0.1, 0.3, 1.0], C):
    kk = sig == s_
    a.loglog(ov[kk] + 1e-12, np.abs(Dm[kk]) + 1e-16, 'o', ms=2.5, color=col, alpha=0.6, label=f'$\\sigma$={s_}')
a.set_xlabel('overlap of the two blurred components'); a.set_ylabel('|D| = |R − f|  (posterior Stein defect)')
a.set_title('(3) mixtures: D needs overlap, grows with $\\sigma$', fontsize=9); a.legend(fontsize=8, frameon=False)
a.set_xlim(1e-12, 2); a.set_ylim(1e-15, 1)
a = ax[3]
for s_, col in zip([0.03, 0.1, 0.3, 1.0], C):
    kk = sig == s_
    a.loglog(als[kk] + 1e-12, np.abs(cov[kk]) + 1e-16, 'o', ms=2.5, color=col, alpha=0.6, label=f'$\\sigma$={s_}')
a.set_xlabel('spread of local anisotropy  std$_{p_Y}[(v^THv - w^THw)/\\mathrm{tr}F]$'); a.set_ylabel('|f − law term|  (focus–stretch covariance)')
a.set_title('(4) where the anisotropy spread enters', fontsize=9); a.legend(fontsize=8, frameon=False)
a.set_xlim(1e-4, 10); a.set_ylim(1e-9, 1)
fig.tight_layout(); fig.savefig('figures/stageB_m2_remainder.png', dpi=150); plt.close(fig)
print('saved')
