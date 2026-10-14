"""Figures for stage C proposals 2 and 4: figures/stageC_D.png, figures/stageC_otherflows.png
(stageC_adversarial.png comes from c1_report.py, stageC_spatial.png from c3_spatial.py)"""
import json, numpy as np, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from obs_eval2 import FLOWS

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ---------------- D
D = json.load(open('results/c2_D.json'))
d = np.array([x['D'] for x in D]); dw = np.array([x['D_within'] for x in D]); db = np.array([x['D_between'] for x in D])
SD = np.array([x['S_D'] for x in D]); rv = np.array([x['resp_var'] for x in D]); gap = np.array([x['stretch_gap'] for x in D])
sig = np.array([x['sigma'] for x in D]); k = np.abs(d) > 1e-8
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
a = ax[0]; a.loglog([1e-9, 1], [1e-9, 1], '-', color=INK, lw=1.2, label='equality')
for s_, col in zip([0.03, 0.1, 0.3, 1.0], C):
    kk = k & (sig == s_); a.loglog(SD[kk], np.abs(d[kk]), 'o', ms=2.5, color=col, alpha=0.6, label=f'$\\sigma$={s_}')
a.set_xlabel('$S_D$ = (spread of the posterior guesses) × curvature / ($\\sigma^4$ tr F)'); a.set_ylabel('|D|')
a.set_title('D against the guess-spread scale $S_D$', fontsize=9); a.legend(fontsize=8, frameon=False); a.set_xlim(1e-9, 1); a.set_ylim(1e-9, 1)
a = ax[1]
for s_, col in zip([0.03, 0.1, 0.3, 1.0], C):
    kk = k & (sig == s_); a.loglog((rv * gap)[kk] + 1e-12, np.abs(d[kk]), 'o', ms=2.5, color=col, alpha=0.6, label=f'$\\sigma$={s_}')
a.set_xlabel('responsibility variance × stretch gap between components'); a.set_ylabel('|D|')
a.set_title('D against the simple product (proposal 2)', fontsize=9); a.legend(fontsize=8, frameon=False); a.set_ylim(1e-9, 1)
a = ax[2]
lim = 0.08
for s_, col in zip([0.03, 0.1, 0.3, 1.0], C):
    kk = k & (sig == s_); a.plot(db[kk], dw[kk], 'o', ms=2.5, color=col, alpha=0.6, label=f'$\\sigma$={s_}')
a.axhline(0, color=MUTED, lw=0.8); a.axvline(0, color=MUTED, lw=0.8)
a.set_xlabel('$D_{between}$ (flow bends between the guesses)'); a.set_ylabel('$D_{within}$ (guesses of different width)')
a.set_title('Exact split D = $D_{within}$ + $D_{between}$', fontsize=9); a.legend(fontsize=8, frameon=False)
a.set_xlim(-lim, lim); a.set_ylim(-lim / 2, lim / 2)
fig.tight_layout(); fig.savefig('figures/stageC_D.png', dpi=150); plt.close(fig)

# ---------------- other flows
S = json.load(open('results/c4_single.json')); Fm = json.load(open('results/c4_family.json')); M = json.load(open('results/c4_mix.json'))
fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
a = ax[0]
for kd, col in [('quartic', C[0]), ('double_well', C[1])]:
    law, rem, pred = [], [], []
    for x in S:
        if x['kind'] != kd: continue
        mu = np.array(x['mus'][0]); Sx = np.array(x['Ss'][0]); F = np.linalg.inv(Sx + x['sigma'] ** 2 * np.eye(2)); K = Sx @ F
        law.append((1 + FLOWS[kd][1](mu[0], Sx[0, 0])) / 2 * 2 * F[0, 1] / np.trace(F))
        rem.append(x['R'] - x['e']); pred.append(6.0 * K[0, 0] * K[0, 1] / np.trace(F))
    a.plot(law, [x['R'] for x in S if x['kind'] == kd], 'o', ms=3, color=col, alpha=0.7, label=f'{kd}: R vs law')
    ax[1].plot(pred, rem, 'o', ms=3, color=col, alpha=0.7, label=kd)
a.plot([-3, 3], [-3, 3], '-', color=INK, lw=1); a.set_xlabel('Gaussian law tr(F Sym $\\bar J$)/tr F'); a.set_ylabel('R (exact)')
a.set_title('single Gaussians: law 1 holds for both flows', fontsize=9); a.legend(fontsize=8, frameon=False)
a = ax[1]; a.plot([-1, 1], [-1, 1], '-', color=INK, lw=1)
a.set_xlabel('$-E_\\rho[g\'\'\'(q)]\\,K_{qq}K_{qp}$/tr F  (here $g\'\'\'=-6$)'); a.set_ylabel('R − e')
a.set_title('single Gaussians: the remainder formula generalises', fontsize=9); a.legend(fontsize=8, frameon=False)
lo = np.array([min(x['R'] - x['e'] for x in S), max(x['R'] - x['e'] for x in S)]); a.set_xlim(lo * 1.1); a.set_ylim(lo * 1.1)
a = ax[2]
m1 = collections.defaultdict(list)
for x in json.load(open('results/m1_slices.json')): m1[x['tag']].append(x)
ratios = [1.0, 2.0, 0.4 / 0.09, 8.0]
pc = [min(m1[t], key=lambda x: x['params']['delta']) for t in ['ratio_1.0', 'ratio_2.0', 'base_sepW', 'ratio_8.0']]
xx = np.arange(4); wdt = 0.25
a.bar(xx - wdt, [x['f'] - x['law_term'] for x in pc], wdt, color=C[2], label='pendulum, centre q=0 ($a\'\'>0$), $l_1$=1.9', edgecolor='white')
for j, (l1, col) in enumerate([(0.5, C[0]), (1.0, C[4])]):
    v = [next(x for x in Fm if x['kind'] == 'quartic' and abs(x['ratio'] - r) < 1e-9 and x['l1'] == l1) for r in ratios]
    a.bar(xx + j * wdt, [x['f'] - x['law_term'] for x in v], wdt, color=col, label=f'quartic q=0 / double well q=1 ($a\'\'<0$), $l_1$={l1}', edgecolor='white')
a.axhline(0, color=INK, lw=0.8); a.set_xticks(xx); a.set_xticklabels(['1', '2', '4.4', '8']); a.set_xlabel('width ratio $n_2/n_1$ (concentric)')
a.set_ylabel('covariance term'); a.set_title('the sign of the covariance term follows the curvature of the stretch', fontsize=9)
a.legend(fontsize=7, frameon=False, loc='lower left')
a = ax[3]
for kd, col in [('quartic', C[0]), ('double_well', C[1])]:
    v = [x for x in M if x['kind'] == kd]
    a.plot([x['lam_comp'] for x in v], [x['R'] for x in v], 'o', ms=2.5, color=col, alpha=0.5, label=f'{kd} mixtures')
xm = max(x['lam_comp'] for x in M) * 1.05; a.plot([0, xm], [0, xm], '-', color=INK, lw=1, label='R = $\\lambda_{comp}$'); a.set_xlim(0, xm)
a.set_xlabel('$\\lambda_{comp}$'); a.set_ylabel('R'); a.set_title('random mixtures: no case above $\\lambda_{comp}$ (800 cases)', fontsize=9)
a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageC_otherflows.png', dpi=150); plt.close(fig)
print('saved')
