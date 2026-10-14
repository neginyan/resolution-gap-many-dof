"""Figures for stage A. figures/stageA_single.png and figures/stageA_law.png"""
import json, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = {'b': '#2a78d6', 'c': '#eb6834', 'd': '#1baf7a', 'e': '#eda100', 'law': '#e87ba4'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
D = json.load(open('results/sweep.json')) + json.load(open('results/sweep_tilt.json'))
os.makedirs('figures', exist_ok=True)

def law(rec, sig):
    S = np.array(rec['S']); ec = np.cos(rec['q0']) * np.exp(-S[0, 0] / 2)
    SJ = np.array([[0, (1 - ec) / 2], [(1 - ec) / 2, 0]]); F = np.linalg.inv(S + sig ** 2 * np.eye(2))
    return np.trace(F @ SJ) / np.trace(F)

# ---- Figure 1: R and candidates vs sigma, sharp-along-stretching shape at three positions + one tilted shape
sel = [('sharp along stretching (0.05 x 0.5)', 1.0), ('sharp along stretching (0.05 x 0.5)', 0.5),
       ('sharp along stretching (0.05 x 0.5)', 0.0), ('tilted 22.5 deg (0.05 x 0.5)', 0.6)]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
for a, (sh, qp) in zip(ax, sel):
    rec = [r for r in D if r['shape'] == sh and abs(r['q0_over_pi'] - qp) < 1e-9][0]
    s = np.array([r['sigma'] for r in rec['rows']])
    a.plot(s, [r['R'] for r in rec['rows']], '-', color=INK, lw=2.4, label='R (exact)', zorder=5)
    a.axhline(1.0, color=MUTED, ls=(0, (6, 3)), lw=1.2, label='(a) global $\\kappa$')
    a.axhline(rec['b'], color=C['b'], ls='--', lw=1.6, label='(b) local stretch at centre')
    a.axhline(rec['c'], color=C['c'], ls='-.', lw=1.6, label='(c) Theorem 9 ($\\sigma\\to0$)')
    a.plot(s, [r['d'] for r in rec['rows']], ls=(0, (1, 1.5)), color=C['d'], lw=2, label='(d) window-averaged stretch')
    a.plot(s, [r['cand_e'] for r in rec['rows']], ls=(0, (4, 1, 1, 1)), color=C['e'], lw=1.6, label='(e) smoothed-score weighted')
    a.plot(s, [law(rec, x) for x in s], 'o', ms=3.5, color=C['law'], mfc='none', label='Gaussian law (Stein)')
    a.set_xscale('log'); a.set_xlabel('window $\\sigma$'); a.set_ylim(-0.05, 1.08)
    a.set_title(f"{sh.split(' (')[0]}, $q_0={qp}\\pi$", fontsize=9, color=INK)
ax[0].set_ylabel('critical rate')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('figures/stageA_single.png', dpi=150)

# ---- Figure 2: (left) collapse R/c for 45-degree shapes; (middle) R vs Gaussian law, all states; (right) mixtures
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
a = ax[0]
for rec in D:
    if rec['shape'].startswith('sharp along stretching'):
        s = [r['sigma'] for r in rec['rows']]
        a.plot(s, [r['R'] / rec['c'] for r in rec['rows']], '-', color=C['b'], lw=1, alpha=0.6)
s = np.logspace(-2, np.log10(2), 100); s1, s2 = 0.05 ** 2, 0.5 ** 2
a.plot(s, (s1 + s2) / (s1 + s2 + 2 * s ** 2), ':', color=INK, lw=2, label='$(s_1^2+s_2^2)/(s_1^2+s_2^2+2\\sigma^2)$')
a.plot([], [], '-', color=C['b'], label='R/(c), 11 positions $q_0=\\pi\\dots0$')
a.set_xscale('log'); a.set_xlabel('window $\\sigma$'); a.set_ylabel('R / (c)'); a.legend(fontsize=8, frameon=False)
a.set_title('Position drops out: R/(c) depends only on shape and $\\sigma$', fontsize=9)
a = ax[1]
Rs, Ls = [], []
for rec in D:
    for r in rec['rows']:
        Rs.append(r['R']); Ls.append(law(rec, r['sigma']))
a.plot([-1, 1], [-1, 1], '-', color=GRID, lw=3)
a.plot(Ls, Rs, 'o', ms=2.5, color=C['b'], alpha=0.5)
a.set_xlabel('tr(F Sym $\\bar J$)/tr F  (Gaussian law)'); a.set_ylabel('R (exact quadrature)')
a.set_title(f'Single Gaussians: {len(Rs)} cases, max |diff| = {max(abs(x - y) for x, y in zip(Rs, Ls)):.1e}', fontsize=9)
a = ax[2]
M = json.load(open('results/mixtures.json'))
ov = np.array([m['overlap'] for m in M]); dev = np.abs(np.array([m['R'] - m['R_sep'] for m in M]))
for sig, col in zip([0.03, 0.1, 0.3, 1.0], [C['b'], C['c'], C['d'], C['e']]):
    k = np.array([m['sigma'] == sig for m in M])
    a.plot(ov[k] + 1e-12, dev[k] + 1e-12, 'o', ms=3, color=col, alpha=0.6, label=f'$\\sigma$={sig}')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(1e-12, 2); a.set_ylim(1e-13, 1)
a.set_xlabel('overlap of the two blurred components (Bhattacharyya)'); a.set_ylabel('|R − R$_{sep}$|')
a.set_title('Two-component mixtures: deviation from the component-wise law', fontsize=9); a.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig('figures/stageA_law.png', dpi=150)
print('saved')
