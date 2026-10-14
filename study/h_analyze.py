"""Stage H analysis: numbers -> results/summary_H.json; figures stageH_rstar.png, stageH_violation.png."""
import json, numpy as np
from fractions import Fraction as Fr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from h1_rstar import rstar, gaussian, one_minus_R
from h2_adversarial import unpack
from obs_eval2 import V, W

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
L = lambda f: json.load(open(f'results/{f}.json'))
Rm = np.stack([V, W], 1)
out = {}

# ---------- task 1 and 3: R*(sigma) ----------
h1 = L('h1_rstar')
coef = [Fr(c) for c in h1['coef_tilted']]; coef_u = [Fr(c) for c in h1['coef_untilted']]
table = []
for s in [1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    t = rstar(s); a = rstar(s, tilt=False); S = Rm.T @ gaussian(s, t['u'], t['p']) @ Rm; ev, U = np.linalg.eigh(S); u = U[:, 1] * np.sign(U[1, 1])
    part = lambda cs, m: sum(float(c) * s ** k for k, c in enumerate(cs[:m + 1]))
    table.append({'sigma': s, 'Rstar': t['Rstar'], 'Rstar_untilted': a['Rstar'], 'tilt_gain': t['Rstar'] - a['Rstar'],
                  'one_minus_Rstar_over_sigma': t['one_minus_Rstar'] / s, 'u': t['u'], 'p': t['p'],
                  'length_W': float(np.sqrt(ev[1])), 'length_over_2sqrt_sigma': float(np.sqrt(ev[1]) / (2 * np.sqrt(s))),
                  'tilt_angle': float(np.arctan2(u[0], u[1])), 'tilt_angle_over_sigma': float(np.arctan2(u[0], u[1]) / s),
                  'series_err': {m: abs(part(coef, m) - t['one_minus_Rstar']) for m in [1, 2, 3, 5, 7, 11]}})
out['rstar_table'] = table
out['coef_tilted'] = h1['coef_tilted']; out['coef_untilted'] = h1['coef_untilted']; out['u_series'] = h1['w_tilted']; out['sigma_c'] = h1['sigma_c']
out['coef_float'] = [float(c) for c in coef]; out['coef_ratio'] = [abs(float(coef[k] / coef[k + 1])) for k in range(1, len(coef) - 1)]
# stage-G NM table versus the true R*
g3 = L('g3_best_single'); dv = [rstar(float(k))['Rstar'] - v[0] for k, v in g3['Rstar_table'].items() if float(k) < 0.9]
out['stageG_NM_vs_true'] = {'max_abs': max(abs(x) for x in dv), 'n': len(dv)}

# ---------- task 2: violations ----------
h2 = L('h2_adversarial'); out['h2'] = [{k: r[k] for k in ['K', 'sigma', 'gstar', 'R', 'Rstar', 'margin']} | {'R16_minus_R10': r['R_per_sd16'] - r['R'], 'mass': r['mass']} for r in h2]
try:
    br = L('h2_brute'); out['brute'] = br
except FileNotFoundError:
    pass
h3 = L('h3_xsearch'); best = max(h3, key=lambda r: r['gstar'])
out['h3'] = [{'K': r['K'], 'seed': r['seed'], 'gstar': r['gstar'], 'scale': r['scale']} for r in h3]
out['h3_best'] = {'K': best['K'], 'seed': best['seed'], 'gstar': best['gstar'], 'scale': best['scale']}
out['max_gstar_all'] = max([r['gstar'] for r in h2] + [r['gstar'] for r in h3])
small = [v for r in h3 for v in r['scale'] if v['sigma'] <= 1e-3]
out['min_one_minus_R_over_sigma_small'] = min(v['one_minus_R_over_sigma'] for v in small)
# structure of the best state
from g1_two_needles import law1
from obs_eval2 import evaluate_full
ws, mus, Ss = unpack(np.array(best['x']), best['K'], 1e-3); s = 1e-3; Rs = rstar(s)['Rstar']
r = evaluate_full(ws, mus, Ss, s, grid='vw', per_sd=10)
comps = []
for w, m, S in zip(ws, mus, Ss):
    Svw = Rm.T @ S @ Rm; ev, U = np.linalg.eigh(Svw); u = U[:, 1] * np.sign(U[1, 1]); Rk, lk = law1(np.array(m), S, s)
    F = np.linalg.inv(S + s ** 2 * np.eye(2))
    comps.append({'weight': w, 'offset_V_over_sigma': float(V @ (m - [np.pi, 0]) / s), 'offset_W_over_2sqrt': float(W @ (m - [np.pi, 0]) / (2 * np.sqrt(s))),
                  'tilt_over_sigma': float(np.arctan2(u[0], u[1]) / s), 'length_over_2sqrt': float(np.sqrt(ev[1]) / (2 * np.sqrt(s))),
                  'thickness_over_sigma': float(np.sqrt(max(ev[0], 0)) / s), 'R_k': Rk, 'gstar_k': (Rk - Rs) / (1 - Rs), 'lam_k': lk, 'w_trF': float(w * np.trace(F))})
tot = sum(c['w_trF'] for c in comps)
for c in comps: c['fisher_share'] = c['w_trF'] / tot
out['best_structure'] = {'sigma': s, 'R': r['R'], 'Rstar': Rs, 'law': r['law_term'], 'cov': r['f'] - r['law_term'], 'D': r['D'], 'lam_rho': r['lam_rho_direct'], 'components': comps}

# ---------- figures ----------
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
a = ax[0]; ss = np.logspace(-4, np.log10(0.92), 120)
a.semilogx(ss, [rstar(x)['one_minus_Rstar'] / x for x in ss], '-', color=C[0], lw=2, label='true R* (tilted line segment)')
a.semilogx(ss, [rstar(x, tilt=False)['one_minus_Rstar'] / x for x in ss], '--', color=C[1], label='untilted needle (stage G closed form)')
for m, col in [(2, C[2]), (3, C[3]), (7, C[4])]:
    a.semilogx(ss, [sum(float(c) * x ** k for k, c in enumerate(coef[:m + 1])) / x for x in ss], ':', color=col, lw=1.4, label=f'series to σ^{m}')
a.set_ylim(0.4, 1.05); a.set_xlabel('window σ'); a.set_ylabel('(1 − R*) / σ'); a.legend(fontsize=7, frameon=False)
a.set_title('(1) the gap of the best single Gaussian, divided by σ', fontsize=9)
a = ax[1]
for m, col in zip([1, 2, 3, 5, 7, 11], C):
    a.loglog(ss[ss < 0.6], [abs(sum(float(c) * x ** k for k, c in enumerate(coef[:m + 1])) - rstar(x)['one_minus_Rstar']) + 1e-19 for x in ss[ss < 0.6]], '-', color=col, label=f'to σ^{m}')
a.set_ylim(1e-18, 1); a.set_xlabel('window σ'); a.set_ylabel('|series − exact|'); a.legend(fontsize=7, frameon=False, ncol=2)
a.set_title('(2) error of the exact series (slope = next order)', fontsize=9)
a = ax[2]; s2 = ss[ss < 0.9]; geo = []
for x in s2:
    t = rstar(x); S = Rm.T @ gaussian(x, t['u'], t['p']) @ Rm; ev, U = np.linalg.eigh(S); u = U[:, 1] * np.sign(U[1, 1])
    geo.append((np.arctan2(u[0], u[1]) / x, np.sqrt(ev[1]) / (2 * np.sqrt(x))))
a.semilogx(s2, [-g[0] for g in geo], '-', color=C[0], label='tilt from W towards −V, in units of σ  (→ 1/4)')
a.semilogx(s2, [g[1] for g in geo], '-', color=C[1], label='length along the segment / 2√σ  (→ 1)')
a.set_xlabel('window σ'); a.legend(fontsize=7, frameon=False); a.set_ylim(0, 1.3)
a.set_title('(3) the optimum: a zero-width segment through the top', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageH_rstar.png', dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
a = ax[0]
for i, K in enumerate([2, 3]):
    v = [r for r in h2 if r['K'] == K]
    a.semilogx([r['sigma'] for r in v], [r['gstar'] for r in v], 'o-', color=C[i], label=f'adversarial search, K = {K}')
for i, r in enumerate(h3):
    a.semilogx([v['sigma'] for v in r['scale']], [v['gstar'] for v in r['scale']], ':', color=C[2 + r['K'] - 2], lw=1, alpha=0.8)
a.semilogx([], [], ':', color=MUTED, label='long-search states rescaled to other σ (K = 2, 3, 4)')
a.axhline(0, color=INK, lw=1); a.set_xlabel('window σ'); a.set_ylabel('g* = (R − R*) / (1 − R*)'); a.legend(fontsize=7, frameon=False)
a.set_title('(1) mixtures beat the best single Gaussian by 1–4 % of its gap', fontsize=9)
a = ax[1]
# density of the best state in V/W coordinates (zoom)
s = 1e-3; vv = np.linspace(-6 * s, 6 * s, 300); wwg = np.linspace(-3 * np.sqrt(s) * 2, 3 * np.sqrt(s) * 2, 300)
A_, B_ = np.meshgrid(vv, wwg, indexing='ij'); Y = np.array([np.pi, 0]) + A_.reshape(-1, 1) * V + B_.reshape(-1, 1) * W
for k, (w, m, S) in enumerate(zip(ws, mus, Ss)):
    Sx = S + (0.03 * s) ** 2 * np.eye(2); d = Y - m; dens = w * np.exp(-0.5 * np.einsum('ni,ij,nj->n', d, np.linalg.inv(Sx), d)) / np.sqrt(np.linalg.det(Sx))
    a.contour(A_ / s, B_ / (2 * np.sqrt(s)), dens.reshape(A_.shape), levels=[dens.max() * f for f in [0.05, 0.3]], colors=[C[k]], linewidths=1.2)
    a.plot([], [], '-', color=C[k], label=f'component {k + 1}: weight {w:.2f}, tilt {comps[k]["tilt_over_sigma"]:+.2f}σ')
a.axvline(0, color=MUTED, lw=0.8, ls=':'); a.axhline(0, color=MUTED, lw=0.8, ls=':')
a.set_xlabel('V / σ (stretching)'); a.set_ylabel('W / 2√σ (contracting)'); a.legend(fontsize=6.5, frameon=True, framealpha=0.9, loc='lower right', bbox_to_anchor=(1.0, 0.0))
a.set_title(f"(2) best state, σ = 10⁻³ (g* = {best['gstar']:+.4f}); prior (unblurred)", fontsize=9)
a = ax[2]
sc = best['scale']
a.semilogx([v['sigma'] for v in sc], [v['one_minus_R_over_sigma'] for v in sc], 'o-', color=C[1], label='best mixture: (1 − R)/σ')
a.semilogx([v['sigma'] for v in sc], [(1 - v['Rstar']) / v['sigma'] for v in sc], 's-', color=C[0], label='best single Gaussian: (1 − R*)/σ')
a.set_xlabel('window σ (same shape, rescaled)'); a.set_ylabel('gap / σ'); a.legend(fontsize=7.5, frameon=False)
a.set_title('(3) the gap stays ∝ σ (R < 1 = κ throughout)', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageH_violation.png', dpi=150); plt.close()
try:
    h4 = L('h4_growk'); out['h4'] = [{'seed': c['seed'], 'by_K': [{k: x[k] for k in ['K', 'gstar10', 'gstar_sigma1e-4', 'R', 'R16']} for x in c['chain']]} for c in h4]
    allk = {}
    for c in h4:
        for x in c['chain']: allk[x['K']] = max(allk.get(x['K'], -1), x['gstar10'])
    for r in h2 + h3: allk[r['K']] = max(allk.get(r['K'], -1), r['gstar'] if r.get('sigma', 1e-3) == 1e-3 else -1)
    out['best_gstar_by_K_sigma1e-3'] = dict(sorted(allk.items()))
    fig, a = plt.subplots(figsize=(5.2, 3.6))
    for i, c in enumerate(h4):
        a.plot([x['K'] for x in c['chain']], [x['gstar10'] for x in c['chain']], 'o-', color=C[i], label=f'chain {i + 1} (components added one by one)')
    a.plot([2, 3, 4], [allk[2], allk[3], max(r['gstar'] for r in h3 if r['K'] == 4)], 's--', color=C[5], label='independent searches (h2, h3)')
    try:
        h5 = L('h5_fan'); a.plot([o['n'] for o in h5 if o['n'] > 1], [o['gstar'] for o in h5 if o['n'] > 1], 'D:', color=C[2], label='symmetric fan of n equal-length segments (5 parameters)')
    except FileNotFoundError:
        pass
    a.set_xscale('log', base=2); a.set_xticks([2, 3, 4, 5, 6, 8, 16, 32]); a.set_xticklabels(['2', '3', '4', '5', '6', '8', '16', '32'])
    a.set_xlabel('number of components K'); a.set_ylabel('g* = (R − R*) / (1 − R*)'); a.set_ylim(0, 0.07)
    a.set_title('σ = 10⁻³: the advantage over R* grows slowly with K', fontsize=9); a.legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig('figures/stageH_growk.png', dpi=150); plt.close()
except FileNotFoundError:
    print('h4_growk.json not found yet')
try:
    h5 = L('h5_fan'); out['h5_fan'] = [{k: o[k] for k in ['n', 'gstar', 'gstar_per_sd16', 'gstar_sigma1e-4', 'gstar_sigma1e-2', 'params', 'cov', 'D']} for o in h5]
except FileNotFoundError:
    pass
json.dump(out, open('results/summary_H.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ['max_gstar_all', 'min_one_minus_R_over_sigma_small', 'sigma_c', 'stageG_NM_vs_true', 'h3_best']}, indent=1, default=float)[:3000])
print(json.dumps(out['best_structure'], indent=1, default=float))
