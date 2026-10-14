"""Stage N analysis: results/summary_N.json and figures stageN_mix / stageN_decomp / stageN_P / stageN_tangent.
Fit / extrapolation / optimisation outputs carry 'opt', 'fit', 'extrap', 'coef' in their JSON paths (compared `loose`)."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
J = lambda f: json.load(open(f'results/{f}.json'))
have = lambda f: os.path.exists(f'results/{f}.json')
SIG = 1e-3; out = {}


def split(parts, k):
    """gap/sigma = (1 - wq) gp + wq gq: gp = plane part, gq = q- part, wq = J_b/(J2 + J_b)."""
    s2 = SIG ** 2; J2, Jb = parts['J2'], parts['Jb']
    gp = (k * s2 * J2 - parts['num2']) / (SIG ** 3 * J2)
    if Jb == 0: return {'gp': gp, 'gq': None, 'wq': 0.0}
    gq = (k * s2 * Jb - parts['extra']) / (SIG ** 3 * Jb); return {'gp': gp, 'gq': gq, 'wq': Jb / (J2 + Jb)}


# ---------------- N1 ----------------
from n_tools import Flow
KAP = {'1d': 1.0, '0.1': Flow(0.1).k, '0.3': Flow(0.3).k}
runs = [dict(o, mode='first') for o in J('n1_mix')] + ([o for o in J('n1_more')] if have('n1_more') else [])
best = {}
for o in runs:
    key = (o['kind'], o['K'])
    if key not in best or o['gap'] < best[key]['gap']: best[key] = o
G = {o['kind']: o for o in J('n1_gauss')} if have('n1_gauss') else {}
from m_tools import coupled, find_kappa, coefficient
share = {}
for e in ['0.1', '0.3']:
    s = coupled(float(e)); k, t = find_kappa(s, starts=50); c = coefficient(s, t[0], k); share[e] = c['diag_terms'][0] / c['coef']
rows = []
for (kind, K), o in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    r = {'kind': kind, 'K': K, 'opt_gap': o['gap'], 'opt_ratio': o['ratio'], 'gain': 1 - o['ratio'], 'from': o.get('mode', 'first'),
         's_b_over_sigma': o.get('s_b_over_sigma'), 'theta': o.get('theta')}
    rows.append(r)
N1 = {'rows': rows, 'plane_share': share, 'gauss_product': {k: {'gap': v['gap'], 'gap_reference': v['gap_reference'], 's_b_over_sigma': v['s_b_over_sigma'],
                                                                   'theta': v['theta']} for k, v in G.items()}}
# (the coordinate split (1 - w_q) g_p + w_q g_q of split() is ill-conditioned for theta != 0 -- large cancelling parts -- and is no longer
#  reported; the decomposition of N1(b) is the hybrid one of n1_decomp.py)
fitA = {}
for K in sorted({r['K'] for r in rows}):
    g1 = [r for r in rows if r['kind'] == '1d' and r['K'] == K]
    if not g1: continue
    for e in ['0.1', '0.3']:
        g3 = [r for r in rows if r['kind'] == e and r['K'] == K]
        if g3 and g3[0]['gain'] > 0 and g1[0]['gain'] > 0: fitA[f'{e}_K{K}'] = float(np.log(g3[0]['gain'] / g1[0]['gain']) / np.log(share[e]))
N1['fit_exponent_a_each'] = fitA
pairs = [(np.log(share[k.split('_')[0]]), np.log([r for r in rows if r['kind'] == k.split('_')[0] and r['K'] == int(k.split('K')[1])][0]['gain'] / [r for r in rows if r['kind'] == '1d' and r['K'] == int(k.split('K')[1])][0]['gain'])) for k in fitA]
if pairs:
    xs, ys = np.array(pairs).T; N1['fit_exponent_a_all'] = float(np.sum(xs * ys) / np.sum(xs * xs))
if have('n1_nonproduct'):
    N1['nonproduct'] = [{k: o[k] for k in ['kind', 'K', 'gap_product_R3', 'gap_nonproduct', 'gain_rel', 'iterations', 'product_components', 'nonproduct_components']} for o in J('n1_nonproduct')]
if have('n1_decomp'): N1['decomp'] = J('n1_decomp')
if have('n1_test'):
    P = J('n1_prediction'); N1['sealed_prediction'] = {'written_utc': P['written_utc'], 'sha256': open('logs/n1_prediction_hash.txt').read().split('=')[-1].strip().split()[0],
                                                       'a_fit': P['a_fit'], 'a_each': P['a_each'], 'gain_1d_K8': P['gain_1d_K8']}
    N1['test'] = [{k: o[k] for k in ['eps', 'runs', 'opt_ratio', 'predicted_ratio', 'predicted_range', 'effective_beta_ratio']} | {'gain_obs': 1 - o['opt_ratio'], 'gain_pred': 1 - o['predicted_ratio']} for o in J('n1_test')]
out['N1'] = N1

# ---------------- N2 ----------------
if have('n2_pmin'):
    P = J('n2_pmin')
    out['N2'] = {'runs': [{k: o.get(k) for k in ['K', 'sigma', 'ymax', 'start', 'P_start', 'P', 'P_fine', 'gap', 'gap_fine', 'gap_over_sqrtP', 'ThetaP', 'rho', 'd', 'y_over_sigma', 'iterations',
                                              'bound_over_sigma']} for o in P],
                 'opt_P_min_unrestricted': min(o['P_fine'] for o in P if o.get('ymax') is None),
                 'opt_gap_at_P_min_unrestricted': min([o for o in P if o.get('ymax') is None], key=lambda o: o['P_fine'])['gap_fine'],
                 'opt_P_min_restricted': {f"{y:g}": min(o['P_fine'] for o in P if o.get('ymax') == y) for y in sorted({o['ymax'] for o in P if o.get('ymax') is not None})},
                 'min_gap_over_sqrtP': min(o['gap_over_sqrtP'] for o in P)}
    if have('m4_dichotomy'):
        D = [o for o in J('m4_dichotomy') if 'error' not in o and o['sigma'] <= 1e-2]; out['N2']['stored_states_P_min'] = min(o['P'] for o in D)
        out['N2']['stored_min_gap_over_sqrtP'] = min(o['gap_over_sigma'] / np.sqrt(o['P']) for o in D)
        out['N2']['stored_max_rho_by_y'] = {f'{Y:g}': max(o['rho'] for o in D if o['y_over_sigma'] <= Y) for Y in [0.5, 1, 5]}
        out['N2']['stored_rho_max_state'] = {k: max(D, key=lambda o: o['rho'])[k] for k in ['rho', 'y_over_sigma', 'd', 'P', 'gap_over_sigma']}

# ---------------- N3 ----------------
if have('n3_tangent'):
    T = J('n3_tangent'); E = sorted(T['ell'], key=lambda o: o['ell']); l = np.array([o['ell'] for o in E]); g = np.array([o['gap'] for o in E])
    sel = (l >= 0.2) & (l <= 1.0)
    f3 = lambda x, g0, a, b: g0 + a / x + b * x ** 2
    p, _ = curve_fit(f3, l[sel], g[sel], p0=[0.5, 0.001, 0.01]); lopt = (p[1] / (2 * p[2])) ** (1 / 3)
    q = np.polyfit(l[(l >= 0.3) & (l <= 0.6)], g[(l >= 0.3) & (l <= 0.6)], 2)
    out['N3'] = {'ell_curve': [{k: o[k] for k in ['ell', 'n', 'gap', 'curvature']} for o in E], 'fit_g0_a_over_l_b_l2': p.tolist(), 'fit_ell_opt': float(lopt),
                 'fit_rms': float(np.sqrt(np.mean((f3(l[sel], *p) - g[sel]) ** 2))), 'fit_parabola_vertex': float(-q[1] / (2 * q[0])), 'fit_parabola_curv': float(q[0]),
                 'by_m': [{k: o[k] for k in ['m', 'kappa', 'gap', 'ell_over_sqrt_sigma', 'curvature', 'estimate_tangent_bundle', 'estimate_straight_gaussian']} | {'gap_over_estimate': o['gap'] / o['estimate_tangent_bundle']} for o in T['m']],
                 'scaling_constant_c': float(np.mean([o['ell_over_sqrt_sigma'] for o in T['m']]) * np.sqrt(1 / np.sqrt(2)))}
json.dump(out, open('results/summary_N.json', 'w'), indent=1, default=float)

# ================= figures =================
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
a = ax[0]
for i, (kind, lab) in enumerate([('1d', '1 degree of freedom'), ('0.1', 'coupled ε = 0.1'), ('0.3', 'coupled ε = 0.3')]):
    v = sorted([r for r in rows if r['kind'] == kind], key=lambda r: r['K'])
    a.plot([r['K'] for r in v], [r['gain'] * 100 for r in v], 'o-', color=C[i], label=lab)
a.set_xscale('log', base=2); a.set_xlabel('K'); a.set_ylabel('gain over the best Gaussian (%)'); a.legend(fontsize=8, frameon=False)
a.set_title('N1: the gain of mixtures, same procedure and effort', fontsize=9)
a = ax[1]
for i, e in enumerate(['0.1', '0.3']):
    v = sorted([r for r in rows if r['kind'] == e], key=lambda r: r['K']); g1 = {r['K']: r['gain'] for r in rows if r['kind'] == '1d'}
    a.plot([r['K'] for r in v], [r['gain'] / g1[r['K']] for r in v if r['K'] in g1], 'o-', color=C[i + 1], label=f'ε = {e}: gain / 1-d gain')
    a.axhline(share[e], color=C[i + 1], ls=':', lw=1); a.axhline(share[e] ** N1.get('fit_exponent_a_all', 1), color=C[i + 1], ls='--', lw=1)
a.set_xscale('log', base=2); a.set_xlabel('K'); a.set_ylabel('relative gain'); a.legend(fontsize=7, frameon=False)
a.set_title(f"dotted: plane share f;  dashed: f^a, a = {N1.get('fit_exponent_a_all', float('nan')):.2f} (fit)", fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageN_mix.png', dpi=150); plt.close()

if 'decomp' in N1:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, e in enumerate(['0.1', '0.3']):
        a = ax[i]; v = sorted([r for r in N1['decomp'] if r['eps'] == e], key=lambda r: r['K']); Ks = [r['K'] for r in v]
        for key, lab, c, ls in [('gain_total', 'total gain', C[0], '-'), ('gain_plane', 'plane mixture alone (q⁻ law of the Gaussian)', C[1], '-'),
                                ('d_cross', 'change of the q⁻ cross term (< 0: costs)', C[4], '-'), ('dilution_prediction', 'pure dilution f × (1-d gain)', MUTED, '--')]:
            a.plot(Ks, [100 * r[key] for r in v], 'o' + ls, color=c, label=lab)
        a.axhline(0, color=MUTED, lw=0.6); a.set_xscale('log', base=2); a.set_xlabel('K'); a.set_ylabel('% of the Gaussian gap')
        a.set_title(f'N1(b): ε = {e}, where the gain comes from', fontsize=9)
    ax[0].legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig('figures/stageN_decomp.png', dpi=150); plt.close()

if 'N2' in out:
    fig, a = plt.subplots(figsize=(6, 4))
    if have('m4_dichotomy'):
        D = [o for o in J('m4_dichotomy') if 'error' not in o and o['sigma'] <= 1e-2 and o['gap_over_sigma'] < 5]
        a.scatter([o['P'] for o in D], [o['gap_over_sigma'] for o in D], s=8, color=MUTED, label='stored states (stages A–L)')
    P = [o for o in J('n2_pmin') if o['gap_fine'] < 5]
    a.scatter([o['P_fine'] for o in P], [o['gap_fine'] for o in P], s=30, color=C[1], marker='D', label='P-minimising searches (N2), y/σ ≤ Y')
    xx = np.linspace(0.8, 1.05, 50); a.plot(xx, np.sqrt(xx), '-', color=INK, lw=1, label='gap = √P')
    a.set_xlim(0.84, 1.02); a.set_ylim(0.88, 2.3)
    a.set_xlabel('P = d² y/σ'); a.set_ylabel('(1 − R)/σ'); a.legend(fontsize=7, frameon=False)
    a.set_title('N2: P and the gap (the unrestricted search reached P = 1e-4 at gap 100, off scale)', fontsize=8)
    plt.tight_layout(); plt.savefig('figures/stageN_P.png', dpi=150); plt.close()

if 'N3' in out:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); a = ax[0]; E = out['N3']['ell_curve']
    a.plot([o['ell'] for o in E], [o['gap'] for o in E], 'o', color=C[0], label='optimised for each piece length (n ≥ 64)')
    xx = np.linspace(0.12, 1.4, 200); p = out['N3']['fit_g0_a_over_l_b_l2']; a.plot(xx, f3(xx, *p), '-', color=INK, lw=1, label=f'fit {p[0]:.4f} + {p[1]:.4f}/ℓ + {p[2]:.4f} ℓ²')
    a.axhline(0.5, color=MUTED, ls=':', lw=1); a.set_xlabel('piece length ℓ / √σ'); a.set_ylabel('(κ − R)/σ'); a.legend(fontsize=7, frameon=False)
    a.set_title('N3: tangent bundle, twist flow (σ = 1e-3)', fontsize=9)
    a = ax[1]; M = out['N3']['by_m']
    a.plot([o['m'] for o in M], [o['gap'] for o in M], 'o', color=C[1], label='optimum (n = 64)'); mm = np.linspace(0.6, 1.5, 50)
    a.plot(mm, mm ** 2 / 2, '-', color=INK, lw=1, label='estimate m²/2 (turning removed)'); a.plot(mm, np.sqrt(mm ** 4 / 4 + mm ** 2 / 2), '--', color=MUTED, lw=1, label='straight Gaussian')
    a.set_xlabel('m  (f = (p, −q)/(1 + r^{2m}),  κ = m/4)'); a.set_ylabel('(κ − R)/σ'); a.legend(fontsize=7, frameon=False); a.set_title('changing the radial profile at fixed turning rate', fontsize=9)
    plt.tight_layout(); plt.savefig('figures/stageN_tangent.png', dpi=150); plt.close()
print(json.dumps({'N1': [(r['kind'], r['K'], round(r['opt_ratio'], 5)) for r in rows], 'a': N1.get('fit_exponent_a_each'), 'a_all': N1.get('fit_exponent_a_all')}, indent=0))
