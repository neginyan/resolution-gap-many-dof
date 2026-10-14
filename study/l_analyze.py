"""Stage L analysis: results/summary_L.json and figures stageL_search / stageL_ratios / stageL_fanprofile / stageL_coupled / stageL_twist.
Outputs of fits, extrapolations, argmax and optimisations carry the substrings 'fit', 'coef', 'extrap', 'opt_' in their JSON paths
(compared `loose` by verification/run_all_L.py); rank correlations ('spearman') are computed with scipy.stats.rankdata after
rounding to relative 1e-10 and compared with an absolute tolerance."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import rankdata

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
J = lambda f: json.load(open(f'results/{f}.json'))
have = lambda f: os.path.exists(f'results/{f}.json')
out = {}


def rk(x):
    x = np.asarray(x, float); e = 10.0 ** np.floor(np.log10(np.abs(x) + 1e-300)); return rankdata(np.round(x / e, 10) * e)


def spearman(a, b):
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


# ---------------- L1(a) ----------------
P = J('l1_post'); DK = ['gap_over_sigma', 'R', 'ThetaP', 'eps_tilde_over_sigma', 'D_over_sigma', 'bound_over_sigma', 'geo', 'slope', 'rho', 'r_sup', 'r_sigma', 'j']
rows = [{**{k: o[k] for k in ['family', 'run', 'K', 'sigma', 'gap_fine', 'gap_other_sigma', 'gap_rel_Rstar', 'gap_rel_Rstar_other']}, **{k: o['diag'][k] for k in DK},
         **({'indep': o['indep']} if 'indep' in o else {})} for o in P]
best = {f'{s:g}': min([r for r in rows if r['sigma'] == s], key=lambda r: r['gap_over_sigma']) for s in [1e-3, 1e-2]}
byK = {}
for s in [1e-3, 1e-2]:
    for fam in ['free (hill climb)', 'free (L-BFGS polish)', 'fan', 'ext']:
        v = [r for r in rows if r['sigma'] == s and r['family'] == fam]
        for K in sorted({r['K'] for r in v}):
            byK.setdefault(f'{s:g}', {}).setdefault(fam, {})[str(K)] = min(r['gap_over_sigma'] for r in v if r['K'] == K)
out['L1a'] = {'rows': rows, 'opt_best': best, 'opt_best_by_family_K': byK, 'stage_H_to_K_min_gap': 0.946,
              'opt_min_over_all': min(r['gap_over_sigma'] for r in rows), 'opt_min_rel_Rstar': min(r['gap_rel_Rstar'] for r in rows),
              'all_bound_le_gap': all(r['bound_over_sigma'] <= r['gap_over_sigma'] + 1e-12 for r in rows)}
if have('l1_shiftscan'):
    S = J('l1_shiftscan'); sc = {}
    for s in [1e-3, 1e-2]:
        for wh in ['v', 'w']:
            v = sorted([o for o in S if o['sigma'] == s and o['which'] == wh], key=lambda o: o['shift']); x = np.array([o['shift'] for o in v]); g = np.array([o['gap'] for o in v])
            p = np.polyfit(x, g, 2)
            sc[f'{s:g}_{wh}'] = {'shift': x.tolist(), 'gap': g.tolist(), 'ThetaP': [o['ThetaP'] for o in v], 'fit_quadratic': p.tolist(), 'fit_argmin': float(-p[1] / (2 * p[0]))}
    out['L1a']['shift_scan'] = sc

# ---------------- L1(b) ----------------
if have('l1_ratios'):
    Rt = [o for o in J('l1_ratios') if 'error' not in o]; small = [o for o in Rt if o['sigma'] <= 1e-2]
    near = [o for o in small if o['gap_over_sigma'] <= 1.2]
    st = {}
    for grp, V in [('all_sigma_le_1e-2', small), ('near_optimal_gap_le_1.2', near)]:
        st[grp] = {'n': len(V)}
        for k in ['rho', 'r_sup', 'r_sigma', 'nu', 'geo', 'slope', 'ThetaP']:
            v = np.array([o[k] for o in V]); st[grp][k] = {'min': float(v.min()), 'max': float(v.max()), 'median': float(np.median(v)), 'argmax_tag': V[int(v.argmax())]['tag'],
                                                            'spearman_with_gap': spearman(v, [o['gap_over_sigma'] for o in V])}
    G0, rho0 = st['near_optimal_gap_le_1.2']['geo']['min'], st['near_optimal_gap_le_1.2']['rho']['max']
    eta = max(o['eps_tilde_over_sigma'] + max(o['D_over_sigma'], 0) for o in near)
    out['L1b'] = {'n_states': len(Rt), 'stats': st, 'r_sigma_scaling_note': 'E|d_w w| ~ 1/sqrt(sigma) for scale-free states, so r_sigma ~ sqrt(sigma)',
                  'conditional_constant': {'G0': G0, 'rho0': rho0, 'eta': eta, 'Theta0': G0 * (1 - rho0), 'c_leading': G0 * (1 - rho0) - eta,
                                           'class': 'states with geo >= G0, rho <= rho0, eps~ + D <= eta sigma (values of the near-optimal states)'},
                  'Theta_ge_geo_1mrho_violations': int(sum(o['geo'] * (1 - o['rho']) > o['ThetaP'] * (1 + 1e-9) + 1e-12 for o in Rt))}
if have('l1_rho_max'):
    out['L1b']['rho_max_search'] = [{'K': r['K'], 'key': r['key'], 'value': r['value'], **{k: r['diag'][k] for k in ['gap_over_sigma', 'ThetaP', 'geo', 'slope', 'rho', 'r_sup', 'bound_over_sigma', 'eps_tilde_over_sigma']}}
                                    for r in sorted(J('l1_rho_max'), key=lambda r: (r['key'], r['K']))]

# ---------------- L1(c) ----------------
if have('l1_profile'):
    out['L1c'] = {'fit_profiles': J('l1_profile'), 'note': 'closed forms are ESTIMATES fitted to optimised fans (6-term Chebyshev log-weights), not derived'}

# ---------------- L2 ----------------
if have('l2_bentfan'):
    B2 = sorted(J('l2_bentfan'), key=lambda r: r['K'])
    out['L2'] = {'straight_gaussian': float(np.sqrt(3) / 2), 'single_bent_needle_stageI': 0.509, 'target_0p95': 0.509 * 0.95,
                 'opt_fans': [{k: r.get(k) for k in ['K', 'gap', 'gap_search', 'gap_fine', 'gap_nseg24', 'gap_sigma1e-4', 'gap_sigma1e-2', 'single_arc_gap', 'single_arc_gap_nseg24',
                                                   'stageI_best_arc_nseg24', 'gap_nseg48', 'single_arc_gap_nseg48', 'R', 'n_components', 'indep']} for r in B2]}
    if have('l2_arc_continuum'):
        A2 = J('l2_arc_continuum'); out['L2']['arc_continuum'] = {k: v for k, v in A2.items() if not k.endswith('_x')}

# ---------------- L3 ----------------
from l3_coupled import gap_coefficient, kappa
S3 = J('l3_single'); L3 = {'table': [], 'coef': {}}
for o in S3:
    L3['table'].append({k: o[k] for k in ['eps', 'sigma', 'kappa', 'Rstar', 'gap_over_sigma', 'gap_law1_plain']} | {'opt_sd': o['sd_in_mode_frame']})
for e in [0.0, 0.1, 0.3]:
    v = sorted([o for o in S3 if o['eps'] == e], key=lambda o: o['sigma']); s = np.array([o['sigma'] for o in v]); g = np.array([o['gap_over_sigma'] for o in v])
    sel = s <= 1e-3; p = np.polyfit(s[sel], g[sel], 1)
    gc = gap_coefficient(e) if e > 0 else {'matrix': 1.0, 'diagonal': 1.0, 'B': None, 'kappa': 1.0}
    entry = {'opt_gap_smallest_sigma': float(g[0]), 'smallest_sigma': float(s[0]), 'extrap_sigma0': float(p[1]), 'extrap_slope': float(p[0]),
             'coef_matrix_formula': gc['matrix'], 'coef_diagonal_formula': gc['diagonal'], 'B': gc['B'], 'kappa': float(kappa(e))}
    if e > 0:
        B = np.array(gc['B']); k = kappa(e)
        entry['coef_parts'] = {'W+ alone (1-d formula 2 sqrt(2 B11 kappa))': float(2 * np.sqrt(2 * B[0, 0] * k)), 'q- alone (2 sqrt(B22 kappa))': float(2 * np.sqrt(B[1, 1] * k))}
    L3['coef'][f'{e:g}'] = entry
L3['coef_small_eps'] = {f'{e:g}': gap_coefficient(e)['matrix'] for e in [1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
L3['check'] = J('l3_check')
L3['eps1'] = [{k: o[k] for k in ['sigma', 'gap_over_sigma', 'matrix_formula']} for o in J('l3_eps1')]
out['L3'] = L3
json.dump(out, open('results/summary_L.json', 'w'), indent=1, default=float)

# ================= figures =================
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
a = ax[0]; fams = [('free (hill climb)', 'o'), ('free (L-BFGS polish)', 's')]
for i, s in enumerate([1e-3, 1e-2]):
    for fam, mk in fams:
        d = byK.get(f'{s:g}', {}).get(fam, {})
        if d: a.plot([int(k) for k in d], list(d.values()), mk + '-', color=C[i], ms=5, lw=1.2, alpha=1 if 'polish' in fam else 0.5, label=f'{fam}, σ = {s:g}')
    for fam, ls in [('fan', ':'), ('ext', '--')]:
        d = byK.get(f'{s:g}', {}).get(fam)
        if d: a.axhline(list(d.values())[0], color=C[i], ls=ls, lw=1)
a.axhline(0.946, color=MUTED, lw=1); a.text(8, 0.9465, 'stages H–K: 0.946', color=MUTED, fontsize=7, va='bottom')
a.set_xlabel('K (number of components)'); a.set_ylabel('(1 − R)/σ'); a.set_title('L1(a): worst states (dotted: fan, dashed: extended fan;\nhill climb at K = 24, 32 started from a perturbed K = 16 state)', fontsize=8.5)
a.legend(fontsize=6.5, frameon=False)
a = ax[1]
for i, s in enumerate([1e-3, 1e-2]):
    v = [r for r in rows if r['sigma'] == s]
    a.scatter([r['rho'] for r in v], [r['gap_over_sigma'] for r in v], color=C[i], s=16, label=f'σ = {s:g}')
    a.scatter([r['rho'] for r in v], [r['ThetaP'] for r in v], color=C[i], s=16, marker='x')
a.set_xlabel('ρ = −E[u ∂_w w] / E[w ∂_w u]'); a.set_ylabel("● (1 − R)/σ     × Θ′"); a.set_title('the worse the state, the larger the focus-slope share ρ', fontsize=9); a.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig('figures/stageL_search.png', dpi=150); plt.close()

if 'L1b' in out:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, k, lab in zip(ax, ['rho', 'r_sup', 'nu'], ['ρ = −E[u ∂_w w]/E[w ∂_w u]', 'E|u ∂_w w| / (j sup|u|)', 'ν = E[(∂_w w)²/w]/J_ww  (unstable, see text)']):
        for grp, sel, c in [('stages A–K', lambda o: not o['tag'].startswith('l1_'), MUTED), ('stage L searches', lambda o: o['tag'].startswith('l1_') and 'rho_max' not in o['tag'], C[0]),
                            ('ρ / r_sup maximisers', lambda o: 'rho_max' in o['tag'], C[1])]:
            v = [o for o in small if sel(o)]
            a.scatter([o['gap_over_sigma'] for o in v], [o[k] for o in v], s=10, color=c, label=grp)
        a.set_xscale('log'); a.set_yscale('symlog', linthresh=1e-2); a.set_xlabel('(1 − R)/σ'); a.set_title(lab, fontsize=9)
    ax[0].legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig('figures/stageL_ratios.png', dpi=150); plt.close()

if 'L1c' in out:
    from l1_profile import profiles
    i1 = J('i1_fan'); x = max(i1, key=lambda o: o['checks']['n40'])['x']; Pf = profiles(x); f0 = out['L1c']['fit_profiles'][0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    a = ax[0]; al = Pf['alpha']
    a.semilogy(al, Pf['weight'], '-', color=INK, lw=2, label='optimised fan (stage I)')
    A, s_ = f0['weight']['sech2']['params']; a.semilogy(al, np.exp(A - 2 * np.log(np.cosh(al / s_))), '--', color=C[0], label=f'sech²(α/{s_:.3f})')
    A, s_ = f0['weight']['gauss']['params']; a.semilogy(al, np.exp(A - al ** 2 / (2 * s_ ** 2)), ':', color=C[1], label=f'Gaussian, s = {s_:.3f}')
    a.set_ylim(Pf['weight'].max() * 1e-4, Pf['weight'].max() * 2); a.set_xlabel('angle α  (units of √σ/2)'); a.set_ylabel('weight per unit angle'); a.legend(fontsize=7, frameon=False)
    a.set_title('L1(c): weight profile (estimate)', fontsize=9)
    a = ax[1]; L0, c_ = f0['length']['exp_a2']['params']
    a.plot(al, Pf['length'], '-', color=INK, lw=2, label='optimised fan'); a.plot(al, L0 * np.exp(c_ * al ** 2), '--', color=C[0], label=f'{L0:.3f} exp({c_:.3f} α²)')
    a.set_xlabel('angle α  (units of √σ/2)'); a.set_ylabel('needle length (units of 2√σ)'); a.legend(fontsize=7, frameon=False); a.set_title('needles get longer away from W', fontsize=9)
    plt.tight_layout(); plt.savefig('figures/stageL_fanprofile.png', dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
a = ax[0]
for i, e in enumerate([0.0, 0.1, 0.3]):
    v = sorted([o for o in S3 if o['eps'] == e], key=lambda o: o['sigma'])
    a.plot([o['sigma'] for o in v], [o['gap_over_sigma'] for o in v], 'o-', color=C[i], lw=1.5, ms=4, label=f'ε = {e:g}')
    a.axhline(L3['coef'][f'{e:g}']['coef_matrix_formula'], color=C[i], ls=':', lw=1)
a.set_xscale('log'); a.set_xlabel('σ'); a.set_ylabel('(κ − R*)/σ'); a.set_title('L3: best Gaussian, coupled pendulums (dotted: formula)', fontsize=9); a.legend(fontsize=8, frameon=False)
a = ax[1]; ee = np.logspace(-4, 0.5, 120)
a.plot(ee, [gap_coefficient(e)['matrix'] for e in ee], '-', color=C[0], lw=2, label='2 tr[(B^½ C B^½)^½]')
a.plot(ee, [2 * np.sqrt(2 * gap_coefficient(e)['B'][0][0] * kappa(e)) for e in ee], '--', color=C[1], lw=1.2, label='W⁺ term alone (1-d formula)')
a.scatter([0.1, 0.3, 1.0], [L3['coef']['0.1']['opt_gap_smallest_sigma'], L3['coef']['0.3']['opt_gap_smallest_sigma'], L3['eps1'][0]['gap_over_sigma']], color=INK, zorder=5, s=22, label='optimum (smallest σ)')
a.set_xscale('log'); a.set_xlabel('coupling ε'); a.set_ylabel('lim (κ − R*)/σ'); a.set_title('the extra configuration direction q⁻ adds 2√(B₂₂κ)', fontsize=9); a.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig('figures/stageL_coupled.png', dpi=150); plt.close()

if 'L2' in out:
    fig, a = plt.subplots(figsize=(6, 3.8)); F = out['L2']['opt_fans']
    a.plot([f['K'] for f in F], [f['gap'] for f in F], 'o-', color=C[0], lw=1.5, label='bent fan (12 segments/needle), σ = 1e-3')
    a.plot([f['K'] for f in F], [f['gap_nseg24'] for f in F], 's--', color=C[1], lw=1, label='same, 24 segments/needle')
    a.plot([f['K'] for f in F], [f['gap_nseg48'] for f in F], 'D:', color=C[2], lw=1, label='same, 48 segments/needle')
    for y, t in [(0.509, 'single bent needle 0.509'), (0.5, '0.5'), (0.509 * 0.95, '0.509 × 0.95')]:
        a.axhline(y, color=MUTED, ls=':', lw=1); a.text(F[0]['K'], y, ' ' + t, fontsize=7, color=MUTED, va='bottom')
    a.set_xscale('log', base=2); a.set_xlabel('K (number of bent needles)'); a.set_ylabel('(κ − R)/σ'); a.set_title('L2: bent fan, twist flow', fontsize=9); a.legend(fontsize=6.5, frameon=False)
    plt.tight_layout(); plt.savefig('figures/stageL_twist.png', dpi=150); plt.close()
    if 'arc_continuum' in out['L2']:
        A2 = out['L2']['arc_continuum']; fig, a = plt.subplots(figsize=(6, 3.8))
        for key, lab, c in [('stageI', 'stage-I bent needle (optimised at 24 segments)', C[0]), ('opt_n64', 're-optimised at 64 segments', C[1])]:
            d = sorted((int(k), v) for k, v in A2[key].items()); a.plot([k for k, v in d], [v for k, v in d], 'o-', color=c, label=lab)
        a.axhline(0.5, color=MUTED, ls=':', lw=1); a.text(24, 0.5, ' prediction 1/2 (β_eff = 1/8)', fontsize=7, color=MUTED, va='bottom')
        a.set_xscale('log', base=2); a.set_xlabel('number of straight pieces n'); a.set_ylabel('(κ − R)/σ  (σ = 1e-3)')
        a.set_title('a bent needle made of n straight pieces: the gap grows with n', fontsize=9); a.legend(fontsize=7, frameon=False)
        plt.tight_layout(); plt.savefig('figures/stageL_arc_pieces.png', dpi=150); plt.close()
print(json.dumps({'best': {k: round(v['gap_over_sigma'], 5) for k, v in best.items()},
                  'L3': {k: (round(v['opt_gap_smallest_sigma'], 6), round(v['coef_matrix_formula'], 6)) for k, v in L3['coef'].items()}}, indent=0))
