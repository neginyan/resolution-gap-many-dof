"""Stage G analysis: numbers -> results/summary_G.json; figures stageG_twoneedles.png, stageG_bestsingle.png,
stageG_coefficient.png, stageG_recheck.png."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
L = lambda f: json.load(open(f'results/{f}.json'))
slope = lambda x, y: float(np.polyfit(np.log(np.abs(x)), np.log(np.abs(y)), 1)[0])
out = {}

# ---------- 1. two needles near the top ----------
g1 = L('g1_families'); fam = {}
for s in [1e-3, 1e-2]:
    for f in ['sym', 'one', 'alongW', 'width']:
        v = [r for r in g1 if r['family'] == f and r['sigma'] == s]; sm = [r for r in v if r['delta'] < 0.3]
        e = np.array([r['excess'] for r in sm]); rec = {'max_excess': max(r['excess'] for r in v), 'max_g': max(r['g'] for r in v),
               'all_negative_beyond_noise': bool(all(r['excess'] < 1e-10 for r in v)), 'slope_vs_delta': slope([r['delta'] for r in sm], e),
               'max_R16_minus_R10': max(abs(r['R_per_sd16'] - r['R']) for r in v)}
        if all(r['d_abar'] > 0 for r in sm): rec['slope_vs_dabar'] = slope([r['d_abar'] for r in sm], e); rec['excess_over_dabar_small'] = float(e[0] / sm[0]['d_abar'])
        if all(r['off2'] > 0 for r in sm): rec['slope_vs_off2'] = slope([r['off2'] for r in sm], e); rec['excess_over_off2_small'] = float(e[0] / sm[0]['off2'])
        fam[f'{f}_sigma{s:g}'] = rec
out['g1'] = fam
M = L('g2_map') + L('g2_map2'); b = max(M, key=lambda r: r['g'])
out['g2_map'] = {'n': len(M), 'max_g': b['g'], 'at': [b['rv'], b['rw']], 'excess': b['excess'], 'gap': b['gap'], 'd_abar': b['d_abar'],
                 'n_g_positive': sum(r['g'] > 0 for r in M), 'max_R16_minus_R10': max(abs(r['R_per_sd16'] - r['R']) for r in M)}
sc = L('g2_scale'); sg = [r['sigma'] for r in sc]
out['g2_scale'] = {'sigma_range': [min(sg), max(sg)], 'slope_excess': slope(sg, [r['excess'] for r in sc]), 'slope_gap': slope(sg, [r['gap'] for r in sc]),
                   'slope_d_abar': slope(sg, [r['d_abar'] for r in sc]), 'g_range': [min(r['g'] for r in sc), max(r['g'] for r in sc)],
                   'excess_over_dabar': [min(r['excess'] / r['d_abar'] for r in sc), max(r['excess'] / r['d_abar'] for r in sc)],
                   'max_R16_minus_R10': max(abs(r['R_per_sd16'] - r['R']) for r in sc)}
sh = L('g2_shift'); out['g2_shift'] = {}
for dr in 'VW':
    v = sorted([r for r in sh if r['direction'] == dr], key=lambda r: r['delta']); m = [r for r in v if r['delta'] < 0.3]
    neg = [r['delta'] for r in v if r['g'] < 0]
    out['g2_shift'][dr] = {'slope_g_drop_vs_delta': slope([r['delta'] for r in m], [b['g'] - r['g'] for r in m]),
                           'slope_g_drop_vs_off2': slope([r['off2'] for r in m], [b['g'] - r['g'] for r in m]),
                           'first_delta_g_negative': min(neg) if neg else None}
se = L('g2_search')
out['g2_search'] = [{'K': r['K'], 'sigma': r['sigma'], 'g': r['g'], 'excess': r['excess'], 'R': r['R'], 'maxRk': r['maxRk'], 'R16_minus_R10': r['R_per_sd16'] - r['R']} for r in se]
g3 = L('g3_best_single'); tags = sorted(set(o['tag'] for o in g3['rows']))
out['g3'] = {'n': g3['n'], 'n_R_gt_Rstar': g3['n_R_gt_Rstar'], 'min_margin': g3['min_margin'], 'max_gstar': g3['max_gstar'],
             'max_gstar_by_tag': {t: max(o['gstar'] for o in g3['rows'] if o['tag'] == t) for t in tags},
             'Rstar_examples': {k: g3['Rstar_table'][k][0] for k in ['1e-05', '0.0001', '0.001', '0.01', '0.1', '0.3', '1.0'] if k in g3['Rstar_table']},
             'max_rel_dev_1mRstar_over_sigma_small': max(abs((1 - v[0]) / float(k) - 1) for k, v in g3['Rstar_table'].items() if float(k) <= 1e-4)}
best_g3 = max(g3['rows'], key=lambda o: o['gstar']); out['g3']['closest'] = best_g3

fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))
a = ax[0, 0]
for i, f in enumerate(['sym', 'one', 'alongW']):
    v = [r for r in g1 if r['family'] == f and r['sigma'] == 1e-3]
    a.loglog([r['off2'] for r in v], [-r['excess'] for r in v], 'o-', ms=3.5, color=C[i], lw=1.4, label={'sym': 'two needles at π ± δ', 'one': 'one at the top, one shifted along V', 'alongW': 'one at the top, one shifted along W'}[f])
x = np.array([1e-13, 1e-6]); a.loglog(x, 3e1 * x, ':', color=MUTED, lw=1, label='slope 1')
a.set_xlabel('(offset of a needle from the top)²  [q − π]²'); a.set_ylabel('−(R − max_k R_k)  (mixing lowers R)')
a.set_title('(1) same-shape needles, σ = 10⁻³: at small offsets the loss ∝ offset² ∝ Δā', fontsize=9); a.legend(fontsize=7.5, frameon=False)
a = ax[0, 1]
a.loglog(sg, [r['excess'] for r in sc], 'o-', color=C[1], label='excess R − max_k R_k')
a.loglog(sg, [r['gap'] for r in sc], 's-', color=C[0], label='gap 1 − max_k R_k')
a.loglog(sg, [r['d_abar'] for r in sc], '^-', color=C[2], label='Δā (stretch difference)')
a.set_xlabel('window σ'); a.set_title(f"(2) concentric needles of different shape (r_v = {b['rv']:.2f}, r_w = {b['rw']:.2f}):\n"
                                      f"all three ∝ σ; excess/gap = g = {sc[0]['g']:.4f} at every σ", fontsize=9); a.legend(fontsize=7.5, frameon=False)
a = ax[1, 0]
for i, dr in enumerate('VW'):
    v = sorted([r for r in sh if r['direction'] == dr], key=lambda r: r['delta'])
    a.loglog([r['delta'] for r in v], [b['g'] - r['g'] for r in v], 'o-', ms=3.5, color=C[i], label=f'second needle shifted along {dr}')
x = np.array([1e-2, 1]); a.loglog(x, 0.05 * x ** 2, ':', color=MUTED, lw=1, label='slope 2')
a.axhline(b['g'], color=INK, lw=0.8, ls='--'); a.text(0.012, b['g'] * 1.3, 'g(0): above this line g < 0', fontsize=7.5, color=INK)
a.set_xlabel('shift δ (in widths along V / lengths along W)'); a.set_ylabel('g(0) − g(δ)')
a.set_title('(3) shifting the second needle off the top: g falls as δ²', fontsize=9); a.legend(fontsize=7.5, frameon=False)
a = ax[1, 1]
gm = np.array([[r['g'] for r in L('g2_map2') if r['rv'] == rv] for rv in sorted(set(r['rv'] for r in L('g2_map2')))])
rvs = sorted(set(r['rv'] for r in L('g2_map2'))); rws = sorted(set(r['rw'] for r in L('g2_map2')))
from matplotlib.colors import TwoSlopeNorm
im = a.pcolormesh(rws, rvs, gm, shading='nearest', cmap='RdBu_r', norm=TwoSlopeNorm(0, gm.min(), max(gm.max(), 1e-3)))
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('r_w (length ratio along W)'); a.set_ylabel('r_v (width ratio along V)')
cb = plt.colorbar(im, ax=a, label='g'); cb.set_ticks([-0.4, -0.2, 0, 0.01, 0.02]); a.set_title('(4) g for two concentric needles (σ = 10⁻³); max 0.027', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageG_twoneedles.png', dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
a = ax[0]
for i, t in enumerate(tags):
    v = [o for o in g3['rows'] if o['tag'] == t]
    a.loglog([1 - o['Rstar'] for o in v], [max(1 - o['R'], 1e-7) for o in v], 'o', ms=2.5, alpha=0.6, color=(C + ['#444', '#999', '#c7a', '#7ac', '#ac7', '#555'])[i], label=t.replace('.json', ''))
x = np.array([1e-5, 1]); a.loglog(x, x, '-', color=INK, lw=1, label='R = R*(σ)')
a.set_xlabel('1 − R*(σ)  (best single Gaussian at the same window)'); a.set_ylabel('1 − R (state)')
a.set_title(f"(1) every state lies above the line: R ≤ R*(σ) ({g3['n']} states)", fontsize=9); a.legend(fontsize=6, frameon=False, ncol=2)
a = ax[1]
a.hist([o['gstar'] for o in g3['rows']], bins=np.linspace(-3, 0.05, 62), color=C[0])
a.axvline(0, color=INK, lw=1); a.set_yscale('log'); a.set_xlabel('g* = (R − R*) / (1 − R*)'); a.set_ylabel('states')
a.set_title(f"(2) largest g* = {g3['max_gstar']:.3f} (always below 0)", fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageG_bestsingle.png', dpi=150); plt.close()

# ---------- 2. coefficient c ----------
g5 = L('g5_coefficient'); rows = g5['rows']; t = np.array([r['t'] for r in rows])
small = rows[-1]
out['g5'] = {k: g5[k] for k in ['q_k', 'S_vv', 'S_ww', 'S_vw', 'sigma0', 'lam_rho0', 'c0', 'c', 'c_phi_part', 'c_lam_part']}
out['g5']['measured_slope_small_t'] = (small['measured'] - g5['c0']) / small['t']
out['g5']['rel_err_exact_prediction'] = [abs(r['prediction_exact_law1'] / r['measured'] - 1) for r in rows]
out['g5']['rel_err_linear'] = [abs(r['prediction_linear'] / r['measured'] - 1) for r in rows]
out['g5']['rel_err_ruler'] = [abs(r['ruler_prediction'] / r['one_minus_fraction'] - 1) for r in rows]
out['g5']['max_R_needle_law1_minus_grid'] = max(abs(r['R_needle_law1'] - r['R_needle_grid']) for r in rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
a = ax[0]
a.loglog(t, [r['measured'] for r in rows], 'o', ms=6, color=C[0], label='1 − ρ·CV·Ω (exact computation, stage F)')
a.loglog(t, [r['prediction_exact_law1'] for r in rows], '-', color=C[1], label='law-1 prediction (1−φ) + φ(1−λ_n)/(1−λ_ρ)')
a.loglog(t, [r['prediction_linear'] for r in rows], '--', color=C[2], label=f"c₀ + c·t,  c = {g5['c']:.5f}")
a.set_xlabel('needle size t'); a.set_title('(1) 1 − ρ·CV·Ω along the needle sequence', fontsize=9); a.legend(fontsize=7.5, frameon=False)
a = ax[1]
a.loglog(t, out['g5']['rel_err_exact_prediction'], 'o-', color=C[1], label='law-1 prediction of 1 − ρ·CV·Ω')
a.loglog(t, out['g5']['rel_err_ruler'], 's-', color=C[3], label='ruler prediction (1 − R_needle)/(1 − law) of 1 − fraction')
a.set_xlabel('needle size t'); a.set_ylabel('relative error'); a.set_title('(2) the predictions become exact as t → 0 (error ∝ t)', fontsize=9)
a.legend(fontsize=7.5, frameon=False)
plt.tight_layout(); plt.savefig('figures/stageG_coefficient.png', dpi=150); plt.close()

# ---------- 3. recheck with the nested grid ----------
try:
    g4 = L('g4_recheck'); cap = [r for r in g4 if 'R_nested' in r]
    out['g4'] = {'n_states': len(g4), 'n_capped': sum(r['capped'] for r in g4), 'n_rechecked': len(cap),
                 'max_abs_nested_minus_stored': max(abs(r['diff']) for r in cap), 'max_abs_refined_minus_nested': max(abs(r['R_nested_refined'] - r['R_nested']) for r in cap),
                 'max_mass_dev': max(abs(r['mass_nested'] - 1) for r in cap), 'max_R_nested': max(r['R_nested'] for r in cap),
                 'n_R_nested_ge_1': sum(r['R_nested'] >= 1 for r in cap), 'by_tag': {},
                 'fallback': [{k: r[k] for k in ['tag', 'sigma', 'R_stored', 'diff_alt']} for r in g4 if 'failed' in r]}
    for tg in sorted(set(r['tag'] for r in cap)):
        v = [r for r in cap if r['tag'] == tg]; out['g4']['by_tag'][tg] = {'n': len(v), 'max_abs_diff': max(abs(r['diff']) for r in v)}
    worst = max(cap, key=lambda r: abs(r['diff'])); out['g4']['worst'] = worst
    fig, a = plt.subplots(figsize=(6, 4))
    for i, tg in enumerate(sorted(set(r['tag'] for r in cap))):
        v = [r for r in cap if r['tag'] == tg]
        a.semilogy([r['R_nested'] for r in v], [max(abs(r['diff']), 1e-16) for r in v], 'o', ms=3, color=C[i % 6], label=tg)
        a.semilogy([r['R_nested'] for r in v], [max(abs(r['R_nested_refined'] - r['R_nested']), 1e-16) for r in v], 'x', ms=3, color=C[i % 6], alpha=0.5)
    a.set_xlabel('R (nested grid)'); a.set_ylabel('|difference|'); a.legend(fontsize=7, frameon=False)
    a.set_title('capped states: ○ nested − stored,  × refined nested − nested', fontsize=9)
    plt.tight_layout(); plt.savefig('figures/stageG_recheck.png', dpi=150); plt.close()
except FileNotFoundError:
    print('g4_recheck.json not found yet')
json.dump(out, open('results/summary_G.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ['g1']}, indent=1, default=float)[:6000])
