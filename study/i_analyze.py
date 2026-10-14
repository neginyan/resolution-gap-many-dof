"""Stage I analysis: numbers -> results/summary_I.json; figures stageI_fan.png, stageI_bound.png, stageI_flows.png."""
import json, numpy as np
from fractions import Fraction as Fr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from i_tools import stretch, V, W, TOPS, KAPPA

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6b6a65']
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#e6e5df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
L = lambda f: json.load(open(f'results/{f}.json'))
out = {}

# ---------- 1. continuous fan ----------
fan = L('i1_fan'); best = max(fan, key=lambda r: r['checks']['n40'])
out['fan'] = [{'start': r['start'], 'seed': r['seed'], 'gstar_search': r['gstar_search'], **r['checks']} for r in fan]
out['fan_best'] = {'start': best['start'], **best['checks'], 'x': best['x']}
from i1_fan import fan as fan_state, unpack
from numpy.polynomial import chebyshev as Ch
xs = np.linspace(-1, 1, 401); lT, cw, cl, ct, co = unpack(np.array(best['x']))
wfun = np.exp(Ch.chebval(xs, cw)); wfun /= np.trapezoid(wfun, xs)
ang = np.exp(lT) * xs                     # angle in units of sqrt(sigma)/2
out['fan_profile'] = {'Theta': float(np.exp(lT)), 'length_range': [float(np.exp(Ch.chebval(xs, cl)).min()), float(np.exp(Ch.chebval(xs, cl)).max())],
                      'thickness_range': [float(np.exp(Ch.chebval(xs, ct)).min()), float(np.exp(Ch.chebval(xs, ct)).max())]}
prev = {'K8_chain_best': max(x['gstar10'] for c in L('h4_growk') for x in c['chain']), 'symmetric_fan_stageH': max(o['gstar'] for o in L('h5_fan'))}
out['fan_compare'] = prev

fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
a = ax[0]
for i, r in enumerate(fan):
    a.plot(np.arange(len(r['trace'])) * 25, r['trace'], '-', color=C[i], lw=1.4, label=f"start {r['start']} (seed {r['seed']}): final {r['checks']['n40']:+.4f}")
a.axhline(prev['K8_chain_best'], color=MUTED, ls='--', lw=1); a.text(10, prev['K8_chain_best'] + 0.001, 'K = 8 (stage H)', fontsize=7, color=MUTED)
a.axhline(prev['symmetric_fan_stageH'], color=MUTED, ls=':', lw=1); a.text(10, prev['symmetric_fan_stageH'] - 0.004, 'symmetric fan (stage H)', fontsize=7, color=MUTED)
a.set_ylim(-0.02, 0.08); a.set_xlabel('search step'); a.set_ylabel('g*'); a.legend(fontsize=6.5, frameon=False, loc='lower right')
a.set_title('(1) functional fan: search progress (σ = 10⁻³)', fontsize=9)
a = ax[1]
a.plot(ang, wfun, '-', color=C[0], lw=2, label='weight density')
a2 = a.twinx(); a2.plot(ang, np.exp(Ch.chebval(xs, cl)), '-', color=C[1], lw=1.5, label='length / 2√σ'); a2.plot(ang, np.exp(Ch.chebval(xs, ct)), '-', color=C[2], lw=1.5, label='thickness / σ')
a2.plot(ang, Ch.chebval(xs, co), '--', color=C[3], lw=1.2, label='V-offset / σ'); a2.spines['right'].set_visible(True)
a.set_xlabel('tilt angle / (√σ/2)'); a.set_ylabel('weight density'); a2.set_ylabel('length, thickness, offset')
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left')
a.set_title(f"(2) best fan: profiles along the angle (g* = {best['checks']['n40']:+.4f})", fontsize=9)
a = ax[2]
ks = ['n20', 'n40', 'n80']; a.plot([20, 40, 80], [best['checks'][k] for k in ks], 'o-', color=C[0], label='number of segments n (σ = 10⁻³)')
a.plot([40], [best['checks']['n40_sigma1e-4']], 's', color=C[1], label='n = 40, σ = 10⁻⁴'); a.plot([40], [best['checks']['n40_sigma1e-2']], '^', color=C[2], label='n = 40, σ = 10⁻²')
a.set_xscale('log', base=2); a.set_xlabel('segments n'); a.set_ylabel('g*'); a.legend(fontsize=7, frameon=False); a.set_title('(3) continuum and scale checks', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageI_fan.png', dpi=150); plt.close()

# ---------- 2. bound ----------
B = [o for o in L('i2_bound') if 'error' not in o]; small = [o for o in B if o['sigma'] <= 1e-2]
out['bound'] = {'n': len(B), 'n_small': len(small), 'identity_max_err': max(abs(o['exact_identity'] - o['one_minus_f']) for o in B),
                'LB_le_one_minus_f': all(o['LB'] <= o['one_minus_f'] + 1e-12 for o in B),
                'min_gap_over_sigma': min(o['gap_over_sigma'] for o in small), 'min_LB_over_sigma': min(o['LB_over_sigma'] for o in small),
                'min_LB_over_gap': min(o['LB_over_sigma'] / o['gap_over_sigma'] for o in small), 'CR_holds': all(o['Q1Jww'] >= o['CR_bound'] - 1e-12 for o in B),
                'single_segment_LB_over_gap': [o['LB_over_sigma'] / o['gap_over_sigma'] for o in B if o['tag'] == 'Rstar_segment'],
                'min_Pi': min(o['Pi'] for o in small), 'max_D': max(o['D'] for o in small)}
tags = sorted(set(o['tag'] for o in small)); out['bound']['by_tag'] = {t: {'n': sum(o['tag'] == t for o in small), 'min_gap': min(o['gap_over_sigma'] for o in small if o['tag'] == t),
                                                                          'min_LB': min(o['LB_over_sigma'] for o in small if o['tag'] == t),
                                                                          'min_j': min(o['j'] for o in small if o['tag'] == t)} for t in tags}
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
a = ax[0]
for i, t in enumerate(tags):
    v = [o for o in small if o['tag'] == t]
    a.scatter([o['gap_over_sigma'] for o in v], [o['LB_over_sigma'] for o in v], s=10, color=(C + ['#444', '#999', '#c7a', '#7ac', '#ac7', '#555'])[i], label=t, alpha=0.8)
x = np.array([0.5, 4]); a.plot(x, x, '-', color=INK, lw=1, label='bound = actual')
a.axhline(out['bound']['min_LB_over_sigma'], color=MUTED, ls=':', lw=1); a.axvline(out['bound']['min_gap_over_sigma'], color=MUTED, ls='--', lw=1)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('actual (1 − R)/σ'); a.set_ylabel('rigorous state-wise bound / σ'); a.legend(fontsize=6, frameon=False, ncol=2)
a.set_title('(1) the chain identity + bathtub, all states with σ ≤ 10⁻²', fontsize=9)
a = ax[1]
for i, t in enumerate(tags):
    v = [o for o in small if o['tag'] == t]
    a.scatter([o['j'] for o in v], [o['LB_over_sigma'] / o['gap_over_sigma'] for o in v], s=10, color=(C + ['#444', '#999', '#c7a', '#7ac', '#ac7', '#555'])[i], alpha=0.8)
a.set_xscale('log'); a.set_xlabel('j = σ² E[(H_vv)₊]  (how close the V-focus is to the limit 1/σ²)'); a.set_ylabel('bound / actual')
a.set_title('(2) the bound is tight when j → 1 and loosens as j falls', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageI_bound.png', dpi=150); plt.close()

# ---------- 3. other flows ----------
S1 = L('i3_single'); out['flows_single'] = S1
beta = {}
for kind in ['pendulum', 'double_well', 'quartic', 'twist']:
    top = TOPS[kind]; t = np.array([-2e-3, -1e-3, 0, 1e-3, 2e-3]); pts = top + t[:, None] * W
    proj = np.einsum('i,nij,j->n', V, stretch(kind, pts)[2], V)
    c2 = np.polyfit(t, proj, 4)[-3]; beta[kind] = float(-c2)       # proj ~ kappa - beta t^2
pred = {k: 2 * np.sqrt(2 * beta[k] * KAPPA[k]) for k in beta}
out['beta'] = beta; out['predicted_gap_coefficient'] = pred
meas = {'pendulum': 1.0}
for k in ['double_well', 'quartic', 'twist']:
    rows = S1['flows'][k]['rows']; s = np.array([r['sigma'] for r in rows[:3]]); g = np.array([r['gap_over_sigma'] for r in rows[:3]])
    meas[k] = float(np.polyval(np.polyfit(s, g, 2), 0.0))                    # extrapolation sigma -> 0
out['measured_gap_coefficient_sigma0'] = meas
try:
    SE = L('i3_search'); out['flows_search'] = [{k: r[k] for k in ['kind', 'K', 'sigma', 'gstar', 'gap_over_sigma', 'Rstar_gap_over_sigma', 'R', 'R_check']} for r in SE]
except FileNotFoundError:
    SE = []
try:
    I5 = L('i5_arc'); b5 = max(I5, key=lambda o: o['gstar_n48'])
    out['twist_arc'] = {'runs': [{k: o[k] for k in o if k != 'x'} for o in I5], 'best': {k: b5[k] for k in b5 if k != 'x'},
                        'predicted_gap_coefficient_arc': 2 * np.sqrt(2 * 0.125 * 0.25)}
except FileNotFoundError:
    I5 = []
try:
    I4 = L('i4_twist'); out['twist_growK'] = [{'seed': c['seed'], 'by_K': [{'K': r['K'], 'gstar': r['gstar'], 'scale': r['scale']} for r in c['chain']]} for c in I4]
except FileNotFoundError:
    I4 = []
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
a = ax[0]
from h1_rstar import rstar
ss = np.logspace(-3, -1, 30); a.semilogx(ss, [(1 - rstar(x)['Rstar']) / x for x in ss], '-', color=C[0], label='pendulum (κ = 1)')
for i, k in enumerate(['double_well', 'quartic', 'twist']):
    rows = S1['flows'][k]['rows']; a.semilogx([r['sigma'] for r in rows], [r['gap_over_sigma'] for r in rows], 'o-', color=C[i + 1], label=f"{k} (κ = {KAPPA[k]:g})")
for i, k in enumerate(['pendulum', 'double_well', 'quartic', 'twist']):
    a.plot([8e-4], [pred[k]], '<', color=C[i], ms=7)
a.set_xlabel('window σ'); a.set_ylabel('(κ − R*) / σ'); a.legend(fontsize=7, frameon=False); a.set_title('(1) best single Gaussian: gap ∝ σ in every flow\n(◂ = prediction 2√(2βκ))', fontsize=9)
a = ax[1]
if SE:
    for i, k in enumerate(['double_well', 'quartic', 'twist']):
        v = [r for r in SE if r['kind'] == k]
        a.plot([r['sigma'] * (1.15 if r['K'] == 3 else 1) for r in v], [r['gstar'] for r in v], 'o', color=C[i + 1], label=k)
    a.set_xscale('log'); a.axhline(0, color=INK, lw=1); a.set_xlabel('window σ (K = 2; K = 3 shifted right)'); a.set_ylabel('g* = (R − R*)/(κ − R*)')
    a.legend(fontsize=7, frameon=False); a.set_title('(2) mixtures near the top versus R*', fontsize=9)
a = ax[2]
if I4:
    for i, c in enumerate(I4):
        a.plot([r['K'] for r in c['chain']], [[v for v in r['scale'] if v['sigma'] == 1e-3][0]['gap_over_sigma'] for r in c['chain']], 'o-', color=C[i], label=f'twist: components added (chain {i + 1})')
if I5:
    a.axhline(b5['gap_over_sigma_n48'], color=C[2], lw=1.5, label=f"curved needle (arc of {48} segments): {b5['gap_over_sigma_n48']:.3f}")
a.axhline(np.sqrt(3) / 2, color=MUTED, ls='--', lw=1, label='single straight Gaussian: √3/2')
a.axhline(0.5, color=INK, ls=':', lw=1, label='prediction for a bent needle: 1/2')
a.set_xlabel('number of components K'); a.set_ylabel('(κ − R)/σ  at σ = 10⁻³'); a.set_ylim(0.4, 0.95); a.legend(fontsize=7, frameon=False)
a.set_title('(3) twist flow: bending the needle removes the turning part of β', fontsize=9)
plt.tight_layout(); plt.savefig('figures/stageI_flows.png', dpi=150); plt.close()
json.dump(out, open('results/summary_I.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ['fan_best', 'fan_compare', 'beta', 'predicted_gap_coefficient', 'measured_gap_coefficient_sigma0']}, indent=1, default=float)[:3000])
print(json.dumps({k: v for k, v in out['bound'].items() if k != 'by_tag'}, indent=1, default=float))
