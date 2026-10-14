"""Stage C, proposal 3: where does the covariance term come from?  Maps of the local stretch E[a|y], the local focus anisotropy
(v^T H v - w^T H w)/tr H, the valley (H not positive definite) and the covariance-term density, for
  A: the measurement-1 base state (concentric, width ratio 4.4, sigma 0.1),
  B: the stage-A counterexample (sigma 0.1),
  C: the largest-R state of the stage-C adversarial search.
Profiles along the long axis W through the weighted centre: p_Y-weighted means per bin of u = (y - c) . W.
Writes figures/stageC_spatial.png and results/c3_spatial.json (numbers quoted in the summary)."""
import numpy as np, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from obs_eval2 import evaluate_full, V, W
from m1_mechanism import state
from adversarial import unpack as unpack_A
from c1_adversarial import unpack as unpack_C

INK, MUTED = '#1f1f1e', '#6b6a65'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})

a = json.load(open('results/adversarial_ratio_0.1.json'))['ratio_0.1']
c1 = json.load(open('results/c1_adversarial.json')); best = max(c1, key=lambda r: r['R'])
states = [('A: measurement-1 base (concentric)', state(0.0), 0.1),
          ('B: stage-A counterexample', unpack_A(np.array(a['x'])), 0.1),
          (f"C: largest R of stage C (K={best['K']})", unpack_C(np.array(best['x']), best['K']), best['sigma'])]
out = {}
fig, ax = plt.subplots(3, 4, figsize=(17, 13.5))
for row, (name, (ws, mus, Ss), sig) in enumerate(states):
    r = evaluate_full(ws, mus, Ss, sig, n=901, return_fields=True); F = r['fields']
    Q, P = np.meshgrid(F['q'], F['p'], indexing='ij'); dA = (F['q'][1] - F['q'][0]) * (F['p'][1] - F['p'][0])
    pY = np.nan_to_num(F['pY']); m = pY > pY.max() * 1e-4
    lev = pY.max() * np.array([0.01, 0.05, 0.2, 0.5, 0.8])
    win = m.any(1); wq = F['q'][win]; winp = m.any(0); wp = F['p'][winp]
    ext = lambda A_: A_.set(xlim=(wq.min(), wq.max()), ylim=(wp.min(), wp.max()))
    A_ = ax[row, 0]
    im = A_.pcolormesh(Q, P, np.where(m, F['a_eff'], np.nan), cmap='Blues', vmin=0, vmax=1, shading='auto')
    A_.contour(Q, P, pY, levels=lev, colors=INK, linewidths=0.7); ext(A_); fig.colorbar(im, ax=A_, fraction=0.046)
    A_.set_title(f'{name}\nlocal stretch E[a|y] (blue) + density $p_Y$ (lines)', fontsize=8.5); A_.set_ylabel('p')
    A_ = ax[row, 1]
    fa = np.where(m, F['focus_aniso'], np.nan)
    im = A_.pcolormesh(Q, P, np.clip(fa, -1.5, 1.5), cmap='PuOr_r', norm=TwoSlopeNorm(0, -1.5, 1.5), shading='auto')
    A_.contourf(Q, P, np.where(m, F['nonPD'], 0), levels=[0.5, 1.5], colors='none', hatches=['////'])
    A_.contour(Q, P, np.where(m, F['nonPD'], 0), levels=[0.5], colors=INK, linewidths=1.0)
    ext(A_); fig.colorbar(im, ax=A_, fraction=0.046)
    A_.set_title('local focus anisotropy $(v^THv-w^THw)/\\mathrm{tr}H$\n(hatched: valley, H not positive definite)', fontsize=8.5)
    A_ = ax[row, 2]
    cd = np.where(m, F['cov_density'], np.nan); vmax = np.nanmax(np.abs(cd))
    im = A_.pcolormesh(Q, P, cd, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), shading='auto')
    A_.contour(Q, P, pY, levels=lev, colors=MUTED, linewidths=0.5); ext(A_); fig.colorbar(im, ax=A_, fraction=0.046)
    A_.set_title(f"covariance-term density (red adds to R)\ncov = {r['f'] - r['law_term']:+.4f}, D = {r['D']:+.4f}, R = {r['R']:.4f}", fontsize=8.5)
    # profile along W through the p_Y-weighted centre
    cq, cp = np.nansum(pY * Q) / pY.sum(), np.nansum(pY * P) / pY.sum()
    u = (Q - cq) * W[0] + (P - cp) * W[1]
    bins = np.linspace(np.percentile(u[m], 1), np.percentile(u[m], 99), 31); mid = (bins[1:] + bins[:-1]) / 2
    idx = np.digitize(u, bins) - 1
    prof = {k: [] for k in ['mass', 'a', 'aniso', 'cov']}
    for b in range(30):
        sel = (idx == b) & m
        wb = pY[sel]; prof['mass'].append(wb.sum() * dA)
        prof['a'].append(np.sum(wb * F['a_eff'][sel]) / wb.sum() if wb.sum() > 0 else np.nan)
        prof['aniso'].append(np.sum(wb * np.nan_to_num(F['H_minus_F_aniso'][sel])) / wb.sum() if wb.sum() > 0 else np.nan)
        prof['cov'].append(np.nansum(F['cov_density'][sel]) * dA)
    A_ = ax[row, 3]
    A_.plot(mid, prof['a'], '-', color=C[0], lw=2, label='mean local stretch E[a|y]')
    A_.plot(mid, prof['aniso'], '--', color=C[1], lw=2, label='mean focus excess (H − F anisotropy)/tr F')
    A2 = A_.twinx(); A2.bar(mid, prof['cov'], width=(bins[1] - bins[0]) * 0.8, color=C[2], alpha=0.45, label='covariance term per bin')
    A2.axhline(0, color=MUTED, lw=0.6); A2.set_ylabel('covariance term per bin', color=C[2])
    A_.set_xlabel('position along the contracting direction W'); A_.set_title('profile along W', fontsize=8.5)
    h1, l1 = A_.get_legend_handles_labels(); h2, l2 = A2.get_legend_handles_labels(); A_.legend(h1 + h2, l1 + l2, fontsize=7, frameon=True, framealpha=0.9, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=1)
    covtot = r['f'] - r['law_term']; covb = np.array(prof['cov']); um = np.abs(mid)
    half = np.percentile(np.abs(u[m]), 50)
    # share of the covariance term from the outer half (|u| above the p_Y-median of |u|)
    outer = np.nansum(np.where(np.abs(u) > half, np.nan_to_num(F['cov_density']), 0)) * dA
    out[name[0]] = {'R': r['R'], 'law': r['law_term'], 'cov': covtot, 'D': r['D'], 'valley_mass': r['H_nonPD_mass'],
                    'cov_outer_half': float(outer), 'cov_inner_half': float(covtot - outer),
                    'corr_a_aniso': float(np.nansum(pY * (F['a_eff'] - np.nansum(pY * F['a_eff']) / pY.sum()) * np.nan_to_num(F['H_minus_F_aniso'])) /
                                          np.sqrt(np.nansum(pY * (F['a_eff'] - np.nansum(pY * F['a_eff']) / pY.sum()) ** 2) *
                                                  np.nansum(pY * (np.nan_to_num(F['H_minus_F_aniso']) - np.nansum(pY * np.nan_to_num(F['H_minus_F_aniso'])) / pY.sum()) ** 2))),
                    'cov_in_valley': float(np.nansum(np.where(F['nonPD'] > 0.5, F['cov_density'], 0)) * dA),
                    'D_in_valley': float(np.nansum(np.where(F['nonPD'] > 0.5, F['D_density'], 0)) * dA)}
for A_ in ax[2, :3]: A_.set_xlabel('q')
fig.tight_layout(); os.makedirs('figures', exist_ok=True); fig.savefig('figures/stageC_spatial.png', dpi=140)
json.dump(out, open('results/c3_spatial.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
