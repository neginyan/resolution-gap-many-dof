"""Stage C, proposal 1 report: table of the adversarial states (R/kappa, valley mass, overlap, D, covariance term, and how much of
the covariance term and of D sits inside the valley), and figures/stageC_adversarial.png:
 (left) law term vs excess (= cov + D) for every state computed so far, with the line R = kappa;
 (right three) the three largest-R states: local stretch, density, components and valley.
Writes results/c1_table.json."""
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from obs_eval2 import evaluate_full
from c1_adversarial import unpack

INK, MUTED = '#1f1f1e', '#6b6a65'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})
ST = json.load(open('results/c1_adversarial.json')) + json.load(open('results/c1_adversarial_room.json'))
rows, fields = [], {}
for i, c in enumerate(ST):
    ws, mus, Ss = unpack(np.array(c['x']), c['K'])
    n = int(min(c['n'], 1201))                                    # fields at <= 1201 points per axis (memory); R compared with full grid
    r = evaluate_full(ws, mus, Ss, c['sigma'], n=n, return_fields=True); F = r['fields']
    dA = (F['q'][1] - F['q'][0]) * (F['p'][1] - F['p'][0]); val = F['nonPD'] > 0.5
    row = {k: c[k] for k in ['K', 'sigma', 'target', 'R', 'R_n1.5', 'law_term', 'cov', 'D', 'H_nonPD_mass', 'overlap_max', 'lam_rho_direct', 'lam_comp', 'room']}
    row.update({'R_fieldgrid_minus_full': float(r['R'] - c['R']),
                'cov_in_valley': float(np.nansum(np.where(val, F['cov_density'], 0)) * dA),
                'D_in_valley': float(np.nansum(np.where(val, F['D_density'], 0)) * dA)})
    rows.append(row); fields[i] = (ws, mus, Ss, c['sigma'], F)
json.dump(rows, open('results/c1_table.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
a = ax[0]
others = json.load(open('results/m2_mix.json')) + json.load(open('results/m1_slices.json')) + json.load(open('results/m1_maps.json'))
a.plot([d['law_term'] for d in others], [d['R'] - d['law_term'] for d in others], 'o', ms=2.2, color=MUTED, alpha=0.35, label='random mixtures, measurement-1 states')
sc = a.scatter([r['law_term'] for r in rows], [r['R'] - r['law_term'] for r in rows], c=[r['H_nonPD_mass'] for r in rows], cmap='viridis',
               s=40, edgecolor=INK, linewidth=0.6, zorder=5, label='stage-C adversarial states')
xx = np.linspace(-0.8, 1, 10); a.plot(xx, 1 - xx, '-', color=C[1], lw=2, label='R = κ (law + excess = 1)')
a.set_xlabel('law term  tr(F Sym $E_\\rho Df$)/tr F'); a.set_ylabel('excess = covariance term + D')
a.set_title('Large excess appears only where the law term is small', fontsize=9); a.legend(fontsize=7, frameon=False, loc='lower left')
fig.colorbar(sc, ax=a, fraction=0.046, label='valley mass')
top = sorted(range(len(rows)), key=lambda i: -rows[i]['R'])[:3]
for a, i in zip(ax[1:], top):
    ws, mus, Ss, sig, F = fields[i]; Q, P = np.meshgrid(F['q'], F['p'], indexing='ij')
    pY = np.nan_to_num(F['pY']); m = pY > pY.max() * 1e-4
    a.pcolormesh(Q, P, np.where(m, F['a_eff'], np.nan), cmap='Blues', vmin=0, vmax=1, shading='auto')
    a.contour(Q, P, pY, levels=pY.max() * np.array([0.01, 0.05, 0.2, 0.5, 0.8]), colors=INK, linewidths=0.6)
    a.contourf(Q, P, np.where(m, F['nonPD'], 0), levels=[0.5, 1.5], colors='none', hatches=['////'])
    a.contour(Q, P, np.where(m, F['nonPD'], 0), levels=[0.5], colors=C[1], linewidths=1.2)
    t = np.linspace(0, 2 * np.pi, 100)
    for w, mu, S in zip(ws, mus, Ss):
        ev, U = np.linalg.eigh(S); E = U @ np.diag(2 * np.sqrt(ev)) @ np.stack([np.cos(t), np.sin(t)])
        a.plot(mu[0] + E[0], mu[1] + E[1], '--', color=C[3], lw=1.3)
    win = m.any(1); winp = m.any(0)
    a.set_xlim(F['q'][win].min(), F['q'][win].max()); a.set_ylim(F['p'][winp].min(), F['p'][winp].max())
    a.axvline(np.pi, color=MUTED, lw=0.8, ls=':')
    r = rows[i]
    a.set_title(f"R/κ = {r['R']:.4f}  (K={r['K']}, σ={r['sigma']})\nlaw {r['law_term']:.3f}, cov {r['cov']:+.3f}, D {r['D']:+.4f}, "
                f"valley {r['H_nonPD_mass']:.3f}, overlap {r['overlap_max']:.2f}", fontsize=8.5)
    a.set_xlabel('q (dotted: hyperbolic point π)'); a.set_ylabel('p')
ax[1].plot([], [], '--', color=C[3], label='components (2 s.d.)'); ax[1].plot([], [], '-', color=C[1], label='valley edge (hatched inside)')
ax[1].legend(fontsize=7, frameon=True, loc='lower left')
fig.tight_layout(); fig.savefig('figures/stageC_adversarial.png', dpi=140)
for r in sorted(rows, key=lambda r: -r['R']):
    print(f"K={r['K']} s={r['sigma']:<5} {r['target']:<6} R={r['R']:.4f} law={r['law_term']:+.3f} cov={r['cov']:+.3f} D={r['D']:+.4f} "
          f"valley={r['H_nonPD_mass']:.3f} ov={r['overlap_max']:.3f} cov_valley={r['cov_in_valley']:+.4f} D_valley={r['D_in_valley']:+.4f} dR={r['R_fieldgrid_minus_full']:.1e}")
