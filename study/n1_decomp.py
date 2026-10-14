"""Stage N, task N1(b): where does the gain of the mixtures come from in the coupled pendulums?
For each best product mixture M (n1_mix / n1_more) and the best product Gaussian G (n1_gauss):
  hybrid H = the plane mixture of M with the b-law (s_b, theta) of G.
  gain_plane = gap(G) - gap(H)   (the plane mixture alone, q- law unchanged)
  gain_b     = gap(H) - gap(M)   (re-tuning the q- law: width s_b and rotation theta)
At fixed b-law the gap is  kappa - (num2 + extra)/(sigma^2 (J2 + J_b)),  extra = the q- cross term (zero for theta = 0).  The change
of the gap from G to H is split exactly into
  d_plane = the part from (num2, J2) with extra frozen at its value for G,   d_cross = the part from the change of extra.
d_cross < 0 means that the plane mixture raises the q- cost (hypothesis (2) of stage M); gain_plane = d_plane + d_cross.  All gaps in units of sigma.
The 1-d reference: the same plane mixture in the 1-d pendulum has gain g1 (n1_mix, '1d', same K) and the plane share f of the
Gaussian coefficient; "pure dilution" would give gain_plane = f g1 x (gap of G).   -> results/n1_decomp.json"""
import numpy as np, json
from n_tools import Flow, R_product, unpack

SIG = 1e-3


def best_states():
    runs = json.load(open('results/n1_mix.json')) + json.load(open('results/n1_more.json')); b = {}
    for o in runs:
        k = (o['kind'], o['K'])
        if k not in b or o['gap'] < b[k]['gap']: b[k] = o
    return b


if __name__ == '__main__':
    B = best_states(); G = {o['kind']: o for o in json.load(open('results/n1_gauss.json'))}; out = []
    from m_tools import coupled, find_kappa, coefficient
    for e in ['0.1', '0.3']:
        fl = Flow(float(e), SIG); s = coupled(float(e)); k, t = find_kappa(s, starts=50); c = coefficient(s, t[0], k); f = c['diag_terms'][0] / c['coef']
        wg, mg, Sg, sbg, thg = unpack(np.array(G[e]['x']), 1, fl)
        Rg, pg = R_product(fl, wg, mg, Sg, sbg, thg, per_sd=6, parts=True); gG = (fl.k - Rg) / SIG
        for K in sorted({kk[1] for kk in B if kk[0] == e}):
            wm, mm, Sm, sbm, thm = unpack(np.array(B[(e, K)]['x']), K, fl)
            Rm, pm = R_product(fl, wm, mm, Sm, sbm, thm, per_sd=6, parts=True); gM = (fl.k - Rm) / SIG
            Rh, ph = R_product(fl, wm, mm, Sm, sbg, thg, per_sd=6, parts=True); gH = (fl.k - Rh) / SIG
            den = SIG ** 2 * (ph['J2'] + ph['Jb'])
            g_frozen = (fl.k - (ph['num2'] + pg['extra']) / den) / SIG
            g1 = 1 - B[('1d', K)]['ratio'] if ('1d', K) in B else None
            out.append({'eps': e, 'K': K, 'plane_share_f': f, 'gap_G': gG, 'gap_H': gH, 'gap_M': gM, 'gain_total': (gG - gM) / gG, 'gain_plane': (gG - gH) / gG,
                        'gain_b': (gH - gM) / gG, 'd_plane': (gG - g_frozen) / gG, 'd_cross': (g_frozen - gH) / gG, 'gain_1d': g1,
                        'dilution_prediction': None if g1 is None else f * g1, 's_b_G': sbg / SIG, 's_b_M': sbm / SIG, 'theta_G': thg, 'theta_M': thm,
                        'extra_G': pg['extra'], 'extra_H': ph['extra']})
            o = out[-1]; print(e, K, {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in o.items() if kk not in ('extra_G', 'extra_H')}, flush=True)
    json.dump(out, open('results/n1_decomp.json', 'w'), indent=1, default=float)
