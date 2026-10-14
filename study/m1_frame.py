"""Stage M, task M1: the precise optimum of the coupled pendulums for every eps of M1 (0.1, 0.3, 1, -0.1, -0.3, 2, 3) by the
full-parameter scaled-frame optimisation (m_tools.frame_series: leading-order start and continuation in sigma) at sigma = 1e-4, 1e-5, 1e-6, quadratic extrapolation to sigma -> 0,
and comparison with the formula; also the reference against which the assumption-free runs of m1_free / m1_range are compared
(none of them may go below it).   -> results/m1_frame.json"""
import numpy as np, json
from m_tools import coupled, find_kappa, coefficient, frame_series

if __name__ == '__main__':
    out = []
    for e in [0.1, 0.3, 1.0, -0.1, -0.3, 2.0, 3.0]:
        s = coupled(e); k, tops = find_kappa(s, starts=200, seed=5); c = coefficient(s, tops[0], k)['coef']; g, _ = frame_series(s, tops[0], k)
        sg = np.array([1e-4, 1e-5, 1e-6]); p = np.polyfit(sg, [g[f'{v:g}'] for v in sg], 2)
        out.append({'eps': e, 'kappa': k, 'top': tops[0].tolist(), 'formula': c, 'opt': g, 'extrap_sigma0': float(p[-1]), 'extrap_slope': float(p[-2]),
                    'opt_rel_diff_extrap_vs_formula': float((p[-1] - c) / c)})
        print(e, k, np.round(tops[0], 5), 'formula', round(c, 8), 'opt', {kk: round(v, 8) for kk, v in g.items()}, 'extrap', round(p[-1], 8), 'rel', f'{(p[-1] - c) / c:.1e}', flush=True)
    json.dump(out, open('results/m1_frame.json', 'w'), indent=1)
