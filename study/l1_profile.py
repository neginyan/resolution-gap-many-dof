"""Stage L, task L1(c): the continuous fan as a functional of its angular profiles (ESTIMATE of closed forms, not derived).
The fan of i1_fan puts its segments at Chebyshev nodes x_j, so the weight per unit angle is  exp(lw(x)) / sqrt(1 - x^2)
(node density 1/(pi sqrt(1 - x^2))), not exp(lw(x)) itself.  Angle alpha = Theta x in units of sqrt(sigma)/2; the length L(alpha)
in units of 2 sqrt(sigma); thickness in units of sigma.
Fitted model families (least squares in log on the core: weight > 1e-2 of its maximum and |alpha| < 0.9 Theta; the tails carry
little weight, are weakly constrained by the objective and are distorted by the cut at |x| = 1 of the Chebyshev series):
  weight:  Gaussian  exp(-alpha^2/(2 s^2));   sech^2(alpha/s);   exp(-a alpha^2 - b alpha^4);   Laplace exp(-|alpha|/s)
  length:  L0 exp(c alpha^2);   L0 (1 + c alpha^2);   L0 cosh(c alpha)
Sources: the best stage-I fan (sigma = 1e-3) and the fans re-optimised in L1(a) (l1_fan_ext.json, kind 'fan', sigma = 1e-3, 1e-2).
-> results/l1_profile.json"""
import numpy as np, json, os
from numpy.polynomial import chebyshev as Ch
from scipy.optimize import curve_fit
from i1_fan import unpack


def profiles(x, m=801):
    lT, cw, cl, ct, co = unpack(np.array(x)); T = float(np.exp(lT))
    xs = np.cos(np.linspace(np.pi, 0, m))[1:-1]; a = T * xs
    dens = np.exp(Ch.chebval(xs, cw)) / np.sqrt(1 - xs ** 2); dens = dens / np.trapezoid(dens, a)
    return {'Theta': T, 'alpha': a, 'weight': dens, 'length': np.exp(Ch.chebval(xs, cl)), 'thickness': np.exp(Ch.chebval(xs, ct)), 'offset': Ch.chebval(xs, co)}


WM = {'gauss': (lambda a, A, s: A - a ** 2 / (2 * s ** 2), [6, 0.5]),
      'sech2': (lambda a, A, s: A - 2 * np.log(np.cosh(a / s)), [6, 0.5]),
      'quartic': (lambda a, A, b, c: A - b * a ** 2 - c * a ** 4, [6, 2, 0]),
      'laplace': (lambda a, A, s: A - np.abs(a) / s, [6, 0.3])}
LM = {'exp_a2': (lambda a, L0, c: np.log(L0) + c * a ** 2, [1, 0.3]),
      'one_plus_a2': (lambda a, L0, c: np.log(L0 * (1 + c * a ** 2)), [1, 0.3]),
      'cosh': (lambda a, L0, c: np.log(L0 * np.cosh(c * a)), [1, 0.5])}


def fit(P):
    sel = (P['weight'] > 1e-2 * P['weight'].max()) & (np.abs(P['alpha']) < 0.9 * P['Theta']); a = P['alpha'][sel]; lw = np.log(P['weight'][sel]); ll = np.log(P['length'][sel])
    out = {'Theta': P['Theta'], 'alpha_range_fitted': [float(a.min()), float(a.max())], 'weight': {}, 'length': {},
           'thickness_max_over_sigma': float(P['thickness'][sel].max()), 'offset_range_over_sigma': [float(P['offset'][sel].min()), float(P['offset'][sel].max())]}
    for k, (f, p0) in WM.items():
        p, _ = curve_fit(f, a, lw, p0=p0, maxfev=20000); r = f(a, *p) - lw
        out['weight'][k] = {'params': [float(v) for v in p], 'rms_log': float(np.sqrt(np.mean(r ** 2))), 'max_abs_log': float(np.max(np.abs(r)))}
    for k, (f, p0) in LM.items():
        p, _ = curve_fit(f, a, ll, p0=p0, maxfev=20000); r = f(a, *p) - ll
        out['length'][k] = {'params': [float(v) for v in p], 'rms_log': float(np.sqrt(np.mean(r ** 2))), 'max_abs_log': float(np.max(np.abs(r)))}
    # mass carried at angle alpha per unit length of the needle near the top: weight/length (the density of the prior along
    # the W direction at the hyperbolic point is sum_j w_j / (sqrt(2 pi) L_j))
    out['weight_over_length_peak_ratio'] = float((P['weight'][sel] / P['length'][sel]).max() / (P['weight'][sel] / P['length'][sel]).min())
    return out


if __name__ == '__main__':
    src = []
    i1 = json.load(open('results/i1_fan.json')); src.append(('stage I best fan, sigma 1e-3', max(i1, key=lambda o: o['checks']['n40'])['x']))
    if os.path.exists('results/l1_fan_ext.json'):
        for r in json.load(open('results/l1_fan_ext.json')):
            if r['kind'] == 'fan': src.append((f"L1 re-optimised fan, sigma {r['sigma']:g}", r['x'][:17]))
    res = []
    for name, x in src:
        P = profiles(x); f = fit(P); f['source'] = name; res.append(f)
        print(name, 'Theta', round(f['Theta'], 3))
        for part in ['weight', 'length']:
            print('  ', part, {k: (np.round(v['params'], 3).tolist(), round(v['rms_log'], 3)) for k, v in f[part].items()})
    json.dump(res, open('results/l1_profile.json', 'w'), indent=1)
