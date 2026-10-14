"""Stage M, task M2: refresh part (B) of results/m2_opt.json with m_tools.frame_series (leading-order start + continuation in sigma),
keeping the random starts (A).  Prediction file hash checked first."""
import json, hashlib, numpy as np
from m_tools import frame_series, frame
from m2_predict import SYSTEMS
from m2_opt import HASH, shape

if __name__ == '__main__':
    assert hashlib.sha256(open('results/m2_prediction.json', 'rb').read()).hexdigest() == HASH
    out = json.load(open('results/m2_opt.json'))
    for o in out:
        idx = [i for i, (kd, n, e, s) in enumerate(SYSTEMS) if (kd, n, e) == (o['kind'], o['n'], o['eps'])][0]; s = SYSTEMS[idx][3]
        P = json.load(open('results/m2_prediction.json'))['systems'][idx]; q0 = np.array(P['tops'][0])
        fr, last = frame_series(s, q0, P['kappa']); f5, st5 = frame_series(s, q0, P['kappa'], (1e-4, 1e-5))
        sg = np.array([1e-4, 1e-5, 1e-6]); p = np.polyfit(sg, [fr[f'{v:g}'] for v in sg], 2)
        o['frame_opt'] = fr; o['extrap_sigma0'] = float(p[-1]); o['extrap_slope'] = float(p[-2]); o['opt_rel_diff_extrap_vs_prediction'] = float((p[-1] - o['prediction']) / o['prediction'])
        o['random_below_frame_opt_1e-5'] = int(sum(r['vals']['1e-05'] < fr['1e-05'] * (1 - 1e-7) for r in o['random']))
        o['random_reaching_1e-4_rel'] = int(sum(abs(r['vals']['1e-05'] - fr['1e-05']) < 1e-4 * fr['1e-05'] for r in o['random']))
        o['shape_1e-5'] = shape(s, np.concatenate([st5[0], st5[1].ravel()]), q0, 1e-5)
        print(o['name'], o['prediction'], fr, o['extrap_sigma0'], f"{o['opt_rel_diff_extrap_vs_prediction']:.1e}", o['random_below_frame_opt_1e-5'], flush=True)
    json.dump(out, open('results/m2_opt.json', 'w'), indent=1)
