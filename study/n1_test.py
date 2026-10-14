"""Stage N, task N1(c): the runs that test the sealed prediction (results/n1_prediction.json, hash in logs/n1_prediction_hash.txt;
checked before running): eps = 1 and -0.3, K = 8, with exactly the procedure of n1_mix (climb 4000 + L-BFGS 400) and of
n1_more 'from1d' (polished 1-d K = 8 start + L-BFGS 1000); the better of the two is the result.   -> results/n1_test.json"""
import json, hashlib
from multiprocessing import Pool
import n1_mix, n1_more

HASH = open('logs/n1_prediction_hash.txt').read().split('=')[-1].strip().split()[0]


def run(a):
    mode, kind = a
    return n1_mix.job((kind, 8, 4000, 400)) if mode == 'first' else n1_more.job(('from1d', kind, 8, 1000))


if __name__ == '__main__':
    assert hashlib.sha256(open('results/n1_prediction.json', 'rb').read()).hexdigest() == HASH
    P = {p['eps']: p for p in json.load(open('results/n1_prediction.json'))['predictions']}
    with Pool(2) as pool: res = pool.map(run, [(m, k) for k in ['1', '-0.3'] for m in ['first', 'from1d']])
    out = []
    for k in ['1', '-0.3']:
        r = min([x for x in res if x['kind'] == k], key=lambda x: x['gap']); p = P[float(k)]
        out.append({'eps': float(k), 'runs': [{kk: x[kk] for kk in ['gap', 'ratio']} | {'mode': x.get('mode', 'first')} for x in res if x['kind'] == k], 'opt_ratio': r['ratio'],
                    'predicted_ratio': p['predicted_ratio'], 'predicted_range': p['predicted_ratio_range'], 'effective_beta_ratio': 1 - p['effective_beta_gain'], 'x': r['x']})
        print(out[-1]['eps'], out[-1]['runs'], 'predicted', p['predicted_ratio'], p['predicted_ratio_range'], flush=True)
    json.dump(out, open('results/n1_test.json', 'w'), indent=1, default=float)
