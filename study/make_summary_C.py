"""Fill the stage-C adversarial table into the Japanese summary from results/summary_C.json (no hand-copied numbers)."""
import json
s = json.load(open('results/summary_C.json'))
tgt = {'R': 'R', 'excess': '超過分', 'room': '余裕比'}
rows = ['| R/κ | K | σ | 目的 | 法則項 | 共分散項 | D | 谷間の質量 | 重なり | 余裕比 | 谷間の共分散 |', '|---|---|---|---|---|---|---|---|---|---|---|']
for r in s['c1']['table']:
    rows.append(f"| {r['R']:.4f} | {r['K']} | {r['sigma']} | {tgt[r['target']]} | {r['law_term']:+.3f} | {r['cov']:+.3f} | {r['D']:+.4f} | "
                f"{r['H_nonPD_mass']:.3f} | {r['overlap_max']:.2f} | {r['room']:.3f} | {r['cov_in_valley']:+.3f} |")
t = open('notes/本筋用要約_観測C.template.md').read().replace('{TABLE}', '\n'.join(rows))
open('notes/本筋用要約_観測C.md', 'w').write(t)
print('written')
