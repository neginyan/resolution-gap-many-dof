"""Fill the chain table of the stage-F summary from results/summary_F.json (no hand-copied numbers)."""
import json
s = json.load(open('results/summary_F.json'))
rows = [f"| {c['name']} | {c['start']:.4f} | {c['ratio']:.4f} | {c['dist']:.3f} | {c['R']:.4f} |" for c in sorted(s['chains'], key=lambda c: -c['ratio'])]
t = open('notes/本筋用要約_観測F.template.md').read().replace('{CHAIN_TABLE}', '\n'.join(rows))
open('notes/本筋用要約_観測F.md', 'w').write(t); print('written')
