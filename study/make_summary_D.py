"""Fill the distance-sweep table of the stage-D summary from results/summary_D.json (no hand-copied numbers)."""
import json
t = json.load(open('results/summary_D.json'))['task3']
rows = [f"| {b['dist']:.4g} | {b['excess']:.3g} | {b['ratio']:.3f} | {b['abar_var']:.2g} | {b['focus_mismatch']:.2f} |"
        for b in sorted(t['best_by_target'], key=lambda b: -b['dist'])]
s = open('notes/本筋用要約_観測D.template.md').read().replace('{SWEEP_TABLE}', '\n'.join(rows))
open('notes/本筋用要約_観測D.md', 'w').write(s); print('written')
