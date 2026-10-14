"""Run the verification scripts of all stages (A-N) and report the total.

Each stage script re-derives every number it reports from the stored raw results and checks the
identities and inequalities used in the paper. Exit code 0 iff every check of every stage passes.
"""
import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [('A', 'run_all.py'), ('B', 'run_all_B.py'), ('C', 'run_all_C.py'), ('C+', 'run_all_C5.py'),
          ('D', 'run_all_D.py'), ('E', 'run_all_E.py'), ('F', 'run_all_F.py'), ('G', 'run_all_G.py'),
          ('H', 'run_all_H.py'), ('I', 'run_all_I.py'), ('J', 'run_all_J.py'), ('K', 'run_all_K.py'), ('L', 'run_all_L.py'), ('M', 'run_all_M.py'), ('N', 'run_all_N.py')]

total = passed = 0
failed_stages = []
t0 = time.time()
for name, script in STAGES:
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, script)], capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    m = re.findall(r'^(\d+)/(\d+) checks passed', p.stdout, re.M)
    n_ok, n = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
    total += n
    passed += n_ok
    ok = p.returncode == 0 and m and n_ok == n
    if not ok:
        failed_stages.append(name)
    print(f'=== stage {name:<2}  {n_ok}/{n} checks passed  ({time.time() - t:.0f} s)  {"OK" if ok else "FAILED"}', flush=True)

print(f'\nALL STAGES: {passed}/{total} checks passed in {time.time() - t0:.0f} s')
if failed_stages:
    print('failed stages: ' + ', '.join(failed_stages))
sys.exit(1 if failed_stages else 0)
