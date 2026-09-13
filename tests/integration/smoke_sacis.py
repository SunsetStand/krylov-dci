#!/usr/bin/env python3
"""Smoke test for the state-average Krylov-dCI workflow (v10_sacis).

Fast (P=400, m=1, ~2 min): runs the CLI and checks that
  (1) it exits cleanly,
  (2) the per-root summary table is produced,
  (3) the ground state is reasonable (|dE0| small),
  (4) the S1/S2 excited roots come out as triplets (<S^2> ~ 2)  -> the singles-seed fix works.

Full sub-mH accuracy needs P>=2000; this only guards against regressions in the
seed construction / spin character, not convergence.

Usage:  python tests/integration/smoke_sacis.py
Exit code 0 = pass.
"""
import os, re, sys, subprocess

# Paths are environment-overridable so the test runs locally as well as on the
# cluster.  The defaults resolve against this checkout and the running
# interpreter rather than a hard-coded cluster location.
ROOT = os.environ.get(
    'KRYLOV_DCI_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PY = os.environ.get('KRYLOV_DCI_PYTHON', sys.executable)
# NOTE: the repository reorganization moved this driver into scripts/archived/,
# which CONTRIBUTING.md marks as historical rather than a current API.  This
# smoke test therefore needs a maintained replacement entry point; until then it
# is pointed at the archived driver so that it can run at all.
SCRIPT = os.environ.get(
    'KRYLOV_DCI_SACIS_SCRIPT',
    os.path.join(ROOT, 'scripts', 'archived', 'phaseA_cas10_v10_sacis.py'))

def main():
    cmd = [PY, SCRIPT, '--P', '400', '--m-max', '1', '--tag', 'smoke']
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    out = r.stdout
    if r.returncode != 0:
        print(out[-2000:]); print("STDERR:", r.stderr[-1000:])
        print("FAIL: non-zero exit"); return 1

    fails = []
    if 'Per-root dE' not in out:
        fails.append("per-root summary table missing")

    # last bare-H_PP <S^2> label line (S1/S2 should be triplets ~2.0)
    s2lines = re.findall(r'\[H_PP bare\].*', out)
    if not s2lines:
        fails.append("no [H_PP bare] <S^2> labels found")
    else:
        last = s2lines[-1]
        s2 = [float(x) for x in re.findall(r'<S2>=([0-9.]+)', last)]
        if len(s2) < 3:
            fails.append(f"could not parse >=3 <S^2> values from: {last}")
        else:
            # among the low excited roots at least two triplets (~2.0) should appear
            n_trip = sum(1 for v in s2[1:] if abs(v - 2.0) < 0.2)
            if n_trip < 2:
                fails.append(f"expected >=2 triplet excited roots, got <S^2>={s2}")

    # ground-state bare dE should be modest at P=400
    m = re.findall(r'dE0\(bare\)=([+-][0-9.]+) mH', out)
    if m:
        dE0 = abs(float(m[-1]))
        if dE0 > 200:
            fails.append(f"ground-state dE0(bare)={dE0} mH too large")
    else:
        fails.append("no dE0(bare) found")

    if fails:
        print(out[-1500:])
        for f in fails: print("FAIL:", f)
        return 1
    print("PASS: pipeline ran, table produced, excited triplets present, ground state sane.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
