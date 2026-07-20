# State-Average Krylov-dCI Workflow (v10_sacis)

Validated excited-state pipeline for N2 CAS(10,10)/cc-pVDZ. Computes the ground
state **and low-lying excited states** in one state-average pass.

## What it does / the key fix (2026-07-13)

Excited states (esp. triplets) are **single-excitation dominated**. Brillouin's
theorem zeroes `<HF|H|singles>`, so the old HFPT2 seed (ground-state PT2) never
selected single excitations → excited roots of `H_PP` were wrong (dE ~ +600 mH).

**Fix:** force **all single excitations** into the initial P-space seed, and label
every root by `<S^2>` so singlet/triplet character is explicit.

## Run (CLI)

```bash
export PATH=/data/home/wangcx/LiYF4_Er3+/env/bin:$PATH
python scripts_new/phaseA_cas10_v10_sacis.py \
    --P 400,800,1200,2000 \
    --m-max 1 \
    --tag mytag
```

Args: `--P` comma-separated P checkpoints, `--m-max` Krylov depth, `--batch`
selection batch (default 200), `--svd-threshold` (default 1e-3), `--tag`.

Output JSON: `checkpoints_phaseA/phaseA_v8_m<M>_svd<thr>_<tag>.json`.

## Validated result (job 15239, P=2000, m=1)

```
       S0     S1     S2     S3     S4     S5
dE    +0.0   +0.8   +0.8   +1.0   +2.5   +42.7  mH
<S2>  0.00   2.00   2.00   0.00   2.00   0.00
FCI: S0/S3 singlet (<S2>=0), S1/S2/S4/S5 triplet (<S2>=2)
```

First three states (S0,S1,S2) converge to **< 1 mH**. (S5 is a window-edge
state-ordering mismatch; not yet tracked.)

## Smoke test

```bash
python tests/smoke_sacis.py
```

Fast check (P=400, m=1, ~2 min): asserts the pipeline runs, the per-root table is
produced, ground state is reasonable, and S1/S2 come out as triplets (<S^2>~2).
Full sub-mH accuracy requires P>=2000.

## Follow-up (not done yet)

- Refactor the monolithic script into importable functions (`run_phaseA(...)`).
- overlap/`<S^2>` state tracking for window-edge / cross-multiplicity states.
