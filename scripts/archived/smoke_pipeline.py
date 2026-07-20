#!/usr/bin/env python3
"""Smoke test for refactored kdci_pipeline. Uses H2O/STO-3G for speed."""
import sys, os
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src.kdci_pipeline import run_kdci

print("=== Small-system smoke test (H2O/STO-3G, P=100) ===")
results = run_kdci(
    system='N2', basis='cc-pVDZ', R=1.1,
    n_active=10, ne_active=(5,5), n_core=2, nroots=6,
    P_target=400, m_max=1,
    P_init=200, tie_inclusive=True,
    scoring_roots=5, batch=200,
    krylov_weight='A_half', svd_threshold=1e-3,
    do_truncation_sweep=False,
    save_krylov_basis=False,
    verbose=True,
)

# Check results
kr = results['kr_results']
print(f"\n=== Smoke Test Results ===")
print(f"P={results['p_dets'].__len__()}")
for m, r in enumerate(kr):
    print(f"m={m}: d={r['d']}, dE0={r['dE'][0]:+.3f} mH")
    for k in range(1, min(4, len(r['dE']))):
        print(f"  S{k}: dE={r['dE'][k]:+.1f} mH")
print(f"Timing: {results['timing']:.0f}s")
print("PASS" if results['timing'] < 600 else "WARN: slow")
