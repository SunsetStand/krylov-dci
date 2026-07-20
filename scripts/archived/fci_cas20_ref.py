#!/usr/bin/env python3
"""FCI reference for CAS(20,10) — single-root first, then 6 roots."""
import numpy as np, time, sys, os
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1

N_CORE, N_ACT = 2, 20
NROOTS, R = 6, 1.1

print(f"=== FCI Ref: N2/cc-pVDZ CAS({N_ACT},10) R={R} ===", flush=True)
t0 = time.time()

mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
print(f"  HF: {mf.e_tot:.8f}", flush=True)

na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
h1a = h1[np.ix_(na_o, na_o)]
era = eri[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
from math import comb
M = comb(N_ACT, ne[0]) * comb(N_ACT, ne[1])
print(f"  M={M:,} ({M/1e6:.1f}M)", flush=True)

# Check available memory
import psutil
avail = psutil.virtual_memory().available / 1e9
print(f"  Available RAM: {avail:.1f} GB", flush=True)

# Try single root first
print(f"\n[1] FCI nroots=1 (sanity check)...", flush=True)
t1 = time.time()
try:
    e1, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=1, verbose=4)
    dt1 = time.time()-t1
    print(f"  FCI(1 root) = {e1:.10f} Ha  [{dt1:.0f}s]", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    sys.exit(1)

# Now 6 roots
print(f"\n[2] FCI nroots={NROOTS}...", flush=True)
t2 = time.time()
try:
    ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=4)
    dt2 = time.time()-t2
    print(f"  FCI({NROOTS} roots) done [{dt2:.0f}s]", flush=True)
    for k, e in enumerate(np.atleast_1d(ef)[:NROOTS]):
        print(f"    S{k}: {e:.12f} Ha", flush=True)
    os.makedirs('/data/home/wangcx/krylov-dci/checkpoints_cas20', exist_ok=True)
    np.savez('/data/home/wangcx/krylov-dci/checkpoints_cas20/dmrg_ref.npz',
             e_dmrg=np.atleast_1d(ef)[:NROOTS], nroots=NROOTS, maxM=0)
    print(f"  Reference saved.", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
