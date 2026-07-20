#!/usr/bin/env python3
"""Quick FCI feasibility test for CAS(14,14) N2/cc-pVDZ.
Also runs DMRG as fallback."""
import time, sys
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1

N_CORE, N_ACT = 2, 14
NROOTS = 6

print("=== CAS(14,14) FCI Feasibility Test ===", flush=True)
t0 = time.time()

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
h1a = h1[np.ix_(na_o, na_o)]
era = eri[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)

from math import comb
M = comb(N_ACT, ne[0]) * comb(N_ACT, ne[1])
print(f"Setup done ({time.time()-t0:.0f}s). M={M:,}", flush=True)

# ── Exact FCI ──
print(f"\n[1] Exact FCI (nroots={NROOTS})...", flush=True)
t1 = time.time()
try:
    ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=3)
    dt_fci = time.time() - t1
    print(f"FCI done: {dt_fci:.0f}s", flush=True)
    for k, e in enumerate(np.atleast_1d(ef)[:NROOTS]):
        print(f"  S{k}: {e:.8f} Ha", flush=True)
    fci_ok = True
except Exception as e:
    print(f"FCI FAILED: {e}", flush=True)
    fci_ok = False
    ef = None

# ── DMRG fallback (if FCI failed or as reference) ──
print(f"\n[2] DMRG-CI (M=2000, nroots={NROOTS})...", flush=True)
t2 = time.time()
try:
    from pyscf import dmrgscf
    mc = dmrgscf.DMRGCI(mf, N_ACT, ne)
    mc.fcisolver.nroots = NROOTS
    mc.fcisolver.maxM = 2000
    e_dmrg = mc.kernel()[0]
    dt_dmrg = time.time() - t2
    print(f"DMRG done: {dt_dmrg:.0f}s", flush=True)
    for k, e in enumerate(np.atleast_1d(e_dmrg)[:NROOTS]):
        diff = '' if ef is None else f'  (vs FCI: {(e-ef[k])*1e3:.2f} mH)'
        print(f"  S{k}: {e:.8f} Ha{diff}", flush=True)
    dmrg_ok = True
except Exception as e:
    print(f"DMRG FAILED: {e}", flush=True)
    dmrg_ok = False

print(f"\n{'='*50}")
if fci_ok:
    print(f"✅ FCI  works! {dt_fci:.0f}s — use exact reference")
elif dmrg_ok:
    print(f"⚠️  FCI failed. Use DMRG (M=2000) as reference")
else:
    print(f"❌ Both FCI and DMRG failed.")
print(f"Total: {time.time()-t0:.0f}s")
