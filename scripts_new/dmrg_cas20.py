#!/usr/bin/env python3
"""DMRG-CI reference for N2/cc-pVDZ CAS(20,10) R=1.1."""
import numpy as np, time, sys, os
from pyscf import gto, scf, mcscf, dmrgscf

N_CORE, N_ACT = 2, 20
NROOTS, R = 6, 1.1

print(f"=== DMRG Ref: N2/cc-pVDZ CAS({N_ACT},10) R={R} ===", flush=True)
t0 = time.time()

mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mol.spin = 0
mf = scf.RHF(mol).run(verbose=0)
print(f"  HF: {mf.e_tot:.8f} Ha", flush=True)

na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = mol.intor('int2e').reshape(norb,norb,norb,norb)
# Transform to MO
from pyscf import ao2mo
eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
from math import comb
M = comb(N_ACT, ne[0]) * comb(N_ACT, ne[1])
print(f"  M={M:,} ({M/1e6:.1f}M)", flush=True)

print(f"\n[1] DMRG-CI (M=2000, nroots={NROOTS})...", flush=True)
t1 = time.time()
try:
    mc = mcscf.CASCI(mf, N_ACT, ne)
    mc.fcisolver = dmrgscf.DMRGCI(mf.mol, maxM=2000, tol=1e-6)
    mc.fcisolver.block_extra_keyword = ["stack_mem 300000000"]
    mc.fcisolver.nroots = NROOTS
    e_dmrg = mc.kernel()[0]
    dt = time.time()-t1
    print(f"  DMRG done: {dt:.0f}s", flush=True)
    # mc.e_cas should have the state energies
    e_states = mc.e_cas if hasattr(mc, 'e_cas') else mc.fcisolver.e
    for k, e in enumerate(np.atleast_1d(e_states)[:NROOTS]):
        print(f"    S{k}: {e:.12f} Ha  ({e*27.2114:.6f} eV)", flush=True)
    os.makedirs('/data/home/wangcx/krylov-dci/checkpoints_cas20', exist_ok=True)
    np.savez('/data/home/wangcx/krylov-dci/checkpoints_cas20/dmrg_ref.npz',
             e_dmrg=np.atleast_1d(e_states)[:NROOTS], nroots=NROOTS, maxM=2000)
    print(f"  Reference saved.", flush=True)
except Exception as e:
    print(f"  DMRG FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
