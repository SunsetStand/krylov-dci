#!/usr/bin/env python3
"""DMRG-CI reference for N2/cc-pVDZ CAS(20,10) R=1.1."""
import numpy as np, time, sys
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1

N_CORE, N_ACT = 2, 20
NROOTS = 6
R = 1.1

print(f"=== DMRG Ref: N2/cc-pVDZ CAS({N_ACT},10) R={R} ===", flush=True)
t0 = time.time()

mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mol.spin = 0  # fix for DMRGCI
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

# DMRG-CI via Block2
print(f"\n[1] DMRG-CI (M=2000, nroots={NROOTS}, block2)...", flush=True)
t1 = time.time()
try:
    from pyscf import dmrgscf
    mc = dmrgscf.DMRGCI(mf, N_ACT, ne)
    mc.fcisolver.nroots = NROOTS
    mc.fcisolver.maxM = 2000
    mc.fcisolver.tol = 1e-6
    e_dmrg = mc.kernel()[0]
    dt = time.time() - t1
    print(f"  DMRG done: {dt:.0f}s", flush=True)
    for k, e in enumerate(np.atleast_1d(e_dmrg)[:NROOTS]):
        print(f"    S{k}: {e:.12f} Ha", flush=True)
    np.savez('/data/home/wangcx/krylov-dci/checkpoints_cas20/dmrg_ref.npz',
             e_dmrg=np.atleast_1d(e_dmrg)[:NROOTS], nroots=NROOTS, maxM=2000)
except Exception as e:
    print(f"  DMRG FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

if not os.path.exists('/data/home/wangcx/krylov-dci/checkpoints_cas20/dmrg_ref.npz'):
    print(f"\n[2] Fallback: FCI...", flush=True)
    try:
        ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=3)
        np.savez('/data/home/wangcx/krylov-dci/checkpoints_cas20/dmrg_ref.npz',
                 e_dmrg=np.atleast_1d(ef)[:NROOTS], nroots=NROOTS, maxM=0)
        print(f"  FCI done: {time.time()-t1:.0f}s", flush=True)
    except Exception as e:
        print(f"  FCI FAILED: {e}", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
