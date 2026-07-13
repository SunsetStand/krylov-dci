#!/usr/bin/env python3
"""DMRG-CI reference for N2/cc-pVDZ CAS(20,10) R=1.1."""
import numpy as np, time, os
from pyscf import gto, scf, mcscf
from pyscf.dmrgscf import DMRGCI
import block2

NTHREADS = int(os.environ.get("OMP_NUM_THREADS", 8))
print(f"  OMP_NUM_THREADS={NTHREADS}", flush=True)
block2.set_omp_num_threads(NTHREADS)
block2.set_mkl_num_threads(NTHREADS)

N_CORE, N_ACT = 2, 20
NROOTS, R = 6, 1.1
MEM_GB = 60  # Block2 memory in GB (PySCF passes as GB)

print(f"=== DMRG Ref: N2/cc-pVDZ CAS({N_ACT},10) R={R} ===", flush=True)
t0 = time.time()

mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pVDZ", verbose=0)
mol.spin = 0
mf = scf.RHF(mol).run(verbose=0)
print(f"  HF: {mf.e_tot:.8f}", flush=True)

na_o = list(range(N_CORE, N_CORE+N_ACT))
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
from math import comb
M = comb(N_ACT, ne[0]) * comb(N_ACT, ne[1])
print(f"  M={M:,} ({M/1e6:.1f}M)  memory={MEM_GB}GB", flush=True)

print(f"\n[1] DMRG-CI (M=2000, nroots={NROOTS}, mem={MEM_GB}GB)...", flush=True)
t1 = time.time()
mc = mcscf.CASCI(mf, N_ACT, ne)
mc.fcisolver = DMRGCI(mol, maxM=2000, tol=1e-6, memory=MEM_GB)
mc.fcisolver.nroots = NROOTS
try:
    e_dmrg = mc.kernel()[0]
    dt = time.time()-t1
    print(f"  DMRG done: {dt:.0f}s", flush=True)
    for k, e in enumerate(np.atleast_1d(e_dmrg)[:NROOTS]):
        print(f"    S{k}: {e:.12f} Ha", flush=True)
    os.makedirs("/data/home/wangcx/krylov-dci/checkpoints_cas20", exist_ok=True)
    np.savez("/data/home/wangcx/krylov-dci/checkpoints_cas20/dmrg_ref.npz",
             e_dmrg=np.atleast_1d(e_dmrg)[:NROOTS], nroots=NROOTS, maxM=2000)
    print(f"  Reference saved.", flush=True)
except Exception as e:
    print(f"  DMRG FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
