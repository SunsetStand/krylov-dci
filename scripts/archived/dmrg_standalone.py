#!/usr/bin/env python3
"""
Standalone pyblock2 DMRG-CI reference for N2/cc-pVDZ CAS(20,10).

Uses block2 directly (not via PySCF DMRGCI wrapper) for better
control over memory, threading, and error reporting.

Reference: block2 CI example at pyblock2/ci.py
"""
import os, time, sys
import numpy as np
from pyscf import gto, scf, ao2mo, lib
from block2 import *
from block2.su2 import *

# ── Parameters ────────────────────────────────────────────────────
N_CORE = 2
N_ACT = 20
NROOTS = 6
R = 1.1  # Angstrom
MAX_M = 2000          # Max bond dimension
MEM_GB = 300           # Total memory for block2 (out of 500GB node)
N_THREADS = 32         # Threads for operator tensors
N_MKL_THREADS = 4      # Threads for MKL (dense matmul)
TOL = 1e-6

SCRATCH = '/data/home/wangcx/krylov-dci/scratch'
OUTDIR = '/data/home/wangcx/krylov-dci/checkpoints_cas20'
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70, flush=True)
print("Standalone Block2 DMRG-CI: N2/cc-pVDZ CAS(20,10)")
print(f"M={MAX_M}  mem={MEM_GB}GB  threads={N_THREADS}/{N_MKL_THREADS}")
print(f"scratch={SCRATCH}")
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════════
# 1. Build integrals (PySCF)
# ═══════════════════════════════════════════════════════════════════
t0 = time.time()
print("\n[1] HF + AO→MO integral transform...", flush=True)
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=3)
mf = scf.RHF(mol).run()
print(f"  HF energy: {mf.e_tot:.12f} Ha", flush=True)

# Active space slice
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb_tot = mf.mo_coeff.shape[1]
ne_act = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)

# Transform integrals to active MO basis
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h1_act = h1_mo[np.ix_(na_o, na_o)]

eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
eri_mo = ao2mo.restore(1, eri_mo, norb_tot)
eri_act = eri_mo[np.ix_(na_o, na_o, na_o, na_o)]

# Count CAS dimension
from math import comb
M_fci = comb(N_ACT, ne_act[0]) * comb(N_ACT, ne_act[1])
print(f"  CAS({N_ACT},{ne_act[0]+ne_act[1]}): {M_fci:,} dets ({M_fci/1e6:.1f}M)", flush=True)
print(f"  Active integrals done ({time.time()-t0:.0f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 2. Initialize Block2
# ═══════════════════════════════════════════════════════════════════
print(f"\n[2] Initializing Block2 (mem={MEM_GB}GB)...", flush=True)
t2 = time.time()

# Set random seed for reproducibility
Random.rand_seed(123456)

# Memory allocation
memory = int(MEM_GB * 1e9)  # bytes
# block2 needs both integer and double scratch space
# isize: integer workspace (configurations)
# dsize: double workspace (operators)
init_memory(isize=int(memory * 0.05),
            dsize=int(memory * 0.95),
            save_dir=SCRATCH)

# Threading
Global.threading = Threading(
    ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
    N_THREADS, N_THREADS, N_MKL_THREADS)
Global.threading.seq_type = SeqTypes.Nothing

# Precision
Global.frame.fp_codec = DoubleFPCodec(1E-16, 1024)
Global.frame.load_buffering = False
Global.frame.save_buffering = False
Global.frame.use_main_stack = False
Global.frame.minimal_disk_usage = True

print(f"  Block2 initialized ({time.time()-t2:.0f}s)", flush=True)
print(f"  Thread config: {Global.threading}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 3. Build FCIDUMP
# ═══════════════════════════════════════════════════════════════════
print(f"\n[3] Building FCIDUMP...", flush=True)
t3 = time.time()

n_act = N_ACT
na, nb = ne_act
fcidump_tol = 1E-13

# Flatten and apply threshold
h1e = h1_act.copy()
g2e = eri_act.copy()
h1e[np.abs(h1e) < fcidump_tol] = 0.0
g2e[np.abs(g2e) < fcidump_tol] = 0.0

ecore = mf.energy_nuc()

fcidump = FCIDUMP()
fcidump.initialize_su2(n_act, na + nb, abs(na - nb), 1, ecore, h1e.ravel(), g2e)
error = fcidump.symmetrize(VectorUInt8([0] * n_act))
print(f"  FCIDUMP symmetry error: {error}", flush=True)
print(f"  ecore = {ecore:.12f} Ha", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 4. Build Hamiltonian MPO
# ═══════════════════════════════════════════════════════════════════
print(f"\n[4] Building Hamiltonian MPO...", flush=True)
t4 = time.time()

vacuum = SU2(0)
target = SU2(na + nb, abs(na - nb), 0)

# Use Simplified Hamiltonian (no big site)
hamil = HamiltonianQC(vacuum, n_act, VectorUInt8([0] * n_act), fcidump)
n_sites = hamil.n_sites
print(f"  n_sites = {n_sites}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 5. Initialize MPS
# ═══════════════════════════════════════════════════════════════════
print(f"\n[5] Initializing MPS...", flush=True)
t5 = time.time()

info = MPSInfo(n_sites, vacuum, target, hamil.basis)
info.tag = "KET"
info.set_bond_dimension(MAX_M)

mps = MPS(n_sites, 0, 2)
mps.initialize(info)
mps.random_canonicalize()
mps.tensors[mps.center].normalize()
mps.save_mutable()
info.save_mutable()
print(f"  MPS bond_dim={MAX_M}, n_sites={n_sites} ({time.time()-t5:.0f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 6. Build MPO
# ═══════════════════════════════════════════════════════════════════
print(f"\n[6] Building MPO...", flush=True)
t6 = time.time()

mpo = MPOQC(hamil, QCTypes.NC)
mpo = SimplifiedMPO(mpo, RuleQC(), True)
print(f"  MPO done ({time.time()-t6:.0f}s)", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 7. DMRG Sweeps
# ═══════════════════════════════════════════════════════════════════
print(f"\n[7] Running DMRG sweeps...", flush=True)
t7 = time.time()

me = MovingEnvironment(mpo, mps, mps, "DMRG")
me.delayed_contraction = OpNamesSet.normal_ops()
me.cached_contraction = True
me.save_partition_info = True
me.init_environments(False)
print(f"  MovingEnvironment done ({time.time()-t7:.0f}s)", flush=True)

# Sweep sequence: start with small M, ramp up
bond_dims = [500, 1000, 1500, 2000]
noises = [1e-4, 1e-5, 1e-6, 0.0]
n_sweeps_per_stage = [4, 4, 4, 2]

all_bonds = []
all_noises = []
for dims, noise, ns in zip(bond_dims, noises, n_sweeps_per_stage):
    all_bonds.extend([dims] * ns)
    all_noises.extend([noise] * ns)

print(f"  Sweep schedule: {list(zip(all_bonds, all_noises))}", flush=True)

dmrg = DMRG(me, VectorUBond([MAX_M]), VectorDouble([0]))
dmrg.davidson_conv_thrds = VectorDouble([1E-12])
dmrg.cutoff = 0
dmrg.iprint = 2  # Show sweep info
dmrg.noise_type = NoiseTypes.Perturbative

ener = dmrg.solve(len(all_bonds), True, 0.0)
t_dmrg = time.time() - t7

# ═══════════════════════════════════════════════════════════════════
# 8. Results
# ═══════════════════════════════════════════════════════════════════
print(f"\n[8] Results", flush=True)
print(f"  Total DMRG time: {t_dmrg:.0f}s", flush=True)
print(f"  DMRG total energy: {ener:.12f} Ha", flush=True)
print(f"  Correlation energy: {ener - mf.e_tot:.12f} Ha", flush=True)

# For state-averaged / excited states, we'd need to do state-specific
# or use the block2 state-averaging API. For now, save the ground state.

# Save to file
outfile = os.path.join(OUTDIR, 'dmrg_ground_state.npz')
np.savez(outfile,
         e_dmrg=float(ener),
         e_hf=float(mf.e_tot),
         maxM=MAX_M,
         n_act=N_ACT,
         ne_act=list(ne_act),
         e_core=float(ecore))
print(f"  Saved to {outfile}", flush=True)

print(f"\n{'='*70}")
print(f"DMRG done. Total wall: {time.time()-t0:.0f}s")
print(f"{'='*70}", flush=True)
