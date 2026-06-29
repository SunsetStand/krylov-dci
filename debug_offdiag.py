import numpy as np
from pyscf import gto, scf
import sys; sys.path.insert(0,'/data/home/wangcx/krylov-dci/src')
from hamiltonian import from_pyscf
from determinants import generate_determinants

mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol); mf.kernel()
ham = from_pyscf(mol, mf)

# Get full FCI determinants and CAS(4,4) subset
dets_all = generate_determinants(7, 5, 5)  # 7 orb, 5 alpha, 5 beta
# CAS: core 0,1,2 doubly occupied, 4 active electrons in orbitals 3-6
core_mask = (1<<3) - 1
cas_dets = [(a,b) for a,b in dets_all 
            if (a & core_mask) == core_mask and (b & core_mask) == core_mask
            and bin(a>>3).count('1') + bin(b>>3).count('1') == 4]

N = len(cas_dets)
print(f"CAS(4,4): {N} dets")

# Build full H_PP and check for large elements
H_PP = np.zeros((N,N))
for i in range(N):
    for j in range(i, N):
        hij = ham.matrix_element(cas_dets[i], cas_dets[j])
        if i == j:
            H_PP[i,i] = hij
        else:
            H_PP[i,j] = hij
            H_PP[j,i] = hij

# Check: are there off-diagonals > 1 Ha?
print("\nLarge off-diagonal elements (>0.5 Ha):")
for i in range(N):
    for j in range(i+1, N):
        if abs(H_PP[i,j]) > 0.5:
            print(f"  H[{i},{j}] = {H_PP[i,j]:.6f}")

# Also check: compare against PySCF FCI Hamiltonian
from pyscf import ao2mo, fci
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 7)

# Get the CAS(4,4) Hamiltonian from PySCF
from pyscf.fci import addons
# Build H in determinant basis using PySCF
norb = 4  # active orbitals
nelec = (2, 2)  # 2 alpha, 2 beta active

# check det 0 (HF in CAS) against det 1:
# det 0: core(3)+active(α in orb3+4, β in orb3+4) 
#   wait, active has 4 electrons in 4 orbitals. HF-CAS: 
#   active alpha: 3,4. active beta: 3,4 → 4 active electrons
# det 1: one excitation? Need to figure out what det 1 is.

# Let me find the first off-diagonal element
print(f"\nFirst few diagonal elements:")
for i in range(5):
    print(f"  H[{i},{i}] = {H_PP[i,i]:.6f}")

print(f"\nFirst row off-diagonal elements:")
for j in range(1, min(10,N)):
    print(f"  H[0,{j}] = {H_PP[0,j]:.10f}")

# Compare: PySCF FCI Hamiltonian for same determinants
# Build in the full determinant basis
from pyscf.fci.direct_nosym import FCI
fci_solver = FCI()
h2e_full = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 7)
# Compute <det_i|H|det_j> for det 0 (HF) and det 1
d0_a, d0_b = cas_dets[0]  # HF
d1_a, d1_b = cas_dets[1]  # excited

# PySCF determinant: convert bit strings to occupation vectors
def bits_to_occvec(bits, norb):
    vec = np.zeros(norb, dtype=int)
    for p in range(norb):
        if bits & (1<<p):
            vec[p] = 1
    return vec

occ_a = bits_to_occvec(d0_a, 7)
occ_b = bits_to_occvec(d0_b, 7)
occ_a2 = bits_to_occvec(d1_a, 7)
occ_b2 = bits_to_occvec(d1_b, 7)

# Use PySCF's built-in
from pyscf.fci import cistring
strs_a = cistring.make_strings(range(7), 5)
strs_b = cistring.make_strings(range(7), 5)

idx0_a = list(strs_a).index(d0_a)
idx0_b = list(strs_b).index(d0_b)
idx0 = cistring.strs2addr(7, 5, idx0_a, idx0_b)

# Actually, let me just use a simpler approach: compare the FCI Hamiltonian
# built by PySCF vs ours for selected determinants
# Use PySCF's pspace function
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
E_nuc = mf.energy_nuc()

# Manual Slater-Condon for a single determinant pair
# HF: α={0,1,2,3,4}, β={0,1,2,3,4}
# Let me check det1: what excitations does it have from HF?
det0_a = d0_a  # bits 0-4 set
det0_b = d0_b  # bits 0-4 set
det1_a = d1_a  # ?
det1_b = d1_b  # ?

alpha_diff = det0_a ^ det1_a
beta_diff = det0_b ^ det1_b
print(f"\ndet0: a={det0_a:07b} b={det0_b:07b}")
print(f"det1: a={det1_a:07b} b={det1_b:07b}")
print(f"α diff: {alpha_diff:07b}, β diff: {beta_diff:07b}")

n_exc_alpha = bin(alpha_diff).count('1')
n_exc_beta = bin(beta_diff).count('1')
print(f"Excitation level: {n_exc_alpha//2 + n_exc_beta//2}")

# Manual from PySCF integrals
# For 4-index integrals, need proper transformation
# Let me use PySCF's pspace module
from pyscf.fci import direct_spin1
h1e_tmp = h1e_mo.copy()
h2e_tmp = h2e_full.copy()

# Just use PySCF FCI pspace
pspace_h1, pspace_h2 = direct_spin1.pspace(h1e_tmp, h2e_tmp, 
    np.array([d0_a, d1_a], dtype=np.int64),
    np.array([d0_b, d1_b], dtype=np.int64),
    np.array([0, 1], dtype=np.int64), E_nuc)
print(f"\nPySCF pspace (2 dets):")
print(f"  H = {pspace_h1}")
diag_diff = pspace_h1[0,0] - H_PP[0,0]
off_diff = pspace_h1[0,1] - H_PP[0,1]
print(f"  Our H[0,0] = {H_PP[0,0]:.10f}, pspace = {pspace_h1[0,0]:.10f}, diff={diag_diff:.6e}")
print(f"  Our H[0,1] = {H_PP[0,1]:.10f}, pspace = {pspace_h1[0,1]:.10f}, diff={off_diff:.6e}")
