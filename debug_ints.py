"""Compare our Hamiltonian matrix elements against PySCF's CASCI Hamiltonian."""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci

mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol); mf.kernel()

# Our Hamiltonian
import sys; sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')
from hamiltonian import from_pyscf
ham = from_pyscf(mol, mf)

# PySCF CASCI Hamiltonian
n_cas, n_elec = 4, 4
mycas = mcscf.CASCI(mf, n_cas, n_elec)
# Get the 1e and 2e integrals CASCI uses
h1_cas, e_core = mycas.get_h1eff()
h2_cas = mycas.get_h2eff()
h2_cas = ao2mo.restore(1, h2_cas, n_cas)  # 4D

print("=== Our h1 (first 4 orbitals) ===")
print(np.array2string(ham.h1[:4, :4], precision=6, suppress_small=True))
print("\n=== PySCF CASCI h1eff ===")
print(np.array2string(h1_cas, precision=6, suppress_small=True))

print("\n=== Our h2[i,i,j,j] for first 4 orbitals ===")
for i in range(4):
    row = [ham.h2[i,i,j,j] for j in range(4)]
    print(f"  {[f'{x:.6f}' for x in row]}")

print("\n=== PySCF CASCI h2[i,i,j,j] ===")
for i in range(4):
    row = [h2_cas[i,i,j,j] for j in range(4)]
    print(f"  {[f'{x:.6f}' for x in row]}")

# Compute diagonal for HF determinant in CAS(4,4) active space
# HF: 3 core doubly occupied, active: both electrons in lowest active orbital (idx 3)
alpha_str = (1<<3) | (1<<2) | (1<<1) | (1<<0) | (1<<3)  # wait this isn't right
# For 7 orbitals, HF has electrons in orbitals 0-4 (10 e, 5 alpha, 5 beta)
# HF: alpha={0,1,2,3,4}, beta={0,1,2,3,4}
alpha_hf = (1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<4)
beta_hf = (1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<4)
E_diag_ours = ham.diagonal_element(alpha_hf, beta_hf)
print(f"\nE_diag(HF determinant) ours = {E_diag_ours:.10f}")
print(f"E_HF from PySCF              = {mf.e_tot:.10f}")
print(f"Difference                   = {(E_diag_ours - mf.e_tot)*1000:.3f} mH")

# Try with PySCF: direct computation of <HF|H|HF>
from pyscf.fci import direct_nosym
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 7)

# HF energy in MO basis: E_HF = 2 Σ_i h_{ii} + Σ_{ij} (2(ii|jj) - (ij|ji)) + E_nuc
# for i,j = 0..4 (occupied)
E_check = mf.energy_nuc()
for i in range(5):
    E_check += 2 * h1e_mo[i,i]
for i in range(5):
    for j in range(5):
        E_check += 2 * h2e_mo[i,i,j,j] - h2e_mo[i,j,j,i]
print(f"\nE_HF (manual from PySCF ints) = {E_check:.10f}")
print(f"E_HF from PySCF               = {mf.e_tot:.10f}")
print(f"Difference                    = {(E_check - mf.e_tot)*1000:.3f} mH")
