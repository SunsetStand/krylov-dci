import numpy as np
from pyscf import gto, scf, mcscf, ao2mo

mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol); mf.kernel()

# PySCF CASCI(4,4)
n_cas = 4
n_elec_cas = 4
mycas = mcscf.CASCI(mf, n_cas, n_elec_cas)
E_casci = mycas.kernel()[0]
print(f"PySCF CASCI(4,4)  = {E_casci:.10f}")
print(f"PySCF HF          = {mf.e_tot:.10f}")
print(f"E_corr(CAS)       = {(E_casci - mf.e_tot)*1000:.2f} mH")

# Now build H_PP manually using from_pyscf
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')
from hamiltonian import from_pyscf
from determinants import generate_determinants, count_bits

ham = from_pyscf(mol, mf)
n_core = (10 - n_elec_cas) // 2
n_orb_total = 7
n_active_orb = 4
n_core_orb = n_core  # = 3

# Generate CAS determinants and build H_PP
from partitioning import partition_cas
dets_all = generate_determinants(n_orb_total, 5, 5)  # n_alpha=n_beta=5
cas_dets = []
for a,b in dets_all:
    # Core occupies lowest n_core_orb=3 orbitals
    core_mask = (1 << n_core_orb) - 1  # bits 0,1,2
    if (a & core_mask) == core_mask and (b & core_mask) == core_mask:
        # Active is in orbitals n_core_orb to n_core_orb+n_active_orb=3 to 7
        # Count active electrons
        a_active = a >> n_core_orb
        b_active = b >> n_core_orb
        n_active_a = bin(a_active).count('1')
        n_active_b = bin(b_active).count('1')
        if n_active_a + n_active_b == n_elec_cas:
            cas_dets.append((a, b))

N = len(cas_dets)
print(f"\nManual CAS(4,4): {N} determinants")
H_PP = np.zeros((N,N))
for i in range(min(N, 5)):  # just check first 5 diag
    E_diag = ham.diagonal_element(cas_dets[i][0], cas_dets[i][1])
    print(f"  det {i}: H_diag = {E_diag:.6f}")
for i in range(N):
    for j in range(N):
        H_PP[i,j] = ham.matrix_element(cas_dets[i], cas_dets[j])
H_PP = 0.5*(H_PP+H_PP.T)
eigvals = np.linalg.eigh(H_PP)[0]
print(f"Manual H_PP lowest  = {eigvals[0]:.10f}")
print(f"Manual H_PP highest = {eigvals[-1]:.10f}")

# Also check the full FCI manually
from pyscf.fci.direct_nosym import FCI
fci_solver = FCI(); fci_solver.verbose = 0
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orb_total)
E_fci_pyscf, _ = fci_solver.kernel(
    h1e_mo, h2e_mo, n_orb_total, (mol.nelec[0], mol.nelec[1]),
    ecore=mf.energy_nuc())
print(f"\nPySCF FCI          = {E_fci_pyscf:.10f}")
print(f"E_corr(FCI)        = {(E_fci_pyscf - mf.e_tot)*1000:.2f} mH")

# Compare: is H_PP eigenvalue consistent with CASCI?
print(f"\nDiscrepancy: CASCI vs manual H_PP = {(E_casci - eigvals[0])*1000:.2f} mH")
