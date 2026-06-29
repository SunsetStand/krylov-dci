"""Debug H2 — with correct FCI (MO 2e integrals)."""
import sys, numpy as np
sys.path.insert(0,'src')
from pyscf import gto, scf, ao2mo
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol); mf.kernel()
ham = from_pyscf(mol, mf)
dets = generate_determinants_ms(2,2,ms=0)

print('Full H (4x4):')
for i in range(4):
    row = [ham.matrix_element(dets[i], dets[j]) for j in range(4)]
    print(f'  {[f"{x:.6f}" for x in row]}')

# CORRECT FCI: use MO 2e integrals
fci_solver = FCI(); fci_solver.verbose = 0
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 2)
E_fci, ci = fci_solver.kernel(
    h1e_mo, h2e_mo, 2, (mol.nelec[0], mol.nelec[1]), ecore=mf.energy_nuc())
print(f'\nE_FCI = {E_fci:.12f} Ha (correct, with MO 2e ints)')
print(f'E_corr = {(E_fci - mf.e_tot)*1000:.2f} mH')

# Diagonalize full 4x4 manually
H_full = np.array([[ham.matrix_element(dets[i], dets[j]) for j in range(4)] for i in range(4)])
eigvals_4 = np.linalg.eigh(H_full)[0]
print(f'\nManual 4x4 eigenvalues: {[f"{e:.6f}" for e in eigvals_4]}')
print(f'Lowest = {eigvals_4[0]:.12f}')

# Now check Löwdin exactness for each P size
print(f'\n{"="*60}')
for Np in [1, 2, 3]:
    p_dets = dets[:Np]
    q_dets = dets[Np:]
    M = 4 - Np
    
    H_PP = np.array([[ham.matrix_element(p_dets[i], p_dets[j]) for j in range(Np)] for i in range(Np)])
    E0 = np.linalg.eigh(H_PP)[0][0]
    
    H_QQ = np.array([[ham.matrix_element(q_dets[i], q_dets[j]) for j in range(M)] for i in range(M)])
    H_QP = np.array([[ham.matrix_element(q_dets[i], p_dets[j]) for j in range(Np)] for i in range(M)])
    
    delta_exact = E_fci - E0
    print(f'P={Np}, Q={M}: E0={E0:.8f}, Δ_exact={delta_exact:.6f}')
    
    # Exact effective H at E = E_FCI
    resolvent = np.linalg.inv((E0 + delta_exact) * np.eye(M) - H_QQ)
    correction = H_QP.T @ resolvent @ H_QP
    H_eff_exact = H_PP + correction + 0  # avoid any mutation issues
    H_eff_exact = 0.5 * (H_eff_exact + H_eff_exact.T)
    
    eigvals_eff = np.linalg.eigh(H_eff_exact)[0]
    print(f'  Correction (exact resolvent at E_FCI):')
    print(f'    {np.array2string(correction, precision=8, suppress_small=True)}')
    print(f'  H_P^eff eigenvalues: {[f"{e:.8f}" for e in eigvals_eff]}')
    print(f'  Lowest = {eigvals_eff[0]:.12f}, ΔE = {(eigvals_eff[0]-E_fci)*1000:.4f} mH')
    
    # Also test: self-consistent H_P^eff(E) = E for P=1
    if Np == 1:
        # Scan E to find roots of H_P^eff(E) - E = 0
        print(f'  Self-consistent scan (H_P^eff(E) - E):')
        for E_test in np.linspace(-2, 1, 13):
            resolv = np.linalg.inv(E_test * np.eye(M) - H_QQ)
            H_eff_test = H_PP[0,0] + (H_QP.T @ resolv @ H_QP)[0,0]
            print(f'    E={E_test:+.4f}: H_eff-E = {H_eff_test-E_test:+.6f}')
    print()
