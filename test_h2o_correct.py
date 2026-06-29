"""H2/STO-3G: verify Krylov+SVD correctness with CORRECT FCI (MO 2e integrals)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from numpy.linalg import eigh

from pyscf import gto, scf, ao2mo
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.partitioning import partition_cas, compute_reference_energy
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.effective_h import (build_effective_H, compute_with_fixed_delta,
                             self_consistent_iteration,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

# ============================================================
# H2O/STO-3G setup with CORRECT FCI
# ============================================================
mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol); mf.kernel()
E_HF = mf.e_tot
ham = from_pyscf(mol, mf)

n_orb, n_elec = 7, 10
dets_all = generate_determinants_ms(n_orb, n_elec, ms=0)
print(f"H2O/STO-3G: FCI dim = {len(dets_all)}, E_HF = {E_HF:.10f}")

# Correct FCI
fci_solver = FCI(); fci_solver.verbose = 0
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orb)
E_fci, _ = fci_solver.kernel(
    h1e_mo, h2e_mo, n_orb, (mol.nelec[0], mol.nelec[1]),
    ecore=mf.energy_nuc())
print(f"E_FCI = {E_fci:.12f}")
print(f"E_corr = {(E_fci-E_HF)*1000:.1f} mH")

# ============================================================
# Compare P-space strategies with CORRECT FCI
# ============================================================
from src.partitioning import partition_energy_window, partition_perturbation

strategies = [
    ('CAS(4,4)', lambda: partition_cas(n_orb, n_elec, 4, 4)),
    ('EW w=1.5', lambda: partition_energy_window(ham, dets_all, E_HF, 1.5)),
    ('EW w=2.0', lambda: partition_energy_window(ham, dets_all, E_HF, 2.0)),
    ('EW w=3.0', lambda: partition_energy_window(ham, dets_all, E_HF, 3.0)),
    ('PT2 1e-3', lambda: partition_perturbation(ham, dets_all, 0, 1e-3)),
    ('PT2 1e-4', lambda: partition_perturbation(ham, dets_all, 0, 1e-4)),
]

print(f"\n{'='*70}")
print(f"{'Strategy':<14s} {'N_P':>5s} {'N_Q':>5s} "
      f"{'E0 (Ha)':>14s} {'Δ exact':>12s} "
      f"{'ΔE fix (mH)':>12s} {'ΔE SCF (mH)':>12s} "
      f"{'iters':>5s} {'basis':>6s}")
print("-" * 75)

for name, fn in strategies:
    p_idx, q_idx = fn()
    N, M = len(p_idx), len(q_idx)
    if N == 0 or M == 0:
        continue
    p_dets = [dets_all[i] for i in p_idx]
    q_dets = [dets_all[i] for i in q_idx]
    
    E0 = compute_reference_energy(ham, dets_all, p_idx)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5*(H_PP + H_PP.T)
    
    diag_H_QQ = np.array([ham.diagonal_element(a,b) for a,b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    H_off = compute_H_off_diag(ham, q_dets)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)
    delta_exact = E_fci - E0
    
    # Build Krylov subspace (all layers until exhausted or max 5)
    all_basis = np.zeros((M, 0))
    layer0_raw = generate_layer_0(H_QP_mat, A_diag)
    basis0, _ = modified_gram_schmidt(layer0_raw, all_basis)
    all_basis = basis0
    
    prev_raw = layer0_raw
    for j in range(1, 5):
        new_raw = propagate_layer(prev_raw, H_off, A_diag, delta_exact)
        new_orth, retained = modified_gram_schmidt(new_raw, all_basis)
        if new_orth.shape[1] == 0:
            break
        all_basis = np.hstack([all_basis, new_orth])
        prev_raw = new_raw[:, retained]
    
    d_total = all_basis.shape[1]
    
    if d_total == 0:
        continue
    
    # Build H blocks (fast: use pre-computed H_QQ)
    H_QQ_full = np.diag(diag_H_QQ) + H_off
    H_PQtilde = build_H_PQtilde(ham, all_basis, p_dets, q_dets)
    H_QQ = build_H_Qtilde_Qtilde(ham, all_basis, q_dets, H_QQ_full=H_QQ_full)
    
    # Fixed Δ
    E_fix, _ = compute_with_fixed_delta(H_PP, H_PQtilde, H_QQ, E0, delta_exact)
    dE_fix = (E_fix - E_fci) * 1000
    
    # SCF Δ
    scf = self_consistent_iteration(H_PP, H_PQtilde, H_QQ, E0,
                                     delta_init=0.0, verbose=False)
    E_scf = scf['E_final']
    dE_scf = (E_scf - E_fci) * 1000
    
    ok = "✅" if abs(dE_scf) < 1.6 else ("🟡" if abs(dE_scf) < 10 else "❌")
    print(f"{name:<14s} {N:5d} {M:5d} {E0:14.8f} {delta_exact:12.6f} "
          f"{dE_fix:12.3f} {dE_scf:12.3f} {scf['n_iter']:5d} {d_total:6d} {ok}")
