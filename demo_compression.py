"""
Demonstrate the FUNDAMENTAL compression of Krylov+SVD:
  M x M → M x N → N x N

The point is NOT threshold truncation — it's that SVD of an M×N matrix
(with N << M) reduces the problem dimensionality from M to N automatically.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from numpy.linalg import eigh, svd

from pyscf import gto, scf
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.partitioning import partition_cas, compute_reference_energy
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.svd_compression import build_weighted_coupling, compress_layer
from src.effective_h import (build_effective_H, compute_with_fixed_delta,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

# ==================================================================
# H2O/STO-3G: CAS(4,4) -> P=36, Q=405, M=405
# ==================================================================
mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
            basis='sto-3g', charge=0, spin=0, verbose=0)
mf = scf.RHF(mol)
mf.kernel()
ham = from_pyscf(mol, mf)

n_orb, n_elec = 7, 10
dets_all = generate_determinants_ms(n_orb, n_elec, ms=0)
n_fci = len(dets_all)

# FCI
fci_solver = FCI(); fci_solver.verbose = 0
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_ao = mol.intor('int2e', aosym='s8')
E_fci, _ = fci_solver.kernel(
    h1e, h2e_ao, n_orb, (mol.nelec[0], mol.nelec[1]),
    ecore=mf.energy_nuc())

# Partition
p_idx, q_idx = partition_cas(n_orb, n_elec, 4, 4)
N, M = len(p_idx), len(q_idx)
p_dets = [dets_all[i] for i in p_idx]
q_dets = [dets_all[i] for i in q_idx]

E0 = compute_reference_energy(ham, dets_all, p_idx)
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)

diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
A_diag = compute_A(E0, diag_H_QQ)
H_off = compute_H_off_diag(ham, q_dets)
H_QP_mat = build_H_QP(ham, p_dets, q_dets)
delta_exact = E_fci - E0

# ==================================================================
# ANALYZE THE EFFECTIVE RANK
# ==================================================================
print("=" * 65)
print("SVD FUNDAMENTAL COMPRESSION: M → min(M,N)")
print("=" * 65)
print(f"  FCI dimension:    {n_fci}")
print(f"  P-space (N):      {N}")
print(f"  Q-space (M):      {M}")
print(f"  P+N =             {N+M}\n")

# Layer 0: raw coupling after A-weighting
layer0_raw = generate_layer_0(H_QP_mat, A_diag)  # M x N
T0 = build_weighted_coupling(layer0_raw, A_diag)  # M x N
# SVD: U(Mxr), Sigma(rxr), Vt(rxN) with r ≤ min(M,N)
U0, sigma0, Vt0 = svd(T0, full_matrices=False)
r0 = len(sigma0)

print(f"  ┌──────────────────────────────────────────────────┐")
print(f"  │ Layer 0: T^(0) is {M} × {N}                             │")
print(f"  │   SVD rank = {r0} = min(M={M}, N={N})                      │")
print(f"  │   COMPRESSION: {M} → {r0} ({M/r0:.1f}×)                       │")
print(f"  │   σ₁ = {sigma0[0]:.4f}  σ_min = {sigma0[-1]:.4f}           │")
print(f"  │   Condition number: {sigma0[0]/sigma0[-1]:.1f}               │")
print(f"  └──────────────────────────────────────────────────┘")

# Layer 1
layer1_raw = propagate_layer(layer0_raw, H_off, A_diag, delta_exact)
T1 = build_weighted_coupling(layer1_raw, A_diag)
U1, sigma1, Vt1 = svd(T1, full_matrices=False)
r1 = len(sigma1)
print(f"  ┌──────────────────────────────────────────────────┐")
print(f"  │ Layer 1: T^(1) is {M} × {N}                             │")
print(f"  │   SVD rank = {r1} = min(M={M}, N={N})                      │")
print(f"  │   COMPRESSION: {M} → {r1} ({M/r1:.1f}×)                       │")
print(f"  │   σ₁ = {sigma1[0]:.4f}  σ_min = {sigma1[-1]:.4f}           │")
print(f"  └──────────────────────────────────────────────────┘")

# All layers combined: at most m × N vectors from M-dimensional Q
m = 3
max_basis = m * N
print(f"\n  After {m} Krylov layers:")
print(f"  ┌──────────────────────────────────────────────────┐")
print(f"  │ Q-space dimension:    M = {M}                          │")
print(f"  │ Krylov basis (max):   m×N = {m}×{N} = {max_basis}                        │")
print(f"  │ Compression:          {M} → ≤{max_basis} ({M/max_basis:.1f}×)                      │")
print(f"  │ Effective H matrix:   ≤{max_basis} × ≤{max_basis}                   │")
print(f"  │ Original H_QQ:        {M} × {M}                          │")
print(f"  └──────────────────────────────────────────────────┘")

# ==================================================================
# SCALING DEMONSTRATION
# ==================================================================
print(f"\n{'='*65}")
print("SCALING: What happens as system size grows?")
print(f"{'='*65}")

print(f"\n  {'System':>12s}  {'M (Q)':>8s}  {'N (P)':>8s}  "
      f"{'M×M':>10s}  {'N×N(basis)':>12s}  {'Saving':>8s}")
print("  " + "-" * 60)

scenarios = [
    ("H2O/STO-3G", 405, 36),
    ("H2O/cc-pVDZ", 500000, 100),
    ("N2/cc-pVDZ", 1.2e7, 200),
    ("C10H8/DZ", 1e10, 500),
]

for name, M_s, N_s in scenarios:
    MxM = M_s**2
    NN = (3*N_s)**2  # ~3 layers of N basis vectors
    saving = MxM / NN if NN > 0 else float('inf')
    print(f"  {name:>12s}  {M_s:>8.0f}  {N_s:>8.0f}  "
          f"{MxM:>10.1e}  {NN:>12.1e}  {saving:>8.0e}×")

print(f"\n  ═══════════════════════════════════════════════════")
print(f"  KEY INSIGHT:")
print(f"  The SVD compresses the M×M H_QQ problem into an")
print(f"  N×N effective Hamiltonian — regardless of M.")
print(f"  The σ-threshold truncation is a SECONDARY tool")
print(f"  to further reduce within the already-compressed")
print(f"  N-dimensional space.")
print(f"  ═══════════════════════════════════════════════════")

# ==================================================================
# Visualize: what does the SVD actually give us?
# ==================================================================
print(f"\n{'='*65}")
print("WHAT SVD GIVES (vs GRAM-SCHMIDT)")
print(f"{'='*65}")

# Gram-Schmidt: just orthonormalizes the N raw vectors
layer0_gs, _ = modified_gram_schmidt(layer0_raw, np.zeros((M, 0)))
# SVD: finds the optimal N-dimensional subspace for the weighted coupling
U_svd, sigma_svd, _ = svd(T0, full_matrices=False)

# Both give N vectors, but SVD's are OPTIMAL for the weighted problem
print(f"\n  Both Gram-Schmidt and SVD give {r0} vectors from {M} Q-dets.")
print(f"\n  Difference:")
print(f"    Gram-Schmidt: orthonormalizes raw Krylov vectors")
print(f"    SVD:          finds the optimal low-dim subspace of T^(j)")
print(f"                  = (E0·I - H_D')^(-1/2) · M^(j)")
print(f"                  → simultaneously weights coupling strength")
print(f"                    AND energetic proximity")
print(f"\n  The SVD basis:")
print(f"    • Is orthonormal (like GS)")
print(f"    • Has singular values that quantify each direction's importance")
print(f"    • Can be threshold-truncated (secondary benefit)")
print(f"    • Is optimal in Frobenius norm (Eckart-Young-Mirsky)")
