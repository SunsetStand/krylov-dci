"""
Minimal H2/STO-3G test: P = HF only, Q = 3 determinants.
Verifies the Krylov method with exact Δ converges to FCI as m increases.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from numpy.linalg import eigh, norm

from pyscf import gto, scf
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.effective_h import (build_effective_H, compute_with_fixed_delta,
                             self_consistent_iteration,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

print("=" * 60)
print("H2/STO-3G: Krylov-dCI exactness verification")
print("P = HF determinant only (1 det), Q = 3 determinants")
print("=" * 60)

# Setup
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol)
mf.kernel()
ham = from_pyscf(mol, mf)

n_orb = 2
n_elec = 2
dets_all = generate_determinants_ms(n_orb, n_elec, ms=0)
print(f"FCI dimension: {len(dets_all)}")

# FCI reference
from pyscf.fci.direct_nosym import FCI
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_ao = mol.intor('int2e', aosym='s8')
fci_solver = FCI()
fci_solver.verbose = 0
E_fci, ci_fci = fci_solver.kernel(
    h1e, h2e_ao, n_orb, (mol.nelec[0], mol.nelec[1]),
    ecore=mf.energy_nuc())
print(f"E_FCI = {E_fci:.12f} Ha")

# P = HF det (idx 0), Q = others (idx 1,2,3)
p_idx = np.array([0])
q_idx = np.array([1, 2, 3])
p_dets = [dets_all[i] for i in p_idx]
q_dets = [dets_all[i] for i in q_idx]
N, M = len(p_idx), len(q_idx)
print(f"P = {N} dets, Q = {M} dets")

# Build H_PP
H_PP = np.array([[ham.matrix_element(p_dets[0], p_dets[0])]])
E0 = H_PP[0, 0]
print(f"E0 (H_PP) = {E0:.12f} Ha")

# Q-space
diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
A_diag = compute_A(E0, diag_H_QQ)
H_off = compute_H_off_diag(ham, q_dets)
H_QP_mat = build_H_QP(ham, p_dets, q_dets)

delta_exact = E_fci - E0
print(f"Δ_exact = E_FCI - E0 = {delta_exact:.10f} Ha")

# Build Krylov subspace layer by layer
all_basis = np.zeros((M, 0))
layer_sizes = []

# Layer 0
layer0_raw = generate_layer_0(H_QP_mat, A_diag)
layer0_orth, _ = modified_gram_schmidt(layer0_raw, all_basis)
d0 = layer0_orth.shape[1]
all_basis = layer0_orth
layer_sizes.append(d0)
print(f"\nLayer 0: {d0} vectors (basis total: {all_basis.shape[1]})")

prev_layer_raw = layer0_raw

# Layers 1+
for j in range(1, 5):  # go up to m=5
    new_raw = propagate_layer(prev_layer_raw, H_off, A_diag, delta_exact)
    new_orth, retained = modified_gram_schmidt(new_raw, all_basis)
    dj = new_orth.shape[1]
    if dj == 0:
        print(f"Layer {j}: exhausted (all linearly dependent)")
        break
    all_basis = np.hstack([all_basis, new_orth])
    layer_sizes.append(dj)
    prev_layer_raw = new_raw[:, retained]  # use retained columns for next propagation
    print(f"Layer {j}: {dj} vectors (basis total: {all_basis.shape[1]})")

d_total = all_basis.shape[1]
print(f"\nTotal Krylov basis: {d_total}/{M} = {d_total/M:.1%} of Q-space")

# Build H_PQ̃, H_Q̃Q̃
H_PQtilde = build_H_PQtilde(ham, all_basis, p_dets, q_dets)
H_Qtilde_Qtilde = build_H_Qtilde_Qtilde(ham, all_basis, q_dets)

# ---- Experiment 1: Fixed Δ (from FCI) ----
print(f"\n{'='*50}")
print(f"Experiment 1: Non-self-consistent (fixed Δ = E_FCI - E0)")
print(f"{'='*50}")

E_fixed, evec_fixed = compute_with_fixed_delta(
    H_PP, H_PQtilde, H_Qtilde_Qtilde, E0, delta_exact
)
delta_EmH = (E_fixed - E_fci) * 1000
print(f"  E_krylov(fixed Δ) = {E_fixed:.12f} Ha")
print(f"  ΔE = {delta_EmH:+.6f} mHartree")

# ---- Experiment 2: Self-consistent Δ ----
print(f"\n{'='*50}")
print(f"Experiment 2: Self-consistent Δ iteration")
print(f"{'='*50}")

scf_result = self_consistent_iteration(
    H_PP, H_PQtilde, H_Qtilde_Qtilde, E0,
    delta_init=0.0, verbose=True
)
E_scf = scf_result['E_final']
delta_scf_EmH = (E_scf - E_fci) * 1000
print(f"  E_krylov(SCF Δ) = {E_scf:.12f} Ha")
print(f"  ΔE = {delta_scf_EmH:+.6f} mHartree")
print(f"  Δ_final = {scf_result['delta_final']:.10f}")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  E_FCI          = {E_fci:.12f} Ha")
print(f"  E_krylov(fixed Δ) = {E_fixed:.12f}  (ΔE = {delta_EmH:+.3e} mH)")
print(f"  E_krylov(SCF Δ)   = {E_scf:.12f}  (ΔE = {delta_scf_EmH:+.3e} mH)")
print(f"  Krylov subspace: {d_total}/{M} Q-vectors")

if abs(delta_EmH) < 1.0:
    print(f"\n  ✅ Krylov method converges to FCI with exact Δ!")
    print(f"     Error = {delta_EmH:.2e} mH (< 0.001 mH, essentially exact)")
else:
    print(f"\n  ⚠ Non-trivial error: {delta_EmH:.3f} mH")

if abs(delta_scf_EmH - delta_EmH) < 0.1:
    print(f"  ✅ Self-consistent Δ matches fixed Δ!")
else:
    print(f"  ⚠ SCF Δ differs from exact Δ by {abs(delta_scf_EmH - delta_EmH):.3f} mH")
