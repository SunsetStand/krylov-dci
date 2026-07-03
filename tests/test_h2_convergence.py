"""
H2/STO-3G: verify Krylov convergence to FCI for different P sizes.
Total: 4 determinants (2e, 2o, Ms=0).
Tests P = 1, 2, 3 (Q = 3, 2, 1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from numpy.linalg import eigh

from pyscf import gto, scf
from pyscf.fci.direct_nosym import FCI
from src.hamiltonian import from_pyscf
from src.determinants import generate_determinants_ms
from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt)
from src.effective_h import (build_effective_H, compute_with_fixed_delta,
                             self_consistent_iteration,
                             build_H_Qtilde_Qtilde, build_H_PQtilde)

# Setup
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = scf.RHF(mol); mf.kernel()
ham = from_pyscf(mol, mf)

dets_all = generate_determinants_ms(2, 2, ms=0)
print(f"H2/STO-3G: {len(dets_all)} determinants")

# FCI reference
fci_solver = FCI(); fci_solver.verbose = 0
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_ao = mol.intor('int2e', aosym='s8')
E_fci, _ = fci_solver.kernel(
    h1e, h2e_ao, 2, (mol.nelec[0], mol.nelec[1]), ecore=mf.energy_nuc())
print(f"E_FCI = {E_fci:.12f}")

# Determinant info
for i, (a, b) in enumerate(dets_all):
    E_diag = ham.diagonal_element(a, b)
    print(f"  det {i}: α={a:04b} β={b:04b}  E_diag={E_diag:.6f}")

# ============================================================
# Test for each P size
# ============================================================
for N_p in [1, 2, 3]:
    print(f"\n{'='*60}")
    print(f"P = {N_p}, Q = {4-N_p}")
    print(f"{'='*60}")

    p_idx = np.arange(N_p)
    q_idx = np.arange(N_p, 4)
    p_dets = [dets_all[i] for i in p_idx]
    q_dets = [dets_all[i] for i in q_idx]
    N, M = N_p, 4 - N_p

    # H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)
    E0_vals, _ = eigh(H_PP)
    E0 = E0_vals[0]
    delta_exact = E_fci - E0

    # Q-space
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    H_off = compute_H_off_diag(ham, q_dets)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)

    # Build Krylov subspace (exhaustive — keep going until no new vectors)
    all_basis = np.zeros((M, 0))
    layer_raws = []
    layer_sizes = []
    m_max = 10  # safety limit

    # Layer 0
    layer0_raw = generate_layer_0(H_QP_mat, A_diag)
    layer_raws.append(layer0_raw)
    layer0_orth, _ = modified_gram_schmidt(layer0_raw, all_basis)
    all_basis = layer0_orth
    layer_sizes.append(layer0_orth.shape[1])

    # Higher layers
    prev_raw = layer0_raw
    for j in range(1, m_max):
        new_raw = propagate_layer(prev_raw, H_off, A_diag, delta_exact)
        if new_raw.shape[1] == 0:
            break
        new_orth, retained = modified_gram_schmidt(new_raw, all_basis)
        dj = new_orth.shape[1]
        if dj == 0:
            break
        all_basis = np.hstack([all_basis, new_orth])
        layer_sizes.append(dj)
        layer_raws.append(new_raw)
        prev_raw = new_raw[:, retained]  # retained columns
    d_total = all_basis.shape[1]

    print(f"  Krylov layers: {layer_sizes}")
    print(f"  Total basis: {d_total}/{M} = {d_total/M:.0%}")
    print(f"  E0 = {E0:.10f}, Δ_exact = {delta_exact:.6f}")

    # Test m=0,1,...,max_layers-1
    if d_total == 0:
        E = eigh(H_PP)[0][0]
        dE = (E - E_fci) * 1000
        print(f"  Q empty → E = {E:.10f}, ΔE = {dE:+.3f} mH")
        continue

    # Build effective H at each m
    basis_m = np.zeros((M, 0))
    cumul_size = 0
    for layer_idx in range(len(layer_sizes)):
        cumul_size += layer_sizes[layer_idx]
        # Reconstruct basis up to this layer (use raw for propagation state)
        # Actually just test with the full basis built so far
        pass

    # Simpler: test with all layers combined
    H_PQtilde = build_H_PQtilde(ham, all_basis, p_dets, q_dets)
    H_QQ = build_H_Qtilde_Qtilde(ham, all_basis, q_dets)

    # Fixed Δ
    E_fixed, _ = compute_with_fixed_delta(
        H_PP, H_PQtilde, H_QQ, E0, delta_exact)
    dE_fixed = (E_fixed - E_fci) * 1000

    # Self-consistent Δ
    scf = self_consistent_iteration(
        H_PP, H_PQtilde, H_QQ, E0, delta_init=0.0, verbose=False)
    E_scf = scf['E_final']
    dE_scf = (E_scf - E_fci) * 1000

    print(f"  Fixed Δ:  E = {E_fixed:.12f}, ΔE = {dE_fixed:+.4f} mH")
    print(f"  SCF Δ:    E = {E_scf:.12f}, ΔE = {dE_scf:+.4f} mH "
          f"(iters={scf['n_iter']}, converged={scf['converged']})")

    # Layer-by-layer convergence
    print(f"\n  Layer-by-layer convergence (fixed Δ):")
    print(f"  {'Layer':>6s} {'n_vecs':>7s} {'E (Ha)':>16s} {'ΔE (mH)':>10s}")
    print(f"  {'-'*42}")
    
    # Build incrementally: add one layer at a time from all_basis
    accum_basis = np.zeros((M, 0))
    basis_idx = 0
    for layer_idx, n_vecs in enumerate(layer_sizes):
        if n_vecs == 0:
            break
        next_chunk = all_basis[:, basis_idx:basis_idx + n_vecs]
        accum_basis = np.hstack([accum_basis, next_chunk])
        basis_idx += n_vecs
        
        H_PQ = build_H_PQtilde(ham, accum_basis, p_dets, q_dets)
        H_QQ_l = build_H_Qtilde_Qtilde(ham, accum_basis, q_dets)
        E_l, _ = compute_with_fixed_delta(H_PP, H_PQ, H_QQ_l, E0, delta_exact)
        dE_l = (E_l - E_fci) * 1000
        
        label = f"m={layer_idx}"
        print(f"  {label:>6s} {accum_basis.shape[1]:7d} {E_l:16.12f} {dE_l:+10.4f}")

print(f"\n{'='*60}")
print("SUMMARY: ΔE (mH) vs FCI")
print(f"{'='*60}")
print(f"  E_FCI = {E_fci:.12f} Ha")
print(f"  For H2/STO-3G, P=1 → Krylov exhausts at 1D → can't reach FCI")
print(f"  Larger P → larger Krylov subspace → closer to exact")
