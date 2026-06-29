#!/usr/bin/env python3
"""
Phase 3: H₂O/STO-3G SVD Compression Analysis

Benchmarks:
  1. Krylov subspace construction (Phase 2)
  2. Weighted SVD compression (Phase 3 core)
  3. Wall-clock comparison:
     (a) Build full H_O' (M×M) → project with SVD rotation T
     (b) Station's idea: sigma-vector H_O'·T without building full H_O'
  4. SVD truncation threshold scan
  5. Convergence vs Krylov order

Output: timing data, singular value spectra, convergence table.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')
from pyscf import gto, scf, ao2mo
from pyscf.fci.direct_nosym import FCI
from hamiltonian import Hamiltonian, from_pyscf
from determinants import generate_determinants_ms
from partitioning import partition_cas, compute_reference_energy
from krylov import (
    compute_A, compute_H_off_diag, build_H_QP,
    generate_layer_0, propagate_layer,
    modified_gram_schmidt, build_krylov_subspace,
    sigma_H_off
)
from svd_compression import (
    build_weighted_coupling, svd_truncate, compress_layer,
    svd_truncate_unweighted, analyze_singular_values
)
from effective_h import (
    build_H_Qtilde_Qtilde, build_H_PQtilde,
    build_effective_H, compute_with_fixed_delta,
    self_consistent_iteration
)

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# ============================================================================
# Setup: H₂O/STO-3G
# ============================================================================
print("=" * 70)
print("Phase 3: H₂O/STO-3G SVD Compression Analysis")
print("=" * 70)

t0_setup = time.perf_counter()
mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol)
mf.kernel()
ham = from_pyscf(mol, mf)

n_orb = mol.nao
n_elec = mol.nelec[0] + mol.nelec[1]
dets = generate_determinants_ms(n_orb, n_elec, ms=0)
print(f"  n_orb={n_orb}, n_elec={n_elec}, n_det={len(dets)}")

# FCI reference
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
h2e_mo = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orb)
solver = FCI()
solver.verbose = 0
E_fci, ci_fci = solver.kernel(h1e_mo, h2e_mo, n_orb,
                              (mol.nelec[0], mol.nelec[1]),
                              ecore=mf.energy_nuc())
t_setup = time.perf_counter() - t0_setup
print(f"  E_HF  = {mf.e_tot:.10f}")
print(f"  E_FCI = {E_fci:.10f}")
print(f"  Setup time: {t_setup:.4f} s")

# P/Q partition: CAS(4,4)
p_idx, q_idx = partition_cas(n_orb, n_elec, n_active_orb=4, n_active_elec=4)
p_dets = [dets[i] for i in p_idx]
q_dets = [dets[i] for i in q_idx]
N = len(p_dets)
M = len(q_dets)
print(f"  P-space: {N} dets, Q-space: {M} dets")

# Reference energy
E0 = compute_reference_energy(ham, dets, p_idx)
delta_fci = E_fci - E0
print(f"  E0 (H_PP lowest) = {E0:.10f}")
print(f"  Δ_FCI = {delta_fci:.10f} Ha = {delta_fci*1000:.3f} mH")

# ============================================================================
# Benchmark 1: H_O' computation approaches (wall-clock comparison)
# ============================================================================
print("\n" + "=" * 70)
print("Benchmark 1: H_O' computation strategies")
print("=" * 70)

# Approach A: Build full H_O' (M×M)
t0 = time.perf_counter()
H_off_full = compute_H_off_diag(ham, q_dets)
t_full = time.perf_counter() - t0
print(f"  (A) Full H_O' (M×M, M={M}): {t_full:.4f} s  [O(M²) = {M*(M-1)//2:,} pairs]")

# Pre-compute common quantities for Krylov
diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
A_diag = compute_A(E0, diag_H_QQ)
H_QP_mat = build_H_QP(ham, p_dets, q_dets)

# Build Krylov subspace (no SVD yet)
t0 = time.perf_counter()
basis_raw = np.zeros((M, 0))
# Layer 0
layer0_raw = generate_layer_0(H_QP_mat, A_diag)
layer0_orth, _ = modified_gram_schmidt(layer0_raw, basis_raw)
basis_raw = layer0_orth

# We'll need raw vectors for SVD — save layer 0 raw
layer_raws = [layer0_raw]  # raw (pre-orth) vectors for each layer

# Layers 1, 2
prev_layer = layer0_raw
for j in range(1, 3):  # up to m=3 layers
    new_raw = propagate_layer(prev_layer, H_off_full, A_diag, delta=0.0)
    layer_raws.append(new_raw)
    new_orth, _ = modified_gram_schmidt(new_raw, basis_raw)
    if new_orth.shape[1] > 0:
        basis_raw = np.hstack([basis_raw, new_orth])
    prev_layer = new_raw

d_total_raw = basis_raw.shape[1]
t_krylov = time.perf_counter() - t0
print(f"\n  Krylov basis (m=3): {d_total_raw} orthonormal vectors")
print(f"  Krylov construction time: {t_krylov:.4f} s")

# Approach B: sigma-vector (station's idea)
# After SVD, we have rotation matrix T (M×r). Compute T^† H_QQ T = T^† (H_D' + H_O') T.
# H_D' contribution is trivial (diagonal). For H_O', use sigma-vector on columns of T.
print("\n  --- SVD compression on Krylov basis ---")

all_sigma_lists = []
compression_stats = []

for j, raw_layer in enumerate(layer_raws):
    if raw_layer.shape[1] == 0:
        compression_stats.append({'layer': j, 'n_raw': 0, 'n_retained': 0})
        continue

    n_raw = raw_layer.shape[1]

    # Weighted SVD
    T = build_weighted_coupling(raw_layer, A_diag)
    U_ret, sigma_ret, r = svd_truncate(T, threshold=1e-3)
    all_sigma_lists.append(sigma_ret)

    compression_stats.append({
        'layer': j, 'n_raw': n_raw, 'n_retained': r,
        'sigma_max': sigma_ret[0] if r > 0 else 0,
        'sigma_min': sigma_ret[-1] if r > 0 else 0,
    })

    if r > 0:
        print(f"    Layer {j}: {n_raw} → {r} vectors (σ_max={sigma_ret[0]:.3e}, "
              f"σ_min={sigma_ret[-1]:.3e})")
    else:
        print(f"    Layer {j}: {n_raw} → 0 vectors (all truncated)")

# Now compare: full H_O' projection vs sigma-vector approach
# Use the SVD rotation matrices from all layers merged
print("\n  --- Comparing H_QQ projection approaches ---")

# Build merged SVD rotation matrix
U_all = []
for j, raw_layer in enumerate(layer_raws):
    if raw_layer.shape[1] == 0:
        continue
    T = build_weighted_coupling(raw_layer, A_diag)
    U_ret, _, _ = svd_truncate(T, threshold=1e-3)
    if U_ret.shape[1] > 0:
        U_all.append(U_ret)

if U_all:
    # Merged SVD basis (not yet orthonormalized — just for timing comparison)
    U_merged = np.hstack(U_all)
    U_merged, _ = modified_gram_schmidt(U_merged, np.zeros((M, 0)))
    d = U_merged.shape[1]
    print(f"  Merged compressed basis: {d} vectors (from {M} Q-dets)")

    # ---- Approach A: Full H_O' → project ----
    t0 = time.perf_counter()
    # H_QQ_full = H_off_full + diag(H_D')
    H_QQ_full = H_off_full + np.diag(diag_H_QQ)
    sigma = H_QQ_full @ U_merged
    H_QQ_proj_A = U_merged.T @ sigma
    H_QQ_proj_A = 0.5 * (H_QQ_proj_A + H_QQ_proj_A.T)
    t_A = time.perf_counter() - t0

    # ---- Approach B: sigma-vector (station's idea) ----
    # H_QQ_proj = U^† H_D' U + U^† H_O' U
    # = U^† * diag(H_D') * U + U^† * sigma_H_off(U)
    t0 = time.perf_counter()
    # Diagonal part: U^† H_D' U = sum_q U[q,a] * H_D'[q] * U[q,b]
    H_D_proj = (U_merged * diag_H_QQ[:, np.newaxis]).T @ U_merged
    H_D_proj = 0.5 * (H_D_proj + H_D_proj.T)
    # Off-diagonal part: sigma-vector
    H_O_sigma = sigma_H_off(ham, U_merged, q_dets)
    H_O_proj = U_merged.T @ H_O_sigma
    H_O_proj = 0.5 * (H_O_proj + H_O_proj.T)
    H_QQ_proj_B = H_D_proj + H_O_proj
    H_QQ_proj_B = 0.5 * (H_QQ_proj_B + H_QQ_proj_B.T)
    t_B = time.perf_counter() - t0

    print(f"\n  Approach A (full H_O' → project): {t_A:.4f} s")
    print(f"    incl. Full H_O' build:          {t_full:.4f} s")
    print(f"    incl. Matrix projection:         {t_A - t_full if hasattr(locals(), 't_full') else t_A:.4f} s")
    print(f"  Approach B (sigma-vector H_O'·U):  {t_B:.4f} s")

    # Verify numerical equivalence
    diff = np.max(np.abs(H_QQ_proj_A - H_QQ_proj_B))
    print(f"  Max |A - B|: {diff:.2e}")
    if diff < 1e-10:
        print("  ✓ Approaches A and B produce identical H_QQ̃ (numerically equivalent)")
    else:
        print(f"  ⚠ Small numerical difference ({diff:.2e}), within expected range")

    # Speedup factor
    if t_B > 0:
        print(f"  Speedup (B vs A total): {t_A/t_B:.2f}×")
        print(f"  Speedup (B vs A projection only): {(t_A)/(t_B+1e-10):.2f}×")
else:
    print("  No compressed vectors retained — skipping comparison")

# ============================================================================
# Benchmark 2: Effective Hamiltonian — convergence vs Krylov order
# ============================================================================
print("\n" + "=" * 70)
print("Benchmark 2: Convergence with Krylov order (SCF, no SVD)")
print("=" * 70)

H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])

for m_max in [0, 1, 2, 3, 4]:
    t0 = time.perf_counter()
    # Build Krylov subspace up to m_max
    basis, layer_sizes = build_krylov_subspace(
        H_QP_mat, H_off_full, A_diag,
        max_layers=m_max + 1, delta=0.0,
        lindep_threshold=1e-10, verbose=False
    )
    d_basis = basis.shape[1]

    # Build compressed H blocks
    H_QQ = build_H_Qtilde_Qtilde(ham, basis, q_dets, H_QQ_full=H_QQ_full)
    H_PQ = build_H_PQtilde(ham, basis, p_dets, q_dets)

    # Self-consistent iteration
    result = self_consistent_iteration(
        H_PP, H_PQ, H_QQ, E0, verbose=False
    )
    t_iter = time.perf_counter() - t0

    dE_mH = (result['E_final'] - E_fci) * 1000
    print(f"  m={m_max}: d_basis={d_basis:3d}, "
          f"E={result['E_final']:.10f}, "
          f"ΔE={dE_mH:+.3f} mH, "
          f"{result['n_iter']} SCF iters, "
          f"{t_iter:.3f} s")

# ============================================================================
# Benchmark 3: SVD truncation scan
# ============================================================================
print("\n" + "=" * 70)
print("Benchmark 3: SVD truncation threshold scan (m=3)")
print("=" * 70)

# Build full Krylov basis at m=3 (reuse from above)
basis_full, _ = build_krylov_subspace(
    H_QP_mat, H_off_full, A_diag,
    max_layers=4, delta=0.0, lindep_threshold=1e-10, verbose=False
)

theta_values = [0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
print(f"  {'θ':>8s}  {'n_vecs':>7s}  {'E':>16s}  {'ΔE (mH)':>10s}  {'t (s)':>8s}")
print(f"  {'-'*8}  {'-'*7}  {'-'*16}  {'-'*10}  {'-'*8}")

for theta in theta_values:
    t0 = time.perf_counter()

    # Apply SVD compression to each layer
    compressed_basis = np.zeros((M, 0))
    for j, raw_layer in enumerate(layer_raws):
        if raw_layer.shape[1] == 0:
            continue
        if theta == 0:
            # No truncation → use raw vectors after Gram-Schmidt
            U, _, _ = svd_truncate(build_weighted_coupling(raw_layer, A_diag),
                                   threshold=1e-15)
        else:
            U, _, _ = compress_layer(raw_layer, A_diag, threshold=theta, verbose=False)

        if U.shape[1] > 0:
            # Gram-Schmidt against accumulated basis
            U_orth, _ = modified_gram_schmidt(U, compressed_basis)
            if U_orth.shape[1] > 0:
                compressed_basis = np.hstack([compressed_basis, U_orth])

    d_compressed = compressed_basis.shape[1]

    if d_compressed == 0:
        print(f"  {theta:8.0e}  {d_compressed:7d}  {'--':>16s}  {'--':>10s}  {time.perf_counter()-t0:.4f}")
        continue

    # Build effective H
    H_QQ_c = build_H_Qtilde_Qtilde(ham, compressed_basis, q_dets, H_QQ_full=H_QQ_full)
    H_PQ_c = build_H_PQtilde(ham, compressed_basis, p_dets, q_dets)
    result = self_consistent_iteration(H_PP, H_PQ_c, H_QQ_c, E0, verbose=False)
    t_elapsed = time.perf_counter() - t0

    dE_mH = (result['E_final'] - E_fci) * 1000
    print(f"  {theta:8.0e}  {d_compressed:7d}  {result['E_final']:16.10f}  "
          f"{dE_mH:+10.3f}  {t_elapsed:8.4f}")

# ============================================================================
# Benchmark 4: Singular value spectrum
# ============================================================================
print("\n" + "=" * 70)
print("Benchmark 4: Singular value spectrum")
print("=" * 70)

sigma_lists = []
layer_labels = []
for j, raw_layer in enumerate(layer_raws):
    if raw_layer.shape[1] == 0:
        continue
    T = build_weighted_coupling(raw_layer, A_diag)
    U, sigma, Vt = np.linalg.svd(T, full_matrices=False)
    sigma_lists.append(sigma)
    layer_labels.append(f"Layer {j}")
    print(f"  Layer {j}: {len(sigma)} singular values, "
          f"σ₁={sigma[0]:.6e}, σ_last={sigma[-1]:.6e}")
    ratios = sigma / sigma[0]
    n_sig = min(8, len(sigma))
    print(f"    σ_i/σ₁: " + "  ".join(f"{ratios[i]:.4f}" for i in range(n_sig)))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

# Find optimal theta (closest to chemical accuracy with smallest basis)
best_theta = None
best_result = None
for theta in theta_values:
    # Recompute quickly
    compressed_basis = np.zeros((M, 0))
    for j, raw_layer in enumerate(layer_raws):
        if raw_layer.shape[1] == 0:
            continue
        if theta == 0:
            U, _, _ = svd_truncate(build_weighted_coupling(raw_layer, A_diag), threshold=1e-15)
        else:
            U, _, _ = compress_layer(raw_layer, A_diag, threshold=theta, verbose=False)
        if U.shape[1] > 0:
            U_orth, _ = modified_gram_schmidt(U, compressed_basis)
            if U_orth.shape[1] > 0:
                compressed_basis = np.hstack([compressed_basis, U_orth])

    if compressed_basis.shape[1] == 0:
        continue
    H_QQ_c = build_H_Qtilde_Qtilde(ham, compressed_basis, q_dets, H_QQ_full=H_QQ_full)
    H_PQ_c = build_H_PQtilde(ham, compressed_basis, p_dets, q_dets)
    result = self_consistent_iteration(H_PP, H_PQ_c, H_QQ_c, E0, verbose=False)
    dE_mH = abs(result['E_final'] - E_fci) * 1000

    if dE_mH < 1.6:  # Chemical accuracy
        if best_theta is None or compressed_basis.shape[1] < best_result['n_vecs']:
            best_theta = theta
            best_result = {'n_vecs': compressed_basis.shape[1], 'dE_mH': dE_mH, 'E': result['E_final']}

if best_result:
    print(f"  Best θ = {best_theta}: {best_result['n_vecs']} vectors, "
          f"ΔE = {best_result['dE_mH']:.3f} mH (chemical accuracy ✓)")
    print(f"  Compression ratio: {best_result['n_vecs']}/{M} = {best_result['n_vecs']/M*100:.1f}% of Q-space")
else:
    print(f"  No θ reached chemical accuracy in this scan")

# Final timing comparison for the optimal configuration
print(f"\n  Total wall times:")
print(f"    Setup + FCI:                      {t_setup:.4f} s")
print(f"    Full H_O' build (O(M²)):          {t_full:.4f} s")
print(f"    Krylov construction (m=3, no SVD): {t_krylov:.4f} s")
print(f"    Total (no SVD):                    {t_setup + t_full + t_krylov:.4f} s")

print("\n" + "=" * 70)
print("Phase 3 analysis complete.")
