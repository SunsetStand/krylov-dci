#!/usr/bin/env python3
"""
Reduced density matrix SVD and Schmidt decomposition.

For each electron-number block n, given the CI coefficient matrix C^(n):

  1.  Construct reduced density matrix  ρ_A^(n) = C^(n) [C^(n)]^†
  2.  SVD: C^(n) = U^(n) Σ^(n) [V^(n)]^†
  3.  Truncate: keep σ_α > ε · σ_max
  4.  Return Schmidt basis coefficients:

      |Ã_α^(n)⟩ = Σ_i U_{iα}^{(n)} |a_i^(n)⟩
      |B̃_α^(n)⟩ = Σ_j V_{jα}^{(n)*} |b_j^(N-n)⟩

For single-state calculations, ρ_A is constructed from one CI vector.
For multi-state, state-averaged ρ_A^SA = (1/N_states) Σ_k C^(n,k) [C^(n,k)]^†
is used, then each state is SVD'd in the shared basis.

References:
  - DensityMatrix_SVD_Embedding_Proposal.md, Sec. 2.3-2.4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def singular_value_threshold(s: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Boolean mask: which singular values are above the threshold.

    Keeps σ > eps * σ_max (with σ_max floored at 1.0 to handle all-zero blocks).

    Args:
        s: Singular values, sorted descending.
        eps: Relative threshold.

    Returns:
        Boolean array of same length as s.
    """
    if len(s) == 0:
        return np.array([], dtype=bool)
    sigma_max = max(s[0], 1.0)
    return s > eps * sigma_max


def svd_truncate_block(
    C: np.ndarray, eps: float = 1e-3,
) -> Dict:
    """Perform SVD on a single C^(n) block and truncate.

    Args:
        C: Coefficient matrix of shape (dim_A, dim_B).
        eps: Truncation threshold (relative to σ_max).

    Returns:
        dict with keys:
          'U':     (dim_A, r) — truncated left singular vectors.
          'sigma': (r,)      — truncated singular values.
          'V':     (dim_B, r) — truncated right singular vectors.
          'r':     int       — number of retained Schmidt pairs.
          'sigma_full': (min(dim_A,dim_B),) — all singular values (for diagnostics).
    """
    dim_A, dim_B = C.shape
    k = min(dim_A, dim_B)

    if k == 0 or np.allclose(C, 0):
        return {
            'U': np.zeros((dim_A, 0)),
            'sigma': np.array([]),
            'V': np.zeros((dim_B, 0)),
            'r': 0,
            'sigma_full': np.array([]),
        }

    # Full SVD (for small matrices this is fine; for larger, use economy SVD)
    U, s, Vh = np.linalg.svd(C, full_matrices=False)
    # Vh is (k, dim_B); V = Vh^T is (dim_B, k)
    V = Vh.T

    keep = singular_value_threshold(s, eps)
    r = int(np.sum(keep))

    return {
        'U': U[:, keep],
        'sigma': s[keep],
        'V': V[:, keep],
        'r': r,
        'sigma_full': s,
    }


def compute_schmidt_decomposition(
    C_blocks: Dict[int, np.ndarray],
    eps: float = 1e-3,
    state_average: Optional[List[Dict[int, np.ndarray]]] = None,
) -> Dict[int, Dict]:
    """Compute Schmidt decomposition for all electron-number blocks.

    Single-state mode: SVD direct on each C^(n).
    Multi-state mode: first compute state-averaged ρ_A^(n),
      then SVD on ρ_A^(n) (same spectrum as C^(n) but shared basis).

    Args:
        C_blocks: Dict[n] → C^(n) matrix of shape (dim_A, dim_B).
        eps: Truncation threshold.
        state_average: If provided, list of C_blocks dicts for multiple states.
                       The state-averaged ρ_A is diagonalized to get a common
                       U basis; then each state's C^(n) is compressed in that basis.

    Returns:
        Dict[n] → schmidt_data dict with keys:
          'U': (dim_A, r) — truncated left basis |Ã⟩.
          'sigma': (r,) — truncated singular values.
          'V': (dim_B, r) — truncated right basis |B̃⟩.
          'r': int — number of retained Schmidt pairs.
          'sigma_full': all singular values (diagnostic).
          'dim_A': int.
          'dim_B': int.
    """
    result = {}

    for n_A in sorted(C_blocks.keys()):
        C = C_blocks[n_A]

        if state_average is not None:
            # Multi-state: state-averaged density matrix
            rho_SA = np.zeros((C.shape[0], C.shape[0]))
            for C_k in state_average:
                Ck = C_k.get(n_A)
                if Ck is not None and Ck.shape == C.shape:
                    rho_SA += Ck @ Ck.T
            rho_SA /= len(state_average)

            # Diagonalize ρ_A^SA to get common U basis
            eigvals, U_SA = np.linalg.eigh(rho_SA)
            # Sort descending
            idx = np.argsort(-eigvals)
            eigvals = eigvals[idx]
            U_SA = U_SA[:, idx]

            # Truncate based on eigenvalues (which are σ²)
            sigma_sq = np.maximum(eigvals, 0.0)
            sigma_est = np.sqrt(sigma_sq)
            keep = singular_value_threshold(sigma_est, eps)
            r = int(np.sum(keep))
            U_trunc = U_SA[:, keep]

            # For each state, project C into this common basis to get V-equivalents
            # C = U Σ V^† ⇒ V = C^† U Σ^{-1}
            # In the truncated basis: Ṽ = C^† U_trunc
            # But for the shared basis we store U_trunc as the A-basis;
            # V for each state is computed later in embedded_hamiltonian.
            result[n_A] = {
                'U': U_trunc,
                'sigma': sigma_est[keep],
                'V': None,  # per-state V will be computed later
                'r': r,
                'sigma_full': sigma_est,
                'dim_A': C.shape[0],
                'dim_B': C.shape[1],
            }
        else:
            # Single-state: direct SVD
            svd_data = svd_truncate_block(C, eps)
            result[n_A] = {
                'U': svd_data['U'],
                'sigma': svd_data['sigma'],
                'V': svd_data['V'],
                'r': svd_data['r'],
                'sigma_full': svd_data['sigma_full'],
                'dim_A': C.shape[0],
                'dim_B': C.shape[1],
            }

    return result


# ---------- utilities for computing compression metrics ----------

def compute_compression_metrics(
    schmidt_data: Dict[int, Dict],
    C_blocks: Dict[int, np.ndarray],
    ci_vector: Optional[np.ndarray] = None,
) -> Dict:
    """Compute compression metrics from Schmidt decomposition.

    Returns:
        dict with:
          'r_total': total number of retained Schmidt pairs Σ r_n.
          'dim_fci': total number of FCI determinants.
          'compression_ratio': r_total / dim_fci.
          'discarded_weight': Σ_{σ_α < ε} σ_α^2 (2-norm error bound).
          'sigma_spectra': Dict[n] → full singular values (for plotting).
          'per_block': Dict[n] → (dim_A, dim_B, r_n, dim_full=F_A×F_B).
    """
    r_total = sum(sd['r'] for sd in schmidt_data.values())
    dim_fci = len(ci_vector) if ci_vector is not None else sum(
        blk.shape[0] * blk.shape[1] for blk in C_blocks.values())

    discarded_weight = 0.0
    sigma_spectra = {}
    per_block = {}

    for n_A, sd in schmidt_data.items():
        s_full = sd['sigma_full']
        s_trunc = sd['sigma']
        # Discarded weight: sum of σ² for σ < threshold
        if len(s_full) > len(s_trunc):
            discarded = s_full[len(s_trunc):]
            discarded_weight += np.sum(discarded ** 2)

        sigma_spectra[n_A] = s_full
        per_block[n_A] = {
            'dim_A': sd['dim_A'],
            'dim_B': sd['dim_B'],
            'r': sd['r'],
            'dim_product': sd['dim_A'] * sd['dim_B'],
        }

    return {
        'r_total': r_total,
        'dim_fci': dim_fci,
        'compression_ratio': r_total / max(dim_fci, 1),
        'discarded_weight': float(discarded_weight),
        'sigma_spectra': sigma_spectra,
        'per_block': per_block,
    }


# ---------- reconstruct CI vector from Schmidt basis (validation) ----------

def reconstruct_ci_vector(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
) -> np.ndarray:
    """Reconstruct the full CI vector from the truncated Schmidt decomposition.

    C^(n) ≈ U_trunc Σ_trunc V_trunc^† → fill back into full CI vector.

    Args:
        schmidt_data: Output of compute_schmidt_decomposition.
        partition: Output of partition_determinants (needed for coeff_map).

    Returns:
        Reconstructed CI vector (same length as FCI space).
    """
    # Determine total FCI dimension
    max_det_idx = 0
    for blk in partition.values():
        for _, _, det_idx in blk['coeff_map']:
            max_det_idx = max(max_det_idx, det_idx)
    ci_recon = np.zeros(max_det_idx + 1)

    for n_A, sd in schmidt_data.items():
        blk = partition.get(n_A)
        if blk is None:
            continue
        U = sd['U']
        sigma = sd['sigma']
        V = sd['V']
        if sd['r'] == 0:
            continue
        # Reconstruct C^(n) ≈ U diag(sigma) V^T
        C_recon = (U * sigma[np.newaxis, :]) @ V.T

        for (i, j, det_idx) in blk['coeff_map']:
            ci_recon[det_idx] = C_recon[i, j]

    return ci_recon


# ---------- tests ----------

def test_svd_small_matrix():
    """Test SVD truncation on a known rank-2 matrix."""
    # Create a rank-2 matrix with noise
    np.random.seed(42)
    u_true = np.random.randn(10, 2)
    v_true = np.random.randn(8, 2)
    sigma_true = np.array([3.0, 1.0])
    C = (u_true * sigma_true[np.newaxis, :]) @ v_true.T
    # Add tiny noise (rank > 2 but with very small singular values)
    C += 1e-14 * np.random.randn(10, 8)

    result = svd_truncate_block(C, eps=1e-3)
    assert result['r'] == 2, f"Expected r=2, got r={result['r']}"
    assert len(result['sigma']) == 2
    print(f"  ✓ Rank-2 matrix: r={result['r']}, σ={result['sigma']}")

    # Tighten threshold: only keep σ > 0.5 * σ_max
    result2 = svd_truncate_block(C, eps=0.5)
    if result2['sigma_full'][1] / result2['sigma_full'][0] < 0.5:
        assert result2['r'] == 1, f"Expected r=1 with eps=0.5, got r={result2['r']}"
        print(f"  ✓ eps=0.5: r={result2['r']}")
    else:
        print(f"  (eps=0.5 test skipped: σ1/σ0 > 0.5)")


def test_full_pipeline_h2o():
    """Integration test: partition → C blocks → SVD → reconstruction on H₂O STO-3G."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices,
    )
    from pyscf import gto, scf, ao2mo, mcscf
    from pyscf.fci import direct_spin1, cistring

    n_act, n_elec = 5, 6
    n_occ = 3
    ms = 0

    # Step 1: Partition
    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=ms)
    print(f"\n  H₂O/STO-3G CAS(5,6): {len(full_dets)} dets, "
          f"{len(partition)} blocks")

    # Step 2: Get CASCI ground-state CI vector
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = 2
    cas.kernel()
    # Get CI vector
    fcivec = cas.ci  # (n_alpha_strs, n_beta_strs)
    ci_flat = fcivec.reshape(-1)

    # Build C^(n) blocks
    C_blocks = build_block_matrices(partition, ci_flat)

    # Step 3: Schmidt decomposition
    schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat)

    print(f"  r_total={metrics['r_total']}, dim_fci={metrics['dim_fci']}, "
          f"ratio={metrics['compression_ratio']:.4f}")
    for n_A in sorted(schmidt.keys()):
        blk = metrics['per_block'][n_A]
        print(f"    n={n_A}: r={blk['r']}/{blk['dim_product']} "
              f"({blk['r']/max(blk['dim_product'],1)*100:.1f}%) "
              f"dim_A={blk['dim_A']}, dim_B={blk['dim_B']}")

    # Step 4: Reconstruct and check fidelity
    ci_recon = reconstruct_ci_vector(schmidt, partition)
    # For an exact CASCI ground state, the Schmidt decomposition should
    # be exact if we keep all singular values. Check that.
    # Actually, since we truncated, check the 2-norm error.
    overlap = np.dot(ci_recon, ci_flat)
    fidelity = overlap ** 2
    error_2norm = np.linalg.norm(ci_flat - ci_recon)
    print(f"  Fidelity: {fidelity:.8f}, |ΔΨ|₂ = {error_2norm:.6e}")

    assert fidelity > 0.9, f"Fidelity too low: {fidelity}"
    print("  ✓ Full pipeline on H₂O/STO-3G passed")


if __name__ == "__main__":
    test_svd_small_matrix()
    test_full_pipeline_h2o()
    print("All density_matrix tests passed.")