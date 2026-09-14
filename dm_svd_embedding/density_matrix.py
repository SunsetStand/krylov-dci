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
For multi-state, state-averaged
ρ_A^SA = Σ_k w_k C^(n,k) [C^(n,k)]^† is used, then each state is
represented in the shared basis.  Equal weights remain the default.

References:
  - DensityMatrix_SVD_Embedding_Proposal.md, Sec. 2.3-2.4
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def normalize_state_weights(
    n_states: int,
    state_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Validate and normalize state-averaging weights.

    Args:
        n_states: Number of states in the average.
        state_weights: Optional non-negative weights.  They need not sum to one.

    Returns:
        A float64 vector of length ``n_states`` that sums to one.
    """
    if n_states <= 0:
        raise ValueError("state averaging requires at least one state")

    if state_weights is None:
        return np.full(n_states, 1.0 / n_states, dtype=float)

    weights = np.asarray(state_weights, dtype=float)
    if weights.ndim != 1 or len(weights) != n_states:
        raise ValueError(
            f"state_weights must have length {n_states}, got shape {weights.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("state_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("state_weights must be non-negative")

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("at least one state weight must be positive")
    return weights / total


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
    state_weights: Optional[np.ndarray] = None,
    rank_mode: str = 'rectangular',
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
        state_weights: Optional non-negative weights for ``state_average``.
                       Equal weights are used when omitted.
        rank_mode: ``'rectangular'`` keeps r_A(n) and r_B(n) independently,
                   which is the method's default.  ``'symmetric'`` forces
                   r_A = r_B = min(r_A, r_B) and exists only as the control
                   for hypothesis H6; a fair comparison must match the total
                   embedded dimension by adjusting ``eps``, not by comparing
                   the two modes at the same threshold.

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
    if rank_mode not in ('rectangular', 'symmetric'):
        raise ValueError(f"unknown rank_mode {rank_mode!r}")
    result = {}
    weights = None
    if state_average is not None:
        weights = normalize_state_weights(len(state_average), state_weights)

    for n_A in sorted(C_blocks.keys()):
        C = C_blocks[n_A]

        if state_average is not None:
            # Multi-state: state-averaged density matrix
            rho_SA = np.zeros((C.shape[0], C.shape[0]))
            for weight, C_k in zip(weights, state_average):
                Ck = C_k.get(n_A)
                if Ck is not None and Ck.shape == C.shape:
                    rho_SA += weight * (Ck @ Ck.T)

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

            # Also diagonalize ρ_B^SA to get common V basis (symmetric to ρ_A^SA)
            rho_B_SA = np.zeros((C.shape[1], C.shape[1]))
            for weight, C_k in zip(weights, state_average):
                Ck = C_k.get(n_A)
                if Ck is not None and Ck.shape == C.shape:
                    rho_B_SA += weight * (Ck.T @ Ck)

            eigvals_B, V_SA = np.linalg.eigh(rho_B_SA)
            idx_B = np.argsort(-eigvals_B)
            eigvals_B = eigvals_B[idx_B]
            V_SA = V_SA[:, idx_B]

            sigma_sq_B = np.maximum(eigvals_B, 0.0)
            sigma_est_B = np.sqrt(sigma_sq_B)
            keep_B = singular_value_threshold(sigma_est_B, eps)
            r_B = int(np.sum(keep_B))
            V_trunc_B = V_SA[:, keep_B]

            # A state average is generally not one pure bipartite state:
            # rho_A^SA and rho_B^SA can therefore have different ranks.  Keep
            # both retained subspaces.  The embedded basis for this block is
            # rectangular, with dimension r_A * r_B.
            sigma_A = sigma_est[keep]
            sigma_B = sigma_est_B[keep_B]
            if rank_mode == 'symmetric':
                # H6 control: discard the asymmetry deliberately.
                r_sym = min(r, r_B)
                U_trunc = U_SA[:, :r_sym]
                V_trunc_B = V_SA[:, :r_sym]
                sigma_A = sigma_est[:r_sym]
                sigma_B = sigma_est_B[:r_sym]
                r = r_sym
                r_B = r_sym
            r_common = min(r, r_B)  # legacy paired-rank diagnostic only

            result[n_A] = {
                'U': U_trunc,
                'sigma': sigma_A[:r_common],
                'sigma_A': sigma_A,
                'sigma_B': sigma_B,
                'V': V_trunc_B,
                'r': r_common,
                'sigma_full': sigma_est,       # ρ_A eigenvalues for diagnostics
                'sigma_full_B': sigma_est_B,   # ρ_B eigenvalues for diagnostics
                'r_A': r,
                'r_B': r_B,
                'dim_A': C.shape[0],
                'dim_B': C.shape[1],
                'rho_A_state_averaged': rho_SA,
                'rho_B_state_averaged': rho_B_SA,
                'state_weights': weights.copy(),
                'rank_mode': rank_mode,
            }
        else:
            # Single-state: direct SVD
            svd_data = svd_truncate_block(C, eps)
            result[n_A] = {
                'U': svd_data['U'],
                'sigma': svd_data['sigma'],
                'V': svd_data['V'],
                'r': svd_data['r'],
                'r_A': svd_data['r'],
                'r_B': svd_data['r'],
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
    r_A_total = sum(sd.get('r_A', sd['r']) for sd in schmidt_data.values())
    r_B_total = sum(sd.get('r_B', sd['r']) for sd in schmidt_data.values())
    product_dim = sum(
        sd.get('r_A', sd['r']) * sd.get('r_B', sd['r'])
        for sd in schmidt_data.values())
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
            'r_A': sd.get('r_A', sd['r']),
            'r_B': sd.get('r_B', sd['r']),
            'product_dim': (
                sd.get('r_A', sd['r']) * sd.get('r_B', sd['r'])),
            'dim_product': sd['dim_A'] * sd['dim_B'],
        }

    return {
        'r_total': r_total,
        'r_A_total': r_A_total,
        'r_B_total': r_B_total,
        'product_dim': product_dim,
        'dim_fci': dim_fci,
        'compression_ratio': r_total / max(dim_fci, 1),
        'product_compression_ratio': product_dim / max(dim_fci, 1),
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
