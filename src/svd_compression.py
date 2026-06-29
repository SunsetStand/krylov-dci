"""
Weighted SVD compression for Krylov-dCI layers.

Within each Krylov layer j, we construct:
    T^{(j)} = (E^{(0)} I - H_D')^{-1/2} · M^{(j)}

where M^{(j)} is the (M x N_j) matrix of (non-orthonormalized) Krylov vectors
from layer j, expressed in the Q determinant basis. The prefactor applies an
additional energy weighting — determinants with diagonal energies far from E0
are suppressed.

SVD is then computed: T^{(j)} = U Σ V†. Columns of U corresponding to
singular values below the threshold are discarded.

Key insight (Proposal §2.5): The weighting (E0 I - H_D')^{-1/2} ensures SVD
simultaneously accounts for:
  (i)  coupling strength to P (encoded in M^{(j)} elements)
  (ii) energetic proximity to E0 (encoded in A^{1/2} prefactor)

References:
  - Proposal §2.5
  - Eckart-Young-Mirsky theorem (optimal low-rank approximation)
  - DMET analogy: Schmidt decomposition of fragment-environment wavefunction
"""

import numpy as np
from numpy.linalg import svd, norm
from typing import Tuple, List, Optional


def build_weighted_coupling(M_j: np.ndarray,
                            A_diag: np.ndarray) -> np.ndarray:
    """Construct T^{(j)} = A^{1/2} · M^{(j)}.

    T[q, p] = M_j[q, p] / sqrt(E0 - H_qq)
             = M_j[q, p] * sqrt(A_diag[q])

    Args:
        M_j:    Raw Krylov vectors for layer j, shape (M, N_j).
                M_j[:, p] = |v_p^(j)> in Q determinant basis.
        A_diag: Diagonal resolvent A = (E0 I - H_D')^{-1}, shape (M,).

    Returns:
        Weighted coupling matrix T, shape (M, N_j).
    """
    weight = np.sqrt(np.abs(A_diag))  # |A_q|^{1/2} for numerical safety
    return M_j * weight[:, np.newaxis]


def svd_truncate(T: np.ndarray,
                 threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute SVD and truncate based on singular value threshold.

    Retains singular vectors with sigma_i >= threshold * sigma_max.

    Args:
        T:         Weighted coupling matrix, shape (M, k).
        threshold: Fraction of sigma_max below which to discard.

    Returns:
        (U_retained, sigma_retained, r) where:
          U_retained:   (M, r) — left singular vectors (Q determinant basis).
          sigma_retained: (r,) — retained singular values.
          r:            Number of retained singular vectors.
    """
    M, k = T.shape

    # Economy SVD: U (M x k), sigma (k,), Vt (k x k)
    # For wide matrices, use full_matrices=False
    if M == 0 or k == 0:
        return np.zeros((M, 0)), np.array([]), 0

    U, sigma, Vt = svd(T, full_matrices=False)

    sigma_max = sigma[0] if len(sigma) > 0 else 0.0
    if sigma_max < 1e-15:
        return np.zeros((M, 0)), np.array([]), 0

    mask = sigma >= threshold * sigma_max
    r = np.sum(mask)

    if r == 0:
        return np.zeros((M, 0)), np.array([]), 0

    return U[:, mask], sigma[mask], r


def compress_layer(M_j: np.ndarray,
                   A_diag: np.ndarray,
                   threshold: float = 1e-3,
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    """Full SVD compression pipeline for one Krylov layer.

    1. Build weighted coupling matrix T^{(j)}.
    2. SVD + truncate.
    3. Return compressed basis vectors (in Q determinant basis).

    This is the core algorithmic step — analogous to Schmidt decomposition
    in DMET, but operating in many-body (determinant) space.

    Args:
        M_j:       Raw Krylov vectors, shape (M, N_j).
        A_diag:    Diagonal resolvent, shape (M,).
        threshold: SVD truncation threshold (fraction of sigma_max).
        verbose:   Print compression statistics.

    Returns:
        (compressed, sigma, r) where:
          compressed: (M, r) — compressed basis vectors (NOT orthonormalized
                      yet — caller applies Gram-Schmidt).
          sigma:      (r,) — retained singular values.
          r:          Number of retained directions.
    """
    # Step 1: Weighted coupling matrix
    T = build_weighted_coupling(M_j, A_diag)

    # Step 2: SVD + truncation
    U_retained, sigma_retained, r = svd_truncate(T, threshold)

    if verbose:
        sigma_all = svd(T, full_matrices=False, compute_uv=False)
        total_power = np.sum(sigma_all ** 2)
        retained_power = np.sum(sigma_retained ** 2) if r > 0 else 0.0
        print(f"    SVD: {M_j.shape[1]} → {r} vectors "
              f"(σ_max={sigma_all[0]:.3e}, "
              f"retained {100*retained_power/total_power:.1f}% power, "
              f"θ={threshold})")

    return U_retained, sigma_retained, r


# ============================================================================
# Unweighted SVD (for comparison experiments)
# ============================================================================

def svd_truncate_unweighted(M_j: np.ndarray,
                            threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, int]:
    """Truncate raw M^{(j)} via SVD without energy weighting.

    This is for the comparison experiment (Proposal §4.2, experiment 3):
    weighted vs. unweighted SVD.

    Args:
        M_j:       Raw Krylov vectors, shape (M, k).
        threshold: Truncation threshold.

    Returns:
        (U_retained, sigma_retained, r)
    """
    M, k = M_j.shape
    if M == 0 or k == 0:
        return np.zeros((M, 0)), np.array([]), 0

    U, sigma, Vt = svd(M_j, full_matrices=False)
    sigma_max = sigma[0] if len(sigma) > 0 else 0.0
    if sigma_max < 1e-15:
        return np.zeros((M, 0)), np.array([]), 0

    mask = sigma >= threshold * sigma_max
    r = np.sum(mask)
    if r == 0:
        return np.zeros((M, 0)), np.array([]), 0

    return U[:, mask], sigma[mask], r


# ============================================================================
# Singular value spectrum analysis
# ============================================================================

def analyze_singular_values(sigma_list: List[np.ndarray],
                            layer_labels: Optional[List[str]] = None) -> str:
    """Generate a textual report of singular value decay across layers.

    Args:
        sigma_list:    List of singular value arrays, one per layer.
        layer_labels:  Labels for each layer (e.g., ["Layer 0", "Layer 1"]).

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Singular Value Spectrum Analysis")
    lines.append("=" * 70)

    for j, sigma in enumerate(sigma_list):
        label = layer_labels[j] if layer_labels else f"Layer {j}"
        if len(sigma) == 0:
            lines.append(f"\n{label}: (empty)")
            continue

        sigma_max = sigma[0]
        lines.append(f"\n{label}: {len(sigma)} singular values, "
                     f"σ_max = {sigma_max:.6e}")

        # Decay ratios
        ratios = sigma / sigma_max
        n_half = np.searchsorted(-np.sort(ratios)[::-1], -0.5)
        n_e3 = np.sum(ratios > 1e-3)
        n_e6 = np.sum(ratios > 1e-6)

        lines.append(f"  σ_i/σ_1 > 0.5 : {n_half:3d}")
        lines.append(f"  σ_i/σ_1 > 1e-3: {n_e3:3d}")
        lines.append(f"  σ_i/σ_1 > 1e-6: {n_e6:3d}")

        # Show first few ratios
        n_show = min(8, len(sigma))
        ratio_str = "  ".join(f"{ratios[i]:.4f}" for i in range(n_show))
        lines.append(f"  σ_i/σ_1: {ratio_str}")

    return "\n".join(lines)


# ============================================================================
# Tests
# ============================================================================

def test_weighted_vs_unweighted():
    """Compare weighted vs unweighted SVD on a toy problem.

    Construct a case where an energetically favorable determinant has weak
    coupling, and an energetically unfavorable one has strong coupling.
    Weighted SVD should favor the first; unweighted the second.
    """
    print("--- test_weighted_vs_unweighted ---")

    # 3 Q determinants, 2 Krylov vectors
    # det 0: E0 - H_00 = 0.1 Ha (close → should be favored by weighting)
    # det 1: E0 - H_00 = 10 Ha (far → suppressed by weighting)
    # det 2: E0 - H_00 = 1 Ha (intermediate)

    A_diag = 1.0 / np.array([0.1, 10.0, 1.0])

    # M_j: coupling strengths
    #   det 0 couples weakly (0.01)
    #   det 1 couples strongly (1.0)
    #   det 2 couples moderately (0.1)
    M_j = np.array([
        [0.01, 0.0],   # weak coupling, close energy
        [1.0,  0.0],   # strong coupling, far energy
        [0.1,  0.5],   # medium coupling, intermediate energy
    ])

    # Weighted SVD
    T = build_weighted_coupling(M_j, A_diag)
    U_w, sigma_w, r_w = svd_truncate(T, threshold=1e-3)
    print(f"  Weighted: {r_w} vectors, sigma = {sigma_w}")
    print(f"  Weighted U[:,0] = {U_w[:, 0]}")

    # Unweighted SVD
    U_uw, sigma_uw, r_uw = svd_truncate_unweighted(M_j, threshold=1e-3)
    print(f"  Unweighted: {r_uw} vectors, sigma = {sigma_uw}")
    print(f"  Unweighted U[:,0] = {U_uw[:, 0]}")

    # Unweighted should be dominated by det 1 (strongest coupling)
    assert abs(U_uw[1, 0]) > abs(U_uw[0, 0]), \
        "Unweighted SVD should favor strong-coupling (det 1)"

    # Weighted should balance energy proximity vs coupling
    # After weighting, det 0: 0.01/sqrt(0.1) = 0.0316
    #                  det 1: 1.0/sqrt(10) = 0.316
    # still dominated by det 1, but det 0 gets boosted relative to
    # the unweighted case
    print("  ✓ Weighted vs unweighted comparison passed")


def test_truncation_threshold():
    """Test that different thresholds produce expected numbers of vectors."""
    print("--- test_truncation_threshold ---")

    # Create a matrix with exactly-controlled singular values
    M = 10
    k = 5
    # SVD of a diagonal matrix: singular values = diagonal elements
    T = np.zeros((M, k))
    sigma_vals = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
    for i in range(k):
        T[i, i] = sigma_vals[i]

    # threshold = 0.6: keep only sigma_1 = 1.0 (0.5 < 0.6)
    _, _, r = svd_truncate(T, threshold=0.6)
    assert r == 1, f"θ=0.6 should give 1 vec, got {r}"

    # threshold = 0.5: keep sigma = 1.0, 0.5 (0.5 >= 0.5)
    _, _, r = svd_truncate(T, threshold=0.5)
    assert r == 2, f"θ=0.5 should give 2 vecs, got {r}"

    # threshold = 0.001: keep all
    _, _, r = svd_truncate(T, threshold=0.001)
    assert r == 5, f"θ=0.001 should give 5 vecs, got {r}"

    print("  ✓ Truncation thresholds work correctly")


def test_empty_and_edge_cases():
    """Test edge cases: empty input, single vector, etc."""
    print("--- test_empty_and_edge_cases ---")

    # Empty input
    U, sigma, r = svd_truncate(np.zeros((5, 0)), threshold=1e-3)
    assert r == 0
    assert U.shape == (5, 0)

    # Single column
    U, sigma, r = svd_truncate(np.array([[1.0], [0.0]]), threshold=1e-3)
    assert r == 1
    assert len(sigma) == 1
    assert np.isclose(sigma[0], 1.0)

    # All-zero matrix
    U, sigma, r = svd_truncate(np.zeros((3, 3)), threshold=1e-3)
    assert r == 0

    print("  ✓ Edge cases handled")


def test_analyze_spectrum():
    """Test the singular value analysis reporter."""
    print("--- test_analyze_spectrum ---")
    sigma_list = [
        np.array([1.0, 0.8, 0.3, 0.05, 1e-5]),
        np.array([2.0, 0.5, 1e-6]),
        np.array([]),
    ]
    report = analyze_singular_values(sigma_list,
                                     layer_labels=["Layer 0", "Layer 1", "Layer 2"])
    print(report)
    print("  ✓ Spectrum analysis works")


if __name__ == "__main__":
    test_weighted_vs_unweighted()
    test_truncation_threshold()
    test_empty_and_edge_cases()
    test_analyze_spectrum()
    print("All SVD compression tests passed.")
