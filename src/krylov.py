"""
Block Krylov subspace generation for downfolding.

Constructs the Krylov subspace:
    K_m = span{ (AB)^j A H_QP |Phi_p> : 0 <= j < m, p = 1,...,N }

where:
    A = (E^(0) I - H_D')^(-1)   (diagonal, trivially invertible)
    B = H_O' - Delta * I         (off-diagonal Q-Q + energy shift)
    H_D' = diag(H_QQ)

Each layer j adds up to N new vectors. Vectors are orthonormalized
via modified Gram-Schmidt with linear dependence detection.

References:
  - Proposal Section 2.4
  - Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed., SIAM 2003.
  - Gutknecht, Acta Numerica 16, 271 (2007). (block Krylov review)
"""

from typing import List, Tuple, Optional
import numpy as np
from numpy.linalg import norm


# ============================================================================
# Core operators: A, B, and Krylov propagation
# ============================================================================

def compute_A(E0: float, diag_H_QQ: np.ndarray) -> np.ndarray:
    """Compute the diagonal resolvent A = (E0*I - H_D')^(-1).

    A_q = 1 / (E0 - H_qq) for each Q-space determinant q.

    Args:
        E0:         Reference energy (lowest eigenvalue of H_PP).
        diag_H_QQ:  Diagonal elements of H_QQ, shape (M,).

    Returns:
        Array of shape (M,) with A_diag[q] = 1/(E0 - H_qq).
    """
    # Check for near-degeneracy (small denominator → large coupling)
    denom = E0 - diag_H_QQ
    if np.any(np.abs(denom) < 1e-12):
        near_zero = np.sum(np.abs(denom) < 1e-12)
        print(f"  [WARNING] {near_zero} Q determinants near-degenerate with E0. "
              f"Consider expanding P-space.")
    return 1.0 / denom


def compute_H_off_diag(ham, q_dets: List[Tuple[int, int]]) -> np.ndarray:
    """Extract the off-diagonal Q-Q block H_O' (with zeros on diagonal).

    Builds the M x M matrix of off-diagonal elements only. The diagonal
    is stored separately as H_D'.

    WARNING: O(M^2) memory and computation. For large M, the sigma-vector
    approach (H_O'|v> on-the-fly via Slater-Condon rules) should be used
    instead. This function is for testing and small systems.

    Args:
        ham:    Hamiltonian object with matrix_element(det1, det2) method.
        q_dets: List of Q-space determinants (alpha_str, beta_str).

    Returns:
        (M, M) array with H_O'[i,j] = <q_i|H|q_j> for i != j, 0 for i == j.
    """
    M = len(q_dets)
    H_off = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            hij = ham.matrix_element(q_dets[i], q_dets[j])
            H_off[i, j] = hij
            H_off[j, i] = hij
    return H_off


def build_H_QP(ham, p_dets: List[Tuple[int, int]],
               q_dets: List[Tuple[int, int]]) -> np.ndarray:
    """Build the P-Q coupling matrix H_QP.

    H_QP[q, p] = <q|H|p> for q in Q, p in P.
    Shape: (M, N) where M = |Q|, N = |P|.

    Args:
        ham:    Hamiltonian object.
        p_dets: P-space determinants.
        q_dets: Q-space determinants.

    Returns:
        (M, N) matrix of coupling elements.
    """
    M = len(q_dets)
    N = len(p_dets)
    H_QP_mat = np.zeros((M, N))
    for q_idx, det_q in enumerate(q_dets):
        for p_idx, det_p in enumerate(p_dets):
            H_QP_mat[q_idx, p_idx] = ham.matrix_element(det_q, det_p)
    return H_QP_mat


# ============================================================================
# Krylov layer generation
# ============================================================================

def generate_layer_0(H_QP_mat: np.ndarray,
                     A_diag: np.ndarray) -> np.ndarray:
    """Generate the 0-th Krylov layer: |xi_p> = A H_QP |Phi_p>.

    These N vectors form the initial block of the Krylov subspace.
    Each vector lives in Q-space (dimension M).

    Args:
        H_QP_mat: P-Q coupling matrix, shape (M, N).
        A_diag:   Diagonal of A (energy denominators), shape (M,).

    Returns:
        Array of shape (M, N) where column p = A * H_QP[:, p].
    """
    M, N = H_QP_mat.shape
    xi = np.zeros((M, N))
    for p in range(N):
        xi[:, p] = A_diag * H_QP_mat[:, p]
    return xi


def propagate_layer(vectors: np.ndarray,
                    H_off: np.ndarray,
                    A_diag: np.ndarray,
                    delta: float = 0.0) -> np.ndarray:
    """Apply the Krylov propagator: new = (AB) * vectors = A (H_O' - delta*I) * vectors.

    Args:
        vectors:  Current layer vectors, shape (M, N_layer).
        H_off:    Off-diagonal Q-Q matrix H_O', shape (M, M).
        A_diag:   Diagonal of A, shape (M,).
        delta:    Current energy shift Delta = E - E0 (default 0 for first pass).

    Returns:
        New vectors after one Krylov step, shape (M, N_layer).
    """
    # B * v = (H_O' - delta*I) * v
    Bv = H_off @ vectors - delta * vectors
    # A * (Bv): element-wise multiplication by A_diag along each column
    return A_diag[:, np.newaxis] * Bv


# ============================================================================
# Orthonormalization: Modified Gram-Schmidt with deflation
# ============================================================================

def modified_gram_schmidt(new_vectors: np.ndarray,
                          existing_basis: np.ndarray,
                          lindep_threshold: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize new vectors against existing basis via modified Gram-Schmidt.

    For each column v of new_vectors:
      1. Orthogonalize against all columns of existing_basis.
      2. Orthogonalize against previously processed new vectors.
      3. If ||v|| < lindep_threshold, discard (linear dependence).
      4. Normalize v → ||v|| = 1.

    Modified (not classical) Gram-Schmidt is used for numerical stability.

    Args:
        new_vectors:     Array of shape (M, k) with k new vectors.
        existing_basis:  Array of shape (M, d) with d existing orthonormal
                         basis vectors (can be empty: k=0).
        lindep_threshold: Vectors with norm below this are discarded.

    Returns:
        (orthonormal_vectors, retained_indices) where:
          orthonormal_vectors: shape (M, r), r <= k, orthonormal columns.
          retained_indices: array of column indices from new_vectors that
                           were kept (non-redundant).
    """
    M, k = new_vectors.shape
    d = existing_basis.shape[1] if existing_basis.size > 0 else 0

    # Work on a copy
    V = new_vectors.copy()
    retained = []

    for j in range(k):
        v = V[:, j]

        # Orthogonalize against existing basis
        if d > 0:
            for i in range(d):
                v -= np.dot(existing_basis[:, i], v) * existing_basis[:, i]

        # Orthogonalize against previously accepted new vectors
        for r_idx in retained:
            w = V[:, r_idx]  # Already orthonormalized
            v -= np.dot(w, v) * w

        nrm = norm(v)
        if nrm > lindep_threshold:
            V[:, j] = v / nrm
            retained.append(j)
        # else: discarded as linearly dependent

    if not retained:
        return np.zeros((M, 0)), np.array([], dtype=int)

    return V[:, retained], np.array(retained)


def detect_linear_dependence(vectors: np.ndarray,
                             threshold: float = 1e-10) -> np.ndarray:
    """Identify linearly independent columns via successive norm check.

    Simpler than full Gram-Schmidt: just checks if a vector has vanishing
    norm after orthogonalizing against previous columns.

    Args:
        vectors:  Array of shape (M, k).
        threshold: Minimum norm to retain.

    Returns:
        Boolean array of length k, True for independent columns.
    """
    M, k = vectors.shape
    independent = np.ones(k, dtype=bool)

    for j in range(k):
        v = vectors[:, j].copy()
        # Orthogonalize against previously accepted vectors
        for i in range(j):
            if independent[i]:
                v -= np.dot(vectors[:, i], v) * vectors[:, i]
        if norm(v) < threshold:
            independent[j] = False

    return independent


# ============================================================================
# Full Krylov subspace construction
# ============================================================================

def build_krylov_subspace(H_QP_mat: np.ndarray,
                          H_off: np.ndarray,
                          A_diag: np.ndarray,
                          max_layers: int = 5,
                          delta: float = 0.0,
                          lindep_threshold: float = 1e-10,
                          verbose: bool = True) -> Tuple[np.ndarray, List[int]]:
    """Build the block Krylov subspace up to max_layers.

    Returns the compressed orthonormal basis spanning K_max_layers.

    Args:
        H_QP_mat:        P-Q coupling matrix, shape (M, N).
        H_off:           Off-diagonal Q-Q, shape (M, M).
        A_diag:          Diagonal resolvent, shape (M,).
        max_layers:      Maximum number of Krylov layers (>= 1).
        delta:           Current energy shift Delta = E - E0.
        lindep_threshold: Linear dependence cutoff.
        verbose:         Print progress.

    Returns:
        (basis, layer_sizes) where:
          basis:       (M, d) orthonormal matrix spanning K_max_layers.
          layer_sizes: list of lengths [d_0, d_1, ..., d_{m-1}] where
                       d_j = number of retained vectors from layer j.
    """
    M, N = H_QP_mat.shape
    all_basis = np.zeros((M, 0))  # Accumulated orthonormal basis
    layer_sizes = []

    # Layer 0
    if verbose:
        print(f"  Layer 0: generating {N} starting vectors...")
    layer0_raw = generate_layer_0(H_QP_mat, A_diag)
    layer0, _ = modified_gram_schmidt(layer0_raw, all_basis, lindep_threshold)
    d0 = layer0.shape[1]
    all_basis = layer0
    layer_sizes.append(d0)
    if verbose:
        print(f"    → retained {d0}/{N} vectors (basis total: {all_basis.shape[1]})")

    if d0 == 0:
        print("  [WARNING] All layer-0 vectors linearly dependent. Check P/Q partition.")
        return all_basis, layer_sizes

    # Layers 1, 2, ..., max_layers - 1
    prev_layer = layer0_raw  # Use raw (pre-orthonormalization) for propagation
    for j in range(1, max_layers):
        if verbose:
            print(f"  Layer {j}: propagating {prev_layer.shape[1]} vectors...")

        # Propagate
        new_raw = propagate_layer(prev_layer, H_off, A_diag, delta)

        # Orthonormalize
        new_orth, retained = modified_gram_schmidt(new_raw, all_basis, lindep_threshold)
        dj = new_orth.shape[1]
        layer_sizes.append(dj)

        if verbose:
            print(f"    → retained {dj}/{new_raw.shape[1]} vectors "
                  f"(basis total: {all_basis.shape[1] + dj})")

        if dj == 0:
            if verbose:
                print(f"  Krylov subspace exhausted at layer {j} (all new vectors "
                      f"linearly dependent).")
            break

        # Append to basis
        all_basis = np.hstack([all_basis, new_orth])

        # Use raw vectors for next propagation
        prev_layer = new_raw[:, retained]

    return all_basis, layer_sizes


# ============================================================================
# Direct sigma-vector: H_O' |v> without storing full H_off matrix
# ============================================================================

def sigma_H_off(ham, vectors: np.ndarray,
                q_dets: List[Tuple[int, int]],
                q_idx_map: Optional[dict] = None) -> np.ndarray:
    """Compute H_O' @ vectors without building the full M x M matrix.

    Uses Slater-Condon rules on-the-fly. For each column of `vectors`,
    loops over all Q determinants and computes <q_i|H|q_j> * v_j.

    NOTE: This is O(M^2 * N_layer) and should be replaced by a proper
    direct-CI sigma-vector routine for production use. This implementation
    is correct but not optimized.

    Args:
        ham:     Hamiltonian object.
        vectors: (M, k) array; column = coefficient vector over Q dets.
        q_dets:  List of Q-space determinants.
        q_idx_map: Dict mapping (alpha_str, beta_str) -> index for fast lookup.

    Returns:
        (M, k) array = H_O' @ vectors.
    """
    M = len(q_dets)
    k = vectors.shape[1]
    result = np.zeros((M, k))

    # Build index map if not provided
    if q_idx_map is None:
        q_idx_map = {(a, b): i for i, (a, b) in enumerate(q_dets)}

    for i in range(M):
        det_i = q_dets[i]
        for j in range(M):
            if i == j:
                continue  # Off-diagonal only
            hij = ham.matrix_element(det_i, q_dets[j])
            if abs(hij) > 1e-14:
                for col in range(k):
                    result[i, col] += hij * vectors[j, col]

    return result


# ============================================================================
# Tests
# ============================================================================

def test_krylov_layer0_h2():
    """Test layer-0 generation for H2/STO-3G.

    H2/STO-3G: 2 electrons, 2 spatial orbitals.
    CAS(2,2) → P = 4 determinants, Q = 0 (degenerate case).
    Let's use a manual 1-det P-space to get non-empty Q.
    """
    from pyscf import gto, scf
    from hamiltonian import Hamiltonian, from_pyscf
    from determinants import generate_determinants_ms, hf_determinant
    from partitioning import partition_cas, extract_subspace, compute_reference_energy

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    ham = from_pyscf(mol, mf)
    n_orb = 2

    # Full space: 4 determinants
    dets = generate_determinants_ms(n_orb, 2, ms=0)

    # Partition: P = CAS(2,2) → all 4 in P, Q empty (degenerate).
    # Instead, let's use a 2-det P space (HF + one excited) for test.
    # P = first 2 dets, Q = last 2 dets.
    p_idx = np.array([0, 1])
    q_idx = np.array([2, 3])
    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]

    # Reference energy from H_PP
    E0 = compute_reference_energy(ham, dets, p_idx)

    # Diagonals of H_QQ
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)

    # H_QP matrix
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)
    assert H_QP_mat.shape == (2, 2)  # M=2, N=2

    # Layer 0
    xi = generate_layer_0(H_QP_mat, A_diag)
    assert xi.shape == (2, 2)

    # Check that A was applied correctly
    for p in range(2):
        expected = np.array([
            H_QP_mat[0, p] / (E0 - diag_H_QQ[0]),
            H_QP_mat[1, p] / (E0 - diag_H_QQ[1])
        ])
        assert np.allclose(xi[:, p], expected), \
            f"Layer-0 vector {p} mismatch: {xi[:, p]} vs {expected}"

    print("  ✓ Layer-0 vectors correctly computed for H2/STO-3G")


def test_krylov_propagation_h2():
    """Test that layer-1 propagation = (AB) * layer-0.
    Uses the same 2-det P / 2-det Q setup as above.
    """
    from pyscf import gto, scf
    from hamiltonian import from_pyscf
    from determinants import generate_determinants_ms
    from partitioning import compute_reference_energy

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    ham = from_pyscf(mol, mf)
    dets = generate_determinants_ms(2, 2, ms=0)

    p_idx = np.array([0, 1])
    q_idx = np.array([2, 3])
    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]

    E0 = compute_reference_energy(ham, dets, p_idx)
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    H_off = compute_H_off_diag(ham, q_dets)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)
    A_diag = compute_A(E0, diag_H_QQ)

    # Manual check: layer1 = A * H_O' * A * H_QP
    layer0 = generate_layer_0(H_QP_mat, A_diag)
    layer1 = propagate_layer(layer0, H_off, A_diag, delta=0.0)
    assert layer1.shape == (2, 2)

    # Alternative: compute (AB) explicitly
    # AB = diag(A) * H_O'  (since B = H_O' when delta=0)
    AB = np.diag(A_diag) @ H_off
    expected_layer1 = AB @ layer0
    assert np.allclose(layer1, expected_layer1), \
        f"Layer-1 mismatch: {layer1} vs {expected_layer1}"

    print("  ✓ Layer-1 propagation = (AB) * layer-0")


def test_gram_schmidt():
    """Test modified Gram-Schmidt orthonormalization."""
    # Two linearly independent 3D vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    new_vecs = np.column_stack([v1, v2])
    existing = np.zeros((3, 0))

    orth, retained = modified_gram_schmidt(new_vecs, existing)

    assert orth.shape == (3, 2), f"Expected 2 orthonormal, got {orth.shape[1]}"
    assert np.allclose(np.dot(orth[:, 0], orth[:, 1]), 0.0), "Not orthogonal"
    assert np.allclose(norm(orth[:, 0]), 1.0), "Not normalized"
    assert np.allclose(norm(orth[:, 1]), 1.0), "Not normalized"
    assert np.array_equal(retained, [0, 1])

    print("  ✓ Gram-Schmidt: 2 vectors → 2 orthonormal")


def test_gram_schmidt_lindep():
    """Test that linearly dependent vectors are discarded."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])  # Parallel to v1
    v3 = np.array([0.0, 1.0, 0.0])  # Independent
    new_vecs = np.column_stack([v1, v2, v3])
    existing = np.zeros((3, 0))

    orth, retained = modified_gram_schmidt(new_vecs, existing)
    assert orth.shape == (3, 2), f"Expected 2 (v2 discarded), got {orth.shape[1]}"
    assert np.array_equal(retained, [0, 2]), f"Expected [0,2], got {retained}"

    print("  ✓ Gram-Schmidt: linear dependence detected and removed")


def test_full_krylov_h2():
    """Build full Krylov subspace for H2/STO-3G and verify dimensions."""
    from pyscf import gto, scf
    from hamiltonian import from_pyscf
    from determinants import generate_determinants_ms
    from partitioning import compute_reference_energy

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    ham = from_pyscf(mol, mf)
    dets = generate_determinants_ms(2, 2, ms=0)

    # P = HF det only (idx 0), Q = other 3
    p_idx = np.array([0])
    q_idx = np.array([1, 2, 3])
    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]

    E0 = compute_reference_energy(ham, dets, p_idx)
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    H_off = compute_H_off_diag(ham, q_dets)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)

    basis, layer_sizes = build_krylov_subspace(
        H_QP_mat, H_off, A_diag,
        max_layers=4, delta=0.0, lindep_threshold=1e-12, verbose=False
    )

    # With 3 Q-dets, Krylov subspace can span at most 3 dimensions
    assert basis.shape[1] <= 3, \
        f"Krylov basis exceeded Q-space dimension: {basis.shape[1]} > 3"
    print(f"  ✓ Full Krylov: {basis.shape[1]}-dim subspace from {sum(layer_sizes)} "
          f"vectors across {len(layer_sizes)} layers")
    print(f"     Layer sizes: {layer_sizes}")


if __name__ == "__main__":
    test_gram_schmidt()
    test_gram_schmidt_lindep()
    test_krylov_layer0_h2()
    test_krylov_propagation_h2()
    test_full_krylov_h2()
    print("All Krylov tests passed.")
