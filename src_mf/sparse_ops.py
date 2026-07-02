"""
Matrix-free sparse operations on Q-space vectors.

All operations avoid storing M-dimensional dense vectors:
  - sigma_sparse: H_QQ @ v via on-the-fly Slater-Condon
  - sparse_mgs: modified Gram-Schmidt on sparse vectors
  - gram_svd: SVD of sparse block via Gram matrix eigen-decomposition
  - build_hqp_block: construct H_QP as sparse vectors via P→Q excitations
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.linalg import eigh

from .sparse_vector import SparseQVector


def generate_connected_determinants(
    alpha_str: int, beta_str: int, n_orb: int
) -> List[Tuple[Tuple[int, int], int, int, int, int]]:
    """Generate all determinants connected via 1-2 spin-orbital excitations."""
    from src.determinants import bit_positions

    alpha_occ = bit_positions(alpha_str)
    beta_occ = bit_positions(beta_str)
    all_orbs = list(range(n_orb))
    alpha_virt = [p for p in all_orbs if p not in alpha_occ]
    beta_virt = [p for p in all_orbs if p not in beta_occ]

    results = []

    # Singles: α→α
    for i in alpha_occ:
        for a in alpha_virt:
            new_a = (alpha_str ^ (1<<i)) | (1<<a)
            results.append(((new_a, beta_str), i, a, -1, -1))

    # Singles: β→β
    for i in beta_occ:
        for a in beta_virt:
            new_b = (beta_str ^ (1<<i)) | (1<<a)
            results.append(((alpha_str, new_b), i, a, -1, -1))

    # Doubles: αα→αα
    if len(alpha_occ) >= 2:
        for idx_i, i in enumerate(alpha_occ):
            for j in alpha_occ[idx_i+1:]:
                for idx_a, va in enumerate(alpha_virt):
                    for vb in alpha_virt[idx_a+1:]:
                        new_a = (alpha_str ^ (1<<i) ^ (1<<j)) | (1<<va) | (1<<vb)
                        results.append(((new_a, beta_str), i, va, j, vb))

    # Doubles: ββ→ββ
    if len(beta_occ) >= 2:
        for idx_i, i in enumerate(beta_occ):
            for j in beta_occ[idx_i+1:]:
                for idx_a, va in enumerate(beta_virt):
                    for vb in beta_virt[idx_a+1:]:
                        new_b = (beta_str ^ (1<<i) ^ (1<<j)) | (1<<va) | (1<<vb)
                        results.append(((alpha_str, new_b), i, va, j, vb))

    # Doubles: αβ→αβ
    for i in alpha_occ:
        for j in beta_occ:
            for va in alpha_virt:
                for vb in beta_virt:
                    new_a = (alpha_str ^ (1<<i)) | (1<<va)
                    new_b = (beta_str ^ (1<<j)) | (1<<vb)
                    results.append(((new_a, new_b), i, va, j, vb))

    return results


def build_hqp_sparse(
    p_dets: List[Tuple[int, int]],
    ham,                # Hamiltonian object with diagonal_element + matrix_element
    A_diag_func,        # callable: (alpha_str, beta_str) → float (resolvent weight)
    n_orb: int,
    skip_P: set = None
) -> List[SparseQVector]:
    """Build H_QP columns as sparse vectors.

    For each P determinant, generates all connected Q determinants and
    computes the coupling h_qp weighted by the diagonal resolvent A_q.

    Args:
        p_dets:    List of P-space (alpha_str, beta_str) tuples.
        ham:       Hamiltonian object.
        A_diag_func:  f(alpha_str, beta_str) → resolvent weight for that Q det.
        n_orb:     Number of spatial orbitals.
        skip_P:    Set of P determinants to exclude from Q (default: None → use all p_dets).

    Returns:
        List of SparseQVector, one per P determinant. Each is A_q · H_QP[:,p].
    """
    if skip_P is None:
        skip_P = set(p_dets)

    columns = []
    for p_idx, (pa, pb) in enumerate(p_dets):
        col = SparseQVector()
        connected = generate_connected_determinants(pa, pb, n_orb)
        for det_q, *_ in connected:
            if det_q in skip_P:
                continue  # Skip P-space determinants
            a_weight = A_diag_func(det_q[0], det_q[1])
            if abs(a_weight) < 1e-14:
                continue
            h_qp = ham.matrix_element(det_q, (pa, pb))
            if abs(h_qp) > 1e-14:
                col[det_q] = a_weight * h_qp
        columns.append(col)
    return columns


def sparse_mgs(
    new_vectors: List[SparseQVector],
    existing_basis: List[SparseQVector],
    lindep_threshold: float = 1e-10
) -> List[SparseQVector]:
    """Modified Gram-Schmidt on sparse vectors.

    For each new vector:
      1. Orthogonalize against existing basis columns.
      2. Orthogonalize against previously accepted new columns.
      3. If residual norm < threshold, discard.
      4. Normalize.

    Args:
        new_vectors:    List of SparseQVector to orthonormalize.
        existing_basis: List of existing orthonormal SparseQVector.
        lindep_threshold: Discard vectors with residual norm below this.

    Returns:
        List of accepted orthonormal SparseQVector.
    """
    # Work on copies
    V = [v.copy() for v in new_vectors]
    d_existing = len(existing_basis)
    retained = []

    for j in range(len(V)):
        v = V[j]

        # Orthogonalize against existing basis
        for i in range(d_existing):
            proj = existing_basis[i].dot(v)
            v.add_scaled(existing_basis[i], alpha=-proj)

        # Orthogonalize against previously accepted new vectors
        for idx in retained:
            w = V[idx]
            proj = w.dot(v)
            v.add_scaled(w, alpha=-proj)

        nrm = v.norm()
        if nrm > lindep_threshold:
            v.scale(1.0 / nrm)
            retained.append(j)

    return [V[j] for j in retained]


def gram_svd(
    vectors: List[SparseQVector],
    weights: Optional[List[float]] = None,
    threshold: float = 1e-12
) -> Tuple[List[SparseQVector], np.ndarray, int]:
    """Compute SVD of a block of sparse vectors via Gram matrix.

    Builds G_{ij} = ⟨w_i · v_i, w_j · v_j⟩ (weighted Gram matrix),
    then eigen-decomposes to get singular values and right singular vectors.
    Left singular vectors are reconstructed as linear combinations of the
    input vectors.

    Args:
        vectors:  List of N SparseQVector (columns of the matrix).
        weights:  Optional per-column weights (A^{1/2} factors). If None, identity.
        threshold: Singular value cutoff.

    Returns:
        (compressed_vectors, singular_values, rank) where:
          compressed_vectors: list of SparseQVector (left singular vectors × Σ)
          singular_values:   array of retained singular values
          rank:              number of retained vectors
    """
    N = len(vectors)
    if N == 0:
        return [], np.array([]), 0

    # Build Gram matrix G_{ij} = ⟨weighted_i, weighted_j⟩
    G = np.zeros((N, N))
    if weights is not None:
        # Weighted vectors — precompute weighted dot products
        for i in range(N):
            wi = vectors[i]
            a_i = weights[i] if weights is not None else 1.0
            for j in range(i, N):
                wj = vectors[j]
                a_j = weights[j] if weights is not None else 1.0
                d = a_i * a_j * wi.dot(wj)
                G[i, j] = d
                G[j, i] = d
    else:
        for i in range(N):
            for j in range(i, N):
                d = vectors[i].dot(vectors[j])
                G[i, j] = d
                G[j, i] = d

    # Eigen-decomposition (G is real symmetric positive semidefinite)
    eigvals, eigvecs = eigh(G)
    # eigvals are ascending; we want descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Compute singular values σ_k = √λ_k
    # and compressed left vectors u_k = Σ_i V_{i,k} · v_i / σ_k
    r = int(np.sum(eigvals > threshold))

    compressed = []
    sigmas = []
    for k in range(r):
        sigma_k = np.sqrt(max(eigvals[k], 0.0))
        sigmas.append(sigma_k)
        # Left singular vector (weighted by sigma for later use):
        # u_k * σ_k = Σ_i V_{i,k} · v_i  (no division by sigma — we store U·Σ)
        uk = SparseQVector()
        for i in range(N):
            coeff = eigvecs[i, k]
            if abs(coeff) > 1e-14:
                uk.add_scaled(vectors[i], alpha=coeff)
        compressed.append(uk)

    return compressed, np.array(sigmas), r


def sigma_sparse(
    vec: SparseQVector,
    ham,                # Hamiltonian object
    n_orb: int,
    diag_func=None      # optional: (a,b)→float for diagonal contribution
) -> SparseQVector:
    """Compute H_QQ @ v for sparse v via on-the-fly Slater-Condon.

    For each non-zero entry in v, generates all connected Q determinants
    and accumulates the off-diagonal contributions h_rq * v[q].
    Diagonal contributions from diag_func are added if provided.

    This NEVER enumerates the full Q-space — only accesses determinants
    connected to the support of v.

    Complexity: O(nnz(v) × n_exc) where n_exc ≈ n_occ²·n_vir².
    """
    result = SparseQVector()

    for det_q, coef in vec.items():
        if abs(coef) < 1e-16:
            continue

        # Diagonal contribution
        if diag_func is not None:
            diag_val = diag_func(det_q[0], det_q[1])
            result[det_q] = result.get(det_q, 0.0) + diag_val * coef

        # Off-diagonal: generate connected Q determinants
        connected = generate_connected_determinants(det_q[0], det_q[1], n_orb)
        for det_r, *_ in connected:
            h_rq = ham.matrix_element(det_r, det_q)
            if abs(h_rq) > 1e-14:
                result[det_r] = result.get(det_r, 0.0) + h_rq * coef

    return result


def project_hqq(
    basis: List[SparseQVector],
    ham,
    n_orb: int,
    diag_func=None
) -> np.ndarray:
    """Compute H_{Q̃Q̃} via streaming double-sum (no full sigma stored).

    H_{Q̃Q̃}[j,k] = Σ_{q ∈ supp(B_j)} Σ_{r ∈ conn(q)} B[j,q] · H_QQ[q,r] · B[k,r]

    The outer loop iterates over each basis vector's support; for each
    connected Q determinant, we immediately dot with all basis vectors.
    The full sigma vectors are never materialized.
    """
    r = len(basis)
    if r == 0:
        return np.zeros((0, 0))

    H_tilde = np.zeros((r, r))

    # Build fast-lookup dicts for each basis vector: {det → coefficient}
    basis_dicts = [dict(b.items()) for b in basis]

    for k in range(r):
        b_k = basis[k]
        b_k_dict = basis_dicts[k]

        # Diagonal contribution: Σ_q B[j,q] · H_QQ[q,q] · B[k,q]
        if diag_func is not None:
            for det_q, coef_k in b_k.items():
                h_qq = diag_func(det_q[0], det_q[1])
                contrib = coef_k * h_qq
                for j in range(r):
                    coef_j = basis_dicts[j].get(det_q, 0.0)
                    if abs(coef_j) > 1e-16:
                        H_tilde[j, k] += coef_j * contrib

        # Off-diagonal: generate connected determinants for each q in supp(B_k)
        for det_q, coef_k in b_k.items():
            connected = generate_connected_determinants(
                det_q[0], det_q[1], n_orb)
            for det_r, *_ in connected:
                h_qr = ham.matrix_element(det_q, det_r)
                if abs(h_qr) < 1e-14:
                    continue
                contrib = coef_k * h_qr
                # Dot with all basis vectors that have det_r in support
                for j in range(r):
                    coef_j = basis_dicts[j].get(det_r, 0.0)
                    if abs(coef_j) > 1e-16:
                        H_tilde[j, k] += coef_j * contrib

    # Symmetrize
    H_tilde = 0.5 * (H_tilde + H_tilde.T)
    return H_tilde


def project_hpq(
    p_dets: List[Tuple[int, int]],
    basis: List[SparseQVector],
    ham,
    n_orb: int,
    skip_P: set = None
) -> np.ndarray:
    """Compute H_{P~Q}[p, k] = ⟨Φ_p|H|b_k⟩.

    For each P determinant, generates connected Q determinants and
    accumulates H_PQ[p,q] · B[q,k].

    Args:
        p_dets:  List of P-space (alpha_str, beta_str) tuples.
        basis:   List of r SparseQVector.
        ham:     Hamiltonian object.
        n_orb:   Number of spatial orbitals.
        skip_P:  Set of P dets to skip (default: use p_dets itself).

    Returns:
        (N, r) matrix H_{P~Q}.
    """
    N = len(p_dets)
    r = len(basis)
    if skip_P is None:
        skip_P = set(p_dets)

    H_PQtilde = np.zeros((N, r))
    for p_idx, (pa, pb) in enumerate(p_dets):
        connected = generate_connected_determinants(pa, pb, n_orb)
        for det_q, *_ in connected:
            if det_q in skip_P:
                continue
            h_pq = ham.matrix_element((pa, pb), det_q)
            if abs(h_pq) < 1e-14:
                continue
            for k in range(r):
                b_qk = basis[k].get(det_q, 0.0)
                if abs(b_qk) > 1e-16:
                    H_PQtilde[p_idx, k] += h_pq * b_qk

    return H_PQtilde
