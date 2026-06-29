"""
Effective Hamiltonian construction and self-consistent iteration.

Given a compressed Krylov subspace basis (from svd_compression.py), builds:
    H_P^eff(Delta) = H_PP + H_{P~Q} ((E0+Delta)I - H_{~Q~Q})^{-1} H_{~Q P}

where ~Q denotes the compressed Q-subspace (dimension d << M).

Supports two modes:
  1. Fixed Delta (non-self-consistent): supply Delta externally (from FCI).
  2. Self-consistent Delta: iterate until energy converges.

References:
  - Proposal §2.6
  - Löwdin partitioning: J. Math. Phys. 3, 969 (1962)
"""

import numpy as np
from numpy.linalg import eigh, norm, inv
from typing import Tuple, List, Optional, Dict
from warnings import warn


# ============================================================================
# Hamiltonian blocks in compressed Krylov basis
# ============================================================================

def build_H_Qtilde_Qtilde(ham, basis: np.ndarray,
                          q_dets: List[Tuple[int, int]],
                          H_QQ_full: np.ndarray = None) -> np.ndarray:
    """Construct H_{~Q~Q} in the compressed Krylov basis.

    H_{~Q~Q}[k,l] = <w_k| H |w_l> = basis^T @ H_QQ @ basis

    If H_QQ_full is provided, uses the pre-computed M×M Hamiltonian.
    Otherwise computes it on the fly (SLOW, O(M²·d)).

    Args:
        ham:       Hamiltonian object.
        basis:     (M, d) orthonormal basis matrix.
        q_dets:    List of Q-space determinants, length M.
        H_QQ_full: Optional pre-computed M×M Q-space Hamiltonian.

    Returns:
        (d, d) matrix H_{~Q~Q}.
    """
    M, d = basis.shape
    
    if H_QQ_full is not None:
        # Fast path: project pre-computed H_QQ
        sigma = H_QQ_full @ basis
        H_QQ_proj = basis.T @ sigma
        return 0.5 * (H_QQ_proj + H_QQ_proj.T)
    
    # Slow path: O(M²·d) — for small systems only
    from src.krylov import compute_H_off_diag
    diag = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    H_off = compute_H_off_diag(ham, q_dets)
    H_QQ_full_mat = H_off + np.diag(diag)
    sigma = H_QQ_full_mat @ basis
    H_QQ_proj = basis.T @ sigma
    return 0.5 * (H_QQ_proj + H_QQ_proj.T)


def build_H_PQtilde(ham, basis: np.ndarray,
                    p_dets: List[Tuple[int, int]],
                    q_dets: List[Tuple[int, int]]) -> np.ndarray:
    """Construct H_{P~Q} in the compressed Krylov basis.

    H_{P~Q}[p, k] = <Phi_p| H |w_k>
                   = sum_j basis[j, k] * <p|H|j>

    Args:
        ham:    Hamiltonian object.
        basis:  (M, d) orthonormal basis in Q determinant basis.
        p_dets: P-space determinant list (length N).
        q_dets: Q-space determinant list (length M).

    Returns:
        (N, d) matrix H_{P~Q}.
    """
    N = len(p_dets)
    M, d = basis.shape
    H_PQ = np.zeros((N, d))

    for p in range(N):
        for k in range(d):
            for j in range(M):
                if abs(basis[j, k]) < 1e-14:
                    continue
                h_pj = ham.matrix_element(p_dets[p], q_dets[j])
                H_PQ[p, k] += basis[j, k] * h_pj

    return H_PQ


# ============================================================================
# Effective Hamiltonian
# ============================================================================

def build_effective_H(H_PP: np.ndarray,
                      H_PQtilde: np.ndarray,
                      H_Qtilde_Qtilde: np.ndarray,
                      E0: float,
                      delta: float = 0.0) -> np.ndarray:
    """Build the effective Hamiltonian at a given Delta.

    H_P^eff(Delta) = H_PP + H_{P~Q} ((E0+Delta)I - H_{~Q~Q})^{-1} H_{~Q P}

    The resolvent is computed by direct inversion (dimension d << M).

    Args:
        H_PP:            (N, N) P-space Hamiltonian.
        H_PQtilde:       (N, d) P-~Q coupling.
        H_Qtilde_Qtilde: (d, d) ~Q Hamiltonian.
        E0:              Reference energy (lowest eigenvalue of H_PP).
        delta:           Energy shift Delta = E - E0.

    Returns:
        (N, N) effective Hamiltonian (Hermitian, real-symmetric).
    """
    # Ensure real symmetric (numerical noise may introduce asymmetry)
    H_PP = 0.5 * (H_PP + H_PP.T)
    H_QQ = 0.5 * (H_Qtilde_Qtilde + H_Qtilde_Qtilde.T)

    d = H_QQ.shape[0]
    if d == 0:
        # No Q-space: effective H = H_PP
        return H_PP.copy()

    # Resolvent: (E I - H_QQ)^{-1} where E = E0 + Delta
    E = E0 + delta
    resolvent = inv(E * np.eye(d) - H_QQ)

    # Correction: H_PQ @ resolvent @ H_QP
    H_QP = H_PQtilde.T
    correction = H_PQtilde @ resolvent @ H_QP

    H_eff = H_PP + correction
    return 0.5 * (H_eff + H_eff.T)  # Hermitize


def diagonalize_effective_H(H_eff: np.ndarray,
                            n_states: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the effective Hamiltonian.

    Args:
        H_eff:    (N, N) effective Hamiltonian (real symmetric).
        n_states: Number of lowest eigenstates to extract.

    Returns:
        (energies, eigenvectors) — eigenvectors[:, i] is in P-space basis.
    """
    if H_eff.shape[0] == 0:
        return np.array([]), np.zeros((0, 0))

    eigvals, eigvecs = eigh(H_eff)
    return eigvals[:n_states], eigvecs[:, :n_states]


# ============================================================================
# Self-consistent iteration
# ============================================================================

def self_consistent_iteration(H_PP: np.ndarray,
                              H_PQtilde: np.ndarray,
                              H_Qtilde_Qtilde: np.ndarray,
                              E0: float,
                              delta_init: float = 0.0,
                              max_iter: int = 50,
                              thr_energy: float = 1e-10,
                              thr_grad: float = 1e-8,
                              damping: float = 0.3,
                              verbose: bool = True,
                              diis: bool = True,
                              n_diis: int = 6) -> Dict:
    """Self-consistent iteration for the energy shift Delta = E - E0.

    Iterates:
      1. Build H_P^eff(Delta)
      2. Diagonalize → new E
      3. Update Delta_new = E_new - E0
      4. Check convergence

    Uses DIIS acceleration for the Delta vector when diis=True.

    Args:
        H_PP:            (N, N) P-space Hamiltonian.
        H_PQtilde:       (N, d) P-~Q coupling.
        H_Qtilde_Qtilde: (d, d) ~Q Hamiltonian.
        E0:              Reference energy.
        delta_init:      Initial guess for Delta.
        max_iter:        Maximum number of iterations.
        thr_energy:      Convergence threshold on |E_new - E_old| (Hartree).
        thr_grad:        Convergence threshold on |Delta_new - Delta_old|.
        damping:         Simple mixing factor (used when diis=False or as
                         fallback).
        verbose:         Print iteration progress.
        diis:            Use DIIS acceleration.
        n_diis:          Number of DIIS vectors to store.

    Returns:
        dict with keys:
          converged:    bool
          n_iter:       number of iterations used
          E_final:      converged energy
          E_vec:        converged eigenvector in P basis
          delta_final:  final Delta
          history:      list of (iter, delta, E) tuples
    """
    delta = delta_init
    E_old = E0 + delta
    history = [(0, delta, E_old)]
    converged = False

    # DIIS storage
    error_vectors = []
    delta_vectors = []

    for it in range(1, max_iter + 1):
        # Build effective H at current Delta
        H_eff = build_effective_H(H_PP, H_PQtilde, H_Qtilde_Qtilde,
                                  E0, delta)

        # Diagonalize
        eigvals, eigvecs = diagonalize_effective_H(H_eff)
        E_new = eigvals[0]
        e_vec_new = eigvecs[:, 0]

        if verbose:
            dE = abs(E_new - E_old)
            print(f"  SCF iter {it:3d}: Δ = {delta:+.10f} Ha, "
                  f"E = {E_new:.10f}, |dE| = {dE:.6e}")

        history.append((it, delta, E_new))

        # Convergence check
        delta_new = E_new - E0
        d_delta = abs(delta_new - delta)
        d_energy = abs(E_new - E_old)

        if d_energy < thr_energy and d_delta < thr_grad:
            converged = True
            if verbose:
                print(f"  ✓ Converged in {it} iterations, "
                      f"E = {E_new:.12f} Ha")
            break

        # DIIS acceleration
        if diis and len(history) >= 2:
            # Error vector: Delta_new - Delta_old
            err = delta_new - delta
            error_vectors.append(err)
            delta_vectors.append(delta_new)

            if len(error_vectors) > n_diis:
                error_vectors.pop(0)
                delta_vectors.pop(0)

            if len(error_vectors) >= 2:
                delta_new = _diis_extrapolate(delta_vectors, error_vectors)
        else:
            # Simple mixing
            delta_new = (1 - damping) * delta + damping * delta_new

        E_old = E_new
        delta = delta_new

    if not converged:
        if verbose:
            print(f"  ⚠ Not converged after {max_iter} iterations")
        # Return best result so far
        eigvals, eigvecs = diagonalize_effective_H(
            build_effective_H(H_PP, H_PQtilde, H_Qtilde_Qtilde, E0, delta))
        E_new = eigvals[0]
        e_vec_new = eigvecs[:, 0]

    return {
        'converged': converged,
        'n_iter': it,
        'E_final': E_new,
        'E_vec': e_vec_new,
        'delta_final': delta,
        'history': history,
    }


def compute_with_fixed_delta(H_PP: np.ndarray,
                             H_PQtilde: np.ndarray,
                             H_Qtilde_Qtilde: np.ndarray,
                             E0: float,
                             delta: float) -> Tuple[float, np.ndarray]:
    """Single-shot effective H diagonalization with fixed Delta.

    This is the "non-self-consistent" mode — Delta comes from FCI.
    Used to verify Krylov method convergence independently of SCF.

    Args:
        H_PP, H_PQtilde, H_Qtilde_Qtilde: Hamiltonian blocks.
        E0:     Reference energy.
        delta:  Exact Delta = E_FCI - E0.

    Returns:
        (E, eigenvector) — lowest eigenvalue of H_P^eff(Delta).
    """
    H_eff = build_effective_H(H_PP, H_PQtilde, H_Qtilde_Qtilde, E0, delta)
    eigvals, eigvecs = diagonalize_effective_H(H_eff, n_states=1)
    return eigvals[0], eigvecs[:, 0]


# ============================================================================
# DIIS extrapolation (scalar case)
# ============================================================================

def _diis_extrapolate(delta_vecs: List[float],
                      err_vecs: List[float]) -> float:
    """DIIS extrapolation for scalar Delta.

    Minimizes ||sum_i c_i * err_i||^2 subject to sum_i c_i = 1,
    then extrapolates: Delta_DIIS = sum_i c_i * Delta_i.

    Args:
        delta_vecs: List of previous Delta values.
        err_vecs:   List of corresponding error vectors.

    Returns:
        DIIS-extrapolated Delta.
    """
    n = len(delta_vecs)
    if n == 0:
        return 0.0
    if n == 1:
        return delta_vecs[0]

    # Build B matrix: B[i,j] = err_i * err_j (scalar product)
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            B[i, j] = err_vecs[i] * err_vecs[j]
        B[i, n] = -1.0
        B[n, i] = -1.0
    B[n, n] = 0.0

    rhs = np.zeros(n + 1)
    rhs[n] = -1.0

    try:
        coeffs = np.linalg.solve(B, rhs)
        c = coeffs[:n]
        return sum(c[i] * delta_vecs[i] for i in range(n))
    except np.linalg.LinAlgError:
        # DIIS failed, use simple extrapolation
        return delta_vecs[-1]


# ============================================================================
# Convergence study: scan over Krylov orders, with or without SCF
# ============================================================================

def convergence_study(build_eff_H_fn,
                      m_max: int,
                      E_FCI: float,
                      delta_mode: str = 'fixed',
                      verbose: bool = True) -> List[Dict]:
    """Study convergence of Krylov-dCI with Krylov order m.

    Args:
        build_eff_H_fn: Function (m, delta) -> H_eff_result dict.
                        See build_krylov_effective_H_pipeline below.
        m_max:          Maximum Krylov order.
        E_FCI:          Exact FCI energy.
        delta_mode:     'fixed' (use E_FCI) or 'scf' (self-consistent).
        verbose:        Print progress.

    Returns:
        List of dicts with keys: m, E, delta_E, delta, n_vecs_total.
    """
    results = []
    E_prev = None

    for m in range(0, m_max + 1):
        if delta_mode == 'fixed':
            # Non-self-consistent: use exact FCI Delta
            E0 = None  # Will be extracted from build_eff_H_fn
            delta_fixed = None  # Filled inside
            result_m = build_eff_H_fn(m, delta='fixed', E_FCI=E_FCI)
        else:
            result_m = build_eff_H_fn(m, delta='scf')

        E_m = result_m['E']
        delta_EmH = (E_m - E_FCI) * 1000.0  # mHartree
        dE_mH = ((E_m - E_prev) * 1000.0) if E_prev is not None else None

        if verbose:
            scf_tag = "SCF" if delta_mode == 'scf' else "fixΔ"
            print(f"  m={m} [{scf_tag}]: E={E_m:.12f}, "
                  f"ΔE={delta_EmH:+.3f} mH, "
                  f"dE={f'{dE_mH:+.3f}' if dE_mH else '--'} mH, "
                  f"n_vecs={result_m.get('n_vecs_total', '?')}")

        results.append({
            'm': m,
            'E': E_m,
            'delta_E_mH': delta_EmH,
            'dE_mH': dE_mH,
            'delta': result_m.get('delta', None),
            'n_vecs_total': result_m.get('n_vecs_total', None),
            'converged': result_m.get('converged', True),
        })

        E_prev = E_m

    return results


# ============================================================================
# Test helpers
# ============================================================================

def test_build_effective_H_h2():
    """Test effective Hamiltonian construction for H2/STO-3G.

    Builds full FCI in P basis (4 dets), then constructs H_eff(m=0)
    and verifies it matches H_PP (since m=0 has no Q-space).
    """
    from pyscf import gto, scf
    import sys
    sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/krylov-dci/src')
    from hamiltonian import from_pyscf
    from determinants import generate_determinants_ms

    print("--- test_build_effective_H_h2 ---")

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    ham = from_pyscf(mol, mf)
    dets = generate_determinants_ms(2, 2, ms=0)

    # Full space: 4 determinants
    # P = first 2, Q = last 2
    p_idx = [0, 1]
    q_idx = [2, 3]
    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]

    # Build H_PP
    N, M = len(p_dets), len(q_dets)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])

    # Reference energy from H_PP
    eigvals_P, _ = eigh(H_PP)
    E0 = eigvals_P[0]

    # Build empty ~Q (m=0 effective H = H_PP)
    H_PQtilde_empty = np.zeros((N, 0))
    H_QQ_empty = np.zeros((0, 0))

    H_eff = build_effective_H(H_PP, H_PQtilde_empty, H_QQ_empty,
                              E0, delta=0.0)
    assert np.allclose(H_eff, H_PP), "m=0: H_eff should equal H_PP"
    print("  ✓ m=0 effective H equals H_PP (no Q-space)")

    eigvals_eff, _ = eigh(H_eff)
    assert np.isclose(eigvals_eff[0], E0), "m=0: eigenvalue should be E0"
    print("  ✓ m=0 eigenvalue = E0")

    # Quick sanity check: diagonalize H_PP vs build_effective_H
    eigvals_P2, _ = eigh(H_PP)
    eigvals_eff2, _ = eigh(H_eff)
    assert np.allclose(eigvals_P2, eigvals_eff2), \
        "Effective H eigenvalues should match H_PP when Q is empty"
    print(f"  E0 (H_PP lowest) = {E0:.10f}")
    print("  ✓ Effective H test passed")


if __name__ == "__main__":
    test_build_effective_H_h2()
    print("All effective Hamiltonian tests passed.")
