#!/usr/bin/env python3
"""
Lightweight Davidson diagonalizer for the growing active-space pipeline.

For H_comp with dimension D × 4 ≤ ~4000 (typical for N₂), direct
numpy.linalg.eigh is fast enough. For larger systems, delegates to
PySCF's davidson iterative solver.

Provides a uniform interface:
    solve_hamiltonian(H, nroots, ...) → (eigenvalues, eigenvectors)

For matrix-free Hamiltonian application, use:
    solve_matvec(matvec_fn, diag, dim, nroots, ...) → (eigenvalues, eigenvectors)
"""

import numpy as np
from numpy.linalg import eigh
from typing import Tuple, Callable, Optional, Union


def solve_hamiltonian(
    H: np.ndarray,
    nroots: int = 1,
    which: str = 'SA',
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize a dense Hamiltonian matrix.

    For dim ≤ 4000, uses direct eigh. For larger, uses PySCF Davidson.

    Args:
        H:       (dim, dim) hermitian matrix.
        nroots:  Number of lowest eigenvalues/vectors.
        which:   'SA' = smallest algebraic (default).
        verbose: Print timing diagnostics.

    Returns:
        (eigvals, eigvecs):
          eigvals: (nroots,) eigenvalues.
          eigvecs: (dim, nroots) eigenvectors as columns.
    """
    dim = H.shape[0]
    if dim == 0:
        return np.array([]), np.zeros((0, 0))

    if dim <= 4000:
        # Direct diagonalization
        import time as time_mod
        t0 = time_mod.perf_counter()
        evals, evecs = eigh(H)
        elapsed = time_mod.perf_counter() - t0
        if verbose:
            print(f"  [Davidson] Direct eigh({dim}×{dim}): {elapsed:.2f}s", flush=True)
        return evals[:nroots], evecs[:, :nroots]
    else:
        # PySCF Davidson
        return _pyscf_davidson(H, nroots, which, verbose)


def solve_matvec(
    matvec: Callable[[np.ndarray], np.ndarray],
    diag: np.ndarray,
    dim: int,
    nroots: int = 1,
    which: str = 'SA',
    max_cycle: int = 100,
    tol: float = 1e-12,
    nguess: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Davidson diagonalization with a matrix-vector product function.

    Args:
        matvec:    Function x → H @ x.
        diag:      (dim,) diagonal of H (preconditioner).
        dim:       Dimension of the Hamiltonian.
        nroots:    Number of lowest eigenvalues.
        which:     'SA' for smallest algebraic.
        max_cycle: Maximum Davidson iterations.
        tol:       Convergence tolerance on residual norm.
        nguess:    Number of initial guess vectors (default: 2*nroots).
        verbose:   Print diagnostics.

    Returns:
        (eigvals, eigvecs):
          eigvals: (nroots,) converged eigenvalues.
          eigvecs: (dim, nroots) eigenvectors as columns.
    """
    if dim == 0:
        return np.array([]), np.zeros((0, 0))

    if dim <= 4000:
        # Build dense matrix from matvec and use direct eigh
        H_dense = _build_dense_from_matvec(matvec, dim)
        return solve_hamiltonian(H_dense, nroots, which, verbose)

    # PySCF Davidson
    return _pyscf_davidson_matvec(matvec, diag, dim, nroots,
                                  which, max_cycle, tol, nguess, verbose)


def _build_dense_from_matvec(
    matvec: Callable[[np.ndarray], np.ndarray],
    dim: int,
) -> np.ndarray:
    """Build dense H from matvec by applying to basis vectors."""
    H = np.zeros((dim, dim))
    for i in range(dim):
        e_i = np.zeros(dim)
        e_i[i] = 1.0
        H[:, i] = matvec(e_i)
    return 0.5 * (H + H.T)


def _pyscf_davidson(
    H: np.ndarray,
    nroots: int,
    which: str,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use PySCF's davidson iterative diagonalizer."""
    import time as time_mod
    from pyscf.lib import davidson

    dim = H.shape[0]
    diag = np.diag(H)

    def matvec(x):
        return H @ x

    t0 = time_mod.perf_counter()
    # PySCF davidson signature: davidson(matvec, x0, precond, ...)
    # For 'SA' (smallest algebraic), we negate or use davidson_nosy
    # Actually, PySCF's davidson finds the lowest eigenvalues by default
    precond = lambda dx, e, x0: dx / (diag - e + 1e-8)

    if nguess is None:
        nguess = min(2 * nroots, dim)

    x0 = np.eye(dim, nguess)  # initial guess

    # davidson doesn't exist in all PySCF versions; fall back to eigsh
    try:
        conv, e, c = davidson(
            matvec, x0, precond,
            tol=1e-10, max_cycle=100,
            nroots=nroots, verbose=0)
    except (ImportError, AttributeError, TypeError):
        # Fallback: use scipy sparse eigensolver on dense matrix
        from scipy.linalg import eigh as seigh
        evals, evecs = seigh(H, subset_by_index=[0, nroots - 1])
        elapsed = time_mod.perf_counter() - t0
        if verbose:
            print(f"  [Davidson] scipy.linalg.eigh({dim}×{dim}): "
                  f"{elapsed:.2f}s (PySCF davidson not available)",
                  flush=True)
        return evals, evecs

    elapsed = time_mod.perf_counter() - t0
    if verbose:
        print(f"  [Davidson] PySCF davidson({dim}×{dim}, {nroots} roots): "
              f"{elapsed:.2f}s", flush=True)

    return e, c


def _pyscf_davidson_matvec(
    matvec: Callable[[np.ndarray], np.ndarray],
    diag: np.ndarray,
    dim: int,
    nroots: int,
    which: str,
    max_cycle: int,
    tol: float,
    nguess: Optional[int],
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """PySCF Davidson with custom matvec."""
    import time as time_mod

    try:
        from pyscf.lib import davidson

        if nguess is None:
            nguess = min(2 * nroots, dim)

        x0 = np.eye(dim, nguess)

        precond = lambda dx, e, x0: dx / (diag - e + 1e-8)

        t0 = time_mod.perf_counter()
        conv, e, c = davidson(
            matvec, x0, precond,
            tol=tol, max_cycle=max_cycle,
            nroots=nroots, verbose=0)
        elapsed = time_mod.perf_counter() - t0

        if verbose:
            print(f"  [Davidson] PySCF davidson-matvec({dim}, {nroots} roots): "
                  f"{elapsed:.2f}s", flush=True)

        return e, c

    except (ImportError, AttributeError, TypeError):
        # Fallback: build dense and eigh
        if verbose:
            print(f"  [Davidson] PySCF davidson unavailable, "
                  f"building dense {dim}×{dim} for direct eigh", flush=True)

        H_dense = _build_dense_from_matvec(matvec, dim)
        return solve_hamiltonian(H_dense, nroots, 'SA', verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_solve_hamiltonian_small():
    """Test direct eigh on a 3x3 matrix."""
    H = np.array([[1.0, 0.2, 0.0],
                  [0.2, 2.0, 0.3],
                  [0.0, 0.3, 3.0]])
    H = 0.5 * (H + H.T)

    evals, evecs = solve_hamiltonian(H, nroots=2, verbose=False)
    assert len(evals) == 2
    assert evals[0] < evals[1]

    # Check eigenvector property: H @ v = λ v
    for k in range(2):
        Hv = H @ evecs[:, k]
        lv = evals[k] * evecs[:, k]
        assert np.allclose(Hv, lv, atol=1e-10)

    print("  ✓ solve_hamiltonian: small matrix test passed")


def test_solve_matvec():
    """Test matrix-free solver on a small system."""
    H = np.array([[1.0, 0.2, 0.0],
                  [0.2, 2.0, 0.3],
                  [0.0, 0.3, 3.0]])
    H = 0.5 * (H + H.T)

    def matvec(x):
        return H @ x

    diag = np.diag(H)
    evals, evecs = solve_matvec(matvec, diag, 3, nroots=2, verbose=False)

    evals_exact, evecs_exact = eigh(H)
    assert np.allclose(evals, evals_exact[:2], atol=1e-10)
    print("  ✓ solve_matvec: matrix-free test passed")


def test_edge_case_empty():
    """Test edge case: zero-dimensional Hamiltonian."""
    H = np.zeros((0, 0))
    evals, evecs = solve_hamiltonian(H, nroots=1, verbose=False)
    assert len(evals) == 0
    assert evecs.shape == (0, 0)
    print("  ✓ edge case: empty Hamiltonian handled")


if __name__ == "__main__":
    test_solve_hamiltonian_small()
    test_solve_matvec()
    test_edge_case_empty()
    print("All davidson_solver tests passed.")
