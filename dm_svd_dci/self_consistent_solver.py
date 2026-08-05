#!/usr/bin/env python3
"""
Self-consistent solver for the k=1 Neumann effective Hamiltonian.

Iteratively updates the reference energy E₀ by:
  1. Building resolvents A_n = (E₀ - D_n)^{-1}
  2. Computing k=0 and k=1 Neumann corrections
  3. Diagonalizing H_PP^{eff} = H_PP + Δ⁽⁰⁾ + Δ⁽¹⁾
  4. Updating E₀ ← lowest eigenvalue of H_PP^{eff}
  5. Checking convergence: |E_new - E_old| < tol

Only the ground state is solved self-consistently per user request.
The density matrix (gs/sa) mode is inherited from the upstream SVD step
and is not modified here.
"""

import numpy as np
from numpy.linalg import eigh
from typing import Dict, Optional, List, Tuple
import time
import copy

from dm_svd_dci.neumann_effective_ham import (
    build_effective_hamiltonian_neumann,
    diagonalize_and_track,
)


# ═══════════════════════════════════════════════════════════════════════════
# Self-consistent ground-state solver
# ═══════════════════════════════════════════════════════════════════════════

def solve_self_consistent(
    H_PP: np.ndarray,
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
    delta: float = 0.0,
    k_max: int = 1,
    tol: float = 1e-8,
    max_iter: int = 100,
    E0_init: Optional[float] = None,
    verbose: bool = True,
) -> Dict:
    """Self-consistent solution of the Neumann effective Hamiltonian.

    Only the ground state is solved self-consistently.  Excited states
    can be obtained by subsequent diagonalization of the converged H^eff
    using per-root tracking.

    Algorithm:
      E₀^(0) = lowest_eigenvalue(H_PP + Δ·I)
      for t = 0, 1, 2, ...:
          build H^eff(E₀^(t)) = H_PP + Δ⁽⁰⁾ + Δ⁽¹⁾
          E₀^(t+1) = lowest_eigenvalue(H^eff)
          if |E₀^(t+1) - E₀^(t)| < tol → converged

    Args:
        H_PP:        (|P|, |P|) P-space Hamiltonian.
        H_PQ:        Dict[n] → H_{P Q_n}  (|P|, d_n).
        H_QQ_blocks: Dict[(m,n)] → H_{Q_m Q_n}  (d_m, d_n).
        D_by_n:      Dict[n] → diag(H_{Q_n Q_n})  (d_n,).
        delta:       Energy shift Δ (default 0, parameter preserved for future
                     exploration).
        k_max:       Neumann order (0 or 1).
        tol:         Convergence threshold on energy (Hartree).
        max_iter:    Maximum number of self-consistent iterations.
        E0_init:     Initial reference energy.  If None, computed from H_PP.
        verbose:     Print per-iteration diagnostics.

    Returns:
        dict:
          'E_conv':          Converged ground-state energy.
          'E_vec':           (|P|,) converged ground-state eigenvector.
          'H_eff_final':     (|P|, |P|) final effective Hamiltonian.
          'n_iter':          Number of iterations performed.
          'converged':       bool — True if |ΔE| < tol.
          'E_history':       List[float] — energy at each iteration.
          'Delta_k0':        (|P|, |P|) final k=0 correction.
          'Delta_k1':        (|P|, |P|) final k=1 correction (if k_max ≥ 1).
          'final_resolvents': Dict[n] → final A_n vectors.
    """
    p_dim = H_PP.shape[0]
    if p_dim == 0:
        return {
            'E_conv': 0.0, 'E_vec': np.zeros(0),
            'H_eff_final': np.zeros((0, 0)), 'n_iter': 0, 'converged': False,
            'E_history': [], 'Delta_k0': np.zeros((0, 0)),
            'Delta_k1': np.zeros((0, 0)),
            'final_resolvents': {},
        }

    # ── Initial reference energy ──
    if E0_init is None:
        E0_init = eigh(H_PP)[0][0] + delta

    E_current = E0_init
    E_history = [E_current]

    if verbose:
        print(f"\n  {'='*60}")
        print(f"  Self-Consistent Neumann Solver (k={k_max}, Δ={delta})")
        print(f"  {'='*60}")
        print(f"  Initial E₀        = {E_current:.12f} Ha")
        print(f"  Convergence tol   = {tol:.1e} Ha")
        print(f"  Max iterations    = {max_iter}")
        print(f"  P-space dimension = {p_dim}")
        print(f"  Active Q blocks   = {sorted(H_PQ.keys())}")
        print(f"  {'-'*60}")

    converged = False
    H_eff_final = None
    Delta_k0_final = None
    Delta_k1_final = None
    A_final = {}

    for it in range(max_iter):
        t_iter = time.perf_counter()

        # Build H^eff at current E₀
        res = build_effective_hamiltonian_neumann(
            H_PP, H_PQ, H_QQ_blocks, D_by_n,
            E_current, delta=0.0,  # delta is already in E_current
            k_max=k_max, verbose=False)

        H_eff = res['H_eff']
        Delta_k0_final = res['Delta_k0'].copy()
        if k_max >= 1:
            Delta_k1_final = res.get('Delta_k1', np.zeros((p_dim, p_dim))).copy()
        else:
            Delta_k1_final = np.zeros((p_dim, p_dim))
        A_final = copy.deepcopy(res['A_by_n'])

        # Diagonalize
        evals, evecs = eigh(H_eff)
        E_new = evals[0]
        E_history.append(E_new)

        dE = E_new - E_current
        elapsed = time.perf_counter() - t_iter

        if verbose:
            print(f"  Iter {it:3d}: E₀ = {E_current:.12f} → {E_new:.12f} Ha  "
                  f"(ΔE = {dE:+.3e} Ha) [{elapsed:.1f}s]", flush=True)

        # Convergence check
        if abs(dE) < tol:
            converged = True
            H_eff_final = H_eff
            if verbose:
                print(f"\n  ✓ Converged at iteration {it}: |ΔE| = {abs(dE):.3e} < {tol:.1e}")
            break

        E_current = E_new

    if not converged:
        H_eff_final = H_eff  # use last iteration
        if verbose:
            print(f"\n  ⚠ Warning: Did not converge in {max_iter} iterations. "
                  f"Final |ΔE| = {abs(dE):.3e}")

    # ── Final ground-state eigenvector ──
    _, evecs_final = eigh(H_eff_final)
    E_vec = evecs_final[:, 0]

    if verbose:
        print(f"  Final E_conv      = {E_history[-1]:.12f} Ha")
        norm_k0 = np.linalg.norm(Delta_k0_final)
        norm_k1 = np.linalg.norm(Delta_k1_final) if k_max >= 1 else 0.0
        norm_pp = np.linalg.norm(H_PP)
        print(f"  ||Δ_k0||/||H_PP|| = {norm_k0/max(norm_pp, 1e-15):.4f}")
        if k_max >= 1:
            print(f"  ||Δ_k1||/||H_PP|| = {norm_k1/max(norm_pp, 1e-15):.4f}")
        print(flush=True)

    return {
        'E_conv': E_history[-1],
        'E_vec': E_vec,
        'H_eff_final': H_eff_final,
        'n_iter': len(E_history) - 1,  # exclude initial guess
        'converged': converged,
        'E_history': E_history,
        'Delta_k0': Delta_k0_final,
        'Delta_k1': Delta_k1_final,
        'final_resolvents': A_final,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Post-convergence: per-root evaluation (non-self-consistent)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_excited_states(
    H_PP: np.ndarray,
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
    E0_gs: float,
    delta: float = 0.0,
    k_max: int = 1,
    n_states: int = 5,
    C_ref: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """Evaluate excited-state energies using the converged ground-state E₀.

    Note: Per user request, excited states are currently NOT solved
    self-consistently.  This function does a single-shot evaluation of
    H^eff at the converged ground-state E₀ and returns the lowest n_states
    eigenvalues via root tracking.

    Future extension: per-root self-consistent iteration.

    Args:
        H_PP, H_PQ, H_QQ_blocks, D_by_n: Hamiltonian blocks.
        E0_gs:   Converged ground-state energy (output of solve_self_consistent).
        delta:   Energy shift.
        k_max:   Neumann order.
        n_states: Number of states to track.
        C_ref:   Reference eigenvectors for root tracking.
        verbose: Print diagnostics.

    Returns:
        dict:
          'E_excited':     (n_states,) excited-state energies.
          'E_vecs':        (|P|, n_states) eigenvectors.
          'overlaps':      (n_states,) max overlaps with reference.
          'H_eff':         (|P|, |P|) effective Hamiltonian used.
    """
    p_dim = H_PP.shape[0]
    if p_dim == 0:
        return {
            'E_excited': np.zeros(0), 'E_vecs': np.zeros((0, 0)),
            'overlaps': np.zeros(0), 'H_eff': np.zeros((0, 0)),
        }

    # Build H^eff at the converged E₀
    res = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, D_by_n,
        E0_gs, delta=delta, k_max=k_max, verbose=verbose)

    H_eff = res['H_eff']

    # Track roots
    if C_ref is None:
        _, C_ref = eigh(H_PP)

    E_excited, E_vecs, overlaps = diagonalize_and_track(
        H_eff, C_ref=C_ref, n_states=n_states)

    if verbose:
        print(f"\n  Excited states (single-shot at E₀={E0_gs:.12f}):")
        for k in range(min(n_states, len(E_excited))):
            exc = (E_excited[k] - E_excited[0]) * 1000 if k > 0 else 0.0
            exc_str = f"  ({exc:+.1f} mH exc)" if k > 0 else ""
            print(f"    S{k}: {E_excited[k]:.12f} Ha{exc_str} "
                  f"(overlap={overlaps[k]:.6f})")

    return {
        'E_excited': E_excited,
        'E_vecs': E_vecs,
        'overlaps': overlaps,
        'H_eff': H_eff,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_self_consistent_convergence():
    """Test that the self-consistent solver converges on a toy system."""
    p_dim = 2
    H_PP = np.array([[0.0, 0.1],
                      [0.1, 0.5]])
    H_PP = 0.5 * (H_PP + H_PP.T)

    H_PQ = {
        1: np.array([[0.3, 0.0],
                      [0.0, 0.2]]),
    }

    H_QQ_blocks = {
        (1, 1): np.array([[2.0, 0.1],
                           [0.1, 3.0]]),
    }

    D_by_n = {
        1: np.array([2.0, 3.0]),
    }

    # Solve self-consistently
    result = solve_self_consistent(
        H_PP, H_PQ, H_QQ_blocks, D_by_n,
        k_max=1, tol=1e-10, max_iter=50, verbose=False)

    assert result['converged']
    assert result['n_iter'] >= 1
    assert result['E_conv'] < eigh(H_PP)[0][0]  # Q coupling should lower energy

    # E_history should be monotonically decreasing (or at least not increasing
    # rapidly — for a well-behaved system)
    E_hist = result['E_history']
    assert len(E_hist) >= 2

    print("  ✓ self_consistent_convergence: converged and energy lowered")


def test_self_consistent_toy():
    """Test self-consistent solver on a known small system with explicit
    comparison to direct diagonalization."""
    # Build a 3×3 system: P has 2 states, Q has 1 state
    # Full H:
    #   H_PP = [[0, 0.1], [0.1, 0.5]]
    #   H_PQ = [[0.3], [0.2]]
    #   H_QQ = [[2.0]]
    # Full D = 3 → we can compare with exact eigh

    H_full = np.array([
        [0.0, 0.1, 0.3],
        [0.1, 0.5, 0.2],
        [0.3, 0.2, 2.0],
    ])
    H_full = 0.5 * (H_full + H_full.T)
    E_exact = eigh(H_full)[0][0]

    # Setup Neumann blocks
    H_PP = H_full[:2, :2]
    H_PQ = {1: H_full[:2, 2:3]}  # single Q₁ block
    H_QQ_blocks = {(1, 1): H_full[2:3, 2:3]}
    D_by_n = {1: np.array([H_full[2, 2]])}

    result = solve_self_consistent(
        H_PP, H_PQ, H_QQ_blocks, D_by_n,
        k_max=1, tol=1e-10, max_iter=50, verbose=False)

    E_neumann = result['E_conv']
    dE_mH = (E_neumann - E_exact) * 1000

    # The Neumann k=1 result should be reasonably close to exact
    # (not necessarily identical — it's a perturbation expansion)
    assert abs(dE_mH) < 50  # within 50 mH for this toy system

    print(f"  ✓ self_consistent_toy: E_exact={E_exact:.8f}, "
          f"E_neumann={E_neumann:.8f}, ΔE={dE_mH:+.3f} mH "
          f"(iter={result['n_iter']})")


def test_non_convergent_edge_case():
    """Test edge case: empty P-space."""
    result = solve_self_consistent(
        np.zeros((0, 0)), {}, {}, {},
        k_max=1, tol=1e-8, max_iter=50, verbose=False)
    assert result['n_iter'] == 0
    assert not result['converged']
    print("  ✓ empty P-space: handled gracefully")


def test_excited_states():
    """Test post-convergence excited-state evaluation."""
    p_dim = 3
    H_PP = np.diag([0.0, 0.3, 0.6])
    H_PQ = {1: np.ones((3, 2)) * 0.1}
    H_QQ_blocks = {
        (1, 1): np.diag([2.0, 3.0]),
    }
    D_by_n = {1: np.array([2.0, 3.0])}

    # First get converged ground state
    sc = solve_self_consistent(
        H_PP, H_PQ, H_QQ_blocks, D_by_n,
        k_max=1, tol=1e-8, max_iter=50, verbose=False)

    # Then evaluate excited states
    exc = evaluate_excited_states(
        H_PP, H_PQ, H_QQ_blocks, D_by_n,
        E0_gs=sc['E_conv'], k_max=1, n_states=3, verbose=False)

    assert len(exc['E_excited']) == 3
    assert exc['overlaps'][0] > 0.5  # ground state should have good overlap
    print("  ✓ excited_states: root tracking works")


if __name__ == "__main__":
    test_self_consistent_convergence()
    test_self_consistent_toy()
    test_non_convergent_edge_case()
    test_excited_states()
    print("All self_consistent_solver tests passed.")