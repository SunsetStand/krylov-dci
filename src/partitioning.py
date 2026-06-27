"""
P/Q space partitioning for Krylov downfolding.

Given a FCI Hamiltonian, splits the complete determinant space into
a model space P and its complement Q = I - P.

Partitioning strategies:
  A) CAS-based: P = determinants within a CAS(n,m) active space
  B) Energy-window: P = determinants with |H_ii - E_ref| < Delta_E
  C) Perturbation-based: P = determinants with significant PT2 contribution

References:
  - Löwdin, J. Math. Phys. 3, 969 (1962)
  - Li & Yang, JPCL 13, 10042 (2022)
"""

from typing import List, Tuple, Optional, Set
import numpy as np
from pyscf import gto, scf  # noqa: F401 (used in real workflows, not tests)


# ============================================================================
# Strategy A: CAS-based partitioning
# ============================================================================

def partition_cas(n_orb: int, n_elec: int,
                  n_active_orb: int, n_active_elec: int,
                  n_core_orb: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Partition determinants based on a CAS(n_active_elec, n_active_orb).

    The orbital space is divided into:
      - Core:    doubly occupied, frozen (n_core_orb orbitals)
      - Active:  all possible occupations within n_active_elec electrons
      - Virtual: empty, frozen (n_orb - n_core_orb - n_active_orb orbitals)

    P-space contains all determinants where:
      - Core orbitals are always doubly occupied
      - Active orbitals have any allowed occupation
      - Virtual orbitals are always empty

    Q-space contains all other determinants.

    Args:
        n_orb:         Total number of spatial orbitals.
        n_elec:        Total number of electrons.
        n_active_orb:  Number of active orbitals.
        n_active_elec: Number of active electrons.
        n_core_orb:    Number of frozen core orbitals (doubly occupied).

    Returns:
        (p_indices, q_indices): 0-based indices into the full determinant list.
    """
    from determinants import generate_determinants, count_bits, bit_positions

    # Total alpha/beta electrons
    n_alpha = n_elec // 2  # Assume closed-shell for now
    n_beta = n_elec - n_alpha

    # Core electrons
    n_core_alpha = n_core_orb
    n_core_beta = n_core_orb

    # Active electrons
    n_act_alpha = n_active_elec // 2
    n_act_beta = n_active_elec - n_act_alpha

    # Virtual starts after core + active
    n_virt_orb = n_orb - n_core_orb - n_active_orb

    # Generate all determinants in the full space
    all_dets = generate_determinants(n_orb, n_alpha, n_beta)
    n_det = len(all_dets)

    # Core occupation mask: first n_core_orb orbitals must be doubly occupied
    core_mask_alpha = (1 << n_core_orb) - 1  # bits 0..n_core_orb-1 set

    # Active orbitals: n_core_orb .. n_core_orb + n_active_orb - 1
    act_start = n_core_orb
    act_end = n_core_orb + n_active_orb

    # Virtual orbitals: act_end .. n_orb - 1
    virt_mask = ((1 << n_orb) - 1) ^ ((1 << act_end) - 1)
    # Virtual should be empty: no bits set in virt range

    p_idx = []
    q_idx = []

    for idx, (a_str, b_str) in enumerate(all_dets):
        # Check core: must be fully occupied
        if (a_str & core_mask_alpha) != core_mask_alpha:
            q_idx.append(idx)
            continue
        if (b_str & core_mask_alpha) != core_mask_alpha:
            q_idx.append(idx)
            continue

        # Check virtual: must be empty
        if (a_str & virt_mask) != 0:
            q_idx.append(idx)
            continue
        if (b_str & virt_mask) != 0:
            q_idx.append(idx)
            continue

        # Count active electrons
        act_a = count_bits(a_str >> act_start)
        act_b = count_bits(b_str >> act_start)
        total_act = act_a + act_b

        if total_act == n_active_elec:
            p_idx.append(idx)
        else:
            q_idx.append(idx)

    return np.array(p_idx), np.array(q_idx)


# ============================================================================
# Strategy B: Energy-window partitioning
# ============================================================================

def partition_energy_window(ham, dets: List[Tuple[int, int]],
                            E_ref: float, window: float) -> Tuple[np.ndarray, np.ndarray]:
    """Partition determinants by diagonal energy proximity to a reference.

    P = {|D> : |H_DD - E_ref| < window}

    Args:
        ham:    Hamiltonian object with diagonal_element() method.
        dets:   List of all determinants (alpha_str, beta_str).
        E_ref:  Reference energy (e.g., E_HF).
        window: Energy window width (Hartree).

    Returns:
        (p_indices, q_indices).
    """
    p_idx = []
    q_idx = []

    for idx, (a_str, b_str) in enumerate(dets):
        E_diag = ham.diagonal_element(a_str, b_str)
        if abs(E_diag - E_ref) < window:
            p_idx.append(idx)
        else:
            q_idx.append(idx)

    return np.array(p_idx), np.array(q_idx)


# ============================================================================
# Strategy C: Perturbation-based partitioning
# ============================================================================

def partition_perturbation(ham, dets: List[Tuple[int, int]],
                           ref_idx: int,
                           threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Partition determinants by first-order PT2 contribution from a reference.

    The PT2 weight of determinant |D> relative to reference |D_0> is:
      w(D) = |<D|H|D_0>|^2 / |E_DD - E_00|

    P = {|D> : w(D) > threshold}

    Args:
        ham:       Hamiltonian object.
        dets:      List of all determinants.
        ref_idx:   Index of the reference determinant in dets.
        threshold: PT2 weight threshold.

    Returns:
        (p_indices, q_indices).
    """
    det0 = dets[ref_idx]
    E00 = ham.diagonal_element(det0[0], det0[1])

    p_idx = [ref_idx]  # Reference always in P
    q_idx = []

    for idx, (a_str, b_str) in enumerate(dets):
        if idx == ref_idx:
            continue

        h_0i = ham.matrix_element(det0, (a_str, b_str))
        E_ii = ham.diagonal_element(a_str, b_str)

        denom = abs(E_ii - E00)
        if denom < 1e-12:
            # Near-degenerate: always include in P
            p_idx.append(idx)
            continue

        weight = abs(h_0i) ** 2 / denom
        if weight > threshold:
            p_idx.append(idx)
        else:
            q_idx.append(idx)

    return np.array(p_idx), np.array(q_idx)


# ============================================================================
# Partition utilities
# ============================================================================

def extract_subspace(ham, dets: List[Tuple[int, int]],
                     indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the Hamiltonian blocks for a subspace.

    Args:
        ham:     Hamiltonian object.
        dets:    Full determinant list.
        indices: Indices of the subspace determinants.

    Returns:
        (H_SS, H_SS_diag, subspace_dets) where:
          H_SS:       Full submatrix (|S| × |S|).
          H_SS_diag:  Diagonal of H_SS, shape (|S|,).
          subspace_dets: List of (alpha_str, beta_str) for the subspace.
    """
    from determinants import count_bits  # noqa: F811 (used implicitly)

    n_sub = len(indices)
    H_SS = np.zeros((n_sub, n_sub))
    H_diag = np.zeros(n_sub)

    sub_dets = [dets[i] for i in indices]

    for i_local, i_global in enumerate(indices):
        det_i = dets[i_global]
        H_diag[i_local] = ham.diagonal_element(det_i[0], det_i[1])
        H_SS[i_local, i_local] = H_diag[i_local]

        for j_local in range(i_local + 1, n_sub):
            j_global = indices[j_local]
            det_j = dets[j_global]
            h_ij = ham.matrix_element(det_i, det_j)
            H_SS[i_local, j_local] = h_ij
            H_SS[j_local, i_local] = h_ij

    return H_SS, H_diag, sub_dets


def compute_reference_energy(ham, dets: List[Tuple[int, int]],
                             p_indices: np.ndarray) -> float:
    """Diagonalize H_PP and return the lowest eigenvalue as E^(0).

    Args:
        ham:       Hamiltonian.
        dets:      Full determinant list.
        p_indices: P-space indices.

    Returns:
        E^(0), the lowest eigenvalue of H_PP.
    """
    H_PP, _, _ = extract_subspace(ham, dets, p_indices)
    eigvals = np.linalg.eigvalsh(H_PP)
    return eigvals[0]


# ============================================================================
# Tests
# ============================================================================

def test_cas_partitioning_h2():
    """Test CAS partitioning for H2/STO-3G (2 spatial orbitals).
    Full space: 2 orbitals, 2 electrons -> 4 determinants.
    CAS(2,2): all 4 determinants are in P (active = full space).
    """
    # n_orb=2, CAS(2,2): active = orbitals 0,1 = all orbitals
    p_idx, q_idx = partition_cas(
        n_orb=2, n_elec=2,
        n_active_orb=2, n_active_elec=2, n_core_orb=0
    )
    assert len(p_idx) == 4, f"Expected 4 P-det, got {len(p_idx)}"
    assert len(q_idx) == 0
    print("  ✓ CAS(2,2), 2 orb: P=4, Q=0 (active = full space)")


def test_cas_partitioning_h2o_sto3g():
    """Test CAS partitioning: active < full space.

    n_orb=4, n_elec=4, CAS(4,4): active = all orbitals.
    Then n_orb=4, n_elec=2, CAS(2,2) with n_core_orb=0: active < full.
    """
    # Case 1: active = full space
    p_idx, q_idx = partition_cas(
        n_orb=4, n_elec=4,
        n_active_orb=4, n_active_elec=4, n_core_orb=0
    )
    # C(4,2)*C(4,2) = 6*6 = 36 dets, all in P since active = full space
    assert len(p_idx) == 36, f"Expected 36 P-det, got {len(p_idx)}"
    assert len(q_idx) == 0
    print(f"  ✓ 4 orb, active=full: P={len(p_idx)}, Q={len(q_idx)}")

    # Case 2: active subset with core frozen
    p_idx, q_idx = partition_cas(
        n_orb=4, n_elec=4,
        n_active_orb=2, n_active_elec=2, n_core_orb=1
    )
    # Core: 1 orbital (2 electrons frozen), Active: 2 orbitals with 2 electrons
    # Full: C(4,2)*C(4,2) = 36 dets
    # P: active electrons = 2 in orbital 1,2 → C(2,1)*C(2,1) = 4
    assert len(p_idx) == 4, f"Expected 4 P-det, got {len(p_idx)}"
    assert len(q_idx) == 32
    print(f"  ✓ 4 orb, CAS(2,2)+core: P={len(p_idx)}, Q={len(q_idx)}")


if __name__ == "__main__":
    test_cas_partitioning_h2()
    test_cas_partitioning_h2o_sto3g()
    print("All partitioning tests passed.")
