#!/usr/bin/env python3
"""
Embedded Hamiltonian construction in Schmidt basis.

Given Schmidt decomposition {U^(n), sigma^(n), V^(n)}, construct:

  H^emb = H_A + H_B + H_AB

in the product Schmidt basis {|Ã_α^(n)⟩ ⊗ |B̃_β^(n)⟩}_{n,α,β}.

Strategy:
  - H_A, H_B: Path C — construct H_A^det in F_A(n) and H_B^det in F_B(N-n),
    then project via U^† H_A^det U and V^† H_B^det V.
    This is mathematically equivalent to 1-RDM/2-RDM contraction because
    ⟨Ã_α|H_A|Ã_γ⟩ = Σ_{i,j} U_{iα}* U_{jγ} ⟨a_i|H_A|a_j⟩.

  - H_AB: sigma-vector projection — expand each Schmidt product state
    into the full CAS CI matrix, compute H·v via PySCF contract_2e (C-level),
    then project onto all other Schmidt states.

H^emb dimension: Σ_n r_n² (product of left and right Schmidt ranks per block).

References:
  - DensityMatrix_SVD_Embedding_Proposal.md, Sec. 2.5-2.7
  - src/hamiltonian.py for Slater-Condon matrix elements
  - src_mf/pyscf_backend.py for C-level sigma-vector
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys, os, time


# ═══════════════════════════════════════════════════════════════════════════
# H_A^det / H_B^det construction via Slater-Condon rules
# ═══════════════════════════════════════════════════════════════════════════

def _build_subspace_hamiltonian(
    dets: List[Tuple[int, int]],
    h1: np.ndarray,
    h2: np.ndarray,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """Build H^det in a subspace of determinants.

    Uses standard Slater-Condon rules via src/hamiltonian.Hamiltonian.

    Args:
        dets: List of (alpha_str, beta_str) for the subspace.
        h1: 1e integrals for the subspace (n_orb × n_orb).
        h2: 2e integrals for the subspace (n_orb × n_orb × n_orb × n_orb).
        n_orb: Number of spatial orbitals in subspace.
        n_alpha, n_beta: Electron counts.

    Returns:
        H_det: (N, N) hermitian matrix.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.hamiltonian import Hamiltonian

    ham = Hamiltonian(h1=h1, h2=h2, E_nuc=0.0, E_HF=0.0)
    N = len(dets)
    H = np.zeros((N, N))

    for i in range(N):
        H[i, i] = ham.diagonal_element(int(dets[i][0]), int(dets[i][1]))
        for j in range(i + 1, N):
            hij = ham.matrix_element(
                (int(dets[i][0]), int(dets[i][1])),
                (int(dets[j][0]), int(dets[j][1])),
            )
            H[i, j] = hij
            H[j, i] = hij

    return H


def _extract_subspace_integrals(
    h1_full: np.ndarray,
    h2_full: np.ndarray,
    subspace_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 1e and 2e integrals for a subspace of orbitals.

    Args:
        h1_full: Full 1e integrals (n_act × n_act).
        h2_full: Full 2e integrals (n_act × n_act × n_act × n_act).
        subspace_indices: Array of orbital indices in the subspace.

    Returns:
        (h1_sub, h2_sub) for the subspace orbitals.
    """
    sub = np.asarray(subspace_indices, dtype=int)
    n_sub = len(sub)
    h1_sub = h1_full[sub[:, None], sub[None, :]]
    h2_sub = h2_full[sub[:, None, None, None],
                      sub[None, :, None, None],
                      sub[None, None, :, None],
                      sub[None, None, None, :]]
    return h1_sub, h2_sub


# ═══════════════════════════════════════════════════════════════════════════
# Schmidt basis vector expansion to full CAS space
# ═══════════════════════════════════════════════════════════════════════════

def _expand_schmidt_product_to_ci_matrix(
    alpha_idx: int,
    beta_idx: int,
    blk_schmidt: Dict,
    partition_blk: Dict,
    n_alpha_strs: int,
    n_beta_strs: int,
    n_occ: int,
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    alpha_to_idx: Dict[int, int],
    beta_to_idx: Dict[int, int],
) -> np.ndarray:
    """Expand |Ã_α^(n)⟩ ⊗ |B̃_β^(n)⟩ to a full CAS CI matrix.

    The Schmidt product state in the full CAS basis is:
      Σ_{i,j} U_{iα} V_{jβ}* |a_i^(n)⟩ ⊗ |b_j^(N-n)⟩

    where |a_i^(n)⟩ ⊗ |b_j^(N-n)⟩ is mapped to a full CAS determinant
    by concatenating A and B bit strings (with B bits shifted by n_occ).

    Args:
        alpha_idx: Index α of the A-basis Schmidt vector.
        beta_idx: Index β of the B-basis Schmidt vector.
        blk_schmidt: Schmidt data dict for this n-block.
        partition_blk: Partition block dict for this n-block.
        n_alpha_strs, n_beta_strs: Number of alpha/beta strings in full CAS.
        n_occ: Number of occupied orbitals (Space A).
        alpha_strs, beta_strs: Full CAS alpha/beta string arrays.
        alpha_to_idx, beta_to_idx: String → index maps.

    Returns:
        CI matrix of shape (n_alpha_strs, n_beta_strs).
    """
    U = blk_schmidt['U']   # (dim_A, r)
    V = blk_schmidt['V']   # (dim_B, r)
    a_dets = partition_blk['a_dets']
    b_dets = partition_blk['b_dets']

    ci_mat = np.zeros((n_alpha_strs, n_beta_strs))

    for i, (aA, bA) in enumerate(a_dets):
        u_coef = U[i, alpha_idx]
        if abs(u_coef) < 1e-14:
            continue
        for j, (aB, bB) in enumerate(b_dets):
            v_coef = V[j, beta_idx]
            if abs(v_coef) < 1e-14:
                continue
            # Reconstruct full-CAS determinant
            a_full = aA | (aB << n_occ)
            b_full = bA | (bB << n_occ)
            ia = alpha_to_idx.get(int(a_full))
            ib = beta_to_idx.get(int(b_full))
            if ia is not None and ib is not None:
                ci_mat[ia, ib] += u_coef * v_coef

    return ci_mat


# ═══════════════════════════════════════════════════════════════════════════
# Main H^emb construction
# ═══════════════════════════════════════════════════════════════════════════

def build_h_emb(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    qspace_index,  # QSpaceIndex
    backend,        # KDCIBackend
    h1_full: np.ndarray,
    h2_full: np.ndarray,
    n_occ: int,
    n_act: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Build the embedded Hamiltonian H^emb in the Schmidt product basis.

    Args:
        schmidt_data: Output of compute_schmidt_decomposition().
        partition: Output of partition_determinants().
        qspace_index: QSpaceIndex for the full CAS space.
        backend: KDCIBackend for sigma-vector computation.
        h1_full: Full 1e integrals (n_act × n_act).
        h2_full: Full 2e integrals (n_act × n_act × n_act × n_act).
        n_occ: Number of occupied orbitals.
        n_act: Total active orbitals.
        verbose: Print progress.

    Returns:
        (H_emb, basis_info, decompositions):
          H_emb: (D, D) hermitian matrix where D = Σ_n r_n².
          basis_info: List of dicts describing each basis state:
            [{n, alpha, beta, flat_idx}, ...].
          decompositions: Dict with 'HA', 'HB', 'HAB' sub-blocks (diagnostic).
    """
    # ── Build basis index map ──
    basis_info = []  # list of (n, alpha, beta, flat_idx)
    offset = 0
    block_offsets = {}  # n → starting flat index

    for n_A in sorted(schmidt_data.keys()):
        sd = schmidt_data[n_A]
        r = sd['r']
        block_offsets[n_A] = offset
        for alpha in range(r):
            for beta in range(r):
                basis_info.append({
                    'n': n_A, 'alpha': alpha, 'beta': beta,
                    'flat_idx': offset + alpha * r + beta,
                })
        offset += r * r

    D = offset
    if D == 0:
        return np.zeros((0, 0)), [], {}

    if verbose:
        print(f"  Schmidt product basis dimension: D = {D}")
        for n_A in sorted(schmidt_data.keys()):
            r = schmidt_data[n_A]['r']
            print(f"    n={n_A}: r={r}, r²={r*r}")

    # ── Full CAS string info ──
    alpha_strs = qspace_index.alpha_strs
    beta_strs = qspace_index.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    # ── Orbitals in A and B spaces ──
    A_orb_indices = np.arange(n_occ, dtype=int)
    B_orb_indices = np.arange(n_occ, n_act, dtype=int)
    n_virt = n_act - n_occ

    if verbose:
        print(f"  Orbitals: A = [0..{n_occ - 1}], B = [{n_occ}..{n_act - 1}]")

    # ── Build H_emb via sigma-vector projection ──
    H_emb = np.zeros((D, D))
    H_emb_HA = np.zeros((D, D))  # diagnostic
    H_emb_HB = np.zeros((D, D))

    if verbose:
        t0 = time.perf_counter()
        print(f"  Computing sigma-vectors for {D} basis states...")

    # Pre-compute all CI matrices for the basis states
    ci_mats = []
    for info in basis_info:
        n_A = info['n']
        alpha = info['alpha']
        beta = info['beta']
        blk_schmidt = schmidt_data[n_A]
        blk_partition = partition[n_A]

        ci_mat = _expand_schmidt_product_to_ci_matrix(
            alpha, beta, blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs,
            alpha_to_idx, beta_to_idx,
        )
        ci_mats.append(ci_mat)

    # Compute sigma for each basis state
    sigmas = []
    for k, ci_mat in enumerate(ci_mats):
        sigma_mat = backend.sigma_full(ci_mat)
        sigmas.append(sigma_mat)

        if verbose and (k + 1) % max(1, D // 5) == 0:
            elapsed = time.perf_counter() - t0
            print(f"      sigma {k + 1}/{D} ({elapsed:.0f}s)", flush=True)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"    Sigma-vectors done in {elapsed:.0f}s "
              f"({elapsed / max(D, 1):.2f}s/vector)")

    # ── Project: H_emb[k,l] = v_l^T · sigma_k ──
    if verbose:
        t1 = time.perf_counter()
        print(f"  Projecting {D}×{D} matrix elements...")

    for k in range(D):
        sigma_flat = sigmas[k].reshape(-1)
        for l in range(D):
            v_l = ci_mats[l].reshape(-1)
            H_emb[l, k] = np.dot(v_l, sigma_flat)

    if verbose:
        elapsed = time.perf_counter() - t1
        print(f"    Projection done in {elapsed:.0f}s")

    # ── Also compute H_A and H_B blocks (both approaches for comparison) ──
    if verbose:
        print(f"  Computing H_A and H_B in Schmidt basis (Path C)...")

    # --- Path C: U^† H_A^det U for H_A ---
    h1_A, h2_A = _extract_subspace_integrals(h1_full, h2_full, A_orb_indices)
    # For H_B, need integrals in B space
    h1_B, h2_B = _extract_subspace_integrals(h1_full, h2_full, B_orb_indices)

    for n_A in sorted(schmidt_data.keys()):
        sd = schmidt_data[n_A]
        blk = partition[n_A]
        r = sd['r']
        if r == 0:
            continue

        # Determine electron counts in A from the block's determinants
        a_dets = blk['a_dets']
        if len(a_dets) > 0:
            nA_alpha = max(blk['a_index'].values())  # just to check population
            # Actually count electrons from any det
            aA0, bA0 = a_dets[0]
            nA_alpha = aA0.bit_count()
            nA_beta = bA0.bit_count()
        else:
            continue

        # Build H_A^det
        if len(a_dets) > 0 and n_occ > 0:
            HA_det = _build_subspace_hamiltonian(
                a_dets, h1_A, h2_A, n_occ, nA_alpha, nA_beta)
            # Project: U^† HA_det U
            HA_schmidt = sd['U'].T @ HA_det @ sd['U']
        else:
            HA_schmidt = np.zeros((r, r))

        # Build H_B^det
        b_dets = blk['b_dets']
        if len(b_dets) > 0 and n_virt > 0:
            bB0, bB0b = b_dets[0]
            nB_alpha = bB0.bit_count()
            nB_beta = bB0b.bit_count()
            HB_det = _build_subspace_hamiltonian(
                b_dets, h1_B, h2_B, n_virt, nB_alpha, nB_beta)
            HB_schmidt = sd['V'].T @ HB_det @ sd['V']
        else:
            HB_schmidt = np.zeros((r, r))

        # Map to H_emb blocks
        offset_n = block_offsets[n_A]
        for alpha in range(r):
            for beta in range(r):
                k_alpha_beta = offset_n + alpha * r + beta
                # H_A: δ_{βδ} HA_schmidt[α,γ]
                for gamma in range(r):
                    l_gamma_beta = offset_n + gamma * r + beta
                    H_emb_HA[l_gamma_beta, k_alpha_beta] = HA_schmidt[gamma, alpha]

                # H_B: δ_{αγ} HB_schmidt[β,δ]
                for delta in range(r):
                    l_alpha_delta = offset_n + alpha * r + delta
                    H_emb_HB[l_alpha_delta, k_alpha_beta] = HB_schmidt[delta, beta]

    # Diagnose: how much of H_emb is from H_A + H_B vs H_AB?
    H_emb_diag = H_emb_HA + H_emb_HB
    H_emb_HAB = H_emb - H_emb_diag
    ha_norm = np.linalg.norm(H_emb_HA)
    hb_norm = np.linalg.norm(H_emb_HB)
    hab_norm = np.linalg.norm(H_emb_HAB)
    total_norm = np.linalg.norm(H_emb)

    decompositions = {
        'HA': H_emb_HA,
        'HB': H_emb_HB,
        'HAB': H_emb_HAB,
        'norm_HA': float(ha_norm),
        'norm_HB': float(hb_norm),
        'norm_HAB': float(hab_norm),
        'norm_total': float(total_norm),
    }

    if verbose and D > 0:
        print(f"  H = H_A + H_B + H_AB decomposition:")
        print(f"    ||H_A||  = {ha_norm:.6f}")
        print(f"    ||H_B||  = {hb_norm:.6f}")
        print(f"    ||H_AB|| = {hab_norm:.6f}")
        print(f"    ||H||    = {total_norm:.6f}")

        # Check symmetry
        asym = np.abs(H_emb - H_emb.T).max()
        print(f"    max|H - H^T| = {asym:.2e}")

    return H_emb, basis_info, decompositions


# ═══════════════════════════════════════════════════════════════════════════
# Quick test on H₂O/STO-3G
# ═══════════════════════════════════════════════════════════════════════════

def _setup_h2o_system():
    """Set up H₂O/STO-3G CAS(5,6) with 2 frozen core."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from pyscf import gto, scf, ao2mo, mcscf
    from pyscf.fci import cistring, direct_spin1
    from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    n_core = 2
    n_act = 5
    n_elec = 6
    n_occ = 3  # Space A: first 3 active orbitals (HF-occupied-like)

    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = n_core
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    cas.kernel()
    fcivec = cas.ci
    ci_flat = fcivec.reshape(-1)
    E_fci = cas.e_tot

    # Build QSpaceIndex
    na, nb = n_elec // 2, n_elec - n_elec // 2
    alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act), nb)
    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, (na, nb), h1eff, h2eff)
    backend = KDCIBackend(q_idx)

    # Unpack 2e integrals to 4D
    from src.hamiltonian import _unpack_4fold
    h2_4d = _unpack_4fold(h2eff, n_act)

    return (mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
            n_act, n_elec, n_occ, na, nb, ecore, E_fci)


def test_embedded_hamiltonian_h2o():
    """End-to-end test: partition → SVD → H^emb → diagonalize on H₂O."""
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices,
    )
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition, compute_compression_metrics,
    )

    (mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
     n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

    print(f"H₂O/STO-3G CAS({n_act},{n_elec}) with {n_occ} A-orbitals, "
          f"{n_act - n_occ} B-orbitals")
    print(f"  FCI energy: {E_fci:.10f}")

    # Step 1: Partition
    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
    C_blocks = build_block_matrices(partition, ci_flat)

    # Step 2: Schmidt decomposition
    schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat)
    print(f"  r_total={metrics['r_total']}, "
          f"compression_ratio={metrics['compression_ratio']:.4f}")

    # Step 3: Build H^emb
    H_emb, basis_info, decomps = build_h_emb(
        schmidt, partition, q_idx, backend, h1eff, h2_4d,
        n_occ, n_act, verbose=True)

    D = H_emb.shape[0]
    print(f"  H^emb dimension: {D}")

    if D == 0:
        print("  SKIP: empty H^emb")
        return

    # Step 4: Diagonalize
    evals, evecs = np.linalg.eigh(H_emb)
    E_emb = evals[0] + ecore
    dE = (E_emb - E_fci) * 1000  # mH

    print(f"  Ground state energies:")
    print(f"    E^emb = {E_emb:.10f}")
    print(f"    E^FCI = {E_fci:.10f}")
    print(f"    ΔE    = {dE:+.3f} mH")

    # Step 5: Check hermiticity
    asym = np.abs(H_emb - H_emb.T).max()
    print(f"  Hermiticity: max|H - H^T| = {asym:.2e}")
    assert asym < 1e-10, f"H_emb not symmetric: {asym}"

    print("  ✓ H^emb construction test passed")


if __name__ == "__main__":
    test_embedded_hamiltonian_h2o()
    print("All embedded_hamiltonian tests passed.")