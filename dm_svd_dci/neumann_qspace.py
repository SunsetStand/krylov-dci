#!/usr/bin/env python3
"""
Q-space generation for Neumann effective Hamiltonian correction.

Given P-space determinants (from Schmidt product basis expansion) and
a set of env (environment) orbitals, generates Q-space determinants via
single (S) and double (D) excitations from P to env, then builds the
H_PQ and H_QQ blocks needed for Neumann k=0 and k=1 corrections.

P/Q partitioning:
  P-space: determinants spanned by the compressed A∪B Schmidt product basis
           (D_emb dimensional, after dmSVD at the current round).
  Q-space: S/D excitations from each P-determinant into env orbitals
           (orbitals within the full CAS that have not yet been added to A∪B).

Selection rules (|Δn| ≤ 2):
  Only Q sectors with electron count difference ≤ 2 from P sectors
  contribute to k=1. This is enforced by the Neumann module's q_pairs.

Integration with Growing CAS DMRG:
  After each round's dmSVD and H_emb diagonalization:
    1. Expand P-space Schmidt basis to raw determinant list p_dets
    2. Generate Q-space via S/D excitations to env orbitals
    3. Compute H_PQ[n] and H_QQ[(m,n)] using full-CAS integrals
    4. Call build_effective_hamiltonian_neumann(H_PP=H_emb, ...)
    5. Diagonalize H_eff → E_corrected

Usage:
    from dm_svd_dci.neumann_qspace import generate_qspace_sd, build_pq_qq
    q_dets, q_labels = generate_qspace_sd(p_dets, env_orbs, n_active, n_elec)
    H_PQ, H_QQ, D_by_n = build_pq_qq(p_dets, q_dets, q_labels, ham, ...)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import itertools
import time


# ═══════════════════════════════════════════════════════════════════════════
# Bit-level determinant manipulation
# ═══════════════════════════════════════════════════════════════════════════

def _bit_count(x: int) -> int:
    """Population count."""
    return x.bit_count()


def _apply_excitation(
    alpha_str: int, beta_str: int,
    occ_orbs: List[int], vir_orbs: List[int],
    spin: str,  # 'a' or 'b'
) -> Optional[Tuple[int, int]]:
    """Apply a single-excitation (occ → vir) to a determinant.
    
    Args:
        alpha_str, beta_str: Bit strings for α and β parts.
        occ_orbs: Orbitals to remove electrons from.
        vir_orbs: Orbitals to add electrons to.
        spin: 'a' for α excitation, 'b' for β.
    
    Returns:
        (new_alpha, new_beta) or None if excitation is invalid.
    """
    if spin == 'a':
        # Check that occ_orb is occupied in alpha
        if not (alpha_str >> occ_orbs[0]) & 1:
            return None
        # Check that vir_orb is empty in alpha
        if (alpha_str >> vir_orbs[0]) & 1:
            return None
        new_a = (alpha_str ^ (1 << occ_orbs[0])) | (1 << vir_orbs[0])
        new_b = beta_str
    else:
        if not (beta_str >> occ_orbs[0]) & 1:
            return None
        if (beta_str >> vir_orbs[0]) & 1:
            return None
        new_a = alpha_str
        new_b = (beta_str ^ (1 << occ_orbs[0])) | (1 << vir_orbs[0])
    return (new_a, new_b)


def _apply_double_excitation(
    alpha_str: int, beta_str: int,
    occ_orbs: List[int], vir_orbs: List[int],
    spin1: str, spin2: str,
) -> Optional[Tuple[int, int]]:
    """Apply a double-excitation (occ1→vir1, occ2→vir2).
    
    spin1/spin2: 'a' or 'b' for each electron.
    """
    # Apply first excitation
    result = _apply_excitation(alpha_str, beta_str,
                                occ_orbs[:1], vir_orbs[:1], spin1)
    if result is None:
        return None
    # Apply second excitation on the result
    return _apply_excitation(result[0], result[1],
                              occ_orbs[1:], vir_orbs[1:], spin2)


# ═══════════════════════════════════════════════════════════════════════════
# Q-space generation via S/D excitations to env
# ═══════════════════════════════════════════════════════════════════════════

def generate_qspace_sd(
    p_dets: List[Tuple[int, int]],
    active_orbs: List[int],
    env_orbs: List[int],
    n_alpha: int,
    n_beta: int,
    max_excit: int = 2,
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:
    """Generate Q-space determinants by S/D excitations to env orbitals.
    
    For each P-space determinant |p⟩, generate all determinants within the
    full CAS (active ∪ env) reachable by:
      - Single excitations: 1 electron from active → env
      - Double excitations: 2 electrons from active → env (or 1→env + 1→env)
    
    These are classified by n_env = number of electrons in env orbitals.
    The Q-space is partitioned as Q_n for each n_env value.
    
    Args:
        p_dets: List of P-space determinants (alpha_str, beta_str).
                Bit positions refer to the full active+env orbital space.
        active_orbs: List of orbital indices belonging to active space.
        env_orbs: List of orbital indices belonging to env space.
        n_alpha, n_beta: Total electron counts.
        max_excit: Maximum excitation level (1 or 2).
        verbose: Print statistics.
    
    Returns:
        (q_dets_unique, q_labels):
          q_dets_unique: List of unique Q-space determinants.
          q_labels: Dict[n_env] → list of indices in q_dets_unique.
    """
    # Convert to sets for fast membership tests
    active_set = set(active_orbs)
    env_set = set(env_orbs)
    
    # Get occupied and virtual orbitals in active space for each P det
    q_set: Set[Tuple[int, int]] = set()
    n_env_counts: Dict[int, List[int]] = defaultdict(list)  # n_env → [det indices]
    
    n_skipped = 0
    
    for p_idx, (a_p, b_p) in enumerate(p_dets):
        # Find occupied orbitals in active space
        occ_a_active = [i for i in active_set if (a_p >> i) & 1]
        occ_b_active = [i for i in active_set if (b_p >> i) & 1]
        
        # Find empty orbitals in env space
        vir_a_env = [i for i in env_set if not ((a_p >> i) & 1)]
        vir_b_env = [i for i in env_set if not ((b_p >> i) & 1)]
        
        # ── Single excitations ──
        if max_excit >= 1:
            # a → a: α electron from active to env α orbital
            for occ in occ_a_active:
                for vir in vir_a_env:
                    new_a = (a_p ^ (1 << occ)) | (1 << vir)
                    new_b = b_p
                    det = (new_a, new_b)
                    if det not in q_set:
                        q_set.add(det)
                        n_env = _bit_count(new_a & sum(1 << i for i in env_set)) + \
                                _bit_count(new_b & sum(1 << i for i in env_set))
            
            # a → b: α electron from active to env β orbital
            for occ in occ_a_active:
                for vir in vir_b_env:
                    new_a = a_p ^ (1 << occ)
                    new_b = b_p | (1 << vir)
                    det = (new_a, new_b)
                    if det not in q_set:
                        q_set.add(det)
            
            # b → a: β electron from active to env α orbital
            for occ in occ_b_active:
                for vir in vir_a_env:
                    new_a = a_p | (1 << vir)
                    new_b = b_p ^ (1 << occ)
                    det = (new_a, new_b)
                    if det not in q_set:
                        q_set.add(det)
            
            # b → b: β electron from active to env β orbital
            for occ in occ_b_active:
                for vir in vir_b_env:
                    new_a = a_p
                    new_b = (b_p ^ (1 << occ)) | (1 << vir)
                    det = (new_a, new_b)
                    if det not in q_set:
                        q_set.add(det)
        
        # ── Double excitations ──
        if max_excit >= 2:
            # aa → aa: two α from active to two α in env
            for occ1, occ2 in itertools.combinations(occ_a_active, 2):
                for vir1, vir2 in itertools.combinations(vir_a_env, 2):
                    new_a = (a_p ^ (1 << occ1) ^ (1 << occ2)) | (1 << vir1) | (1 << vir2)
                    new_b = b_p
                    det = (new_a, new_b)
                    if det not in q_set:
                        q_set.add(det)
            
            # bb → bb: two β from active to two β in env
            for occ1, occ2 in itertools.combinations(occ_b_active, 2):
                for vir1, vir2 in itertools.combinations(vir_b_env, 2):
                    new_a = a_p
                    new_b = (b_p ^ (1 << occ1) ^ (1 << occ2)) | (1 << vir1) | (1 << vir2)
                    det = (new_a, new_b)
                    if det not in q_set:
                        q_set.add(det)
            
            # ab → ab: one α + one β from active to env
            for occ_a in occ_a_active:
                for occ_b in occ_b_active:
                    for vir_a in vir_a_env:
                        for vir_b in vir_b_env:
                            new_a = (a_p ^ (1 << occ_a)) | (1 << vir_a)
                            new_b = (b_p ^ (1 << occ_b)) | (1 << vir_b)
                            det = (new_a, new_b)
                            if det not in q_set:
                                q_set.add(det)
            
            # aa → ab: two α from active, one to α-env + one to β-env
            for occ1, occ2 in itertools.combinations(occ_a_active, 2):
                for vir_a in vir_a_env:
                    for vir_b in vir_b_env:
                        new_a = (a_p ^ (1 << occ1)) | (1 << vir_a)
                        new_b = b_p | (1 << vir_b)
                        det = (new_a, new_b)
                        if det not in q_set:
                            q_set.add(det)
                        
                        new_a2 = (a_p ^ (1 << occ2)) | (1 << vir_a)
                        det2 = (new_a2, new_b)
                        if det2 not in q_set:
                            q_set.add(det2)
            
            # bb → ab: two β from active, one to α-env + one to β-env
            for occ1, occ2 in itertools.combinations(occ_b_active, 2):
                for vir_a in vir_a_env:
                    for vir_b in vir_b_env:
                        new_a = a_p | (1 << vir_a)
                        new_b = (b_p ^ (1 << occ1)) | (1 << vir_b)
                        det = (new_a, new_b)
                        if det not in q_set:
                            q_set.add(det)
                        
                        new_b2 = (b_p ^ (1 << occ2)) | (1 << vir_b)
                        det2 = (new_a, new_b2)
                        if det2 not in q_set:
                            q_set.add(det2)
    
    q_dets = list(q_set)
    
    # Build n_env partition
    env_mask = sum(1 << i for i in env_set)
    n_env_to_dets: Dict[int, List[int]] = defaultdict(list)
    for q_idx, (a_q, b_q) in enumerate(q_dets):
        n_env = _bit_count(a_q & env_mask) + _bit_count(b_q & env_mask)
        n_env_to_dets[n_env].append(q_idx)
    
    if verbose:
        print(f"  Q-space generation: |P|={len(p_dets)}, |Q|={len(q_dets)}")
        for n_env in sorted(n_env_to_dets.keys()):
            print(f"    n_env={n_env}: {len(n_env_to_dets[n_env])} determinants")
    
    return q_dets, dict(n_env_to_dets)


# ═══════════════════════════════════════════════════════════════════════════
# P-det expansion from Schmidt basis
# ═══════════════════════════════════════════════════════════════════════════

def expand_p_dets_from_schmidt(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    n_occ: int,
) -> List[Tuple[int, int]]:
    """Expand P-space Schmidt basis to raw determinant list.
    
    For each Schmidt product basis state |Ã_α^(n)⟩⊗|B̃_β^(n)⟩,
    expand to raw determinants in the A∪B space and collect all
    determinants that have significant coefficient (|c| > 1e-12).
    
    Args:
        schmidt_data: Output of compute_schmidt_decomposition().
        partition: Output of partition_determinants().
        n_occ: Number of A-space orbitals.
    
    Returns:
        List of unique (alpha_str, beta_str) tuples in the A∪B space.
    """
    det_set: Set[Tuple[int, int]] = set()
    
    for n_A, sd in schmidt_data.items():
        r = sd['r']
        if r == 0:
            continue
        U = sd['U']      # (d_A, r)
        V = sd['V']      # (d_B, r)
        blk = partition[n_A]
        a_dets = blk['a_dets']
        b_dets = blk['b_dets']
        
        for alpha in range(r):
            for beta in range(r):
                for i, (aA_alpha, bA_alpha) in enumerate(a_dets):
                    if abs(U[i, alpha]) < 1e-12:
                        continue
                    for j, (aB, bB) in enumerate(b_dets):
                        if abs(V[j, beta]) < 1e-12:
                            continue
                        a_full = aA_alpha | (aB << n_occ)
                        b_full = bA_alpha | (bB << n_occ)
                        det_set.add((a_full, b_full))
    
    return list(det_set)


# ═══════════════════════════════════════════════════════════════════════════
# Build H_PQ and H_QQ blocks
# ═══════════════════════════════════════════════════════════════════════════

def build_pq_qq_blocks(
    p_dets: List[Tuple[int, int]],
    q_dets: List[Tuple[int, int]],
    q_by_n: Dict[int, List[int]],
    ham,  # src.hamiltonian.Hamiltonian instance
    hdiag_full: np.ndarray,
    det_to_full_idx: Dict[Tuple[int, int], int],
    p_indices: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int, int], np.ndarray], Dict[int, np.ndarray]]:
    """Build H_PQ[n] and H_QQ[(m,n)] blocks for Neumann correction.
    
    Uses the Slater-Condon Hamiltonian to compute matrix elements
    between P and Q determinants.
    
    H_PQ[n]: ⟨p_i|H|q_j⟩ for q_j in Q_n sector.
    H_QQ[(m,n)]: ⟨q_i|H|q_j⟩ for q_i ∈ Q_m, q_j ∈ Q_n.
    D_by_n[n]: diag(H_{Q_n Q_n}) = hdiag for Q_n determinants.
    
    Args:
        p_dets: P-space determinant list.
        q_dets: Q-space determinant list.
        q_by_n: Dict[n_env] → indices into q_dets.
        ham: src.hamiltonian.Hamiltonian object with full integrals.
        hdiag_full: H diagonal for ALL determinants (from QSpaceIndex.hdiag).
        det_to_full_idx: Map (alpha_str, beta_str) → flat index in full CAS.
        p_indices: Optional flat indices of P dets in full CAS.
        verbose: Print diagnostics.
    
    Returns:
        (H_PQ, H_QQ_blocks, D_by_n):
          H_PQ: Dict[n] → (|P|, d_n) array.
          H_QQ_blocks: Dict[(m,n)] → (d_m, d_n) array.
          D_by_n: Dict[n] → (d_n,) array of diagonal elements.
    """
    n_p = len(p_dets)
    
    H_PQ: Dict[int, np.ndarray] = {}
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray] = {}
    D_by_n: Dict[int, np.ndarray] = {}
    
    t0 = time.perf_counter()
    
    # Build P-index map for fast lookup
    p_set = set(p_dets)
    p_to_idx = {d: i for i, d in enumerate(p_dets)}
    
    for n_env in sorted(q_by_n.keys()):
        q_indices = q_by_n[n_env]
        d_n = len(q_indices)
        if d_n == 0:
            continue
        
        q_dets_n = [q_dets[idx] for idx in q_indices]
        
        # H_PQ[n]: (n_p, d_n)
        HPQ_n = np.zeros((n_p, d_n))
        
        for j, det_q in enumerate(q_dets_n):
            for i, det_p in enumerate(p_dets):
                hij = ham.matrix_element(det_p, det_q)
                if abs(hij) > 1e-14:
                    HPQ_n[i, j] = hij
        
        H_PQ[n_env] = HPQ_n
        
        # H_QQ[(n, n)]: (d_n, d_n)
        HQQ_nn = np.zeros((d_n, d_n))
        D_n = np.zeros(d_n)
        
        for i, det_q in enumerate(q_dets_n):
            # Diagonal
            if det_q in det_to_full_idx:
                D_n[i] = hdiag_full[det_to_full_idx[det_q]]
            else:
                D_n[i] = ham.matrix_element(det_q, det_q)
            
            # Off-diagonal (only upper triangle, symmetrize later)
            for j in range(i + 1, d_n):
                det_qj = q_dets_n[j]
                hij = ham.matrix_element(det_q, det_qj)
                if abs(hij) > 1e-14:
                    HQQ_nn[i, j] = hij
        
        # Symmetrize
        HQQ_nn = HQQ_nn + HQQ_nn.T + np.diag(D_n)
        H_QQ_blocks[(n_env, n_env)] = HQQ_nn
        D_by_n[n_env] = D_n
        
        if verbose:
            nnz_pq = np.count_nonzero(np.abs(HPQ_n) > 1e-14)
            nnz_qq = np.count_nonzero(np.abs(HQQ_nn - np.diag(D_n)) > 1e-14)
            print(f"    n_env={n_env}: d_n={d_n}, nnz(H_PQ)={nnz_pq}, "
                  f"nnz(H_QQ offdiag)={nnz_qq}")
        
        # Cross-block H_QQ for neighboring n_env values (|m-n| ≤ 2)
        for m_env in sorted(q_by_n.keys()):
            if m_env >= n_env:
                continue
            if abs(m_env - n_env) > 2:
                continue
            
            q_indices_m = q_by_n[m_env]
            d_m = len(q_indices_m)
            q_dets_m = [q_dets[idx] for idx in q_indices_m]
            
            HQQ_mn = np.zeros((d_m, d_n))
            for i, det_qi in enumerate(q_dets_m):
                for j, det_qj in enumerate(q_dets_n):
                    hij = ham.matrix_element(det_qi, det_qj)
                    if abs(hij) > 1e-14:
                        HQQ_mn[i, j] = hij
            
            if np.count_nonzero(np.abs(HQQ_mn) > 1e-14) > 0:
                H_QQ_blocks[(m_env, n_env)] = HQQ_mn
    
    if verbose:
        elapsed = time.perf_counter() - t0
        total_q = sum(len(v) for v in q_by_n.values())
        print(f"  H_PQ/H_QQ built: |P|={n_p}, |Q|={total_q}, "
              f"n_sectors={len(q_by_n)}, {elapsed:.1f}s")
    
    return H_PQ, H_QQ_blocks, D_by_n


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Full Neumann correction pipeline for a round
# ═══════════════════════════════════════════════════════════════════════════

def apply_neumann_correction(
    H_PP: np.ndarray,
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    n_occ: int,
    n_act: int,
    n_active_full: int,
    active_orbs: List[int],
    env_orbs: List[int],
    ham,  # src.hamiltonian.Hamiltonian
    hdiag_full: np.ndarray,
    alpha_strs_full: np.ndarray,
    beta_strs_full: np.ndarray,
    n_alpha: int,
    n_beta: int,
    ecore: float = 0.0,
    k_max: int = 1,
    verbose: bool = True,
) -> Dict:
    """Apply Neumann correction to H_emb for a Growing CAS DMRG round.
    
    Complete pipeline:
      1. Expand P-space from Schmidt basis → p_dets
      2. Generate Q-space via S/D excitations to env
      3. Build H_PQ and H_QQ blocks
      4. Build Neumann effective Hamiltonian (k=0,1)
      5. Diagonalize and return corrected energy
    
    Args:
        H_PP: H_emb matrix (D_emb, D_emb) — P-space Hamiltonian.
        schmidt_data: Output of SVD.
        partition: Output of partition_determinants().
        n_occ: Number of A-space (old) orbitals.
        n_act: Number of active orbitals in current round.
        n_active_full: Total target active orbitals (full CAS).
        active_orbs: Orbital indices in current active space.
        env_orbs: Orbital indices in env (remaining) space.
        ham: src.hamiltonian.Hamiltonian with full-space integrals.
        hdiag_full: H diagonal for full CAS.
        alpha_strs_full, beta_strs_full: Full CAS string arrays.
        n_alpha, n_beta: Total electron counts.
        ecore: Core energy.
        k_max: Neumann order (0 or 1).
        verbose: Print diagnostics.
    
    Returns:
        dict with:
          'E_corrected': Corrected ground state energy.
          'dE_neumann_mH': Neumann correction in mH.
          'E_pp': H_PP diagonalization energy.
          'H_eff': Effective Hamiltonian.
          'Delta_k0': k=0 correction matrix.
          'Delta_k1': k=1 correction matrix.
    """
    from dm_svd_dci.neumann_effective_ham import (
        build_effective_hamiltonian_neumann,
    )
    
    t0 = time.perf_counter()
    
    # Step 1: Expand P-space to raw determinants
    if verbose:
        print(f"\n  ── Neumann correction (k={k_max}) ──")
        print(f"  Step 1: Expanding P-space from Schmidt basis...")
    
    p_dets = expand_p_dets_from_schmidt(schmidt_data, partition, n_occ)
    
    if verbose:
        print(f"    |P| = {len(p_dets)} raw determinants")
    
    if len(p_dets) == 0:
        # No P determinants — Neumann correction not applicable
        evals_pp, _ = np.linalg.eigh(H_PP)
        E_pp = evals_pp[0] + ecore
        return {
            'E_corrected': E_pp,
            'dE_neumann_mH': 0.0,
            'E_pp': E_pp,
            'H_eff': H_PP,
            'Delta_k0': np.zeros_like(H_PP),
            'Delta_k1': np.zeros_like(H_PP),
        }
    
    # Step 2: Generate Q-space
    if verbose:
        print(f"  Step 2: Generating Q-space via S/D excitations to env...")
        print(f"    env orbitals: {env_orbs}")
    
    q_dets, q_by_n = generate_qspace_sd(
        p_dets, list(active_orbs), list(env_orbs),
        n_alpha, n_beta, max_excit=2, verbose=verbose)
    
    if len(q_dets) == 0:
        if verbose:
            print(f"  ⚠ No Q determinants generated (env may be empty). "
                  f"Skipping Neumann correction.")
        evals_pp, _ = np.linalg.eigh(H_PP)
        E_pp = evals_pp[0] + ecore
        return {
            'E_corrected': E_pp,
            'dE_neumann_mH': 0.0,
            'E_pp': E_pp,
            'H_eff': H_PP,
            'Delta_k0': np.zeros_like(H_PP),
            'Delta_k1': np.zeros_like(H_PP),
        }
    
    # Step 3: Build H_PQ and H_QQ
    if verbose:
        print(f"  Step 3: Building H_PQ and H_QQ blocks...")
    
    # Build det_to_full_idx map
    det_to_full_idx = {}
    for ia, a in enumerate(alpha_strs_full):
        for ib, b in enumerate(beta_strs_full):
            det_to_full_idx[(int(a), int(b))] = ia * len(beta_strs_full) + ib
    
    H_PQ, H_QQ_blocks, D_by_n = build_pq_qq_blocks(
        p_dets, q_dets, q_by_n, ham, hdiag_full, det_to_full_idx,
        verbose=verbose)
    
    if len(H_PQ) == 0:
        if verbose:
            print(f"  ⚠ No H_PQ blocks built. Skipping Neumann correction.")
        evals_pp, _ = np.linalg.eigh(H_PP)
        E_pp = evals_pp[0] + ecore
        return {
            'E_corrected': E_pp,
            'dE_neumann_mH': 0.0,
            'E_pp': E_pp,
            'H_eff': H_PP,
            'Delta_k0': np.zeros_like(H_PP),
            'Delta_k1': np.zeros_like(H_PP),
        }
    
    # Step 4: Build Neumann effective Hamiltonian
    # Get E0 from H_PP diagonalization
    if H_PP.shape[0] > 0:
        evals_pp, _ = np.linalg.eigh(H_PP)
        E0 = float(evals_pp[0])
    else:
        E0 = 0.0
    
    if verbose:
        print(f"  Step 4: Building Neumann effective Hamiltonian...")
        print(f"    E0 (from H_PP) = {E0 + ecore:.12f} Ha")
    
    result = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, D_by_n,
        E0=E0, delta=0.0, k_max=k_max, verbose=verbose)
    
    H_eff = result['H_eff']
    
    # Step 5: Diagonalize
    if H_eff.shape[0] > 0:
        evals_eff, _ = np.linalg.eigh(H_eff)
        E_corrected = float(evals_eff[0]) + ecore
    else:
        E_corrected = E0 + ecore
    
    dE_neumann = (E_corrected - (E0 + ecore)) * 1000  # mH
    
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  Neumann correction results:")
        print(f"    E(H^emb)     = {E0 + ecore:.12f} Ha")
        print(f"    E(Neumann)   = {E_corrected:.12f} Ha")
        print(f"    ΔE_neumann   = {dE_neumann:+.3f} mH")
        print(f"    Time:        {elapsed:.1f}s")
    
    return {
        'E_corrected': E_corrected,
        'dE_neumann_mH': float(dE_neumann),
        'E_pp': float(E0 + ecore),
        'H_eff': H_eff,
        'Delta_k0': result.get('Delta_k0', np.zeros_like(H_PP)),
        'Delta_k1': result.get('Delta_k1', np.zeros_like(H_PP)),
        'n_p': len(p_dets),
        'n_q': len(q_dets),
        'n_q_sectors': len(q_by_n),
    }