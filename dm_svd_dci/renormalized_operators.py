#!/usr/bin/env python3
"""
Renormalized operators in compressed Schmidt basis (Route A).

Rather than expanding Schmidt basis states to the full determinant space
in every round, we store the CI-coefficient matrices of each S-basis state
and use them to construct composite Hamiltonians via sigma-vector projection.

Mathematical structure:
  t=0: dmSVD → S₀, store CI matrix expansion per S-basis state.
  t≥1: Build H_comp from stored CI matrices + new orbital → diagonalize
       → block-SVD → store new CI matrices for compressed S_{t+1}.

The "renormalization" is performed via:
  CI_new[α] = Σ_{d,o} W[d×4+o, α] · (CI_old[d] extended by occupation o)

This is equivalent to:
  |s_α^{t+1}⟩ = Σ_{d,o} W_{d×4+o, α} · |s_d^t⟩ ⊗ |b_o⟩

where |b_o⟩ is a determinant occupation state of the new orbital.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


# ═══════════════════════════════════════════════════════════════════════════
# Bare 1-orbital operators (4-state Fock basis: |0⟩, |↑⟩, |↓⟩, |↑↓⟩)
# ═══════════════════════════════════════════════════════════════════════════

def get_bare_operators_1orb() -> Dict[str, np.ndarray]:
    """Return bare creation/annihilation/number operators for 1 spatial orbital.

    Returns dict with keys:
      'a_up', 'a_dn':      annihilation (4×4)
      'a_dag_up', 'a_dag_dn': creation (4×4) = a†
    """
    a_dag_up = np.zeros((4, 4))
    a_dag_up[1, 0] = 1.0    # |↑⟩ = a_↑†|0⟩
    a_dag_up[3, 2] = -1.0   # |↑↓⟩ = a_↑†|↓⟩

    a_dag_dn = np.zeros((4, 4))
    a_dag_dn[2, 0] = 1.0    # |↓⟩ = a_↓†|0⟩
    a_dag_dn[3, 1] = 1.0    # |↑↓⟩ = a_↓†|↑⟩

    return {
        'a_up': a_dag_up.T,
        'a_dn': a_dag_dn.T,
        'a_dag_up': a_dag_up,
        'a_dag_dn': a_dag_dn,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Expand S-basis state to CI matrix in current active space
# ═══════════════════════════════════════════════════════════════════════════

def _expand_schmidt_state_to_ci(
    info: Dict,
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    n_occ: int,
    alpha_to_idx: Dict[int, int],
    beta_to_idx: Dict[int, int],
) -> np.ndarray:
    """Expand one Schmidt product state to full CAS CI matrix.

    Reuses _expand_schmidt_product_to_ci_matrix from embedded_hamiltonian.
    """
    from dm_svd_embedding.embedded_hamiltonian import (
        _expand_schmidt_product_to_ci_matrix)
    return _expand_schmidt_product_to_ci_matrix(
        info['alpha'], info['beta'],
        schmidt_data[info['n']], partition[info['n']],
        len(alpha_strs), len(beta_strs), n_occ,
        alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)


class RenormalizedOperators:
    """Manages S-basis states and their CI matrix expansions.

    This is the "Route A" implementation: instead of expanding to the
    full determinant space in every round, we store the CI matrix
    representation of each compressed basis state and use them to
    construct composite Hamiltonians.
    """

    def __init__(self):
        self.D: int = 0               # current S-basis dimension
        self.n_orbitals: int = 0      # total spatial orbitals in active set
        self.H_S: np.ndarray = np.zeros((0, 0))  # Hamiltonian in S-basis
        self.ci_mats: List[np.ndarray] = []      # CI matrix per S-basis state
        self.states_info: List[Dict] = []         # [{n, alpha, beta}, ...]
        # For back-compat: creation/annihilation in S-basis (computed on demand)
        self._a_ann_up: Optional[List[np.ndarray]] = None
        self._a_ann_dn: Optional[List[np.ndarray]] = None

    # ── Bootstrap from dmSVD ──

    @classmethod
    def bootstrap_from_schmidt(
        cls,
        schmidt_data: Dict[int, Dict],
        partition: Dict[int, Dict],
        backend,
        n_occ: int,
        n_act: int,
        h1eff: np.ndarray,
        h2_4d: np.ndarray,
        ecore: float,
        verbose: bool = True,
    ) -> 'RenormalizedOperators':
        """Bootstrap: expand S₀ states to CI matrices, build H_S.

        Args:
            schmidt_data: dmSVD output (compute_schmidt_decomposition).
            partition:    occ_virt_partition (setup_partition).
            backend:      KDCIBackend for sigma_full.
            n_occ:        Number of A-space orbitals.
            n_act:        Total active orbitals.
            h1eff, h2_4d: 1e and 2e integrals (full active space).
            ecore:        Core energy.
            verbose:      Print progress.

        Returns:
            RenormalizedOperators with H_S and CI matrix expansions.
        """
        t0 = time.perf_counter()

        # ── Get determinant string info ──
        alpha_strs = backend.q_idx.alpha_strs
        beta_strs = backend.q_idx.beta_strs
        alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
        beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

        # ── Build global Schmidt basis index mapping ──
        n_sorted = sorted(schmidt_data.keys())
        ci_mats = []
        states_info = []
        D = 0

        for n_A in n_sorted:
            r = schmidt_data[n_A].get('r', 0)
            if r == 0:
                continue
            blk_schmidt = schmidt_data[n_A]
            blk_partition = partition[n_A]

            for alpha in range(r):
                for beta in range(r):
                    ci_mat = _expand_schmidt_state_to_ci(
                        {'n': n_A, 'alpha': alpha, 'beta': beta},
                        schmidt_data, partition,
                        alpha_strs, beta_strs, n_occ,
                        alpha_to_idx, beta_to_idx)
                    ci_mats.append(ci_mat)
                    states_info.append({'n': n_A, 'alpha': alpha, 'beta': beta})
            D += r * r

        if verbose:
            print(f"  [OpBootstrap] Expanded {D} S-basis states to CI mats "
                  f"({alpha_strs.shape[0]}×{beta_strs.shape[0]}), "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

        # ── Build H_S via sigma-vector projection (reuse existing code) ──
        from dm_svd_embedding.embedded_hamiltonian import \
            _expand_schmidt_product_to_ci_matrix
        from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

        t_sigma = time.perf_counter()
        sigmas = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats, n_workers=1, verbose=verbose)

        # H_S[l,k] = ⟨l|H|k⟩ = dot(ci_mats[l], sigmas[k])
        M = ci_mats[0].size
        C_flat = np.empty((M, D))
        S_flat = np.empty((M, D))
        for k in range(D):
            C_flat[:, k] = ci_mats[k].ravel()
            S_flat[:, k] = sigmas[k].ravel()
        H_S = C_flat.T @ S_flat
        H_S = 0.5 * (H_S + H_S.T)
        if D > 0:
            H_S += ecore * np.eye(D)

        del C_flat, S_flat, sigmas

        if verbose:
            print(f"  [OpBootstrap] H_S ({D}×{D}) built, "
                  f"{time.perf_counter() - t_sigma:.1f}s", flush=True)

        # ── Assemble ──
        ops = cls()
        ops.D = D
        ops.n_orbitals = n_act
        ops.H_S = H_S
        ops.ci_mats = ci_mats
        ops.states_info = states_info

        if verbose:
            print(f"  [OpBootstrap] Done: {ops.n_orbitals} orbitals, "
                  f"D={D}, {time.perf_counter() - t0:.1f}s", flush=True)

        return ops

    # ── Build composite Hamiltonian for new orbital ──

    def build_composite_hamiltonian(
        self,
        backend,
        new_orbital_index: int,
        h1eff: np.ndarray,
        h2_4d: np.ndarray,
        ecore: float,
        n_workers: int = 1,
        verbose: bool = True,
    ) -> np.ndarray:
        """Build H_comp in S_t ⊗ F(B_new) using sigma-vector expansion.

        For each S-basis state |s_α⟩ and each B occupation |b_o⟩:
          1. Form combined CI matrix = |s_α⟩ extended with occupation |b_o⟩
          2. Compute sigma-vector
          3. Project: H_comp[(α,o), (β,o')] = ⟨s_α,b_o|H|s_β,b_o'⟩

        Args:
            backend:     KDCIBackend for sigma_full.
            new_orbital_index: Global orbital index of the new orbital (0-based).
            h1eff:       Full active-space 1e integrals (n_act+1, n_act+1).
            h2_4d:       Full active-space 2e integrals as 4d array.
            ecore:       Core energy.
            n_workers:   Parallel sigma workers.
            verbose:     Print progress.

        Returns:
            H_comp: (D×4, D×4) hermitian matrix.
        """
        from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

        t0 = time.perf_counter()
        D = self.D
        dim_comp = D * 4

        # ── Build expanded CI matrices for all D×4 basis states ──
        # We need a new QSpaceIndex for the expanded active space
        from src_mf.pyscf_backend import QSpaceIndex

        # The new active space has self.n_orbitals + 1 orbitals
        n_act_new = self.n_orbitals + 1

        # But the CI matrices of S-basis states are in the OLD active space
        # (self.n_orbitals orbitals). To extend them by new_orbital_index:
        #
        # The new orbital is at position new_orbital_index in the new space.
        # We need to insert it into the existing CI matrix structure.
        #
        # Since the existing CI matrices use orbital indices 0..n_act_old-1,
        # and new_orbital_index is the global index in the new space,
        # we need to map old orbital indices to new ones.
        #
        # Simplest case: new orbital is appended at the end.
        # For now, assume new_orbital_index = n_act_old (appended).

        # Build the extended CI matrices for all (α, o) pairs
        ci_mats_ext = []

        # Get old CI matrix shape
        na_old, nb_old = self.ci_mats[0].shape

        # For each S-basis state and each B occupation, form the extended CI matrix
        # B occupations: 0=|0⟩, 1=|↑⟩, 2=|↓⟩, 3=|↑↓⟩
        # In the extended space:
        #   |0⟩:  old alpha unchanged, old beta unchanged
        #   |↑⟩: add one electron in new orbital (alpha)
        #   |↓⟩: add one electron in new orbital (beta)
        #   |↑↓⟩: add electron in new orbital (both alpha and beta)

        # We need to build the new alpha/beta strings with the extra orbital
        # The new orbital contributes 1 bit (index new_orbital_index)
        new_orb_bit = np.uint64(1) << np.uint64(new_orbital_index)

        for alpha_idx in range(D):
            ci_old = self.ci_mats[alpha_idx]

            # B=0: no change
            ci_mats_ext.append(ci_old.copy())

            # B=↑: add alpha electron in new orbital
            na_new = na_old + 1 if new_orbital_index >= self.n_orbitals else na_old
            # Actually this is complex. Let me simplify by using PySCF string generation.

        # ── Simpler approach: build new QSpaceIndex and expand from there ──
        # This requires the full determinant space of the new active space.
        # Instead, let me build the extended CI matrices by manipulating
        # the bit-packed strings.

        from pyscf.fci import cistring

        # Get old alpha/beta strings
        old_alpha_strs = backend.q_idx.alpha_strs
        old_beta_strs = backend.q_idx.beta_strs

        n_alpha_old = old_alpha_strs.shape[0]
        n_beta_old = old_beta_strs.shape[0]
        n_act_old = self.n_orbitals

        # Generate new strings with the extra orbital
        # The new orbital extends the active space to n_act_old + 1
        # Old orbitals: 0..n_act_old-1, new orbital: new_orbital_index
        # 
        # For now assume new orbital is appended:
        # new orbitals = 0..n_act_old (total n_act_old + 1)

        # Count alpha/beta electrons in old space
        na_old_elec = old_alpha_strs[0].bit_count() if n_alpha_old > 0 else 0
        nb_old_elec = old_beta_strs[0].bit_count() if n_beta_old > 0 else 0

        # Generate all new alpha strings
        new_alpha_strs_all = cistring.gen_strings4orblist(
            range(n_act_old + 1), na_old_elec)       # same count → |0⟩, |↑↓⟩
        new_alpha_strs_plus = cistring.gen_strings4orblist(
            range(n_act_old + 1), na_old_elec + 1)   # one more → |↑⟩
        new_alpha_strs_minus = cistring.gen_strings4orblist(
            range(n_act_old + 1), na_old_elec - 1)   # one less

        new_beta_strs_all = cistring.gen_strings4orblist(
            range(n_act_old + 1), nb_old_elec)
        new_beta_strs_plus = cistring.gen_strings4orblist(
            range(n_act_old + 1), nb_old_elec + 1)
        new_beta_strs_minus = cistring.gen_strings4orblist(
            range(n_act_old + 1), nb_old_elec - 1)

        # For each B occupation, determine which alpha/beta string sets to use
        # B=0: nα_new = nα_old, nβ_new = nβ_old
        # B=↑: nα_new = nα_old+1, nβ_new = nβ_old
        # B=↓: nα_new = nα_old, nβ_new = nβ_old+1
        # B=↑↓: nα_new = nα_old+1, nβ_new = nβ_old+1

        occupation_configs = [
            (new_alpha_strs_all, new_beta_strs_all),     # B=0
            (new_alpha_strs_plus, new_beta_strs_all),     # B=↑
            (new_alpha_strs_all, new_beta_strs_plus),     # B=↓
            (new_alpha_strs_plus, new_beta_strs_plus),    # B=↑↓
        ]

        # Build extended CI matrices by mapping old→new determinant indices
        # For each B occupation, the old CI matrix is copied into the
        # appropriate sub-block of the new CI matrix

        if verbose:
            print(f"  [BuildHcomp] Expanding {D}×4={dim_comp} states "
                  f"to ({n_act_old+1})-orbital space...", flush=True)

        for alpha_idx in range(D):
            ci_old = self.ci_mats[alpha_idx]

            for b_occ, (alpha_strs_new, beta_strs_new) in enumerate(occupation_configs):
                na_new = alpha_strs_new.shape[0]
                nb_new = beta_strs_new.shape[0]
                ci_new = np.zeros((na_new, nb_new))

                if b_occ == 0:
                    # |0⟩: new orbital empty
                    # Old strings map to new strings by inserting 0 at new_orbital_index
                    self._map_ci_expand(ci_old, ci_new,
                                        old_alpha_strs, old_beta_strs,
                                        alpha_strs_new, beta_strs_new,
                                        new_orbital_index, 0, 0)
                elif b_occ == 1:
                    # |↑⟩: new orbital has alpha electron
                    self._map_ci_expand(ci_old, ci_new,
                                        old_alpha_strs, old_beta_strs,
                                        alpha_strs_new, beta_strs_new,
                                        new_orbital_index, 1, 0)
                elif b_occ == 2:
                    # |↓⟩: new orbital has beta electron
                    self._map_ci_expand(ci_old, ci_new,
                                        old_alpha_strs, old_beta_strs,
                                        alpha_strs_new, beta_strs_new,
                                        new_orbital_index, 0, 1)
                else:
                    # |↑↓⟩: new orbital has both alpha and beta electrons
                    self._map_ci_expand(ci_old, ci_new,
                                        old_alpha_strs, old_beta_strs,
                                        alpha_strs_new, beta_strs_new,
                                        new_orbital_index, 1, 1)

                ci_mats_ext.append(ci_new)

        if verbose:
            print(f"  [BuildHcomp] Expanded {len(ci_mats_ext)} CI mats, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

        # ── Compute sigma-vectors ──
        t_sigma = time.perf_counter()
        sigmas = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats_ext,
            n_workers=n_workers, verbose=verbose)

        # ── Project to get H_comp ──
        M_ext = ci_mats_ext[0].size
        C_flat = np.empty((M_ext, dim_comp))
        S_flat = np.empty((M_ext, dim_comp))
        for k in range(dim_comp):
            C_flat[:, k] = ci_mats_ext[k].ravel()
            S_flat[:, k] = sigmas[k].ravel()
        H_comp = C_flat.T @ S_flat
        H_comp = 0.5 * (H_comp + H_comp.T)

        if dim_comp > 0:
            H_comp += ecore * np.eye(dim_comp)

        del C_flat, S_flat, sigmas

        if verbose:
            print(f"  [BuildHcomp] H_comp ({dim_comp}×{dim_comp}) built, "
                  f"{time.perf_counter() - t_sigma:.1f}s", flush=True)

        return H_comp

    def _map_ci_expand(
        self,
        ci_old: np.ndarray,
        ci_new: np.ndarray,
        old_alpha_strs: np.ndarray,
        old_beta_strs: np.ndarray,
        new_alpha_strs: np.ndarray,
        new_beta_strs: np.ndarray,
        new_orb_idx: int,
        occ_alpha: int,
        occ_beta: int,
    ) -> None:
        """Map old CI coefficients to new CI matrix with extended orbital.

        The new orbital at index `new_orb_idx` is set to have `occ_alpha`
        alpha electrons and `occ_beta` beta electrons.

        For each old determinant (α_old, β_old), the new determinant is
        (α_old | (occ_alpha << new_orb_idx), β_old | (occ_beta << new_orb_idx)).
        The sign is (+1) since the new orbital is at the end (or at a fixed position
        relative to the existing electrons, determined by fermionic ordering).
        """
        # Build maps from old→new alpha/beta strings
        alpha_map = {}
        new_bit_a = np.uint64(occ_alpha) << np.uint64(new_orb_idx)
        for i_old, s_old in enumerate(old_alpha_strs):
            s_new_val = int(s_old) | int(new_bit_a)
            # Find this in new strings
            for i_new, s_new in enumerate(new_alpha_strs):
                if int(s_new) == s_new_val:
                    # Fermionic sign: (-1)^{# electrons below new_orb_idx in s_old}
                    n_below = bin(int(s_old) & ((1 << new_orb_idx) - 1)).count('1')
                    alpha_map[i_old] = (i_new, 1.0 if n_below % 2 == 0 else -1.0)
                    break

        beta_map = {}
        new_bit_b = np.uint64(occ_beta) << np.uint64(new_orb_idx)
        for i_old, s_old in enumerate(old_beta_strs):
            s_new_val = int(s_old) | int(new_bit_b)
            for i_new, s_new in enumerate(new_beta_strs):
                if int(s_new) == s_new_val:
                    n_below = bin(int(s_old) & ((1 << new_orb_idx) - 1)).count('1')
                    beta_map[i_old] = (i_new, 1.0 if n_below % 2 == 0 else -1.0)
                    break

        # Copy coefficients with proper signs
        for i_old in range(ci_old.shape[0]):
            if i_old not in alpha_map:
                continue
            i_new, sign_a = alpha_map[i_old]
            for j_old in range(ci_old.shape[1]):
                if j_old not in beta_map:
                    continue
                j_new, sign_b = beta_map[j_old]
                ci_new[i_new, j_new] = ci_old[i_old, j_old] * sign_a * sign_b

    # ── Recompress after block-SVD ──

    def recompress(
        self,
        W: np.ndarray,
        backend,
        h1eff: np.ndarray,
        h2_4d: np.ndarray,
        ecore: float,
        verbose: bool = True,
    ) -> None:
        """Recompress S-basis states after block-SVD.

        The new CI matrices are linear combinations of the old extended CI
        matrices: CI_new[α] = Σ_{d,o} W[d×4+o, α] · CI_ext[d×4+o]

        Then rebuild H_S for the new compressed basis.

        Args:
            W:          (D_old×4, D_new) isometry from block-SVD.
            backend:    KDCIBackend.
            h1eff:      1e integrals for new active space.
            h2_4d:      2e integrals for new active space.
            ecore:      Core energy.
            verbose:    Print progress.
        """
        from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

        t0 = time.perf_counter()
        D_old = self.D
        D_new = W.shape[1]
        n_act_new = self.n_orbitals + 1

        if verbose:
            print(f"  [Recompress] D: {D_old} → {D_new}, "
                  f"n_orb: {self.n_orbitals} → {n_act_new}", flush=True)

        # ── Build extended CI matrices for all (old, B_occ) pairs ──
        # This is the same as in build_composite_hamiltonian
        # But we can reuse the same mapping logic
        # For now, build them fresh

        from pyscf.fci import cistring
        old_alpha_strs = backend.q_idx.alpha_strs
        old_beta_strs = backend.q_idx.beta_strs
        new_orb_idx = self.n_orbitals  # appended at end

        na_old_elec = old_alpha_strs[0].bit_count() if old_alpha_strs.shape[0] > 0 else 0
        nb_old_elec = old_beta_strs[0].bit_count() if old_beta_strs.shape[0] > 0 else 0

        new_alpha_strs_a = cistring.gen_strings4orblist(range(n_act_new), na_old_elec)
        new_alpha_strs_p = cistring.gen_strings4orblist(range(n_act_new), na_old_elec + 1)
        new_beta_strs_a = cistring.gen_strings4orblist(range(n_act_new), nb_old_elec)
        new_beta_strs_p = cistring.gen_strings4orblist(range(n_act_new), nb_old_elec + 1)

        occupation_configs = [
            (new_alpha_strs_a, new_beta_strs_a, 0, 0),    # B=0
            (new_alpha_strs_p, new_beta_strs_a, 1, 0),    # B=↑
            (new_alpha_strs_a, new_beta_strs_p, 0, 1),    # B=↓
            (new_alpha_strs_p, new_beta_strs_p, 1, 1),    # B=↑↓
        ]

        ci_mats_ext = []
        for alpha_idx in range(D_old):
            ci_old = self.ci_mats[alpha_idx]
            for (a_strs, b_strs, occ_a, occ_b) in occupation_configs:
                ci_new = np.zeros((len(a_strs), len(b_strs)))
                self._map_ci_expand(ci_old, ci_new,
                                    old_alpha_strs, old_beta_strs,
                                    a_strs, b_strs,
                                    new_orb_idx, occ_a, occ_b)
                ci_mats_ext.append(ci_new)

        # ── Form new CI matrices as linear combinations ──
        new_ci_mats = []
        for alpha_new in range(D_new):
            ci_combined = np.zeros_like(ci_mats_ext[0])
            for k in range(D_old * 4):
                w_val = W[k, alpha_new]
                if abs(w_val) < 1e-15:
                    continue
                ci_combined += w_val * ci_mats_ext[k]
            new_ci_mats.append(ci_combined)

        del ci_mats_ext

        # ── Rebuild H_S ──
        QSpaceIndex, KDCIBackend = self._get_backend_classes()
        from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

        # Build new backend for the expanded space
        na, nb = self._get_elec_counts(backend)
        new_alpha_strs_all = cistring.gen_strings4orblist(range(n_act_new), na)
        new_beta_strs_all = cistring.gen_strings4orblist(range(n_act_new), nb)
        new_qidx = QSpaceIndex(new_alpha_strs_all, new_beta_strs_all,
                                n_act_new, (na, nb), h1eff, h2_4d)
        new_backend = KDCIBackend(new_qidx)

        sigmas = compute_sigma_vectors_parallel(
            new_backend.sigma_full, new_ci_mats,
            n_workers=1, verbose=verbose)

        M = new_ci_mats[0].size
        C_flat = np.empty((M, D_new))
        S_flat = np.empty((M, D_new))
        for k in range(D_new):
            C_flat[:, k] = new_ci_mats[k].ravel()
            S_flat[:, k] = sigmas[k].ravel()
        H_S = C_flat.T @ S_flat
        H_S = 0.5 * (H_S + H_S.T)
        if D_new > 0:
            H_S += ecore * np.eye(D_new)

        del C_flat, S_flat, sigmas

        # ── Update state ──
        self.D = D_new
        self.n_orbitals = n_act_new
        self.H_S = H_S
        self.ci_mats = new_ci_mats
        # States info is no longer needed (basis is compressed)

        if verbose:
            print(f"  [Recompress] Done: D={D_new}, H_S built, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    def _get_elec_counts(self, backend):
        """Extract (na, nb) from backend."""
        na = backend.q_idx.nelec[0]
        nb = backend.q_idx.nelec[1]
        return na, nb

    def _get_backend_classes(self):
        """Lazy import."""
        from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
        return QSpaceIndex, KDCIBackend


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_bare_operators():
    """Test bare 1-orbital operators."""
    ops = get_bare_operators_1orb()
    for k in ['a_up', 'a_dn', 'a_dag_up', 'a_dag_dn']:
        assert ops[k].shape == (4, 4)

    # a† = a^T
    assert np.allclose(ops['a_dag_up'], ops['a_up'].T)
    assert np.allclose(ops['a_dag_dn'], ops['a_dn'].T)

    # Anticommutation: {a_↑, a_↑†} = I (holds on all 4 states)
    anti = ops['a_up'] @ ops['a_dag_up'] + ops['a_dag_up'] @ ops['a_up']
    assert np.allclose(anti, np.eye(4), atol=1e-12)

    print("  ✓ bare_operators: correct")


def test_renorm_init():
    """Test basic RenormalizedOperators init."""
    ops = RenormalizedOperators()
    assert ops.D == 0
    assert ops.n_orbitals == 0
    print("  ✓ RenormalizedOperators: init correct")


if __name__ == "__main__":
    test_bare_operators()
    test_renorm_init()
    print("All renormalized_operators tests passed.")
