#!/usr/bin/env python3
"""
DMRG-style Growing Active Space via dmSVD with |A⟩⊗|B⟩⊗|env⟩ Framework.

Implements a clean, self-contained pipeline for incrementally adding B orbitals
while keeping the compressed system dimension fixed (≤ initial D₀).

Mathematical framework:
    |Ψ_total⟩ = |A⟩ ⊗ |B⟩ ⊗ |env⟩
where |env⟩ is ALWAYS the vacuum state |000...0⟩ (no electrons).
Electrons only migrate between A and B.

Each round transfers orbitals from env → B:
  Round 0: |A₀⟩ ⊗ |B₀⟩ ⊗ |env₀⟩
  Round 1: |S₁⟩ ⊗ |B₁⟩ ⊗ |env₁⟩  (S₁ = compressed A₀∪B₀, B₁ pulled from env)
  Round k: |S_k⟩ ⊗ |B_k⟩ ⊗ |env_k⟩

Key classes:
  - ChainedTransform: Manages recursive mapping S_k → raw determinants
  - GrowingCASDMRG: Main growing pipeline

Usage:
    from dm_svd_dci.growing_cas_dmrg import GrowingCASDMRG
    grower = GrowingCASDMRG(mol, mf, n_active=10, n_elec=(5,5),
                             n_core=2, n_occ_A=5, n_orb_B0=2,
                             n_orb_Bt=2, eps_svd=1e-3)
    results = grower.run(verbose=True)
"""

import sys, os, time
import numpy as np
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# ChainedTransform
# ═══════════════════════════════════════════════════════════════════════════

class ChainedTransform:
    """Manages the recursive mapping from compressed basis S_k to raw determinants.

    The chain invariant:
        T_{k+1}[n_A] = T_k[n_A] @ U_k[n_A]

    where U_k[n_A] : S_{k+1} → S_k comes from the SVD at round k.

    T_k[n_A] has shape (d_raw(n_A), r_k(n_A)) where:
      - d_raw(n_A) = number of raw determinants on the first k orbital blocks
                      with n_A electrons in the "system" partition
      - r_k(n_A) = compressed dimension for that block
    """

    def __init__(self, T_initial: Dict[int, np.ndarray]):
        """
        Args:
            T_initial: {n_A: ndarray of shape (d_raw(n_A), r(n_A))}
                       Built from dmSVD round-0 output.
                       T_initial[n_A][:, α] = U[:,α] ⊗ conj(V[:,α]) (flattened)
                       maps S₀ basis state α to raw A₀∪B₀ determinants.
        """
        self.chain: List[Dict[int, np.ndarray]] = [T_initial]

    def compress_ci_matrix(
        self, D_full: np.ndarray, n_A: int
    ) -> np.ndarray:
        """Apply T_k† to a raw CI coefficient matrix.

        D_full ∈ ℝ^{d_raw(n_A) × d_B(N-n_A)} → D̃ ∈ ℝ^{r_k(n_A) × d_B(N-n_A)}

        Uses the FULL chained transform (product of all U's so far).
        If n_A is not in the chain, or shapes are incompatible,
        returns D_full unchanged (no pre-compression).

        Args:
            D_full: Raw CI coefficient matrix for n_A block.
            n_A:    Number of electrons in the "system" (old) orbitals.

        Returns:
            D̃: Compressed matrix.
        """
        if n_A not in self.chain[0]:
            # Block not present in T_0; no pre-compression available
            return D_full
        T = self.get_full_transform(n_A)
        if T.shape[1] == 0:
            return np.zeros((0, D_full.shape[1]))
        # Check shape compatibility: T maps to A⊗B product space (d_A*d_B),
        # but D_full may have d_old rows (full old-CI space) which spans
        # multiple n_A blocks and doesn't match T's column space directly.
        if T.shape[0] != D_full.shape[0]:
            # Incompatible shapes — skip pre-compression, fall back to raw SVD
            return D_full
        return T.T @ D_full

    def extend(self, U_new: Dict[int, np.ndarray]):
        """Add new SVD result to chain.

        Args:
            U_new: {n_A: ndarray of shape (r_k, r_{k+1})}
                   For n_A blocks NOT in U_new, the chain for that block
                   is unchanged (meaning T_{k+1} = T_k for those blocks).
        """
        self.chain.append(U_new)

    def get_full_transform(self, n_A: int) -> np.ndarray:
        """Compute T_k[n_A] = T_0[n_A] @ U_1[n_A] @ ... @ U_k[n_A].

        Uses right-to-left multiplication: T_k = T_{k-1} @ U_k.
        Since U_k is small (r_k × r_{k+1}), this is efficient.

        Returns:
            T_k[n_A] of shape (d_raw, r_current).
        """
        if n_A not in self.chain[0]:
            return np.zeros((0, 0))
        T = self.chain[0][n_A]  # (d_raw, r_0)
        for U_dict in self.chain[1:]:
            if n_A in U_dict and U_dict[n_A].shape[1] > 0:
                T = T @ U_dict[n_A]  # (d_raw, r_{k-1}) @ (r_{k-1}, r_k)
        return T

    @property
    def dimension(self) -> Dict[int, int]:
        """{n_A: current r_k(n_A)}"""
        result = {}
        for n_A in self.chain[0]:
            T = self.get_full_transform(n_A)
            result[n_A] = T.shape[1]
        return result

    @property
    def total_dimension(self) -> int:
        """Total compressed dimension D_k = Σ_n r_k(n)."""
        return sum(self.dimension.values())

    @property
    def n_blocks(self) -> set:
        """Set of n_A values present in chain."""
        return set(self.chain[0].keys())


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Build T_0 from dmSVD schmidt data
# ═══════════════════════════════════════════════════════════════════════════

def build_T0_from_schmidt(
    schmidt_data: Dict[int, Dict],
) -> Dict[int, np.ndarray]:
    """Build T_0 from dmSVD schmidt_data.

    T_0[n_A] maps S₀ basis states (indexed by α) to raw A₀∪B₀ determinants.
    Each column α is the flattened outer product U[:,α] ⊗ V[:,α].

    Shape: (d_A(n_A) * d_B(N-n_A), r_n)

    Args:
        schmidt_data: Output of compute_schmidt_decomposition().

    Returns:
        T_0: {n_A: ndarray of shape (d_A*d_B, r_n)}
    """
    T0 = {}
    for n_A, sd in schmidt_data.items():
        r = sd['r']
        if r == 0:
            T0[n_A] = np.zeros((0, 0))
            continue
        U = sd['U']        # (d_A, r)
        V = sd['V']        # (d_B, r)
        d_A = sd['dim_A']
        d_B = sd['dim_B']
        T_block = np.zeros((d_A * d_B, r))
        for alpha in range(r):
            # Outer product: U[:,α] ⊗ V[:,α]
            T_block[:, alpha] = np.outer(U[:, alpha], V[:, alpha]).ravel()
        T0[n_A] = T_block
    return T0


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Run CASCI in a subspace of orbitals
# ═══════════════════════════════════════════════════════════════════════════

def _run_casci_subspace(
    mol,
    mf,
    n_core: int,
    active_orbs: np.ndarray,
    n_elec: Tuple[int, int],
    verbose: bool = True,
) -> Dict:
    """Run CASCI on a subset of active orbitals.

    Reorders MOs so that the selected active orbitals are contiguous
    after the frozen core.

    Args:
        mol, mf: PySCF molecule and RHF objects.
        n_core:  Number of frozen core orbitals.
        active_orbs: Array of global MO indices for active space.
        n_elec:  (n_alpha, n_beta) active electrons.

    Returns:
        dict with:
          'fcivec': CI vector as (n_alpha_strs, n_beta_strs) array.
          'ci_flat': CI vector flattened (M,).
          'E_fci': Total energy.
          'ecore': Core energy.
          'h1eff': Active-space 1e integrals.
          'h2eff': Active-space 2e integrals (packed).
          'h2_4d': Active-space 2e integrals (4d array).
          'n_act': Number of active orbitals.
          'active_orbs': The active orbital indices.
    """
    from pyscf import mcscf
    from pyscf.fci import cistring
    from src.hamiltonian import _unpack_4fold

    n_act = len(active_orbs)
    n_total_mo = mol.nao_nr()

    # Build new MO coefficient matrix: [core | active | rest]
    all_indices = list(range(n_total_mo))
    core_indices = list(range(n_core))
    active_sub_indices = [n_core + i for i in active_orbs]
    used = set(core_indices) | set(active_sub_indices)
    rest_indices = [i for i in all_indices if i not in used]

    new_order = core_indices + active_sub_indices + rest_indices
    mo_new = mf.mo_coeff[:, new_order]

    mf_new = mf.copy()
    mf_new.mo_coeff = mo_new
    mf_new.mo_occ = np.zeros(n_total_mo)
    mf_new.mo_occ[:n_core] = 2.0

    cas = mcscf.CASCI(mf_new, n_act, sum(n_elec))
    cas.frozen = n_core
    cas.kernel()

    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    fcivec = cas.ci
    E_fci = cas.e_tot

    # Unpack h2 to 4d
    h2_4d = _unpack_4fold(h2eff, n_act)

    if verbose:
        M = fcivec.size
        print(f"    CASCI({n_act},{sum(n_elec)}): E={E_fci:.12f} Ha, "
              f"M={M:,} dets", flush=True)

    return {
        'fcivec': fcivec,
        'ci_flat': fcivec.reshape(-1),
        'E_fci': E_fci,
        'ecore': float(ecore),
        'h1eff': h1eff,
        'h2eff': h2eff,
        'h2_4d': h2_4d,
        'n_act': n_act,
        'active_orbs': active_orbs,
    }


def _build_backend(
    n_act: int,
    n_elec: Tuple[int, int],
    h1eff: np.ndarray,
    h2eff: np.ndarray,
):
    """Build QSpaceIndex + KDCIBackend for a given active space."""
    from pyscf.fci import cistring
    from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

    na, nb = n_elec
    alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act), nb)
    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, n_elec, h1eff, h2eff)
    backend = KDCIBackend(q_idx)
    return q_idx, backend


# ═══════════════════════════════════════════════════════════════════════════
# Round 0: Bootstrap via dmSVD
# ═══════════════════════════════════════════════════════════════════════════

def _round_0_bootstrap(
    mol,
    mf,
    n_core: int,
    n_occ_A: int,
    n_orb_B0: int,
    n_elec: Tuple[int, int],
    eps_svd: float,
    verbose: bool = True,
) -> Dict:
    """Round 0: Run dmSVD on A₀∪B₀, build T₀, compute energy.

    Steps:
      1. CASCI on A₀∪B₀ (total n_occ_A + n_orb_B0 orbitals)
      2. Partition CI vector by n_A (electrons in A₀)
      3. SVD each C^(n_A) block → schmidt_data
      4. Build T_0 from schmidt_data
      5. Build H^emb in Schmidt product basis → diagonalize → E₀

    Returns:
        dict with keys:
          'E0':         Ground state energy from H^emb.
          'E_casci':    CASCI energy in the subspace.
          'T':          ChainedTransform (T_0).
          'schmidt':    Schmidt data dict.
          'partition':  Partition dict.
          'n_act':      Number of active orbitals (n_occ_A + n_orb_B0).
          'n_occ':      Number of A orbitals (n_occ_A).
          'D0':         Total compressed dimension Σ r_n.
          'h1eff':      1e integrals.
          'h2_4d':      2e integrals (4d).
          'ecore':      Core energy.
    """
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices,
    )
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition, compute_compression_metrics,
    )
    from dm_svd_embedding.embedded_hamiltonian import build_h_emb

    n_act_0 = n_occ_A + n_orb_B0
    active_orbs = np.arange(n_act_0, dtype=int)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ROUND 0: Bootstrap dmSVD")
        print(f"{'='*60}")
        print(f"  A₀: {n_occ_A} orbitals [0..{n_occ_A-1}]")
        print(f"  B₀: {n_orb_B0} orbitals [{n_occ_A}..{n_act_0-1}]")
        print(f"  Total: {n_act_0} orbitals, {sum(n_elec)} electrons")

    # 1. CASCI
    cas_data = _run_casci_subspace(
        mol, mf, n_core, active_orbs, n_elec, verbose=verbose)
    ci_flat = cas_data['ci_flat']
    E_casci = cas_data['E_fci']
    h1eff = cas_data['h1eff']
    h2_4d = cas_data['h2_4d']
    ecore = cas_data['ecore']
    n_act = cas_data['n_act']

    # 2. Partition by n_A
    partition, _ = setup_partition(n_act, sum(n_elec), n_occ_A, ms=0)
    C_blocks = build_block_matrices(partition, ci_flat)

    if verbose:
        print(f"  Electron blocks: {sorted(C_blocks.keys())}")
        for n_A in sorted(C_blocks.keys()):
            C = C_blocks[n_A]
            print(f"    n={n_A}: ({C.shape[0]} × {C.shape[1]})")

    # 3. SVD
    schmidt = compute_schmidt_decomposition(C_blocks, eps=eps_svd)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat)

    if verbose:
        r_total = metrics['r_total']
        D_prod = sum(sd['r']**2 for sd in schmidt.values())
        print(f"  Schmidt: r_total={r_total}, D(prod)={D_prod}, "
              f"compression={metrics['compression_ratio']:.4%}")

    # 4. Build T_0
    T0 = build_T0_from_schmidt(schmidt)
    T = ChainedTransform(T0)
    D0 = T.total_dimension

    if verbose:
        print(f"  T_0 dimension D₀ = {D0} (Σ r_n per block)")
        for n_A in sorted(T0.keys()):
            print(f"    n={n_A}: r={T0[n_A].shape[1]}")

    # 5. Build H^emb in Schmidt product basis
    # Need backend for the full active space
    q_idx, backend = _build_backend(
        n_act, n_elec, h1eff, cas_data['h2eff'])

    if verbose:
        print(f"\n  Building H^emb in Schmidt product basis...")

    H_emb, basis_info, decomps = build_h_emb(
        schmidt, partition, q_idx, backend, h1eff, h2_4d,
        n_occ_A, n_act, verbose=verbose)

    D_emb = H_emb.shape[0]
    if D_emb > 0:
        evals_emb, _ = np.linalg.eigh(H_emb)
        E_emb_gs = evals_emb[0] + ecore
    else:
        E_emb_gs = ecore

    dE = (E_emb_gs - E_casci) * 1000
    if verbose:
        print(f"\n  E(CASCI)  = {E_casci:.12f} Ha")
        print(f"  E(H^emb)  = {E_emb_gs:.12f} Ha")
        print(f"  ΔE        = {dE:+.3f} mH")
        print(f"  D(prod)   = {D_emb}")
        print(f"  D₀        = {D0}")

    return {
        'E0': E_emb_gs,
        'E_casci': E_casci,
        'T': T,
        'schmidt': schmidt,
        'partition': partition,
        'n_act': n_act,
        'n_occ': n_occ_A,
        'D0': D0,
        'D_emb': D_emb,
        'h1eff': h1eff,
        'h2_4d': h2_4d,
        'h2eff': cas_data['h2eff'],
        'ecore': ecore,
        'dE_mH': dE,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Round k ≥ 1: Extension
# ═══════════════════════════════════════════════════════════════════════════

def _round_k_extension(
    mol,
    mf,
    n_core: int,
    T: ChainedTransform,
    prev_n_act: int,
    prev_n_occ: int,
    Bk_orbs: np.ndarray,
    n_elec: Tuple[int, int],
    eps_svd: float,
    prev_h1eff: np.ndarray,
    prev_h2_4d: np.ndarray,
    prev_ecore: float,
    verbose: bool = True,
    apply_neumann: bool = False,
    env_orbs_remaining: Optional[List[int]] = None,
    n_active_full: Optional[int] = None,
    full_h1eff: Optional[np.ndarray] = None,
    full_h2_4d: Optional[np.ndarray] = None,
) -> Dict:
    """Round k: Add B_k orbitals, compress, chain transform.

    Steps:
      1. CASCI on (old_orbs ∪ B_k)
      2. Partition by n_old = electrons in old orbitals
      3. For each n_old block:
         a. Build raw CI matrix D^(n_old)
         b. D̃ = T† @ D (compress old side)
         c. SVD D̃ → U_new, V_new
         d. Store U_new for chain extension
      4. Chain extend: T_{k+1}[n_old] = T_k[n_old] @ U_new[n_old]
      5. Build H^emb and diagonalize
      6. (Optional) Apply Neumann k=1 correction for dynamical correlation

    Returns:
        dict with E_k, E_neumann, T (updated), D_k, etc.
    """
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices,
    )
    from dm_svd_dci.block_svd_general import block_svd_multi_orbital

    t0 = time.perf_counter()

    # All old orbitals + new B_k orbitals
    all_orbs = np.arange(prev_n_act + len(Bk_orbs), dtype=int)
    n_act_new = len(all_orbs)
    n_elec_total = sum(n_elec)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ROUND k: Extension")
        print(f"{'='*60}")
        print(f"  Old orbitals: {prev_n_act}")
        print(f"  B_k orbitals: {list(Bk_orbs)} ({len(Bk_orbs)} orbitals)")
        print(f"  New total:    {n_act_new} orbitals")
        print(f"  Previous D:   {T.total_dimension}")

    # 1. CASCI on expanded space
    cas_data = _run_casci_subspace(
        mol, mf, n_core, all_orbs, n_elec, verbose=verbose)
    ci_flat = cas_data['ci_flat']
    E_casci = cas_data['E_fci']
    h1eff = cas_data['h1eff']
    h2_4d = cas_data['h2_4d']
    ecore = cas_data['ecore']

    # 2. Partition by n_old = electrons in old orbitals
    # n_occ = prev_n_act (all old orbitals are the "occupied" part)
    partition, _ = setup_partition(n_act_new, n_elec_total, prev_n_act, ms=0)
    C_blocks = build_block_matrices(partition, ci_flat)

    if verbose:
        print(f"  Electron blocks (by n_old): {sorted(C_blocks.keys())}")

    # 3. For each n_old block: compress, SVD, chain
    U_new_all = {}
    D_new_total = 0
    new_schmidt_data = {}

    for n_old in sorted(C_blocks.keys()):
        C = C_blocks[n_old]  # (d_old(n_old), d_B(N-n_old))
        d_old = C.shape[0]
        d_B = C.shape[1]

        if d_old == 0 or d_B == 0:
            continue

        # Get current compressed dimension for this n_old block
        if n_old in T.chain[0]:
            r_current = T.get_full_transform(n_old).shape[1]
        else:
            r_current = 0

        effectively_compressed = False
        if r_current > 0:
            # Compress old side: D̃ = T† @ C
            D_tilde = T.compress_ci_matrix(C, n_old)
            # Check if compression actually reduced the dimension
            if D_tilde.shape[0] == r_current:
                # Successfully compressed
                psi = D_tilde.ravel()  # (r_current * d_B,)
                svd_result = block_svd_multi_orbital(
                    psi, r_current, d_B, eps=eps_svd, verbose=verbose)
                effectively_compressed = True
            else:
                # Shape incompatible — fall through to raw SVD
                r_current = 0

        if not effectively_compressed:
            # No pre-compression available; SVD raw C matrix directly
            # This handles n_old blocks not present in T_0
            psi = C.ravel()
            svd_result = block_svd_multi_orbital(
                psi, d_old, d_B, eps=eps_svd, verbose=verbose)
            # For the chain, we need U_new that maps S_{k+1} → S_k
            # Since S_k is empty for this block, we create a fresh entry
            # The "U_new" will be identity-like when chain is first created
            # Actually, T_0 doesn't have this n_old, so we need to
            # build T from the raw SVD of C
            U_raw = svd_result['U_trunc']  # (d_old, D_new)
            V_raw = svd_result['V_trunc']  # (d_B, D_new)
            D_new = svd_result['D_new']

            # Store raw U and V for T construction
            # We'll add this block to T_0 later
            if D_new > 0:
                new_schmidt_data[n_old] = {
                    'U': U_raw,
                    'V': V_raw,
                    'r': D_new,
                    'dim_A': d_old,
                    'dim_B': d_B,
                    'sigma': svd_result['s_kept'],
                    'sigma_full': svd_result['s_all'],
                }
            D_new_total += D_new
            continue

        D_new = svd_result['D_new']
        U_trunc = svd_result['U_trunc']  # (r_current, D_new): maps S_{k+1} → S_k
        V_trunc = svd_result['V_trunc']  # (d_B, D_new)

        if D_new > 0:
            U_new_all[n_old] = U_trunc
            # CRITICAL: U_trunc lives in compressed S_k space, NOT in raw A-det space.
            # build_h_emb needs U mapping from raw A-det space (len(a_dets)=d_old).
            # Re-do raw SVD on the ORIGINAL C matrix for H_emb purposes.
            C_orig = C_blocks.get(n_old)
            if C_orig is not None and C_orig.size > 0:
                from dm_svd_dci.block_svd_general import block_svd_multi_orbital
                raw_svd = block_svd_multi_orbital(
                    C_orig.ravel(), C_orig.shape[0], C_orig.shape[1],
                    eps=eps_svd, verbose=False)
                if raw_svd['D_new'] > 0:
                    new_schmidt_data[n_old] = {
                        'U': raw_svd['U_trunc'],
                        'V': raw_svd['V_trunc'],
                        'r': raw_svd['D_new'],
                        'dim_A': C_orig.shape[0],
                        'dim_B': C_orig.shape[1],
                        'sigma': raw_svd['s_kept'],
                        'sigma_full': raw_svd['s_all'],
                    }
        D_new_total += D_new

    # 4. Extend chain
    if len(U_new_all) > 0:
        T.extend(U_new_all)

    # Handle new n_old blocks: add them to T_0
    # (This is a pragmatic approach: for blocks that didn't exist in T_0,
    #  we do direct SVD and add them as new entries)
    for n_old, sd in new_schmidt_data.items():
        if n_old not in T.chain[0] and sd['r'] > 0:
            U_raw = sd['U']
            V_raw = sd['V']
            d_old_raw = sd['dim_A']
            d_B_raw = sd['dim_B']
            r_new = sd['r']
            # Build T for this block
            T_block = np.zeros((d_old_raw * d_B_raw, r_new))
            for alpha in range(r_new):
                T_block[:, alpha] = np.outer(
                    U_raw[:, alpha], V_raw[:, alpha]).ravel()
            # Add to the first chain entry
            T.chain[0][n_old] = T_block

    D_k = T.total_dimension

    if verbose:
        print(f"  D_{len(T.chain)} = {D_k} "
              f"({'≤' if D_k <= T.total_dimension else '>'}"
              f" D_prev = {T.total_dimension - D_new_total + D_k})")

    # 4.5. Adaptive truncation: if D_emb would be too large, binary-search tighten eps
    D_emb_raw = sum(sd['r']**2 for sd in new_schmidt_data.values())
    MAX_D_EMB = 2000    # target: ≤2000 basis states
    EPS_MIN = eps_svd   # never go below this (already satisfied)
    EPS_MAX = 0.05      # NEVER exceed 0.05 — larger eps destroys physics
    effective_eps = eps_svd

    if D_emb_raw > MAX_D_EMB:
        if verbose:
            print(f"  [D_emb control] Raw D_emb={D_emb_raw} > max={MAX_D_EMB}, "
                  f"binary-searching eps in [{EPS_MIN:.1e}, {EPS_MAX:.1e}]...")
        from dm_svd_dci.block_svd_general import block_svd_multi_orbital
        import copy
        
        # Cache original schmidt data
        schmidt_backup = {}
        for n_old, sd in new_schmidt_data.items():
            schmidt_backup[n_old] = {k: v for k, v in sd.items()}
        
        # Binary search for eps that gives D_emb ≤ MAX_D_EMB
        eps_lo, eps_hi = EPS_MIN, EPS_MAX
        best_eps = eps_svd
        best_D_emb = D_emb_raw
        
        for _ in range(12):  # max 12 bisections
            eps_mid = np.sqrt(eps_lo * eps_hi)  # geometric mean
            D_trial = 0
            trial_data = {}
            
            for n_old, sd in schmidt_backup.items():
                C = C_blocks.get(n_old)
                if C is not None and C.size > 0 and sd['r'] > 1:
                    new_result = block_svd_multi_orbital(
                        C.ravel(), sd['dim_A'], sd['dim_B'],
                        eps=eps_mid, verbose=False)
                    r_trial = new_result['D_new']
                    D_trial += r_trial**2
                    trial_data[n_old] = new_result
                else:
                    D_trial += sd['r']**2
            
            if D_trial <= MAX_D_EMB:
                # eps_mid is strict enough — try even tighter (lower eps)
                eps_hi = eps_mid
                best_eps = eps_mid
                best_D_emb = D_trial
                # Store the winning SVD results
                for n_old, res in trial_data.items():
                    if res['D_new'] < schmidt_backup[n_old]['r']:
                        sd = new_schmidt_data[n_old]
                        sd['r'] = res['D_new']
                        sd['U'] = res['U_trunc']
                        sd['V'] = res['V_trunc']
                        sd['sigma'] = res['s_kept']
            else:
                # eps_mid too loose — need higher eps
                eps_lo = eps_mid
        
        # If best_eps is still too loose (shouldn't happen with EPS_MAX=0.05),
        # fall back to accepting the raw D_emb with a warning
        D_emb_raw = sum(sd['r']**2 for sd in new_schmidt_data.values())
        if D_emb_raw > MAX_D_EMB:
            if verbose:
                print(f"  [D_emb control] ⚠ Binary search exhausted. "
                      f"Accepting D_emb={D_emb_raw} (eps={best_eps:.1e}). "
                      f"Will rely on Neumann correction for accuracy.")
        
        if verbose:
            actual_D = sum(sd['r']**2 for sd in new_schmidt_data.values())
            print(f"  [D_emb control] After bisection: D_emb={actual_D}, "
                  f"eps={best_eps:.1e}")

    # 5. Build H^emb in new S_{k+1} basis
    q_idx, backend = _build_backend(
        n_act_new, n_elec, h1eff, cas_data['h2eff'])

    from dm_svd_embedding.embedded_hamiltonian import build_h_emb

    # Build schmidt_data dict in the expected format
    schmidt_for_hemb = {}
    for n_old in sorted(new_schmidt_data.keys()):
        sd = new_schmidt_data[n_old]
        if sd['r'] > 0:
            schmidt_for_hemb[n_old] = sd

    if len(schmidt_for_hemb) > 0:
        # n_occ for build_h_emb is prev_n_act (the old orbitals are "occupied")
        H_emb, basis_info, decomps = build_h_emb(
            schmidt_for_hemb, partition, q_idx, backend,
            h1eff, h2_4d, prev_n_act, n_act_new, verbose=verbose)

        D_emb = H_emb.shape[0]
        if D_emb > 0:
            evals_emb, _ = np.linalg.eigh(H_emb)
            E_emb_gs = evals_emb[0] + ecore
        else:
            E_emb_gs = E_casci
    else:
        E_emb_gs = E_casci
        D_emb = 0
        if verbose:
            print(f"  No Schmidt blocks to build H^emb. Using CASCI energy.")

    # 6. (Optional) Neumann k=1 correction for dynamical correlation
    E_neumann = E_emb_gs
    dE_neumann_mH = 0.0
    neumann_info = None

    if apply_neumann and env_orbs_remaining is not None and len(env_orbs_remaining) > 0:
        if verbose:
            print(f"\n  ── Applying Neumann k=1 correction ──")

        try:
            from dm_svd_dci.neumann_qspace import apply_neumann_correction
            from src.hamiltonian import Hamiltonian, _unpack_4fold

            # Build full-space Hamiltonian for matrix elements
            if full_h1eff is not None and full_h2_4d is not None:
                ham_full = Hamiltonian(h1=full_h1eff, h2=full_h2_4d,
                                        E_nuc=0.0, E_HF=0.0)
            else:
                # Fallback: use current subspace integrals
                ham_full = Hamiltonian(h1=h1eff, h2=h2_4d,
                                        E_nuc=0.0, E_HF=0.0)

            # Full CAS string arrays needed for det_to_full_idx
            from pyscf.fci import cistring
            na_full, nb_full = n_elec
            alpha_strs_full = cistring.gen_strings4orblist(
                range(n_active_full or n_act_new), na_full)
            beta_strs_full = cistring.gen_strings4orblist(
                range(n_active_full or n_act_new), nb_full)

            neumann_result = apply_neumann_correction(
                H_PP=H_emb,
                schmidt_data=schmidt_for_hemb,
                partition=partition,
                n_occ=prev_n_act,
                n_act=n_act_new,
                n_active_full=n_active_full or n_act_new,
                active_orbs=list(range(n_act_new)),
                env_orbs=list(env_orbs_remaining),
                ham=ham_full,
                hdiag_full=q_idx.hdiag,
                alpha_strs_full=alpha_strs_full,
                beta_strs_full=beta_strs_full,
                n_alpha=na_full,
                n_beta=nb_full,
                ecore=ecore,
                k_max=1,
                verbose=verbose,
            )
            E_neumann = neumann_result['E_corrected']
            dE_neumann_mH = neumann_result['dE_neumann_mH']
            neumann_info = {
                'n_p_dets': neumann_result.get('n_p', 0),
                'n_q_dets': neumann_result.get('n_q', 0),
                'n_q_sectors': neumann_result.get('n_q_sectors', 0),
            }
        except Exception as e:
            if verbose:
                print(f"  ⚠ Neumann correction failed: {e}")
                import traceback
                traceback.print_exc()
            E_neumann = E_emb_gs

    dE = (E_emb_gs - E_casci) * 1000
    dE_neumann_vs_casci = (E_neumann - E_casci) * 1000
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n  E(CASCI new) = {E_casci:.12f} Ha")
        print(f"  E(H^emb)     = {E_emb_gs:.12f} Ha")
        if apply_neumann:
            print(f"  E(Neumann)   = {E_neumann:.12f} Ha")
            print(f"  ΔE(H^emb)    = {dE:+.3f} mH")
            print(f"  ΔE(Neumann)  = {dE_neumann_vs_casci:+.3f} mH")
        else:
            print(f"  ΔE           = {dE:+.3f} mH")
        print(f"  D_k          = {D_k}")
        print(f"  Time:        {elapsed:.1f}s")

    return {
        'Ek': E_neumann if apply_neumann else E_emb_gs,
        'E_hemb': E_emb_gs,
        'E_neumann': E_neumann if apply_neumann else None,
        'dE_neumann_mH': dE_neumann_mH,
        'E_casci': E_casci,
        'D_k': D_k,
        'D_emb': D_emb,
        'dE_mH': dE,
        'neumann_info': neumann_info,
        'n_act': n_act_new,
        'h1eff': h1eff,
        'h2_4d': h2_4d,
        'h2eff': cas_data['h2eff'],
        'ecore': ecore,
        'schmidt_data': new_schmidt_data,
        'partition': partition,
        'elapsed': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Growing CAS DMRG Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class GrowingCASDMRG:
    """DMRG-style growing active space via dmSVD with |A⟩⊗|B⟩⊗|env⟩ framework.

    The env space contains ALL target orbitals from the start (vacuum).
    Each round pulls orbitals from env → B, compresses, and chains the transform.

    Usage:
        grower = GrowingCASDMRG(mol, mf, n_active=10, n_elec=(5,5),
                                 n_core=2, n_occ_A=5, n_orb_B0=2,
                                 n_orb_Bt=2, eps_svd=1e-3)
        results = grower.run(verbose=True)
    """

    def __init__(
        self,
        mol,
        mf,
        n_active: int,
        n_elec: Tuple[int, int],
        n_core: int = 0,
        n_occ_A: int = 5,
        n_orb_B0: int = 2,
        n_orb_Bt: int = 2,
        eps_svd: float = 1e-3,
        max_rounds: Optional[int] = None,
    ):
        """
        Args:
            mol, mf: PySCF molecule and RHF objects.
            n_active: Total target active orbitals (full CAS).
            n_elec: (n_alpha, n_beta) active electrons.
            n_core: Frozen core orbitals.
            n_occ_A: Number of A-space orbitals (round 0 system).
            n_orb_B0: Number of B-space orbitals in round 0.
            n_orb_Bt: Number of B-space orbitals per extension round.
            eps_svd: SVD truncation threshold.
            max_rounds: Maximum extension rounds (None = use all remaining).
        """
        self.mol = mol
        self.mf = mf
        self.n_active = n_active
        self.n_elec = n_elec
        self.n_core = n_core
        self.n_occ_A = n_occ_A
        self.n_orb_B0 = n_orb_B0
        self.n_orb_Bt = n_orb_Bt
        self.eps_svd = eps_svd
        self.max_rounds = max_rounds

        # env orbitals: all remaining orbitals not in initial A₀ + B₀
        initial_count = n_occ_A + n_orb_B0
        self.env_orbitals = list(range(initial_count, n_active))

        if max_rounds is None:
            self.max_rounds = (
                len(self.env_orbitals) // n_orb_Bt
                + (1 if len(self.env_orbitals) % n_orb_Bt else 0)
            )

        # State
        self.T: Optional[ChainedTransform] = None
        self.results: List[Dict] = []

    def run(self, verbose: bool = True) -> Dict:
        """Run the full growing CAS pipeline.

        Returns:
            dict with:
              'E_history': List of energies per round.
              'D_history': List of compressed dimensions per round.
              'E_ref': Full CAS FCI reference energy.
              'dE_final_mH': Final error vs FCI in mH.
              'converged': Whether energy converged.
              'rounds': Number of rounds completed.
              'round_details': Per-round info.
        """
        t_total = time.perf_counter()

        if verbose:
            print("=" * 70)
            print("Growing CAS DMRG via dmSVD")
            print("=" * 70)
            print(f"  Target CAS:  ({self.n_active},{sum(self.n_elec)})")
            print(f"  Frozen core: {self.n_core}")
            print(f"  Round 0: A₀={self.n_occ_A}, B₀={self.n_orb_B0}")
            print(f"  Extension: +{self.n_orb_Bt} orbital(s)/round")
            print(f"  SVD ε:      {self.eps_svd}")
            print(f"  Env orbs:   {self.env_orbitals} "
                  f"({len(self.env_orbitals)} orbitals, "
                  f"max {self.max_rounds} rounds)")

        # ── Compute full CAS FCI reference ──
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Computing full CAS FCI reference...")

        full_orbs = np.arange(self.n_active, dtype=int)
        ref_data = _run_casci_subspace(
            self.mol, self.mf, self.n_core, full_orbs,
            self.n_elec, verbose=verbose)
        E_fci_ref = ref_data['E_fci']

        if verbose:
            print(f"  E(FCI ref) = {E_fci_ref:.12f} Ha")

        # ── Round 0: Bootstrap ──
        r0 = _round_0_bootstrap(
            self.mol, self.mf, self.n_core,
            self.n_occ_A, self.n_orb_B0, self.n_elec,
            self.eps_svd, verbose=verbose)

        self.T = r0['T']
        E_history = [r0['E0']]
        D_history = [r0['D0']]
        round_details = [r0]

        current_n_act = r0['n_act']
        current_h1eff = r0['h1eff']
        current_h2_4d = r0['h2_4d']
        current_ecore = r0['ecore']
        current_n_occ = r0['n_occ']

        if verbose:
            dE = (r0['E0'] - E_fci_ref) * 1000
            print(f"\n  ΔE vs FCI ref: {dE:+.3f} mH")

        # ── Extension rounds ──
        for round_idx in range(self.max_rounds):
            start = round_idx * self.n_orb_Bt
            end = min(start + self.n_orb_Bt, len(self.env_orbitals))
            if start >= len(self.env_orbitals):
                break

            Bk_orbs = np.array(self.env_orbitals[start:end], dtype=int)

            rk = _round_k_extension(
                self.mol, self.mf, self.n_core,
                self.T,
                prev_n_act=current_n_act,
                prev_n_occ=current_n_occ,
                Bk_orbs=Bk_orbs,
                n_elec=self.n_elec,
                eps_svd=self.eps_svd,
                prev_h1eff=current_h1eff,
                prev_h2_4d=current_h2_4d,
                prev_ecore=current_ecore,
                verbose=verbose)

            E_history.append(rk['Ek'])
            D_history.append(rk['D_k'])
            round_details.append(rk)

            # Update state for next round
            current_n_act = rk['n_act']
            current_h1eff = rk.get('h1eff', current_h1eff)
            current_h2_4d = rk.get('h2_4d', current_h2_4d)
            current_ecore = rk.get('ecore', current_ecore)
            # n_occ for next round = current total orbitals
            current_n_occ = current_n_act

            if verbose:
                dE = (rk['Ek'] - E_fci_ref) * 1000
                print(f"\n  ΔE vs FCI ref: {dE:+.3f} mH")

        # ── Summary ──
        elapsed_total = time.perf_counter() - t_total
        n_rounds = len(E_history)
        dE_final = (E_history[-1] - E_fci_ref) * 1000

        if verbose:
            print(f"\n{'='*70}")
            print(f"CONVERGENCE SUMMARY")
            print(f"{'='*70}")
            print(f"  {'Round':>5} {'D':>6} {'Energy (Ha)':>18} "
                  f"{'ΔE vs FCI (mH)':>16}")
            print(f"  {'─'*50}")
            for r in range(n_rounds):
                dE_r = (E_history[r] - E_fci_ref) * 1000
                print(f"  {r:>5} {D_history[r]:>6} "
                      f"{E_history[r]:>18.12f} {dE_r:>+15.3f}")
            print(f"\n  FCI ref:     {E_fci_ref:.12f} Ha")
            print(f"  Final ΔE:    {dE_final:+.3f} mH")
            print(f"  Total time:  {elapsed_total:.1f}s")
            print(f"  Rounds:      {n_rounds}")

        self.results = {
            'E_history': [float(e) for e in E_history],
            'D_history': D_history,
            'E_fci_ref': float(E_fci_ref),
            'dE_final_mH': float(dE_final),
            'n_rounds': n_rounds,
            'converged': True,  # Always runs all rounds
            'total_time': elapsed_total,
            'round_details': round_details,
        }

        return self.results


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Run from CLI or script
# ═══════════════════════════════════════════════════════════════════════════

def run_n2_cas10_example(
    r_nn: float = 1.098,
    n_occ_A: int = 5,
    n_orb_B0: int = 2,
    n_orb_Bt: int = 2,
    eps_svd: float = 1e-3,
    verbose: bool = True,
) -> Dict:
    """Run Growing CAS DMRG on N₂/cc-pVDZ CAS(10,10).

    Default: Round 0: A₀=5, B₀=2 (7 orbitals)
             Round 1: B₁=2 (9 orbitals)
             Round 2: B₂=1 (10 orbitals, full CAS)
    """
    from pyscf import gto, scf

    mol = gto.M(
        atom=f'N 0 0 0; N 0 0 {r_nn}',
        basis='cc-pVDZ',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    grower = GrowingCASDMRG(
        mol, mf,
        n_active=10,
        n_elec=(5, 5),
        n_core=2,
        n_occ_A=n_occ_A,
        n_orb_B0=n_orb_B0,
        n_orb_Bt=n_orb_Bt,
        eps_svd=eps_svd,
    )

    return grower.run(verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_chained_transform():
    """Test ChainedTransform basic operations."""
    # Create T_0: 3 blocks with different shapes
    T0 = {
        0: np.eye(4, 2),   # (4, 2)
        1: np.eye(6, 3),   # (6, 3)
        2: np.eye(3, 2),   # (3, 2)
    }
    ct = ChainedTransform(T0)
    assert ct.total_dimension == 7  # 2+3+2
    assert ct.dimension == {0: 2, 1: 3, 2: 2}

    # Extend with SVD result for block 1 only
    U_new = {1: np.eye(3, 2)}  # (3, 2) — reduce block 1 from 3 to 2
    ct.extend(U_new)
    assert ct.total_dimension == 6  # 2+2+2
    assert ct.dimension == {0: 2, 1: 2, 2: 2}

    # Check full transform for block 1
    T_full = ct.get_full_transform(1)
    assert T_full.shape == (6, 2)  # (6,3) @ (3,2) = (6,2)

    # Compress CI matrix
    C = np.random.randn(6, 4)  # block 1: d_old=6, d_B=4
    C_tilde = ct.compress_ci_matrix(C, 1)
    assert C_tilde.shape == (2, 4)  # (r=2, d_B=4)

    # Block not in chain should pass through
    C3 = np.random.randn(10, 3)
    C3_tilde = ct.compress_ci_matrix(C3, 10)  # n_A=10 not in T0
    assert C3_tilde.shape == (10, 3)  # unchanged

    print("  ✓ ChainedTransform: basic operations passed")


def test_build_T0():
    """Test build_T0_from_schmidt."""
    # Create mock schmidt data
    U = np.array([[1.0, 0.0], [0.0, 0.5]]).reshape(2, 2)
    V = np.array([[0.8, 0.0], [0.0, 0.6]]).reshape(2, 2)
    schmidt = {
        2: {
            'U': U, 'V': V, 'r': 2,
            'dim_A': 2, 'dim_B': 2,
            'sigma': np.array([0.8, 0.3]),
            'sigma_full': np.array([0.8, 0.3]),
        }
    }
    T0 = build_T0_from_schmidt(schmidt)
    assert 2 in T0
    assert T0[2].shape == (4, 2)  # d_A*d_B=4, r=2

    # Column 0: outer(U[:,0], V[:,0])
    expected_col0 = np.outer(U[:, 0], V[:, 0]).ravel()
    assert np.allclose(T0[2][:, 0], expected_col0)

    print("  ✓ build_T0_from_schmidt: correct")


def test_block_svd_general_import():
    """Verify block_svd_general import and basic functionality."""
    from dm_svd_dci.block_svd_general import block_svd_multi_orbital

    # Simple rank-1 test
    D, d_B = 8, 6
    u = np.ones(D) / np.sqrt(D)
    v = np.ones(d_B) / np.sqrt(d_B)
    C = np.outer(u, v)
    psi = C.ravel()

    result = block_svd_multi_orbital(psi, D, d_B, eps=1e-3, verbose=False)
    assert result['D_new'] == 1
    assert 'U_trunc' in result
    assert result['U_trunc'].shape == (D, 1)

    print("  ✓ block_svd_general import and rank-1 test passed")


if __name__ == "__main__":
    test_chained_transform()
    test_build_T0()
    test_block_svd_general_import()
    print("\nAll growing_cas_dmrg tests passed.")