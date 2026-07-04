"""
PySCF C-level backend for Krylov-dCI dense operations.

Replaces Python-level Slater-Condon rules with PySCF's libfci
(selected_ci.contract_2e + direct_spin1.contract_1e) for all
Hamiltonian-vector products.

Architecture:
  Q-space = full CAS CI space (all alpha × all beta strings).
  P-space = selected subset of this full space.

Since Q uses the same α/β string set as the full CAS, we can use:
  - selected_ci.contract_2e  for H_2e · vec  (C-level, selected strings)
  - direct_spin1.contract_1e for H_1e · vec  (C-level, full CI matrix)

Key classes:
  QSpaceIndex  — Q-space indexing + C-level link tables + hdiag
  KDCIBackend  — H_QP, sigma, MGS, projected blocks

References:
  - PySCF selected_ci.contract_2e: SCIcontract_2e_aaaa + SCIcontract_2e_bbaa
  - link_index = _all_linkstr_index(ci_strs, norb, nelec)
"""

import numpy as np
from numpy.linalg import eigh
from typing import Tuple, List, Optional
import time



from .qspace import QSpaceIndex
from .kdci_dense import KDCIBackend
from .sparse_vector import SparseQVector


class KDCISparse(KDCIBackend):
    """Matrix-free sparse Krylov-dCI backend.

    Builds Krylov bases WITHOUT storing dense H_QP (M x N).
    Uses streaming MGS + indexed sparse projection.
    """

    def __init__(self, q_idx: QSpaceIndex):
        self.q_idx = q_idx

    def build_basis_streaming(self, p_dets: List[Tuple[int, int]],
                              E0_P: float,
                              lindep_threshold: float = 1e-10,
                              verbose: bool = True
                              ) -> Tuple[List, int]:
        """Build orthonormal basis WITHOUT storing dense (M, N) H_QP.

        Streams one P-det at a time:
          1. contract_2e on unit CI → dense sigma (temporary)
          2. Extract sparse: A_q * sigma[q] for q not in P
          3. MGS against existing basis vectors (sparse ops)
          4. If linearly independent: normalize, add to basis
          5. Discard dense sigma

        Persistent storage: only d SparseQVector objects.
        Temporary: one (na, nb) dense CI matrix per iteration.
        """
        from .sparse_vector import SparseQVector

        N = len(p_dets)
        na = self.q_idx.n_alpha
        nb = self.q_idx.n_beta

        denom = E0_P - self.q_idx.hdiag
        mask = np.abs(denom) > 1e-10
        A_q = np.zeros(self.q_idx.M)
        A_q[mask] = 1.0 / denom[mask]

        p_indices = set()
        for pa, pb in p_dets:
            idx = self.q_idx.flat_index(int(pa), int(pb))
            p_indices.add(idx)

        if verbose:
            t0 = time.perf_counter()
            print(f"    Streaming MGS: {N} columns → basis...", flush=True)

        basis = []
        for p in range(N):
            pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
            ia = self.q_idx._alpha_idx.get(pa)
            ib = self.q_idx._beta_idx.get(pb)
            if ia is None or ib is None:
                continue

            ci_unit = np.zeros((na, nb))
            ci_unit[ia, ib] = 1.0
            sigma_mat = self.sigma_full(ci_unit)
            sigma_flat = sigma_mat.reshape(-1)

            w_p = SparseQVector()
            nnz = np.where(np.abs(sigma_flat) > 1e-14)[0]
            for q in nnz:
                if q in p_indices:
                    continue
                val = A_q[q] * sigma_flat[q]
                if abs(val) > 1e-14:
                    a_str = int(self.q_idx.alpha_strs[q // nb])
                    b_str = int(self.q_idx.beta_strs[q % nb])
                    w_p[(a_str, b_str)] = float(val)

            for b in basis:
                proj = b.dot(w_p)
                w_p.add_scaled(b, alpha=-proj)

            nrm = w_p.norm()
            if nrm > lindep_threshold:
                w_p.scale(1.0 / nrm)
                basis.append(w_p)

            if verbose and (p + 1) % max(1, N // 5) == 0:
                print(f"      col {p+1}/{N}, basis={len(basis)} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

        d = len(basis)
        if verbose:
            total_nnz = sum(b.nnz() for b in basis)
            elapsed = time.perf_counter() - t0
            print(f"    Streaming MGS: {N} → {d} vectors, "
                  f"{total_nnz} total nnz, {elapsed:.0f}s", flush=True)

        return basis, d

    def build_projected_blocks_sparse(self,
                                       basis: List,
                                       p_dets: List[Tuple[int, int]],
                                       verbose: bool = True
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Build H_{Q̃Q̃} and H_{PQ̃} from SPARSE basis vectors (optimized).

        Pre-computes flat index arrays to replace Python dict lookups
        with numpy advanced indexing. Key optimization:
          Old: d² × avg_nnz dict lookups (50ns each → ~8s for 160M)
          New: d × avg_nnz numpy gather (vectorized → ~0.5s)

        Only ONE dense M-vector exists at a time.
        """
        from .sparse_vector import SparseQVector

        d = len(basis)
        N = len(p_dets)
        na = self.q_idx.n_alpha
        nb = self.q_idx.n_beta

        p_indices = self.q_idx.p_indices(p_dets)
        p_valid = p_indices >= 0
        p_flat = p_indices[p_valid]

        # Pre-compute flat index arrays for all basis vectors
        # Replaces d² dict iterations with numpy advanced indexing
        basis_idx = []   # list of (indices_array, values_array)
        for b in basis:
            idx_list = []
            val_list = []
            for (a_str, b_str), coef in b.items():
                flat = self.q_idx.flat_index(int(a_str), int(b_str))
                idx_list.append(flat)
                val_list.append(coef)
            basis_idx.append((
                np.array(idx_list, dtype=np.int64),
                np.array(val_list)
            ))

        H_QQ_tilde = np.zeros((d, d))
        H_PQ_tilde = np.zeros((N, d))

        if verbose:
            t0 = time.perf_counter()
            print(f"    Projecting {d} basis vectors (sparse, indexed)...",
                  flush=True)

        for k, b_k in enumerate(basis):
            # Materialize dense CI matrix from sparse b_k
            ci_mat = np.zeros((na, nb))
            for (a_str, b_str), coef in b_k.items():
                ia = self.q_idx._alpha_idx.get(int(a_str))
                ib = self.q_idx._beta_idx.get(int(b_str))
                if ia is not None and ib is not None:
                    ci_mat[ia, ib] = coef

            sigma_mat = self.sigma_full(ci_mat)
            sigma_flat = sigma_mat.reshape(-1)

            # Vectorized sparse-dense dots: gather sigma at basis indices
            for j, (idxs, vals) in enumerate(basis_idx):
                H_QQ_tilde[j, k] = np.dot(vals, sigma_flat[idxs])

            H_PQ_tilde[p_valid, k] = sigma_flat[p_flat]

            if verbose and (k + 1) % max(1, d // 5) == 0:
                print(f"      basis {k+1}/{d} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"    Sparse projection done in {elapsed:.0f}s "
                  f"({elapsed/max(d,1):.2f}s/vector)", flush=True)

        H_QQ_tilde = 0.5 * (H_QQ_tilde + H_QQ_tilde.T)
        return H_QQ_tilde, H_PQ_tilde

    def _build_projected_blocks_sparse_slow(self,
                                             basis: List,
                                             p_dets: List[Tuple[int, int]],
                                             verbose: bool = True
                                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy sparse projection with per-entry dict lookups (kept for validation)."""
        d = len(basis)
        N = len(p_dets)
        na = self.q_idx.n_alpha
        nb = self.q_idx.n_beta

        p_indices = self.q_idx.p_indices(p_dets)
        p_valid = p_indices >= 0
        p_flat = p_indices[p_valid]

        H_QQ_tilde = np.zeros((d, d))
        H_PQ_tilde = np.zeros((N, d))

        if verbose:
            t0 = time.perf_counter()
            print(f"    Projecting {d} basis vectors (sparse, dict)...", flush=True)

        for k, b_k in enumerate(basis):
            ci_mat = np.zeros((na, nb))
            for (a_str, b_str), coef in b_k.items():
                ia = self.q_idx._alpha_idx.get(int(a_str))
                ib = self.q_idx._beta_idx.get(int(b_str))
                if ia is not None and ib is not None:
                    ci_mat[ia, ib] = coef

            sigma_mat = self.sigma_full(ci_mat)
            sigma_flat = sigma_mat.reshape(-1)

            for j, b_j in enumerate(basis):
                dot_val = 0.0
                for (a_str, b_str), coef in b_j.items():
                    idx = self.q_idx.flat_index(int(a_str), int(b_str))
                    dot_val += coef * sigma_flat[idx]
                H_QQ_tilde[j, k] = dot_val

            H_PQ_tilde[p_valid, k] = sigma_flat[p_flat]

            if verbose and (k + 1) % max(1, d // 5) == 0:
                print(f"      basis {k+1}/{d} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"    Sparse projection done in {elapsed:.0f}s "
                  f"({elapsed/max(d,1):.2f}s/vector)", flush=True)

        H_QQ_tilde = 0.5 * (H_QQ_tilde + H_QQ_tilde.T)
        return H_QQ_tilde, H_PQ_tilde

    # ═══════════════════════════════════════════════════════════════
    # Diagonal-only sigma (fast approximate path, not for production)
    # ═══════════════════════════════════════════════════════════════

    def sigma_diagonal(self, vec: np.ndarray) -> np.ndarray:
        """H_QQ @ vec using only diagonal (includes 1e + 2e diag)."""
        return self.q_idx.hdiag * vec


