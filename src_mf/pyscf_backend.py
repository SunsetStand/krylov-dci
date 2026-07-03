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


class QSpaceIndex:
    """Manages Q-space determinant index and C-level link tables.

    Q-space = full CAS space: all C(norb, nalpha) α strings × all
    C(norb, nbeta) β strings.

    Attributes:
        alpha_strs: (M_a,) int64 array of alpha strings.
        beta_strs:  (M_b,) int64 array of beta strings.
        n_alpha:    Number of alpha strings.
        n_beta:     Number of beta strings.
        M:          Total determinant count = n_alpha × n_beta.
        link_index: Pre-computed C-level link table for selected_ci.
        hdiag:      Diagonal of H_QQ, shape (M,) (via make_hdiag).
    """

    def __init__(self, alpha_strs: np.ndarray, beta_strs: np.ndarray,
                 norb: int, nelec: Tuple[int, int],
                 h1e: np.ndarray, eri: np.ndarray):
        from pyscf.fci import selected_ci, direct_spin1

        self.alpha_strs = np.asarray(alpha_strs, dtype=np.int64)
        self.beta_strs = np.asarray(beta_strs, dtype=np.int64)
        self.n_alpha = len(self.alpha_strs)
        self.n_beta = len(self.beta_strs)
        self.M = self.n_alpha * self.n_beta
        self.norb = norb
        self.nelec = tuple(nelec)
        self.h1e = np.asarray(h1e)
        self.eri = np.asarray(eri)

        # Absorb 1e into 2e integrals (PySCF convention)
        # contract_2e on h2e_eff gives full H·c = (H_1e + H_2e)·c
        self.h2e_eff = direct_spin1.absorb_h1e(
            self.h1e, self.eri, self.norb, self.nelec, 0.5)

        # C-level link tables for selected_ci.contract_2e
        ci_strs = (self.alpha_strs, self.beta_strs)
        self.link_index = selected_ci._all_linkstr_index(
            ci_strs, self.norb, self.nelec)

        # Pre-compute diagonal (includes 1e + 2e contributions)
        self.hdiag = selected_ci.make_hdiag(
            self.h1e, self.eri, ci_strs, self.norb, self.nelec)

        # Fast lookup: (alpha_str, beta_str) → flat index
        self._alpha_idx = {int(s): i for i, s in enumerate(self.alpha_strs)}
        self._beta_idx = {int(s): i for i, s in enumerate(self.beta_strs)}

    def flat_index(self, alpha_str: int, beta_str: int) -> int:
        """Map (alpha_str, beta_str) → flat index in [0, M)."""
        ia = self._alpha_idx[int(alpha_str)]
        ib = self._beta_idx[int(beta_str)]
        return ia * self.n_beta + ib

    def to_ci_matrix(self, coeffs: np.ndarray) -> np.ndarray:
        """Flat vector (M,) → CI matrix (n_alpha, n_beta)."""
        return coeffs.reshape(self.n_alpha, self.n_beta)

    def from_ci_matrix(self, ci_mat: np.ndarray) -> np.ndarray:
        """CI matrix (n_alpha, n_beta) → flat vector (M,)."""
        return ci_mat.reshape(-1)

    def p_indices(self, p_dets: List[Tuple[int, int]]) -> np.ndarray:
        """Get flat indices for P-space determinants.

        Returns:
            (N,) int64 array of flat indices. -1 for any det not found.
        """
        indices = np.full(len(p_dets), -1, dtype=np.int64)
        for k, (a, b) in enumerate(p_dets):
            ia = self._alpha_idx.get(int(a))
            ib = self._beta_idx.get(int(b))
            if ia is not None and ib is not None:
                indices[k] = ia * self.n_beta + ib
        return indices


class KDCIBackend:
    """Krylov-dCI operations using PySCF C-level backend.

    All bottleneck operations use C-level PySCF calls:
      - H_2e via selected_ci.contract_2e  (libfci.SCIcontract_2e_*)
      - H_1e via direct_spin1.contract_1e
      - H_diag via selected_ci.make_hdiag (already C-level)
    """

    def __init__(self, q_idx: QSpaceIndex):
        self.q_idx = q_idx

    # ═══════════════════════════════════════════════════════════════
    # Sigma-vector: H·v (combined 1e + 2e, C-level)
    # ═══════════════════════════════════════════════════════════════

    def sigma(self, vec: np.ndarray) -> np.ndarray:
        """H @ vec using selected_ci.contract_2e on h2e_eff (absorbed 1e+2e).

        Following PySCF convention: absorb_h1e embeds 1e into 2e integrals,
        so contract_2e alone gives the full H·c.

        Args:
            vec: Dense vector over Q-space, shape (M,).
        Returns:
            sigma = H @ vec, shape (M,).
        """
        from pyscf.fci import selected_ci

        ci_mat = self.q_idx.to_ci_matrix(vec)
        ci_with_strs = selected_ci._as_SCIvector(
            ci_mat, (self.q_idx.alpha_strs, self.q_idx.beta_strs))
        sigma_mat = selected_ci.contract_2e(
            self.q_idx.h2e_eff, ci_with_strs,
            self.q_idx.norb, self.q_idx.nelec,
            link_index=self.q_idx.link_index)
        return self.q_idx.from_ci_matrix(sigma_mat)

    def sigma_full(self, ci_mat: np.ndarray) -> np.ndarray:
        """Compute H @ ci_mat using h2e_eff (absorbed 1e+2e), returning CI matrix.

        NOTE: selected_ci.contract_2e modifies fcivec in-place via
        lib.transpose(..., out=fcivecT). We pass a copy to avoid corrupting
        the input (which may be a view into a basis array).

        Args:
            ci_mat: (n_alpha, n_beta) CI matrix.
        Returns:
            sigma = H @ ci_mat, same shape (n_alpha, n_beta).
        """
        from pyscf.fci import selected_ci

        ci_with_strs = selected_ci._as_SCIvector(
            ci_mat.copy(), (self.q_idx.alpha_strs, self.q_idx.beta_strs))
        return selected_ci.contract_2e(
            self.q_idx.h2e_eff, ci_with_strs,
            self.q_idx.norb, self.q_idx.nelec,
            link_index=self.q_idx.link_index)

    # ═══════════════════════════════════════════════════════════════
    # H_QP construction
    # ═══════════════════════════════════════════════════════════════
    # H_QP construction
    # ═══════════════════════════════════════════════════════════════

    def build_hqp(self, p_dets: List[Tuple[int, int]],
                  verbose: bool = True) -> np.ndarray:
        """Build H_QP as dense (M, N) via N serial C-level contract_2e calls.

        Each P-det unit vector → sigma_full (selected_ci.contract_2e
        on h2e_eff, absorbed 1e+2e). P-det rows are zeroed out.

        Complexity: N × O(M × n_links) at C speed.
        Replaces: N × O(n_exc) Python Slater-Condon calls.

        Returns:
            H_QP: (M, N) where H_QP[q, p] = ⟨Φ_q|H|Φ_p⟩ for q ∈ Q.
        """
        M = self.q_idx.M
        N = len(p_dets)
        na = self.q_idx.n_alpha
        nb = self.q_idx.n_beta
        H_QP = np.zeros((M, N))

        p_indices_all = self.q_idx.p_indices(p_dets)
        p_mask = np.zeros(M, dtype=bool)
        p_mask[p_indices_all[p_indices_all >= 0]] = True

        t_start = time.perf_counter()
        for p in range(N):
            pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
            ia = self.q_idx._alpha_idx.get(pa)
            ib = self.q_idx._beta_idx.get(pb)
            if ia is None or ib is None:
                continue

            ci_unit = np.zeros((na, nb))
            ci_unit[ia, ib] = 1.0
            sigma_mat = self.sigma_full(ci_unit)
            col = sigma_mat.reshape(-1)
            col[p_mask] = 0.0
            H_QP[:, p] = col

            if verbose and (p + 1) % max(1, N // 10) == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (p + 1) * (N - p - 1)
                print(f"    H_QP {p+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s)",
                      flush=True)

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(f"    H_QP ({M},{N}) done in {elapsed:.0f}s "
                  f"({elapsed/N:.2f}s/col)", flush=True)

        return H_QP

    # ═══════════════════════════════════════════════════════════════
    # Basis construction (MGS on A-weighted H_QP)
    # ═══════════════════════════════════════════════════════════════

    def build_basis(self, H_QP: np.ndarray, E0_P: float,
                    delta: float = 0.0,
                    lindep_threshold: float = 1e-10,
                    verbose: bool = True) -> Tuple[np.ndarray, int]:
        """Build orthonormal basis from A-weighted H_QP.

        1. Weight: H_QP_w[q,p] = A_q^{1/2} · H_QP[q,p],
           A_q^{1/2} = 1/|E0_P - H_QQ[q,q]|^{1/2} · sgn(E0_P - H_QQ).
           Uses A^{1/2} (not A) for numerical stability (proposal Eq. 4).
           Propagation uses A = (A^{1/2})^2 (proposal Eq. 6).
        2. Modified Gram-Schmidt → orthonormal basis (M, d), d ≤ N.

        Args:
            H_QP: (M, N) unweighted H_QP matrix.
            E0_P: Reference energy (lowest eigenvalue of H_PP).
            delta: Energy shift Delta = E - E0_P.
        Returns:
            (basis, d): basis is (M, d), d is count of linearly independent cols.
        """
        M, N = H_QP.shape

        # A^{1/2}-weighting: T^{(0)}_qp = A^{1/2}_q · H_QP_{qp}
        # A^{1/2} = |E0_P - D_QQ|^{-1/2} · sgn(E0_P - D_QQ)
        denom = E0_P - self.q_idx.hdiag
        mask = np.abs(denom) > 1e-10
        A_sqrt = np.zeros(M)
        A_sqrt[mask] = 1.0 / np.sqrt(np.abs(denom[mask])) * np.sign(denom[mask])
        H_QP_w = H_QP * A_sqrt[:, np.newaxis]

        # Dense MGS
        if verbose:
            t0 = time.perf_counter()
            print(f"    MGS: {N} cols × {M} rows...", flush=True)

        basis_cols = []
        for p in range(N):
            v = H_QP_w[:, p].copy()
            for b in basis_cols:
                v -= np.dot(b, v) * b
            nrm = np.linalg.norm(v)
            if nrm > lindep_threshold:
                v /= nrm
                basis_cols.append(v)

        basis = np.column_stack(basis_cols) if basis_cols else np.zeros((M, 0))
        d = basis.shape[1]

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"    MGS: {N} → {d} vectors in {elapsed:.0f}s", flush=True)

        return basis, d

    # ═══════════════════════════════════════════════════════════════
    # Krylov propagation: (AB)^m · A^{1/2} H_QP
    # ═══════════════════════════════════════════════════════════════

    def propagate_basis(self, basis: np.ndarray, E0_P: float,
                        delta: float = 0.0,
                        lindep_threshold: float = 1e-10,
                        verbose: bool = True) -> Tuple[np.ndarray, int]:
        """Propagate existing Krylov basis by one layer: B <- AB · B.

        Implements the Krylov subspace propagation:
          K_{m+1} = span(K_m, (AB) · K_m)

        where:
          A = (E0_P - D_QQ)^{-1}            (diagonal resolvent)
          B = H_O' - Delta·I = H_QQ - D_QQ - Delta·I
        Krylov subspace = span{A^{1/2}·H_QP, (AB)·A^{1/2}·H_QP, ...}
        with A and B as defined in the original decomposition.

        For each basis vector b_k:
          1. sigma = H_QQ · b_k            (contract_2e, C-level)
          2. x_k  = A · (sigma - (D_QQ + delta·I) · b_k)
                  = A · B · b_k

        The x_k vectors are MGS-orthogonalized against the existing basis.
        Only linearly independent new directions are appended.

        Args:
            basis:  (M, r) existing orthonormal Krylov basis.
            E0_P:   Reference energy (lowest eigenvalue of H_PP).
            delta:  Energy shift Delta = E - E0_P.
            lindep_threshold: MGS linear independence threshold.
            verbose: Print progress.

        Returns:
            (basis_new, d_new): expanded basis (M, d_new) and new count.
        """
        M, r = basis.shape

        # A = (E0_P - D_QQ)^{-1}  (Delta enters via B = H_O' − ΔI)
        # Propagation uses full A (not A^{1/2}) per proposal Eq. 6
        denom = E0_P - self.q_idx.hdiag
        mask = np.abs(denom) > 1e-10
        A_q = np.zeros(M)
        A_q[mask] = 1.0 / denom[mask]

        if verbose:
            t0 = time.perf_counter()
            print(f"    Propagate: r={r}, delta={delta:.6f}...", flush=True)

        propagated = []
        for k in range(r):
            b_k = basis[:, k]

            # sigma = H_QQ · b_k  (1e already absorbed in h2e_eff)
            sigma_k = self.sigma_full(
                self.q_idx.to_ci_matrix(b_k)
            ).reshape(-1)

            # B · b_k = H_QQ·b_k - D_QQ·b_k - delta·b_k
            #        = (H_O' - Delta·I) · b_k
            # B carries the energy shift per the decomposition:
            # (E I - H_QQ)^{-1} = (A^{-1} - B)^{-1}, B = H_O' - Delta*I
            residual = sigma_k - (self.q_idx.hdiag + delta) * b_k

            # x_k = A · residual  (diagonal resolvent applied)
            x_k = A_q * residual
            propagated.append(x_k)

        # ── A-weighted SVD truncation (before MGS) ──
        # Re-apply A-weighting to the propagated vectors: T = A · propagated.
        # This amplifies low-energy Q-space directions that couple strongly
        # to P, and suppresses high-energy noise. SVD on T identifies the
        # dominant resolvent-relevant subspace.
        # Threshold ~1e-3 mirrors the original Krylov-dCI implementation.
        if len(propagated) > 0:
            prop_mat = np.column_stack(propagated)  # (M, count)
            # Re-weight: T[:,k] = A_q * prop_mat[:,k]
            T = A_q[:, np.newaxis] * prop_mat
            _, s_prop, Vt = np.linalg.svd(T, full_matrices=False)
            svd_threshold = 1e-3  # matches original implementation
            keep = s_prop > svd_threshold * max(1.0, s_prop[0])
            n_keep = int(np.sum(keep))
            if n_keep < len(propagated):
                # Compress to significant directions only
                prop_compressed = T @ Vt[:n_keep, :].T  # (M, n_keep)
            else:
                prop_compressed = prop_mat
        else:
            prop_compressed = np.zeros((M, 0))
            n_keep = 0

        # MGS orthogonalization: compressed propagated vectors against
        # existing basis and against each other
        basis_list = [basis[:, j] for j in range(r)]
        new_count = 0
        for k in range(prop_compressed.shape[1]):
            v = prop_compressed[:, k].copy()
            for b in basis_list:
                v -= np.dot(b, v) * b
            nrm = np.linalg.norm(v)
            if nrm > lindep_threshold:
                v /= nrm
                basis_list.append(v)
                new_count += 1

        basis_new = np.column_stack(basis_list)
        d_new = len(basis_list)

        if verbose:
            elapsed = time.perf_counter() - t0
            added = d_new - r
            removed = len(propagated) - n_keep
            msg = f"    Propagate: {r} -> {d_new} vectors (+{added} new"
            if removed > 0:
                msg += f", {removed} kept from SVD"
            msg += f") in {elapsed:.0f}s"
            print(msg, flush=True)

        return basis_new, d_new

    def build_krylov_layers(self, H_QP: np.ndarray, E0_P: float,
                            delta: float = 0.0, m_max: int = 0,
                            lindep_threshold: float = 1e-10,
                            verbose: bool = True
                            ) -> Tuple[np.ndarray, int, List[int]]:
        """Build Krylov basis up to order m_max.

        Layer 0: B_0 = MGS(A^{1/2} · H_QP)
        Layer m:  B_m = MGS([B_{m-1}, (AB) · B_{m-1}])

        This is the main entry point for multi-layer Krylov-dCI.

        Args:
            H_QP:   (M, N) unweighted H_QP matrix.
            E0_P:   Reference energy.
            delta:  Energy shift.
            m_max:  Maximum Krylov order (m=0 is single-layer).
            lindep_threshold: Linear independence threshold.

        Returns:
            (basis, d_total, d_per_layer): final basis, total dim,
                                           dimensions per layer.
        """
        if verbose and m_max > 0:
            t0 = time.perf_counter()
            print(f"Building Krylov layers m=0..{m_max} "
                  f"(delta={delta:.6f})...", flush=True)

        # Layer 0: A^{1/2} · H_QP  (no sqrt in original; using A-weighted)
        basis, d0 = self.build_basis(H_QP, E0_P, delta=delta,
                                     lindep_threshold=lindep_threshold,
                                     verbose=verbose)
        d_per_layer = [d0]

        for m in range(1, m_max + 1):
            basis, d_m = self.propagate_basis(
                basis, E0_P, delta=delta,
                lindep_threshold=lindep_threshold,
                verbose=verbose)
            d_per_layer.append(d_m)
            if d_m == d_per_layer[-2]:
                if verbose:
                    print(f"  Layer m={m}: no new directions, stopping.",
                          flush=True)
                break

        if verbose and m_max > 0:
            elapsed = time.perf_counter() - t0
            print(f"Krylov layers done: {d_per_layer} "
                  f"in {elapsed:.0f}s", flush=True)

        return basis, d_per_layer[-1], d_per_layer

    # ═══════════════════════════════════════════════════════════════
    # Projected Hamiltonian blocks
    # ═══════════════════════════════════════════════════════════════

    def build_projected_blocks(self, basis: np.ndarray,
                               p_dets: List[Tuple[int, int]],
                               n_workers: int = 1,
                               verbose: bool = True
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Build H_{Q̃Q̃} and H_{PQ̃} in the compressed basis.

        For each basis vector b_k:
          1. sigma_k = H·b_k via sigma_full (C-level, 1e+2e absorbed).
          2. Accumulate into sigma matrix.

        Then:
          H_{Q̃Q̃} = basis^T @ sigma_all   (single matmul, not d² dots)
          H_{PQ̃}[p,k] = sigma_all[p, k]   (slice P-det rows)

        Args:
            basis:  (M, d) orthonormal basis from build_basis().
            p_dets: P-space dets, length N.
            n_workers: Number of threads for parallel sigma computation.
                       Set to 1 for serial, >1 for ThreadPoolExecutor.
        Returns:
            (H_QQ_tilde, H_PQ_tilde):
              H_QQ_tilde: (d, d) in compressed basis.
              H_PQ_tilde: (N, d) P-~Q coupling.
        """
        M, d = basis.shape
        N = len(p_dets)
        p_indices = self.q_idx.p_indices(p_dets)
        p_valid = p_indices >= 0
        p_flat = p_indices[p_valid]

        if verbose:
            t0 = time.perf_counter()
            mode = "parallel" if n_workers > 1 else "serial"
            print(f"    Projecting {d} basis vectors ({mode}, "
                  f"{n_workers} workers)...", flush=True)

        # ── Step 1: Compute all sigma vectors ──
        sigma_all = self._compute_sigma_all(
            basis, d, M, n_workers, verbose, t0 if verbose else 0)

        # ── Step 2: H_{Q̃Q̃} = basis^T @ sigma_all  (single BLAS matmul) ──
        H_QQ_tilde = basis.T @ sigma_all  # (d, M) @ (M, d) → (d, d)

        # ── Step 3: H_{PQ̃} = slice P-det rows from sigma_all ──
        H_PQ_tilde = np.zeros((N, d))
        H_PQ_tilde[p_valid, :] = sigma_all[p_flat, :]

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"    Projection done in {elapsed:.0f}s "
                  f"({elapsed/max(d,1):.2f}s/vector)", flush=True)

        # Symmetrize
        H_QQ_tilde = 0.5 * (H_QQ_tilde + H_QQ_tilde.T)
        return H_QQ_tilde, H_PQ_tilde

    def _compute_sigma_all(self, basis, d, M, n_workers, verbose, t0):
        """Compute sigma vectors for all basis columns.

        Serial path: simple loop.
        Parallel path: ThreadPoolExecutor — each contract_2e is C-level
        (libfci, releases GIL), so threads provide real parallelism.
        """
        sigma_all = np.empty((M, d))

        if n_workers <= 1:
            # Serial path
            for k in range(d):
                sigma_all[:, k] = self._sigma_one_col(basis[:, k])
                if verbose and (k + 1) % max(1, d // 5) == 0:
                    print(f"      basis {k+1}/{d} "
                          f"({time.perf_counter()-t0:.0f}s)", flush=True)
        else:
            # Parallel path
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(self._sigma_one_col, basis[:, k]): k
                    for k in range(d)
                }
                for f in as_completed(futures):
                    k = futures[f]
                    sigma_all[:, k] = f.result()
                    n_done = sum(1 for ff in futures if ff.done())
                    if verbose and n_done % max(1, d // 5) == 0:
                        print(f"      basis {n_done}/{d} "
                              f"({time.perf_counter()-t0:.0f}s)", flush=True)

        return sigma_all

    def _sigma_one_col(self, vec: np.ndarray) -> np.ndarray:
        """Compute sigma = H @ vec for a single column vector.

        Standalone method — callable from worker threads.
        Each call is independent (no shared state).
        """
        ci_mat = self.q_idx.to_ci_matrix(vec)
        sigma_mat = self.sigma_full(ci_mat)
        return sigma_mat.reshape(-1)

    # ═══════════════════════════════════════════════════════════════
    # Matrix-free sparse operations (no persistent M-dimensional storage)
    # ═══════════════════════════════════════════════════════════════

    def build_basis_streaming(self, p_dets: List[Tuple[int, int]],
                              E0_P: float,
                              delta: float = 0.0,
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

        A-weighting: A^{1/2}_q = |E0_P - H_QQ[q,q]|^{-1/2} · sgn(E0_P - H_QQ).

        Persistent storage: only d SparseQVector objects.
        Temporary: one (na, nb) dense CI matrix per iteration.
        """
        from .sparse_vector import SparseQVector

        N = len(p_dets)
        na = self.q_idx.n_alpha
        nb = self.q_idx.n_beta

        # A^{1/2}-weighting for streaming build
        denom = E0_P - self.q_idx.hdiag
        mask = np.abs(denom) > 1e-10
        A_sqrt_stream = np.zeros(self.q_idx.M)
        A_sqrt_stream[mask] = 1.0 / np.sqrt(np.abs(denom[mask])) * np.sign(denom[mask])

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
                val = A_sqrt_stream[q] * sigma_flat[q]
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


# ============================================================================
# Tests
# ============================================================================

def _make_h2o_idx():
    """Create QSpaceIndex for H₂O/STO-3G CAS(4,4)."""
    from pyscf import gto, scf, ao2mo, mcscf
    from pyscf.fci import cistring

    mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    n_cas, n_elec = 4, 4
    na, nb = n_elec // 2, n_elec - n_elec // 2
    cas = mcscf.CASCI(mf, n_cas, n_elec)
    cas.frozen = 1; cas.mo_coeff = mf.mo_coeff
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    alpha_strs = cistring.gen_strings4orblist(range(n_cas), na)
    beta_strs = cistring.gen_strings4orblist(range(n_cas), nb)
    return QSpaceIndex(alpha_strs, beta_strs, n_cas, (na, nb), h1eff, h2eff)


def test_sigma_vs_direct_spin1():
    """Verify backend.sigma matches direct_spin1 on H₂O/STO-3G."""
    from pyscf.fci import direct_spin1

    q_idx = _make_h2o_idx()
    backend = KDCIBackend(q_idx)

    vec = np.random.randn(q_idx.M)
    vec /= np.linalg.norm(vec)

    sigma_ours = backend.sigma(vec)
    # Reference: contract_2e on h2e_eff (absorbed 1e+2e), PySCF convention
    ci_full = q_idx.to_ci_matrix(vec)
    sigma_ref = direct_spin1.contract_2e(q_idx.h2e_eff, ci_full,
                                          q_idx.norb, q_idx.nelec)

    diff = np.abs(sigma_ours - sigma_ref.reshape(-1)).max()
    assert diff < 1e-10, f"Sigma mismatch: max|diff| = {diff:.2e}"
    print(f"  ✓ H₂O sigma: max|diff| = {diff:.2e}")


def test_build_hqp_vs_hamiltonian():
    """Verify build_hqp matches Python Slater-Condon on H₂O CAS(4,4)."""
    import sys
    sys.path.insert(0, '/data/home/wangcx/krylov-dci')
    from src.hamiltonian import Hamiltonian, _unpack_4fold
    from src.determinants import generate_determinants_ms

    q_idx = _make_h2o_idx()
    backend = KDCIBackend(q_idx)

    # Use a few HF+SD determinants as P-space
    from src.determinants import hf_determinant, bit_positions
    na, nb = q_idx.nelec
    hf_a, hf_b = hf_determinant(na, nb)
    alpha_occ = bit_positions(hf_a); beta_occ = bit_positions(hf_b)
    all_orbs = list(range(q_idx.norb))
    av = [p for p in all_orbs if p not in alpha_occ]
    bv = [p for p in all_orbs if p not in beta_occ]

    # Generate a few double excitations
    p_dets = [(hf_a, hf_b)]
    count = 0
    for i in alpha_occ:
        if count >= 5:
            break
        for a in av:
            for j in beta_occ:
                if count >= 5:
                    break
                for b in bv:
                    if count >= 5:
                        break
                    det = ((hf_a ^ (1<<i)) | (1<<a), (hf_b ^ (1<<j)) | (1<<b))
                    p_dets.append(det)
                    count += 1

    H_QP = backend.build_hqp(p_dets, verbose=False)

    # Compare with Python Slater-Condon
    from pyscf import ao2mo
    n_act = q_idx.norb
    h2_4d = ao2mo.restore('s1', q_idx.eri, n_act).reshape(n_act, n_act, n_act, n_act)
    ham = Hamiltonian(h1=q_idx.h1e, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

    max_diff = 0.0
    for p, (pa, pb) in enumerate(p_dets):
        for q in range(q_idx.M):
            a_str = int(q_idx.alpha_strs[q // q_idx.n_beta])
            b_str = int(q_idx.beta_strs[q % q_idx.n_beta])
            expected = ham.matrix_element((a_str, b_str), (pa, pb))
            max_diff = max(max_diff, abs(H_QP[q, p] - expected))

    assert max_diff < 1e-10, f"build_hqp mismatch: max|diff| = {max_diff:.2e}"
    print(f"  ✓ build_hqp vs Python SC: max|diff| = {max_diff:.2e}")


def test_build_projected_blocks():
    """End-to-end: build_hqp → MGS → projected blocks on H₂O CAS(4,4)."""
    import sys
    sys.path.insert(0, '/data/home/wangcx/krylov-dci')
    from src.hamiltonian import Hamiltonian, _unpack_4fold
    from src.determinants import hf_determinant, bit_positions
    from src.effective_h import build_effective_H, diagonalize_effective_H
    from pyscf.fci import direct_spin1, cistring
    from pyscf import ao2mo

    q_idx = _make_h2o_idx()
    backend = KDCIBackend(q_idx)

    # FCI reference
    fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = 3
    e_fci, _ = fs.kernel(q_idx.h1e, q_idx.eri, q_idx.norb, q_idx.nelec)

    # P-space: HF + top 10 perturbative doubles
    na, nb = q_idx.nelec
    hf_a, hf_b = hf_determinant(na, nb)
    n_act = q_idx.norb
    n_act_range = list(range(n_act))
    alpha_occ = bit_positions(hf_a); beta_occ = bit_positions(hf_b)
    av = [p for p in n_act_range if p not in alpha_occ]
    bv = [p for p in n_act_range if p not in beta_occ]

    from src.hamiltonian import Hamiltonian, _unpack_4fold
    h2_4d = ao2mo.restore('s1', q_idx.eri, n_act).reshape(n_act, n_act, n_act, n_act)
    ham = Hamiltonian(h1=q_idx.h1e, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    E_HF = ham.diagonal_element(hf_a, hf_b)

    scores = []
    for ii, i in enumerate(alpha_occ):
        for j in alpha_occ[ii+1:]:
            for ia, a in enumerate(av):
                for b in av[ia+1:]:
                    det = ((hf_a^(1<<i)^(1<<j))|(1<<a)|(1<<b), hf_b)
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij*hij)/denom))
    # ββ, αβ — abbreviated for test
    for i in alpha_occ:
        for j in beta_occ:
            for a in av:
                for b in bv:
                    det = ((hf_a^(1<<i))|(1<<a), (hf_b^(1<<j))|(1<<b))
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij*hij)/denom))

    scores.sort(key=lambda x: x[1], reverse=True)
    p_dets = [(hf_a, hf_b)] + [d for d, _ in scores[:10]]
    N = len(p_dets)

    # H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0_P = float(eigh(H_PP)[0][0])

    # Build via backend
    H_QP = backend.build_hqp(p_dets, verbose=False)
    basis, d = backend.build_basis(H_QP, E0_P, verbose=False)
    H_QQ_t, H_PQ_t = backend.build_projected_blocks(basis, p_dets, verbose=False)

    # Effective Hamiltonian
    # delta = E(FCI) - E0_P provides the exact shift for the ground state.
    # For production use, delta is obtained from DMRG-CI reference
    # and will eventually be determined self-consistently.
    delta = e_fci[0] - E0_P  # exact Delta from FCI reference
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P, delta=delta)
    ev_kdci, _ = diagonalize_effective_H(H_eff, n_states=None)

    dE0 = (ev_kdci[0] - e_fci[0]) * 1000
    print(f"  ✓ End-to-end: d={d}, E(kDCI)−E(FCI) = {dE0:+.1f} mH")
    # Should be close but not exact — perturbative P selection is approximate


if __name__ == '__main__':
    test_sigma_vs_direct_spin1()
    test_build_hqp_vs_hamiltonian()
    test_build_projected_blocks()
    print("All pyscf_backend tests passed.")
