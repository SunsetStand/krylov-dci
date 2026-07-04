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
                    lindep_threshold: float = 1e-10,
                    verbose: bool = True) -> Tuple[np.ndarray, int]:
        """Build orthonormal basis via SVD on A^2-weighted H_QP.

        Layer 0 of the Krylov-dCI pipeline:
          1. L0 = A * H_QP,  A_q = 1/(E0_P - D_qq)
          2. T = A * L0 = A^2 * H_QP
          3. SVD(T), keep sigma > 1e-3 * sigma_max
          4. Return U (left singular vectors, orthonormal)

        This matches the original Krylov-dCI build_weighted_coupling +
        svd_truncate pipeline (phase6b/7/12).
        """
        M, N = H_QP.shape

        denom = E0_P - self.q_idx.hdiag
        A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)
        L0 = H_QP * A_q[:, np.newaxis]
        T = A_q[:, np.newaxis] * L0

        if verbose:
            t0 = time.perf_counter()

        U_svd, s, _ = np.linalg.svd(T, full_matrices=False)
        svd_threshold = 1e-3
        keep = s > svd_threshold * max(1.0, s[0])
        d = int(np.sum(keep))
        basis = U_svd[:, keep]

        if verbose:
            elapsed = time.perf_counter() - t0
            print("    SVD: %d -> %d vectors in %ds" % (N, d, int(elapsed)),
                  flush=True)

        return basis, d

    def build_projected_blocks(self, basis: np.ndarray,
                               p_dets: List[Tuple[int, int]],
                               H_QP: np.ndarray = None,
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

        # ── Step 3: H_{PQ̃} = H_QP^T @ basis (excludes P-P coupling) ──
        # sigma_all at P-det rows contains P-P contributions for
        # propagated basis vectors that have P-space components.
        # H_QP has zero P-det rows by construction, so H_QP^T @ basis
        # correctly isolates P-Q coupling only.
        if H_QP is not None:
            H_PQ_tilde = H_QP.T @ basis
        else:
            # Fallback: slice from sigma_all (legacy, may be wrong)
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

    
    # ═══════════════════════════════════════════════════════════════
    # Krylov propagation: B_{m+1} = MGS([B_m, compressed(AB_m)])
    # ═══════════════════════════════════════════════════════════════

    def propagate_basis(self, basis: np.ndarray, E0_P: float,
                        lindep_threshold: float = 1e-10,
                        svd_threshold: float = 1e-3,
                        verbose: bool = True) -> Tuple[np.ndarray, int]:
        """Propagate Krylov basis: B_{m+1} = MGS([B_m, SVD(A*H_O'*B_m)]).

        1. X_k = A * H_O' * b_k = A * (H_QQ * b_k - D_QQ * b_k)
        2. T = A * X  (re-weight for resolvent importance)
        3. SVD(T), keep sigma > svd_threshold * sigma_max
        4. MGS(U_trunc) against existing basis

        Delta is NOT used in the propagation (B = H_O'). The energy
        shift enters only in the final H^eff resolvent and the
        self-consistent iteration. This avoids numerical instability
        from the A * Delta product amplification.
        """
        M, r = basis.shape

        # A = (E0_P - D_QQ)^{-1}
        denom = E0_P - self.q_idx.hdiag
        A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)

        if verbose:
            t0 = time.perf_counter()
            print(f"    Propagate: r={r}...", flush=True)

        # Step 1: X = A * H_O' * B_m
        propagated = []
        for k in range(r):
            b_k = basis[:, k]
            sigma_k = self.sigma_full(
                self.q_idx.to_ci_matrix(b_k)
            ).reshape(-1)
            # H_O' * b_k = H_QQ * b_k - D_QQ * b_k
            residual = sigma_k - self.q_idx.hdiag * b_k
            x_k = A_q * residual
            propagated.append(x_k)

        prop_mat = np.column_stack(propagated)  # (M, r)

        # Step 2-3: T = A * X, SVD truncation
        T = A_q[:, np.newaxis] * prop_mat
        U_svd, s, _ = np.linalg.svd(T, full_matrices=False)
        keep = s > svd_threshold * max(1.0, s[0])
        n_keep = int(np.sum(keep))
        U_trunc = U_svd[:, keep]  # (M, n_keep)

        # Step 4: MGS against existing basis
        basis_list = [basis[:, j] for j in range(r)]
        new_count = 0
        for k in range(U_trunc.shape[1]):
            v = U_trunc[:, k].copy()
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
            print(f"    Propagate: {r} -> {d_new} vectors "
                  f"(+{added} new, SVD {n_keep}/{r}) in {elapsed:.0f}s",
                  flush=True)

        return basis_new, d_new

    def build_krylov_layers(self, H_QP: np.ndarray, E0_P: float,
                            m_max: int = 0,
                            lindep_threshold: float = 1e-10,
                            svd_threshold: float = 1e-3,
                            verbose: bool = True
                            ) -> Tuple[np.ndarray, int, List[int]]:
        """Build Krylov basis up to order m_max.

        Layer 0: B_0 = MGS(A * H_QP)
        Layer m:  B_m = MGS([B_{m-1}, SVD(A * H_O' * B_{m-1})])

        Returns:
            (basis, d_total, d_per_layer): final basis, total dim,
                                           dimensions per layer.
        """
        if verbose and m_max > 0:
            t0 = time.perf_counter()
            print(f"Building Krylov layers m=0..{m_max}...", flush=True)

        # Layer 0
        basis, d0 = self.build_basis(H_QP, E0_P,
                                     lindep_threshold=lindep_threshold,
                                     verbose=verbose)
        d_per_layer = [d0]

        for m in range(1, m_max + 1):
            basis, d_m = self.propagate_basis(
                basis, E0_P,
                lindep_threshold=lindep_threshold,
                svd_threshold=svd_threshold,
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

