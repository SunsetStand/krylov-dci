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


