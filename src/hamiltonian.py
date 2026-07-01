"""
Hamiltonian matrix elements via Slater-Condon rules + PySCF.

The Hamiltonian class stores MO-basis integrals and provides:
  1. Diagonal matrix elements  (via PySCF selected_ci.make_hdiag)
  2. Off-diagonal matrix elements (Slater-Condon rules II/III)
  3. Sigma-vector H·|Ψ⟩  (via PySCF direct_spin1.contract_2e)

Slater-Condon rules II and III are kept because PySCF does not expose
a single-determinant-pair matrix element H[i,j]. All phase computations
within them now delegate to PySCF cistring.cre_des_sign.

References:
  - Slater, Phys. Rev. 34, 1293 (1929)
  - Condon, Phys. Rev. 36, 1121 (1930)
  - Szabo & Ostlund, "Modern Quantum Chemistry", Ch. 2.3
"""

from typing import Tuple, Optional, List
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci
try:
    from .determinants import (
        count_bits, bit_positions,
        excitation_level, find_excitations,
    )
except ImportError:
    from determinants import (
        count_bits, bit_positions,
        excitation_level, find_excitations,
    )


class Hamiltonian:
    """Molecular Hamiltonian in a Slater determinant basis.

    Attributes:
        n_orb:  Number of spatial MOs.
        h1:     One-electron integrals (n_orb × n_orb), chemist's notation.
        h2:     Two-electron integrals (n_orb × n_orb × n_orb × n_orb),
                chemist's notation (ij|kl).
        E_nuc:  Nuclear repulsion energy.
        E_HF:   Hartree-Fock reference energy.
    """

    def __init__(self, h1: np.ndarray, h2: np.ndarray,
                 E_nuc: float = 0.0, E_HF: float = 0.0):
        self.n_orb = h1.shape[0]
        self.h1 = h1
        self.E_nuc = E_nuc
        self.E_HF = E_HF

        if h2.ndim == 4:
            self.h2 = h2
        elif h2.ndim == 2:
            self.h2 = _unpack_4fold(h2, self.n_orb)
        else:
            raise ValueError(f"h2 must be 2D (packed) or 4D, got {h2.ndim}D")

    # ------------------------------------------------------------------
    # Diagonal elements → PySCF make_hdiag
    # ------------------------------------------------------------------

    def diagonal_element(self, alpha_str: int, beta_str: int) -> float:
        """Compute <D|H|D> for a single determinant.

        Hand-rolled Slater-Condon Rule I. PySCF's make_hdiag computes
        all alpha×beta pairs — wasteful for individual elements.
        For bulk, use diagonal_elements_bulk which calls make_hdiag.

        Args:
            alpha_str, beta_str: Bit strings.

        Returns:
            Diagonal Hamiltonian matrix element (includes E_nuc).
        """
        try:
            from .determinants import bit_positions
        except ImportError:
            from determinants import bit_positions

        e = 0.0
        alpha_occ = bit_positions(int(alpha_str))
        beta_occ = bit_positions(int(beta_str))

        # One-electron contribution
        for p in alpha_occ:
            e += self.h1[p, p]
        for p in beta_occ:
            e += self.h1[p, p]

        # Two-electron contribution
        all_occ = [(p, 'a') for p in alpha_occ] + [(p, 'b') for p in beta_occ]
        n_occ = len(all_occ)

        for i_idx in range(n_occ):
            for j_idx in range(n_occ):
                p, spin_p = all_occ[i_idx]
                q, spin_q = all_occ[j_idx]
                if i_idx == j_idx:
                    continue
                e += 0.5 * self.h2[p, p, q, q]        # Coulomb
                if spin_p == spin_q:
                    e -= 0.5 * self.h2[p, q, q, p]    # Exchange

        return e + self.E_nuc

    def diagonal_elements_bulk(self,
                                dets: List[Tuple[int, int]]) -> np.ndarray:
        """Compute <D_i|H|D_i> for a list of determinants.

        Uses PySCF selected_ci.make_hdiag. Note: make_hdiag computes
        hdiag for ALL combinations of unique alpha × beta strings.
        We pass unique strings, compute full grid, then index.

        For Q-space where all alpha×beta pairs exist, this is O(M) C-speed.

        Args:
            dets: List of (alpha_str, beta_str) tuples.

        Returns:
            Array of diagonal energies, length len(dets).
        """
        if not dets:
            return np.array([])

        n_orb = self.n_orb

        # Get unique alpha and beta strings
        alpha_unique = sorted(set(int(d[0]) for d in dets))
        beta_unique = sorted(set(int(d[1]) for d in dets))

        alpha_strs = np.array(alpha_unique, dtype=np.int64)
        beta_strs = np.array(beta_unique, dtype=np.int64)

        n_alpha = max(int(d[0]).bit_count() for d in dets)
        n_beta = max(int(d[1]).bit_count() for d in dets)

        from pyscf import ao2mo
        eri_packed = ao2mo.restore(1,
            self.h2.reshape(n_orb * n_orb, n_orb * n_orb), n_orb)

        # make_hdiag returns hdiag for all alpha×beta pairs: shape (na*nb,)
        hdiag_full = selected_ci.make_hdiag(
            self.h1, eri_packed, (alpha_strs, beta_strs),
            n_orb, (n_alpha, n_beta)
        )

        # Map back to input ordering
        na = len(alpha_unique)
        a_idx = {s: i for i, s in enumerate(alpha_unique)}
        b_idx = {s: i for i, s in enumerate(beta_unique)}

        result = np.zeros(len(dets))
        for k, (a_s, b_s) in enumerate(dets):
            ia = a_idx[int(a_s)]
            ib = b_idx[int(b_s)]
            result[k] = hdiag_full[ia * len(beta_unique) + ib]

        return result + self.E_nuc

    # ------------------------------------------------------------------
    # Off-diagonal matrix elements — Slater-Condon rules II & III
    # (Kept because PySCF does not expose H[i,j] for arbitrary det pairs)
    # ------------------------------------------------------------------

    def matrix_element(self, det1: Tuple[int, int],
                       det2: Tuple[int, int]) -> float:
        """Compute <det1|H|det2> using Slater-Condon rules.

        Returns 0 for excitation level > 2 (Slater-Condon rule IV).

        Args:
            det1: (alpha_str, beta_str) for bra.
            det2: (alpha_str, beta_str) for ket.

        Returns:
            <det1|H|det2>.
        """
        level = excitation_level(det1, det2)

        if level == 0:
            return self.diagonal_element(det1[0], det1[1])
        elif level == 1:
            return self._sc_rule_ii(det1, det2)
        elif level == 2:
            return self._sc_rule_iii(det1, det2)
        else:
            return 0.0

    def _sc_rule_ii(self, det1: Tuple[int, int],
                    det2: Tuple[int, int]) -> float:
        """Slater-Condon rule II: single excitations.

        <D|H|D_i^a> = phase · [h_{ia} + Σ_k ((ik|ak) - δ_{σσ_k} (ik|ka))]

        Phase is computed via PySCF cistring.cre_des_sign.
        """
        a1, b1 = det1
        holes, particles = find_excitations(det1, det2)

        if len(holes) != 1 or len(particles) != 1:
            return 0.0

        i, spin_i = holes[0]
        a, spin_a = particles[0]

        # Phase via PySCF cistring.cre_des_sign(create, destroy, string)
        if spin_i == 'alpha':
            phase = cistring.cre_des_sign(a, i, int(a1))
        else:
            phase = cistring.cre_des_sign(a, i, int(b1))

        # One-electron part
        result = self.h1[i, a]

        # Two-electron part: Σ_k (ia|kk) - δ_{σσ_k} (ik|ka)
        alpha_occ = bit_positions(a1)
        beta_occ = bit_positions(b1)

        for p in alpha_occ:
            if spin_i == 'alpha' and p == i:
                continue
            result += self.h2[i, a, p, p]       # (ia|pp) Coulomb
            if spin_i == 'alpha':
                result -= self.h2[i, p, p, a]   # (ip|pa) Exchange
        for p in beta_occ:
            if spin_i == 'beta' and p == i:
                continue
            result += self.h2[i, a, p, p]       # (ia|pp) Coulomb
            if spin_i == 'beta':
                result -= self.h2[i, p, p, a]   # (ip|pa) Exchange

        return int(phase) * result

    def _sc_rule_iii(self, det1: Tuple[int, int],
                     det2: Tuple[int, int]) -> float:
        """Slater-Condon rule III: double excitations.

        <D|H|D_{ij}^{ab}> = phase · [(ia|jb) - δ_{σ_i σ_j} (ib|ja)]

        Phase is computed by applying two single excitations sequentially.
        """
        try:
            from .determinants import apply_single_excitation
        except ImportError:
            from determinants import apply_single_excitation

        a1, b1 = det1
        holes, particles = find_excitations(det1, det2)

        if len(holes) != 2 or len(particles) != 2:
            return 0.0

        i, spin_i = holes[0]
        j, spin_j = holes[1]
        a, spin_a = particles[0]
        b, spin_b = particles[1]

        # Phase: apply two single excitations sequentially
        tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, i, a, spin_i)
        if tmp_a != 0 or tmp_b != 0:
            final_a, final_b, ph2 = apply_single_excitation(
                tmp_a, tmp_b, j, b, spin_j)
            if final_a != 0 or final_b != 0:
                phase = ph1 * ph2
            else:
                tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, j, b, spin_j)
                final_a, final_b, ph2 = apply_single_excitation(
                    tmp_a, tmp_b, i, a, spin_i)
                phase = ph1 * ph2
        else:
            tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, j, b, spin_j)
            final_a, final_b, ph2 = apply_single_excitation(
                tmp_a, tmp_b, i, a, spin_i)
            phase = ph1 * ph2

        # Two-electron integral: (ia|jb) - δ_{σ_i,σ_j} (ib|ja)
        result = self.h2[i, a, j, b]
        if spin_i == spin_j:
            result -= self.h2[i, b, j, a]

        return int(phase) * result

    # ------------------------------------------------------------------
    # Sigma-vector: H·|Ψ⟩  → PySCF contract_2e
    # ------------------------------------------------------------------

    def sigma_vector_pyscf(self, c_vec: np.ndarray,
                           alpha_strs: np.ndarray,
                           beta_strs: np.ndarray,
                           nelec: Tuple[int, int]) -> np.ndarray:
        """Compute sigma = H @ c using PySCF direct_spin1.contract_2e.

        This is the production sigma-vector: C-level speed, O(nnz) scaling.

        Args:
            c_vec:      CI coefficient vector, length n_det.
            alpha_strs: numpy int64 array of alpha strings, length n_det.
            beta_strs:  numpy int64 array of beta strings, length n_det.
            nelec:      (n_alpha, n_beta) electron counts.

        Returns:
            sigma = H @ c_vec, length n_det.
        """
        n_orb = self.n_orb

        # Build the CI vector in the correct shape
        # cistring gives us the full space dimension
        na_strs_full = cistring.gen_strings4orblist(range(n_orb), nelec[0])
        nb_strs_full = cistring.gen_strings4orblist(range(n_orb), nelec[1])

        # Build a full CI vector (sparse — only selected dets have non-zero)
        from pyscf.fci import cistring as _cs
        na = len(na_strs_full)
        nb = len(nb_strs_full)

        # Map each (alpha_str, beta_str) → flat index
        qa_idx = {int(s): i for i, s in enumerate(na_strs_full)}
        qb_idx = {int(s): i for i, s in enumerate(nb_strs_full)}

        ci_full = np.zeros((na, nb))
        for k, (a_s, b_s) in enumerate(zip(alpha_strs, beta_strs)):
            ia = qa_idx[int(a_s)]
            ib = qb_idx[int(b_s)]
            ci_full[ia, ib] = c_vec[k]

        # Contract
        from pyscf import ao2mo
        eri_packed = ao2mo.restore(1,
            self.h2.reshape(n_orb * n_orb, n_orb * n_orb), n_orb)

        sigma_full = direct_spin1.contract_2e(
            eri_packed, ci_full, n_orb, nelec
        ) + direct_spin1.contract_1e(self.h1, ci_full, n_orb, nelec)

        # Extract selected entries
        sigma = np.zeros(len(alpha_strs))
        for k, (a_s, b_s) in enumerate(zip(alpha_strs, beta_strs)):
            ia = qa_idx[int(a_s)]
            ib = qb_idx[int(b_s)]
            sigma[k] = sigma_full[ia, ib]

        return sigma

    # ------------------------------------------------------------------
    # Full matrix (testing only)
    # ------------------------------------------------------------------

    def build_full_matrix(self, dets: List[Tuple[int, int]]) -> np.ndarray:
        """Build the full Hamiltonian matrix (O(N²), testing only).

        Args:
            dets: List of determinants.

        Returns:
            H matrix, shape (n_det, n_det), symmetric.
        """
        n_det = len(dets)
        H = np.zeros((n_det, n_det))

        for i in range(n_det):
            H[i, i] = self.diagonal_element(dets[i][0], dets[i][1])
            for j in range(i + 1, n_det):
                h_ij = self.matrix_element(dets[i], dets[j])
                H[i, j] = h_ij
                H[j, i] = h_ij

        return H


# ============================================================================
# PySCF interface
# ============================================================================

def from_pyscf(mol: gto.Mole, mf: Optional[scf.hf.RHF] = None,
               mo_coeff: Optional[np.ndarray] = None) -> Hamiltonian:
    """Build a Hamiltonian from PySCF molecule + mean-field.

    Args:
        mol:      PySCF Mole object.
        mf:       PySCF RHF object (computed if not provided).
        mo_coeff: MO coefficient matrix (uses mf.mo_coeff if not).

    Returns:
        Hamiltonian with MO-basis integrals.
    """
    if mf is None:
        mf = scf.RHF(mol)
        mf.kernel()

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    n_orb = mo_coeff.shape[1]

    # One-electron integrals: AO → MO
    h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    h1_mo = mo_coeff.T @ h1_ao @ mo_coeff

    # Two-electron integrals: AO → MO
    eri_ao = mol.intor('int2e')
    eri_mo_packed = ao2mo.full(eri_ao, mo_coeff)
    h2_mo = _unpack_4fold(eri_mo_packed, n_orb)

    return Hamiltonian(
        h1=h1_mo,
        h2=h2_mo,
        E_nuc=mol.energy_nuc(),
        E_HF=mf.e_tot
    )


def _unpack_4fold(packed: np.ndarray, n_orb: int) -> np.ndarray:
    """Unpack PySCF 2D integral format to 4-index chemist's notation."""
    from pyscf import ao2mo
    h2 = ao2mo.restore('s1', packed, n_orb)
    return h2.reshape(n_orb, n_orb, n_orb, n_orb)


# ============================================================================
# Tests
# ============================================================================

def test_h2_sto3g_hamiltonian():
    """Build full H matrix for H₂/STO-3G, diagonalize, compare to PySCF FCI."""
    from pyscf import gto, scf
    from .determinants import generate_determinants_ms

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    ham = from_pyscf(mol, mf)
    assert ham.n_orb == 2

    dets = generate_determinants_ms(n_orb=2, n_elec=2, ms=0)
    assert len(dets) == 4

    H = ham.build_full_matrix(dets)
    eigvals = np.linalg.eigvalsh(H)
    E_ground = eigvals[0]

    # Compare to PySCF FCI
    norb = mol.nao
    nelec = (mol.nelec[0], mol.nelec[1])
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    from pyscf import ao2mo as a2m
    eri = a2m.restore(1, a2m.incore.full(mol.intor('int2e'), mf.mo_coeff), norb)
    E_fci, _ = direct_spin1.FCI().kernel(h1e, eri, norb, nelec,
                                          ecore=mf.energy_nuc())

    diff = abs(E_ground - E_fci)
    assert diff < 1e-10, f"FCI mismatch: {E_ground:.12f} vs {E_fci:.12f}"
    print(f"  ✓ H₂/STO-3G: E(our)={E_ground:.12f}, E(FCI)={E_fci:.12f}, "
          f"diff={diff:.2e}")


def test_diagonal_elements_bulk():
    """Verify bulk diagonal matches individual diagonal_element calls."""
    from pyscf import gto, scf
    from .determinants import generate_determinants_ms

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    ham = from_pyscf(mol, mf)
    dets = generate_determinants_ms(2, 2, ms=0)

    # Individual
    diag_indiv = np.array([ham.diagonal_element(a, b) for a, b in dets])
    # Bulk
    diag_bulk = ham.diagonal_elements_bulk(dets)

    assert np.allclose(diag_indiv, diag_bulk, atol=1e-12)
    print("  ✓ Bulk diag matches individual calls")


if __name__ == "__main__":
    test_h2_sto3g_hamiltonian()
    test_diagonal_elements_bulk()
    print("All Hamiltonian tests passed.")
