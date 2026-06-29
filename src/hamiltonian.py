"""
Hamiltonian matrix construction via Slater-Condon rules.

Implements the direct CI approach: Hamiltonian matrix elements between
Slater determinants are computed on-the-fly using one- and two-electron
integrals from PySCF.

References:
  - Slater, Phys. Rev. 34, 1293 (1929)
  - Condon, Phys. Rev. 36, 1121 (1930)
  - Szabo & Ostlund, "Modern Quantum Chemistry", Ch. 2.3
  - Knowles & Handy, Comput. Phys. Commun. 54, 75 (1989)
"""

from typing import Tuple, Optional
import numpy as np
from pyscf import gto, scf, ao2mo
from determinants import (
    count_bits, bit_positions,
    excitation_level, find_excitations,
    excitation_phase_alpha, excitation_phase,
)


class Hamiltonian:
    """Molecular Hamiltonian in a Slater determinant basis.

    Stores one- and two-electron integrals in the molecular orbital (MO)
    basis and provides methods to compute matrix elements between any
    pair of Slater determinants.

    Attributes:
        n_orb:      Number of spatial molecular orbitals.
        h1:         One-electron integrals (n_orb × n_orb), chemist's notation.
        h2:         Two-electron integrals (n_orb × n_orb × n_orb × n_orb),
                    chemist's notation (ij|kl).
        E_nuc:      Nuclear repulsion energy.
        E_HF:       Hartree-Fock reference energy.
    """

    def __init__(self, h1: np.ndarray, h2: np.ndarray,
                 E_nuc: float = 0.0, E_HF: float = 0.0):
        """Initialize with MO-basis integrals.

        Args:
            h1:    One-electron integrals (i|h|j) in chemist's notation,
                   shape (n_orb, n_orb).
            h2:    Two-electron integrals (ij|kl) in chemist's notation.
                   Shape can be (n_orb, n_orb, n_orb, n_orb) or
                   a lower-dimensional packed form — will be stored as 4D.
            E_nuc: Nuclear repulsion energy.
            E_HF:  Hartree-Fock reference energy.
        """
        self.n_orb = h1.shape[0]
        self.h1 = h1
        self.E_nuc = E_nuc
        self.E_HF = E_HF

        # Ensure h2 is 4-index
        if h2.ndim == 4:
            self.h2 = h2
        elif h2.ndim == 2:
            # PySCF ao2mo output: (n_orb*(n_orb+1)/2, n_orb*(n_orb+1)/2)
            # Expand to 4-index for convenience
            self.h2 = _unpack_4fold(h2, self.n_orb)
        else:
            raise ValueError(f"h2 must be 2D (packed) or 4D, got {h2.ndim}D")

    # ------------------------------------------------------------------
    # Diagonal matrix element
    # ------------------------------------------------------------------

    def diagonal_element(self, alpha_str: int, beta_str: int) -> float:
        """Compute <D|H|D> for a single determinant.

        E_D = sum_i h_{ii} + 1/2 sum_{i!=j} (J_{ij} - K_{ij})
        where i,j run over occupied spin-orbitals.

        For a determinant with alpha and beta strings:
          E = sum_p h1[p,p] * occ_p + 1/2 sum_{p,q} (pq|pq) * occ_p * occ_q
            - 1/2 sum_{p,q, same spin} (pq|qp) * occ_p * occ_q

        where occ_p = occupation number of spatial orbital p (0, 1, or 2).

        Args:
            alpha_str, beta_str: Bit strings.

        Returns:
            Diagonal Hamiltonian matrix element (includes E_nuc).
        """
        e = 0.0

        # Collect occupied spatial orbitals with spin info
        alpha_occ = bit_positions(alpha_str)
        beta_occ = bit_positions(beta_str)

        # One-electron contribution
        for p in alpha_occ:
            e += self.h1[p, p]
        for p in beta_occ:
            e += self.h1[p, p]

        # Two-electron contribution
        # For each pair of spatial orbitals (p, q):
        #   J_{pq} = (pq|pq) contributes for all occupied pairs regardless of spin
        #   K_{pq} = (pq|qp) contributes only for same-spin pairs

        all_occ = [(p, 'a') for p in alpha_occ] + [(p, 'b') for p in beta_occ]
        n_occ = len(all_occ)

        for i_idx in range(n_occ):
            for j_idx in range(n_occ):
                p, spin_p = all_occ[i_idx]
                q, spin_q = all_occ[j_idx]

                if i_idx == j_idx:
                    continue  # self-interaction: J_{pp} = K_{pp}, cancels

                # Coulomb: (pq|pq) * 1/2
                e += 0.5 * self.h2[p, p, q, q]

                # Exchange: only if same spin
                if spin_p == spin_q:
                    e -= 0.5 * self.h2[p, q, q, p]

        return e + self.E_nuc

    # ------------------------------------------------------------------
    # Off-diagonal matrix element via Slater-Condon rules
    # ------------------------------------------------------------------

    def matrix_element(self, det1: Tuple[int, int],
                       det2: Tuple[int, int]) -> float:
        """Compute <det1|H|det2> using Slater-Condon rules.

        Only non-zero for determinants differing by 0, 1, or 2 spin-orbitals
        (Slater-Condon rules I, II, III). Returns 0 for higher excitations.

        Args:
            det1: (alpha_str, beta_str) for bra.
            det2: (alpha_str, beta_str) for ket.

        Returns:
            <det1|H|det2>.
        """
        level = excitation_level(det1, det2)

        if level == 0:
            # Rule I: Diagonal
            return self.diagonal_element(det1[0], det1[1])

        elif level == 1:
            # Rule II: Single excitation
            return self._sc_rule_ii(det1, det2)

        elif level == 2:
            # Rule III: Double excitation
            return self._sc_rule_iii(det1, det2)

        else:
            # Rule IV: Zero for triple and higher excitations
            return 0.0

    def _sc_rule_ii(self, det1: Tuple[int, int],
                    det2: Tuple[int, int]) -> float:
        """Slater-Condon rule for single excitations.

        <D|H|D_i^a> = Gamma * [h_{ia} + sum_k ((ik|ak) - (ik|ka)) * delta_same_spin]

        where i→a is the single excitation, k runs over occupied orbitals
        in the bra determinant, and Gamma is the phase factor.

        Args:
            det1: Bra determinant.
            det2: Ket determinant.

        Returns:
            <det1|H|det2>.
        """
        a1, b1 = det1
        a2, b2 = det2

        holes, particles = find_excitations(det1, det2)

        if len(holes) != 1 or len(particles) != 1:
            return 0.0

        # Extract source and destination
        i, spin_i = holes[0]
        a, spin_a = particles[0]

        # Phase factor
        if spin_i == 'alpha':
            phase = excitation_phase_alpha(a1, i, a)
        else:
            phase = excitation_phase_alpha(b1, i, a)

        # One-electron part
        result = self.h1[i, a]

        # Two-electron part: sum over occupied orbitals in bra
        alpha_occ = bit_positions(a1)
        beta_occ = bit_positions(b1)

        # Coulomb-exchange contributions from occupied spin-orbitals
        # Sum over j ∈ D, j ≠ (i, σ_i): (ia|jj) - δ_{σ_i,σ_j} (ij|ja)
        for p in alpha_occ:
            if spin_i == 'alpha' and p == i:
                continue  # Exclude hole spin-orbital
            result += self.h2[i, a, p, p]  # (ia|pp) Coulomb
            if spin_i == 'alpha':
                result -= self.h2[i, p, p, a]  # (ip|pa) Exchange
        for p in beta_occ:
            if spin_i == 'beta' and p == i:
                continue  # Exclude hole spin-orbital
            result += self.h2[i, a, p, p]  # (ia|pp) Coulomb
            if spin_i == 'beta':
                result -= self.h2[i, p, p, a]  # (ip|pa) Exchange

        return phase * result

    def _sc_rule_iii(self, det1: Tuple[int, int],
                     det2: Tuple[int, int]) -> float:
        """Slater-Condon rule for double excitations.

        <D|H|D_{ij}^{ab}> = Gamma * [(ia|jb) - (ib|ja) * delta_same_spin]

        where i→a and j→b are the two single excitations, and Gamma
        is the overall phase factor.

        Note: There is no one-electron contribution for double excitations.

        Args:
            det1: Bra determinant.
            det2: Ket determinant.

        Returns:
            <det1|H|det2>.
        """
        a1, b1 = det1

        holes, particles = find_excitations(det1, det2)

        if len(holes) != 2 or len(particles) != 2:
            return 0.0

        i, spin_i = holes[0]
        j, spin_j = holes[1]
        a, spin_a = particles[0]
        b, spin_b = particles[1]

        # Phase factor: product of phases for the two single excitations
        # applied sequentially. We compute the overall phase by comparing
        # the bit strings before and after.

        # Apply excitation i→a first, then j→b
        from determinants import apply_single_excitation

        # Try both orderings and take the one that works
        tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, i, a, spin_i)
        if tmp_a != 0 or tmp_b != 0:
            final_a, final_b, ph2 = apply_single_excitation(tmp_a, tmp_b, j, b, spin_j)
            if final_a != 0 or final_b != 0:
                phase = ph1 * ph2
            else:
                # Try reverse order
                tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, j, b, spin_j)
                final_a, final_b, ph2 = apply_single_excitation(tmp_a, tmp_b, i, a, spin_i)
                phase = ph1 * ph2
        else:
            tmp_a, tmp_b, ph1 = apply_single_excitation(a1, b1, j, b, spin_j)
            final_a, final_b, ph2 = apply_single_excitation(tmp_a, tmp_b, i, a, spin_i)
            phase = ph1 * ph2

        # Two-electron integral: (ia|jb) - (ib|ja) * delta(spin_i, spin_j)
        # Using chemist's notation: (pq|rs) = ∫ p*(1)q(1) r*(2)s(2) / r12
        result = self.h2[i, a, j, b]

        # Exchange contribution: only if i and j have the same spin
        if spin_i == spin_j:
            result -= self.h2[i, b, j, a]

        return phase * result

    # ------------------------------------------------------------------
    # Sigma-vector: H|Psi> for direct CI
    # ------------------------------------------------------------------

    def sigma_vector(self, c_vec: np.ndarray,
                     dets: list) -> np.ndarray:
        """Compute sigma = H @ c for a CI wavefunction.

        This is the core operation for iterative diagonalization (Davidson).
        Computed on-the-fly without storing the full H matrix.

        Args:
            c_vec: CI coefficient vector, shape (n_det,).
            dets:  List of determinants (alpha_str, beta_str) in the
                   same order as c_vec.

        Returns:
            sigma = H @ c_vec, shape (n_det,).
        """
        n_det = len(dets)
        sigma = np.zeros(n_det)

        for idx_I, detI in enumerate(dets):
            if abs(c_vec[idx_I]) < 1e-14:
                continue

            # Diagonal contribution
            sigma[idx_I] += self.diagonal_element(detI[0], detI[1]) * c_vec[idx_I]

            # Off-diagonal: loop over all determinants J
            # (This is O(N^2) — full matrix. For large CI spaces,
            #  use screening or direct-CI sigma-vector construction.)
            for idx_J, detJ in enumerate(dets):
                if idx_I >= idx_J:
                    continue  # Only upper triangle, add hermitian conjugate

                h_ij = self.matrix_element(detI, detJ)
                if abs(h_ij) > 1e-14:
                    sigma[idx_I] += h_ij * c_vec[idx_J]
                    sigma[idx_J] += h_ij * c_vec[idx_I]  # Hermitian

        return sigma

    # ------------------------------------------------------------------
    # Full Hamiltonian matrix (for small spaces only)
    # ------------------------------------------------------------------

    def build_full_matrix(self, dets: list) -> np.ndarray:
        """Build the full Hamiltonian matrix in the determinant basis.

        WARNING: O(N_det^2) memory. Only for small test systems.

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
    """Build a Hamiltonian from a PySCF molecule and mean-field calculation.

    If no mean-field is provided, runs RHF. Integrals are transformed to
    the MO basis using the provided MO coefficients.

    Args:
        mol:      PySCF Mole object.
        mf:       PySCF RHF object (optional; computed if not provided).
        mo_coeff: MO coefficient matrix (optional; uses mf.mo_coeff if not).

    Returns:
        Hamiltonian object with MO-basis integrals.
    """
    if mf is None:
        mf = scf.RHF(mol)
        mf.kernel()

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    n_orb = mo_coeff.shape[1]

    # One-electron integrals in AO basis → MO basis
    h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    h1_mo = mo_coeff.T @ h1_ao @ mo_coeff

    # Two-electron integrals in AO basis → MO basis
    # (ij|kl) = sum_{mu,nu,lambda,sigma} C_{mu i} C_{nu j} (mu nu|lambda sigma) C_{lambda k} C_{sigma l}
    # PySCF's ao2mo does this efficiently
    eri_ao = mol.intor('int2e')
    eri_mo_packed = ao2mo.full(eri_ao, mo_coeff)

    # Expand to 4-index
    h2_mo = _unpack_4fold(eri_mo_packed, n_orb)

    return Hamiltonian(
        h1=h1_mo,
        h2=h2_mo,
        E_nuc=mol.energy_nuc(),
        E_HF=mf.e_tot
    )


def _unpack_4fold(packed: np.ndarray, n_orb: int) -> np.ndarray:
    """Unpack PySCF's 2D integral format to 4-index.

    Uses PySCF's ao2mo.restore to expand from packed triangular
    format to full 4-index array in chemist's notation (ij|kl).

    Args:
        packed: 2D array of packed integrals from ao2mo.full().
        n_orb:  Number of spatial orbitals.

    Returns:
        4D array h2[i, j, k, l] in chemist's notation.
    """
    from pyscf import ao2mo
    # ao2mo.restore expands the packed 2D format to 4-index
    # 's1' = full symmetry (no index permutation symmetry assumed)
    h2 = ao2mo.restore('s1', packed, n_orb)
    # Reshape from (n_orb, n_orb, n_orb, n_orb) in Fortran-like order
    # restore returns a 4-index tensor with shape (n_orb, n_orb, n_orb, n_orb)
    return h2.reshape(n_orb, n_orb, n_orb, n_orb)


# ============================================================================
# Tests
# ============================================================================

def test_h2_sto3g_hamiltonian():
    """Build the full H matrix for H2/STO-3G and check against FCI."""
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74',
        basis='sto-3g',
        verbose=0
    )
    mf = scf.RHF(mol)
    mf.kernel()

    ham = from_pyscf(mol, mf)
    assert ham.n_orb == 2

    # Generate all determinants: 2 electrons, Ms=0
    from determinants import generate_determinants_ms
    dets = generate_determinants_ms(n_orb=2, n_elec=2, ms=0)

    # 4 determinants for H2/STO-3G
    assert len(dets) == 4

    H = ham.build_full_matrix(dets)

    # Diagonalize
    eigvals = np.linalg.eigvalsh(H)
    E_ground = eigvals[0]

    # Check against PySCF FCI
    from pyscf import fci
    h1_mo = mo_coeff_from_mf(mf).T @ (
        mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    ) @ mo_coeff_from_mf(mf)

    # Use our own FCI for verification (simpler path: compare to PySCF FCI)
    cisolver = fci.FCI(mf)
    E_fci, _ = cisolver.kernel()

    print(f"  E(HF)    = {mf.e_tot:.10f}")
    print(f"  E(FCI)   = {E_fci:.10f}")
    print(f"  E(manual)= {E_ground:.10f}")
    print(f"  Diff     = {abs(E_ground - E_fci):.2e}")

    assert abs(E_ground - E_fci) < 1e-9, \
        f"Manual FCI ({E_ground:.10f}) != PySCF FCI ({E_fci:.10f})"

    print("  ✓ H2/STO-3G Hamiltonian matches PySCF FCI.")


def mo_coeff_from_mf(mf):
    """Helper: get MO coefficients from a mean-field object."""
    return mf.mo_coeff


if __name__ == "__main__":
    test_h2_sto3g_hamiltonian()
    print("All Hamiltonian tests passed.")
