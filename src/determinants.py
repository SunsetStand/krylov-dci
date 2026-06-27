"""
Slater determinant representation using bit-string encoding.

Each determinant is represented as a pair of integers (alpha, beta),
where each bit position corresponds to a spatial orbital:
  bit j set → spin-orbital j is occupied.

Conventions:
  - Orbital indices: 0-based, consistent with PySCF
  - Bit position 0 = least significant bit = orbital 0
  - alpha_str, beta_str are Python integers (arbitrary precision)

References:
  - Knowles & Handy, Comput. Phys. Commun. 54, 75 (1989)
  - Olsen et al., J. Chem. Phys. 89, 2185 (1988)
"""

from typing import List, Tuple, Generator
import itertools
import numpy as np


# ============================================================================
# Bit-string utilities
# ============================================================================

def count_bits(x: int) -> int:
    """Population count (number of set bits). Uses Python's built-in bit_count."""
    return x.bit_count()


def bit_positions(x: int) -> List[int]:
    """Return list of orbital indices where bits are set, sorted ascending.

    Args:
        x: Bit string as integer.

    Returns:
        List of 0-based orbital indices.

    Example:
        bit_positions(0b10110) → [1, 2, 4]
    """
    positions = []
    idx = 0
    while x:
        if x & 1:
            positions.append(idx)
        x >>= 1
        idx += 1
    return positions


def create_bit_string(orbitals: List[int]) -> int:
    """Create a bit string with the given orbital indices set.

    Args:
        orbitals: List of 0-based orbital indices.

    Returns:
        Integer with bits set at the specified positions.
    """
    result = 0
    for orb in orbitals:
        result |= (1 << orb)
    return result


def occupation_number(alpha_str: int, beta_str: int, orb: int) -> int:
    """Return occupation number (0, 1, or 2) of spatial orbital `orb`.

    Args:
        alpha_str: Alpha electron bit string.
        beta_str:  Beta electron bit string.
        orb:       Spatial orbital index.

    Returns:
        0 = empty, 1 = singly occupied, 2 = doubly occupied.
    """
    return ((alpha_str >> orb) & 1) + ((beta_str >> orb) & 1)


# ============================================================================
# Single excitations between determinants
# ============================================================================

def excitation_phase_alpha(alpha_str: int, i: int, a: int) -> int:
    """Compute the phase factor for exciting an alpha electron from i to a.

    The phase is (-1)^k where k is the number of occupied alpha orbitals
    between i and a.

    Args:
        alpha_str: Alpha bit string before excitation.
        i:         Occupied orbital (source).
        a:         Virtual orbital (destination, must be empty in alpha_str).

    Returns:
        +1 or -1.
    """
    if i < a:
        # Count set bits in positions (i, a), exclusive
        mask = ((1 << (a - i - 1)) - 1) << (i + 1)
        n_between = count_bits(alpha_str & mask)
    else:
        mask = ((1 << (i - a - 1)) - 1) << (a + 1)
        n_between = count_bits(alpha_str & mask)
    return -1 if (n_between & 1) else 1


def excitation_phase(alpha_str: int, beta_str: int,
                     i: int, a: int, spin: str) -> int:
    """Compute the phase factor for a single excitation.

    Args:
        alpha_str, beta_str: Bit strings of the initial (bra) determinant.
        i:    Occupied spin-orbital index (absolute).
        a:    Virtual spin-orbital index (absolute).
        spin: 'alpha' or 'beta'.

    Returns:
        +1 or -1.
    """
    if spin == 'alpha':
        return excitation_phase_alpha(alpha_str, i, a)
    else:
        return excitation_phase_alpha(beta_str, i, a)


# ============================================================================
# Determinant generation
# ============================================================================

def generate_determinants(n_orb: int, n_alpha: int, n_beta: int) -> List[Tuple[int, int]]:
    """Generate all Slater determinants for given electron counts.

    Generates the full CI space with n_alpha alpha electrons and n_beta
    beta electrons in n_orb spatial orbitals.

    Args:
        n_orb:   Number of spatial orbitals.
        n_alpha: Number of alpha electrons.
        n_beta:  Number of beta electrons.

    Returns:
        List of (alpha_str, beta_str) tuples, one per determinant.
    """
    # All possible alpha strings: combinations of n_alpha bits from n_orb
    alpha_strings = []
    for combo in itertools.combinations(range(n_orb), n_alpha):
        alpha_strings.append(create_bit_string(list(combo)))

    # All possible beta strings: combinations of n_beta bits from n_orb
    beta_strings = []
    for combo in itertools.combinations(range(n_orb), n_beta):
        beta_strings.append(create_bit_string(list(combo)))

    # Cartesian product: each alpha string with each beta string
    dets = []
    for a_str in alpha_strings:
        for b_str in beta_strings:
            dets.append((a_str, b_str))

    return dets


def generate_determinants_ms(n_orb: int, n_elec: int, ms: int = 0) -> List[Tuple[int, int]]:
    """Generate determinants with a given total Ms value.

    Ms = (n_alpha - n_beta) / 2.

    Args:
        n_orb:  Number of spatial orbitals.
        n_elec: Total number of electrons.
        ms:     Ms quantum number (default 0 for singlet).

    Returns:
        List of (alpha_str, beta_str) tuples.
    """
    # n_alpha + n_beta = n_elec
    # n_alpha - n_beta = 2*ms
    n_alpha = (n_elec + 2 * ms) // 2
    n_beta = n_elec - n_alpha

    if n_alpha < 0 or n_beta < 0 or n_alpha > n_orb or n_beta > n_orb:
        raise ValueError(f"Invalid electron configuration: "
                         f"n_elec={n_elec}, ms={ms}, n_orb={n_orb}")

    return generate_determinants(n_orb, n_alpha, n_beta)


# ============================================================================
# Excitation operators
# ============================================================================

def apply_single_excitation(alpha_str: int, beta_str: int,
                            i: int, a: int, spin: str) -> Tuple[int, int, int]:
    """Apply a single excitation to a determinant.

    Args:
        alpha_str, beta_str: Initial bit strings.
        i:    Source orbital (0-based spatial index).
        a:    Destination orbital.
        spin: 'alpha' or 'beta'.

    Returns:
        (alpha_str', beta_str', phase) where phase is +1 or -1.
        Returns (0, 0, 0) if excitation is invalid (i not occupied or a occupied).
    """
    if spin == 'alpha':
        # Check validity: i must be occupied, a must be empty
        if not (alpha_str >> i) & 1:
            return 0, 0, 0
        if (alpha_str >> a) & 1:
            return 0, 0, 0
        phase = excitation_phase_alpha(alpha_str, i, a)
        new_alpha = (alpha_str & ~(1 << i)) | (1 << a)
        return new_alpha, beta_str, phase
    else:
        if not (beta_str >> i) & 1:
            return 0, 0, 0
        if (beta_str >> a) & 1:
            return 0, 0, 0
        phase = excitation_phase_alpha(beta_str, i, a)
        new_beta = (beta_str & ~(1 << i)) | (1 << a)
        return alpha_str, new_beta, phase


def apply_double_excitation(alpha_str: int, beta_str: int,
                            i: int, a: int, j: int, b: int,
                            spin1: str, spin2: str) -> Tuple[int, int, int]:
    """Apply a double excitation i→a, j→b to a determinant.

    Args:
        alpha_str, beta_str: Initial bit strings.
        i, j:  Source orbitals.
        a, b:  Destination orbitals.
        spin1: Spin of first excitation ('alpha' or 'beta').
        spin2: Spin of second excitation ('alpha' or 'beta').

    Returns:
        (alpha_str', beta_str', phase). Returns (0, 0, 0) if invalid.
    """
    # Apply first excitation
    a_new, b_new, phase1 = apply_single_excitation(
        alpha_str, beta_str, i, a, spin1)
    if a_new == 0 and b_new == 0:
        return 0, 0, 0

    # Apply second excitation on the intermediate determinant
    a_final, b_final, phase2 = apply_single_excitation(
        a_new, b_new, j, b, spin2)
    if a_final == 0 and b_final == 0:
        return 0, 0, 0

    return a_final, b_final, phase1 * phase2


# ============================================================================
# Determinant comparison utilities
# ============================================================================

def excitation_level(det1: Tuple[int, int], det2: Tuple[int, int]) -> int:
    """Determine the excitation level between two determinants.

    Args:
        det1, det2: (alpha_str, beta_str) tuples.

    Returns:
        0 = same determinant, 1 = single, 2 = double, >2 = higher.
    """
    a1, b1 = det1
    a2, b2 = det2

    # XOR gives bits that differ
    a_diff = count_bits(a1 ^ a2)
    b_diff = count_bits(b1 ^ b2)

    # Excitation level = (number of different alpha bits +
    #                     number of different beta bits) / 2
    return (a_diff + b_diff) // 2


def find_excitations(det1: Tuple[int, int],
                     det2: Tuple[int, int]) -> Tuple[List, List]:
    """Find the occupied→virtual pairs connecting two determinants.

    Used for computing Hamiltonian matrix elements via Slater-Condon rules.

    Args:
        det1: Initial determinant (alpha_str, beta_str).
        det2: Final determinant (alpha_str, beta_str).

    Returns:
        (holes, particles) where:
          holes[i]     = (orb_index, 'alpha'|'beta')
          particles[i] = (orb_index, 'alpha'|'beta')
    """
    a1, b1 = det1
    a2, b2 = det2

    # Alpha holes: bits set in det1 but not det2
    a_holes = a1 & ~a2
    a_parts = a2 & ~a1

    # Beta holes and particles
    b_holes = b1 & ~b2
    b_parts = b2 & ~b1

    holes = []
    particles = []

    for orb in bit_positions(a_holes):
        holes.append((orb, 'alpha'))
    for orb in bit_positions(a_parts):
        particles.append((orb, 'alpha'))
    for orb in bit_positions(b_holes):
        holes.append((orb, 'beta'))
    for orb in bit_positions(b_parts):
        particles.append((orb, 'beta'))

    return holes, particles


# ============================================================================
# Reference determinant
# ============================================================================

def hf_determinant(n_alpha: int, n_beta: int) -> Tuple[int, int]:
    """Create the Hartree-Fock reference determinant.

    The lowest n_alpha orbitals are doubly occupied, and the next
    (n_beta - n_alpha) orbitals are singly occupied (if n_beta > n_alpha).

    Args:
        n_alpha: Number of alpha electrons.
        n_beta:  Number of beta electrons.

    Returns:
        (alpha_str, beta_str) for the HF reference.
    """
    alpha_str = (1 << n_alpha) - 1
    beta_str = (1 << n_beta) - 1
    return alpha_str, beta_str


# ============================================================================
# Spatial symmetry filtering (optional)
# ============================================================================

def filter_by_occupation(dets: List[Tuple[int, int]],
                         n_orb: int,
                         allowed_occ: int) -> List[Tuple[int, int]]:
    """Filter determinants by occupation number pattern.

    Useful for selecting determinants within an active space.

    Args:
        dets:        List of determinants (alpha_str, beta_str).
        n_orb:       Number of spatial orbitals.
        allowed_occ: Bit mask: bit j=1 means orbital j is in the active space,
                     bit j=0 means it is frozen (must remain doubly occupied
                     or empty as in the reference).

    Returns:
        Filtered list.
    """
    result = []
    for a_str, b_str in dets:
        ok = True
        for orb in range(n_orb):
            if not (allowed_occ >> orb) & 1:
                # Frozen orbital: must match reference occupation
                # (which is typically doubly occupied or empty)
                pass  # Placeholder: implement if needed
        result.append((a_str, b_str))
    return result


# ============================================================================
# Tests (run with pytest)
# ============================================================================

def test_count_bits():
    assert count_bits(0) == 0
    assert count_bits(0b10110) == 3
    assert count_bits(0b11111111) == 8


def test_create_bit_string():
    assert create_bit_string([0, 2, 4]) == 0b10101


def test_generate_h2_sto3g():
    """H2/STO-3G: 2 electrons, 4 spin-orbitals (2 spatial orbitals)."""
    dets = generate_determinants(n_orb=2, n_alpha=1, n_beta=1)
    # C(2,1) * C(2,1) = 2 * 2 = 4 determinants
    assert len(dets) == 4
    # Verify all have correct electron counts
    for a_str, b_str in dets:
        assert count_bits(a_str) == 1
        assert count_bits(b_str) == 1


def test_hf_determinant():
    a_str, b_str = hf_determinant(5, 5)
    assert a_str == 0b11111
    assert b_str == 0b11111


def test_excitation_level():
    """Test that excitation_level correctly identifies excitation ranks."""
    # Same determinant
    det = (0b0011, 0b0011)
    assert excitation_level(det, det) == 0
    # Single excitation: alpha in orb 0→2
    det2 = (0b0101, 0b0011)
    assert excitation_level(det, det2) == 1


def test_phase_convention():
    """Verify that the excitation phase matches the standard convention.
    For a single excitation from orb 0→2 with orb 1 occupied,
    phase = +1 (no occupied orbitals between 0 and 2)."""
    alpha = 0b0011  # orbs 0,1 occupied
    phase = excitation_phase_alpha(alpha, 0, 2)
    # i=0, a=2, orbs 0,1 occupied
    # Between i+1=1 and a-1=1: orb 1 IS occupied → n_between=1 → phase=-1
    assert phase == -1


if __name__ == "__main__":
    test_count_bits()
    test_create_bit_string()
    test_generate_h2_sto3g()
    test_hf_determinant()
    test_excitation_level()
    test_phase_convention()
    print("All tests passed.")
