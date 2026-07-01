"""
Slater determinant representation using PySCF's cistring module.

Each determinant is a pair of integers (alpha, beta) where each bit
corresponds to a spatial orbital. All phase/sign operations delegate
to PySCF's cistring.cre_des_sign.

Conventions:
  - Orbital indices: 0-based, consistent with PySCF
  - Bit position 0 = LSB = orbital 0
  - (alpha_str, beta_str) are Python ints (converted from numpy int64)

PySCF references:
  - cistring.gen_strings4orblist  — generates all strings
  - cistring.cre_des_sign        — sign for i→a excitation
  - cistring.num_strings         — count of strings
"""

from typing import List, Tuple
import numpy as np
from pyscf.fci import cistring


# ============================================================================
# Bit-string utilities (kept — no PySCF equivalent for these)
# ============================================================================

def count_bits(x: int) -> int:
    """Population count (number of set bits)."""
    return x.bit_count()


def bit_positions(x: int) -> List[int]:
    """Return list of orbital indices where bits are set, sorted ascending.

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
    """Create a bit string with the given orbital indices set."""
    result = 0
    for orb in orbitals:
        result |= (1 << orb)
    return result


def occupation_number(alpha_str: int, beta_str: int, orb: int) -> int:
    """Return occupation number (0, 1, or 2) of spatial orbital `orb`."""
    return ((alpha_str >> orb) & 1) + ((beta_str >> orb) & 1)


# ============================================================================
# Single-excitation phase → PySCF cistring.cre_des_sign
# ============================================================================

def excitation_phase_alpha(alpha_str: int, i: int, a: int) -> int:
    """Compute the phase factor for exciting an alpha electron from i to a.

    Delegates to PySCF cistring.cre_des_sign(create, destroy, string).
    Note argument order: cre_des_sign(p=CREATE, q=DESTROY, string).

    Args:
        alpha_str: Alpha bit string before excitation.
        i:         Occupied orbital (source, to be destroyed).
        a:         Virtual orbital (destination, to be created).

    Returns:
        +1 or -1.
    """
    return cistring.cre_des_sign(a, i, int(alpha_str))


def excitation_phase(alpha_str: int, beta_str: int,
                     i: int, a: int, spin: str) -> int:
    """Compute the phase factor for a single excitation.

    Args:
        alpha_str, beta_str: Bit strings of the initial (bra) determinant.
        i:    Occupied orbital index (spatial).
        a:    Virtual orbital index (spatial).
        spin: 'alpha' or 'beta'.

    Returns:
        +1 or -1.
    """
    if spin == 'alpha':
        return cistring.cre_des_sign(a, i, int(alpha_str))
    else:
        return cistring.cre_des_sign(a, i, int(beta_str))


# ============================================================================
# Determinant generation → PySCF cistring.gen_strings4orblist
# ============================================================================

def _generate_alpha_beta_strings(n_orb: int, n_alpha: int, n_beta: int):
    """Generate all alpha and beta strings via PySCF cistring.

    Returns:
        (alpha_strs, beta_strs) as lists of Python ints.
    """
    orb_list = list(range(n_orb))
    alphas = cistring.gen_strings4orblist(orb_list, n_alpha)
    betas = cistring.gen_strings4orblist(orb_list, n_beta)
    return [int(a) for a in alphas], [int(b) for b in betas]


def generate_determinants(n_orb: int, n_alpha: int, n_beta: int) -> List[Tuple[int, int]]:
    """Generate all Slater determinants for given electron counts.

    Uses PySCF cistring.gen_strings4orblist for string generation,
    then Cartesian product.

    Args:
        n_orb:   Number of spatial orbitals.
        n_alpha: Number of alpha electrons.
        n_beta:  Number of beta electrons.

    Returns:
        List of (alpha_str, beta_str) tuples.
    """
    alpha_strs, beta_strs = _generate_alpha_beta_strings(n_orb, n_alpha, n_beta)
    dets = []
    for a_str in alpha_strs:
        for b_str in beta_strs:
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

    Uses PySCF cistring.cre_des_sign for the phase.

    Args:
        alpha_str, beta_str: Initial bit strings.
        i:    Source orbital (0-based spatial index).
        a:    Destination orbital.
        spin: 'alpha' or 'beta'.

    Returns:
        (alpha_str', beta_str', phase) where phase is +1 or -1.
        Returns (0, 0, 0) if excitation is invalid.
    """
    if spin == 'alpha':
        if not (alpha_str >> i) & 1:
            return 0, 0, 0
        if (alpha_str >> a) & 1:
            return 0, 0, 0
        phase = cistring.cre_des_sign(a, i, int(alpha_str))
        new_alpha = (alpha_str & ~(1 << i)) | (1 << a)
        return new_alpha, beta_str, int(phase)
    else:
        if not (beta_str >> i) & 1:
            return 0, 0, 0
        if (beta_str >> a) & 1:
            return 0, 0, 0
        phase = cistring.cre_des_sign(a, i, int(beta_str))
        new_beta = (beta_str & ~(1 << i)) | (1 << a)
        return alpha_str, new_beta, int(phase)


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
    a_new, b_new, phase1 = apply_single_excitation(
        alpha_str, beta_str, i, a, spin1)
    if a_new == 0 and b_new == 0:
        return 0, 0, 0

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

    a_diff = count_bits(a1 ^ a2)
    b_diff = count_bits(b1 ^ b2)

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

    a_holes = a1 & ~a2
    a_parts = a2 & ~a1

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

    The lowest n_alpha alpha orbitals and n_beta beta orbitals are occupied.

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
# Tests
# ============================================================================

def test_cre_des_sign_consistency():
    """Verify our excitation_phase_alpha matches PySCF cre_des_sign."""
    from pyscf.fci import cistring as _cs

    # Test case 1: i=0, a=1, string=0b01 (orb 0 occupied → orb 1)
    a_str = 0b01
    phase_ours = excitation_phase_alpha(a_str, 0, 1)
    assert phase_ours in (+1, -1), f"Phase should be ±1, got {phase_ours}"
    print(f"  cre_des_sign(0→1, 0b01) = {phase_ours:+d}")

    # Test case 2: i=0, a=2, string=0b11 (orb 0→2, orb 1 occupied between)
    a_str = 0b11
    phase = excitation_phase_alpha(a_str, 0, 2)
    assert phase == -1, f"Expected -1, got {phase}"
    print(f"  cre_des_sign(0→2, 0b11) = {phase:+d} ✓")

    # Test case 3: None between
    a_str = 0b01
    phase = excitation_phase_alpha(a_str, 0, 1)
    print(f"  cre_des_sign(0→1, 0b01) = {phase:+d}")

    print("  ✓ cre_des_sign consistency")


def test_generate_determinants_pyscf():
    """Verify determinant generation matches PySCF cistring."""
    n_orb, n_alpha, n_beta = 2, 1, 1
    dets = generate_determinants(n_orb, n_alpha, n_beta)
    assert len(dets) == 4, f"Expected 4 dets, got {len(dets)}"
    for a_str, b_str in dets:
        assert count_bits(a_str) == n_alpha
        assert count_bits(b_str) == n_beta

    # Verify against cistring directly
    from pyscf.fci import cistring as _cs
    alphas = [int(s) for s in _cs.gen_strings4orblist(range(n_orb), n_alpha)]
    betas = [int(s) for s in _cs.gen_strings4orblist(range(n_orb), n_beta)]
    expected = set()
    for a in alphas:
        for b in betas:
            expected.add((a, b))
    actual = set(dets)
    assert actual == expected, f"Mismatch: {actual - expected} vs {expected - actual}"
    print("  ✓ Generate determinants matches PySCF cistring")


def test_excitation_level():
    """Test that excitation_level correctly identifies excitation ranks."""
    det = (0b0011, 0b0011)
    assert excitation_level(det, det) == 0
    det2 = (0b0101, 0b0011)
    assert excitation_level(det, det2) == 1


if __name__ == "__main__":
    test_cre_des_sign_consistency()
    test_generate_determinants_pyscf()
    test_excitation_level()
    print("All determinant tests passed.")
