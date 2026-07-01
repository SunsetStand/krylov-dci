#!/usr/bin/env python3
"""
CAS-only determinant generation and frozen-core Hamiltonian construction.

For N₂/cc-pVDZ (28 orbitals, 14 electrons), full FCI has ~10^12 dets —
impossible to enumerate. This module generates only CAS-space determinants
and builds effective Hamiltonians with frozen core.

Refactored (2026-07-01): Builds the frozen-core active-space Hamiltonian
by delegating to PySCF's mcscf.CASCI.get_h1eff/get_h2eff. This guarantees
exact agreement with PySCF's CASCI energy.
"""

import numpy as np
from typing import List, Tuple


def generate_cas_determinants(
    n_active_orb: int,
    n_active_elec: int,
    ms: int = 0
) -> List[Tuple[int, int]]:
    """Generate determinants within a CAS(n,m) active space.

    Only generates determinants for the active space, ignoring core
    (always doubly occupied) and virtual (always empty) orbitals.

    Args:
        n_active_orb:  Number of active spatial orbitals.
        n_active_elec: Number of active electrons.
        ms:            2*Sz = n_alpha - n_beta. Default 0 (singlet).

    Returns:
        List of (alpha_str, beta_str) tuples.
    """
    try:
        from .determinants import generate_determinants_ms
    except ImportError:
        from determinants import generate_determinants_ms
    return generate_determinants_ms(n_active_orb, n_active_elec, ms=ms)


def build_cas_hamiltonian(mol, mf, n_core_orb: int, n_active_orb: int):
    """Build an effective Hamiltonian for the CAS space with frozen core.

    DELEGATES to PySCF mcscf.CASCI.get_h1eff/get_h2eff to compute
    the frozen-core-corrected integrals. This guarantees exact agreement
    with PySCF's CASCI total energy.

    Args:
        mol:          PySCF Mole object.
        mf:           PySCF RHF object.
        n_core_orb:   Number of frozen core spatial orbitals.
        n_active_orb: Number of active spatial orbitals.

    Returns:
        dict with keys:
          h1_eff: Active-space one-electron integrals (n_active × n_active)
          h2_eff: Active-space two-electron integrals (n_active⁴)
          h2_packed: Packed 4-fold format for PySCF FCI solvers
          E_core: Core energy contribution (scalar, for FCI ecore parameter)
          active_mo_coeff: MO coefficients for active orbitals
          n_active_orb, n_active_elec
    """
    from pyscf import mcscf
    from pyscf import ao2mo

    n_active_elec = mol.nelec[0] + mol.nelec[1] - 2 * n_core_orb

    # Build a temporary CASCI object to get h1eff and ecore from PySCF
    cas = mcscf.CASCI(mf, n_active_orb, n_active_elec)
    cas.frozen = n_core_orb
    # Need to set mo_coeff for get_h1eff to work
    cas.mo_coeff = mf.mo_coeff

    # Get integrals from PySCF (guaranteed correct frozen-core treatment)
    h1_eff, E_core = cas.get_h1eff()
    h2_eff_packed = cas.get_h2eff()

    # 4D unpacked version for our Hamiltonian class
    from pyscf import ao2mo as a2m
    n_act = n_active_orb
    h2_eff_4d = a2m.restore('s1', h2_eff_packed, n_act).reshape(
        n_act, n_act, n_act, n_act)

    active_mo = mf.mo_coeff[:, n_core_orb:n_core_orb + n_active_orb]

    return {
        'h1_eff': h1_eff,
        'h2_eff': h2_eff_4d,
        'h2_packed': np.asarray(h2_eff_packed),
        'E_core': float(E_core),
        'active_mo_coeff': active_mo,
        'n_active_orb': n_act,
        'n_active_elec': n_active_elec,
    }


def compute_casci_energy(cas_data: dict, n_active_elec: int,
                         ms: int = 0) -> float:
    """Compute exact CASCI energy using PySCF FCI solver.

    Args:
        cas_data: Output of build_cas_hamiltonian.
        n_active_elec: Number of active electrons.
        ms: 2*Sz quantum number.

    Returns:
        CASCI total energy (includes core contribution from E_core).
    """
    from pyscf.fci.direct_nosym import FCI

    n_act = cas_data['n_active_orb']
    n_alpha = (n_active_elec + ms) // 2
    n_beta = (n_active_elec - ms) // 2

    solver = FCI()
    solver.verbose = 0
    E_casci, _ = solver.kernel(
        cas_data['h1_eff'], cas_data['h2_packed'], n_act,
        (n_alpha, n_beta), ecore=cas_data['E_core']
    )
    return E_casci


def build_hamiltonian_from_cas(cas_data: dict):
    """Build a Hamiltonian object from CAS data.

    Args:
        cas_data: Output of build_cas_hamiltonian.

    Returns:
        Hamiltonian object for the active space.
    """
    try:
        from .hamiltonian import Hamiltonian
    except ImportError:
        from hamiltonian import Hamiltonian
    return Hamiltonian(
        h1=cas_data['h1_eff'],
        h2=cas_data['h2_eff'],
        E_nuc=0.0,          # E_core already includes nuclear repulsion
        E_HF=0.0
    )


# ============================================================================
# Test
# ============================================================================

def test_cas_hamiltonian_h2o():
    """Verify frozen-core CAS Hamiltonian matches CASCI for H₂O/STO-3G."""
    from pyscf import gto, scf, mcscf
    import sys

    print("--- test_cas_hamiltonian_h2o (refactored) ---")
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    # CAS(5,6) with 2 frozen core
    cas_data = build_cas_hamiltonian(mol, mf, n_core_orb=2, n_active_orb=5)
    n_act_elec = cas_data['n_active_elec']
    E_our = compute_casci_energy(cas_data, n_act_elec, ms=0)
    print(f"  Our CASCI(5,6) frozen-core energy: {E_our:.10f}")

    # PySCF reference
    mycas = mcscf.CASCI(mf, 5, 6)
    mycas.frozen = 2
    mycas.kernel()
    E_pyscf = mycas.e_tot
    print(f"  PySCF CASCI(5,6) energy: {E_pyscf:.10f}")

    diff = abs(E_our - E_pyscf)
    print(f"  |diff| = {diff:.6e}")
    assert diff < 1e-8, f"CAS energy mismatch: {diff}"
    print("  ✓ Frozen-core CAS Hamiltonian matches PySCF CASCI (via get_h1eff)")


if __name__ == "__main__":
    test_cas_hamiltonian_h2o()
    print("All CAS Hamiltonian tests passed.")
