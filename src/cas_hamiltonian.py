#!/usr/bin/env python3
"""
CAS-only determinant generation and frozen-core Hamiltonian construction.

For N₂/cc-pVDZ (28 orbitals, 14 electrons), full FCI has ~10^12 dets —
impossible to enumerate. This module generates only CAS-space determinants
and builds effective Hamiltonians with frozen core.

Frozen core means core orbitals are always doubly occupied in every
determinant. The core contribution is folded into the effective one-electron
integrals of the active space.
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
    This avoids the combinatorial explosion of full FCI generation.

    Args:
        n_active_orb:  Number of active spatial orbitals.
        n_active_elec: Number of active electrons.
        ms:            2*Sz = n_alpha - n_beta. Default 0 (singlet).

    Returns:
        List of (alpha_str, beta_str) tuples, where bit strings encode
        occupations within the active space (0 .. n_active_orb-1).
    """
    from determinants import generate_determinants_ms
    return generate_determinants_ms(n_active_orb, n_active_elec, ms=ms)


def build_cas_hamiltonian(mol, mf, n_core_orb: int, n_active_orb: int):
    """Build an effective Hamiltonian for the CAS space with frozen core.

    Core orbitals (indices 0 .. n_core_orb-1) are frozen doubly occupied.
    Active orbitals (indices n_core_orb .. n_core_orb + n_active_orb - 1)
    receive modified one-electron integrals that include core contributions.

    The effective active-space Hamiltonian has:
        h1_eff[p,q] = h1[p,q] + Σ_{c∈core} [2*(pc|qc) - (pc|cq)]
        h2_eff = h2 in active space (unchanged)
        E_core = 2 Σ_c h1[c,c] + Σ_{c,d∈core} [2*(cc|dd) - (cd|dc)]

    Args:
        mol:          PySCF Mole object.
        mf:           PySCF RHF object.
        n_core_orb:   Number of frozen core spatial orbitals.
        n_active_orb: Number of active spatial orbitals.

    Returns:
        dict with keys:
          h1_eff: Active-space one-electron integrals (n_active_orb × n_active_orb)
          h2_eff: Active-space two-electron integrals (n_active_orb⁴)
          E_core: Core energy contribution (scalar)
          active_mo_coeff: MO coefficients for active orbitals (n_ao × n_active_orb)
          n_active_orb, n_active_elec
    """
    from pyscf import ao2mo
    from hamiltonian import _unpack_4fold

    n_ao = mol.nao
    n_orb_total = mf.mo_coeff.shape[1]
    n_active_elec = mol.nelec[0] + mol.nelec[1] - 2 * n_core_orb

    # MO coefficients
    mo_coeff = mf.mo_coeff
    core_mo = mo_coeff[:, :n_core_orb]
    active_mo = mo_coeff[:, n_core_orb:n_core_orb + n_active_orb]

    # ---- All integrals in AO basis ----
    h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    eri_ao = mol.intor('int2e')  # (μν|λσ) in chemist notation

    # ---- Core energy ----
    # Use ao2mo.incore with 2D arrays (n_ao, 1) for each MO.
    h1_core_mo = core_mo.T @ h1_ao @ core_mo
    eri_core_packed = ao2mo.incore.full(eri_ao, core_mo)
    eri_core_4d = _unpack_4fold(eri_core_packed, n_core_orb)
    
    E_core = 0.0
    for c in range(n_core_orb):
        E_core += 2.0 * h1_core_mo[c, c]
        for d in range(n_core_orb):
            E_core += 2.0 * eri_core_4d[c, c, d, d] - eri_core_4d[c, d, d, c]

    # ---- Effective one-electron integrals in active space ----
    # h1_eff[p,q] = h1[p,q] + Σ_{c∈core} [2*(pc|qc) - (pc|cq)]
    h1_active_mo = active_mo.T @ h1_ao @ active_mo
    h1_eff = h1_active_mo.copy()
    n_act = n_active_orb
    
    for c in range(n_core_orb):
        c_mo = core_mo[:, c:c+1]  # (n_ao, 1)
        for p in range(n_act):
            p_mo = active_mo[:, p:p+1]
            for q in range(n_act):
                q_mo = active_mo[:, q:q+1]
                eri_pcqc = ao2mo.incore.general(
                    eri_ao, [p_mo, c_mo, q_mo, c_mo], compact=False
                )[0, 0, 0, 0]
                eri_pccq = ao2mo.incore.general(
                    eri_ao, [p_mo, c_mo, c_mo, q_mo], compact=False
                )[0, 0, 0, 0]
                h1_eff[p, q] += 2.0 * eri_pcqc - eri_pccq

    # ---- Two-electron integrals in active space ----
    eri_active_packed = ao2mo.incore.full(eri_ao, active_mo)
    h2_eff = _unpack_4fold(eri_active_packed, n_act)

    return {
        'h1_eff': h1_eff,
        'h2_eff': h2_eff,
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
        CASCI total energy (includes core contribution).
    """
    from pyscf.fci.direct_nosym import FCI

    n_act = cas_data['n_active_orb']
    n_alpha = (n_active_elec + ms) // 2
    n_beta = (n_active_elec - ms) // 2

    solver = FCI()
    solver.verbose = 0
    E_casci, _ = solver.kernel(
        cas_data['h1_eff'], cas_data['h2_eff'], n_act,
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
    from hamiltonian import Hamiltonian
    return Hamiltonian(
        h1=cas_data['h1_eff'],
        h2=cas_data['h2_eff'],
        E_nuc=cas_data['E_core'],
        E_HF=0.0  # Not used for CAS HF reference
    )


# ============================================================================
# Test
# ============================================================================

def test_cas_hamiltonian_h2o():
    """Verify frozen-core CAS Hamiltonian matches CASCI for H₂O/STO-3G."""
    from pyscf import gto, scf
    import sys
    sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/krylov-dci/src')

    print("--- test_cas_hamiltonian_h2o ---")
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    # CAS(6,5): 2 core orbitals (O 1s, O 2s?), 5 active, 6 active electrons
    # H₂O: 10 e, 7 MOs. Core = 2 (O 1s, O 2s-like), Active = 5 (O 2p-like + H 1s ×2)
    cas = build_cas_hamiltonian(mol, mf, n_core_orb=2, n_active_orb=5)
    E_casci = compute_casci_energy(cas, 6, ms=0)
    print(f"  CAS(6,5) energy: {E_casci:.10f}")

    # Compare to PySCF CASCI with frozen core
    from pyscf import mcscf
    ncas = 5
    nelecas = 6
    mycas = mcscf.CASCI(mf, ncas, nelecas)
    mycas.frozen = 2  # Freeze first 2 orbitals as core
    mycas.kernel()
    E_casci_pyscf = mycas.e_tot
    print(f"  PySCF CASCI(6,5) energy: {E_casci_pyscf:.10f}")
    
    diff = abs(E_casci - E_casci_pyscf)
    print(f"  |diff| = {diff:.6e}")
    assert diff < 1e-8, f"CAS energy mismatch: {diff}"
    print("  ✓ Frozen-core CAS Hamiltonian matches PySCF CASCI")


if __name__ == "__main__":
    test_cas_hamiltonian_h2o()
    print("All CAS Hamiltonian tests passed.")
