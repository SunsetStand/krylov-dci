#!/usr/bin/env python3
"""
Diagnostic: check ecore, H_emb[0,0], and active-space FCI energy.
Minimal script — does NOT run dmSVD or full H^emb build.
Just checks integral convention and basic numbers.

Usage:
    python notebooks/debug_hemb_diag.py
"""
import sys, os
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

import numpy as np
from pyscf import gto, scf, mcscf

# ── N2/cc-pVDZ CAS(10,10), R=1.098, n_core=2 ──
mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

n_act = 10
n_core = 2
n_elec = (5, 5)

cas = mcscf.CASCI(mf, n_act, sum(n_elec))
cas.frozen = n_core
h1eff, ecore = cas.get_h1eff()
h2eff = cas.get_h2eff()
cas.kernel()
E_fci = cas.e_tot

print("=" * 60)
print("DIAGNOSTIC: N₂/cc-pVDZ CAS(10,10)")
print("=" * 60)
print(f"  E_fci (total)     = {E_fci:.12f} Ha")
print(f"  ecore (frozen+nuc)= {ecore:.12f} Ha")
print(f"  E_active (FCI)    = {E_fci - ecore:.12f} Ha")
print(f"  h1eff shape       = {h1eff.shape}")
print(f"  h2eff shape       = {h2eff.shape}")

# ── Build full H in determinant basis via PySCF make_hdiag ──
from pyscf.fci import cistring, selected_ci, direct_spin1

as_ = cistring.gen_strings4orblist(range(n_act), n_elec[0])
bs_ = cistring.gen_strings4orblist(range(n_act), n_elec[1])
na, nb = len(as_), len(bs_)
M = na * nb
print(f"  Full CAS dim      = {M:,} ({na} α × {nb} β)")

# PySCF hdiag (all determinants)
hdiag = selected_ci.make_hdiag(h1eff, h2eff, (as_, bs_), n_act, n_elec)
print(f"  min(hdiag)        = {hdiag.min():.12f} Ha")
print(f"  max(hdiag)        = {hdiag.max():.12f} Ha")
print(f"  mean(hdiag)       = {hdiag.mean():.12f} Ha")
print(f"  hdiag[0]          = {hdiag[0]:.12f} Ha  (HF-like det)")

# ── Check: sigma-vector on unit HF det vs manual diagonal ──
# HF determinant: first 5 orbitals occupied (α and β)
hf_a = sum(1 << i for i in range(5))
hf_b = sum(1 << i for i in range(5))

# Unit vector at HF det
ia = {int(s): i for i, s in enumerate(as_)}
ib = {int(s): i for i, s in enumerate(bs_)}
hf_idx = ia[hf_a] * nb + ib[hf_b]

unit = np.zeros(M)
unit[hf_idx] = 1.0
ci_mat = unit.reshape(na, nb)

# Sigma via PySCF absorb_h1e + contract_2e
h2e_eff = direct_spin1.absorb_h1e(h1eff, h2eff, n_act, n_elec, 0.5)
ci_with_strs = selected_ci._as_SCIvector(ci_mat.copy(), (as_, bs_))
sigma_mat = selected_ci.contract_2e(
    h2e_eff, ci_with_strs, n_act, n_elec,
    link_index=selected_ci._all_linkstr_index((as_, bs_), n_act, n_elec))
sigma_flat = sigma_mat.reshape(-1)

# H[HF, HF] = ⟨HF|H|HF⟩ = σ[HF_idx] (since σ = H·unit, and unit is at HF)
H_hfhf_pyscf = sigma_flat[hf_idx]

# Same via Hamiltonian class
from src.hamiltonian import Hamiltonian, _unpack_4fold
h2_4d = _unpack_4fold(h2eff, n_act)
ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
H_hfhf_manual = ham.diagonal_element(hf_a, hf_b)

print(f"\n  HF determinant:")
print(f"    PySCF σ-proj       = {H_hfhf_pyscf:.12f} Ha")
print(f"    Manual SC-rule     = {H_hfhf_manual:.12f} Ha")
print(f"    Difference         = {abs(H_hfhf_pyscf - H_hfhf_manual):.2e} Ha")
print(f"    hdiag[HF]          = {hdiag[hf_idx]:.12f} Ha")

# ── Also check with ecore included ──
ham_with_ecore = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=ecore, E_HF=0.0)
H_hfhf_with_ecore = ham_with_ecore.diagonal_element(hf_a, hf_b)
print(f"    Manual + ecore     = {H_hfhf_with_ecore:.12f} Ha")

# ── Check off-diagonal: pick a single excitation ──
# |HF_i^a⟩ where i=4 (HOMO), a=5 (LUMO), α spin
sd_a = hf_a ^ (1 << 4) | (1 << 5)
sd_b = hf_b
sd_idx = ia[sd_a] * nb + ib[sd_b]

# Unit vector at single excitation
unit2 = np.zeros(M)
unit2[sd_idx] = 1.0
ci_mat2 = unit2.reshape(na, nb)
ci_with_strs2 = selected_ci._as_SCIvector(ci_mat2.copy(), (as_, bs_))
sigma_mat2 = selected_ci.contract_2e(
    h2e_eff, ci_with_strs2, n_act, n_elec,
    link_index=selected_ci._all_linkstr_index((as_, bs_), n_act, n_elec))
sigma_flat2 = sigma_mat2.reshape(-1)

H_hfsd_pyscf = sigma_flat2[hf_idx]  # ⟨HF|H|sd⟩ from σ-projection
H_sdhf_pyscf = sigma_flat[sd_idx]   # ⟨sd|H|HF⟩ from σ-projection

H_hfsd_manual = ham.matrix_element((hf_a, hf_b), (sd_a, sd_b))

print(f"\n  Off-diagonal HF → single-excitation (α: 4→5):")
print(f"    PySCF σ-proj ⟨HF|H|sd⟩ = {H_hfsd_pyscf:.12f}")
print(f"    PySCF σ-proj ⟨sd|H|HF⟩ = {H_sdhf_pyscf:.12f}")
print(f"    Manual SC-rule         = {H_hfsd_manual:.12f}")
print(f"    |⟨HF|H|sd⟩ - ⟨sd|H|HF⟩| = {abs(H_hfsd_pyscf - H_sdhf_pyscf):.2e}")

# ── Summary ──
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  If PySCF σ ≈ Manual SC: integral convention is consistent ✓")
print(f"  If H[HF,HF] ≈ hdiag[HF]: hdiag matches σ-projection ✓")
print(f"  Expected active E ≈ {E_fci - ecore:.4f} Ha")
print(f"  H_emb eigenvalues should be near this value")
print(f"  (Not -31.6 Ha if H_emb is correctly constructed)")