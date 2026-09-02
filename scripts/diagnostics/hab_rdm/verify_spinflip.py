#!/usr/bin/env python3
"""
Confirm: Slater-Condon matrix_element misclassifies SPIN-FLIP determinant pairs
as level-1 excitations, producing spurious nonzero elements.

H conserves n_alpha and n_beta SEPARATELY.  A pair of determinants with
different (n_alpha, n_beta) must have <det1|H|det2> = 0 exactly.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from src.hamiltonian import Hamiltonian
from src.determinants import excitation_level, find_excitations, count_bits

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs); n_bs = len(beta_strs)

# Find a spin-flip pair: same total n_elec, but different (n_alpha, n_beta)
# det1: 2 alpha + 4 beta, det2: 3 alpha + 3 beta (both 6 electrons)
# Actually find any pair differing in n_alpha.
found = 0
for ia in range(n_as):
    for ib in range(n_bs):
        d1 = (int(alpha_strs[ia]), int(beta_strs[ib]))
        n1 = (d1[0].bit_count(), d1[1].bit_count())
        for ja in range(n_as):
            for jb in range(n_bs):
                d2 = (int(alpha_strs[ja]), int(beta_strs[jb]))
                n2 = (d2[0].bit_count(), d2[1].bit_count())
                if n1 != n2 and (n1[0]+n1[1]) == (n2[0]+n2[1]):
                    # cross-sector pair with same total electrons
                    lvl = excitation_level(d1, d2)
                    sc = ham.matrix_element(d1, d2)
                    # sigma reference for this pair
                    ci_ket = np.zeros((n_as, n_bs)); ci_ket[ja, jb] = 1.0
                    sig = backend.sigma_full(ci_ket)
                    sigma_val = sig[ia, ib]
                    print(f"det1=({d1[0]},{d1[1]}) nα,β={n1}")
                    print(f"det2=({d2[0]},{d2[1]}) nα,β={n2}")
                    print(f"  excitation_level = {lvl}")
                    print(f"  holes/particles  = {find_excitations(d1, d2)}")
                    print(f"  ham.matrix_element (Slater-Condon) = {sc:.10f}")
                    print(f"  sigma_full (correct)               = {sigma_val:.10f}")
                    print(f"  => spurious nonzero: {abs(sc) > 1e-10}")
                    found = 1
                    break
            if found: break
        if found: break
    if found: break

if not found:
    print("No cross-sector pair found (unexpected)")
