#!/usr/bin/env python3
"""
Decisive: does src/hamiltonian.py (hand-rolled Slater-Condon)
agree with PySCF contract_2e (sigma_full) for single determinant pairs?
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from src.hamiltonian import Hamiltonian

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs); n_bs = len(beta_strs)

# Build all determinant pairs (alpha_idx, beta_idx)
# Compare <d|H|d'> via two methods for several random pairs
rng = np.random.default_rng(42)

# Method 1: ham.matrix_element (Slater-Condon)
# Method 2: sigma_full — build CI matrix with 1.0 at ket, apply H, read bra
max_diff = 0.0
n_checked = 0
for trial in range(200):
    ia, ib = rng.integers(0, n_as), rng.integers(0, n_bs)
    ja, jb = rng.integers(0, n_as), rng.integers(0, n_bs)
    det1 = (int(alpha_strs[ia]), int(beta_strs[ib]))
    det2 = (int(alpha_strs[ja]), int(beta_strs[jb]))

    # Slater-Condon
    h_sc = ham.matrix_element(det1, det2)

    # sigma_full: H @ e_{ket}
    ci_ket = np.zeros((n_as, n_bs))
    ci_ket[ja, jb] = 1.0
    sigma = backend.sigma_full(ci_ket)
    h_sigma = sigma[ia, ib]

    diff = abs(h_sc - h_sigma)
    max_diff = max(max_diff, diff)
    n_checked += 1
    if diff > 1e-10:
        print(f"  MISMATCH: <d{ia},{ib}|H|d{ja},{jb}>: SC={h_sc:.10f}, sigma={h_sigma:.10f}, diff={diff:.2e}")
        if n_checked > 5:
            break

print(f"Checked {n_checked} random determinant pairs")
print(f"max |SC - sigma| = {max_diff:.2e}")
