#!/usr/bin/env python3
"""
Decisive check: is _expand_schmidt_product_to_ci_matrix a correct resolution
of the identity?  Two tests:

  T1: orthonormality of expanded Schmidt product states.
  T2: completeness — does sum_{alpha,beta} sigma_alpha * |A_alpha B_alpha>
      reproduce the original CASCI CI vector?
  T3: single-element sigma vs brute-force, with intermediate diagnostics.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _expand_schmidt_product_to_ci_matrix,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs); n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

# ---- T1: orthonormality ----
print("=== T1: orthonormality of expanded Schmidt states ===")
max_offdiag = 0.0
min_diag = 1e9
for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]; r = sd['r']
    if r == 0: continue
    blk = partition[n_A]
    for a in range(r):
        for b in range(r):
            c1 = _expand_schmidt_product_to_ci_matrix(
                a, b, sd, blk, n_as, n_bs, n_occ,
                alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
            n1 = np.sum(c1 * c1)
            min_diag = min(min_diag, abs(n1 - 1.0))
            for c in range(r):
                for d in range(r):
                    if (a, b) == (c, d): continue
                    c2 = _expand_schmidt_product_to_ci_matrix(
                        c, d, sd, blk, n_as, n_bs, n_occ,
                        alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
                    ov = np.sum(c1 * c2)
                    max_offdiag = max(max_offdiag, abs(ov))
print(f"  min |norm-1| = {min_diag:.2e}")
print(f"  max |offdiag overlap| = {max_offdiag:.2e}")

# ---- T2: completeness (reconstruct CI from Schmidt) ----
print("\n=== T2: completeness (reconstruct CASCI CI vector) ===")
ci_recon = np.zeros((n_as, n_bs))
for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]; r = sd['r']
    if r == 0: continue
    blk = partition[n_A]
    for a in range(r):
        cm = _expand_schmidt_product_to_ci_matrix(
            a, a, sd, blk, n_as, n_bs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        ci_recon += sd['sigma'][a] * cm
# Compare to original CASCI vector
ci_orig = fcivec.reshape(n_as, n_bs)
overlap = np.sum(ci_orig * ci_recon)
norm_recon = np.linalg.norm(ci_recon)
norm_orig = np.linalg.norm(ci_orig)
err = np.linalg.norm(ci_orig - ci_recon)
print(f"  <orig|recon> = {overlap:.8f}")
print(f"  ||recon|| = {norm_recon:.8f}, ||orig|| = {norm_orig:.8f}")
print(f"  ||orig - recon|| = {err:.6e}")
