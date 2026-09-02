#!/usr/bin/env python3
"""Decisive: locate the sigma-vs-bruteforce discrepancy for element (A0B0 -> A1B0)."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _expand_schmidt_product_to_ci_matrix,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from src.hamiltonian import Hamiltonian

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs); n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

n_A = 3
sd = schmidt[n_A]; r = sd['r']; blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']; b_dets = blk['b_dets']

a_src, b_src, a_dst, b_dst = 0, 0, 1, 0

def prod_to_full(i_a, k_b):
    aA, aB = a_dets[i_a]; bA, bB = b_dets[k_b]
    return (aA | (bA << n_occ), aB | (bB << n_occ))

# ---- (1) sigma ref: <A1 B0 | H | A0 B0> ----
ci_ket = _expand_schmidt_product_to_ci_matrix(
    a_src, b_src, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
ci_bra = _expand_schmidt_product_to_ci_matrix(
    a_dst, b_dst, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
sigma = backend.sigma_full(ci_ket)
H_sigma = np.sum(ci_bra * sigma)
print(f"[1] sigma <A1B0|H|A0B0> = {H_sigma:.10f}")

# ---- (2) brute via ham.matrix_element: <A0B0|H|A1B0> ----
H_bf = 0.0
for i in range(len(a_dets)):
    ui = U[i, a_src]
    for k in range(len(b_dets)):
        vk = V[k, b_src]
        for j in range(len(a_dets)):
            uj = U[j, a_dst]
            for l in range(len(b_dets)):
                vl = V[l, b_dst]
                w = ui*vk*uj*vl
                if abs(w) < 1e-14: continue
                H_bf += w * ham.matrix_element(prod_to_full(i,k), prod_to_full(j,l))
print(f"[2] brute <A0B0|H|A1B0> = {H_bf:.10f}")

# ---- (3) brute using sigma_full on full-det basis (same ordering as (2)) ----
# Build a ket = sum U[i,a_src]V[k,b_src] |full(i,k)>, bra = sum U[j,a_dst]V[l,b_dst]|full(j,l)>
ci_ket3 = np.zeros((n_as, n_bs))
ci_bra3 = np.zeros((n_as, n_bs))
for i in range(len(a_dets)):
    for k in range(len(b_dets)):
        f = prod_to_full(i, k)
        ia = alpha_to_idx[f[0]]; ib = beta_to_idx[f[1]]
        ci_ket3[ia, ib] += U[i, a_src] * V[k, b_src]
        ci_bra3[ia, ib] += U[j, a_dst] * V[l, b_dst] if False else 0.0
# fill bra properly
for j in range(len(a_dets)):
    for l in range(len(b_dets)):
        f = prod_to_full(j, l)
        ia = alpha_to_idx[f[0]]; ib = beta_to_idx[f[1]]
        ci_bra3[ia, ib] += U[j, a_dst] * V[l, b_dst]

sig3 = backend.sigma_full(ci_ket3)
H_sigma3 = np.sum(ci_bra3 * sig3)
print(f"[3] sigma on manually-expanded (ket=A0B0, bra=A1B0): {H_sigma3:.10f}")

# ---- (4) compare ci_bra3/ci_ket3 vs _expand output ----
print(f"    diff ci_ket (expand vs manual) = {np.abs(ci_ket - ci_ket3).max():.2e}")
print(f"    diff ci_bra (expand vs manual) = {np.abs(ci_bra - ci_bra3).max():.2e}")
