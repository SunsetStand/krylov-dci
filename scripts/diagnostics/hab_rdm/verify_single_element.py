#!/usr/bin/env python3
"""
Definitive single-element decomposition of within-block H_AB.

For n_A=3 block of H2O, pick element (alpha=0,beta=0)->(gamma=1,delta=0).
Compute it FOUR ways and decompose by operator type:
  (1) sigma-vector reference (ground truth)
  (2) brute-force Slater-Condon in product basis, classified by (A-exc, B-exc)
  (3) transition-matrix "direct"  (spin-summed, (AA|BB))
  (4) transition-matrix "exchange" (spin-explicit, (AB|BA))
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _expand_schmidt_product_to_ci_matrix,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import (
    compute_transition_matrices, _create_sign, _annihilate_sign,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']
b_dets = blk['b_dets']
a_index = blk['a_index']
b_index = blk['b_index']

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs); n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

# target element
a_src, b_src, a_dst, b_dst = 0, 0, 1, 0
print(f"Element: (alpha={a_src},beta={b_src}) -> (alpha={a_dst},beta={b_dst}), r={r}")

# ---- (1) sigma reference ----
ci_ket = _expand_schmidt_product_to_ci_matrix(
    a_src, b_src, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
ci_bra = _expand_schmidt_product_to_ci_matrix(
    a_dst, b_dst, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
sigma = backend.sigma_full(ci_ket)
H_ref = np.sum(ci_bra * sigma)
print(f"[1] sigma ref H = {H_ref:.10f}")

# ---- (2) brute force Slater-Condon in product basis, classify by B-exc ----
# Expand bra/ket in product basis: |a_i>|b_k>
from src.hamiltonian import Hamiltonian
ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

def prod_to_full(i_a, k_b):
    aA, aB = a_dets[i_a]; bA, bB = b_dets[k_b]
    return (aA | (bA << n_occ), aB | (bB << n_occ))

# ket coefficients: U[i,a_src] * V[k,b_src]
# bra coefficients: U[j,a_dst] * V[l,b_dst]
H_bf = 0.0
contrib = {}
for i in range(len(a_dets)):
    ui = U[i, a_src]
    if abs(ui) < 1e-12: continue
    for k in range(len(b_dets)):
        vk = V[k, b_src]
        if abs(vk) < 1e-12: continue
        ket_w = ui * vk
        for j in range(len(a_dets)):
            uj = U[j, a_dst]
            if abs(uj) < 1e-12: continue
            for l in range(len(b_dets)):
                vl = V[l, b_dst]
                if abs(vl) < 1e-12: continue
                bra_w = uj * vl
                w = bra_w * ket_w
                if abs(w) < 1e-14: continue
                hij = ham.matrix_element(prod_to_full(i, k), prod_to_full(j, l))
                # classify A-exc and B-exc
                aA_i, aB_i = a_dets[i]; aA_j, aB_j = a_dets[j]
                bA_k, bB_k = b_dets[k]; bA_l, bB_l = b_dets[l]
                def exc(d1, d2):
                    x = (d1[0]^d2[0]).bit_count() + (d1[1]^d2[1]).bit_count()
                    return x // 2
                eA = exc(a_dets[i], a_dets[j])
                eB = exc(b_dets[k], b_dets[l])
                key = (eA, eB)
                contrib[key] = contrib.get(key, 0.0) + w * hij
                H_bf += w * hij

print(f"[2] brute-force H = {H_bf:.10f}")
for key in sorted(contrib.keys()):
    print(f"    (A-exc,B-exc)={key}: {contrib[key]:+.10f}")

# ---- (3)+(4) transition matrices ----
def spin_explicit_1body(dets, idx, n_orb):
    d = len(dets)
    comps = {k: np.zeros((d, d, n_orb, n_orb)) for k in ('aa','ab','ba','bb')}
    for jj, (aA_j, bA_j) in enumerate(dets):
        # aa
        for q in range(n_orb):
            if not ((aA_j >> q) & 1): continue
            phq, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (a1 >> p) & 1: continue
                if p == q: comps['aa'][jj,jj,p,q]=1.0; continue
                php, a2 = _create_sign(a1, p)
                if php==0: continue
                ii = idx.get((a2, bA_j))
                if ii is not None: comps['aa'][ii,jj,p,q]=phq*php
        # bb
        for q in range(n_orb):
            if not ((bA_j >> q) & 1): continue
            phq, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (b1 >> p) & 1: continue
                if p == q: comps['bb'][jj,jj,p,q]=1.0; continue
                php, b2 = _create_sign(b1, p)
                if php==0: continue
                ii = idx.get((aA_j, b2))
                if ii is not None: comps['bb'][ii,jj,p,q]=phq*php
        # ab (create alpha, annihilate beta)
        for q in range(n_orb):
            if not ((bA_j >> q) & 1): continue
            phq, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (aA_j >> p) & 1: continue
                php, a2 = _create_sign(aA_j, p)
                if php==0: continue
                ii = idx.get((a2, b1))
                if ii is not None: comps['ab'][ii,jj,p,q]=phq*php
        # ba (create beta, annihilate alpha)
        for q in range(n_orb):
            if not ((aA_j >> q) & 1): continue
            phq, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (bA_j >> p) & 1: continue
                php, b2 = _create_sign(bA_j, p)
                if php==0: continue
                ii = idx.get((a1, b2))
                if ii is not None: comps['ba'][ii,jj,p,q]=phq*php
    return comps

compA = spin_explicit_1body(a_dets, a_index, n_occ)
compB = spin_explicit_1body(b_dets, b_index, n_virt)
TA = {k: np.einsum('ijpq,ia,jg->agpq', compA[k], U, U) for k in compA}
TB = {k: np.einsum('ijpq,ia,jg->agpq', compB[k], V, V) for k in compB}
TA_sum = TA['aa'] + TA['bb']
TB_sum = TB['aa'] + TB['bb']

# direct: sum_{ij in A, kl in B} (ij|kl) TA_sum[i,j] TB_sum[k,l]
H_direct = 0.0
for i in range(n_occ):
    for j in range(n_occ):
        ta = TA_sum[a_dst, a_src, i, j]
        if abs(ta) < 1e-14: continue
        for k in range(n_virt):
            for l in range(n_virt):
                tb = TB_sum[b_dst, b_src, k, l]
                if abs(tb) < 1e-14: continue
                H_direct += h2_4d[i, j, k + n_occ, l + n_occ] * ta * tb

# exchange: -sum_{il in A, jk in B} (ij|kl) sum_{ss'} TA_ss'[i,l] TB_s's[k,j]
H_exch = 0.0
spin_pairs = [('aa','aa'), ('bb','bb'), ('ab','ba'), ('ba','ab')]
for i in range(n_occ):
    for l in range(n_occ):
        for j in range(n_virt):
            for k in range(n_virt):
                v = h2_4d[i, j + n_occ, k + n_occ, l]
                if abs(v) < 1e-14: continue
                s = 0.0
                for sA, sB in spin_pairs:
                    ta = TA[sA][a_dst, a_src, i, l]
                    tb = TB[sB][b_dst, b_src, k, j]
                    s += ta * tb
                H_exch += v * s
H_exch = -H_exch

print(f"[3] direct   = {H_direct:+.10f}")
print(f"[4] exchange = {H_exch:+.10f}")
print(f"    direct+exchange = {H_direct + H_exch:+.10f}")
print(f"    ref              = {H_ref:+.10f}")
print(f"    residual         = {H_ref - H_direct - H_exch:+.10f}")
