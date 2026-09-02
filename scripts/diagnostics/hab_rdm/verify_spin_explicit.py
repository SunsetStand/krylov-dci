#!/usr/bin/env python3
"""
Verify: the n_A-conserved 2e part of H_AB requires SPIN-EXPLICIT
transition matrices, not spin-summed ones.

block2 (Chan et al. JCP 145, 014102 (2016)) normal/complementary form:
  H_AB (n_A-conserved, within same n block) =
      + sum_{ij in A} B_ij(A)  x Q^B_ij              [direct,  same-spin]
      - sum_{il in A, ss'}  B'_il,ss'(A) x Q'^B_il,ss'  [exchange, cross-spin]

where
  B_ij        = sum_s a^+_is a_js          (spin-summed excitation)
  B'_il,ss'   = a^+_is a_ls'               (spin-EXPLICIT excitation)
  Q^B_ij      = sum_{kl in B, s'} v_ijkl a^+_ks' a_ls'
  Q'^B_il,ss' = sum_{jk in B}     v_ijkl a^+_ks' a_js

The "direct" term is a product of two spin-summed operators -> trans_1 works.
The "exchange" term is a product of spin-EXPLICIT operators a^+_is a_ls',
which requires the four spin components (aa, ab, ba, bb) separately.

This script computes, for each n_A block:
  (a) reference within-block H_AB (from sigma-vector build_h_emb)
  (b) direct term using spin-summed trans_1
  (c) exchange term using spin-explicit components
and checks (b)+(c) == (a).
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import (
    _create_sign, _annihilate_sign,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)
HAB_ref = decomps_ref['HAB']

# block offsets
block_offsets = {}
off = 0
for na_ in sorted(schmidt.keys()):
    block_offsets[na_] = off
    off += schmidt[na_]['r'] ** 2
D = off


def spin_explicit_1body(a_dets, a_index, n_orb):
    """Return dict with 4 spin components of <d_i| a^+_ps a_qs' |d_j>.

    keys 'aa','ab','ba','bb' -> (d, d, n_orb, n_orb) arrays.
    'aa' = a^+_pa a_qa (create alpha, annihilate alpha)
    'ab' = a^+_pa a_qb (create alpha, annihilate beta)
    etc.
    """
    d = len(a_dets)
    comps = {k: np.zeros((d, d, n_orb, n_orb)) for k in ('aa', 'ab', 'ba', 'bb')}
    for j, (aA_j, bA_j) in enumerate(a_dets):
        # --- alpha -> alpha: a^+_pa a_qa ---
        for q in range(n_orb):
            if not ((aA_j >> q) & 1):
                continue
            phq, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (a1 >> p) & 1:
                    continue
                if p == q:
                    comps['aa'][j, j, p, q] = 1.0
                    continue
                php, a2 = _create_sign(a1, p)
                if php == 0:
                    continue
                i = a_index.get((a2, bA_j))
                if i is not None:
                    comps['aa'][i, j, p, q] = phq * php
        # --- beta -> beta: a^+_pb a_qb ---
        for q in range(n_orb):
            if not ((bA_j >> q) & 1):
                continue
            phq, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (b1 >> p) & 1:
                    continue
                if p == q:
                    comps['bb'][j, j, p, q] = 1.0
                    continue
                php, b2 = _create_sign(b1, p)
                if php == 0:
                    continue
                i = a_index.get((aA_j, b2))
                if i is not None:
                    comps['bb'][i, j, p, q] = phq * php
        # --- alpha -> beta: a^+_pa a_qb (create alpha, annihilate beta) ---
        # Fock ordering: |det> = alpha-string THEN beta-string.
        # a_qb (beta annihilator) must pass through all n_alpha alpha ops:
        #   cross sign = (-1)^{n_alpha(source)}.
        n_alpha_j = aA_j.bit_count()
        cross_ab = -1.0 if (n_alpha_j % 2 == 1) else 1.0
        for q in range(n_orb):
            if not ((bA_j >> q) & 1):
                continue
            phq, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (aA_j >> p) & 1:
                    continue
                php, a2 = _create_sign(aA_j, p)
                if php == 0:
                    continue
                i = a_index.get((a2, b1))
                if i is not None:
                    comps['ab'][i, j, p, q] = cross_ab * phq * php
        # --- beta -> alpha: a^+_pb a_qa (create beta, annihilate alpha) ---
        # a_qa acts on alpha-string directly (leftmost): no cross sign.
        for q in range(n_orb):
            if not ((aA_j >> q) & 1):
                continue
            phq, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (bA_j >> p) & 1:
                    continue
                php, b2 = _create_sign(bA_j, p)
                if php == 0:
                    continue
                i = a_index.get((a1, b2))
                if i is not None:
                    comps['ba'][i, j, p, q] = phq * php
    return comps


def transform_1body(T_det, U_dst, U_src):
    return np.einsum('ijpq,ia,jg->agpq', T_det, U_dst, U_src)


# Build spin-explicit transition matrices for A and B
transA_explicit = {}   # n_A -> dict of 4 spin comps (r,r,n_orb,n_orb)
transB_explicit = {}

for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]
    r = sd['r']
    blk = partition[n_A]
    if r == 0:
        continue
    # A-space (n_occ orbitals)
    a_dets = blk['a_dets']
    a_idx = blk['a_index']
    U = sd['U']
    compA = spin_explicit_1body(a_dets, a_idx, n_occ)
    transA_explicit[n_A] = {k: transform_1body(compA[k], U, U) for k in compA}
    # B-space (n_virt orbitals)
    b_dets = blk['b_dets']
    b_idx = blk['b_index']
    V = sd['V']
    compB = spin_explicit_1body(b_dets, b_idx, n_virt)
    transB_explicit[n_A] = {k: transform_1body(compB[k], V, V) for k in compB}

# Now build the within-block n_A-conserved H_AB and compare with reference.
print("=" * 72)
print("Block-by-block: within-block H_AB  (reference vs direct+exchange)")
print("=" * 72)
total_max_diff = 0.0
for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]
    r = sd['r']
    if r == 0:
        continue
    bo = block_offsets[n_A]
    # reference within-block H_AB
    ref = HAB_ref[bo:bo + r * r, bo:bo + r * r]

    # direct term (spin-summed): sum_{ij in A, kl in B} v_ijkl TA[i,j] TB[k,l]
    #   TA = aa+bb, TB = aa+bb
    TA = transA_explicit[n_A]
    TB = transB_explicit[n_A]
    TA_sum = TA['aa'] + TA['bb']
    TB_sum = TB['aa'] + TB['bb']

    direct = np.zeros((r * r, r * r))
    for a_src in range(r):
        for b_src in range(r):
            ik = a_src * r + b_src
            for a_dst in range(r):
                for b_dst in range(r):
                    ib = a_dst * r + b_dst
                    val = 0.0
                    for i in range(n_occ):
                        for j in range(n_occ):
                            ta = TA_sum[a_dst, a_src, i, j]
                            if abs(ta) < 1e-14:
                                continue
                            for k in range(n_virt):
                                for l in range(n_virt):
                                    tb = TB_sum[b_dst, b_src, k, l]
                                    if abs(tb) < 1e-14:
                                        continue
                                    val += h2_4d[i, j, k + n_occ, l + n_occ] * ta * tb
                    direct[ib, ik] = val

    # exchange term (spin-explicit):
    #   -sum_{il in A, jk in B} v_ijkl sum_{ss'} TA_ss'[i,l] TB_s's[k,j]
    #   with TA_ss' = a^+_is a_ls' (A), TB_s's = a^+_ks' a_js (B)
    #   v_ijkl = h2_4d[i, j+n_occ, k+n_occ, l]
    exch = np.zeros((r * r, r * r))
    # spin-pairing map: A (create s, annih s'), B (create s', annih s)
    spin_pairs = [('aa', 'aa'), ('bb', 'bb'), ('ab', 'ba'), ('ba', 'ab')]
    for a_src in range(r):
        for b_src in range(r):
            ik = a_src * r + b_src
            for a_dst in range(r):
                for b_dst in range(r):
                    ib = a_dst * r + b_dst
                    val = 0.0
                    for i in range(n_occ):
                        for l in range(n_occ):
                            for j in range(n_virt):
                                for k in range(n_virt):
                                    v = h2_4d[i, j + n_occ, k + n_occ, l]
                                    if abs(v) < 1e-14:
                                        continue
                                    s = 0.0
                                    for sA, sB in spin_pairs:
                                        ta = TA[sA][a_dst, a_src, i, l]
                                        tb = TB[sB][b_dst, b_src, k, j]
                                        s += ta * tb
                                    val += v * s
                    exch[ib, ik] = -val

    total = direct + exch
    diff = np.abs(ref - total).max()
    nr = np.linalg.norm(ref)
    nd = np.linalg.norm(total)
    total_max_diff = max(total_max_diff, diff)
    print(f"  n={n_A:2d} r={r:2d}: ||ref||={nr:8.4f}  ||direct+exch||={nd:8.4f}  "
          f"max|diff|={diff:.3e}")

print("=" * 72)
print(f"  TOTAL max|ref - (direct+exchange)| = {total_max_diff:.6e}")
print("=" * 72)
