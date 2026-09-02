#!/usr/bin/env python3
"""Test the Jordan-Wigner phase for pair transfer on odd vs even n_A."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition


def apply_op(d, ops):
    aA, bA = d
    phase = 1
    for kind, spin, orb in ops:
        if spin == 'a':
            ph, aA = (_create_sign(aA, orb) if kind == 'c' else _annihilate_sign(aA, orb))
        else:
            ph, bA = (_create_sign(bA, orb) if kind == 'c' else _annihilate_sign(bA, orb))
        if ph == 0:
            return None
        phase *= ph
    return (aA, bA), phase


def brute_pair_element(full_src, full_dst, h2, n_occ, n_act):
    """Full-determinant brute force <dst| H_pair |src> (includes JW automatically)."""
    val = 0.0
    n_virt = n_act - n_occ
    for p in range(n_occ):
        for q in range(n_occ):
            for r in range(n_virt):
                for s in range(n_virt):
                    g = h2[p, q, r + n_occ, s + n_occ]
                    if abs(g) < 1e-14:
                        continue
                    for sig in ('a', 'b'):
                        for tau in ('a', 'b'):
                            res = apply_op(full_src, [('a', sig, r + n_occ), ('a', tau, s + n_occ),
                                                       ('c', tau, q), ('c', sig, p)])
                            if res is None:
                                continue
                            final, phase = res
                            if final == full_dst:
                                val += 0.5 * g * phase
    return val


def main():
    (mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
     n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
    n_virt = n_act - n_occ

    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
    C_blocks = build_block_matrices(partition, ci_flat)
    schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

    from dm_svd_embedding.transition_rdm import (
        _compute_det_pair_creation_explicit, _compute_det_pair_annihilation_explicit,
    )

    # For each source block n_A (3 -> 5 and 2 -> 4), compare brute vs separate-subspace contraction.
    for ns in (2, 3):
        nd = ns + 2
        if ns not in partition or nd not in partition:
            continue
        dets_src = partition[ns]['a_dets']
        dets_dst = partition[nd]['a_dets']
        idx_dst = partition[nd]['a_index']
        c2_A = _compute_det_pair_creation_explicit(dets_src, dets_dst, idx_dst, n_occ)

        bsrc = partition[ns]['b_dets']
        bdst = partition[nd]['b_dets']
        bidx = partition[nd]['b_index']
        a2_B = _compute_det_pair_annihilation_explicit(bsrc, bdst, bidx, n_virt)

        spin_pairs = [('aa', 'aa'), ('ab', 'ba'), ('ba', 'ab'), ('bb', 'bb')]

        # test first (a_src, b_src) pair with nonzero coupling
        found = False
        for aj in range(len(dets_src)):
            for bj in range(len(bsrc)):
                # brute for all dst
                full_src = (dets_src[aj][0] | (bsrc[bj][0] << n_occ),
                            dets_src[aj][1] | (bsrc[bj][1] << n_occ))
                # contraction
                cval = 0.0
                for sA, sB in spin_pairs:
                    for ai in range(len(dets_dst)):
                        for bi in range(len(bdst)):
                            full_dst = (dets_dst[ai][0] | (bdst[bi][0] << n_occ),
                                        dets_dst[ai][1] | (bdst[bi][1] << n_occ))
                            bval = brute_pair_element(full_src, full_dst, h2_4d, n_occ, n_act)
                            cv = 0.0
                            for i in range(n_occ):
                                for k in range(n_occ):
                                    for j in range(n_virt):
                                        for l in range(n_virt):
                                            g = h2_4d[i, k, j + n_occ, l + n_occ]
                                            if abs(g) < 1e-14:
                                                continue
                                            cv += 0.5 * g * c2_A[sA][ai, aj, i, k] * a2_B[sB][bi, bj, l, j]
                            if abs(bval) > 1e-8 and abs(cv) > 1e-8:
                                print(f"n_A={ns}: brute={bval:+.6f} separate(no JW)={cv:+.6f} ratio={cv/bval:+.4f}")
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break


if __name__ == "__main__":
    main()
