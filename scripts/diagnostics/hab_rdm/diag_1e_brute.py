#!/usr/bin/env python3
"""Brute-force verify the 1e cross term (single create/annihilate + JW phase)."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import (
    _compute_det_creation_explicit, _compute_det_annihilation_explicit,
)


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


def main():
    (mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
     n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
    n_virt = n_act - n_occ

    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)

    # 1e cross: A create (n_A -> n_A+1), B annihilate.  Use source n_A=2 -> 3.
    dets_src = partition[2]['a_dets']
    dets_dst = partition[3]['a_dets']
    idx_dst = partition[3]['a_index']
    c1_A = _compute_det_creation_explicit(dets_src, dets_dst, idx_dst, n_occ)

    bsrc = partition[2]['b_dets']
    bdst = partition[3]['b_dets']
    bidx = partition[3]['b_index']
    a1_B = _compute_det_annihilation_explicit(bsrc, bdst, bidx, n_virt)

    # pick a source (a_src, b_src) and compare brute vs separate contraction, for each spin.
    for aj in range(len(dets_src)):
        for bj in range(len(bsrc)):
            full_src = (dets_src[aj][0] | (bsrc[bj][0] << n_occ),
                        dets_src[aj][1] | (bsrc[bj][1] << n_occ))
            for ai in range(len(dets_dst)):
                for bi in range(len(bdst)):
                    full_dst = (dets_dst[ai][0] | (bdst[bi][0] << n_occ),
                                dets_dst[ai][1] | (bdst[bi][1] << n_occ))
                    # brute: sum over p in A, r in B, spin, of h[p,r] <dst| c_p,sp a_r,sp |src>
                    bval = 0.0
                    for p in range(n_occ):
                        for r in range(n_virt):
                            h = h1eff[p, r + n_occ]
                            if abs(h) < 1e-14:
                                continue
                            for sp in ('a', 'b'):
                                res = apply_op(full_src, [('a', sp, r + n_occ), ('c', sp, p)])
                                if res is None:
                                    continue
                                final, phase = res
                                if final == full_dst:
                                    bval += h * phase
                    # contraction (separate subspaces, no JW):
                    cval = 0.0
                    for p in range(n_occ):
                        for r in range(n_virt):
                            h = h1eff[p, r + n_occ]
                            if abs(h) < 1e-14:
                                continue
                            cval += h * (c1_A['a'][ai, aj, p] * a1_B['a'][bi, bj, r]
                                         + c1_A['b'][ai, aj, p] * a1_B['b'][bi, bj, r])
                    if abs(bval) > 1e-8 or abs(cval) > 1e-8:
                        # JW phase = (-1)^{n_alpha(A_src)} for 'a', (-1)^{n_beta} for 'b'
                        na_src = dets_src[aj][0].bit_count()
                        nb_src = dets_src[aj][1].bit_count()
                        cval_jw = 0.0
                        for p in range(n_occ):
                            for r in range(n_virt):
                                h = h1eff[p, r + n_occ]
                                if abs(h) < 1e-14:
                                    continue
                                cval_jw += h * (c1_A['a'][ai, aj, p] * a1_B['a'][bi, bj, r] * (-1)**na_src
                                                + c1_A['b'][ai, aj, p] * a1_B['b'][bi, bj, r] * (-1)**nb_src)
                        print(f"n_A=2: brute={bval:+.8f} sep(noJW)={cval:+.8f} sep(JW)={cval_jw:+.8f}")
                        return


if __name__ == "__main__":
    main()
