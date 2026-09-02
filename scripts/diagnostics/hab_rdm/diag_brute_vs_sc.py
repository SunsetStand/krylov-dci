#!/usr/bin/env python3
"""Compare my brute-force 2e operator application vs Hamiltonian.matrix_element (Slater-Condon)."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from src.hamiltonian import Hamiltonian

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

ham = Hamiltonian(h1=h1eff, h2=h2_4d)


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


def brute_2e_element(src, dst, h2, norb):
    """<dst| H_2e |src> = 1/2 sum_{pqrs} sum_st (pq|rs) <dst| c_p,st c_q,t a_s,t a_r,st |src>."""
    val = 0.0
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    g = h2[p, q, r, s]
                    if abs(g) < 1e-14:
                        continue
                    for sig in ('a', 'b'):
                        for tau in ('a', 'b'):
                            res = apply_op(src, [('a', sig, r), ('a', tau, s),
                                                 ('c', tau, q), ('c', sig, p)])
                            if res is None:
                                continue
                            final, phase = res
                            if final == dst:
                                val += 0.5 * g * phase
    return val


# Get all determinant strings
alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
dets = [(int(a), int(b)) for a in alpha_strs for b in beta_strs]

# Compare for the diagonal and a few off-diagonal pairs
import random
random.seed(1)
idx_pairs = [(0, 0)]  # diagonal (HF det)
# add some random pairs
for _ in range(20):
    idx_pairs.append((random.randint(0, len(dets)-1), random.randint(0, len(dets)-1)))

mismatch = 0
for ia, ib in idx_pairs:
    d_src = dets[ia]
    d_dst = dets[ib]
    # only compare where excitation level <= 2 (Slater-Condon III), else matrix_element returns 0
    # (our brute force handles all levels)
    brute = brute_2e_element(d_src, d_dst, h2_4d, n_act)
    ref = ham.matrix_element(d_src, d_dst) - ham_1e_element(d_src, d_dst) if False else None
    # ref 2e-only: matrix_element includes 1e+2e.  Subtract 1e part.
    # Compute 1e part separately.
    a1, b1 = d_src; a2, b2 = d_dst
    # 1e: sum over single excitations
    ref_full = ham.matrix_element(d_src, d_dst)
    # 1e contribution: h1[p,a] for single excitation
    # simpler: use Hamiltonian with h2=0
    ham1 = Hamiltonian(h1=h1eff, h2=np.zeros_like(h2_4d))
    ref_1e = ham1.matrix_element(d_src, d_dst)
    ref_2e = ref_full - ref_1e
    diff = abs(brute - ref_2e)
    if diff > 1e-8:
        mismatch += 1
        if mismatch <= 12:
            print(f"({ia},{ib}): brute_2e={brute:+.10f} ref_2e={ref_2e:+.10f} diff={diff:.2e}")

print(f"\nmismatches: {mismatch}/{len(idx_pairs)}")
