#!/usr/bin/env python3
"""
Second bug hypothesis: Path C places H_A as delta_{beta,delta} HA_schmidt[alpha,gamma]
for ALL beta, ignoring SPIN-SECTOR COMPATIBILITY between A and B Schmidt states.

A product state |A_alpha B_beta> is physical only if sector(alpha)+sector(beta) = total
(n_alpha, n_beta).  Incompatible products have zero full-CI expansion, so their
H_A (and H_B, H_AB) matrix elements must be ZERO.

Check: for an incompatible pair, does Path C assign nonzero while the true value is 0?
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _build_subspace_hamiltonian,
    _extract_subspace_integrals,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

n_A = 3
sd = schmidt[n_A]; r = sd['r']; blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']; b_dets = blk['b_dets']

# A-sector of each U column, B-sector of each V column
def a_sector(alpha):
    s = set()
    for i, d in enumerate(a_dets):
        if abs(U[i, alpha]) > 1e-10:
            s.add((d[0].bit_count(), d[1].bit_count()))
    return s

def b_sector(beta):
    s = set()
    for k, d in enumerate(b_dets):
        if abs(V[k, beta]) > 1e-10:
            s.add((d[0].bit_count(), d[1].bit_count()))
    return s

print(f"n_A=3 block, r={r}")
for alpha in range(r):
    print(f"  U col {alpha}: A-sector {a_sector(alpha)}")
for beta in range(r):
    print(f"  V col {beta}: B-sector {b_sector(beta)}")

# total sector = (na, nb) = (3,3). compatible if A_sector + B_sector == (3,3)
def compatible(alpha, beta):
    for sa in a_sector(alpha):
        for sb in b_sector(beta):
            if (sa[0]+sb[0], sa[1]+sb[1]) == (na, nb):
                return True
    return False

print("\ncompatibility matrix (alpha, beta):")
for alpha in range(r):
    row = []
    for beta in range(r):
        row.append('Y' if compatible(alpha, beta) else '.')
    print(f"  alpha={alpha}: {''.join(row)}")

# Path C HA: H[(gamma,beta),(alpha,beta)] = HA_schmidt[gamma,alpha]
_, _, decomps = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HA_ref = decomps['HA']
block_offsets = {}
off = 0
for nA_ in sorted(schmidt.keys()):
    block_offsets[nA_] = off; off += schmidt[nA_]['r']**2
bo = block_offsets[n_A]
HA_blk = HA_ref[bo:bo+r*r, bo:bo+r*r]

# For an INCOMPATIBLE product state (alpha, beta), the diagonal element
# HA_blk[(alpha,beta),(alpha,beta)] should be ZERO (physical) but Path C gives HA_schmidt[alpha,alpha].
print("\nDiagonal of HA at incompatible product states (should be 0 physically):")
for alpha in range(r):
    for beta in range(r):
        if not compatible(alpha, beta):
            idx = alpha*r + beta
            val = HA_blk[idx, idx]
            print(f"  (alpha={alpha},beta={beta}): PathC HA diag = {val:+.6f}  [physically 0]")
