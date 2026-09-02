#!/usr/bin/env python3
"""
Confirm the spin-conservation bug in src/hamiltonian.py matrix_element.

The Hamiltonian conserves n_alpha and n_beta SEPARATELY (it is spin-conserving).
Therefore <det1|H|det2> = 0 EXACTLY whenever det1 and det2 have different
(n_alpha, n_beta).

In the A-subspace block (n_A=3), determinants split into sectors (2,1) and (1,2).
matrix_element must return 0 for any cross-sector pair.  We show it does not.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _build_subspace_hamiltonian, _extract_subspace_integrals,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from src.hamiltonian import Hamiltonian
from src.determinants import excitation_level, find_excitations

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

n_A = 3
blk = partition[n_A]
a_dets = blk['a_dets']

h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
hamA = Hamiltonian(h1=h1_A, h2=h2_A, E_nuc=0.0, E_HF=0.0)

# group A-dets by spin sector
sector_dets = {}
for i, d in enumerate(a_dets):
    s = (d[0].bit_count(), d[1].bit_count())
    sector_dets.setdefault(s, []).append((i, d))

print(f"A-space sectors: { {k: len(v) for k,v in sector_dets.items()} }")

# pick one (2,1) and one (1,2) det that are related by a single spin flip
sec21 = sector_dets[(2,1)] if (2,1) in sector_dets else sector_dets[(1,2)]
sec12 = sector_dets[(1,2)] if (1,2) in sector_dets else sector_dets[(2,1)]

# find a pair with minimal bit-difference (likely a pure spin flip)
best = None
for i1, d1 in sec21:
    for i2, d2 in sec12:
        diff = (d1[0]^d2[0]).bit_count() + (d1[1]^d2[1]).bit_count()
        if best is None or diff < best[0]:
            best = (diff, i1, d1, i2, d2)

diff, i1, d1, i2, d2 = best
print(f"\nclosest cross-sector pair, bit-diff = {diff}:")
print(f"  det1 = ({d1[0]:0{n_occ}b}, {d1[1]:0{n_occ}b})  sector (2,1) [a,b strings]")
print(f"  det2 = ({d2[0]:0{n_occ}b}, {d2[1]:0{n_occ}b})  sector (1,2)")
print(f"  excitation_level = {excitation_level(d1, d2)}")
print(f"  holes/particles  = {find_excitations(d1, d2)}")
print(f"  matrix_element (Slater-Condon) = {hamA.matrix_element(d1, d2):.10f}")
print(f"  (correct value = 0.0 by spin conservation)")
