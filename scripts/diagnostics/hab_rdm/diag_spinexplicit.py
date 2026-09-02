#!/usr/bin/env python3
"""Verify spin-explicit transition matrices against brute-force Slater-Condon."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.transition_rdm import (
    _create_sign, _annihilate_sign,
    _compute_det_pair_creation_explicit,
    _compute_det_pair_annihilation_explicit,
    _compute_det_create2_annih1_explicit,
    _compute_det_create1_annih2_explicit,
)


def all_dets(n_orb, na, nb):
    from itertools import combinations
    alpha = []
    for comb in combinations(range(n_orb), na):
        s = 0
        for p in comb:
            s |= (1 << p)
        alpha.append(s)
    beta = []
    for comb in combinations(range(n_orb), nb):
        s = 0
        for p in comb:
            s |= (1 << p)
        beta.append(s)
    return [(a, b) for a in alpha for b in beta]


def brute_pair_creation(dets_src, dets_dst, idx_dst, n_orb, s1, s2):
    d_dst, d_src = len(dets_dst), len(dets_src)
    T = np.zeros((d_dst, d_src, n_orb, n_orb))
    for j, (aA, bA) in enumerate(dets_src):
        for q in range(n_orb):  # a†_q,s2 acts first
            if s2 == 'a':
                ph_q, a1 = _create_sign(aA, q)
                if ph_q == 0:
                    continue
                st = (a1, bA)
            else:
                ph_q, b1 = _create_sign(bA, q)
                if ph_q == 0:
                    continue
                st = (aA, b1)
            for p in range(n_orb):  # a†_p,s1 acts second
                if s1 == 'a':
                    ph_p, a2 = _create_sign(st[0], p)
                    if ph_p == 0:
                        continue
                    final = (a2, st[1])
                else:
                    ph_p, b2 = _create_sign(st[1], p)
                    if ph_p == 0:
                        continue
                    final = (st[0], b2)
                i = idx_dst.get(final)
                if i is not None:
                    T[i, j, p, q] = ph_q * ph_p
    return T


def brute_pair_annihilation(dets_src, dets_dst, idx_dst, n_orb, s1, s2):
    d_dst, d_src = len(dets_dst), len(dets_src)
    T = np.zeros((d_dst, d_src, n_orb, n_orb))
    for j, (aA, bA) in enumerate(dets_src):
        for q in range(n_orb):  # a_q,s2 acts first
            if s2 == 'a':
                ph_q, a1 = _annihilate_sign(aA, q)
                if ph_q == 0:
                    continue
                st = (a1, bA)
            else:
                ph_q, b1 = _annihilate_sign(bA, q)
                if ph_q == 0:
                    continue
                st = (aA, b1)
            for p in range(n_orb):  # a_p,s1 acts second
                if s1 == 'a':
                    ph_p, a2 = _annihilate_sign(st[0], p)
                    if ph_p == 0:
                        continue
                    final = (a2, st[1])
                else:
                    ph_p, b2 = _annihilate_sign(st[1], p)
                    if ph_p == 0:
                        continue
                    final = (st[0], b2)
                i = idx_dst.get(final)
                if i is not None:
                    T[i, j, p, q] = ph_q * ph_p
    return T


def check(name, got, ref):
    if got.size == 0 and ref.size == 0:
        print(f"  {name:20s} (both empty) OK")
        return True
    diff = np.abs(got - ref).max()
    status = 'OK' if diff < 1e-12 else 'FAIL'
    print(f"  {name:20s} max|diff| = {diff:.3e}  {status}")
    return diff < 1e-12


def main():
    n_orb = 4
    ok = True

    print("=== pair creation explicit vs brute (n_orb=4) ===")
    src = all_dets(n_orb, 2, 1)
    for c, (na, nb) in [('aa', (4, 1)), ('ab', (3, 2)), ('ba', (3, 2)), ('bb', (2, 3))]:
        dst = all_dets(n_orb, na, nb)
        idx = {d: i for i, d in enumerate(dst)}
        got = _compute_det_pair_creation_explicit(src, dst, idx, n_orb)[c]
        ref = brute_pair_creation(src, dst, idx, n_orb, c[0], c[1])
        ok &= check(f"create_2[{c}]", got, ref)

    print("=== pair annihilation explicit vs brute (n_orb=4) ===")
    src2 = all_dets(n_orb, 4, 2)
    for c, (na, nb) in [('aa', (2, 2)), ('ab', (3, 1)), ('ba', (3, 1)), ('bb', (4, 0))]:
        dst = all_dets(n_orb, na, nb)
        idx = {d: i for i, d in enumerate(dst)}
        got = _compute_det_pair_annihilation_explicit(src2, dst, idx, n_orb)[c]
        ref = brute_pair_annihilation(src2, dst, idx, n_orb, c[0], c[1])
        ok &= check(f"annihilate_2[{c}]", got, ref)

    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
