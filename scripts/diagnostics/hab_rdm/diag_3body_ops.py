#!/usr/bin/env python3
"""Verify 3-body (c2a1 / c1a2) spin-explicit transition matrices vs brute force."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.transition_rdm import (
    _create_sign, _annihilate_sign,
    _compute_det_create2_annih1_explicit,
    _compute_det_create1_annih2_explicit,
)


def all_dets(n_orb, na, nb):
    from itertools import combinations
    def gen(k):
        out = []
        for c in combinations(range(n_orb), k):
            s = 0
            for p in c:
                s |= (1 << p)
            out.append(s)
        return out
    A = gen(na); B = gen(nb)
    return [(a, b) for a in A for b in B]


def apply(d, ops):
    aA, bA = d; ph = 1
    for kind, spin, orb in ops:
        if spin == 'a':
            p, aA = (_create_sign(aA, orb) if kind == 'c' else _annihilate_sign(aA, orb))
        else:
            p, bA = (_create_sign(bA, orb) if kind == 'c' else _annihilate_sign(bA, orb))
        if p == 0:
            return None
        ph *= p
    return (aA, bA), ph


def brute_c2a1(dets_src, dets_dst, idx_dst, n_orb, s1, s2, s3):
    """<d_dst| a+_p,s1 a+_q,s2 a_r,s3 |d_src>  (a_r first, then a+_q, then a+_p)."""
    T = np.zeros((len(dets_dst), len(dets_src), n_orb, n_orb, n_orb))
    for j, d in enumerate(dets_src):
        for r in range(n_orb):
            res = apply(d, [('a', s3, r)])
            if res is None:
                continue
            st, ph_r = res
            for q in range(n_orb):
                res2 = apply(st, [('c', s2, q)])
                if res2 is None:
                    continue
                st2, ph_q = res2
                for p in range(n_orb):
                    res3 = apply(st2, [('c', s1, p)])
                    if res3 is None:
                        continue
                    final, ph_p = res3
                    i = idx_dst.get(final)
                    if i is not None:
                        T[i, j, p, q, r] = ph_r * ph_q * ph_p
    return T


def brute_c1a2(dets_src, dets_dst, idx_dst, n_orb, s1, s2, s3):
    """<d_dst| a+_p,s1 a_q,s2 a_r,s3 |d_src>  (a_r first, then a_q, then a+_p)."""
    T = np.zeros((len(dets_dst), len(dets_src), n_orb, n_orb, n_orb))
    for j, d in enumerate(dets_src):
        for r in range(n_orb):
            res = apply(d, [('a', s3, r)])
            if res is None:
                continue
            st, ph_r = res
            for q in range(n_orb):
                res2 = apply(st, [('a', s2, q)])
                if res2 is None:
                    continue
                st2, ph_q = res2
                for p in range(n_orb):
                    res3 = apply(st2, [('c', s1, p)])
                    if res3 is None:
                        continue
                    final, ph_p = res3
                    i = idx_dst.get(final)
                    if i is not None:
                        T[i, j, p, q, r] = ph_r * ph_q * ph_p
    return T


def main():
    n_orb = 4
    # c2a1: source n=2 (1a+1b), dest n=3
    src = all_dets(n_orb, 1, 1)
    combos = {
        'aaa': ('a', 'a', 'a', (2, 1)),   # create a,a; annih a -> net (2a,1b)?
        'aba': ('a', 'b', 'a', (2, 1)),
        'abb': ('a', 'b', 'b', (1, 2)),
        'bab': ('b', 'a', 'b', (1, 2)),
        'baa': ('b', 'a', 'a', (2, 1)),
        'bbb': ('b', 'b', 'b', (1, 2)),
    }
    print("=== c2a1 (create2+annih1) ===")
    ok = True
    for c, (s1, s2, s3, (na, nb)) in combos.items():
        dst = all_dets(n_orb, na, nb)
        idx = {d: i for i, d in enumerate(dst)}
        got = _compute_det_create2_annih1_explicit(src, dst, idx, n_orb)[c]
        ref = brute_c2a1(src, dst, idx, n_orb, s1, s2, s3)
        diff = np.abs(got - ref).max()
        print(f"  c2a1[{c}] max|diff| = {diff:.2e} {'OK' if diff < 1e-12 else 'FAIL'}")
        ok &= diff < 1e-12

    # c1a2: source n=3 (2a+1b), dest n=2
    src2 = all_dets(n_orb, 2, 1)
    combos2 = {
        'aaa': ('a', 'a', 'a', (1, 1)),
        'aab': ('a', 'a', 'b', (1, 1)),
        'aba': ('a', 'b', 'a', (1, 1)),
        'bab': ('b', 'a', 'b', (1, 1)),
        'bba': ('b', 'b', 'a', (1, 1)),
        'bbb': ('b', 'b', 'b', (1, 1)),
    }
    print("=== c1a2 (create1+annih2) ===")
    for c, (s1, s2, s3, (na, nb)) in combos2.items():
        dst = all_dets(n_orb, na, nb)
        idx = {d: i for i, d in enumerate(dst)}
        got = _compute_det_create1_annih2_explicit(src2, dst, idx, n_orb)[c]
        ref = brute_c1a2(src2, dst, idx, n_orb, s1, s2, s3)
        diff = np.abs(got - ref).max()
        print(f"  c1a2[{c}] max|diff| = {diff:.2e} {'OK' if diff < 1e-12 else 'FAIL'}")
        ok &= diff < 1e-12
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
