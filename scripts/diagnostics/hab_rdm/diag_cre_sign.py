#!/usr/bin/env python3
"""Compare hand-rolled _create_sign/_annihilate_sign vs PySCF cistring phases."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from pyscf.fci import cistring
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign

# For a given string and a single excitation i -> a (annihilate i, create a),
# PySCF's cre_des_sign(a, i, str) gives the phase of a†_a a_i |str>.
# My version: _annihilate_sign(str, i) then _create_sign(result, a), multiply phases.

def my_cre_des(str_, i, a):
    ph_i, s1 = _annihilate_sign(str_, i)
    if ph_i == 0:
        return 0
    ph_a, s2 = _create_sign(s1, a)
    if ph_a == 0:
        return 0
    return ph_i * ph_a

# test a bunch of random strings and excitations
import random
random.seed(0)
n_orb = 5
mismatch = 0
for trial in range(200):
    str_ = random.randint(0, (1 << n_orb) - 1)
    i = random.randint(0, n_orb - 1)
    a = random.randint(0, n_orb - 1)
    if i == a:
        continue
    # only valid if i occupied and a empty
    if not ((str_ >> i) & 1) or ((str_ >> a) & 1):
        continue
    mine = my_cre_des(str_, i, a)
    py = cistring.cre_des_sign(a, i, str_)
    if mine != py:
        mismatch += 1
        if mismatch <= 10:
            print(f"MISMATCH str={str_:05b} i={i} a={a}: mine={mine} pyscf={py}")

print(f"\ntotal mismatches: {mismatch} / 200 trials")
