#!/usr/bin/env python3
"""Verify backend2.sigma_full (2e, selected_ci.contract_2e on h2eff) == chemist brute."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

h2eff = cas.get_h2eff()
q2 = QSpaceIndex(q_idx.alpha_strs, q_idx.beta_strs, n_act, (na, nb),
                 np.zeros_like(h1eff), h2eff)
backend2 = KDCIBackend(q2)


def apply_op(d, ops):
    aA, bA = d; phase = 1
    for kind, spin, orb in ops:
        if spin == 'a':
            ph, aA = (_create_sign(aA, orb) if kind == 'c' else _annihilate_sign(aA, orb))
        else:
            ph, bA = (_create_sign(bA, orb) if kind == 'c' else _annihilate_sign(bA, orb))
        if ph == 0:
            return None
        phase *= ph
    return (aA, bA), phase


def brute(full_src, full_dst):
    val = 0.0
    for p in range(n_act):
        for q in range(n_act):
            for r in range(n_act):
                for s in range(n_act):
                    g = h2_4d[p,q,r,s]
                    if abs(g) < 1e-14: continue
                    for sig in ('a','b'):
                        for tau in ('a','b'):
                            res = apply_op(full_src, [('a',sig,q),('a',tau,s),('c',tau,r),('c',sig,p)])
                            if res is None: continue
                            f, ph = res
                            if f == full_dst: val += 0.5*g*ph
    return val


# build sigma_full matrix in determinant basis: S = H_2e applied to each unit det
dets = [(int(a), int(b)) for a in q_idx.alpha_strs for b in q_idx.beta_strs]
N = len(dets)
idx = {d: i for i, d in enumerate(dets)}

# sigma_full for a single det (unit CI vector)
import numpy as np
ci = np.zeros((len(q_idx.alpha_strs), len(q_idx.beta_strs)))
maxdiff = 0.0; worst = None
for k in range(min(N, 15)):
    d = dets[k]
    ia = q_idx._alpha_idx[d[0]]; ib = q_idx._beta_idx[d[1]]
    ci[:] = 0.0
    ci[ia, ib] = 1.0
    s = backend2.sigma_full(ci)  # (n_alpha, n_beta)
    # compare each column element to brute
    for ia2 in range(len(q_idx.alpha_strs)):
        for ib2 in range(len(q_idx.beta_strs)):
            bv = brute(d, (int(q_idx.alpha_strs[ia2]), int(q_idx.beta_strs[ib2])))
            sv = s[ia2, ib2]
            d2 = abs(bv - sv)
            if d2 > maxdiff:
                maxdiff = d2; worst = (d, (int(q_idx.alpha_strs[ia2]), int(q_idx.beta_strs[ib2])), bv, sv)

print(f"max |brute - sigma_full| = {maxdiff:.3e}")
if worst:
    d, dst, bv, sv = worst
    print(f"worst: src={d} dst={dst} brute={bv:+.8f} sigma={sv:+.8f}")
