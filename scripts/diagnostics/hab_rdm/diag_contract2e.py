#!/usr/bin/env python3
"""Compare selected_ci.contract_2e vs direct_spin1.contract_2e vs brute force,
for H_2e acting on a single determinant."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

from pyscf import ao2mo
from pyscf.fci import selected_ci, direct_spin1

h2eff = cas.get_h2eff()
norb = n_act
nelec = (na, nb)

# pick a specific determinant (full CI string)
alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
# pick the HF-like determinant: lowest alpha/beta strings
a_str = int(alpha_strs[0])
b_str = int(beta_strs[0])
ci = np.zeros((len(alpha_strs), len(beta_strs)))
ci[0, 0] = 1.0

# 1) selected_ci.contract_2e (sigma_full path)
ci_with_strs = selected_ci._as_SCIvector(ci.copy(), (alpha_strs, beta_strs))
s_selected = selected_ci.contract_2e(
    h2eff, ci_with_strs, norb, nelec, link_index=backend.q_idx.link_index)

# 2) direct_spin1.contract_2e
eri_packed = ao2mo.restore(1, h2eff, norb)
s_direct = direct_spin1.contract_2e(eri_packed, ci, norb, nelec)

# 3) brute force on this determinant
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

def brute_sigma(src_a, src_b, h2, norb):
    """H_2e |src> = 1/2 sum_{pqrs} sum_st (pq|rs) c_p,st c_q,t a_s,t a_r,st |src>."""
    n_alpha = norb
    # result dict (final det -> coeff)
    result = {}
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    g = h2[p, q, r, s]
                    if abs(g) < 1e-14:
                        continue
                    for sig in ('a', 'b'):
                        for tau in ('a', 'b'):
                            res = apply_op((src_a, src_b), [('a', sig, r), ('a', tau, s),
                                                             ('c', tau, q), ('c', sig, p)])
                            if res is None:
                                continue
                            final, phase = res
                            result[final] = result.get(final, 0.0) + 0.5 * g * phase
    return result

br = brute_sigma(a_str, b_str, h2_4d, norb)

# compare selected vs direct
print("selected vs direct max diff:", np.abs(s_selected - s_direct).max())

# compare brute force to selected (for the diagonal element)
# find the diagonal element of the HF det
ia = 0; ib = 0  # HF det is (0,0) in the CI matrix
print(f"selected[0,0] = {s_selected[0,0]:+.10f}")
print(f"direct[0,0]   = {s_direct[0,0]:+.10f}")
# brute force diagonal element (final == src)
diag = br.get((a_str, b_str), 0.0)
print(f"brute[diag]   = {diag:+.10f}")

# also check a specific off-diagonal: final = (a_str with one electron moved)
# find first nonzero off-diagonal in selected and check brute
print("\nOff-diagonal check (selected vs brute):")
checked = 0
for i in range(len(alpha_strs)):
    for j in range(len(beta_strs)):
        if (i, j) == (0, 0):
            continue
        val_sel = s_selected[i, j]
        if abs(val_sel) > 1e-6:
            fa, fb = int(alpha_strs[i]), int(beta_strs[j])
            val_br = br.get((fa, fb), 0.0)
            print(f"  det({i},{j}): selected={val_sel:+.10f} brute={val_br:+.10f} ratio={val_br/val_sel:+.4f}")
            checked += 1
            if checked >= 5:
                break
    if checked >= 5:
        break
