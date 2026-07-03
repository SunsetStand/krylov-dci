#!/usr/bin/env python3
"""
Phase 4b: P-space strategies in CAS(10,10) — P << Q with sparse methods

Key difference from Phase 4:
  - CAS(10,10): 63,504 determinants (vs 4,900 for CAS(8,8))
  - P=10-200 (PT2-selected) << Q ~63k → SVD M>>N compression
  - Sparse methods: only compute H_QP for Q-neighborhood (connected dets)
  - No full H_QQ build needed
  - Reference: exact CASCI(10,10) in active space

Bond lengths: Re=1.10, 1.5Re=1.65, 2.0Re=2.20, 2.5Re=2.75
"""

import sys, time
import numpy as np

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci.direct_nosym import FCI
from hamiltonian import Hamiltonian, _unpack_4fold
from determinants import (
    bit_positions, excitation_level
)
from krylov import compute_A, modified_gram_schmidt, generate_layer_0
from effective_h import self_consistent_iteration
from svd_compression import build_weighted_coupling, svd_truncate

np.set_printoptions(linewidth=120, precision=6, suppress=True)

N_CORE = 2       # freeze 2 core orbitals (4 electrons)
N_ACT = 10       # 10 active orbitals
N_ELEC_ACT = 10  # 10 active electrons
BOND_LENGTHS = [1.10, 1.65, 2.20, 2.75]
P_SIZES = [10, 25, 50, 100, 200]
PT2_THRESHOLD = 1e-8  # PT2 score cutoff (very permissive)


def setup_system(R):
    """Build N2, run CASCI(10,10), extract active-space Hamiltonian and reference."""
    t0 = time.perf_counter()
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    # CASCI for reference energy
    mycas = mcscf.CASCI(mf, N_ACT, N_ELEC_ACT)
    mycas.frozen = N_CORE
    mycas.verbose = 0
    mycas.kernel()
    E_ref = mycas.e_tot

    # Active-space MO integrals
    mo_act = mycas.mo_coeff[:, N_CORE:N_CORE+N_ACT]
    h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    eri_ao = mol.intor('int2e')
    h1_act = mo_act.T @ h1_ao @ mo_act
    h2_act = _unpack_4fold(ao2mo.incore.full(eri_ao, mo_act), N_ACT)

    # Compute ecore = everything outside active space
    solver = FCI(); solver.verbose = 0
    n_a = N_ELEC_ACT // 2; n_b = N_ELEC_ACT - n_a
    E_active, _ = solver.kernel(h1_act, h2_act, N_ACT, (n_a, n_b), ecore=0.0)
    ecore = E_ref - E_active

    # Hamiltonian in active space
    ham = Hamiltonian(h1=h1_act, h2=h2_act, E_nuc=0.0, E_HF=mf.e_tot)
    t_setup = time.perf_counter() - t0
    return ham, E_ref, ecore, t_setup


def generate_pt2_candidates_in_cas(hf_det, n_orb, n_alpha, n_beta):
    """Enumerate all 1-2 excited determinants from HF within the active space."""
    a_str, b_str = hf_det
    a_occ = bit_positions(a_str); b_occ = bit_positions(b_str)
    a_vir = [p for p in range(n_orb) if p not in a_occ]
    b_vir = [p for p in range(n_orb) if p not in b_occ]
    results = {(a_str, b_str)}

    for i in a_occ:
        for a in a_vir:
            results.add(((a_str ^ (1<<i)) | (1<<a), b_str))
    for i in b_occ:
        for a in b_vir:
            results.add((a_str, (b_str ^ (1<<i)) | (1<<a)))

    if len(a_occ) >= 2:
        for idx_i, i in enumerate(a_occ):
            for j in a_occ[idx_i+1:]:
                for idx_a, va in enumerate(a_vir):
                    for vb in a_vir[idx_a+1:]:
                        results.add((a_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb), b_str))
    if len(b_occ) >= 2:
        for idx_i, i in enumerate(b_occ):
            for j in b_occ[idx_i+1:]:
                for idx_a, va in enumerate(b_vir):
                    for vb in b_vir[idx_a+1:]:
                        results.add((a_str, b_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb)))
    for i in a_occ:
        for j in b_occ:
            for va in a_vir:
                for vb in b_vir:
                    results.add(((a_str ^ (1<<i)) | (1<<va), (b_str ^ (1<<j)) | (1<<vb)))
    return list(results)


def select_p_by_pt2(ham, hf_det, candidates, n_p):
    """PT2 scoring: |<HF|H|i>|^2 / |E_HF - H_ii|. Return top n_p in descending order."""
    E_hf = ham.diagonal_element(*hf_det)
    scores = [(1e10, hf_det)]
    for det_i in candidates:
        if det_i == hf_det: continue
        hij = ham.matrix_element(hf_det, det_i)
        if abs(hij) < PT2_THRESHOLD: continue
        e_i = ham.diagonal_element(*det_i)
        denom = E_hf - e_i
        if abs(denom) < 1e-14: scores.append((1e9, det_i)); continue
        scores.append((abs(hij * hij / denom), det_i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [det for _, det in scores[:n_p]]


def compute_h_qp_sparse(ham, p_dets, q_dets, n_orb):
    """Compute H_QP only for connected (p,q) pairs. Returns (M,N) dense matrix."""
    M, N = len(q_dets), len(p_dets)
    H_QP = np.zeros((M, N))
    q_map = {det: i for i, det in enumerate(q_dets)}

    for p_idx, (a,b) in enumerate(p_dets):
        a_occ = bit_positions(a); b_occ = bit_positions(b)
        a_vir = [p for p in range(n_orb) if p not in a_occ]
        b_vir = [p for p in range(n_orb) if p not in b_occ]

        conn = []
        for i in a_occ:
            for v in a_vir: conn.append(((a ^ (1<<i)) | (1<<v), b))
        for i in b_occ:
            for v in b_vir: conn.append((a, (b ^ (1<<i)) | (1<<v)))

        if len(a_occ) >= 2:
            for idx_i, i in enumerate(a_occ):
                for j in a_occ[idx_i+1:]:
                    for idx_a, va in enumerate(a_vir):
                        for vb in a_vir[idx_a+1:]:
                            conn.append((a ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb), b))
        if len(b_occ) >= 2:
            for idx_i, i in enumerate(b_occ):
                for j in b_occ[idx_i+1:]:
                    for idx_a, va in enumerate(b_vir):
                        for vb in b_vir[idx_a+1:]:
                            conn.append((a, b ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb)))
        for i in a_occ:
            for j in b_occ:
                for va in a_vir:
                    for vb in b_vir:
                        conn.append(((a ^ (1<<i)) | (1<<va), (b ^ (1<<j)) | (1<<vb)))

        for det_q in conn:
            q_idx = q_map.get(det_q)
            if q_idx is not None:
                hij = ham.matrix_element((a,b), det_q)
                if abs(hij) > 1e-14:
                    H_QP[q_idx, p_idx] = hij
    return H_QP


def run_m0(ham, p_dets, q_dets, H_QP_mat, diag_H_QQ, E_ref, ecore):
    """Krylov-dCI m=0 with SVD compression in active space."""
    t0 = time.perf_counter()
    N, M = len(p_dets), len(q_dets)

    # H_PP
    H_PP = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0 = np.linalg.eigh(H_PP)[0][0]

    A_diag = compute_A(E0, diag_H_QQ)
    L0 = H_QP_mat * A_diag[:, np.newaxis]  # layer 0
    T = build_weighted_coupling(L0, A_diag)
    U_comp, sigma, r = svd_truncate(T, threshold=1e-3)

    if r == 0:
        dE = (E0 + ecore - E_ref) * 1000
        return {'n_p': N, 'n_q': M, 'd': 0, 'dE_mH': dE, 't': time.perf_counter()-t0,
                'sigma1': 0, 'sigma_last': 0, 'n_iter': 1}

    U_orth, _ = modified_gram_schmidt(U_comp, np.zeros((M,0)))
    d = U_orth.shape[1]

    # Compressed H blocks (diagonal H_QQ approximation at m=0)
    H_QQ_t = (U_orth * diag_H_QQ[:, None]).T @ U_orth
    H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
    H_PQ_t = (U_orth.T @ H_QP_mat).T

    result = self_consistent_iteration(H_PP, H_PQ_t, H_QQ_t, E0, verbose=False)
    E_total = result['E_final'] + ecore
    dE_mH = (E_total - E_ref) * 1000
    return {'n_p': N, 'n_q': M, 'd': d, 'dE_mH': dE_mH,
            't': time.perf_counter()-t0, 'sigma1': sigma[0], 'sigma_last': sigma[-1],
            'n_iter': result['n_iter']}


def main():
    print("=" * 80)
    print("Phase 4b: CAS(10,10) PT2 P-space (P << Q, sparse, m=0)")
    print("=" * 80)

    for R in BOND_LENGTHS:
        lbl = f"R={R:.2f}"
        if abs(R-1.10)<0.01: lbl+=" Re"
        elif abs(R-1.65)<0.01: lbl+=" 1.5Re"
        elif abs(R-2.20)<0.01: lbl+=" 2.0Re"
        elif abs(R-2.75)<0.01: lbl+=" 2.5Re"
        print(f"\n{'='*60}\n{lbl}\n{'='*60}")

        ham, E_ref, ecore, t_setup = setup_system(R)
        n_act = N_ACT; n_el = N_ELEC_ACT
        n_a = n_el//2; n_b = n_el-n_a
        print(f"  CAS(10,10) dim = 63504, E_ref = {E_ref:.10f}")
        print(f"  Setup: {t_setup:.1f}s", flush=True)

        hf_a = (1<<n_a)-1; hf_b = (1<<n_b)-1; hf_d = (hf_a, hf_b)

        # PT2 candidates in CAS(10,10)
        t0 = time.perf_counter()
        cands = generate_pt2_candidates_in_cas(hf_d, n_act, n_a, n_b)
        print(f"  PT2 candidates: {len(cands)} (in CAS space)", flush=True)

        # Compute PT2 for all candidates
        E_hf = ham.diagonal_element(*hf_d)
        scores = [(1e10, hf_d)]
        for det_i in cands:
            if det_i == hf_d: continue
            hij = ham.matrix_element(hf_d, det_i)
            if abs(hij) < 1e-14: continue
            e_i = ham.diagonal_element(*det_i)
            d = E_hf - e_i
            s = 1e9 if abs(d)<1e-14 else abs(hij*hij/d)
            if s > PT2_THRESHOLD: scores.append((s, det_i))
        scores.sort(key=lambda x: x[0], reverse=True)
        p_pool = [det for _, det in scores]
        print(f"  PT2 pool: {len(p_pool)} dets (score > {PT2_THRESHOLD:.0e}), "
              f"{time.perf_counter()-t0:.1f}s", flush=True)

        print(f"\n  {'P':>5s}  {'Q':>8s}  {'d':>5s}  "
              f"{'ΔE(mH)':>10s}  {'SCF':>4s}  {'t(s)':>7s}  {'σ₁':>8s}  {'σ_last':>8s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}")

        for n_p in P_SIZES:
            if n_p > len(p_pool): continue
            this_p = p_pool[:n_p]
            p_set = set(this_p)

            # Q-neighborhood: all CAS dets connected to any P det (sparse, not full CAS)
            t0 = time.perf_counter()
            q_set = set()
            for det_p in this_p:
                a,b = det_p
                a_occ=bit_positions(a);b_occ=bit_positions(b)
                a_vir=[p for p in range(n_act) if p not in a_occ]
                b_vir=[p for p in range(n_act) if p not in b_occ]
                for i in a_occ:
                    for v in a_vir:
                        nd=((a^(1<<i))|(1<<v),b)
                        if nd not in p_set: q_set.add(nd)
                for i in b_occ:
                    for v in b_vir:
                        nd=(a,(b^(1<<i))|(1<<v))
                        if nd not in p_set: q_set.add(nd)
                if len(a_occ)>=2:
                    for ii,i in enumerate(a_occ):
                        for j in a_occ[ii+1:]:
                            for ia,va in enumerate(a_vir):
                                for vb in a_vir[ia+1:]:
                                    nd=(a^(1<<i)^(1<<j)|(1<<va)|(1<<vb),b)
                                    if nd not in p_set: q_set.add(nd)
                if len(b_occ)>=2:
                    for ii,i in enumerate(b_occ):
                        for j in b_occ[ii+1:]:
                            for ia,va in enumerate(b_vir):
                                for vb in b_vir[ia+1:]:
                                    nd=(a,b^(1<<i)^(1<<j)|(1<<va)|(1<<vb))
                                    if nd not in p_set: q_set.add(nd)
                for i in a_occ:
                    for j in b_occ:
                        for va in a_vir:
                            for vb in b_vir:
                                nd=((a^(1<<i))|(1<<va),(b^(1<<j))|(1<<vb))
                                if nd not in p_set: q_set.add(nd)
            q_dets = list(q_set)
            M = len(q_dets)  # Q-neighborhood size

            # H_QP (sparse — only connected pairs)
            H_QP_mat = compute_h_qp_sparse(ham, this_p, q_dets, n_act)

            # H_D' for Q-neighborhood
            diag = np.array([ham.diagonal_element(a,b) for a,b in q_dets])

            t_qbuild = time.perf_counter() - t0

            r = run_m0(ham, this_p, q_dets, H_QP_mat, diag, E_ref, ecore)
            print(f"  {r['n_p']:5d}  {r['n_q']:8d}  {r['d']:5d}  "
                  f"{r['dE_mH']:+10.1f}  {r['n_iter']:4d}  "
                  f"{r['t']+t_qbuild:7.1f}  {r['sigma1']:8.2e}  "
                  f"{r['sigma_last']:8.2e}")

    print("\n" + "=" * 80)
    print("Phase 4b complete.")
    print("=" * 80)


if __name__ == '__main__':
    main()
