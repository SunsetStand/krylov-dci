#!/usr/bin/env python3
"""
Phase 5: N₂/cc-pVDZ Krylov-dCI using PySCF make_hdiag (C-level H_D')

Key improvements over hand-rolled code:
  - make_hdiag: H_D' at C level (~0.002us/det, 24,000x faster than Python SC)
  - H_QP: sparse enumeration (only connected pairs, Python SC, ~0.3s for P=100)
  - All matrix ops: NumPy matmul (BLAS)

All SLURM-submitted. No direct SSH runs for production tests.
"""

import sys, time, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci, cistring
from hamiltonian import Hamiltonian, _unpack_4fold

np.set_printoptions(linewidth=120, precision=6, suppress=True)


def make_string(s, val): return val  # PySCF string is just an int bitmask (same as ours)


def get_q_neighborhood_strings(p_dets, norb):
    """Collect alpha and beta strings connected to any P det by 1-2 excitations."""
    a_set, b_set = set(), set()
    p_a_set = {d[0] for d in p_dets}
    p_b_set = {d[1] for d in p_dets}
    
    for a_str, b_str in p_dets:
        a_occ = [i for i in range(norb) if (a_str>>i)&1]
        b_occ = [i for i in range(norb) if (b_str>>i)&1]
        a_vir = [i for i in range(norb) if i not in a_occ]
        b_vir = [i for i in range(norb) if i not in b_occ]
        nao, nbo = len(a_occ), len(b_occ)
        
        for i in a_occ:
            for v in a_vir:
                a_set.add((a_str ^ (1<<i)) | (1<<v))
                b_set.add(b_str)
        for i in b_occ:
            for v in b_vir:
                a_set.add(a_str)
                b_set.add((b_str ^ (1<<i)) | (1<<v))
        
        if nao >= 2:
            for ii, i in enumerate(a_occ):
                for j in a_occ[ii+1:]:
                    for ia, va in enumerate(a_vir):
                        for vb in a_vir[ia+1:]:
                            a_set.add(a_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb))
                            b_set.add(b_str)
        if nbo >= 2:
            for ii, i in enumerate(b_occ):
                for j in b_occ[ii+1:]:
                    for ia, va in enumerate(b_vir):
                        for vb in b_vir[ia+1:]:
                            a_set.add(a_str)
                            b_set.add(b_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb))
        for i in a_occ:
            for j in b_occ:
                for va in a_vir:
                    for vb in b_vir:
                        a_set.add((a_str ^ (1<<i)) | (1<<va))
                        b_set.add((b_str ^ (1<<j)) | (1<<vb))
    
    a_set.update(p_a_set); b_set.update(p_b_set)
    return sorted(a_set), sorted(b_set)


def compute_h_qp_sparse(ham, p_dets, q_a_strs, q_b_strs, norb):
    """H_QP via sparse connected-pair Slater-Condon. Only for connected (p,q)."""
    M = len(q_a_strs) * len(q_b_strs)
    N = len(p_dets)
    H_QP = np.zeros((M, N))
    q_a_map = {s: i for i, s in enumerate(q_a_strs)}
    q_b_map = {s: i for i, s in enumerate(q_b_strs)}
    nb_q = len(q_b_strs)
    p_a_set = {d[0] for d in p_dets}
    p_b_set = {d[1] for d in p_dets}
    
    for p_idx, (pa, pb) in enumerate(p_dets):
        a_occ = [i for i in range(norb) if (pa>>i)&1]
        b_occ = [i for i in range(norb) if (pb>>i)&1]
        a_vir = [i for i in range(norb) if i not in a_occ]
        b_vir = [i for i in range(norb) if i not in b_occ]
        nao, nbo = len(a_occ), len(b_occ)
        
        conn = []
        for i in a_occ:
            for v in a_vir: conn.append(((pa ^ (1<<i)) | (1<<v), pb))
        for i in b_occ:
            for v in b_vir: conn.append((pa, (pb ^ (1<<i)) | (1<<v)))
        if nao >= 2:
            for ii, i in enumerate(a_occ):
                for j in a_occ[ii+1:]:
                    for ia, va in enumerate(a_vir):
                        for vb in a_vir[ia+1:]:
                            conn.append((pa ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb), pb))
        if nbo >= 2:
            for ii, i in enumerate(b_occ):
                for j in b_occ[ii+1:]:
                    for ia, va in enumerate(b_vir):
                        for vb in b_vir[ia+1:]:
                            conn.append((pa, pb ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb)))
        for i in a_occ:
            for j in b_occ:
                for va in a_vir:
                    for vb in b_vir:
                        conn.append(((pa ^ (1<<i)) | (1<<va), (pb ^ (1<<j)) | (1<<vb)))
        
        for (qa, qb) in conn:
            ia = q_a_map.get(qa); ib = q_b_map.get(qb)
            if ia is not None and ib is not None:
                # Skip P×P pairs (these belong to H_PP, not H_QP)
                if qa in p_a_set and qb in p_b_set:
                    continue
                hij = ham.matrix_element((pa,pb), (qa,qb))
                if abs(hij) > 1e-14:
                    H_QP[ia * nb_q + ib, p_idx] = hij
    return H_QP


def generate_pt2_candidates(a_occ, b_occ, norb):
    """Enumerate SD excitations from HF occupations."""
    a_str = sum(1<<p for p in a_occ); b_str = sum(1<<p for p in b_occ)
    a_vir = [p for p in range(norb) if p not in a_occ]
    b_vir = [p for p in range(norb) if p not in b_occ]
    results = {(a_str, b_str)}
    
    for i in a_occ:
        for a in a_vir: results.add(((a_str ^ (1<<i)) | (1<<a), b_str))
    for i in b_occ:
        for a in b_vir: results.add((a_str, (b_str ^ (1<<i)) | (1<<a)))
    if len(a_occ)>=2:
        for ii,i in enumerate(a_occ):
            for j in a_occ[ii+1:]:
                for ia,va in enumerate(a_vir):
                    for vb in a_vir[ia+1:]:
                        results.add((a_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb), b_str))
    if len(b_occ)>=2:
        for ii,i in enumerate(b_occ):
            for j in b_occ[ii+1:]:
                for ia,va in enumerate(b_vir):
                    for vb in b_vir[ia+1:]:
                        results.add((a_str, b_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb)))
    for i in a_occ:
        for j in b_occ:
            for va in a_vir:
                for vb in b_vir:
                    results.add(((a_str ^ (1<<i)) | (1<<va), (b_str ^ (1<<j)) | (1<<vb)))
    return list(results)


def score_pt2(ham, hf_det, candidates):
    E_hf = ham.diagonal_element(*hf_det)
    scores = [(1e10, hf_det)]
    for det_i in candidates:
        if det_i == hf_det: continue
        hij = ham.matrix_element(hf_det, det_i)
        if abs(hij) < 1e-14: continue
        e_i = ham.diagonal_element(*det_i)
        d = E_hf - e_i
        s = 1e9 if abs(d) < 1e-14 else abs(hij*hij/d)
        if s > 1e-8: scores.append((s, det_i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scores]


def run_m0(ham, h1e, eri, p_dets, q_a_strs, q_b_strs, norb, nelec, E_ref, ecore):
    t0 = time.perf_counter()
    N = len(p_dets)
    nb_q = len(q_b_strs)
    M = len(q_a_strs) * nb_q
    
    # H_D' via PySCF make_hdiag (C-level!!!)
    ci_strs = (np.asarray(q_a_strs, dtype=np.int64),
               np.asarray(q_b_strs, dtype=np.int64))
    hdiag_q = selected_ci.make_hdiag(h1e, eri, ci_strs, norb, nelec[:2])
    
    # H_PP via Python SC (small)
    H_PP = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0 = np.linalg.eigh(H_PP)[0][0]
    
    # H_QP via sparse SC
    H_QP_mat = compute_h_qp_sparse(ham, p_dets, q_a_strs, q_b_strs, norb)
    nnz = np.count_nonzero(np.abs(H_QP_mat) > 1e-14)
    
    # A and layer 0
    A_diag = 1.0 / np.maximum(np.abs(E0 - hdiag_q), 1e-12)
    L0 = H_QP_mat * A_diag[:, np.newaxis]
    
    # Weighted SVD
    from svd_compression import build_weighted_coupling, svd_truncate
    from krylov import modified_gram_schmidt
    from effective_h import self_consistent_iteration
    
    T = build_weighted_coupling(L0, A_diag)
    U_comp, sigma, r = svd_truncate(T, threshold=1e-3)
    
    if r == 0:
        dE = (E0 + ecore - E_ref) * 1000
        return {'n_p': N, 'n_q': M, 'd': 0, 'dE_mH': dE, 't': time.perf_counter()-t0,
                'sigma1': 0, 'sigma_last': 0, 'n_iter': 1, 'nnz': nnz}
    
    U_orth, _ = modified_gram_schmidt(U_comp, np.zeros((M,0)))
    d = U_orth.shape[1]
    
    H_QQ_t = (U_orth * hdiag_q[:, None]).T @ U_orth
    H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
    H_PQ_t = (U_orth.T @ H_QP_mat).T
    
    result = self_consistent_iteration(H_PP, H_PQ_t, H_QQ_t, E0, verbose=False)
    E_total = result['E_final'] + ecore
    dE_mH = (E_total - E_ref) * 1000
    return {'n_p': N, 'n_q': M, 'd': d, 'dE_mH': dE_mH, 't': time.perf_counter()-t0,
            'sigma1': sigma[0], 'sigma_last': sigma[-1], 'n_iter': result['n_iter'], 'nnz': nnz}


def main():
    print("=" * 80)
    print("Phase 5: N₂/cc-pVDZ — PySCF make_hdiag + sparse H_QP (m=0, SLURM)")
    print("=" * 80)
    
    BOND_LENGTHS = [1.10, 1.65, 2.20, 2.75]
    P_SIZES = [10, 25, 50, 100]
    N_CORE, N_ACT, N_ELEC_ACT = 2, 10, 10
    
    for R in BOND_LENGTHS:
        lbl = f"R={R:.2f}"
        if abs(R-1.10)<0.01: lbl+=" Re"
        elif abs(R-1.65)<0.01: lbl+=" 1.5Re"
        elif abs(R-2.20)<0.01: lbl+=" 2.0Re"
        elif abs(R-2.75)<0.01: lbl+=" 2.5Re"
        print(f"\n{'='*60}\n{lbl}\n{'='*60}")
        
        t_start = time.perf_counter()
        mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
        mf = scf.RHF(mol); mf.kernel()
        
        mycas = mcscf.CASCI(mf, N_ACT, N_ELEC_ACT)
        mycas.frozen = N_CORE; mycas.verbose = 0; mycas.kernel()
        E_ref = mycas.e_tot
        
        mo_act = mycas.mo_coeff[:, N_CORE:N_CORE+N_ACT]
        h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        eri_ao = mol.intor('int2e')
        h1_act = mo_act.T @ h1_ao @ mo_act
        h2_act = _unpack_4fold(ao2mo.incore.full(eri_ao, mo_act), N_ACT)
        eri_act_packed = ao2mo.restore(1, ao2mo.incore.full(eri_ao, mo_act), N_ACT)
        
        from pyscf.fci.direct_nosym import FCI
        solver = FCI(); solver.verbose = 0
        n_a = N_ELEC_ACT//2; n_b = N_ELEC_ACT - n_a
        E_active, _ = solver.kernel(h1_act, h2_act, N_ACT, (n_a,n_b), ecore=0.0)
        ecore = E_ref - E_active
        
        ham = Hamiltonian(h1=h1_act, h2=h2_act, E_nuc=0.0, E_HF=mf.e_tot)
        norb, nelec = N_ACT, (n_a, n_b)
        
        print(f"  CAS(10,10) E_ref={E_ref:.10f}  setup={time.perf_counter()-t_start:.1f}s",
              flush=True)
        
        # PT2
        t0 = time.perf_counter()
        hf_ao = list(range(n_a)); hf_bo = list(range(n_b))
        hf_a = sum(1<<p for p in hf_ao); hf_b = sum(1<<p for p in hf_bo)
        hf_det = (hf_a, hf_b)
        cands = generate_pt2_candidates(hf_ao, hf_bo, norb)
        p_pool = score_pt2(ham, hf_det, cands)
        print(f"  PT2 pool: {len(p_pool)} ({time.perf_counter()-t0:.1f}s)", flush=True)
        
        print(f"\n  {'P':>5s}  {'Q':>8s}  {'d':>5s}  {'ΔE(mH)':>10s}  "
              f"{'SCF':>4s}  {'t(s)':>7s}  {'σ₁':>8s}  {'nnz':>8s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}")
        
        for n_p in P_SIZES:
            if n_p > len(p_pool): continue
            this_p = p_pool[:n_p]
            
            t0 = time.perf_counter()
            q_a_strs, q_b_strs = get_q_neighborhood_strings(this_p, norb)
            t_qgen = time.perf_counter() - t0
            
            r = run_m0(ham, h1_act, eri_act_packed, this_p, q_a_strs, q_b_strs,
                       norb, nelec, E_ref, ecore)
            t_total = r['t'] + t_qgen
            
            print(f"  {r['n_p']:5d}  {r['n_q']:8d}  {r['d']:5d}  "
                  f"{r['dE_mH']:+10.1f}  {r['n_iter']:4d}  "
                  f"{t_total:7.1f}  {r['sigma1']:8.2e}  {r['nnz']:8d}")
    
    print("\n" + "=" * 80 + "\nPhase 5 complete.\n" + "=" * 80)


if __name__ == '__main__':
    main()
