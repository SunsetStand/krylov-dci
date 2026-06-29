#!/usr/bin/env python3
"""
Phase 5b: N₂/cc-pVDZ P-Space Strategy Exploration (PySCF make_hdiag, m=0)

Re-does Phase 4's strategy comparison with PySCF C-level H_D'.
CAS(10,10), all strategies at m=0.

Strategies:
  A. sub-CAS(4,4): 36 dets
  B. sub-CAS(6,6): 400 dets  
  C. PT2 (multiple thresholds)
  D. Energy Window (multiple widths)
  E. Single-det (P=1)
"""

import sys, time, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci, cistring
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H

np.set_printoptions(linewidth=120, precision=6, suppress=True)

N_CORE, N_ACT, N_ELEC_ACT = 2, 10, 10
BOND_LENGTHS = [1.10, 1.65, 2.20, 2.75]

# ============================================================================
# Helpers
# ============================================================================

def get_q_neighborhood_strings(p_dets, norb):
    a_set, b_set = set(), set()
    p_a = {d[0] for d in p_dets}; p_b = {d[1] for d in p_dets}
    for a_str, b_str in p_dets:
        a_occ = [i for i in range(norb) if (a_str>>i)&1]
        b_occ = [i for i in range(norb) if (b_str>>i)&1]
        a_vir = [i for i in range(norb) if i not in a_occ]
        b_vir = [i for i in range(norb) if i not in b_occ]
        nao, nbo = len(a_occ), len(b_occ)
        for i in a_occ:
            for v in a_vir:
                a_set.add((a_str ^ (1<<i)) | (1<<v)); b_set.add(b_str)
        for i in b_occ:
            for v in b_vir:
                a_set.add(a_str); b_set.add((b_str ^ (1<<i)) | (1<<v))
        if nao >= 2:
            for ii,i in enumerate(a_occ):
                for j in a_occ[ii+1:]:
                    for ia,va in enumerate(a_vir):
                        for vb in a_vir[ia+1:]:
                            a_set.add(a_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb))
                            b_set.add(b_str)
        if nbo >= 2:
            for ii,i in enumerate(b_occ):
                for j in b_occ[ii+1:]:
                    for ia,va in enumerate(b_vir):
                        for vb in b_vir[ia+1:]:
                            a_set.add(a_str)
                            b_set.add(b_str ^ (1<<i) ^ (1<<j) | (1<<va) | (1<<vb))
        for i in a_occ:
            for j in b_occ:
                for va in a_vir:
                    for vb in b_vir:
                        a_set.add((a_str ^ (1<<i)) | (1<<va))
                        b_set.add((b_str ^ (1<<j)) | (1<<vb))
    a_set.update(p_a); b_set.update(p_b)
    return sorted(a_set), sorted(b_set)


def compute_h_qp_sparse(ham, p_dets, q_a_strs, q_b_strs, norb):
    M = len(q_a_strs) * len(q_b_strs); N = len(p_dets)
    H_QP = np.zeros((M, N))
    qa_m = {s:i for i,s in enumerate(q_a_strs)}
    qb_m = {s:i for i,s in enumerate(q_b_strs)}
    nb_q = len(q_b_strs)
    pa_s = {d[0] for d in p_dets}; pb_s = {d[1] for d in p_dets}
    for p_idx, (pa,pb) in enumerate(p_dets):
        a_occ=[i for i in range(norb) if (pa>>i)&1]
        b_occ=[i for i in range(norb) if (pb>>i)&1]
        a_vir=[i for i in range(norb) if i not in a_occ]
        b_vir=[i for i in range(norb) if i not in b_occ]
        nao,nbo=len(a_occ),len(b_occ)
        conn=[]
        for i in a_occ:
            for v in a_vir: conn.append(((pa^(1<<i))|(1<<v), pb))
        for i in b_occ:
            for v in b_vir: conn.append((pa, (pb^(1<<i))|(1<<v)))
        if nao>=2:
            for ii,i in enumerate(a_occ):
                for j in a_occ[ii+1:]:
                    for ia,va in enumerate(a_vir):
                        for vb in a_vir[ia+1:]:
                            conn.append((pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb), pb))
        if nbo>=2:
            for ii,i in enumerate(b_occ):
                for j in b_occ[ii+1:]:
                    for ia,va in enumerate(b_vir):
                        for vb in b_vir[ia+1:]:
                            conn.append((pa, pb^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
        for i in a_occ:
            for j in b_occ:
                for va in a_vir:
                    for vb in b_vir:
                        conn.append(((pa^(1<<i))|(1<<va), (pb^(1<<j))|(1<<vb)))
        for qa,qb in conn:
            ia=qa_m.get(qa); ib=qb_m.get(qb)
            if ia is not None and ib is not None:
                if qa in pa_s and qb in pb_s: continue
                hij=ham.matrix_element((pa,pb),(qa,qb))
                if abs(hij)>1e-14: H_QP[ia*nb_q+ib, p_idx]=hij
    return H_QP


def run_m0(ham, h1e, eri, p_dets, q_a_strs, q_b_strs, norb, nelec, E_ref, ecore):
    t0=time.perf_counter()
    N=len(p_dets); nb_q=len(q_b_strs); M=len(q_a_strs)*nb_q

    ci_strs=(np.asarray(q_a_strs,dtype=np.int64), np.asarray(q_b_strs,dtype=np.int64))
    hdiag_q=selected_ci.make_hdiag(h1e, eri, ci_strs, norb, nelec[:2])

    H_PP=np.zeros((N,N))
    for i in range(N):
        for j in range(N): H_PP[i,j]=ham.matrix_element(p_dets[i],p_dets[j])
    E0=np.linalg.eigh(H_PP)[0][0]

    H_QP_mat=compute_h_qp_sparse(ham, p_dets, q_a_strs, q_b_strs, norb)
    nnz=np.count_nonzero(np.abs(H_QP_mat)>1e-14)

    A_diag=1.0/np.maximum(np.abs(E0-hdiag_q),1e-12)
    L0=H_QP_mat*A_diag[:,np.newaxis]
    T=build_weighted_coupling(L0,A_diag)
    U_comp,sigma,r=svd_truncate(T,threshold=1e-3)

    if r==0:
        dE=(E0+ecore-E_ref)*1000
        return {'n_p':N,'n_q':M,'d':0,'dE_mH':dE,'t':time.perf_counter()-t0,
                'sigma1':0,'sigma_last':0,'n_iter':1,'nnz':nnz}

    U_orth,_=modified_gram_schmidt(U_comp,np.zeros((M,0)))
    d=U_orth.shape[1]
    H_QQ_t=(U_orth*hdiag_q[:,None]).T@U_orth; H_QQ_t=0.5*(H_QQ_t+H_QQ_t.T)
    H_PQ_t=(U_orth.T@H_QP_mat).T
    # m=0: no B matrix, no Δ. Single-shot effective H diagonalization at Δ=0.
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
    eigvals, _ = diagonalize_effective_H(H_eff)
    E_total = eigvals[0] + ecore
    dE_mH = (E_total - E_ref) * 1000
    return {'n_p':N,'n_q':M,'d':d,'dE_mH':dE_mH,'t':time.perf_counter()-t0,
            'sigma1':sigma[0],'sigma_last':sigma[-1],'n_iter':1,'nnz':nnz}


# ============================================================================
# Strategy functions
# ============================================================================

def strat_sub_cas(ham, dets, norb, nelec_total, sub_n_orb, sub_n_elec):
    """P = full CAS(sub_n_orb, sub_n_elec) within active space."""
    from partitioning import partition_cas
    p_idx, q_idx = partition_cas(norb, nelec_total, n_active_orb=sub_n_orb, n_active_elec=sub_n_elec)
    return [dets[i] for i in p_idx]

def strat_single_det(dets, hf_det):
    return [hf_det]

def strat_energy_window(ham, dets, E_hf, de):
    p = []
    for d in dets:
        if abs(ham.diagonal_element(*d) - E_hf) < de:
            p.append(d)
    return p

def strat_pt2(ham, hf_det, candidates, threshold):
    E_hf = ham.diagonal_element(*hf_det)
    scores = [(1e10, hf_det)]
    for det_i in candidates:
        if det_i == hf_det: continue
        hij = ham.matrix_element(hf_det, det_i)
        if abs(hij) < 1e-14: continue
        e_i = ham.diagonal_element(*det_i)
        d = E_hf - e_i
        s = 1e9 if abs(d) < 1e-14 else abs(hij*hij/d)
        if s > threshold: scores.append((s, det_i))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scores]


# ============================================================================
# Generate PT2 candidates (SD excitations from HF in CAS)
# ============================================================================

def generate_sd_candidates(a_occ, b_occ, norb):
    a_str=sum(1<<p for p in a_occ); b_str=sum(1<<p for p in b_occ)
    a_vir=[p for p in range(norb) if p not in a_occ]
    b_vir=[p for p in range(norb) if p not in b_occ]
    res={(a_str,b_str)}
    for i in a_occ:
        for a in a_vir: res.add(((a_str^(1<<i))|(1<<a), b_str))
    for i in b_occ:
        for a in b_vir: res.add((a_str, (b_str^(1<<i))|(1<<a)))
    if len(a_occ)>=2:
        for ii,i in enumerate(a_occ):
            for j in a_occ[ii+1:]:
                for ia,va in enumerate(a_vir):
                    for vb in a_vir[ia+1:]:
                        res.add((a_str^(1<<i)^(1<<j)|(1<<va)|(1<<vb), b_str))
    if len(b_occ)>=2:
        for ii,i in enumerate(b_occ):
            for j in b_occ[ii+1:]:
                for ia,va in enumerate(b_vir):
                    for vb in b_vir[ia+1:]:
                        res.add((a_str, b_str^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
    for i in a_occ:
        for j in b_occ:
            for va in a_vir:
                for vb in b_vir:
                    res.add(((a_str^(1<<i))|(1<<va), (b_str^(1<<j))|(1<<vb)))
    return list(res)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Phase 5b: P-Space Strategy Exploration — PySCF make_hdiag, CAS(10,10), m=0")
    print("=" * 80)

    for R in BOND_LENGTHS:
        lbl = f"R={R:.2f}"
        if abs(R-1.10)<0.01: lbl+=" Re"
        elif abs(R-1.65)<0.01: lbl+=" 1.5Re"
        elif abs(R-2.20)<0.01: lbl+=" 2.0Re"
        else: lbl+=" 2.5Re"
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
        eri_packed = ao2mo.restore(1, ao2mo.incore.full(eri_ao, mo_act), N_ACT)

        from pyscf.fci.direct_nosym import FCI
        s=FCI(); s.verbose=0
        na=N_ELEC_ACT//2; nb=N_ELEC_ACT-na
        E_active,_=s.kernel(h1_act,h2_act,N_ACT,(na,nb),ecore=0.0)
        ecore=E_ref-E_active

        ham=Hamiltonian(h1=h1_act,h2=h2_act,E_nuc=0.0,E_HF=mf.e_tot)
        norb,nelec=N_ACT,(na,nb)

        # Generate all CAS dets (for EW strategy)
        from determinants import generate_determinants_ms
        all_dets=generate_determinants_ms(norb, N_ELEC_ACT, ms=0)

        hf_ao=list(range(na)); hf_bo=list(range(nb))
        hf_a=sum(1<<p for p in hf_ao); hf_b=sum(1<<p for p in hf_bo)
        hf_det=(hf_a,hf_b); E_hf=ham.diagonal_element(hf_a,hf_b)
        cands=generate_sd_candidates(hf_ao,hf_bo,norb)

        print(f"  CAS(10,10) E_ref={E_ref:.10f}  setup={time.perf_counter()-t_start:.1f}s",
              flush=True)

        strategies = []

        # A: CAS(4,4) = 36 dets
        sp = strat_sub_cas(ham, all_dets, norb, N_ELEC_ACT, 4, 4)
        strategies.append(("A: CAS(4,4)    ", sp))

        # B: CAS(6,6) = 400 dets
        sp = strat_sub_cas(ham, all_dets, norb, N_ELEC_ACT, 6, 6)
        strategies.append(("B: CAS(6,6)    ", sp))

        # C: Energy Window (skip if too wide → Q too large)
        for de in [0.5, 1.0, 2.0]:
            sp = strat_energy_window(ham, all_dets, E_hf, de)
            if len(sp) > 1:
                strategies.append((f"C: EW {de:.1f}Ha  ", sp))

        # D: PT2
        for th in [1e-5, 1e-4, 1e-3, 1e-2]:
            sp = strat_pt2(ham, hf_det, cands, th)
            if len(sp) > 1:
                strategies.append((f"D: PT2 {th:.0e}", sp))

        # E: Single-det
        sp = strat_single_det(all_dets, hf_det)
        strategies.append(("E: Single-det  ", sp))

        print(f"  {'Strategy':20s} {'P':>5s} {'Q':>8s} {'d':>5s} {'ΔE(mH)':>10s} "
              f"{'SCF':>4s} {'t(s)':>7s} {'nnz':>7s}")
        print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*5} {'-'*10} {'-'*4} {'-'*7} {'-'*7}")

        for name, p_dets in strategies:
            # Skip if P is too large (would be too expensive)
            if len(p_dets) > 500:
                print(f"  {name:20s} {'--':>5s} {'(P too large, skipped)':>30s}")
                continue
            
            t0 = time.perf_counter()
            q_a, q_b = get_q_neighborhood_strings(p_dets, norb)
            t_qgen = time.perf_counter() - t0
            
            nq = len(q_a) * len(q_b)
            if nq > 100000:
                print(f"  {name:20s} {'--':>5s} {'(Q={:d} too large, skipped)':>30s}".format(nq))
                continue

            r = run_m0(ham, h1_act, eri_packed, p_dets, q_a, q_b, norb, nelec, E_ref, ecore)
            t_total = r['t'] + t_qgen

            print(f"  {name:20s} {r['n_p']:5d} {r['n_q']:8d} {r['d']:5d} "
                  f"{r['dE_mH']:+10.1f} {r['n_iter']:4d} {t_total:7.1f} {r['nnz']:7d}")

    print("\n" + "=" * 80 + "\nPhase 5b complete.\n" + "=" * 80)


if __name__ == '__main__':
    main()
