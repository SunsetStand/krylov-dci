#!/usr/bin/env python3
"""
Phase 6b: Krylov Convergence — FIXED propagation (AB acts on compressed basis, not raw)

Fix: prev_compressed = U_orth (SVD+MGS output), NOT L0/propagated (full raw vectors).
This ensures d_layer should decline as the subspace converges.

System: N₂/cc-pVDZ, CAS(8,8), PT2 P=100, Re only
"""

import sys, time, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import build_effective_H, diagonalize_effective_H
from determinants import bit_positions

np.set_printoptions(linewidth=120, precision=6, suppress=True)

N_CORE, N_ACT, N_ELEC_ACT = 3, 8, 8
BOND_LENGTHS = [1.10]  # Re only — stretched bonds need larger P (per static correlation)
P_TARGET = 100
SVD_THRESHOLD = 1e-3
MAX_KRYLOV = 5  # m=0..5 to see convergence behaviour
LEVEL_SHIFT = 0.3


# ============================================================================
# Helpers (same as Phase 6)
# ============================================================================

def get_q_strings(p_dets, norb):
    a_set, b_set = set(), set()
    p_a = {d[0] for d in p_dets}; p_b = {d[1] for d in p_dets}
    for a_str, b_str in p_dets:
        a_occ = [i for i in range(norb) if (a_str>>i)&1]
        b_occ = [i for i in range(norb) if (b_str>>i)&1]
        a_vir = [i for i in range(norb) if i not in a_occ]
        b_vir = [i for i in range(norb) if i not in b_occ]
        nao, nbo = len(a_occ), len(b_occ)
        for i in a_occ:
            for v in a_vir: a_set.add((a_str ^ (1<<i)) | (1<<v)); b_set.add(b_str)
        for i in b_occ:
            for v in b_vir: a_set.add(a_str); b_set.add((b_str ^ (1<<i)) | (1<<v))
        if nao >= 2:
            for ii,i in enumerate(a_occ):
                for j in a_occ[ii+1:]:
                    for ia,va in enumerate(a_vir):
                        for vb in b_vir[ia+1:]:
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


def build_sparse_h_qq(ham, q_a_strs, q_b_strs, norb):
    na, nb = len(q_a_strs), len(q_b_strs)
    M = na * nb
    off_diag = [[] for _ in range(M)]
    qa_list = list(q_a_strs); qb_list = list(q_b_strs)
    qa_map = {s: i for i, s in enumerate(qa_list)}
    qb_map = {s: i for i, s in enumerate(qb_list)}
    nnz = 0
    for idx_a in range(na):
        for idx_b in range(nb):
            i = idx_a * nb + idx_b
            a_str, b_str = qa_list[idx_a], qb_list[idx_b]
            a_occ = [p for p in range(norb) if (a_str>>p)&1]
            b_occ = [p for p in range(norb) if (b_str>>p)&1]
            a_vir = [p for p in range(norb) if p not in a_occ]
            b_vir = [p for p in range(norb) if p not in b_occ]
            nao, nbo = len(a_occ), len(b_occ)
            connected = set()
            for ii in a_occ:
                for v in a_vir:
                    connected.add((((a_str^(1<<ii))|(1<<v), b_str)))
            for ii in b_occ:
                for v in b_vir:
                    connected.add(((a_str, (b_str^(1<<ii))|(1<<v))))
            if nao >= 2:
                for ii,ia in enumerate(a_occ):
                    for j in a_occ[ii+1:]:
                        for iaa,va in enumerate(a_vir):
                            for vb in b_vir[iaa+1:]:
                                connected.add(((a_str^(1<<ia)^(1<<j)|(1<<va)|(1<<vb), b_str)))
            if nbo >= 2:
                for ii,ia in enumerate(b_occ):
                    for j in b_occ[ii+1:]:
                        for iaa,va in enumerate(b_vir):
                            for vb in b_vir[iaa+1:]:
                                connected.add(((a_str, b_str^(1<<ia)^(1<<j)|(1<<va)|(1<<vb))))
            for ii in a_occ:
                for j in b_occ:
                    for va in a_vir:
                        for vb in b_vir:
                            connected.add((((a_str^(1<<ii))|(1<<va), (b_str^(1<<j))|(1<<vb))))
            for (qa_str, qb_str) in connected:
                ja = qa_map.get(qa_str); jb = qb_map.get(qb_str)
                if ja is not None and jb is not None:
                    j = ja * nb + jb
                    if j > i:
                        hij = ham.matrix_element((a_str, b_str), (qa_str, qb_str))
                        if abs(hij) > 1e-14:
                            off_diag[i].append((j, hij))
                            nnz += 1
    return off_diag, nnz


def sigma_h_qq_sparse(off_diag, diag, vec):
    M = len(diag)
    result = diag * vec
    for i in range(M):
        for (j, hij) in off_diag[i]:
            result[i] += hij * vec[j]
            result[j] += hij * vec[i]
    return result


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
                        for vb in b_vir[ia+1:]:
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
        for qa_str,qb_str in conn:
            ia=qa_m.get(qa_str); ib=qb_m.get(qb_str)
            if ia is not None and ib is not None:
                if qa_str in pa_s and qb_str in pb_s: continue
                hij=ham.matrix_element((pa,pb),(qa_str,qb_str))
                if abs(hij)>1e-14: H_QP[ia*nb_q+ib, p_idx]=hij
    return H_QP


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
                    for vb in b_vir[ia+1:]:
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
    print("Phase 6b: Krylov Convergence — FIXED propagation, sparse H_QQ, CAS(8,8)")
    print("=" * 80)

    for R in BOND_LENGTHS:
        lbl = f"R={R:.2f} Re"
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

        # PT2 P-space
        hf_ao=list(range(na)); hf_bo=list(range(nb))
        hf_a=sum(1<<p for p in hf_ao); hf_b=sum(1<<p for p in hf_bo)
        hf_det=(hf_a,hf_b); E_hf=ham.diagonal_element(hf_a,hf_b)
        cands=generate_sd_candidates(hf_ao,hf_bo,norb)
        scores=[(1e10,hf_det)]
        for d in cands:
            if d==hf_det: continue
            hij=ham.matrix_element(hf_det,d)
            if abs(hij)<1e-14: continue
            ei=ham.diagonal_element(*d);de=E_hf-ei
            ss=1e9 if abs(de)<1e-14 else abs(hij*hij/de)
            if ss>1e-5: scores.append((ss,d))
        scores.sort(key=lambda x:x[0],reverse=True)
        p_dets=[d for _,d in scores[:min(P_TARGET,len(scores))]]
        N=len(p_dets)

        # Q-neighborhood strings
        qa,qb=get_q_strings(p_dets,norb)
        nb_q=len(qb); M=len(qa)*nb_q

        # H_D' via PySCF
        ci_strs=(np.asarray(qa,dtype=np.int64),np.asarray(qb,dtype=np.int64))
        hdiag=selected_ci.make_hdiag(h1_act,eri_packed,ci_strs,norb,nelec[:2])

        # H_QP sparse
        t0=time.perf_counter()
        H_QP=compute_h_qp_sparse(ham,p_dets,qa,qb,norb)
        t_hqp=time.perf_counter()-t0

        # H_PP
        H_PP=np.zeros((N,N))
        for i in range(N):
            for j in range(N): H_PP[i,j]=ham.matrix_element(p_dets[i],p_dets[j])
        E0=np.linalg.eigh(H_PP)[0][0]
        A_diag=1.0/(E0 - hdiag + LEVEL_SHIFT)

        # Build sparse H_QQ
        print(f"  CAS(8,8), P={N}, Q={M}, E_ref={E_ref:.10f}")
        t0=time.perf_counter()
        off_diag,nnz_hqq=build_sparse_h_qq(ham,qa,qb,norb)
        t_hqq=time.perf_counter()-t0
        print(f"  H_QQ sparse: {nnz_hqq} off-diag pairs ({t_hqq:.1f}s)", flush=True)
        print(f"  Setup total: {time.perf_counter()-t_start:.1f}s", flush=True)

        # ============================================================
        # Krylov layers — FIXED: propagate compressed basis, not raw
        # ============================================================
        print(f"\n  {'m':>3s}  {'d_basis':>7s}  {'d_layer':>7s}  "
              f"{'ΔE(mH)':>10s}  {'t(s)':>7s}")
        print(f"  {'-'*3}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*7}")

        accumulated_basis = np.zeros((M, 0))
        prev_compressed = None  # FIXED: compressed (SVD+MGS) vectors for next layer

        for m in range(MAX_KRYLOV + 1):
            t_layer = time.perf_counter()

            if m == 0:
                # Layer 0: L0 = A·H_QP  (M × N)
                L0 = H_QP * A_diag[:, np.newaxis]
                T = build_weighted_coupling(L0, A_diag)
                U_comp, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
                if r == 0:
                    dE = ((E0 + ecore) - E_ref) * 1000
                    print(f"  {m:3d}  {0:7d}  {0:7d}  {dE:+10.1f}  {time.perf_counter()-t_layer:7.1f}")
                    continue
            else:
                # FIXED: propagate = A·H_O'·(prev_compressed)
                # prev_compressed: (M, d_{m-1}) — SVD+MGS compressed basis from layer m-1
                d_prev = prev_compressed.shape[1]
                propagated = np.zeros((M, d_prev))
                for k in range(d_prev):
                    # H_O'·v = H_QQ·v - H_D'·v
                    propagated[:, k] = A_diag * (
                        sigma_h_qq_sparse(off_diag, hdiag, prev_compressed[:, k])
                        - hdiag * prev_compressed[:, k]
                    )
                T = build_weighted_coupling(propagated, A_diag)
                U_comp, sigma, r = svd_truncate(T, threshold=SVD_THRESHOLD)
                if r == 0:
                    dE = _compute_dE(accumulated_basis, H_PP, H_QP, off_diag, hdiag,
                                    E0, ecore, E_ref, LEVEL_SHIFT)
                    print(f"  {m:3d}  {accumulated_basis.shape[1]:7d}  {0:7d}  "
                          f"{dE:+10.1f}  {time.perf_counter()-t_layer:7.1f} (exhausted)")
                    break

            # Gram-Schmidt against accumulated basis
            U_orth, retained = modified_gram_schmidt(U_comp, accumulated_basis)
            d_layer = U_orth.shape[1]

            if d_layer == 0:
                dE = _compute_dE(accumulated_basis, H_PP, H_QP, off_diag, hdiag,
                                E0, ecore, E_ref, LEVEL_SHIFT)
                print(f"  {m:3d}  {accumulated_basis.shape[1]:7d}  {0:7d}  "
                      f"{dE:+10.1f}  {time.perf_counter()-t_layer:7.1f} (lindep)")
                break

            # Append to accumulated basis
            accumulated_basis = np.hstack([accumulated_basis, U_orth])
            d_total = accumulated_basis.shape[1]

            # Build effective H with compressed basis
            sigma_basis = np.zeros((M, d_total))
            for k in range(d_total):
                sigma_basis[:, k] = sigma_h_qq_sparse(
                    off_diag, hdiag, accumulated_basis[:, k]
                )
            H_QQ_t = accumulated_basis.T @ sigma_basis
            H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
            H_PQ_t = (accumulated_basis.T @ H_QP).T

            E_shifted = E0 + LEVEL_SHIFT
            H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E_shifted, delta=0.0)
            eigvals, _ = diagonalize_effective_H(H_eff)
            E_method = eigvals[0] + ecore
            dE_mH = (E_method - E_ref) * 1000

            print(f"  {m:3d}  {d_total:7d}  {d_layer:7d}  "
                  f"{dE_mH:+10.1f}  {time.perf_counter()-t_layer:7.1f}")

            # FIXED: save compressed basis (not raw) for next layer propagation
            if m < MAX_KRYLOV:
                prev_compressed = U_orth

    print("\n" + "=" * 80 + "\nPhase 6b complete.\n" + "=" * 80)


def _compute_dE(basis, H_PP, H_QP, off_diag, hdiag, E0, ecore, E_ref, level_shift):
    """Compute ΔE for a given accumulated basis (used in early-exit paths)."""
    d = basis.shape[1]
    if d == 0:
        return ((E0 + ecore) - E_ref) * 1000
    sigma_basis = np.zeros((basis.shape[0], d))
    for k in range(d):
        sigma_basis[:, k] = sigma_h_qq_sparse(off_diag, hdiag, basis[:, k])
    H_QQ_t = basis.T @ sigma_basis
    H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
    H_PQ_t = (basis.T @ H_QP).T
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0 + level_shift, delta=0.0)
    eigvals, _ = diagonalize_effective_H(H_eff)
    return (eigvals[0] + ecore - E_ref) * 1000


if __name__ == '__main__':
    main()
